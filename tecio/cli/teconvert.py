r"""Convert between Tecplot data file formats.

The three Tecplot file formats, SZL binary (``.szplt``), PLT binary (``.plt``), and
ASCII DAT (``.dat``), are not always interchangeable in practice. Tecplot itself
stores all data internally in binary form and provides license-free command-line
utilities (``preplot``, ``tec360 -convert``) for converting from lower-level formats
upward to SZL, but no supported path exists for the reverse without a active license.
This is a practical limitation when working with visualisation tools such as ParaView
that support PLT and DAT but not SZL, or when direct inspection of file contents in a
text editor is required.  ``teconvert`` fills this gap by supporting conversion in any
direction between all three formats without requiring a Tecplot installation.

:Usage:

.. code:: bash

    teconvert [-h] (-szplt | -plt | -dat) [-f] [-o PATH] FILE


:Positional Arguments:
    ``FILE``
        Path to the input Tecplot file (``.plt``, ``.szplt``, or ``.dat``) to convert.

:Options:
    ``-szplt``
        Convert to Tecplot SZL binary format (``.szplt``). Exactly one format flag is
        required.

    ``-plt``
        Convert to Tecplot PLT binary format (``.plt``). Exactly one format flag is
        required.

    ``-dat``
        Convert to Tecplot ASCII DAT format (``.dat``). Exactly one format flag is
        required.

    ``-o PATH``, ``--output PATH``
        Explicit output file path. Defaults to the input file stem with the new
        extension in the same directory as the input file.

    ``-f``, ``--force``
        Overwrite the output file if it already exists. Without this flag the command
        exits with an error rather than silently clobbering an existing file.

:Returns:
    A new Tecplot file written to the output path in the requested format. Exit code is
    ``0`` on success and non-zero if the input file cannot be read, no format flag is
    supplied, or the output file already exists and ``--force`` is not set.

Examples:
    Convert SZL to ASCII DAT::

        $ teconvert -dat flow.szplt

    Convert PLT to SZL::

        $ teconvert -szplt flow.plt

    Convert to PLT with an explicit output path::

        $ teconvert -plt -o /tmp/out.plt flow.dat

    Overwrite an existing output file::

        $ teconvert --force -dat flow.szplt

    Call directly from a Python session::

        import tecio.cli.teconvert.main as teconvert

        teconvert(["-dat", "-o", "flow.dat", "flow.szplt"])

See Also:
    :mod:`tecio.cli.tecextract` - Extract a zone/variable subset while simultaneously
    converting format via the output file extension.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import (
    TecplotFEZoneReader,
    TecplotOrderedZoneReader,
    TecplotPolyZoneReader,
    TecplotReader,
    TecplotSzlWriter,
    TecplotWriter,
    ZoneType,
)
from .. import open as tecio_open
from .._writer import poly_face_kwargs

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

#: Canonical extension for each format flag (flag name -> extension).
_FLAG_TO_EXT: dict[str, str] = {
    "szplt": ".szplt",
    "plt": ".plt",
    "dat": ".dat",
}

#: FE zone types supported by the Write API.
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="teconvert",
        description="Convert a Tecplot file to a different format.",
        epilog=(
            # -|--------------------|---------------------------------------------|
            "Output format flags (exactly one required):\n"
            "  -szplt  ->  Tecplot SZL binary (.szplt)\n"
            "  -plt    ->  Tecplot PLT binary (.plt)\n"
            "  -dat    ->  Tecplot ASCII DAT (.dat)\n\n"
            "Example usage:\n"
            "  Convert SZL -> DAT\n"
            "    $ teconvert -dat flow.szplt\n"
            "  Convert PLT -> SZL\n"
            "    $ teconvert -szplt flow.plt\n"
            "  Explicit output path\n"
            "    $ teconvert -plt -o /tmp/out.plt flow.dat\n"
            "  Overwrite existing\n"
            "    $ teconvert --force -dat flow.szplt\n"
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=70, max_help_position=24
        ),
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Input Tecplot file to convert.",
    )

    # Mutually exclusive format flags -- exactly one must be given.
    fmt_group = parser.add_mutually_exclusive_group(required=True)
    fmt_group.add_argument(
        "-szplt",
        action="store_const",
        const="szplt",
        dest="format",
        help="Convert to Tecplot SZL binary format (.szplt).",
    )
    fmt_group.add_argument(
        "-plt",
        action="store_const",
        const="plt",
        dest="format",
        help="Convert to Tecplot PLT binary format (.plt).",
    )
    fmt_group.add_argument(
        "-dat",
        action="store_const",
        const="dat",
        dest="format",
        help="Convert to Tecplot ASCII DAT format (.dat).",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Explicit output file path. Defaults to the input file stem with "
            "the new extension in the same directory as the input."
        ),
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# Zone copying helper
# --------------------------------------------------------------------------------------


def _copy_zones(reader: TecplotReader, writer: TecplotWriter) -> None:
    """Stream all zones from *reader* into the open *writer*.

    Each zone is reproduced at its original data type and value location.  Connectivity
    (node maps) is copied verbatim for FE zones, along with face-neighbor connections
    where present. Zone-level auxiliary data is forwarded as well.

    Args:
        reader: An open reader instance (SZL, PLT, or DAT).
        writer: An open writer instance for the target format.

    Raises:
        NotImplementedError: If attempting to pass a poly zone to a SZL writer.

    """
    for zone in reader.zones:
        zt = zone.zone_type

        # Collect per-variable arrays and metadata. Only active variables are
        # appended; passive_vars/var_sharing name the rest sparsely.
        data: list[np.ndarray] = []
        value_locations: list[Any] = []
        passive_vars: set[str] = set()
        var_sharing: dict[str, int] = {}

        for var in zone.variables:
            sv = var.shared_zone
            if var.is_passive():
                passive_vars.add(var.name)
            elif sv is not None:
                var_sharing[var.name] = sv
            else:
                data.append(var.values)  # ty: ignore[invalid-argument-type]
                value_locations.append(var.value_location)

        # Collect existing zone-level aux data.
        zone_aux: dict[str, str] | None = None
        if len(zone.auxdata) > 0:
            zone_aux = dict(zone.auxdata.items())

        common_kw: dict[str, Any] = {
            "title": zone.title,
            "value_locations": value_locations,
            "passive_vars": passive_vars,
            "var_sharing": var_sharing,
            "solution_time": zone.solution_time,
            "strand_id": zone.strand_id,
            "aux": zone_aux,
        }

        if isinstance(zone, TecplotOrderedZoneReader):
            writer.write_ordered_zone(data=data, **common_kw)
        elif isinstance(zone, TecplotFEZoneReader):
            con_sharing = zone.shared_connectivity
            fe_kw = common_kw.copy()

            # Face-neighbor connections
            if zone.face_neighbor_mode is not None and not isinstance(
                writer, TecplotSzlWriter
            ):
                fe_kw["face_neighbors"] = zone.get_face_connections()
                fe_kw["face_neighbor_mode"] = zone.face_neighbor_mode
            writer.write_fe_zone(
                zone_type=zt,
                data=data,
                node_map=None if con_sharing else zone.node_map,
                con_sharing=con_sharing,
                **fe_kw,
            )
        elif isinstance(zone, TecplotPolyZoneReader):
            con_sharing = zone.shared_connectivity
            writer.write_poly_zone(
                zone_type=zt,
                data=data,
                con_sharing=con_sharing,
                **({} if con_sharing is not None else poly_face_kwargs(zone)),
                **common_kw,
            )
        else:
            raise NotImplementedError(
                f"Zone '{zone.title}' is not an ordered, FE or poly zone; "
                "conversion is not supported for it."
            )


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Convert a Tecplot file to the requested format.

    Returns:
        Exit code -- ``0`` on success, ``1`` on error.

    """
    args = _parse_args(argv)

    src = Path(args.filename)
    if not src.exists():
        print(f"Error: input file not found: {src}", file=sys.stderr)
        return 1

    src_ext = src.suffix.lower()
    dst_ext = _FLAG_TO_EXT[args.format]

    if src_ext == dst_ext:
        print(
            f"Warning: source and destination formats are identical ({dst_ext}). "
            "Nothing to do.",
            file=sys.stderr,
        )
        return 0

    # Determine output path.
    dst = Path(args.output) if args.output is not None else src.with_suffix(dst_ext)

    if dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\nUse --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    print(f"Converting: {src}  ->  {dst}")

    try:
        with tecio_open(str(src), "r") as reader:
            # Propagate dataset-level metadata to the writer.
            title: str = reader.title
            variables: list[str] = reader.variables
            file_type = reader.file_type

            with tecio_open(
                str(dst),
                "w",
                title=title,
                variables=variables,
                file_type=file_type,
            ) as writer:
                # Forward dataset-level aux data.
                if len(reader.auxdata) > 0:
                    writer.add_auxdataset_dict(dict(reader.auxdata.items()))

                # Forward variable-level aux data.
                auxvar: dict[int, dict[str, str]] = {}
                for i in range(reader.num_vars):
                    var_aux = reader.get_var_auxdata(i)
                    if len(var_aux) > 0:
                        auxvar[i] = dict(var_aux.items())
                if auxvar:
                    writer.add_auxvar_dict(auxvar)

                _copy_zones(reader, writer)

    except NotImplementedError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        # Remove incomplete output so the user is not left with a partial file.
        dst.unlink(missing_ok=True)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error during conversion: {exc}", file=sys.stderr)
        dst.unlink(missing_ok=True)
        return 1

    print(f"Done. Output written to: {dst}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
