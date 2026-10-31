r"""Extract a subset of zones and/or variables from a Tecplot data file.

Large Tecplot files produced by CFD solvers frequently contain many zones and variables,
only a fraction of which are relevant to a given analysis. Loading the full file into
Tecplot or writing a dedicated extraction script is impractical when the goal is simply
to isolate a surface zone or a handful of flow variables for downstream processing.
``tecextract`` addresses this by writing a new Tecplot file containing only the
requested subset, with the output format determined by the file extension of the output
path. This makes it straightforward to reduce file size, change format, or prepare a
subset for use with another tool in a single command.

:Usage:

.. code:: bash

    tecextract [-h] [-z LIST] [-v LIST] [-o PATH] [-f] PATH

:Positional Arguments:
    ``PATH``
        Path to the input Tecplot binary file (``.plt`` or ``.szplt``).

:Options:
    ``-z LIST``, ``--zones LIST``
        Comma-separated list of 0-based zone indices to extract (e.g. ``-z
        0,2,4``). If omitted, all zones are written to the output. A poly zone loses its
        boundary connections to zones that are not extracted, with a warning, and a
        boundary face left with no connection becomes an ordinary face.

    ``-v LIST``, ``--variables LIST``
        Comma-separated list of 0-based variable indices or exact variable
        names to extract (e.g. ``-v 0,1,4`` or ``-v x,y,pressure``).
        Indices and names cannot be mixed in the same list. A name containing a
        comma isn't supported by this syntax. If omitted, all variables are written
        to the output.

    ``-o PATH``, ``--output PATH``
        Output file path. The extension controls the output format: ``.szplt``,
        ``.plt``, or ``.dat``. Defaults to ``<stem>_extract<ext>`` in the same directory
        as the input file.

    ``-f``, ``--force``
        Overwrite the output file if it already exists. Without this flag the command
        exits with an error rather than silently clobbering an existing file.

:Returns:
    A new Tecplot file written to the output path containing only the requested zones
    and variables. Exit code is ``0`` on success and non-zero if the input file cannot
    be read, an invalid index is supplied, or the output file already exists and
    ``--force`` is not set.

Examples:
    Extract zones 0 and 2::

        $ tecextract -z 0,2 solution.szplt

    Extract variables 0, 1, and 4::

        $ tecextract -v 0,1,4 solution.szplt

    Extract variables by name::

        $ tecextract -v x,y,pressure solution.szplt

    Extract a zone subset and convert to ASCII in one step::

        $ tecextract -z 0,1 -o subset.dat solution.szplt

    Call directly from a Tecplot macro or Python session, passing arguments as a list of
    strings::

        import tecio.cli.tecextract.main as tecextract

        tecextract([
            "-z",
            "0,1",
            "-v",
            "0,1,4",
            "-o",
            "subset.szplt",
            "solution.szplt",
        ])

See Also:
    * :mod:`tecio.cli.tecsplit` - Split a file into separate grid and solution files
      rather than extracting a subset into a single output.
    * :mod:`tecio.cli.tecslice` - Slice a Tecplot file containing structured data along
      IJK indices and/or solution time.
    * :mod:`tecio.cli.tecmerge` - Merge zones from multiple files into a single output —
      the inverse operation to ``tecextract``.
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
    TecplotSzlWriter,
)
from .. import open as tecio_open
from .._writer import poly_face_kwargs

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _parse_index_list(value: str) -> list[int]:
    """Parse a comma-separated string of 0-based integers.

    Args:
        value: String like ``"0,2,4"`` or ``"1"``.

    Returns:
        List of integers.

    Raises:
        argparse.ArgumentTypeError: On invalid input.

    """
    try:
        return [int(v.strip()) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated integers, got: {value!r}"
        ) from exc


def _parse_index_or_name_list(value: str) -> list[int | str]:
    """Parse a comma-separated string of 0-based integers and/or variable names.

    Each token is parsed as an integer where possible; anything else is kept as a name
    string, resolved against the dataset's real variable list once the file is open
    (this function alone can't do that resolution, it doesn't have access to the file
    yet).

    Args:
        value: String like ``"0,2,pressure"`` or ``"x,y,z"``.

    Returns:
        List of ``int`` (0-based index) and/or ``str`` (variable name) tokens, in the
        order given.

    """
    tokens: list[int | str] = []
    for raw in value.split(","):
        token = raw.strip()
        try:
            tokens.append(int(token))
        except ValueError:
            tokens.append(token)
    return tokens


def _resolve_variable_tokens(
    tokens: list[int | str], num_vars: int, var_names: list[str]
) -> list[int]:
    """Resolve an index-only or name-only token list into 0-based indices.

    Args:
        tokens: Output of :func:`_parse_index_or_name_list`.
        num_vars: Total variable count in the dataset, for range checks.
        var_names: Dataset variable names in order, for exact-match lookup.

    Returns:
        List of validated 0-based indices, same order as *tokens*.

    Raises:
        ValueError: If *tokens* mixes indices and names, an integer token is out of
            range, or a name token doesn't exactly match any variable in *var_names*.

    Example:
        >>> _resolve_variable_tokens(["x", "y"], 3, ["x", "y", "z"])
        [0, 1]
    """
    has_int = any(isinstance(t, int) for t in tokens)
    has_name = any(isinstance(t, str) for t in tokens)
    if has_int and has_name:
        raise ValueError(
            "variable indices and names cannot be mixed in the same "
            f"-v/--variables list; use all indices or all names, got: "
            f"{','.join(str(t) for t in tokens)!r}."
        )
    resolved: list[int] = []
    for token in tokens:
        if isinstance(token, int):
            if token < 0 or token >= num_vars:
                raise ValueError(
                    f"variable index {token} out of range [0, {num_vars - 1}]."
                )
            resolved.append(token)
        else:
            try:
                resolved.append(var_names.index(token))
            except ValueError:
                raise ValueError(
                    f"variable name {token!r} not found; available names: "
                    f"{', '.join(var_names)}."
                ) from None
    return resolved


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tecextract",
        description=(
            # -|--------------------|---------------------------------------------|
            "Extract a subset of zones and/or variables from a Tecplot file.\n"
            "Output format is determined by the -o extension."
        ),
        epilog=(
            # -|--------------------|---------------------------------------------|
            "Example usage:\n"
            "  Extract zones 0 and 2\n"
            "    $ tecextract -z 0,2 <file>\n"
            "  Extract variables 0, 1, 4\n"
            "    $ tecextract -v 0,1,4 <file>\n"
            "  Extract variables by name\n"
            "    $ tecextract -v x,y,pressure <file>\n"
            "  Extract and convert format\n"
            "    $ tecextract -z 0,1 -o subset.dat <file>\n"
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=70, max_help_position=24
        ),
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Input Tecplot file.",
    )
    parser.add_argument(
        "-z",
        "--zones",
        type=_parse_index_list,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated list of 0-based zone indices to extract "
            "(e.g. -z 0,2,4). Default is all zones."
        ),
    )
    parser.add_argument(
        "-v",
        "--variables",
        type=_parse_index_or_name_list,
        default=None,
        metavar="LIST",
        help=(
            # -|--------------------|---------------------------------------------|
            "Comma-separated list of 0-based variable indices or exact variable "
            "names to extract. Indices and names cannot be mixed in the same list. "
            "A name containing a comma isn't supported by this syntax. Default is "
            "all variables."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Output file path. The extension controls the output format "
            "(.szplt, .plt, .dat). Defaults to <stem>_extract<ext> in "
            "the same directory as the input."
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Extract zones and/or variables from a Tecplot file.

    Returns:
        Exit code -- ``0`` on success, ``1`` on error.

    """
    args = _parse_args(argv)

    src = Path(args.filename)
    if not src.exists():
        print(f"Error: input file not found: {src}", file=sys.stderr)
        return 1

    dst = (
        Path(args.output)
        if args.output is not None
        else src.with_stem(src.stem + "_extract")
    )

    if dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\nUse --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    try:
        with tecio_open(str(src), "r") as reader:
            num_vars: int = reader.num_vars
            all_var_names: list[str] = reader.variables

            # Resolve and validate zone filter.
            if args.zones is not None:
                for z in args.zones:
                    if z < 0 or z >= reader.num_zones:
                        print(
                            f"Error: zone index {z} out of range "
                            f"[0, {reader.num_zones - 1}].",
                            file=sys.stderr,
                        )
                        return 1
                zone_set: set[int] = set(args.zones)
            else:
                zone_set = set(range(reader.num_zones))

            # Resolve and validate variable filter (index or name tokens).
            if args.variables is not None:
                try:
                    resolved_vars = _resolve_variable_tokens(
                        args.variables, num_vars, all_var_names
                    )
                except ValueError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    return 1
                var_set: set[int] = set(resolved_vars)
                # Ordered list preserving original index order.
                out_var_indices: list[int] = [
                    v for v in range(num_vars) if v in var_set
                ]
            else:
                out_var_indices = list(range(num_vars))

            out_var_names: list[str] = [all_var_names[v] for v in out_var_indices]

            print(f"Extracting: {src}  ->  {dst}")
            print(f"  Zones     : {sorted(zone_set)} of {reader.num_zones}")
            print(f"  Variables : {out_var_indices} of {num_vars} ({out_var_names})")

            with tecio_open(
                str(dst),
                "w",
                title=reader.title,
                variables=out_var_names,
                file_type=reader.file_type,
            ) as writer:
                # Forward dataset-level aux data.
                if len(reader.auxdata) > 0:
                    writer.add_auxdataset_dict(dict(reader.auxdata.items()))

                # Forward variable-level aux data for kept variables.
                auxvar: dict[int, dict[str, str]] = {}
                for new_idx, orig_idx in enumerate(out_var_indices):
                    var_aux = reader.get_var_auxdata(orig_idx)
                    if len(var_aux) > 0:
                        auxvar[new_idx] = dict(var_aux.items())
                if auxvar:
                    writer.add_auxvar_dict(auxvar)

                # Maps a source zone's 0-based index to its 0-based index in the output
                # file, populated as zones are actually written:
                # - A variable/connectivity shared from a zone that's also in this map
                #   can have its sharing preserved (just pointing at the new, compacted
                #   index) instead of being materialized as independent data
                # - Only a share whose source zone was excluded from the extraction
                #   genuinely has nowhere to point and must fall back to real data.
                zone_index_map: dict[int, int] = {}

                for zone_num, zone in enumerate(reader.zones):
                    if zone_num not in zone_set:
                        continue

                    zt = zone.zone_type
                    zones_before = writer.current_zone

                    # Build per-variable metadata for the requested subset.
                    data: list[np.ndarray] = []
                    value_locations: list[Any] = []
                    passive_vars: set[str] = set()
                    var_sharing: dict[str, int] = {}

                    for orig_idx in out_var_indices:
                        var = zone.variables[orig_idx]
                        is_passive = var.is_passive()

                        sv = var.shared_zone
                        remapped = zone_index_map.get(sv) if sv is not None else None

                        if is_passive:
                            passive_vars.add(var.name)
                        elif remapped is not None:
                            # Source zone was also extracted -> preserve the data
                            # sharing relationship
                            var_sharing[var.name] = remapped
                        else:
                            # Variables not shared at all, or shared from a zone that is
                            # not output, in which case branch sharing -> write the
                            # actual values as independent data
                            arr = var.values
                            if arr is None or arr.size == 0:
                                passive_vars.add(var.name)
                            else:
                                data.append(arr)
                                value_locations.append(var.value_location)

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
                    elif isinstance(zone, TecplotPolyZoneReader):
                        con_src = zone.shared_connectivity
                        con_remapped = (
                            zone_index_map.get(con_src) if con_src is not None else None
                        )
                        # Boundary connections follow the compacted zone numbering, a
                        # zone may also be connected to itself.
                        zone_map = {**zone_index_map, zone_num: writer.current_zone + 1}
                        writer.write_poly_zone(
                            zone_type=zt,
                            data=data,
                            con_sharing=con_remapped,
                            **(
                                {}
                                if con_remapped is not None
                                else poly_face_kwargs(zone, zone_map)
                            ),
                            **common_kw,
                        )
                    elif isinstance(zone, TecplotFEZoneReader):
                        con_src = zone.shared_connectivity
                        con_remapped = (
                            zone_index_map.get(con_src) if con_src is not None else None
                        )
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
                            node_map=(
                                None if con_remapped is not None else zone.node_map
                            ),
                            con_sharing=con_remapped,
                            **fe_kw,
                        )
                    else:
                        raise NotImplementedError(
                            f"Zone '{zone.title}' is neither an ordered, classic FE "
                            "nor poly zone; extraction is not supported for it."
                        )

                    # Record where this source zone landed in the output, so a later
                    # zone sharing from it can point at the real (compacted) index
                    # instead of falling back to independent data.  A zone with no
                    # active data (every variable shared or passive) is skipped by the
                    # writer and has no output index.
                    if writer.current_zone > zones_before:
                        zone_index_map[zone_num] = writer.current_zone

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        dst.unlink(missing_ok=True)
        return 1

    print(f"Done. Output written to: {dst}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
