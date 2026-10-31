r"""Scale and/or offset variable arrays in a Tecplot data file.

CFD solvers and experimental data sources frequently produce output in differing unit
systems, and converting between them (e.g.pressure in Pa to kPa, temperature in Kelvin
to Celsius or lengths in meters to feet) is a routine step before visualisation or
comparison. Performing these transformations would otherwise require loading the file
in Tecplot or writing a dedicated script. ``tecscale`` applies the linear transformation

.. math::

    v' = a \cdot v + b

to every value in a selected variable array, where :math:`a` is the scale factor and
:math:`b` is the additive offset. Variable transformations with ``-v``/``--variable``
are repeatable for multiple scale and offset operations to be applied in a single pass
over the file. The transformation can be restricted to a single zone and the output
format is controlled by the output file extension, allowing unit conversion and format
conversion to be performed in a single step.

:Usage:

.. code:: bash

    tecscale [-h] -v INDEX_OR_NAME [-s SCALE] [--offset OFFSET]
             [-v ... [-s ...] [--offset ...]] [-z INDEX] [-o PATH] [-f] PATH

:Positional Arguments:
    ``PATH``
        Path to the input Tecplot binary file (``.plt`` or ``.szplt``) to transform.

:Options:
    ``-v INDEX_OR_NAME``, ``--variable INDEX_OR_NAME``
        A variable to transform (index or name, case-insensitive), applied as ``new =
        old * scale + offset``. Repeatable: give one ``-v`` per variable to transform
        several variables in one pass. Transformations applied to the same variable more
        than once keeps only the last occurrence.

    ``-s SCALE``, ``--scale SCALE``
        Multiplicative scale factor :math:`s` for the most recent ``-v``. Defaults to
        ``1`` (no scaling). Must follow the ``-v`` it applies to.

    ``--offset OFFSET``
        Additive offset :math:`b` for the most recent ``-v``. Defaults to ``0`` (no
        shift). Must follow the ``-v`` it applies to.

    ``-z INDEX``, ``--zone INDEX``
        0-based zone index to restrict the transformation to. If omitted, all zones
        are processed.

    ``-o PATH``, ``--output PATH``
        Output file path. The extension controls the output format: ``.szplt``,
        ``.plt``, or ``.dat``. Defaults to ``<stem>_scaled<ext>`` in the same directory
        as the input file.

    ``-f``, ``--force``
        Overwrite the output file if it already exists. Without this flag the command
        exits with an error rather than silently clobbering an existing file.

:Returns:
    A new Tecplot file written to the output path with the selected variable(s)
    transformed in all processed zones. Exit code ``0`` on success and non-zero if the
    input file cannot be read, a variable cannot be resolved, a ``SCALE``/``OFFSET``
    isn't a valid number, or the output file already exists and ``--force`` is not set.

Examples:
    Convert pressure from kPa to psi by index (scale only, offset 0)::

        $ tecscale -v 3 -s 0.145038 flow.szplt

    Same conversion using the variable name::

        $ tecscale -v Pressure -s 0.145038 flow.szplt

    Shift temperature from Kelvin to Celsius in the second zone only (offset only)::

        $ tecscale -v Temperature --offset -273.15 -z 1 flow.szplt

    Scale pressure and shift temperature in a single pass::

        $ tecscale -v Pressure -s 0.145038 -v Temperature --offset -273.15 flow.szplt

    Scale and offset in one step, writing to ASCII DAT::

        $ tecscale -v 2 -s 0.3048 flow.szplt -o flow_ft.dat

    Call directly from a Python session::

        import tecio.cli.tecscale.main as tecscale

        tecscale(["-v", "Pressure", "-s", "1e-3", "flow.szplt"])

See Also:
    :mod:`tecio.cli.teconvert` - Convert between Tecplot file formats without applying
    any variable transformation.

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
# Argument parsing
# --------------------------------------------------------------------------------------


class _AddVariable(argparse.Action):
    """Start a new ``[spec, scale, offset]`` entry with default scale 1, offset 0."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        """Append a default-valued entry for the variable spec in ``values``."""
        entries = getattr(namespace, self.dest, None) or []
        entries.append([values, "1", "0"])
        setattr(namespace, self.dest, entries)


class _SetField(argparse.Action):
    """Set one field of the most recent ``-v`` entry."""

    field = 0

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        """Store ``values`` in this action's field of the last ``-v`` entry."""
        entries = getattr(namespace, self.dest, None)
        if not entries:
            parser.error(f"{option_string} must follow a -v/--variable.")
        entries[-1][self.field] = values


class _SetScale(_SetField):
    field = 1


class _SetOffset(_SetField):
    field = 2


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tecscale",
        description=(
            # -|-------------------|---------------------------------------------|
            "Scale and/or offset one or more variables in a Tecplot file.\n"
            "Transformation: new = old * scale + offset."
        ),
        epilog=(
            "Example usage:\n"
            "  Scale variable 3 by 1e-3 (Pa -> kPa)\n"
            "    $ tecscale -v 3 -s 1e-3 <file>\n"
            "  Same using variable name\n"
            "    $ tecscale -v Pressure -s 1e-3 <file>\n"
            "  Offset temperature in zone 1 only\n"
            "    $ tecscale -v Temperature --offset -273.15 -z 1 <file>\n"
            "  Scale two variables in one pass\n"
            "    $ tecscale -v Pressure -s 1e-3 -v Temperature --offset -273.15 <file>\n"
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
        "-v",
        "--variable",
        dest="variable",
        action=_AddVariable,
        required=True,
        metavar="INDEX_OR_NAME",
        help=(
            "A variable to transform (0-based integer index or name, "
            "case-insensitive): new = old * SCALE + OFFSET, with SCALE and OFFSET "
            "from the -s/--scale and --offset that follow it (defaults 1 and 0). "
            "Repeatable: give one -v per variable to transform several in a "
            "single pass."
        ),
    )
    parser.add_argument(
        "-s",
        "--scale",
        dest="variable",
        action=_SetScale,
        metavar="SCALE",
        help="Scale factor for the preceding -v. Default 1 (no scaling).",
    )
    parser.add_argument(
        "--offset",
        dest="variable",
        action=_SetOffset,
        metavar="OFFSET",
        help="Additive offset for the preceding -v. Default 0 (no shift).",
    )
    parser.add_argument(
        "-z",
        "--zone",
        type=int,
        default=None,
        metavar="INDEX",
        help=(
            "0-based zone index to apply the transformation to. Default is all zones."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Output file path. The extension controls the output format.  "
            "Defaults to <stem>_scaled<ext> in the same directory."
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


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _resolve_variable(spec: str, var_names: list[str]) -> int | None:
    """Return a 0-based variable index from a name or 0-based integer string.

    Args:
        spec:      User-supplied string (e.g. ``"2"`` or ``"Pressure"``).
        var_names: Ordered list of variable names from the reader.

    Returns:
        0-based variable index.

    Raises:
        SystemExit: If the spec cannot be resolved.

    """
    # Try integer first.
    try:
        idx = int(spec)
        if idx < 0 or idx >= len(var_names):
            raise IndexError(
                f"Error: variable index {idx} out of range [0, {len(var_names) - 1}]."
            )
        return idx
    except ValueError:
        pass

    # Try case-insensitive name match.
    spec_lower = spec.lower()
    for i, name in enumerate(var_names):
        if name.lower() == spec_lower:
            return i

    print(
        f"Error: variable '{spec}' not found. Available: {var_names}",
        file=sys.stderr,
    )
    return None


def _resolve_transforms(
    raw_specs: list[list[str]], var_names: list[str]
) -> dict[int, tuple[float, float]] | None:
    """Resolve repeated ``-v`` entries into a ``{var_index: (scale, offset)}`` map.

    Args:
        raw_specs: ``args.variable`` as the argparse actions built it: one
            ``[spec, scale_str, offset_str]`` entry per ``-v`` occurrence, with the
            defaults ``"1"`` and ``"0"`` where ``-s``/``--offset`` were omitted.
        var_names: Ordered list of variable names from the reader.

    Returns:
        A mapping from 0-based variable index to ``(scale, offset)``, or ``None`` if
        any entry fails to resolve (an error is already printed to stderr). A variable
        named more than once keeps only its last occurrence.

    """
    transforms: dict[int, tuple[float, float]] = {}
    for spec, scale_str, offset_str in raw_specs:
        var_idx = _resolve_variable(spec, var_names)
        if var_idx is None:
            return None
        try:
            scale = float(scale_str)
            offset = float(offset_str)
        except ValueError:
            print(
                f"Error: scale/offset for variable '{spec}' must be numbers, "
                f"got scale={scale_str!r} offset={offset_str!r}.",
                file=sys.stderr,
            )
            return None
        transforms[var_idx] = (scale, offset)
    return transforms


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Scale and/or offset one or more variables in a Tecplot file.

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
        else src.with_stem(src.stem + "_scaled")
    )

    if dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\nUse --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    try:
        with tecio_open(str(src), "r") as reader:
            var_names: list[str] = reader.variables
            num_vars: int = reader.num_vars

            # Resolve every -v occurrence into {var_index: (scale, offset)}.
            transforms = _resolve_transforms(args.variable, var_names)
            if transforms is None:
                return 1

            # Validate zone if specified.
            if args.zone is not None and (
                args.zone < 0 or args.zone >= reader.num_zones
            ):
                print(
                    f"Error: zone index {args.zone} out of range "
                    f"[0, {reader.num_zones - 1}].",
                    file=sys.stderr,
                )
                return 1

            for var_idx, (scale, offset) in sorted(transforms.items()):
                print(
                    f"Scaling '{var_names[var_idx]}' (var {var_idx}): "
                    f"new = old * {scale} + {offset}"
                )
            if args.zone is not None:
                print(f"  Zone {args.zone} only.")
            else:
                print(f"  All {reader.num_zones} zones.")

            with tecio_open(
                str(dst),
                "w",
                title=reader.title,
                variables=var_names,
                file_type=reader.file_type,
            ) as writer:
                # Forward dataset-level aux data.
                if len(reader.auxdata) > 0:
                    writer.add_auxdataset_dict(dict(reader.auxdata.items()))

                # Forward all variable-level aux data.
                auxvar: dict[int, dict[str, str]] = {}
                for i in range(num_vars):
                    var_aux = reader.get_var_auxdata(i)
                    if len(var_aux) > 0:
                        auxvar[i] = dict(var_aux.items())
                if auxvar:
                    writer.add_auxvar_dict(auxvar)

                for zone_num, zone in enumerate(reader.zones):
                    zt = zone.zone_type

                    apply_scale = args.zone is None or zone_num == args.zone

                    data: list[np.ndarray] = []
                    value_locations: list[Any] = []
                    passive_vars: set[str] = set()
                    var_sharing: dict[str, int] = {}

                    for j, var in enumerate(zone.variables):
                        sv = var.shared_zone
                        if var.is_passive():
                            passive_vars.add(var.name)
                            continue
                        if sv is not None:
                            var_sharing[var.name] = sv
                            continue

                        arr = var.values
                        if arr is None or arr.size == 0:
                            passive_vars.add(var.name)
                            continue

                        if apply_scale and j in transforms:
                            # Apply transformation, promoting to float64 to
                            # preserve precision across the operation.
                            scale, offset = transforms[j]
                            arr = arr.astype(np.float64)
                            arr = arr * scale + offset

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
                        con_sharing = zone.shared_connectivity
                        writer.write_poly_zone(
                            zone_type=zt,
                            data=data,
                            con_sharing=con_sharing,
                            **(
                                {}
                                if con_sharing is not None
                                else poly_face_kwargs(zone)
                            ),
                            **common_kw,
                        )
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
                            node_map=None if con_sharing is not None else zone.node_map,
                            con_sharing=con_sharing,
                            **fe_kw,
                        )
                    else:
                        raise NotImplementedError(
                            f"Zone '{zone.title}' is neither an ordered nor a "
                            "classic FE or poly zone; scaling is not supported for it."
                        )

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        dst.unlink(missing_ok=True)
        return 1

    print(f"Done. Output written to: {dst}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
