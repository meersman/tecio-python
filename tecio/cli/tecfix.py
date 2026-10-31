r"""Rewrite a Tecplot data file with invalid variable arrays set to passive.

Numerical solvers occasionally produce output containing ``NaN`` or ``Inf`` values,
whether due to localised solver divergence, incomplete initialisation, or variables that
are formally undefined in certain zones. These invalid values propagate silently
through post-processing pipelines and can cause unexpected behaviour in Tecplot and
downstream tools. ``tecfix`` addresses this by scanning every variable array in a file
and rewriting any array containing invalid values as passive, recording the affected
variable names in zone-level auxiliary data for traceability. A dry-run mode is
available to produce a diagnostic report without writing any output, making the tool
useful as a validation step before committing to a fix.

:Usage:

.. code:: bash

    tecfix [-h] [-o PATH] [-f] [--dry-run] PATH

:Positional Arguments:
    ``filename``
        Path to the input Tecplot binary file (``.plt`` or ``.szplt``) to inspect and
        fix.

:Options:
    ``-o PATH``, ``--output PATH``
        Output file path. Defaults to ``<stem>_fixed<ext>`` in the same directory as the
        input file.

    ``-f``, ``--force``
        Overwrite the output file if it already exists. Without this flag the command
        exits with an error rather than silently clobbering an existing file.

    ``--dry-run``
        Scan all variable arrays for invalid values and print a report without writing
        any output file. Useful for diagnosing whether a file requires fixing before
        modifying it.

:Returns:
    A new Tecplot file written to the output path with all invalid variable arrays set
    to passive. Affected variables are recorded in zone-level auxiliary data. If
    ``--dry-run`` is set, output is written to standard output only and no file is
    produced. Exit code is ``0`` on success and non-zero if the input file cannot be
    read or the output file already exists and ``--force`` is not set.

Examples:
    Fix a file with the default output naming::

        $ tecfix flow.szplt

    Write the fixed file to an explicit path::

        $ tecfix flow.szplt --output clean.szplt

    Overwrite an existing fixed file::

        $ tecfix --force flow.szplt

    Report bad variables without writing any output::

        $ tecfix --dry-run flow.szplt

    Call directly from a Python session::

        import tecio.cli.tecfix.main as tecfix

        tecfix(["flow.szplt", "--output", "clean.szplt"])

See Also:
    * :mod:`tecio.cli.tecdump` - Inspect file contents and metadata to
      identify zones or variables that may require fixing.
    * :mod:`tecio.cli.tecstats` - Compute per-zone min, max, and mean
      statistics without printing raw values.

Note:
    Only floating-point variables (``FLOAT`` and ``DOUBLE``) are inspected;
    integer variables cannot represent NaN / Inf and are always copied as-is.

Note:
    Variables that are already passive or shared in the source file are
    forwarded unchanged (they remain passive / shared).

Note:
    If a zone has no bad variables its aux data is not modified beyond what
    was already present in the source file.

Note:
    The output format matches the input format (extension is preserved).

"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .. import (
    DataType,
    TecplotFEZoneReader,
    TecplotOrderedZoneReader,
    TecplotPolyZoneReader,
    TecplotSzlWriter,
)
from .. import open as tecio_open
from .._writer import poly_face_kwargs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: DataTypes that can hold NaN / Inf.
_FLOAT_TYPES: frozenset[DataType] = frozenset({DataType.FLOAT, DataType.DOUBLE})


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tecfix",
        description=(
            # -|--------------------|--------------------------------------------|
            "Rewrite a Tecplot file with NaN / Inf variable arrays set to passive.\n"
            "Affected variables are documented in zone-level auxiliary data."
        ),
        epilog=(
            # -|--------------------|--------------------------------------------|
            "Example usage:\n"
            "  Output flow.szplt without diverging zones/variables as\n"
            "  flow_fixed.szplt\n"
            "    $ tecfix flow.szplt\n"
            "  Specify output file name\n"
            "    $ tecfix --output clean.szplt flow.szplt\n"
            "  Overwrite existing _fixed file\n"
            "    $ tecfix --force flow.szplt\n"
            "  Report bad variables and zones without output\n"
            "    $ tecfix --dry-run flow.szplt\n"
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=70, max_help_position=24
        ),
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Input Tecplot file to inspect and fix.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Explicit output file path. Defaults to <stem>_fixed<ext> "
            "in the same directory as the input file."
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help=(
            "Scan for bad values and print a report without writing any output file."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------


def _classify_array(arr: npt.NDArray) -> str | None:
    """Return a short description of any NaN / Inf in *arr*, or ``None``.

    Args:
        arr: A flat or shaped NumPy array of a floating-point dtype.

    Returns:
        ``"NaN"``, ``"Inf"``, ``"NaN and Inf"``, or ``None`` if the array
        is clean or empty.

    """
    # Some readers (e.g. DAT) may return an empty or None array for passive
    # or otherwise unavailable variables. Treat these as clean -> there is
    # nothing to inspect and numpy operations like isnan would error or
    # return misleading results on a zero-size array.
    if arr is None or arr.size == 0:
        return None

    has_nan = bool(np.any(np.isnan(arr)))
    has_inf = bool(np.any(np.isinf(arr)))

    if has_nan and has_inf:
        return "NaN and Inf"
    if has_nan:
        return "NaN"
    if has_inf:
        return "Inf"
    return None


# --------------------------------------------------------------------------------------
# Per-zone processing
# --------------------------------------------------------------------------------------


def _process_zone(
    zone: Any,
    num_vars: int,
    var_names: list[str],
) -> tuple[
    list[np.ndarray],
    list[Any],
    set[str],
    dict[str, int],
    dict[str, str],
    dict[str, str],
]:
    """Inspect one zone and build the data/metadata needed by the writer.

    For each variable:
    - If the variable is already passive or shared, it is forwarded exactly
      as-is. Sharing references are 0-based zone indices that are identical
      in the output file because tecfix writes every zone in order without
      skipping any.
    - If the variable is a floating-point type and contains NaN or Inf it is
      marked passive and recorded in *fix_auxdata*.
    - Otherwise it is included in the active data list verbatim.

    Args:
        zone:      A ``ReadZone`` instance from the source file.
        num_vars:  Total number of variables in the dataset.
        var_names: Ordered list of variable names (0-based).

    Returns:
        A 6-tuple of:
        ``(data, value_locations, passive_vars, var_sharing,
           existing_aux, fix_auxdata)``

        ``data``            – arrays for active, non-shared variables.
        ``value_locations`` – value locations aligned with *data*.
        ``passive_vars``    – set of variable names marked passive.
        ``var_sharing``     – {variable name: 0-based source zone} mapping.
        ``existing_aux``    – dict copied from the zone's original aux data.
        ``fix_auxdata``     – dict describing variables set passive by this tool.

    """
    data: list[np.ndarray] = []
    value_locations: list[Any] = []
    passive_vars: set[str] = set()
    var_sharing: dict[str, int] = {}
    fix_auxdata: dict[str, str] = {}

    for j, var in enumerate(zone.variables):
        sv = var.shared_zone  # 0-based source zone index, or None

        if var.is_passive():
            passive_vars.add(var.name)
            continue
        if sv is not None:
            # Pass sharing through verbatim.  tecfix writes all zones in the
            # same order as the source, so source zone N is always output zone N.
            var_sharing[var.name] = sv
            continue

        # Load the array.
        arr: np.ndarray = var.values

        # Guard against readers that return None or an empty array for
        # variables that are unavailable (e.g. DAT reader passive vars).
        if arr is None or arr.size == 0:
            passive_vars.add(var.name)
            continue

        # Only floating-point arrays can hold NaN / Inf.
        bad_label: str | None = None
        if var.data_type in _FLOAT_TYPES:
            bad_label = _classify_array(arr)

        if bad_label is not None:
            # Mark as passive; record the reason.
            passive_vars.add(var.name)
            key = f"Variable{j + 1}"
            fix_auxdata[key] = bad_label
        else:
            data.append(arr)
            value_locations.append(var.value_location)

    # Collect existing zone aux data.
    existing_aux: dict[str, str] = {}
    if len(zone.auxdata) > 0:
        existing_aux = dict(zone.auxdata.items())

    return (
        data,
        value_locations,
        passive_vars,
        var_sharing,
        existing_aux,
        fix_auxdata,
    )


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect and fix a Tecplot file.

    Returns:
        Exit code — ``0`` on success (including clean files), ``1`` on error.

    """
    args = _parse_args(argv)

    src = Path(args.filename)
    if not src.exists():
        print(f"Error: input file not found: {src}", file=sys.stderr)
        return 1

    # Determine output path (irrelevant for --dry-run but computed early so we
    # can catch conflicts before doing any work).
    dst = (
        Path(args.output)
        if args.output is not None
        else src.with_stem(src.stem + "_fixed")
    )

    if not args.dry_run and dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\nUse --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    # -- Scan / fix --------------------------------------------------------------------

    total_fixed = 0  # count of (zone, variable) pairs made passive

    try:
        with tecio_open(str(src), "r") as reader:
            num_vars: int = reader.num_vars
            var_names: list[str] = reader.variables

            if args.dry_run:
                # Scan only — report and exit without writing.
                print(f"Scanning: {src}")
                print(f"  Variables : {num_vars}")
                print(f"  Zones     : {reader.num_zones}")

                any_bad = False
                for i, zone in enumerate(reader.zones):
                    zone_bad: dict[str, str] = {}
                    for j, var in enumerate(zone.variables):
                        if var.is_passive() or var.shared_zone is not None:
                            continue
                        if var.data_type not in _FLOAT_TYPES:
                            continue
                        label = _classify_array(np.asarray(var.values))
                        if label is not None:
                            zone_bad[f"Variable{j + 1}"] = label

                    if zone_bad:
                        any_bad = True
                        total_fixed += len(zone_bad)
                        print(
                            f"\n  Zone {i + 1}: '{zone.title}' — "
                            f"{len(zone_bad)} bad variable(s)"
                        )
                        for key, label in zone_bad.items():
                            idx = int(key.replace("Variable", "")) - 1
                            print(f"    {key} ({var_names[idx]}): {label}")

                if not any_bad:
                    print("\nNo NaN or Inf values found. File is clean.")
                else:
                    print(
                        f"\nTotal: {total_fixed} variable(s) across "
                        f"{reader.num_zones} zone(s) contain NaN or Inf."
                    )
                return 0

            # -- Write fixed output ----------------------------------------------------

            print(f"Fixing: {src}  →  {dst}")

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

                # Forward variable-level aux data.
                auxvar: dict[int, dict[str, str]] = {}
                for i in range(num_vars):
                    var_aux = reader.get_var_auxdata(i)
                    if len(var_aux) > 0:
                        auxvar[i] = dict(var_aux.items())
                if auxvar:
                    writer.add_auxvar_dict(auxvar)

                for i, zone in enumerate(reader.zones):
                    zt = zone.zone_type

                    (
                        writer_data,
                        writer_locs,
                        passive_vars,
                        var_sharing,
                        existing_aux,
                        fix_auxdata,
                    ) = _process_zone(zone, num_vars, var_names)

                    if fix_auxdata:
                        n = len(fix_auxdata)
                        total_fixed += n
                        print(
                            f"  Zone {i + 1}: '{zone.title}' — "
                            f"setting {n} variable(s) passive:"
                        )
                        for key, label in fix_auxdata.items():
                            idx = int(key.replace("Variable", "")) - 1
                            print(f"    {key} ({var_names[idx]}): {label}")

                    # Merge existing aux data with any fix records.
                    merged_aux: dict[str, str] | None = {
                        **existing_aux,
                        **fix_auxdata,
                    } or None

                    # Sanitize solution_time: if it is NaN the zone metadata
                    # is corrupt. Reset both solution_time and strand_id to 0
                    # so Tecplot treats the zone as stationary and the writer
                    # does not propagate the bad value downstream.
                    sol_time = zone.solution_time
                    strand = zone.strand_id
                    if math.isnan(sol_time):
                        sol_time = 0.0
                        strand = 0
                        fix_auxdata["SolutionTime"] = "NaN"
                        total_fixed += 1
                        print(
                            f"  Zone {i + 1}: '{zone.title}' — "
                            "solution_time is NaN, reset to 0.0 (strand_id -> 0)"
                        )
                        # Rebuild merged_aux to pick up the SolutionTime entry.
                        merged_aux = {**existing_aux, **fix_auxdata} or None

                    common_kw: dict[str, Any] = {
                        "title": zone.title,
                        "value_locations": writer_locs,
                        "passive_vars": passive_vars,
                        "var_sharing": var_sharing,
                        "solution_time": sol_time,
                        "strand_id": strand,
                        "aux": merged_aux,
                    }

                    if isinstance(zone, TecplotOrderedZoneReader):
                        writer.write_ordered_zone(data=writer_data, **common_kw)
                    elif isinstance(zone, TecplotPolyZoneReader):
                        con_sharing = zone.shared_connectivity
                        writer.write_poly_zone(
                            zone_type=zt,
                            data=writer_data,
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
                            data=writer_data,
                            node_map=None if con_sharing is not None else zone.node_map,
                            con_sharing=con_sharing,
                            **fe_kw,
                        )
                    else:
                        raise NotImplementedError(
                            f"Zone '{zone.title}' is neither an ordered nor a "
                            "classic FE or poly zone; fixing is not supported for it."
                        )

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        if not args.dry_run:
            dst.unlink(missing_ok=True)
        return 1

    if total_fixed == 0:
        print("No NaN or Inf values found. Output file is a clean copy.")
    else:
        print(
            f"\nDone. {total_fixed} variable array(s) set passive. "
            f"Output written to: {dst}"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
