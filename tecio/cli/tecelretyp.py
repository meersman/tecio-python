r"""Retype finite-element zones whose cells are all stored as a larger cell type.

Most unstructured mesh multi-physics solvers support volume cells with degenerate edges
such that multiple cell types can be stored in the same block with square connectivity
arrays. However, if all cells are upcast when they are written to disk as ``FEBRICK``
for all classical volume cells and ``FEQUADRILATERAL`` for surface cells, node indices
of upcast cells are repeated describe the smaller cells and incurring more
memory. Therefore, zones where all cells are upcast can be rewritten with the zone type
with the least required memory allocation and a compact node map. For example a solution
file containing tetrahedra volume cells and triangle surface cells can be written with
hexahedral and quadralateral cells respectively. ``tecelretyp`` finds such zones and
downcasts them as:

* ``FEBRICK`` zones whose cells are all tetrahedra become ``FETETRAHEDRON``.
* ``FEQUADRILATERAL`` zones whose cells are all triangles become ``FETRIANGLE``.

Detection does not assume a repeat convention (see :func:`tecio.classify_cells`), and
the handedness of each cell is kept. Node indices are not renumbered, so every variable,
nodal or cell-centered, is copied unchanged. Zones that share connectivity are retyped
together. A zone that holds any pyramid or prism cell has no classic zone type to
collapse to and is copied unchanged, and so is every other zone.

:Usage:

.. code:: bash

    tecelretyp [-h] [-z LIST] [--dry-run] [--check] [--drop-face-neighbors]
               [-o PATH] [-f] PATH

:Positional Arguments:
    ``PATH``
        Path to the input Tecplot file (``.szplt``, ``.plt`` or ``.dat``).

:Options:
    ``-z LIST``, ``--zones LIST``
        Comma-separated list of 0-based zone indices to consider (e.g. ``-z 0,2``).
        Other zones are copied unchanged. A group of zones sharing connectivity is only
        retyped if every zone in it is listed. If omitted, all zones are considered.

    ``--dry-run``
        Print what would change and write nothing.

    ``--check``
        Like ``--dry-run``, and exit with status ``1`` if any zone could be retyped (and
        ``0`` if none could), for use in scripts. Errors exit with status ``2``.

    ``--drop-face-neighbors``
        Retype zones that have face-neighbor data, discarding that data. Face numbers
        depend on the cell type, so the data cannot be carried over, and without this
        flag such zones are left as they are. Ignored for ``.szplt`` output, which does
        not store face-neighbor data.

    ``-o PATH``, ``--output PATH``
        Output file path. The extension controls the output format: ``.szplt``,
        ``.plt`` or ``.dat``. Defaults to ``<stem>_retyped<ext>`` in the same directory
        as the input file.

    ``-f``, ``--force``
        Overwrite the output file if it already exists.

:Returns:
    A new Tecplot file with the retypable zones rewritten, all other zones unchanged.
    If no zone can be retyped, no file is written. Exit code ``0`` on success and ``1``
    if the input cannot be read, a zone index is out of range, the output exists
    without ``--force``, or writing fails (``2`` for errors under ``--check``).

Examples:
    See what could be retyped without writing anything::

        $ tecelretyp --dry-run flow.plt

    Retype and write to a new file::

        $ tecelretyp flow.plt -o flow_compact.plt

    Retype and convert to SZL in one step::

        $ tecelretyp flow.plt -o flow_compact.szplt

    Fail a pipeline if a file still holds upcast cells::

        $ tecelretyp --check flow.plt

    Call directly from a Python session::

        from tecio.cli.tecelretyp import main as tecelretyp

        tecelretyp(["flow.plt", "-o", "flow_compact.plt"])

See Also:
    :func:`tecio.collapse_cells` and :func:`tecio.classify_cells` - The node map
    functions this tool is built on.

    :mod:`tecio.cli.teconvert` - Convert between Tecplot file formats.

"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import (
    FeCellShape,
    TecplotFEZoneReader,
    TecplotOrderedZoneReader,
    TecplotPolyZoneReader,
    TecplotReader,
    TecplotSzlWriter,
    ZoneType,
    collapse_cells,
    collapsed_zone_type,
    shape_counts,
)
from .. import open as tecio_open
from .._writer import poly_face_kwargs
from .tecextract import _parse_index_list

# Zone types that can collapse to a smaller type, and the number of nodes per cell
# before and after.
_COLLAPSIBLE: dict[ZoneType, tuple[int, int]] = {
    ZoneType.FEBRICK: (8, 4),
    ZoneType.FEQUADRILATERAL: (4, 3),
}

# Bytes per node index in the binary formats' node maps (32-bit), for estimates.
_BYTES_PER_INDEX = 4

_TITLE_WIDTH = 24

# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tecelretyp",
        description=(
            # -|-------------------|---------------------------------------------|
            "Rewrite FEBRICK zones that hold only tetrahedra as FETETRAHEDRON and\n"
            "FEQUADRILATERAL zones that hold only triangles as FETRIANGLE."
        ),
        epilog=(
            "Example usage:\n"
            "  Show what could be retyped\n"
            "    $ tecelretyp --dry-run <file>\n"
            "  Retype and write a new file\n"
            "    $ tecelretyp <file> -o compact.plt\n"
            "  Retype only zones 0 and 2\n"
            "    $ tecelretyp -z 0,2 <file>\n"
            "  Exit with status 1 if anything could be retyped\n"
            "    $ tecelretyp --check <file>\n"
        ),
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, width=70, max_help_position=24
        ),
    )
    parser.add_argument("filename", type=str, help="Input Tecplot file.")
    parser.add_argument(
        "-z",
        "--zones",
        type=_parse_index_list,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated list of 0-based zone indices to consider "
            "(e.g. -z 0,2). Default is all zones."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would change and write nothing.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=False,
        help=(
            "Like --dry-run, but exit with status 1 if any zone could be retyped "
            "(2 on error)."
        ),
    )
    parser.add_argument(
        "--drop-face-neighbors",
        action="store_true",
        default=False,
        help=(
            "Retype zones with face-neighbor data, discarding that data. Face "
            "numbers depend on the cell type, so it cannot be kept."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Output file path. The extension controls the output format. "
            "Defaults to <stem>_retyped<ext> in the same directory."
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
# Planning
# --------------------------------------------------------------------------------------


@dataclass
class _Plan:
    """What to do with one zone."""

    index: int
    title: str
    zone_type: ZoneType | None
    num_cells: int | None
    target: ZoneType | None = None
    note: str = ""
    saved_entries: int = 0


def _describe_shapes(counts: dict[FeCellShape | None, int]) -> str:
    """Summarize cell counts for a zone that cannot be retyped.

    Returns:
        Text such as ``"cells: 12 pyramid, 40 tetrahedron"``.
    """
    parts = [
        f"{n} {'unrecognized' if shape is None else shape.name.lower()}"
        for shape, n in counts.items()
    ]
    return "cells: " + ", ".join(parts)


def _root_zone(reader: TecplotReader, index: int) -> int:
    """Follow connectivity sharing back to the zone that owns the node map.

    Returns:
        0-based index of the zone that holds the connectivity.
    """
    while True:
        zone = reader.zones[index]
        source = (
            zone.shared_connectivity if isinstance(zone, TecplotFEZoneReader) else None
        )
        if source is None:
            return index
        index = source


def _make_plan(
    reader: TecplotReader, selected: set[int], *, protect_face_neighbors: bool
) -> list[_Plan]:
    """Decide which zones to retype.

    Zones that share connectivity form a group that is retyped together or not at all.

    Args:
        reader: Open reader of the input file.
        selected: Indices of the zones to consider.
        protect_face_neighbors: True if retyping would silently discard face-neighbor
            data, so zones that have it must be left alone.

    Returns:
        One plan per zone, in file order.
    """
    plans: list[_Plan] = []
    fe_zones: dict[int, TecplotFEZoneReader] = {}
    groups: dict[int, list[int]] = {}
    for i, zone in enumerate(reader.zones):
        plans.append(
            _Plan(
                i,
                zone.title or "",
                zone.zone_type,
                zone.num_elements if isinstance(zone, TecplotFEZoneReader) else None,
            )
        )
        if isinstance(zone, TecplotFEZoneReader):
            fe_zones[i] = zone
            groups.setdefault(_root_zone(reader, i), []).append(i)

    for root, members in groups.items():
        owner = fe_zones[root]
        zone_type = owner.zone_type
        if zone_type not in _COLLAPSIBLE:
            continue
        blocked = _blocking_reason(
            [fe_zones[i] for i in members],
            root,
            members,
            selected,
            protect_face_neighbors=protect_face_neighbors,
        )
        target = None
        if blocked is None:
            node_map = owner.node_map
            if node_map is None or node_map.size == 0:
                blocked = "no connectivity"
            else:
                target = collapsed_zone_type(node_map, zone_type)
                if target is None:
                    blocked = _describe_shapes(shape_counts(node_map, zone_type))
        for i in members:
            plans[i].note = blocked or ""
            if target is not None:
                plans[i].target = target
        if target is not None:
            before, after = _COLLAPSIBLE[zone_type]
            plans[root].saved_entries = (before - after) * int(owner.num_elements)
            for i in members:
                if i != root:
                    plans[i].note = f"shares connectivity with zone {root}"
    return plans


def _blocking_reason(
    zones: list[TecplotFEZoneReader],
    root: int,
    members: list[int],
    selected: set[int],
    *,
    protect_face_neighbors: bool,
) -> str | None:
    """Explain why a group of zones must be left as it is, if it must.

    Args:
        zones: The zones of the group.
        root: Index of the zone that owns the connectivity.
        members: Indices of the zones of the group.
        selected: Indices of the zones to consider.
        protect_face_neighbors: True if retyping would discard face-neighbor data.

    Returns:
        The reason, or ``None`` if the group may be retyped.
    """
    unselected = [i for i in members if i not in selected]
    if root in unselected:
        return "not selected"
    if unselected:
        return f"shares connectivity with unselected zone {unselected[0]}"
    if protect_face_neighbors and any(z.face_neighbor_mode is not None for z in zones):
        return "has face-neighbor data (see --drop-face-neighbors)"
    return None


def _print_plan(plans: list[_Plan]) -> None:
    """Print one line per zone and a summary."""
    print(
        f"{'Zone':>5}  {'Title':<{_TITLE_WIDTH}}  {'Type':<16}  {'Cells':>10}  Result"
    )
    for plan in plans:
        title = plan.title[:_TITLE_WIDTH]
        zone_type = plan.zone_type.name if plan.zone_type is not None else "?"
        cells = "" if plan.num_cells is None else str(plan.num_cells)
        if plan.target is not None:
            result = f"-> {plan.target.name}"
            if plan.note:
                result += f" ({plan.note})"
        else:
            result = f"unchanged ({plan.note})" if plan.note else "unchanged"
        print(
            f"{plan.index:>5}  {title:<{_TITLE_WIDTH}}  {zone_type:<16}  {cells:>10}  {result}"
        )
    retyped = [p for p in plans if p.target is not None]
    saved = sum(p.saved_entries for p in plans)
    print(
        f"{len(retyped)} of {len(plans)} zones can be retyped, removing {saved} node "
        f"map entries (about {saved * _BYTES_PER_INDEX / 2**20:.1f} MiB in a binary "
        "file)."
    )


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------


def _write_zones(reader: TecplotReader, writer: Any, plans: list[_Plan]) -> None:
    """Copy every zone of *reader* to *writer*, retyping the planned ones.

    Args:
        reader: Open reader of the input file.
        writer: Open writer of the output file.
        plans: Output of :func:`_make_plan`. Retyped zones lose any face-neighbor data,
            other zones keep it unless the writer cannot store it.

    Raises:
        NotImplementedError: If a zone is not an ordered, classic FE or poly zone.
    """
    for plan, zone in zip(plans, reader.zones, strict=True):
        data: list[np.ndarray] = []
        value_locations: list[Any] = []
        passive_vars: set[str] = set()
        var_sharing: dict[str, int] = {}
        for var in zone.variables:
            shared = var.shared_zone
            if var.is_passive():
                passive_vars.add(var.name)
            elif shared is not None:
                var_sharing[var.name] = shared
            elif var.values is None or var.values.size == 0:
                passive_vars.add(var.name)
            else:
                data.append(var.values)
                value_locations.append(var.value_location)

        common_kw: dict[str, Any] = {
            "title": zone.title,
            "value_locations": value_locations,
            "passive_vars": passive_vars,
            "var_sharing": var_sharing,
            "solution_time": zone.solution_time,
            "strand_id": zone.strand_id,
            "aux": dict(zone.auxdata.items()) if len(zone.auxdata) > 0 else None,
        }

        if isinstance(zone, TecplotOrderedZoneReader):
            writer.write_ordered_zone(data=data, **common_kw)
        elif isinstance(zone, TecplotPolyZoneReader):
            con_sharing = zone.shared_connectivity
            writer.write_poly_zone(
                zone_type=zone.zone_type,
                data=data,
                con_sharing=con_sharing,
                **({} if con_sharing is not None else poly_face_kwargs(zone)),
                **common_kw,
            )
        elif isinstance(zone, TecplotFEZoneReader):
            con_sharing = zone.shared_connectivity
            zone_type = zone.zone_type
            node_map = None if con_sharing is not None else zone.node_map
            if plan.target is not None:
                if node_map is not None:
                    node_map = collapse_cells(node_map, zone_type, plan.target)
                zone_type = plan.target
            elif zone.face_neighbor_mode is not None and not isinstance(
                writer, TecplotSzlWriter
            ):
                common_kw["face_neighbors"] = zone.get_face_connections()
                common_kw["face_neighbor_mode"] = zone.face_neighbor_mode
            writer.write_fe_zone(
                zone_type=zone_type,
                data=data,
                node_map=node_map,
                con_sharing=con_sharing,
                **common_kw,
            )
        else:
            raise NotImplementedError(
                f"Zone '{zone.title}' is not an ordered, FE or poly zone; it cannot "
                "be copied."
            )


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Retype finite-element zones that hold only smaller cells.

    Returns:
        Exit code. ``0`` on success and ``1`` on error. With ``--check``, ``0`` if no
        zone could be retyped, ``1`` if some could, and ``2`` on error.
    """
    args = _parse_args(argv)
    error = 2 if args.check else 1

    src = Path(args.filename)
    if not src.exists():
        print(f"Error: input file not found: {src}", file=sys.stderr)
        return error

    dst = (
        Path(args.output)
        if args.output is not None
        else src.with_stem(src.stem + "_retyped")
    )
    dry = args.dry_run or args.check
    if not dry and dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\nUse --force to overwrite.",
            file=sys.stderr,
        )
        return error

    # An SZL file cannot hold face-neighbor data, so there is nothing to protect.
    protect_face_neighbors = not (
        args.drop_face_neighbors or dst.suffix.lower() == ".szplt"
    )

    try:
        with tecio_open(str(src), "r") as reader:
            if args.zones is not None:
                for z in args.zones:
                    if z < 0 or z >= reader.num_zones:
                        print(
                            f"Error: zone index {z} out of range "
                            f"[0, {reader.num_zones - 1}].",
                            file=sys.stderr,
                        )
                        return error
                selected = set(args.zones)
            else:
                selected = set(range(reader.num_zones))

            plans = _make_plan(
                reader, selected, protect_face_neighbors=protect_face_neighbors
            )
            _print_plan(plans)
            retypable = any(p.target is not None for p in plans)

            if dry:
                return int(args.check and retypable)
            if not retypable:
                print("Nothing to retype. No file written.")
                return 0

            with tecio_open(
                str(dst),
                "w",
                title=reader.title,
                variables=reader.variables,
                file_type=reader.file_type,
            ) as writer:
                if len(reader.auxdata) > 0:
                    writer.add_auxdataset_dict(dict(reader.auxdata.items()))
                auxvar: dict[int, dict[str, str]] = {}
                for i in range(reader.num_vars):
                    var_aux = reader.get_var_auxdata(i)
                    if len(var_aux) > 0:
                        auxvar[i] = dict(var_aux.items())
                if auxvar:
                    writer.add_auxvar_dict(auxvar)
                _write_zones(reader, writer, plans)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        if not dry:
            dst.unlink(missing_ok=True)
        return error

    print(f"Done. Output written to: {dst}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
