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
together.

A zone that holds several cell shapes, such as tetrahedra with pyramids and prisms, has
no classic zone type to collapse to. With ``--mixed`` it is written as a ``FEMIXED``
zone instead, with one section per shape (see :func:`tecio.split_cells_by_shape`). The
cells are then grouped by shape, so cell-centered variables are reordered to match.
``FEMIXED`` zones need ``.szplt`` output. Every other zone is copied unchanged.

``--dry-run`` lists what would change and ends with the estimated size of the
connectivity (the node maps, which are the part retyping changes) under each option,
including as polyhedral or polygonal zones, and with the commands that give each saving.
Poly output cannot be written by this tool and is only an estimate.

:Usage:

.. code:: bash

    tecelretyp [-h] [-z LIST] [--mixed] [--dry-run] [--check]
               [--drop-face-neighbors] [-o PATH] [-f] PATH

:Positional Arguments:
    ``PATH``
        Path to the input Tecplot file (``.szplt``, ``.plt`` or ``.dat``).

:Options:
    ``-z LIST``, ``--zones LIST``
        Comma-separated list of 0-based zone indices to consider (e.g. ``-z 0,2``).
        Other zones are copied unchanged. A group of zones sharing connectivity is only
        retyped if every zone in it is listed. If omitted, all zones are considered.

    ``--mixed``
        Write zones that hold several cell shapes as ``FEMIXED`` zones. Needs ``.szplt``
        output, which is then the default for the output name. Face-neighbor data is
        discarded, as for any ``.szplt`` output.

    ``--dry-run``
        Print what would change, the estimated sizes and suggested commands, and write
        nothing.

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
    without ``--force``, ``--mixed`` is given with ``.plt`` or ``.dat`` output, or
    writing fails (``2`` for errors under ``--check``).

Examples:
    See what could be retyped, and how much it would save, without writing anything::

        $ tecelretyp --dry-run flow.plt

    Retype and write to a new file::

        $ tecelretyp flow.plt -o flow_compact.plt

    Retype and convert to SZL in one step::

        $ tecelretyp flow.plt -o flow_compact.szplt

    Also write zones of mixed cell shapes as ``FEMIXED``::

        $ tecelretyp --mixed flow.plt -o flow_compact.szplt

    Fail a pipeline if a file still holds upcast cells::

        $ tecelretyp --check flow.plt

    Call directly from a Python session::

        from tecio.cli.tecelretyp import main as tecelretyp

        tecelretyp(["flow.plt", "-o", "flow_compact.plt"])

See Also:
    :func:`tecio.collapse_cells`, :func:`tecio.split_cells_by_shape` and
    :func:`tecio.classify_cells` - The node map functions this tool is built on.

    :mod:`tecio.cli.teconvert` - Convert between Tecplot file formats.

"""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .. import (
    FeCellShape,
    TecplotFEZoneReader,
    TecplotMixedZoneReader,
    TecplotOrderedZoneReader,
    TecplotPolyZoneReader,
    TecplotReader,
    TecplotSzlWriter,
    ValueLocation,
    ZoneType,
    collapse_cells,
    shape_counts,
    split_cells_by_shape,
)
from .. import open as tecio_open
from .._elements import CELL_SHAPE_DIMENSION, CELL_SHAPE_NUM_NODES
from .._writer import poly_face_kwargs
from .tecextract import _parse_index_list

# Zone types that can collapse to a smaller type, with the zone type that all-small
# cells collapse to, the shape of those cells and the shape of a cell that uses every
# node.
_COLLAPSIBLE: dict[ZoneType, tuple[ZoneType, FeCellShape, FeCellShape]] = {
    ZoneType.FEBRICK: (
        ZoneType.FETETRAHEDRON,
        FeCellShape.TETRAHEDRON,
        FeCellShape.HEXAHEDRON,
    ),
    ZoneType.FEQUADRILATERAL: (
        ZoneType.FETRIANGLE,
        FeCellShape.TRIANGLE,
        FeCellShape.QUADRILATERAL,
    ),
}

# Shape of every cell in the other classic zone types.
_NATURAL_SHAPE: dict[ZoneType, FeCellShape] = {
    ZoneType.FELINESEG: FeCellShape.BAR,
    ZoneType.FETRIANGLE: FeCellShape.TRIANGLE,
    ZoneType.FETETRAHEDRON: FeCellShape.TETRAHEDRON,
}

# Faces and node references over all faces of one cell, for the poly size estimate.
_POLY_CELL: dict[FeCellShape, tuple[int, int]] = {
    FeCellShape.TRIANGLE: (3, 6),
    FeCellShape.QUADRILATERAL: (4, 8),
    FeCellShape.TETRAHEDRON: (4, 12),
    FeCellShape.HEXAHEDRON: (6, 24),
    FeCellShape.PYRAMID: (5, 16),
    FeCellShape.PRISM: (5, 18),
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
            "FEQUADRILATERAL zones that hold only triangles as FETRIANGLE. With\n"
            "--mixed, zones of several cell shapes become FEMIXED zones."
        ),
        epilog=(
            "Example usage:\n"
            "  Show what could be retyped and how much it would save\n"
            "    $ tecelretyp --dry-run <file>\n"
            "  Retype and write a new file\n"
            "    $ tecelretyp <file> -o compact.plt\n"
            "  Also write mixed cell shapes as FEMIXED zones\n"
            "    $ tecelretyp --mixed <file> -o compact.szplt\n"
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
        "--mixed",
        action="store_true",
        default=False,
        help=(
            "Write zones of several cell shapes as FEMIXED zones. Needs .szplt "
            "output, which is the default output format with this flag."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would change, size estimates and commands. Write nothing.",
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
# Analysis
# --------------------------------------------------------------------------------------


@dataclass
class _Group:
    """Zones that share one node map, owned by the zone ``root``."""

    root: int
    members: list[int]
    zone_type: ZoneType
    num_cells: int
    counts: dict[FeCellShape | None, int]  # empty if there is no connectivity
    face_neighbors: bool
    target: ZoneType | None = None  # classic type that every cell collapses to
    mixed: dict[FeCellShape, int] | None = None  # shapes, if FEMIXED would be smaller
    current: int = 0  # bytes of the node map as it is
    retyped: int = 0  # bytes as ``target``
    as_mixed: int = 0  # bytes as FEMIXED
    as_poly: int = 0  # estimated bytes of the faces of a poly zone


@dataclass
class _Analysis:
    """What the file holds, found once and used to plan any set of options."""

    groups: dict[int, _Group] = field(default_factory=dict)
    # Bytes of connectivity of each zone that owns it, for every kind of zone.
    current: dict[int, int] = field(default_factory=dict)
    poly: dict[int, int] = field(default_factory=dict)
    # Pairs (zone, source zone) where the zone shares a cell-centered variable.
    cell_links: list[tuple[int, int]] = field(default_factory=list)


def _shape_list(counts: Mapping[Any, int]) -> str:
    """List cell counts by shape.

    Returns:
        Text such as ``"12 pyramid, 40 tetrahedron"``.
    """
    return ", ".join(
        f"{n} {'unrecognized' if shape is None else shape.name.lower()}"
        for shape, n in counts.items()
    )


def _node_map_bytes(counts: Mapping[FeCellShape, int]) -> int:
    """Return the bytes of the node maps of cells of the given shapes."""
    return _BYTES_PER_INDEX * sum(
        n * CELL_SHAPE_NUM_NODES[shape] for shape, n in counts.items()
    )


def _poly_bytes(counts: Mapping[FeCellShape, int]) -> int:
    """Estimate the bytes of the face arrays of cells of the given shapes as poly.

    Every interior face is shared by two cells, so each cell is charged half of its
    faces. A face stores its nodes, the elements on its two sides and, in a
    polyhedron, its node count. Boundary faces are not counted.

    Returns:
        Estimated bytes of the node, count and element arrays of the faces.
    """
    total = 0
    for shape, n in counts.items():
        if shape not in _POLY_CELL:
            total += n * CELL_SHAPE_NUM_NODES[shape] * _BYTES_PER_INDEX
            continue
        faces, refs = _POLY_CELL[shape]
        per_face = 3 if CELL_SHAPE_DIMENSION[shape] == 3 else 2
        total += n * _BYTES_PER_INDEX * (refs + per_face * faces) // 2
    return total


def _poly_zone_bytes(zone: TecplotPolyZoneReader) -> int:
    """Return the bytes of the face arrays of a poly zone (boundary faces excluded)."""
    faces = int(zone.num_faces)
    if zone.zone_type == ZoneType.FEPOLYGON:
        return _BYTES_PER_INDEX * 4 * faces
    nodes = int(zone.face_node_counts.sum())
    return _BYTES_PER_INDEX * (nodes + 3 * faces)


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


def _value_location(reader: TecplotReader, zone: int, index: int) -> ValueLocation:
    """Return the value location of a variable, as set by the zone holding its values.

    A shared variable can report the default location, so follow it to its source.

    Returns:
        The location of variable *index* of zone *zone*.
    """
    while True:
        var = reader.zones[zone].variables[index]
        source = var.shared_zone
        if source is None or source >= zone:
            return var.value_location
        zone = source


def _build_group(reader: TecplotReader, root: int, members: list[int]) -> _Group:
    """Classify the cells of the zones that share the node map of zone *root*.

    Returns:
        The group, with the sizes of each way it could be stored.
    """
    owner = reader.zones[root]
    assert isinstance(owner, TecplotFEZoneReader)
    zone_type = owner.zone_type
    num_cells = int(owner.num_elements)
    group = _Group(
        root,
        members,
        zone_type,
        num_cells,
        {},
        any(
            isinstance(z := reader.zones[i], TecplotFEZoneReader)
            and z.face_neighbor_mode is not None
            for i in members
        ),
    )
    if zone_type not in _COLLAPSIBLE:
        shape = _NATURAL_SHAPE.get(zone_type)
        if shape is not None and num_cells:
            group.counts = {shape: num_cells}
            group.current = _node_map_bytes({shape: num_cells})
            group.as_poly = _poly_bytes({shape: num_cells})
        return group

    target, small, full = _COLLAPSIBLE[zone_type]
    node_map = owner.node_map
    if node_map is None or node_map.size == 0:
        return group
    group.counts = shape_counts(node_map, zone_type)
    group.current = num_cells * CELL_SHAPE_NUM_NODES[full] * _BYTES_PER_INDEX
    known = {s: n for s, n in group.counts.items() if s is not None}
    known[full] = known.get(full, 0) + group.counts.get(None, 0)
    group.as_poly = _poly_bytes(known)
    if set(group.counts) == {small}:
        group.target = target
        group.retyped = num_cells * CELL_SHAPE_NUM_NODES[small] * _BYTES_PER_INDEX
    elif None not in group.counts and set(group.counts) != {full}:
        group.mixed = known
        group.as_mixed = _node_map_bytes(known)
    return group


def _analyze(reader: TecplotReader) -> _Analysis:
    """Find the cell shapes, sizes and variable links of every zone of a file.

    Returns:
        The analysis.
    """
    analysis = _Analysis()
    members: dict[int, list[int]] = {}
    for i, zone in enumerate(reader.zones):
        for j, var in enumerate(zone.variables):
            source = var.shared_zone
            if (
                source is not None
                and _value_location(reader, i, j) == ValueLocation.CELL_CENTERED
            ):
                analysis.cell_links.append((i, source))
        if isinstance(zone, TecplotFEZoneReader):
            members.setdefault(_root_zone(reader, i), []).append(i)
        elif isinstance(zone, TecplotMixedZoneReader):
            if zone.shared_connectivity is None:
                counts: dict[FeCellShape, int] = {}
                for shape, n in zip(
                    zone.section_shapes, zone.section_num_elements, strict=True
                ):
                    counts[shape] = counts.get(shape, 0) + int(n)
                analysis.current[i] = _node_map_bytes(counts)
                analysis.poly[i] = _poly_bytes(counts)
        elif (
            isinstance(zone, TecplotPolyZoneReader) and zone.shared_connectivity is None
        ):
            analysis.current[i] = analysis.poly[i] = _poly_zone_bytes(zone)
    for root, group_members in members.items():
        group = _build_group(reader, root, group_members)
        analysis.groups[root] = group
        analysis.current[root] = group.current
        analysis.poly[root] = group.as_poly
    return analysis


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
    root: int = -1  # zone that owns the node map
    target: ZoneType | None = None
    to_mixed: bool = False
    mixed_counts: dict[FeCellShape, int] | None = None  # set on the owner only
    note: str = ""
    saved_entries: int = 0


def _describe_shapes(counts: dict[FeCellShape | None, int]) -> str:
    """Summarize cell counts for a zone that cannot be retyped.

    Returns:
        Text such as ``"cells: 12 pyramid, 40 tetrahedron"``.
    """
    return "cells: " + _shape_list(counts)


def _make_plan(
    reader: TecplotReader,
    analysis: _Analysis,
    selected: set[int],
    *,
    protect_face_neighbors: bool,
    mixed: bool,
) -> list[_Plan]:
    """Decide which zones to retype.

    Zones that share connectivity form a group that is retyped together or not at all.

    Args:
        reader: Open reader of the input file.
        analysis: Output of :func:`_analyze`.
        selected: Indices of the zones to consider.
        protect_face_neighbors: True if retyping would silently discard face-neighbor
            data, so zones that have it must be left alone.
        mixed: True to write zones of several cell shapes as ``FEMIXED``.

    Returns:
        One plan per zone, in file order.
    """
    plans = [
        _Plan(
            i,
            zone.title or "",
            zone.zone_type,
            zone.num_elements if isinstance(zone, TecplotFEZoneReader) else None,
            root=i,
        )
        for i, zone in enumerate(reader.zones)
    ]
    for root, group in analysis.groups.items():
        for i in group.members:
            plans[i].root = root
        if group.zone_type not in _COLLAPSIBLE:
            continue
        if not group.counts:
            for i in group.members:
                plans[i].note = "no connectivity"
            continue
        # FEMIXED needs an SZL file, which has nowhere to keep face-neighbor data.
        wanted_mixed = group.target is None and group.mixed is not None
        blocked = _blocking_reason(
            group,
            selected,
            protect_face_neighbors=protect_face_neighbors and not wanted_mixed,
        )
        note = blocked or ""
        target = group.target if blocked is None else None
        to_mixed = wanted_mixed and mixed and blocked is None
        if target is None and not to_mixed and blocked is None:
            note = _describe_shapes(group.counts)
            if wanted_mixed:
                note += "; could be FEMIXED, see --mixed"
        for i in group.members:
            plans[i].note = note
            plans[i].target = target
            plans[i].to_mixed = to_mixed
        before = group.current // _BYTES_PER_INDEX
        if target is not None:
            plans[root].saved_entries = before - group.retyped // _BYTES_PER_INDEX
        elif to_mixed:
            plans[root].mixed_counts = group.mixed
            plans[root].saved_entries = before - group.as_mixed // _BYTES_PER_INDEX
        if target is not None or to_mixed:
            for i in group.members:
                if i != root:
                    plans[i].note = f"shares connectivity with zone {root}"
    _keep_cell_values_consistent(plans, analysis.cell_links)
    return plans


def _keep_cell_values_consistent(
    plans: list[_Plan], links: list[tuple[int, int]]
) -> None:
    """Leave a zone as it is if reordering its cells would break a shared variable.

    A ``FEMIXED`` zone lists its cells by shape, so its cell-centered variables are
    reordered. Zones that share such a variable must agree on the order, which holds if
    they share connectivity and so share the plan, but not otherwise.

    Args:
        plans: Plans to correct in place.
        links: Pairs of (zone, source zone) that share a cell-centered variable.
    """
    changed = True
    while changed:
        changed = False
        owner = {p.index: p.root for p in plans if p.to_mixed}
        for zone, source in links:
            a, b = owner.get(zone), owner.get(source)
            if a == b:
                continue
            for root, other in ((a, source), (b, zone)):
                if root is None:
                    continue
                for plan in plans:
                    if plan.root == root and plan.to_mixed:
                        plan.to_mixed = False
                        plan.mixed_counts = None
                        plan.saved_entries = 0
                        plan.note = f"shares a cell-centered variable with zone {other}"
            changed = True


def _blocking_reason(
    group: _Group,
    selected: set[int],
    *,
    protect_face_neighbors: bool,
) -> str | None:
    """Explain why a group of zones must be left as it is, if it must.

    Args:
        group: The zones that share one node map.
        selected: Indices of the zones to consider.
        protect_face_neighbors: True if retyping would discard face-neighbor data.

    Returns:
        The reason, or ``None`` if the group may be retyped.
    """
    unselected = [i for i in group.members if i not in selected]
    if group.root in unselected:
        return "not selected"
    if unselected:
        return f"shares connectivity with unselected zone {unselected[0]}"
    if protect_face_neighbors and group.face_neighbors:
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
        new_type = "FEMIXED" if plan.to_mixed else plan.target and plan.target.name
        if new_type:
            result = f"-> {new_type}"
            if plan.mixed_counts is not None:
                result += f": {_shape_list(plan.mixed_counts)}"
            if plan.note:
                result += f" ({plan.note})"
        else:
            result = f"unchanged ({plan.note})" if plan.note else "unchanged"
        print(
            f"{plan.index:>5}  {title:<{_TITLE_WIDTH}}  {zone_type:<16}  {cells:>10}  {result}"
        )
    retyped = [p for p in plans if p.target is not None or p.to_mixed]
    saved = sum(p.saved_entries for p in plans)
    as_mixed = sum(p.to_mixed for p in plans)
    detail = f" ({as_mixed} as FEMIXED)" if as_mixed else ""
    print(
        f"{len(retyped)} of {len(plans)} zones can be retyped{detail}, removing "
        f"{saved} node map entries (about {saved * _BYTES_PER_INDEX / 2**20:.1f} MiB "
        "in a binary file)."
    )


# --------------------------------------------------------------------------------------
# Estimates
# --------------------------------------------------------------------------------------


def _size(nbytes: float, *, signed: bool = False) -> str:
    """Format a byte count with a binary unit.

    Returns:
        Text such as ``"12.3 MiB"``, with a leading sign if *signed*.
    """
    if nbytes == 0:
        return "0 B"
    units = ("B", "KiB", "MiB", "GiB")
    value = float(nbytes)
    step = 0
    while abs(value) >= 1024 and step < len(units) - 1:
        value /= 1024
        step += 1
    unit = units[step]
    digits = 0 if step == 0 else 1
    return f"{value:+.{digits}f} {unit}" if signed else f"{value:.{digits}f} {unit}"


def _after_bytes(analysis: _Analysis, plans: list[_Plan]) -> int:
    """Return the bytes of connectivity left after carrying out *plans*."""
    total = 0
    for root, current in analysis.current.items():
        group = analysis.groups.get(root)
        if group is not None and plans[root].target is not None:
            total += group.retyped
        elif group is not None and plans[root].to_mixed:
            total += group.as_mixed
        else:
            total += current
    return total


def _suggestions(
    analysis: _Analysis, variants: list[tuple[list[str], list[_Plan]]]
) -> list[tuple[list[str], int, str]]:
    """Pick the option sets that save more than every weaker set before them.

    Args:
        analysis: Output of :func:`_analyze`.
        variants: Options and the plan they give, weakest first.

    Returns:
        The options, the bytes they save and a description of what they do.
    """
    current = sum(analysis.current.values())
    best = 0
    found = []
    for flags, plans in variants:
        saved = current - _after_bytes(analysis, plans)
        if saved <= best:
            continue
        best = saved
        retyped = sum(p.target is not None for p in plans)
        mixed = sum(p.to_mixed for p in plans)
        parts = []
        if retyped:
            parts.append(f"retypes {retyped} {'zone' if retyped == 1 else 'zones'}")
        if mixed:
            parts.append(
                f"writes {mixed} {'zone' if mixed == 1 else 'zones'} as FEMIXED"
            )
        acted = {p.root for p in plans if p.target is not None or p.to_mixed}
        if any(analysis.groups[r].face_neighbors for r in acted):
            parts.append("discards face-neighbor data")
        found.append((flags, saved, ", ".join(parts)))
    return found


def _print_estimates(
    src: Path,
    command: list[str],
    analysis: _Analysis,
    variants: list[tuple[list[str], list[_Plan]]],
) -> None:
    """Print the estimated connectivity size of each option and suggested commands.

    Args:
        src: Input file, whose size gives the percentages.
        command: Words of the command that start every suggestion.
        analysis: Output of :func:`_analyze`.
        variants: Options and the plan they give, weakest first. The first is plain
            retyping and the last also writes ``FEMIXED`` zones.
    """
    current = sum(analysis.current.values())
    if current == 0:
        return
    file_size = src.stat().st_size
    rows = [
        ("Current", current),
        ("Retype", _after_bytes(analysis, variants[0][1])),
        ("Retype + FEMIXED", _after_bytes(analysis, variants[-1][1])),
        ("Poly (estimate only)", sum(analysis.poly.values())),
    ]
    print()
    print("Estimated size of the connectivity (node maps and faces)")
    print(f"  {'Option':<22}{'Size':>12}{'Change':>13}{'Of file':>9}")
    for name, size in rows:
        if name == "Current":
            print(f"  {name:<22}{_size(size):>12}")
            continue
        change = size - current
        share = f"{100 * change / file_size:+.1f}%" if file_size else ""
        print(
            f"  {name:<22}{_size(size):>12}{_size(change, signed=True):>13}{share:>9}"
        )
    print(
        "  Counts raw 32-bit entries, so a compressed .szplt saves somewhat less.\n"
        "  Poly ignores boundary faces, and this tool cannot write it."
    )

    suggestions = _suggestions(analysis, variants)
    print()
    if not suggestions:
        print("Suggested commands: none, nothing can be made smaller.")
        return
    print("Suggested commands")
    for flags, saved, what in suggestions:
        words = [*command[:1], *flags, *command[1:]]
        share = f" ({-100 * saved / file_size:+.1f}% of the file)" if file_size else ""
        print(f"  {' '.join(words)}")
        print(f"      {_size(-saved, signed=True)}{share}: {what}")


# --------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------


@dataclass
class _MixedLayout:
    """The sections a group of zones is written as, shared by the whole group."""

    shapes: list[FeCellShape]
    node_maps: list[npt.NDArray[np.integer]]
    order: npt.NDArray[np.intp]  # input cell of each cell of the new zone


def _mixed_layout(owner: TecplotFEZoneReader) -> _MixedLayout:
    """Split the cells of *owner* by shape.

    Returns:
        The sections and the order of the cells.
    """
    node_map = owner.node_map
    assert node_map is not None
    parts = split_cells_by_shape(node_map, owner.zone_type)
    return _MixedLayout(
        list(parts),
        [nodes for _, nodes in parts.values()],
        np.concatenate([cells for cells, _ in parts.values()]),
    )


def _write_zones(reader: TecplotReader, writer: Any, plans: list[_Plan]) -> None:
    """Copy every zone of *reader* to *writer*, retyping the planned ones.

    Args:
        reader: Open reader of the input file.
        writer: Open writer of the output file.
        plans: Output of :func:`_make_plan`. Retyped zones lose any face-neighbor data,
            other zones keep it unless the writer cannot store it.

    Raises:
        NotImplementedError: If a zone is not an ordered, classic FE, mixed or poly
            zone.
    """
    cell_centered = ValueLocation.CELL_CENTERED
    layouts: dict[int, _MixedLayout] = {}
    last_member = {p.root: p.index for p in plans if p.to_mixed}
    for plan, zone in zip(plans, reader.zones, strict=True):
        layout = None
        if plan.to_mixed:
            layout = layouts.get(plan.root)
            if layout is None:
                owner = reader.zones[plan.root]
                assert isinstance(owner, TecplotFEZoneReader)
                layout = layouts[plan.root] = _mixed_layout(owner)
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
                values = var.values
                if layout is not None and var.value_location == cell_centered:
                    values = values[layout.order]
                data.append(values)
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
            if layout is not None:
                writer.write_mixed_zone(
                    data=data,
                    cell_shapes=None if con_sharing is not None else layout.shapes,
                    node_maps=None if con_sharing is not None else layout.node_maps,
                    con_sharing=con_sharing,
                    **common_kw,
                )
                if last_member[plan.root] == plan.index:
                    del layouts[plan.root]  # the group is done, free the node maps
            else:
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
        elif isinstance(zone, TecplotMixedZoneReader):
            con_sharing = zone.shared_connectivity
            writer.write_mixed_zone(
                data=data,
                cell_shapes=None if con_sharing is not None else zone.section_shapes,
                node_maps=None if con_sharing is not None else zone.node_maps,
                con_sharing=con_sharing,
                **common_kw,
            )
        else:
            raise NotImplementedError(
                f"Zone '{zone.title}' is not an ordered, FE, mixed or poly zone; it "
                "cannot be copied."
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

    if args.output is not None:
        dst = Path(args.output)
    else:
        dst = src.with_stem(src.stem + "_retyped")
        if args.mixed:
            dst = dst.with_suffix(".szplt")
    szplt_output = dst.suffix.lower() == ".szplt"
    if args.mixed and not szplt_output:
        print(
            "Error: --mixed needs .szplt output, FEMIXED zones are not supported by "
            f"{dst.suffix or 'this'} files.",
            file=sys.stderr,
        )
        return error
    dry = args.dry_run or args.check
    if not dry and dst.exists() and not args.force:
        print(
            f"Error: output file already exists: {dst}\nUse --force to overwrite.",
            file=sys.stderr,
        )
        return error

    # An SZL file cannot hold face-neighbor data, so there is nothing to protect.
    protect_face_neighbors = not (args.drop_face_neighbors or szplt_output)

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

            analysis = _analyze(reader)
            plans = _make_plan(
                reader,
                analysis,
                selected,
                protect_face_neighbors=protect_face_neighbors,
                mixed=args.mixed,
            )
            _print_plan(plans)
            retypable = any(p.target is not None or p.to_mixed for p in plans)

            if dry:
                command = ["tecelretyp"]
                if args.zones is not None:
                    command += ["-z", ",".join(str(z) for z in args.zones)]
                command.append(shlex.quote(args.filename))
                variants = [
                    (
                        [],
                        _make_plan(
                            reader,
                            analysis,
                            selected,
                            protect_face_neighbors=not szplt_output,
                            mixed=False,
                        ),
                    ),
                    (
                        ["--drop-face-neighbors"],
                        _make_plan(
                            reader,
                            analysis,
                            selected,
                            protect_face_neighbors=False,
                            mixed=False,
                        ),
                    ),
                    (
                        ["--mixed"],
                        _make_plan(
                            reader,
                            analysis,
                            selected,
                            protect_face_neighbors=False,
                            mixed=True,
                        ),
                    ),
                ]
                _print_estimates(src, command, analysis, variants)
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
