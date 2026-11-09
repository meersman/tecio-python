"""Node-map utilities for unstructured finite-element cells.

Many solvers write every volume cell as an ``FEBRICK`` and every surface cell as an
``FEQUADRILATERAL``, repeating node indices to describe tetrahedra, pyramids, prisms and
triangles. The functions here detect such cells, collapse them to their natural zone
type (``FEBRICK`` to ``FETETRAHEDRON``, ``FEQUADRILATERAL`` to ``FETRIANGLE``) and
expand them back.

Detection does not assume a particular repeat convention. The nodes of a cell are
compared pairwise and the resulting pattern is matched against every way a brick or quad
can collapse onto a valid solid (a closed surface made of triangles and quads). Repeats
need not be adjacent in the node list, for example a brick can have a vertical edge
collapsed, which repeats node ``i`` at positions ``i`` and ``i + 4``. When a cell is
collapsed, the handedness of the original cell is kept: the new node order is the order
of first appearance, with the last two nodes swapped where a collapse would otherwise
mirror the cell (this affects 18 of the 102 ways a brick can collapse to a tetrahedron).

Node indices are never changed, so nodal and cell-centered variables stay valid for a
collapsed zone. Only the node map changes.

:func:`cell_face_neighbors` finds, for every face of every cell, the cell on the other
side from the node map alone.

:func:`faces_by_element` and :func:`nodes_by_element` list the faces and the nodes of
every element from the left and right element of each face for polygon or polyhedral
elements.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Iterator
from fractions import Fraction
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from ._constants import FeCellShape, ZoneType
from ._hashing import lowbias32, splitmix64

__all__ = [
    "UNKNOWN_SHAPE",
    "cell_face_neighbors",
    "classify_cells",
    "collapse_cells",
    "collapsed_zone_type",
    "degenerate_edges",
    "expand_cells",
    "shape_counts",
]

#: Value returned by :func:`classify_cells` for a cell that is not a recognized shape.
UNKNOWN_SHAPE = -1

# Cells processed at a time, bounds the size of temporary arrays.
_CHUNK_CELLS = 2**18

_NODES_PER_CELL: dict[ZoneType, int] = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FETETRAHEDRON: 4,
    ZoneType.FEBRICK: 8,
}

# Shape of every cell in a zone type that has no repeated-node variants.
_FIXED_SHAPE: dict[ZoneType, FeCellShape] = {
    ZoneType.FELINESEG: FeCellShape.BAR,
    ZoneType.FETRIANGLE: FeCellShape.TRIANGLE,
    ZoneType.FETETRAHEDRON: FeCellShape.TETRAHEDRON,
}

# Edges of each cell, as pairs of positions in the node list.
_EDGES: dict[ZoneType, tuple[tuple[int, int], ...]] = {
    ZoneType.FELINESEG: ((0, 1),),
    ZoneType.FETRIANGLE: ((0, 1), (1, 2), (2, 0)),
    ZoneType.FEQUADRILATERAL: ((0, 1), (1, 2), (2, 3), (3, 0)),
    ZoneType.FETETRAHEDRON: ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    ZoneType.FEBRICK: (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ),
}

# Brick nodes follow the Tecplot data formatting guide ordering: 0-3 run around the
# bottom face and 4-7 around the top face, with node 4 above node 0.
_BRICK_CORNERS = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)
_BRICK_FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)

# Nodes of each face as positions in the cell's node list.
_FACE_NODES: dict[ZoneType, tuple[tuple[int, ...], ...]] = {
    ZoneType.FETRIANGLE: ((0, 1), (1, 2), (2, 0)),
    ZoneType.FEQUADRILATERAL: ((0, 1), (1, 2), (2, 3), (3, 0)),
    ZoneType.FETETRAHEDRON: ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)),
    ZoneType.FEBRICK: _BRICK_FACES,
}


# Zone type each collapsible zone type can be reduced to, with the cell shape of its
# cells and the node positions used to expand them again.
_COLLAPSE: dict[ZoneType, tuple[ZoneType, FeCellShape]] = {
    ZoneType.FEBRICK: (ZoneType.FETETRAHEDRON, FeCellShape.TETRAHEDRON),
    ZoneType.FEQUADRILATERAL: (ZoneType.FETRIANGLE, FeCellShape.TRIANGLE),
}
_EXPAND: dict[tuple[ZoneType, ZoneType], tuple[int, ...]] = {
    (ZoneType.FETETRAHEDRON, ZoneType.FEBRICK): (0, 1, 2, 2, 3, 3, 3, 3),
    (ZoneType.FETRIANGLE, ZoneType.FEQUADRILATERAL): (0, 1, 2, 2),
}


class _Table(NamedTuple):
    """Collapse patterns of one zone type, keyed by node-equality bit mask."""

    masks: npt.NDArray[np.int64]  # sorted
    shapes: npt.NDArray[np.int8]
    orders: npt.NDArray[np.int8]  # node positions of the collapsed cell, -1 if unused


# --------------------------------------------------------------------------------------
# Pattern tables
# --------------------------------------------------------------------------------------


def _label_strings(n: int) -> Iterator[tuple[int, ...]]:
    """Yield every partition of *n* positions as labels in order of first appearance.

    Yields:
        Tuples of ``n`` labels, where each label first appears after all smaller ones.
    """

    def extend(prefix: tuple[int, ...], used: int) -> Iterator[tuple[int, ...]]:
        if len(prefix) == n:
            yield prefix
            return
        for label in range(used + 1):
            yield from extend((*prefix, label), used + (label == used))

    yield from extend((), 0)


def _polygons(
    labels: tuple[int, ...], faces: tuple[tuple[int, ...], ...]
) -> list[tuple[int, ...]] | None:
    """Return the faces of a collapsed cell as polygons of node labels.

    Returns:
        The faces left with three or more distinct nodes, or ``None`` if a face folds
        over itself (a node repeated at non-adjacent positions of the face).
    """
    polygons = []
    for face in faces:
        ring = [labels[i] for i in face]
        kept = [ring[i] for i in range(len(ring)) if ring[i] != ring[i - 1]]
        if len(kept) <= 2:
            continue  # collapsed to an edge or a point
        if len(set(kept)) != len(kept):
            return None
        polygons.append(tuple(kept))
    return polygons


def _brick_shape(labels: tuple[int, ...]) -> FeCellShape | None:
    """Name the solid a collapsed brick forms.

    Returns:
        The shape, or ``None`` if the collapsed cell is not a closed solid or is not a
        tetrahedron, pyramid, prism or hexahedron.
    """
    polygons = _polygons(labels, _BRICK_FACES)
    if polygons is None:
        return None
    edge_uses: dict[frozenset[int], int] = {}
    for polygon in polygons:
        for i in range(len(polygon)):
            edge = frozenset((polygon[i], polygon[i - 1]))
            edge_uses[edge] = edge_uses.get(edge, 0) + 1
    if any(uses != 2 for uses in edge_uses.values()):
        return None
    num_nodes = len(set(labels))
    triangles = sum(len(p) == 3 for p in polygons)
    quads = sum(len(p) == 4 for p in polygons)
    return {
        (4, 4, 0): FeCellShape.TETRAHEDRON,
        (5, 4, 1): FeCellShape.PYRAMID,
        (6, 2, 3): FeCellShape.PRISM,
        (8, 0, 6): FeCellShape.HEXAHEDRON,
    }.get((num_nodes, triangles, quads))


def _quad_shape(labels: tuple[int, ...]) -> FeCellShape | None:
    """Name the polygon a collapsed quadrilateral forms.

    Returns:
        ``QUADRILATERAL`` or ``TRIANGLE``, or ``None`` if the cell has fewer than three
        distinct nodes or repeats a node at non-adjacent positions.
    """
    polygons = _polygons(labels, ((0, 1, 2, 3),))
    if not polygons:
        return None
    return FeCellShape.TRIANGLE if len(polygons[0]) == 3 else FeCellShape.QUADRILATERAL


def _simpson_volume(corners: list[tuple[int, ...]]) -> Fraction:
    """Return the exact volume of a trilinear hexahedron.

    The Jacobian determinant is quadratic in each coordinate, so a three-point Simpson
    rule integrates it exactly.

    Returns:
        Signed volume, positive for a right-handed brick.
    """
    points = (-1, 0, 1)
    weights = (Fraction(1, 3), Fraction(4, 3), Fraction(1, 3))
    signs = [tuple(2 * c - 1 for c in corner) for corner in _BRICK_CORNERS]
    total = Fraction(0)
    for (a, wa), (b, wb), (c, wc) in itertools.product(
        zip(points, weights, strict=True), repeat=3
    ):
        jac = [[Fraction(0)] * 3 for _ in range(3)]
        for node, (sx, sy, sz) in enumerate(signs):
            slopes = (
                sx * (1 + sy * b) * (1 + sz * c),
                sy * (1 + sx * a) * (1 + sz * c),
                sz * (1 + sx * a) * (1 + sy * b),
            )
            for i in range(3):
                for k in range(3):
                    jac[i][k] += Fraction(slopes[i], 8) * corners[node][k]
        det = (
            jac[0][0] * (jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1])
            - jac[0][1] * (jac[1][0] * jac[2][2] - jac[1][2] * jac[2][0])
            + jac[0][2] * (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0])
        )
        total += wa * wb * wc * det
    return total


def _first_positions(labels: tuple[int, ...]) -> list[int]:
    """Return the first position of each label, in label order."""
    return [labels.index(label) for label in range(max(labels) + 1)]


def _tet_order(labels: tuple[int, ...]) -> tuple[int, ...]:
    """Return the brick positions that give a tetrahedron of the brick's handedness.

    The nodes are taken in order of first appearance. If a unit tetrahedron put on
    those nodes has the opposite sign to the collapsed brick, the last two swap.

    Returns:
        Four brick positions, in the order of the new node map.
    """
    unit = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))  # volume +1/6
    brick_volume = _simpson_volume([unit[label] for label in labels])
    order = _first_positions(labels)
    if brick_volume < 0:
        order[2], order[3] = order[3], order[2]
    return tuple(order)


@functools.cache
def _table(zone_type: ZoneType) -> _Table:
    """Build the collapse patterns of ``FEBRICK`` or ``FEQUADRILATERAL``.

    Returns:
        Table of patterns, sorted by mask, for the zone type.
    """
    n = _NODES_PER_CELL[zone_type]
    pairs = tuple(itertools.combinations(range(n), 2))
    width = 4 if zone_type == ZoneType.FEBRICK else 3
    rows: list[tuple[int, FeCellShape, tuple[int, ...]]] = []
    for labels in _label_strings(n):
        is_brick = zone_type == ZoneType.FEBRICK
        shape = _brick_shape(labels) if is_brick else _quad_shape(labels)
        if shape is None:
            continue
        mask = sum(1 << k for k, (i, j) in enumerate(pairs) if labels[i] == labels[j])
        order: tuple[int, ...] = ()
        if shape == FeCellShape.TETRAHEDRON:
            order = _tet_order(labels)
        elif shape == FeCellShape.TRIANGLE:
            order = tuple(_first_positions(labels))
        rows.append((mask, shape, order))
    rows.sort(key=lambda row: row[0])
    orders = np.full((len(rows), width), -1, dtype=np.int8)
    for r, (_, _, order) in enumerate(rows):
        orders[r, : len(order)] = order
    return _Table(
        np.array([r[0] for r in rows], dtype=np.int64),
        np.array([r[1].value for r in rows], dtype=np.int8),
        orders,
    )


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _as_node_map(
    node_map: npt.ArrayLike, zone_type: ZoneType
) -> npt.NDArray[np.integer]:
    """Validate *node_map* for *zone_type* and shape it ``(num_cells, nodes_per_cell)``.

    Returns:
        The node map as a two-dimensional integer array.

    Raises:
        ValueError: If *zone_type* is not a classic finite-element zone type, the node
            map has the wrong number of nodes per cell, or it is not an integer array.
    """
    if zone_type not in _NODES_PER_CELL:
        raise ValueError(
            f"zone_type must be a classic finite-element zone type, got {zone_type}."
        )
    arr = np.asarray(node_map)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"node_map must hold integers, got dtype {arr.dtype}.")
    nodes_per_cell = _NODES_PER_CELL[zone_type]
    if arr.ndim == 1 and arr.size % nodes_per_cell == 0:
        arr = arr.reshape(-1, nodes_per_cell)
    if arr.ndim != 2 or arr.shape[1] != nodes_per_cell:
        raise ValueError(
            f"node_map for {zone_type.name} must have {nodes_per_cell} nodes per cell, "
            f"got shape {arr.shape}."
        )
    return arr


def _equality_masks(nodes: npt.NDArray[np.integer]) -> npt.NDArray[np.int32]:
    """Pack the pairwise node equalities of every cell into one integer per cell.

    Returns:
        Bit ``k`` is set where the ``k``-th pair of node positions holds the same index.
    """
    mask = np.zeros(len(nodes), dtype=np.int32)
    pairs = itertools.combinations(range(nodes.shape[1]), 2)
    for bit, (i, j) in enumerate(pairs):
        mask |= (nodes[:, i] == nodes[:, j]).astype(np.int32) << bit
    return mask


def _classify_chunk(
    nodes: npt.NDArray[np.integer], zone_type: ZoneType
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.intp]]:
    """Classify the cells of *nodes*.

    Returns:
        The shape value of each cell (:data:`UNKNOWN_SHAPE` if not recognized) and the
        index of its row in the pattern table (meaningful only where the shape is known
        and the zone type has a table).
    """
    masks = _equality_masks(nodes)
    fixed = _FIXED_SHAPE.get(zone_type)
    if fixed is not None:
        shapes = np.where(masks == 0, fixed.value, UNKNOWN_SHAPE).astype(np.int8)
        return shapes, np.zeros(len(nodes), dtype=np.intp)
    table = _table(zone_type)
    rows = np.minimum(np.searchsorted(table.masks, masks), len(table.masks) - 1)
    hit = table.masks[rows] == masks
    shapes = np.where(hit, table.shapes[rows], UNKNOWN_SHAPE).astype(np.int8)
    return shapes, rows


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def classify_cells(
    node_map: npt.ArrayLike, zone_type: ZoneType
) -> npt.NDArray[np.int8]:
    """Identify the shape of every cell, looking through repeated node indices.

    ``FEBRICK`` cells are reported as ``TETRAHEDRON``, ``PYRAMID``, ``PRISM`` or
    ``HEXAHEDRON`` and ``FEQUADRILATERAL`` cells as ``TRIANGLE`` or ``QUADRILATERAL``.
    The other classic zone types report their own shape, unless a cell repeats a node.
    A cell that repeats nodes in any other way (a collapsed edge that leaves a solid
    that is not one of these shapes, or a cell flattened to a face) is unknown.

    Args:
        node_map: Node indices with shape ``(num_cells, nodes_per_cell)``, or the same
            values flattened.
        zone_type: The zone type that *node_map* belongs to.

    Returns:
        Array of :class:`~tecio.FeCellShape` values, one per cell, with
        :data:`UNKNOWN_SHAPE` for a cell that is not recognized.

    Raises:
        ValueError: If *zone_type* is not a classic finite-element type or *node_map*
            does not fit it.
    """
    nodes = _as_node_map(node_map, zone_type)
    shapes = np.empty(len(nodes), dtype=np.int8)
    for lo in range(0, len(nodes), _CHUNK_CELLS):
        hi = lo + _CHUNK_CELLS
        shapes[lo:hi] = _classify_chunk(nodes[lo:hi], zone_type)[0]
    return shapes


def shape_counts(
    node_map: npt.ArrayLike, zone_type: ZoneType
) -> dict[FeCellShape | None, int]:
    """Count the cells of each shape (see :func:`classify_cells`).

    Args:
        node_map: Node indices with shape ``(num_cells, nodes_per_cell)``.
        zone_type: The zone type that *node_map* belongs to.

    Returns:
        Mapping from shape to cell count, ordered by shape value, with ``None`` as the
        key for unrecognized cells. Shapes that do not occur are left out.
    """
    shapes = classify_cells(node_map, zone_type)
    counts = np.bincount(shapes.astype(np.intp) + 1, minlength=len(FeCellShape) + 1)
    result: dict[FeCellShape | None, int] = {}
    if counts[0]:
        result[None] = int(counts[0])
    for shape in FeCellShape:
        if counts[shape.value + 1]:
            result[shape] = int(counts[shape.value + 1])
    return result


def degenerate_edges(
    node_map: npt.ArrayLike, zone_type: ZoneType
) -> npt.NDArray[np.bool_]:
    """Flag the edges of each cell whose two end nodes are the same index.

    Args:
        node_map: Node indices with shape ``(num_cells, nodes_per_cell)``.
        zone_type: The zone type that *node_map* belongs to.

    Returns:
        Boolean array ``(num_cells, num_edges)``. The edges are those of the zone type:
        1 for ``FELINESEG``, 3 for ``FETRIANGLE``, 4 for ``FEQUADRILATERAL``, 6 for
        ``FETETRAHEDRON`` and 12 for ``FEBRICK`` (bottom ring, top ring, then the four
        vertical edges, in the order of Tecplot's node numbering).

    Raises:
        ValueError: If *zone_type* is not a classic finite-element type or *node_map*
            does not fit it.
    """
    nodes = _as_node_map(node_map, zone_type)
    return np.stack([nodes[:, i] == nodes[:, j] for i, j in _EDGES[zone_type]], axis=1)


def collapsed_zone_type(
    node_map: npt.ArrayLike, zone_type: ZoneType
) -> ZoneType | None:
    """Return the zone type a zone can be collapsed to, if every cell allows it.

    Only ``FEBRICK`` zones whose cells are all tetrahedra and ``FEQUADRILATERAL`` zones
    whose cells are all triangles can be collapsed. Pyramids and prisms have no classic
    zone type, so a zone that holds any is left as it is.

    Args:
        node_map: Node indices with shape ``(num_cells, nodes_per_cell)``.
        zone_type: The zone type that *node_map* belongs to.

    Returns:
        ``FETETRAHEDRON`` or ``FETRIANGLE``, or ``None`` if the zone cannot be collapsed
        (including a zone with no cells).

    Raises:
        ValueError: If *zone_type* is not a classic finite-element type or *node_map*
            does not fit it.
    """
    nodes = _as_node_map(node_map, zone_type)
    if zone_type not in _COLLAPSE or len(nodes) == 0:
        return None
    target, shape = _COLLAPSE[zone_type]
    for lo in range(0, len(nodes), _CHUNK_CELLS):
        shapes = _classify_chunk(nodes[lo : lo + _CHUNK_CELLS], zone_type)[0]
        if not np.all(shapes == shape.value):
            return None
    return target


def collapse_cells(
    node_map: npt.ArrayLike, zone_type: ZoneType, target: ZoneType
) -> npt.NDArray[np.integer]:
    """Collapse every cell to a smaller zone type, keeping its handedness.

    ``FEBRICK`` cells that are tetrahedra become ``FETETRAHEDRON`` cells and
    ``FEQUADRILATERAL`` cells that are triangles become ``FETRIANGLE`` cells. The
    repeated nodes are dropped and the rest are kept in order of first appearance, with
    the last two tetrahedron nodes swapped where the order would otherwise mirror the
    cell. Node indices are unchanged.

    Args:
        node_map: Node indices with shape ``(num_cells, nodes_per_cell)``.
        zone_type: The zone type that *node_map* belongs to.
        target: The zone type to collapse to, ``FETETRAHEDRON`` for ``FEBRICK`` and
            ``FETRIANGLE`` for ``FEQUADRILATERAL``.

    Returns:
        New node map with 4 (tetrahedron) or 3 (triangle) nodes per cell and the dtype
        of *node_map*.

    Raises:
        ValueError: If *zone_type* cannot collapse to *target*, *node_map* does not fit
            *zone_type*, or any cell is not a tetrahedron or triangle respectively.
    """
    nodes = _as_node_map(node_map, zone_type)
    if zone_type not in _COLLAPSE or _COLLAPSE[zone_type][0] != target:
        raise ValueError(f"{zone_type.name} cannot be collapsed to {target.name}.")
    shape = _COLLAPSE[zone_type][1]
    table = _table(zone_type)
    out = np.empty((len(nodes), table.orders.shape[1]), dtype=nodes.dtype)
    for lo in range(0, len(nodes), _CHUNK_CELLS):
        chunk = nodes[lo : lo + _CHUNK_CELLS]
        shapes, rows = _classify_chunk(chunk, zone_type)
        wrong = np.flatnonzero(shapes != shape.value)
        if wrong.size:
            raise ValueError(
                f"cannot collapse {zone_type.name} to {target.name}: cell "
                f"{lo + int(wrong[0])} is not a {shape.name.lower()}."
            )
        positions = table.orders[rows].astype(np.intp)
        out[lo : lo + len(chunk)] = np.take_along_axis(chunk, positions, axis=1)
    return out


def expand_cells(
    node_map: npt.ArrayLike, zone_type: ZoneType, target: ZoneType
) -> npt.NDArray[np.integer]:
    """Expand every cell to a larger zone type by repeating nodes.

    ``FETETRAHEDRON`` cells become ``FEBRICK`` cells as ``(n0, n1, n2, n2, n3, n3, n3,
    n3)`` and ``FETRIANGLE`` cells become ``FEQUADRILATERAL`` cells as ``(n0, n1, n2,
    n2)``. The result keeps the handedness of the input and collapses back to it with
    :func:`collapse_cells`.

    Args:
        node_map: Node indices with shape ``(num_cells, nodes_per_cell)``.
        zone_type: The zone type that *node_map* belongs to.
        target: The zone type to expand to, ``FEBRICK`` for ``FETETRAHEDRON`` and
            ``FEQUADRILATERAL`` for ``FETRIANGLE``.

    Returns:
        New node map with 8 (brick) or 4 (quadrilateral) nodes per cell and the dtype
        of *node_map*.

    Raises:
        ValueError: If *zone_type* cannot expand to *target* or *node_map* does not fit
            *zone_type*.
    """
    nodes = _as_node_map(node_map, zone_type)
    columns = _EXPAND.get((zone_type, target))
    if columns is None:
        raise ValueError(f"{zone_type.name} cannot be expanded to {target.name}.")
    return nodes[:, list(columns)]


# --------------------------------------------------------------------------------------
# Face neighbors
# --------------------------------------------------------------------------------------


def _node_hashes(
    num_nodes: int,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint32]]:
    """Return two independent hash values for each node index.

    The 64-bit hash is :func:`~tecio._hashing.splitmix64` of the index. The 32-bit hash
    is :func:`~tecio._hashing.lowbias32` of the index plus one, with the high half
    folded into the low half for indices of 2**32 or more. Both are bijections for
    indices below 2**32.

    Returns:
        A 64-bit and a 32-bit hash per node.
    """
    index = np.arange(num_nodes, dtype=np.uint64)
    one_based = index + np.uint64(1)
    one_based ^= one_based >> np.uint64(32)
    return splitmix64(index), lowbias32(one_based.astype(np.uint32))


def _hash_faces(
    nodes: npt.NDArray[np.integer],
    table: tuple[tuple[int, ...], ...],
    min_nodes: int,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint32], npt.NDArray[np.bool_]]:
    """Hash the set of nodes of every face of every cell.

    A node repeated within a face counts once, so a face of a degenerate cell hashes the
    same as a face with the same distinct nodes. Faces are numbered ``face * num_cells +
    cell``.

    Args:
        nodes: Node map ``(num_cells, nodes_per_cell)``.
        table: Node positions of each face of a cell.
        min_nodes: Fewest distinct nodes a face needs to be a face.

    Returns:
        A 64-bit hash, an independent 32-bit hash, and a flag for faces with at least
        *min_nodes* distinct nodes.
    """
    num_cells = len(nodes)
    wide, narrow = _node_hashes(int(np.max(nodes)) + 1)
    wide_sum = np.empty(len(table) * num_cells, dtype=np.uint64)
    narrow_sum = np.empty(len(table) * num_cells, dtype=np.uint32)
    valid = np.empty(len(table) * num_cells, dtype=bool)
    for lo in range(0, num_cells, _CHUNK_CELLS):
        hi = min(lo + _CHUNK_CELLS, num_cells)
        columns = np.ascontiguousarray(nodes[lo:hi].T)
        wide_node, narrow_node = wide.take(columns), narrow.take(columns)
        for f, positions in enumerate(table):
            out = slice(f * num_cells + lo, f * num_cells + hi)
            total_wide = wide_node[positions[0]].copy()
            total_narrow = narrow_node[positions[0]].copy()
            distinct = np.ones(hi - lo, dtype=np.int8)
            for j in range(1, len(positions)):
                repeated = columns[positions[j]] == columns[positions[0]]
                for k in range(1, j):
                    repeated |= columns[positions[j]] == columns[positions[k]]
                new = ~repeated
                np.add(total_wide, wide_node[positions[j]], out=total_wide, where=new)
                np.add(
                    total_narrow, narrow_node[positions[j]], out=total_narrow, where=new
                )
                distinct += new
            wide_sum[out] = total_wide
            narrow_sum[out] = total_narrow
            valid[out] = distinct >= min_nodes
    return wide_sum, narrow_sum, valid


def _face_rows(
    nodes: npt.NDArray[np.integer], table: npt.NDArray[np.intp], face_ids: npt.NDArray
) -> npt.NDArray[np.integer]:
    """Return the node indices of the faces with the given face-major ids."""
    face, cell = np.divmod(face_ids, len(nodes))
    flat = nodes.reshape(-1)
    return flat.take((cell * nodes.shape[1])[:, None] + table[face])


def _pair_exact(
    nodes: npt.NDArray[np.integer],
    table: npt.NDArray[np.intp],
    face_ids: npt.NDArray[np.intp],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Pair up faces with the same distinct nodes, comparing the nodes themselves.

    Used for the faces whose hashes alone leave doubt: a node set shared by more than
    two cells, or a hash collision.

    Returns:
        The two face ids of every node set that exactly two of *face_ids* have.
    """
    rows = _face_rows(nodes, table, face_ids)
    rows.sort(axis=1)
    repeated = np.zeros(rows.shape, dtype=bool)
    repeated[:, 1:] = rows[:, 1:] == rows[:, :-1]
    rows[repeated] = -1
    rows.sort(axis=1)
    order = np.lexsort(rows.T[::-1])
    rows = rows[order]
    starts = np.flatnonzero(np.r_[True, (rows[1:] != rows[:-1]).any(axis=1), True])
    first = starts[:-1][np.diff(starts) == 2]
    return face_ids[order[first]], face_ids[order[first + 1]]


def cell_face_neighbors(
    node_map: npt.ArrayLike, zone_type: ZoneType
) -> npt.NDArray[np.int64]:
    """Find the cell on the other side of every face of every cell.

    Two cells are neighbors across a face when they have the same nodes on that face. A
    node repeated within a face counts once, so cells written as bricks with repeated
    nodes (see :func:`classify_cells`) match their neighbors through the faces that
    remain. Faces collapsed to an edge or a point have no neighbor, and neither does a
    face shared by more than two cells. Nodes are compared by index, not by position, so
    nodes at the same location with different indices are not merged.

    Faces are matched by sorting a 96-bit hash of each face's node set, which makes the
    cost grow as ``num_cells * log(num_cells)``. Doubt about a match (a node set shared
    by more than two cells) is settled by comparing the nodes themselves.

    The faces of ``FETRIANGLE`` and ``FEQUADRILATERAL`` cells are the edges from node
    ``i`` to node ``i + 1``. The order of the faces of ``FETETRAHEDRON`` and ``FEBRICK``
    cells is provisional: it has not yet been checked against the face numbering figures
    of the Tecplot data format guide, so the neighbor across each face is certain, but
    the column it is reported in is not.

    Args:
        node_map: Node indices with shape ``(num_cells, nodes_per_cell)``.
        zone_type: One of ``FETRIANGLE``, ``FEQUADRILATERAL``, ``FETETRAHEDRON`` or
            ``FEBRICK``.

    Returns:
        Array ``(num_cells, num_faces)`` holding the 0-based index of the neighbor of
        each face, or ``-1`` where there is none.

    Raises:
        ValueError: If *zone_type* has no faces to match (``FELINESEG`` or a zone type
            that is not finite-element), *node_map* does not fit it, or a node index is
            negative.
    """
    nodes = _as_node_map(node_map, zone_type)
    if zone_type not in _FACE_NODES:
        raise ValueError(f"{zone_type.name} cells have no faces to match.")
    table = np.array(_FACE_NODES[zone_type], dtype=np.intp)
    num_cells, num_faces = len(nodes), len(table)
    if num_cells == 0:
        return np.full((0, num_faces), -1, dtype=np.int64)
    if np.min(nodes) < 0:
        raise ValueError("node_map holds negative node indices.")
    nodes = np.ascontiguousarray(nodes)

    wide, narrow, valid = _hash_faces(
        nodes, _FACE_NODES[zone_type], 2 if table.shape[1] == 2 else 3
    )
    if valid.all():
        keys = wide
        order = np.argsort(keys)
        ids = order
    else:
        usable = np.flatnonzero(valid)
        keys = wide[usable]
        order = np.argsort(keys)
        ids = usable[order]
    keys = keys[order]
    del order, valid

    # A run of equal hashes is a pair of faces, or a collision or a node set shared by
    # many cells.
    equal = keys[1:] == keys[:-1]
    follows = np.zeros(len(keys), dtype=bool)
    follows[1:] = equal
    precedes = np.zeros(len(keys), dtype=bool)
    precedes[:-1] = equal
    first = precedes & ~follows
    first[:-1] &= ~precedes[1:]
    start = np.flatnonzero(first)
    in_pair = np.zeros(len(keys), dtype=bool)
    in_pair[start] = in_pair[start + 1] = True
    in_run = follows | precedes
    ambiguous = ids[in_run & ~in_pair]
    a, b = ids[start], ids[start + 1]
    confirmed = narrow[a] == narrow[b]
    a, b = a[confirmed], b[confirmed]
    if ambiguous.size:
        extra_a, extra_b = _pair_exact(nodes, table, ambiguous)
        a, b = np.r_[a, extra_a], np.r_[b, extra_b]
    cell_a, cell_b = a % num_cells, b % num_cells
    other = cell_a != cell_b  # a cell cannot be its own neighbor
    a, b, cell_a, cell_b = a[other], b[other], cell_a[other], cell_b[other]
    neighbors = np.full(num_faces * num_cells, -1, dtype=np.int64)
    neighbors[a] = cell_b
    neighbors[b] = cell_a
    return np.ascontiguousarray(neighbors.reshape(num_faces, num_cells).T)


# --------------------------------------------------------------------------------------
# Elements of face-based zones
# --------------------------------------------------------------------------------------


def faces_by_element(
    face_left: npt.ArrayLike, face_right: npt.ArrayLike, num_elements: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """List the faces of every element of a face-based zone.

    Each face names the element on its left and on its right. This regroups that into
    one list of faces per element. A side that is not an element of the zone (``-1`` for
    none, or a boundary reference below that) is skipped. Within an element the faces
    are in ascending order.

    The cost is one stable sort of the ``2 * num_faces`` sides. No faces are matched.

    Args:
        face_left: Element on the left of each face, shape ``(num_faces,)``.
        face_right: Element on the right of each face, same shape.
        num_elements: Number of elements of the zone.

    Returns:
        The ``offsets`` with shape ``(num_elements + 1,)`` and the flat ``faces``. The
        faces of element ``e`` are ``faces[offsets[e]:offsets[e + 1]]``.

    Raises:
        ValueError: If the arrays are not one dimensional and of the same length, or an
            element index is not below *num_elements*.
    """
    left = np.asarray(face_left, dtype=np.int64)
    right = np.asarray(face_right, dtype=np.int64)
    if left.ndim != 1 or left.shape != right.shape:
        raise ValueError("face_left and face_right must be 1-D arrays of one length.")
    sides = np.empty(2 * left.size, dtype=np.int64)  # face f is at 2 * f and 2 * f + 1
    sides[0::2] = left
    sides[1::2] = right
    entries = np.flatnonzero(sides >= 0)
    elements = sides[entries]
    if elements.size and np.max(elements) >= num_elements:
        raise ValueError(f"An element index is not below num_elements={num_elements}.")
    order = np.argsort(elements, kind="stable")
    faces = (entries[order] >> 1).astype(np.int64, copy=False)
    offsets = np.zeros(num_elements + 1, dtype=np.int64)
    np.cumsum(np.bincount(elements, minlength=num_elements), out=offsets[1:])
    return offsets, faces


def _unique_pairs(
    first: npt.NDArray[np.int64],
    second: npt.NDArray[np.int64],
    num_first: int,
    num_second: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return the distinct ``(first, second)`` pairs, sorted by *first* then *second*.

    Args:
        first: Values in ``[0, num_first)``.
        second: Values in ``[0, num_second)``, the same length as *first*.
        num_first: Exclusive upper bound of *first*.
        num_second: Exclusive upper bound of *second*.
    """
    if first.size == 0:
        return first, second
    if num_first * num_second < 2**62:  # one integer key per pair is the fast path
        key = np.unique(first * num_second + second)
        return np.divmod(key, num_second)
    order = np.lexsort((second, first))
    first, second = first[order], second[order]
    keep = np.ones(first.size, dtype=bool)
    keep[1:] = (first[1:] != first[:-1]) | (second[1:] != second[:-1])
    return first[keep], second[keep]


def nodes_by_element(
    face_nodes: npt.ArrayLike,
    face_node_offsets: npt.ArrayLike,
    element_offsets: npt.NDArray[np.int64],
    element_faces: npt.NDArray[np.int64],
    num_nodes: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """List the distinct nodes of every element of a face-based zone.

    An element is defined by its faces, so its nodes are the nodes of its faces with
    repeats removed. They come out in ascending order. That is a set, not a cell node
    order: a polyhedron has no natural node order.

    Args:
        face_nodes: Flat node indices of all faces.
        face_node_offsets: Running offsets into *face_nodes*, ``(num_faces + 1,)``.
        element_offsets: Offsets from :func:`faces_by_element`.
        element_faces: Faces from :func:`faces_by_element`.
        num_nodes: Number of nodes of the zone.

    Returns:
        The ``offsets`` with shape ``(num_elements + 1,)`` and the flat ``nodes``. The
        nodes of element ``e`` are ``nodes[offsets[e]:offsets[e + 1]]``.

    Raises:
        ValueError: If a node index is outside ``[0, num_nodes)``.
    """
    flat = np.asarray(face_nodes, dtype=np.int64)
    starts = np.asarray(face_node_offsets, dtype=np.int64)
    num_elements = len(element_offsets) - 1
    counts = (starts[1:] - starts[:-1])[element_faces]  # nodes of each listed face
    # Position in `flat` of every node of every listed face.
    take = np.arange(int(counts.sum()), dtype=np.int64)
    take += np.repeat(starts[element_faces] - (np.cumsum(counts) - counts), counts)
    nodes = flat[take]
    if nodes.size and (np.min(nodes) < 0 or np.max(nodes) >= num_nodes):
        raise ValueError(f"A node index is outside [0, {num_nodes}).")
    elements = np.repeat(
        np.arange(num_elements, dtype=np.int64), np.diff(element_offsets)
    )
    elements, nodes = _unique_pairs(
        np.repeat(elements, counts), nodes, num_elements, num_nodes
    )
    offsets = np.zeros(num_elements + 1, dtype=np.int64)
    np.cumsum(np.bincount(elements, minlength=num_elements), out=offsets[1:])
    return offsets, nodes
