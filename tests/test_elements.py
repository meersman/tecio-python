#!/usr/bin/env python3
"""pytest test suite for the node-map utilities and the ``tecelretyp`` CLI tool.

The node-map tests need no Tecplot library. They enumerate every way a brick can
collapse onto a valid solid (4140 partitions of its 8 node positions), and check the
tetrahedra against the exact volume of the degenerate trilinear brick. The CLI tests
write small synthetic files in all three formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tecio
from tecio import FaceNeighborMode, FeCellShape, ValueLocation, ZoneType
from tecio import _elements as elements
from tecio._hashing import lowbias32, splitmix64
from tecio.cli.tecelretyp import main as tecelretyp

BRICK = ZoneType.FEBRICK
TET = ZoneType.FETETRAHEDRON
QUAD = ZoneType.FEQUADRILATERAL
TRI = ZoneType.FETRIANGLE
_FORMATS = ["szplt", "plt", "dat"]

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

_CORNERS = np.array(
    [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)]
    + [(0, 1, 1)],
    dtype=float,
)


def _brick_patterns(shape: FeCellShape) -> list[tuple[int, ...]]:
    """Return every collapse pattern of a brick that forms *shape*.

    Returns:
        Node-position labels of each pattern.
    """
    patterns = list(elements._label_strings(8))
    found = tecio.classify_cells(np.array(patterns), BRICK)
    return [p for p, s in zip(patterns, found, strict=True) if s == shape.value]


def _hex_volume(corners: np.ndarray) -> float:
    """Volume of a trilinear hexahedron by 2 x 2 x 2 Gauss quadrature (exact).

    Returns:
        Signed volume, positive for a right-handed brick.
    """
    gauss = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    signs = 2 * _CORNERS - 1
    volume = 0.0
    for a in gauss:
        for b in gauss:
            for c in gauss:
                slopes = (
                    np.stack(
                        [
                            signs[:, 0] * (1 + signs[:, 1] * b) * (1 + signs[:, 2] * c),
                            signs[:, 1] * (1 + signs[:, 0] * a) * (1 + signs[:, 2] * c),
                            signs[:, 2] * (1 + signs[:, 0] * a) * (1 + signs[:, 1] * b),
                        ],
                        axis=1,
                    )
                    / 8
                )
                volume += np.linalg.det(slopes.T @ corners)
    return volume


def _tet_volumes(xyz: np.ndarray, node_map: np.ndarray) -> np.ndarray:
    """Signed volumes of tetrahedra, positive for a right-handed node order.

    Returns:
        One volume per cell.
    """
    a, b, c, d = (xyz[node_map[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def _triangle_areas(xy: np.ndarray, node_map: np.ndarray) -> np.ndarray:
    """Signed areas of triangles (or of degenerate quads), positive counterclockwise.

    Returns:
        One area per cell.
    """
    ring = xy[node_map]
    nxt = np.roll(ring, -1, axis=1)
    return 0.5 * (ring[..., 0] * nxt[..., 1] - nxt[..., 0] * ring[..., 1]).sum(axis=1)


# --------------------------------------------------------------------------------------
# Node-map functions
# --------------------------------------------------------------------------------------


class TestPatterns:
    """The collapse patterns found by walking every partition of a cell's nodes."""

    def test_brick_pattern_counts(self) -> None:
        """A brick collapses to 102 tetrahedra, 102 pyramids and 36 prisms."""
        assert len(_brick_patterns(FeCellShape.TETRAHEDRON)) == 102
        assert len(_brick_patterns(FeCellShape.PYRAMID)) == 102
        assert len(_brick_patterns(FeCellShape.PRISM)) == 36
        assert len(_brick_patterns(FeCellShape.HEXAHEDRON)) == 1

    def test_quad_pattern_counts(self) -> None:
        """A quad collapses to a triangle in 4 ways."""
        patterns = np.array(list(elements._label_strings(4)))
        shapes = tecio.classify_cells(patterns, QUAD)
        assert np.count_nonzero(shapes == FeCellShape.TRIANGLE.value) == 4
        assert np.count_nonzero(shapes == FeCellShape.QUADRILATERAL.value) == 1

    def test_only_six_tetrahedron_collapses_repeat_adjacent_nodes(self) -> None:
        """Dropping repeats rarely gives the node order: 96 of 102 repeat apart."""

        def runs(labels: tuple[int, ...]) -> bool:
            return all(
                labels[i + 1] in (labels[i], max(labels[: i + 1]) + 1) for i in range(7)
            )

        patterns = _brick_patterns(FeCellShape.TETRAHEDRON)
        assert sum(runs(p) for p in patterns) == 6

    def test_exactly_18_tetrahedron_collapses_need_a_swap(self) -> None:
        """Node order of first appearance mirrors the cell in 18 of 102 cases."""
        patterns = _brick_patterns(FeCellShape.TETRAHEDRON)
        collapsed = tecio.collapse_cells(np.array(patterns), BRICK, TET)
        mirrored = [p for p, c in zip(patterns, collapsed, strict=True) if c[2] > c[3]]
        assert len(mirrored) == 18

    def test_collapse_matches_exact_brick_volume(self) -> None:
        """Every tetrahedron keeps the sign and size of the brick it came from."""
        rng = np.random.default_rng(7)
        patterns = np.array(_brick_patterns(FeCellShape.TETRAHEDRON))
        collapsed = tecio.collapse_cells(patterns, BRICK, TET)
        for nodes, tet in zip(patterns, collapsed, strict=True):
            xyz = rng.normal(size=(4, 3))
            brick = _hex_volume(xyz[nodes])
            assert _tet_volumes(xyz, tet[None, :])[0] == pytest.approx(brick, abs=1e-9)

    def test_collapse_matches_exact_area(self) -> None:
        """Every triangle keeps the sign and size of the quadrilateral."""
        rng = np.random.default_rng(3)
        patterns = np.array([
            p
            for p in elements._label_strings(4)
            if tecio.classify_cells(np.array([p]), QUAD)[0]
            == FeCellShape.TRIANGLE.value
        ])
        collapsed = tecio.collapse_cells(patterns, QUAD, TRI)
        for nodes, tri in zip(patterns, collapsed, strict=True):
            xy = rng.normal(size=(3, 2))
            assert _triangle_areas(xy, tri[None, :])[0] == pytest.approx(
                _triangle_areas(xy, nodes[None, :])[0]
            )


class TestClassify:
    """classify_cells, shape_counts and degenerate_edges."""

    def test_named_conventions(self) -> None:
        """Layouts that solvers use for each shape are recognized."""
        cells = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7],  # hexahedron
            [0, 1, 2, 2, 3, 3, 3, 3],  # tetrahedron, repeats adjacent
            [0, 0, 0, 0, 1, 2, 3, 3],  # tetrahedron, apex first
            [0, 0, 0, 0, 1, 2, 3, 1],  # tetrahedron, repeats apart
            [0, 0, 0, 0, 4, 5, 6, 7],  # pyramid, apex first
            [0, 1, 2, 3, 4, 4, 4, 4],  # pyramid, apex last
            [0, 1, 2, 2, 3, 4, 5, 5],  # prism
        ])
        shapes = [FeCellShape(s) for s in tecio.classify_cells(cells, BRICK)]
        assert shapes == [
            FeCellShape.HEXAHEDRON,
            FeCellShape.TETRAHEDRON,
            FeCellShape.TETRAHEDRON,
            FeCellShape.TETRAHEDRON,
            FeCellShape.PYRAMID,
            FeCellShape.PYRAMID,
            FeCellShape.PRISM,
        ]

    def test_unrecognized_cells(self) -> None:
        """Flat cells, folded faces and one collapsed edge are unknown."""
        cells = np.array([
            [0, 1, 2, 3, 3, 3, 3, 3],  # flat, a single face
            [0, 1, 0, 2, 3, 4, 5, 6],  # bottom face repeats a node across
            [0, 0, 2, 3, 4, 5, 6, 7],  # one collapsed edge, 7 nodes
            [0, 0, 0, 0, 0, 0, 0, 0],  # a point
        ])
        assert set(tecio.classify_cells(cells, BRICK)) == {elements.UNKNOWN_SHAPE}

    def test_quad_and_fixed_zone_types(self) -> None:
        """Quads classify as triangles; other zone types report their shape."""
        quads = np.array([[0, 1, 2, 3], [0, 1, 2, 2], [3, 0, 1, 3], [0, 1, 0, 2]])
        assert list(tecio.classify_cells(quads, QUAD)) == [
            FeCellShape.QUADRILATERAL.value,
            FeCellShape.TRIANGLE.value,
            FeCellShape.TRIANGLE.value,
            elements.UNKNOWN_SHAPE,
        ]
        tets = np.array([[0, 1, 2, 3], [0, 1, 2, 2]])
        assert list(tecio.classify_cells(tets, TET)) == [
            FeCellShape.TETRAHEDRON.value,
            elements.UNKNOWN_SHAPE,
        ]
        assert list(tecio.classify_cells([[0, 1], [4, 4]], ZoneType.FELINESEG)) == [
            FeCellShape.BAR.value,
            elements.UNKNOWN_SHAPE,
        ]
        assert list(tecio.classify_cells([[0, 1, 2]], TRI)) == [
            FeCellShape.TRIANGLE.value
        ]

    def test_shape_counts(self) -> None:
        """Counts are keyed by shape, with None for unrecognized cells."""
        cells = np.array(
            [[0, 1, 2, 2, 3, 3, 3, 3]] * 3
            + [[0, 0, 0, 0, 1, 2, 3, 4]]
            + [[0, 1, 2, 3, 3, 3, 3, 3]] * 2
        )
        assert tecio.shape_counts(cells, BRICK) == {
            None: 2,
            FeCellShape.TETRAHEDRON: 3,
            FeCellShape.PYRAMID: 1,
        }

    def test_degenerate_edges(self) -> None:
        """Edges are flagged where both end nodes are the same index."""
        flags = tecio.degenerate_edges([[0, 1, 2, 2, 3, 3, 3, 3]], BRICK)
        assert flags.shape == (1, 12)
        # edge 2 (nodes 2-3) and the whole top ring (edges 4-7) have collapsed
        assert list(np.flatnonzero(flags[0])) == [2, 4, 5, 6, 7]
        assert not tecio.degenerate_edges([[0, 1, 2, 3, 4, 5, 6, 7]], BRICK).any()
        assert tecio.degenerate_edges([[0, 1, 2, 2]], QUAD).tolist() == [
            [False, False, True, False]
        ]
        assert tecio.degenerate_edges([[0, 1, 2, 3]], TET).shape == (1, 6)
        assert tecio.degenerate_edges([[0, 1, 2]], TRI).shape == (1, 3)
        assert tecio.degenerate_edges([[0, 1]], ZoneType.FELINESEG).shape == (1, 1)

    def test_flat_node_map_is_accepted(self) -> None:
        """A flattened node map is reshaped using the zone type."""
        flat = np.array([0, 1, 2, 2, 3, 3, 3, 3, 0, 1, 2, 3, 4, 5, 6, 7])
        assert list(tecio.classify_cells(flat, BRICK)) == [
            FeCellShape.TETRAHEDRON.value,
            FeCellShape.HEXAHEDRON.value,
        ]

    def test_empty_node_map(self) -> None:
        """No cells give empty results."""
        empty = np.empty((0, 8), dtype=np.int64)
        assert tecio.classify_cells(empty, BRICK).size == 0
        assert tecio.shape_counts(empty, BRICK) == {}
        assert tecio.collapsed_zone_type(empty, BRICK) is None

    def test_invalid_input(self) -> None:
        """Wrong zone types, widths and dtypes are rejected."""
        with pytest.raises(ValueError, match="classic finite-element"):
            tecio.classify_cells([[0, 1]], ZoneType.ORDERED)
        with pytest.raises(ValueError, match="8 nodes per cell"):
            tecio.classify_cells(np.zeros((2, 4), dtype=int), BRICK)
        with pytest.raises(ValueError, match="integers"):
            tecio.classify_cells(np.zeros((2, 8)), BRICK)

    def test_chunking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Results do not depend on how many cells are processed at a time."""
        rng = np.random.default_rng(0)
        cells = rng.integers(0, 3, size=(50, 8))
        cells[::3] = [0, 1, 2, 2, 3, 3, 3, 3]
        expected = tecio.classify_cells(cells, BRICK)
        monkeypatch.setattr(elements, "_CHUNK_CELLS", 7)
        np.testing.assert_array_equal(tecio.classify_cells(cells, BRICK), expected)


class TestCollapseExpand:
    """collapsed_zone_type, collapse_cells and expand_cells."""

    def test_collapse_standard_tetrahedra(self) -> None:
        """The Tecplot layout (n0, n1, n2, n2, n3, n3, n3, n3) gives (n0..n3)."""
        cells = np.array([[5, 6, 7, 7, 8, 8, 8, 8], [1, 2, 3, 3, 9, 9, 9, 9]])
        assert tecio.collapsed_zone_type(cells, BRICK) == TET
        np.testing.assert_array_equal(
            tecio.collapse_cells(cells, BRICK, TET), [[5, 6, 7, 8], [1, 2, 3, 9]]
        )

    def test_collapse_keeps_handedness_for_a_mirrored_pattern(self) -> None:
        """Dropping repeats from (a, a, b, b, a, a, c, d) would mirror the cell."""
        xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1.0]])
        cell = np.array([[1, 1, 2, 2, 1, 1, 4, 3]])
        assert _hex_volume(xyz[cell[0]]) > 0
        out = tecio.collapse_cells(cell, BRICK, TET)
        np.testing.assert_array_equal(out, [[1, 2, 3, 4]])
        assert _tet_volumes(xyz, out)[0] > 0

    def test_collapse_quads(self) -> None:
        """Repeats at the end, middle and wrapped around all collapse."""
        cells = np.array([[0, 1, 2, 2], [0, 0, 2, 3], [0, 1, 1, 3], [3, 0, 1, 3]])
        assert tecio.collapsed_zone_type(cells, QUAD) == TRI
        np.testing.assert_array_equal(
            tecio.collapse_cells(cells, QUAD, TRI),
            [[0, 1, 2], [0, 2, 3], [0, 1, 3], [3, 0, 1]],
        )

    def test_dtype_is_kept(self) -> None:
        """Output has the dtype of the input."""
        cells = np.array([[0, 1, 2, 2, 3, 3, 3, 3]], dtype=np.int32)
        assert tecio.collapse_cells(cells, BRICK, TET).dtype == np.int32
        assert tecio.expand_cells(cells[:, :4], TET, BRICK).dtype == np.int32

    def test_zone_that_cannot_collapse(self) -> None:
        """A zone with any pyramid, prism, hexahedron or odd cell stays as it is."""
        tet = [0, 1, 2, 2, 3, 3, 3, 3]
        for other in (
            [0, 0, 0, 0, 1, 2, 3, 4],  # pyramid
            [0, 1, 2, 2, 3, 4, 5, 5],  # prism
            [0, 1, 2, 3, 4, 5, 6, 7],  # hexahedron
            [0, 1, 2, 3, 3, 3, 3, 3],  # flat
        ):
            assert tecio.collapsed_zone_type(np.array([tet, other]), BRICK) is None
        assert tecio.collapsed_zone_type([[0, 1, 2, 3]], TET) is None
        assert tecio.collapsed_zone_type([[0, 1, 2, 3]], QUAD) is None

    def test_collapse_rejects_wrong_cells_and_targets(self) -> None:
        """Collapsing needs every cell to be a tetrahedron or triangle."""
        cells = np.array([[0, 1, 2, 2, 3, 3, 3, 3], [0, 1, 2, 3, 4, 5, 6, 7]])
        with pytest.raises(ValueError, match="cell 1 is not a tetrahedron"):
            tecio.collapse_cells(cells, BRICK, TET)
        with pytest.raises(ValueError, match="cannot be collapsed"):
            tecio.collapse_cells(cells, BRICK, TRI)
        with pytest.raises(ValueError, match="cannot be collapsed"):
            tecio.collapse_cells([[0, 1, 2, 3]], TET, TRI)

    def test_chunked_error_reports_global_cell_index(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The offending cell is numbered within the whole zone."""
        monkeypatch.setattr(elements, "_CHUNK_CELLS", 4)
        cells = np.tile([0, 1, 2, 2, 3, 3, 3, 3], (10, 1))
        cells[6] = [0, 1, 2, 3, 4, 5, 6, 7]
        with pytest.raises(ValueError, match="cell 6 is not"):
            tecio.collapse_cells(cells, BRICK, TET)
        cells[6] = [0, 1, 2, 2, 3, 3, 3, 3]
        assert tecio.collapsed_zone_type(cells, BRICK) == TET
        assert tecio.collapse_cells(cells, BRICK, TET).shape == (10, 4)
        cells[9] = [0, 1, 2, 3, 3, 3, 3, 3]
        assert tecio.collapsed_zone_type(cells, BRICK) is None

    def test_expand_and_collapse_round_trip(self) -> None:
        """expand_cells inverts collapse_cells for tetrahedra and triangles."""
        tets = np.array([[0, 1, 2, 3], [4, 3, 2, 1]])
        up = tecio.expand_cells(tets, TET, BRICK)
        np.testing.assert_array_equal(up[0], [0, 1, 2, 2, 3, 3, 3, 3])
        np.testing.assert_array_equal(tecio.collapse_cells(up, BRICK, TET), tets)
        tris = np.array([[0, 1, 2], [5, 4, 3]])
        up = tecio.expand_cells(tris, TRI, QUAD)
        np.testing.assert_array_equal(up[1], [5, 4, 3, 3])
        np.testing.assert_array_equal(tecio.collapse_cells(up, QUAD, TRI), tris)

    def test_expand_rejects_unsupported_targets(self) -> None:
        """Only tetrahedra to bricks and triangles to quads are available."""
        with pytest.raises(ValueError, match="cannot be expanded"):
            tecio.expand_cells([[0, 1, 2, 3]], TET, QUAD)
        with pytest.raises(ValueError, match="cannot be expanded"):
            tecio.expand_cells([[0, 1, 2]], TRI, BRICK)

    def test_public_names_report_tecio(self) -> None:
        """Functions show their public module in help and docs."""
        assert tecio.collapse_cells.__module__ == "tecio"
        assert "classify_cells" in tecio.__all__


# --------------------------------------------------------------------------------------
# Face neighbors
# --------------------------------------------------------------------------------------


def _cube_mesh(n: int, shuffle: bool = False) -> np.ndarray:
    """Return the node map of an ``n`` cubed brick mesh, optionally shuffled."""
    idx = np.arange((n + 1) ** 3).reshape(n + 1, n + 1, n + 1)
    i, j, k = (a.ravel() for a in np.meshgrid(*[np.arange(n)] * 3, indexing="ij"))
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    corners += [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    node_map = np.stack([idx[i + a, j + b, k + c] for a, b, c in corners], axis=1)
    if shuffle:
        rng = np.random.default_rng(1)
        node_map = rng.permutation((n + 1) ** 3)[node_map]
        node_map = node_map[rng.permutation(len(node_map))]
    return node_map


def _kuhn_tets(n: int, shuffle: bool = False) -> np.ndarray:
    """Return the node map of a conforming six tetrahedra per cube mesh."""
    bricks = _cube_mesh(n, shuffle)
    paths = [(0, 1, 2, 6), (0, 1, 5, 6), (0, 3, 2, 6), (0, 3, 7, 6)]
    paths += [(0, 4, 5, 6), (0, 4, 7, 6)]
    return np.concatenate([bricks[:, p] for p in paths])


def _brute_neighbors(node_map: np.ndarray, zone_type: ZoneType) -> np.ndarray:
    """Return neighbors found with a dict keyed by the set of face nodes."""
    faces = elements._FACE_NODES[zone_type]
    min_nodes = 2 if len(faces[0]) == 2 else 3
    seen: dict[frozenset[int], list[tuple[int, int]]] = {}
    for cell, row in enumerate(node_map):
        for face, positions in enumerate(faces):
            key = frozenset(int(row[p]) for p in positions)
            if len(key) >= min_nodes:
                seen.setdefault(key, []).append((cell, face))
    out = np.full((len(node_map), len(faces)), -1, dtype=np.int64)
    for pair in seen.values():
        if len(pair) == 2 and pair[0][0] != pair[1][0]:
            (c0, f0), (c1, f1) = pair
            out[c0, f0], out[c1, f1] = c1, c0
    return out


class TestNodeHashes:
    """Tests of :func:`tecio._elements._node_hashes`."""

    def test_uses_the_hash_functions(self) -> None:
        wide, narrow = elements._node_hashes(1000)
        index = np.arange(1000, dtype=np.uint64)
        assert np.array_equal(wide, splitmix64(index))
        assert np.array_equal(narrow, lowbias32((index + 1).astype(np.uint32)))

    def test_dtypes(self) -> None:
        wide, narrow = elements._node_hashes(10)
        assert wide.dtype == np.uint64
        assert narrow.dtype == np.uint32

    def test_hashes_do_not_repeat(self) -> None:
        """Both hash functions are bijections, so distinct nodes never collide."""
        wide, narrow = elements._node_hashes(1 << 18)
        assert len(np.unique(wide)) == len(wide)
        assert len(np.unique(narrow)) == len(narrow)

    def test_empty(self) -> None:
        wide, narrow = elements._node_hashes(0)
        assert wide.size == narrow.size == 0


class TestCellFaceNeighbors:
    """Tests of :func:`tecio.cell_face_neighbors`."""

    @pytest.mark.parametrize("shuffle", [False, True])
    def test_bricks_match_reference(self, shuffle: bool) -> None:
        node_map = _cube_mesh(5, shuffle)
        result = tecio.cell_face_neighbors(node_map, BRICK)
        assert result.dtype == np.int64
        assert result.shape == (125, 6)
        assert np.array_equal(result, _brute_neighbors(node_map, BRICK))

    @pytest.mark.parametrize("shuffle", [False, True])
    def test_tets_match_reference(self, shuffle: bool) -> None:
        node_map = _kuhn_tets(4, shuffle)
        result = tecio.cell_face_neighbors(node_map, TET)
        assert np.array_equal(result, _brute_neighbors(node_map, TET))

    def test_boundary_face_counts(self) -> None:
        n = 4
        bricks = tecio.cell_face_neighbors(_cube_mesh(n), BRICK)
        assert (bricks < 0).sum() == 6 * n * n
        tets = tecio.cell_face_neighbors(_kuhn_tets(n), TET)
        assert (tets < 0).sum() == 12 * n * n

    def test_neighbors_are_symmetric(self) -> None:
        result = tecio.cell_face_neighbors(_kuhn_tets(4, True), TET)
        cells, faces = np.nonzero(result >= 0)
        other = result[cells, faces]
        assert all(cell in result[o] for cell, o in zip(cells, other, strict=True))

    def test_upcast_tets_have_same_neighbors(self) -> None:
        tets = _kuhn_tets(3, True)
        up = tecio.expand_cells(tets, TET, BRICK)
        small = tecio.cell_face_neighbors(tets, TET)
        big = tecio.cell_face_neighbors(up, BRICK)
        assert np.array_equal(big, _brute_neighbors(up, BRICK))
        assert np.array_equal((big >= 0).sum(axis=1), (small >= 0).sum(axis=1))
        assert sorted(big[big >= 0]) == sorted(small[small >= 0])

    @pytest.mark.parametrize(
        ("zone_type", "width"), [(BRICK, 8), (TET, 4), (QUAD, 4), (TRI, 3)]
    )
    @pytest.mark.parametrize("num_nodes", [3, 5, 9])
    def test_random_connectivity(
        self, zone_type: ZoneType, width: int, num_nodes: int
    ) -> None:
        """Repeated nodes, shared node sets and faces shared by many cells."""
        rng = np.random.default_rng(num_nodes)
        node_map = rng.integers(0, num_nodes, size=(300, width))
        result = tecio.cell_face_neighbors(node_map, zone_type)
        assert np.array_equal(result, _brute_neighbors(node_map, zone_type))

    def test_quads_and_triangles(self) -> None:
        quads = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
        assert tecio.cell_face_neighbors(quads, QUAD).tolist() == [
            [-1, 1, -1, -1],
            [-1, -1, -1, 0],
        ]
        tris = np.array([[0, 1, 2], [2, 1, 3]])
        assert tecio.cell_face_neighbors(tris, TRI).tolist() == [
            [-1, 1, -1],
            [0, -1, -1],
        ]

    def test_non_manifold_face_has_no_neighbor(self) -> None:
        tris = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
        assert (tecio.cell_face_neighbors(tris, TRI) == -1).all()

    def test_collapsed_face_has_no_neighbor(self) -> None:
        """Two cells meeting only at a face collapsed to an edge are not neighbors."""
        bricks = np.array([[0, 1, 2, 2, 3, 3, 3, 3], [0, 1, 4, 4, 5, 5, 5, 5]])
        result = tecio.cell_face_neighbors(bricks, BRICK)
        assert np.array_equal(result, _brute_neighbors(bricks, BRICK))
        assert (result == -1).all()

    def test_same_cell_is_not_its_own_neighbor(self) -> None:
        tris = np.array([[0, 1, 0]])
        assert (tecio.cell_face_neighbors(tris, TRI) == -1).all()

    def test_hash_collision_is_resolved(self, monkeypatch) -> None:
        """A 64-bit collision of different node sets must not create a neighbor."""
        real = elements._node_hashes

        def colliding(num_nodes):
            wide, narrow = real(num_nodes)
            return np.zeros_like(wide), narrow

        monkeypatch.setattr(elements, "_node_hashes", colliding)
        node_map = _cube_mesh(3, True)
        result = tecio.cell_face_neighbors(node_map, BRICK)
        assert np.array_equal(result, _brute_neighbors(node_map, BRICK))

    def test_second_hash_rejects_false_pair(self, monkeypatch) -> None:
        """Faces with equal 64-bit hashes but different 32-bit hashes do not pair."""
        real = elements._node_hashes

        def wide_collision(num_nodes):
            wide, narrow = real(num_nodes)
            return np.zeros_like(wide), narrow

        monkeypatch.setattr(elements, "_node_hashes", wide_collision)
        node_map = np.array([[0, 1, 2], [3, 4, 5]])
        assert (tecio.cell_face_neighbors(node_map, TRI) == -1).all()

    def test_chunking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Results do not depend on how many cells are hashed at a time."""
        monkeypatch.setattr(elements, "_CHUNK_CELLS", 7)
        node_map = _cube_mesh(4, True)
        result = tecio.cell_face_neighbors(node_map, BRICK)
        assert np.array_equal(result, _brute_neighbors(node_map, BRICK))

    def test_empty(self) -> None:
        result = tecio.cell_face_neighbors(np.empty((0, 8), dtype=np.int64), BRICK)
        assert result.shape == (0, 6)

    def test_int32_and_list_input(self) -> None:
        node_map = _cube_mesh(2)
        expected = tecio.cell_face_neighbors(node_map, BRICK)
        as_int32 = tecio.cell_face_neighbors(node_map.astype(np.int32), BRICK)
        as_list = tecio.cell_face_neighbors(node_map.tolist(), BRICK)
        assert np.array_equal(as_int32, expected)
        assert np.array_equal(as_list, expected)

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="no faces"):
            tecio.cell_face_neighbors(np.zeros((1, 2), dtype=int), ZoneType.FELINESEG)
        with pytest.raises(ValueError):
            tecio.cell_face_neighbors(np.zeros((1, 5), dtype=int), BRICK)
        with pytest.raises(ValueError, match="negative"):
            tecio.cell_face_neighbors(np.array([[0, 1, -1]]), TRI)


# --------------------------------------------------------------------------------------
# Face-based zones
# --------------------------------------------------------------------------------------

POLYHEDRON = ZoneType.FEPOLYHEDRON
POLYGON = ZoneType.FEPOLYGON


def _face_map(node_map: np.ndarray, zone_type: ZoneType) -> dict[str, np.ndarray]:
    """Return the face map of the cells of *node_map* as writer arguments."""
    owners: dict[frozenset[int], list[int]] = {}
    nodes: dict[frozenset[int], tuple[int, ...]] = {}
    for cell, row in enumerate(node_map):
        for positions in elements._FACE_NODES[zone_type]:
            face = tuple(int(row[p]) for p in positions)
            owners.setdefault(frozenset(face), []).append(cell)
            nodes.setdefault(frozenset(face), face)
    return {
        "face_nodes": np.array([n for key in owners for n in nodes[key]]),
        "face_node_counts": np.array([len(nodes[key]) for key in owners]),
        "face_left_elements": np.array([c[0] for c in owners.values()]),
        "face_right_elements": np.array([
            c[1] if len(c) > 1 else -1 for c in owners.values()
        ]),
    }


def _check_by_element(
    node_map: np.ndarray,
    face_map: dict[str, np.ndarray],
    face_offsets: np.ndarray,
    faces: np.ndarray,
    node_offsets: np.ndarray,
    nodes: np.ndarray,
) -> None:
    """Compare faces and nodes of each cell with what its own row of node_map gives."""
    left = face_map["face_left_elements"]
    right = face_map["face_right_elements"]
    assert face_offsets.dtype == faces.dtype == np.int64
    assert node_offsets.dtype == nodes.dtype == np.int64
    assert len(face_offsets) == len(node_offsets) == len(node_map) + 1
    for cell, row in enumerate(node_map):
        expected_faces = np.flatnonzero((left == cell) | (right == cell))
        assert np.array_equal(
            faces[face_offsets[cell] : face_offsets[cell + 1]], expected_faces
        )
        found = nodes[node_offsets[cell] : node_offsets[cell + 1]]
        assert found.tolist() == sorted(set(row.tolist()))


class TestFaceBasedElements:
    """Tests of :func:`faces_by_element` and :func:`nodes_by_element`."""

    @staticmethod
    def _run(
        node_map: np.ndarray, zone_type: ZoneType
    ) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, ...]]:
        fm = _face_map(node_map, zone_type)
        face_offsets, faces = elements.faces_by_element(
            fm["face_left_elements"], fm["face_right_elements"], len(node_map)
        )
        starts = np.r_[0, np.cumsum(fm["face_node_counts"])]
        node_offsets, nodes = elements.nodes_by_element(
            fm["face_nodes"], starts, face_offsets, faces, int(node_map.max()) + 1
        )
        return fm, (face_offsets, faces, node_offsets, nodes)

    @pytest.mark.parametrize("shuffle", [False, True])
    def test_tetrahedra(self, shuffle: bool) -> None:
        node_map = _kuhn_tets(2, shuffle)
        fm, result = self._run(node_map, TET)
        _check_by_element(node_map, fm, *result)
        assert (np.diff(result[0]) == 4).all()
        assert (np.diff(result[2]) == 4).all()

    def test_bricks_have_quadrilateral_faces(self) -> None:
        node_map = _cube_mesh(3, shuffle=True)
        fm, result = self._run(node_map, BRICK)
        _check_by_element(node_map, fm, *result)
        assert (np.diff(result[0]) == 6).all()
        assert (np.diff(result[2]) == 8).all()

    def test_polygons_have_edge_faces(self) -> None:
        node_map = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
        fm, result = self._run(node_map, TRI)
        _check_by_element(node_map, fm, *result)

    def test_boundary_references_and_no_element_are_skipped(self) -> None:
        left = [0, 0, -1, -3]
        right = [1, -1, 1, 0]
        offsets, faces = elements.faces_by_element(left, right, 2)
        assert offsets.tolist() == [0, 3, 5]
        assert faces.tolist() == [0, 1, 3, 0, 2]

    def test_element_without_faces(self) -> None:
        offsets, faces = elements.faces_by_element([0, 2], [-1, 2], 4)
        assert offsets.tolist() == [0, 1, 1, 3, 3]
        assert faces.tolist() == [0, 1, 1]
        node_offsets, nodes = elements.nodes_by_element(
            [0, 1, 1, 2], [0, 2, 4], offsets, faces, 3
        )
        assert node_offsets.tolist() == [0, 2, 2, 4, 4]
        assert nodes.tolist() == [0, 1, 1, 2]

    def test_empty_zone(self) -> None:
        offsets, faces = elements.faces_by_element([], [], 0)
        assert offsets.tolist() == [0]
        assert faces.size == 0
        node_offsets, nodes = elements.nodes_by_element([], [0], offsets, faces, 0)
        assert node_offsets.tolist() == [0]
        assert nodes.size == 0

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="same|one length"):
            elements.faces_by_element([0, 1], [0], 2)
        with pytest.raises(ValueError, match="not below"):
            elements.faces_by_element([0, 2], [1, -1], 2)
        offsets, faces = elements.faces_by_element([0], [-1], 1)
        with pytest.raises(ValueError, match="outside"):
            elements.nodes_by_element([0, 5], [0, 2], offsets, faces, 3)
        with pytest.raises(ValueError, match="outside"):
            elements.nodes_by_element([0, -1], [0, 2], offsets, faces, 3)

    def test_unique_pairs_without_a_single_key(self) -> None:
        """Bounds too large for one integer key per pair use the sorting path."""
        rng = np.random.default_rng(0)
        first = rng.integers(0, 50, size=500)
        second = rng.integers(0, 50, size=500)
        small = elements._unique_pairs(first, second, 50, 50)
        large = elements._unique_pairs(first, second, 2**40, 2**40)
        assert np.array_equal(small[0], large[0])
        assert np.array_equal(small[1], large[1])
        assert len(small[0]) == len({
            (a, b) for a, b in zip(first, second, strict=True)
        })


@pytest.fixture(params=["plt", "dat"])
def face_based_file(request, tmp_path: Path) -> str:
    """File with a polyhedron zone and a polygon zone in each format.

    Returns:
        Path of the written file.
    """
    tets = _kuhn_tets(2, shuffle=True)
    tris = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
    path = str(tmp_path / f"faces.{request.param}")
    with tecio.open(path, "w", variables=["x", "y", "z"]) as w:
        for zone_type, cells, face_map in [
            (POLYHEDRON, tets, _face_map(tets, TET)),
            (POLYGON, tris, _face_map(tris, TRI)),
        ]:
            num_nodes = int(cells.max()) + 1
            if zone_type is POLYGON:
                face_map.pop("face_node_counts")
            w.write_poly_zone(
                data=[np.arange(num_nodes, dtype=float)] * 3,
                zone_type=zone_type,
                **face_map,
            )
    return path


class TestPolyZoneReaderElements:
    """Tests of ``element_faces`` and ``element_nodes`` of a poly zone reader."""

    def test_polyhedron_zone(self, face_based_file: str) -> None:
        tets = _kuhn_tets(2, shuffle=True)
        with tecio.open(face_based_file) as reader:
            zone = reader.zones[0]
            assert isinstance(zone, tecio.TecplotPolyZoneReader)
            face_offsets, faces = zone.element_faces()
            node_offsets, nodes = zone.element_nodes()
        _check_by_element(
            tets, _face_map(tets, TET), face_offsets, faces, node_offsets, nodes
        )

    def test_polygon_zone(self, face_based_file: str) -> None:
        tris = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])
        with tecio.open(face_based_file) as reader:
            zone = reader.zones[1]
            assert isinstance(zone, tecio.TecplotPolyZoneReader)
            face_offsets, faces = zone.element_faces()
            node_offsets, nodes = zone.element_nodes()
        _check_by_element(
            tris, _face_map(tris, TRI), face_offsets, faces, node_offsets, nodes
        )

    def test_methods_do_not_change_the_zone(self, face_based_file: str) -> None:
        with tecio.open(face_based_file) as reader:
            zone = reader.zones[0]
            assert isinstance(zone, tecio.TecplotPolyZoneReader)
            before = zone.face_left_elements.copy()
            zone.element_nodes()
            assert np.array_equal(zone.face_left_elements, before)


# --------------------------------------------------------------------------------------
# tecelretyp CLI
# --------------------------------------------------------------------------------------

_XYZ = np.array(
    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)], dtype=np.float64
)
# Three tetrahedra written as bricks: the Tecplot layout, one that drops to the wrong
# handedness if repeats are just removed, and one that repeats apart.
_UPCAST_TETS = np.array([
    [0, 1, 2, 2, 3, 3, 3, 3],
    [1, 1, 2, 2, 1, 1, 4, 3],
    [0, 0, 0, 0, 1, 2, 3, 1],
])
_TETS = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [0, 1, 2, 3]])
_PYRAMID = [0, 0, 0, 0, 1, 2, 3, 4]

_XY = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float64)
_UPCAST_TRIS = np.array([[0, 1, 2, 2], [0, 0, 2, 3], [3, 0, 1, 3]])
_TRIS = np.array([[0, 1, 2], [0, 2, 3], [3, 0, 1]])


def _write_fe(
    writer: Any,
    zone_type: ZoneType,
    xyz: np.ndarray,
    node_map: np.ndarray,
    *,
    title: str,
    **kwargs: Any,
) -> None:
    """Write an FE zone with x, y, z, a nodal p and a cell-centered c."""
    num_nodes, num_cells = len(xyz), len(node_map)
    writer.write_fe_zone(
        zone_type=zone_type,
        data=[
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            np.arange(num_nodes) * 1.5,
            np.arange(num_cells) + 10.0,
        ],
        node_map=node_map,
        value_locations=[ValueLocation.NODAL] * 4 + [ValueLocation.CELL_CENTERED],
        title=title,
        **kwargs,
    )


def _write_dataset(path: Path) -> None:
    """Write upcast tetrahedra, upcast triangles, a hybrid zone and an ordered zone."""
    xy0 = np.column_stack([_XY, np.zeros(4)])
    with tecio.open(str(path), "w", variables=["x", "y", "z", "p", "c"]) as w:
        _write_fe(
            w,
            BRICK,
            _XYZ,
            _UPCAST_TETS,
            title="tets",
            strand_id=3,
            solution_time=2.5,
            aux={"Common.Note": "kept"},
        )
        _write_fe(w, QUAD, xy0, _UPCAST_TRIS, title="tris")
        _write_fe(
            w,
            BRICK,
            _XYZ,
            np.array([_UPCAST_TETS[0], _PYRAMID]),
            title="hybrid",
        )
        w.write_ordered_zone(
            data=[np.zeros((2, 2, 1))] * 4,
            passive_vars={"c"},
            title="ordered",
        )


def _write_shared(path: Path) -> None:
    """Write three upcast zones sharing one node map (zone 0 owns it)."""
    with tecio.open(str(path), "w", variables=["x", "y", "z", "p", "c"]) as w:
        _write_fe(w, BRICK, _XYZ, _UPCAST_TETS, title="owner", strand_id=1)
        for t in (1, 2):
            w.write_fe_zone(
                zone_type=BRICK,
                data=[np.arange(len(_XYZ)) * 2.0 * t, np.arange(3) + 20.0 * t],
                var_sharing={"x": 0, "y": 0, "z": 0},
                value_locations=[ValueLocation.NODAL, ValueLocation.CELL_CENTERED],
                con_sharing=0,
                title=f"sharer{t}",
                strand_id=1,
                solution_time=float(t),
            )


@pytest.fixture(params=_FORMATS)
def dataset(request, tmp_path: Path) -> Path:
    """Dataset file in each format.

    Returns:
        Path of the written file.
    """
    path = tmp_path / f"data.{request.param}"
    _write_dataset(path)
    return path


@pytest.fixture(params=_FORMATS)
def shared(request, tmp_path: Path) -> Path:
    """Shared-connectivity dataset in each format.

    Returns:
        Path of the written file.
    """
    path = tmp_path / f"shared.{request.param}"
    _write_shared(path)
    return path


class TestTecelretyp:
    """tecelretyp command-line tool."""

    def test_retypes_tetrahedra_and_triangles(
        self, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Brick zones of tetrahedra and quad zones of triangles are rewritten."""
        out = tmp_path / f"out{dataset.suffix}"
        assert tecelretyp([str(dataset), "-o", str(out)]) == 0
        with tecio.open(str(out), "r") as r:
            assert [z.zone_type for z in r.zones][:3] == [TET, TRI, BRICK]
            tets = r.zones[0]
            np.testing.assert_array_equal(tets.node_map, _TETS)
            assert _tet_volumes(_XYZ, tets.node_map).min() > 0
            tris = r.zones[1]
            np.testing.assert_array_equal(tris.node_map, _TRIS)
            assert _triangle_areas(_XY, tris.node_map).min() > 0
        assert "2 of 4 zones can be retyped" in capsys.readouterr().out

    def test_other_zones_and_metadata_are_unchanged(
        self, dataset: Path, tmp_path: Path
    ) -> None:
        """Hybrid and ordered zones, titles, times, aux data and variables carry over."""
        out = tmp_path / f"out{dataset.suffix}"
        assert tecelretyp([str(dataset), "-o", str(out)]) == 0
        with tecio.open(str(dataset), "r") as a, tecio.open(str(out), "r") as b:
            assert b.variables == a.variables
            assert b.num_zones == a.num_zones
            for za, zb in zip(a.zones, b.zones, strict=True):
                assert zb.title == za.title
                assert zb.strand_id == za.strand_id
                assert zb.solution_time == za.solution_time
                assert dict(zb.auxdata.items()) == dict(za.auxdata.items())
                for va, vb in zip(za.variables, zb.variables, strict=True):
                    assert vb.value_location == va.value_location
                    if va.values is None:
                        assert vb.values is None  # passive
                    else:
                        np.testing.assert_allclose(vb.values, va.values)
            assert b.zones[0].strand_id == 3
            assert b.zones[0].solution_time == 2.5
            np.testing.assert_array_equal(b.zones[2].node_map, a.zones[2].node_map)
            assert b.zones[2].zone_type == BRICK
            assert b.zones[3].zone_type == ZoneType.ORDERED

    def test_default_output_name_and_force(
        self, dataset: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Output defaults to <stem>_retyped and is not overwritten without --force."""
        expected = dataset.with_stem("data_retyped")
        assert tecelretyp([str(dataset)]) == 0
        assert expected.exists()
        assert tecelretyp([str(dataset)]) == 1
        assert "already exists" in capsys.readouterr().err
        assert tecelretyp([str(dataset), "--force"]) == 0

    def test_dry_run_reports_and_writes_nothing(
        self, dataset: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """--dry-run lists each zone and what would happen."""
        assert tecelretyp([str(dataset), "--dry-run"]) == 0
        assert not dataset.with_stem("data_retyped").exists()
        out = capsys.readouterr().out
        assert "-> FETETRAHEDRON" in out
        assert "-> FETRIANGLE" in out
        assert "unchanged (cells: 1 tetrahedron, 1 pyramid)" in out
        assert "removing 15 node map entries" in out

    def test_check_exit_codes(
        self, dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """--check exits 1 if a zone could be retyped, 0 if none could, 2 on error."""
        assert tecelretyp(["--check", str(dataset)]) == 1
        assert not dataset.with_stem("data_retyped").exists()
        compact = tmp_path / f"compact{dataset.suffix}"
        assert tecelretyp([str(dataset), "-o", str(compact)]) == 0
        # only the hybrid zone is left, and it cannot collapse
        assert tecelretyp(["--check", "-z", "2", str(compact)]) == 0
        assert tecelretyp(["--check", str(tmp_path / "missing.plt")]) == 2
        assert tecelretyp(["--check", "-z", "9", str(dataset)]) == 2
        assert "out of range" in capsys.readouterr().err

    def test_nothing_to_retype_writes_no_file(
        self, dataset: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A file with no retypable zone is not copied."""
        assert tecelretyp([str(dataset), "-z", "2,3"]) == 0
        assert not dataset.with_stem("data_retyped").exists()
        assert "Nothing to retype" in capsys.readouterr().out

    def test_zone_selection(self, dataset: Path, tmp_path: Path) -> None:
        """Only the listed zones are retyped."""
        out = tmp_path / f"out{dataset.suffix}"
        assert tecelretyp([str(dataset), "-z", "1", "-o", str(out)]) == 0
        with tecio.open(str(out), "r") as r:
            assert [z.zone_type for z in r.zones][:2] == [BRICK, TRI]

    def test_errors(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Missing input and bad zone indices return 1."""
        assert tecelretyp([str(tmp_path / "missing.plt")]) == 1
        assert "not found" in capsys.readouterr().err
        path = tmp_path / "d.plt"
        _write_dataset(path)
        assert tecelretyp([str(path), "-z", "7"]) == 1
        assert "out of range" in capsys.readouterr().err

    def test_converts_format_in_one_step(self, tmp_path: Path) -> None:
        """The output extension picks the format."""
        src = tmp_path / "in.plt"
        _write_dataset(src)
        out = tmp_path / "out.szplt"
        assert tecelretyp([str(src), "-o", str(out)]) == 0
        with tecio.open(str(out), "r") as r:
            assert r.zones[0].zone_type == TET

    def test_shared_connectivity_is_retyped_together(
        self, shared: Path, tmp_path: Path
    ) -> None:
        """A zone group is retyped as one and keeps sharing from the owner."""
        out = tmp_path / f"out{shared.suffix}"
        assert tecelretyp([str(shared), "-o", str(out)]) == 0
        with tecio.open(str(out), "r") as r:
            assert [z.zone_type for z in r.zones] == [TET, TET, TET]
            assert [z.shared_connectivity for z in r.zones] == [None, 0, 0]
            np.testing.assert_array_equal(r.zones[2].node_map, _TETS)
            np.testing.assert_allclose(
                r.zones[2].variables[3].values, [0, 4, 8, 12, 16]
            )
            assert r.zones[1].variables[0].shared_zone == 0

    def test_dry_run_lists_zones_that_share_connectivity(
        self, shared: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Each zone of a group is listed, with the owner named for the others."""
        assert tecelretyp(["--dry-run", str(shared)]) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[1].endswith("-> FETETRAHEDRON")
        assert lines[2].endswith("-> FETETRAHEDRON (shares connectivity with zone 0)")
        assert lines[3].endswith("-> FETETRAHEDRON (shares connectivity with zone 0)")
        assert "removing 12 node map entries" in lines[4]

    def test_shared_group_needs_every_zone_selected(
        self, shared: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A group with an unselected member is left alone."""
        assert tecelretyp(["--dry-run", "-z", "0,1", str(shared)]) == 0
        out = capsys.readouterr().out
        assert "shares connectivity with unselected zone 2" in out
        assert "0 of 3 zones can be retyped" in out
        assert tecelretyp(["--dry-run", "-z", "1,2", str(shared)]) == 0
        assert "unchanged (not selected)" in capsys.readouterr().out

    @pytest.mark.parametrize("fmt", ["plt", "dat"])
    def test_face_neighbors_block_retyping_unless_dropped(
        self, fmt: str, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Face numbers depend on the cell type, so the data is never kept silently."""
        src = tmp_path / f"fn.{fmt}"
        with tecio.open(str(src), "w", variables=["x", "y", "z", "p", "c"]) as w:
            _write_fe(
                w,
                BRICK,
                _XYZ,
                _UPCAST_TETS,
                title="fn",
                face_neighbors=np.array([0, 1, 1]),
                face_neighbor_mode=FaceNeighborMode.LOCAL_ONE_TO_ONE,
            )
        assert tecelretyp(["--check", str(src)]) == 0
        assert "has face-neighbor data" in capsys.readouterr().out
        out = tmp_path / f"out.{fmt}"
        assert tecelretyp([str(src), "--drop-face-neighbors", "-o", str(out)]) == 0
        with tecio.open(str(out), "r") as r:
            assert r.zones[0].zone_type == TET
            assert r.zones[0].face_neighbor_mode is None

    def test_face_neighbors_of_other_zones_are_kept(self, tmp_path: Path) -> None:
        """Zones that are not retyped keep their face-neighbor data."""
        src = tmp_path / "fn.plt"
        with tecio.open(str(src), "w", variables=["x", "y", "z", "p", "c"]) as w:
            _write_fe(w, BRICK, _XYZ, _UPCAST_TETS, title="plain")
            _write_fe(
                w,
                BRICK,
                _XYZ,
                np.array([_UPCAST_TETS[0], _PYRAMID]),
                title="hybrid",
                face_neighbors=np.array([0, 1, 1]),
                face_neighbor_mode=FaceNeighborMode.LOCAL_ONE_TO_ONE,
            )
        out = tmp_path / "out.plt"
        assert tecelretyp([str(src), "--drop-face-neighbors", "-o", str(out)]) == 0
        with tecio.open(str(out), "r") as r:
            assert r.zones[0].zone_type == TET
            assert r.zones[1].zone_type == BRICK
            assert r.zones[1].face_neighbor_mode == FaceNeighborMode.LOCAL_ONE_TO_ONE

    def test_szplt_output_does_not_block_on_face_neighbors(
        self, tmp_path: Path
    ) -> None:
        """SZL files store no face neighbors, so nothing is lost by retyping."""
        src = tmp_path / "fn.plt"
        with tecio.open(str(src), "w", variables=["x", "y", "z", "p", "c"]) as w:
            _write_fe(
                w,
                BRICK,
                _XYZ,
                _UPCAST_TETS,
                title="fn",
                face_neighbors=np.array([0, 1, 1]),
                face_neighbor_mode=FaceNeighborMode.LOCAL_ONE_TO_ONE,
            )
        out = tmp_path / "out.szplt"
        assert tecelretyp([str(src), "-o", str(out)]) == 0
        with tecio.open(str(out), "r") as r:
            assert r.zones[0].zone_type == TET


def _write_with_aux_and_polygon(path: Path) -> None:
    """Write upcast tetrahedra and a polygon zone, with dataset and variable aux data."""
    xy = np.array([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)], dtype=np.float64)
    with tecio.open(str(path), "w", variables=["x", "y", "z", "p", "c"]) as w:
        w.add_auxdataset_dict({"Common.Note": "dataset"})
        w.add_auxvar_dict({3: {"Units": "Pa"}})
        _write_fe(w, BRICK, _XYZ, _UPCAST_TETS, title="tets")
        w.write_poly_zone(
            zone_type=ZoneType.FEPOLYGON,
            data=[
                xy[:, 0],
                xy[:, 1],
                np.zeros(6),
                np.arange(6.0),
                np.array([1.0, 2.0]),
            ],
            value_locations=[ValueLocation.NODAL] * 4 + [ValueLocation.CELL_CENTERED],
            face_nodes=[0, 1, 1, 4, 4, 3, 3, 0, 1, 2, 2, 5, 5, 4],
            face_left_elements=[0, 0, 0, 0, 1, 1, 1],
            face_right_elements=[-1, 1, -1, -1, -1, -1, -1],
            title="polygons",
        )


class TestTecelretypOtherContent:
    """Zones and metadata that tecelretyp passes through."""

    def test_aux_data_and_polygon_zones_are_copied(self, tmp_path: Path) -> None:
        """Dataset aux, variable aux and a polygon zone survive next to a retyped zone."""
        src = tmp_path / "in.plt"
        _write_with_aux_and_polygon(src)
        out = tmp_path / "out.plt"
        assert tecelretyp([str(src), "-o", str(out)]) == 0
        with tecio.open(str(src), "r") as a, tecio.open(str(out), "r") as b:
            assert dict(b.auxdata.items()) == {"Common.Note": "dataset"}
            assert dict(b.get_var_auxdata(3).items()) == {"Units": "Pa"}
            assert [z.zone_type for z in b.zones] == [TET, ZoneType.FEPOLYGON]
            for name in ("face_nodes", "face_left_elements", "face_right_elements"):
                np.testing.assert_array_equal(
                    getattr(b.zones[1], name), getattr(a.zones[1], name)
                )

    def test_unwritable_zone_removes_the_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A polygon zone cannot go to SZL: the error is reported and no file is left."""
        src = tmp_path / "in.plt"
        _write_with_aux_and_polygon(src)
        out = tmp_path / "out.szplt"
        assert tecelretyp([str(src), "-o", str(out)]) == 1
        assert not out.exists()
        assert capsys.readouterr().err.startswith("Error:")
