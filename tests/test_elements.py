#!/usr/bin/env python3
"""pytest test suite for the node-map utilities and the ``tecelretyp`` CLI tool.

The node-map tests need no Tecplot library. They enumerate every way a brick can
collapse onto a valid solid (4140 partitions of its 8 node positions), and check the
tetrahedra against the exact volume of the degenerate trilinear brick. The CLI tests
write small synthetic files in all three formats.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from create_test_data import (
    create_fe_padded_brick,
    create_fe_tet,
    create_femixed_block,
    create_femixed_volume,
)
from helpers import reordered_node_maps, scramble_zone

import tecio
from tecio import FaceNeighborMode, FeCellShape, ValueLocation, ZoneType
from tecio import _elements as elements
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
        """A brick collapses to 102 tetrahedra, 54 pyramids and 12 prisms."""
        assert len(_brick_patterns(FeCellShape.TETRAHEDRON)) == 102
        assert len(_brick_patterns(FeCellShape.PYRAMID)) == 54
        assert len(_brick_patterns(FeCellShape.PRISM)) == 12
        assert len(_brick_patterns(FeCellShape.HEXAHEDRON)) == 1

    def test_folded_sliver_is_not_a_pyramid_or_prism(self) -> None:
        """Right face counts are not enough, a node with two edges pinches the cell.

        The cell below has the faces of a prism (two triangles and three quads) but the
        node ``1`` touches only two faces that lie on top of each other.
        """
        folded = np.array([[0, 0, 0, 1, 2, 3, 4, 5]])
        assert tecio.classify_cells(folded, BRICK)[0] == elements.UNKNOWN_SHAPE
        with pytest.raises(ValueError, match="not a recognized shape"):
            tecio.split_cells_by_shape(folded, BRICK)

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
        assert tecio.split_cells_by_shape.__module__ == "tecio"
        assert "classify_cells" in tecio.__all__
        assert "split_cells_by_shape" in tecio.__all__


def _solid_volume(shape: FeCellShape, xyz: np.ndarray, nodes: np.ndarray) -> float:
    """Signed volume of a tetrahedron, pyramid or prism in FEMIXED node order.

    The solids are cut into tetrahedra along their node order, so the result is positive
    for the right-handed layouts of the data format guide.

    Returns:
        The volume.
    """
    cuts = {
        FeCellShape.TETRAHEDRON: [(0, 1, 2, 3)],
        FeCellShape.PYRAMID: [(0, 1, 2, 4), (0, 2, 3, 4)],
        FeCellShape.PRISM: [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)],
    }[shape]
    return sum(_tet_volumes(xyz, nodes[None, list(cut)])[0] for cut in cuts)


def _cell_volume(shape: FeCellShape, xyz: np.ndarray, nodes: np.ndarray) -> float:
    """Signed volume of a cell of any solid shape.

    Returns:
        The volume.
    """
    if shape == FeCellShape.HEXAHEDRON:
        return _hex_volume(xyz[nodes])
    return _solid_volume(shape, xyz, nodes)


class TestSplitCells:
    """split_cells_by_shape separates the cells of a zone, one section per shape."""

    def test_split_brick_zone_into_sections(self, monkeypatch) -> None:
        """A FEBRICK zone of collapsed cells gives one section for each solid shape.

        Demonstrates:
        - Sections come in the order tetrahedron, hexahedron, pyramid, prism, each with
          the ascending indices of its cells in the input
        - Repeated nodes are dropped, a hexahedron is unchanged and the input dtype is
          kept
        - A pyramid lists its base then its apex and a prism one triangle then the other
        - Cells are processed in chunks, which does not change the sections
        """
        _, _, _, cells = create_fe_padded_brick()
        cells = cells.astype(np.int32)

        result = tecio.split_cells_by_shape(cells, BRICK)

        # Sections, cell indices, shapes and dtypes
        assert list(result) == [
            FeCellShape.TETRAHEDRON,
            FeCellShape.HEXAHEDRON,
            FeCellShape.PYRAMID,
            FeCellShape.PRISM,
        ]
        assert [r[0].tolist() for r in result.values()] == [[0, 3], [2], [1, 5], [4]]
        assert [r[1].shape for r in result.values()] == [(2, 4), (1, 8), (2, 5), (1, 6)]
        assert all(r[1].dtype == np.int32 for r in result.values())
        np.testing.assert_array_equal(result[FeCellShape.HEXAHEDRON][1], cells[[2]])
        # A flat node map gives the same sections
        flat = tecio.split_cells_by_shape(cells.ravel(), BRICK)
        for shape, (idx, nodes) in result.items():
            np.testing.assert_array_equal(flat[shape][0], idx)
            np.testing.assert_array_equal(flat[shape][1], nodes)

        # Node order of pyramids and prisms
        pyramids = result[FeCellShape.PYRAMID][1]
        prisms = result[FeCellShape.PRISM][1]
        assert pyramids[:, 4].tolist() == [8, 9]  # the apex is last
        assert sorted(pyramids[0, :4]) == [4, 5, 6, 7]
        assert sorted(prisms[0]) == [1, 2, 5, 6, 10, 11]
        # node i + 3 is joined to node i by an edge of the prism
        edges = {
            tuple(sorted(e)) for e in zip(prisms[0, :3], prisms[0, 3:], strict=True)
        }
        assert edges == {(1, 5), (2, 6), (10, 11)}

        # Chunks of 7 cells give the same sections as one pass
        rng = np.random.default_rng(5)
        many = cells[rng.integers(len(cells), size=200)] + 100
        whole = tecio.split_cells_by_shape(many, BRICK)
        monkeypatch.setattr(elements, "_CHUNK_CELLS", 7)
        chunked = tecio.split_cells_by_shape(many, BRICK)
        assert chunked.keys() == whole.keys()
        for shape, (idx, nodes) in whole.items():
            np.testing.assert_array_equal(chunked[shape][0], idx)
            np.testing.assert_array_equal(chunked[shape][1], nodes)

    @pytest.mark.parametrize("mirror", [False, True])
    def test_every_collapse_keeps_the_handedness(self, mirror: bool) -> None:
        """Pyramids, prisms and tetrahedra have the volume of the brick they came from.

        The cells have flat faces, so their volume is the exact volume of the degenerate
        brick. Mirroring the points makes every brick left-handed.
        """
        rng = np.random.default_rng(11)
        for shape in (
            FeCellShape.TETRAHEDRON,
            FeCellShape.PYRAMID,
            FeCellShape.PRISM,
        ):
            patterns = np.array(_brick_patterns(shape))
            for labels in patterns:
                nodes = np.unique(labels)
                # Put every node where the brick corner of its first appearance sits,
                # which gives flat faces, then nudge it so that no volume is zero.
                xyz = np.array([_CORNERS[list(labels).index(n)] for n in nodes])
                xyz = xyz + rng.normal(scale=0.02, size=xyz.shape)
                if mirror:
                    xyz[:, 0] *= -1
                brick = _hex_volume(xyz[labels])
                ((found, (_, cell)),) = tecio.split_cells_by_shape(
                    labels[None, :], BRICK
                ).items()
                assert found == shape
                assert len(set(cell[0])) == len(cell[0])
                # Compare signs only: the brick is not flat-faced for every pattern.
                assert np.sign(_solid_volume(shape, xyz, cell[0])) == np.sign(brick)

    def test_split_single_shape_and_empty_zones(self) -> None:
        """A zone of one cell shape comes back as one section, an empty zone as none.

        Demonstrates:
        - The node map is unchanged, and a flat node map gives the same section
        - A zone without cells has no sections
        """
        _, _, _, tets = create_fe_tet()

        ((shape, (cells, nodes)),) = tecio.split_cells_by_shape(tets, TET).items()

        assert shape == FeCellShape.TETRAHEDRON
        assert cells.tolist() == [0, 1]
        np.testing.assert_array_equal(nodes, tets)
        assert tecio.split_cells_by_shape(tets.ravel(), TET).keys() == {shape}
        assert tecio.split_cells_by_shape(np.zeros((0, 8), dtype=int), BRICK) == {}

    def test_split_rejects_unusable_input(self) -> None:
        """Unrecognized cells and unsupported zone types are rejected."""
        # A brick that is not any solid shape, here a sliver that has repeated edges
        with pytest.raises(ValueError, match="cell 1 is not a recognized shape"):
            tecio.split_cells_by_shape(
                [[0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 0, 0, 0, 0, 1, 2]], BRICK
            )
        # Only classic finite-element zone types have cells to split
        with pytest.raises(ValueError, match="classic finite-element"):
            tecio.split_cells_by_shape([[0, 1, 2]], ZoneType.FEPOLYGON)
        # The node count of a cell must match the zone type
        with pytest.raises(ValueError, match="nodes per cell"):
            tecio.split_cells_by_shape([[0, 1, 2]], BRICK)


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
        assert np.array_equal(wide, elements._splitmix64(index))
        assert np.array_equal(
            narrow, elements._lowbias32((index + 1).astype(np.uint32))
        )

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
# Matching zones stored in another order
# --------------------------------------------------------------------------------------


class TestMatchNodesAndCells:
    """Tests of :func:`tecio.match_nodes` and :func:`tecio.match_cells`."""

    def test_match_recovers_a_scrambled_zone(self) -> None:
        """Nodes, node maps and data compare exactly once the zone is matched.

        Demonstrates:
        - ``match_nodes`` gives the order that puts nodal arrays in the reference order
        - ``np.argsort`` of that order gives the reference index of each node
        - ``match_cells`` gives the order for cell-centered arrays, counting cells
          through the sections
        - Node maps of another integer type match, and ``node_order`` may be left out
          when both zones number their nodes alike
        """
        x, y, z, _, maps = create_femixed_block(4)
        coords = [x, y, z]
        cell_values = np.arange(sum(len(m) for m in maps)) * 0.5
        new_coords, new_maps, perm, cell_perm = scramble_zone(coords, maps, seed=3)
        new_maps = [m.astype(np.int32) for m in new_maps]

        node_order = tecio.match_nodes(coords, new_coords)
        assert node_order is not None
        for got, want in zip(new_coords, coords, strict=True):
            np.testing.assert_array_equal(got[node_order], want)
        np.testing.assert_array_equal(np.argsort(node_order), perm)

        cell_order = tecio.match_cells(maps, new_maps, node_order)
        assert cell_order is not None
        np.testing.assert_array_equal(cell_values[cell_perm][cell_order], cell_values)
        got_maps = reordered_node_maps(new_maps, node_order, cell_order)
        for got, want in zip(got_maps, maps, strict=True):
            np.testing.assert_array_equal(got, want)

        reversed_maps = [m[::-1] for m in maps]
        cell_order = tecio.match_cells(maps, reversed_maps)
        starts = np.cumsum([0, *(len(m) for m in maps[:-1])])
        expected = [
            lo + np.arange(len(m))[::-1] for lo, m in zip(starts, maps, strict=True)
        ]
        np.testing.assert_array_equal(cell_order, np.concatenate(expected))

    def test_match_a_zone_written_to_plt_and_szplt(self, tmp_path: Path) -> None:
        """The same classic zone in PLT and SZPLT is matched, whichever order each uses.

        Demonstrates:
        - A classic zone is passed to ``match_cells`` as a list of one node map
        - Nodal and cell-centered variables of one file are put in the order of the
          other with ``values[node_order]`` and ``values[cell_order]``
        - The zone is large enough for SZPLT to store it in another order than PLT
        """
        n = 9
        node_map = _cube_mesh(n)
        lattice = np.arange(n + 1, dtype=np.float64)
        x, y, z = (
            a.ravel() for a in np.meshgrid(lattice, lattice, lattice, indexing="ij")
        )
        pressure = x + 10.0 * y + 100.0 * z
        density = np.arange(len(node_map), dtype=np.float64)

        zones = {}
        for suffix in ("plt", "szplt"):
            path = tmp_path / f"cube.{suffix}"
            with tecio.open(str(path), "w") as w:
                w.write_fe_zone(
                    zone_type=BRICK,
                    data=[x, y, z, pressure, density],
                    node_map=node_map,
                    variables=["x", "y", "z", "p", "d"],
                    value_locations=[ValueLocation.NODAL] * 4
                    + [ValueLocation.CELL_CENTERED],
                )
            with tecio.open(str(path), "r") as r:
                zone = r.zones[0]
                zones[suffix] = (
                    zone.get_array(["x", "y", "z"]),
                    zone.node_map,
                    zone.get_array("p"),
                    zone.get_array("d"),
                )
        (ref_coords, ref_map, ref_p, ref_d) = zones["plt"]
        (coords, node_map_szplt, p, d) = zones["szplt"]

        node_order = tecio.match_nodes(ref_coords, coords)
        assert node_order is not None
        cell_order = tecio.match_cells([ref_map], [node_map_szplt], node_order)
        assert cell_order is not None

        np.testing.assert_array_equal(p[node_order], ref_p)
        np.testing.assert_array_equal(d[cell_order], ref_d)
        (got,) = reordered_node_maps([node_map_szplt], node_order, cell_order)
        np.testing.assert_array_equal(got, ref_map)

    def test_match_ignores_the_order_of_nodes_within_cells(self) -> None:
        """Cells match by their set of nodes, so a rotated cell is still found."""
        _, _, _, cells = create_fe_padded_brick()
        maps = [cells]
        rolled = [np.roll(m, 1, axis=1) for m in maps]

        cell_order = tecio.match_cells(maps, rolled)

        assert not np.array_equal(rolled[0], maps[0])
        np.testing.assert_array_equal(cell_order, np.arange(len(maps[0])))

    def test_match_reports_zones_that_differ(self) -> None:
        """``None`` means the zones are not the same grid or cannot be told apart.

        Demonstrates:
        - A coordinate that changed, other node counts and other coordinate counts
        - Coordinates that repeat, which leave the match ambiguous
        - Other sections, a cell with another node, and identical cells in the
          reference
        """
        x, y, z, _, maps = create_femixed_block(3)
        coords = [x, y, z]
        new_coords, new_maps, _, _ = scramble_zone(coords, maps, seed=1)

        moved = [c.copy() for c in new_coords]
        moved[0][0] += 0.25
        assert tecio.match_nodes(coords, moved) is None
        assert tecio.match_nodes(coords, [c[:-1] for c in new_coords]) is None
        assert tecio.match_nodes([x, y], new_coords) is None
        repeated = [c.copy() for c in coords]
        for c in repeated:
            c[1] = c[0]
        assert tecio.match_nodes(repeated, repeated) is None

        node_order = tecio.match_nodes(coords, new_coords)
        assert tecio.match_cells(maps, new_maps, node_order) is not None
        assert tecio.match_cells(maps, new_maps[:-1], node_order) is None
        assert tecio.match_cells(maps, new_maps[::-1], node_order) is None
        wrong = [m.copy() for m in new_maps]
        wrong[1][0, 0] = wrong[1][1, 1]
        assert tecio.match_cells(maps, wrong, node_order) is None
        doubled = [np.concatenate([m, m[:1]]) for m in maps]
        assert tecio.match_cells(doubled, doubled) is None

    def test_match_rejects_unusable_input(self) -> None:
        """Arrays that are not coordinates or node maps are errors, not mismatches."""
        line = np.arange(3.0)
        with pytest.raises(ValueError, match="1-D"):
            tecio.match_nodes([], [])
        with pytest.raises(ValueError, match="1-D"):
            tecio.match_nodes([np.zeros((3, 3))], [line])
        with pytest.raises(ValueError, match="same length"):
            tecio.match_nodes([line, line[:2]], [line, line])

        cell = np.array([[0, 1, 2]])
        with pytest.raises(ValueError, match="integer"):
            tecio.match_cells([cell.astype(float)], [cell])
        with pytest.raises(ValueError, match="two-dimensional"):
            tecio.match_cells([cell[0]], [cell])
        with pytest.raises(ValueError, match="negative"):
            tecio.match_cells([cell], [cell - 1])
        with pytest.raises(ValueError, match="beyond"):
            tecio.match_cells([cell], [cell + 1], node_order=[0, 1, 2])

    def test_match_cells_in_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cells are hashed in chunks, which does not change the match."""
        x, y, z, _, maps = create_femixed_block(4)
        new_coords, new_maps, _, _ = scramble_zone([x, y, z], maps, seed=5)
        node_order = tecio.match_nodes([x, y, z], new_coords)
        whole = tecio.match_cells(maps, new_maps, node_order)
        assert whole is not None

        monkeypatch.setattr(elements, "_CHUNK_CELLS", 7)
        chunked = tecio.match_cells(maps, new_maps, node_order)

        np.testing.assert_array_equal(chunked, whole)

    def test_match_cells_checks_a_second_hash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two cells whose 64-bit hashes collide are not taken for the same cell."""
        wide = np.array([1, 2, 3, 0], dtype=np.uint64)  # nodes 0, 1 and 2, 3 collide
        narrow = np.array([10, 20, 30, 40], dtype=np.uint32)
        monkeypatch.setattr(elements, "_node_hashes", lambda n: (wide, narrow))
        first, second = np.array([[0, 1]]), np.array([[2, 3]])

        assert tecio.match_cells([first], [second]) is None
        np.testing.assert_array_equal(tecio.match_cells([first], [first]), [0])


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
        assert (
            "unchanged (cells: 1 tetrahedron, 1 pyramid; "
            "could be FEMIXED, see --femixed)" in out
        )
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

    @pytest.mark.parametrize("mirror", [False, True])
    def test_femixed_keeps_geometry_and_cell_values(
        self, tmp_path: Path, mirror: bool
    ) -> None:
        """--femixed writes a zone of several shapes as FEMIXED, cell for cell.

        Demonstrates:
        - A FEBRICK zone of tetrahedra, pyramids, a prism and a hexahedron becomes one
          section for each shape, in the order tetrahedron, hexahedron, pyramid, prism
        - Every cell keeps its nodes, its volume, its handedness and its cell-centered
          value, whether the grid is right-handed or mirrored to left-handed
        """
        x, y, z, cells = create_fe_padded_brick()
        if mirror:
            x = -x
        xyz = np.column_stack([x, y, z])
        src = tmp_path / "solids.plt"
        with tecio.open(str(src), "w") as w:
            w.write_fe_zone(
                zone_type=BRICK,
                data=[x, y, z, np.arange(len(x)) * 1.5, np.arange(len(cells)) + 10.0],
                node_map=cells,
                variables=["x", "y", "z", "p", "c"],
                value_locations=[ValueLocation.NODAL] * 4
                + [ValueLocation.CELL_CENTERED],
                title="solids",
            )
        out = tmp_path / "solids.szplt"

        assert tecelretyp([str(src), "--femixed", "-o", str(out)]) == 0

        with tecio.open(str(out), "r") as r:
            zone = r.zones[0]
            assert isinstance(zone, tecio.TecplotFEMixedZoneReader)
            assert zone.title == "solids"
            assert zone.section_shapes == (
                FeCellShape.TETRAHEDRON,
                FeCellShape.HEXAHEDRON,
                FeCellShape.PYRAMID,
                FeCellShape.PRISM,
            )
            assert zone.section_num_elements == (2, 1, 2, 1)
            # Nodes may be stored in another order, so match them by coordinates
            node_order = tecio.match_nodes([x, y, z], zone.get_array(["x", "y", "z"]))
            assert node_order is not None
            renumber = np.argsort(node_order)
            np.testing.assert_allclose(
                zone.variables["p"].values[node_order], np.arange(len(x)) * 1.5
            )
            # Sections are in shape order, so the values show where each cell came from
            values = zone.variables["c"].values
            assert sorted(values) == [10, 11, 12, 13, 14, 15]
            assert not np.array_equal(values, np.arange(6) + 10.0)
            k = 0
            for shape, nodes in zip(zone.section_shapes, zone.node_maps, strict=True):
                for cell in renumber[nodes]:
                    source = cells[int(values[k]) - 10]
                    assert set(cell) == set(source)
                    expected = _hex_volume(xyz[source])
                    assert _cell_volume(shape, xyz, cell) == pytest.approx(expected)
                    assert (expected < 0) == mirror
                    k += 1

    @pytest.mark.parametrize("fmt", _FORMATS)
    def test_femixed_needs_szplt_output(
        self, fmt: str, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """FEMIXED zones need SZPLT, so the default output name and format change.

        Demonstrates:
        - Without ``-o`` the output is ``<stem>_retyped.szplt``, whatever the input
        - ``-o`` with a PLT or DAT name is an error and writes no file
        """
        x, y, z, cells = create_fe_padded_brick()
        src = tmp_path / f"solids.{fmt}"
        with tecio.open(str(src), "w") as w:
            w.write_fe_zone(
                zone_type=BRICK,
                data=[x, y, z],
                node_map=cells,
                variables=["x", "y", "z"],
            )

        assert tecelretyp([str(src), "--femixed"]) == 0
        assert tmp_path.joinpath("solids_retyped.szplt").exists()

        for ext in ("plt", "dat"):
            refused = tmp_path / f"refused.{ext}"
            assert tecelretyp([str(src), "--femixed", "-o", str(refused)]) == 1
            assert "--femixed needs .szplt output" in capsys.readouterr().err
            assert not refused.exists()

    @pytest.mark.parametrize("fmt", _FORMATS)
    def test_femixed_is_reported_by_dry_run_and_check(
        self, fmt: str, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """The dry run points out --femixed, and --check only counts it when given.

        Demonstrates:
        - Without ``--femixed`` the zone is listed as unchanged with its cell shapes
          and a hint, and ``--check`` exits 0
        - With ``--femixed`` the zone is listed as FEMIXED, the summary counts it and
          ``--check`` exits 1
        - ``--zones`` limits ``--femixed`` as it limits retyping
        """
        x, y, z, cells = create_fe_padded_brick()
        src = tmp_path / f"solids.{fmt}"
        with tecio.open(str(src), "w") as w:
            w.write_fe_zone(
                zone_type=BRICK,
                data=[x, y, z],
                node_map=cells,
                variables=["x", "y", "z"],
            )

        assert tecelretyp(["--check", str(src)]) == 0
        out = capsys.readouterr().out
        assert (
            "unchanged (cells: 2 tetrahedron, 1 hexahedron, 2 pyramid, 1 prism" in out
        )
        assert "could be FEMIXED, see --femixed" in out

        assert tecelretyp(["--check", "--femixed", str(src)]) == 1
        out = capsys.readouterr().out
        assert "-> FEMIXED: 2 tetrahedron, 1 hexahedron, 2 pyramid, 1 prism" in out
        assert "1 of 1 zones can be retyped (1 as FEMIXED), removing 16 node map" in out

        assert tecelretyp(["--check", "--femixed", "-z", "0", str(src)]) == 1
        assert tecelretyp(["--check", "--femixed", "-z", "5", str(src)]) == 2
        assert "out of range" in capsys.readouterr().err

    def test_femixed_is_used_only_where_it_helps(self, tmp_path: Path) -> None:
        """Each zone of a file gets the smallest zone type that holds its cells.

        Demonstrates:
        - Cells of several shapes become a FEMIXED zone
        - A zone of one shape that has a classic zone type is retyped instead
        - A zone of one shape that has none, like pyramids, is a one-section zone
        - A zone of hexahedra already uses every node, so it is not touched
        """
        x, y, z, cells = create_fe_padded_brick()
        # Zone title, zone type, node map and the nodes that its cells use
        zones = [
            ("solids", BRICK, cells, np.column_stack([x, y, z])),
            ("tets", BRICK, cells[[0, 3]], np.column_stack([x, y, z])[:11]),
            ("pyramids", BRICK, cells[[1, 5]], np.column_stack([x, y, z])[:10]),
            ("hexahedron", BRICK, cells[[2]], np.column_stack([x, y, z])[:8]),
        ]
        src = tmp_path / "zones.plt"
        with tecio.open(str(src), "w", variables=["x", "y", "z", "p", "c"]) as w:
            for title, zone_type, node_map, xyz in zones:
                w.write_fe_zone(
                    zone_type=zone_type,
                    data=[
                        *xyz.T,
                        np.arange(len(xyz)) * 1.5,
                        np.arange(len(node_map)) + 10.0,
                    ],
                    node_map=node_map,
                    value_locations=[ValueLocation.NODAL] * 4
                    + [ValueLocation.CELL_CENTERED],
                    title=title,
                )
        out = tmp_path / "zones.szplt"

        assert tecelretyp([str(src), "--femixed", "-o", str(out)]) == 0

        with tecio.open(str(out), "r") as r:
            solids, tets, pyramids, hexahedron = r.zones
            assert isinstance(solids, tecio.TecplotFEMixedZoneReader)
            assert solids.section_num_elements == (2, 1, 2, 1)

            assert tets.zone_type == TET
            tet_maps = [np.array([[0, 1, 3, 4], [1, 10, 2, 6]])]
            node_order = tecio.match_nodes(
                [x[:11], y[:11], z[:11]], tets.get_array(["x", "y", "z"])
            )
            assert node_order is not None
            cell_order = tecio.match_cells(tet_maps, [tets.node_map], node_order)
            assert cell_order is not None
            (got,) = reordered_node_maps([tets.node_map], node_order, cell_order)
            np.testing.assert_array_equal(got, tet_maps[0])

            assert isinstance(pyramids, tecio.TecplotFEMixedZoneReader)
            assert pyramids.section_shapes == (FeCellShape.PYRAMID,)
            assert pyramids.section_num_elements == (2,)

            assert hexahedron.zone_type == BRICK
            node_order = tecio.match_nodes(
                [x[:8], y[:8], z[:8]], hexahedron.get_array(["x", "y", "z"])
            )
            assert node_order is not None
            cell_order = tecio.match_cells(
                [cells[[2]]], [hexahedron.node_map], node_order
            )
            assert cell_order is not None
            (got,) = reordered_node_maps([hexahedron.node_map], node_order, cell_order)
            np.testing.assert_array_equal(got, cells[[2]])

    def test_femixed_zones_sharing_connectivity(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Zones that share one node map are written alike, or all left alone.

        Demonstrates:
        - A zone group becomes FEMIXED together, every zone keeps sharing from the owner
          and has its cell-centered values reordered like the owner's
        - Two zones with node maps of their own that share a cell-centered variable
          cannot be reordered independently, so they are left unchanged
        """
        x, y, z, cells = create_fe_padded_brick()
        locations = [ValueLocation.NODAL, ValueLocation.CELL_CENTERED]
        src = tmp_path / "shared.plt"
        with tecio.open(str(src), "w", variables=["x", "y", "z", "p", "c"]) as w:
            w.write_fe_zone(
                zone_type=BRICK,
                data=[x, y, z, np.arange(len(x)) * 1.5, np.arange(6) + 10.0],
                node_map=cells,
                value_locations=[ValueLocation.NODAL] * 4
                + [ValueLocation.CELL_CENTERED],
                title="owner",
            )
            w.write_fe_zone(
                zone_type=BRICK,
                data=[np.arange(len(x)) * 2.0, np.arange(6) + 100.0],
                var_sharing={"x": 0, "y": 0, "z": 0},
                value_locations=locations,
                con_sharing=0,
                title="sharer",
            )
        out = tmp_path / "shared.szplt"

        assert tecelretyp([str(src), "--femixed", "-o", str(out)]) == 0

        with tecio.open(str(out), "r") as r:
            owner, sharer = r.zones
            assert isinstance(owner, tecio.TecplotFEMixedZoneReader)
            assert isinstance(sharer, tecio.TecplotFEMixedZoneReader)
            assert sharer.shared_connectivity == 0
            assert sharer.variables["x"].shared_zone == 0
            np.testing.assert_array_equal(
                sharer.variables["c"].values - 100, owner.variables["c"].values - 10
            )

        linked = tmp_path / "linked.plt"
        with tecio.open(str(linked), "w", variables=["x", "y", "z", "p", "c"]) as w:
            w.write_fe_zone(
                zone_type=BRICK,
                data=[x, y, z, np.arange(len(x)) * 1.5, np.arange(6) + 10.0],
                node_map=cells,
                value_locations=[ValueLocation.NODAL] * 4
                + [ValueLocation.CELL_CENTERED],
                title="a",
            )
            w.write_fe_zone(
                zone_type=BRICK,
                data=[np.arange(len(x)) * 2.0],
                node_map=cells,
                var_sharing={"x": 0, "y": 0, "z": 0, "c": 0},
                value_locations=[ValueLocation.NODAL],
                title="b",
            )

        assert tecelretyp(["--check", "--femixed", str(linked)]) == 0

        out = capsys.readouterr().out
        assert "unchanged (shares a cell-centered variable with zone 1)" in out
        assert "unchanged (shares a cell-centered variable with zone 0)" in out

    @pytest.mark.parametrize("fmt", ["plt", "dat"])
    def test_femixed_discards_face_neighbors(
        self, fmt: str, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """SZPLT stores no face neighbors, so --femixed does not stop for them.

        Demonstrates:
        - The dry run says that the face-neighbor data is discarded
        - The FEMIXED zone is written without it, where retyping alone would refuse
        """
        x, y, z, cells = create_fe_padded_brick()
        src = tmp_path / f"fn.{fmt}"
        with tecio.open(str(src), "w") as w:
            w.write_fe_zone(
                zone_type=BRICK,
                data=[x, y, z],
                node_map=cells,
                variables=["x", "y", "z"],
                face_neighbors=np.array([0, 1, 1]),
                face_neighbor_mode=FaceNeighborMode.LOCAL_ONE_TO_ONE,
            )

        assert tecelretyp(["--dry-run", str(src)]) == 0

        out = capsys.readouterr().out
        assert "could be FEMIXED" in out
        assert "writes 1 zone as FEMIXED, discards face-neighbor data" in out
        written = tmp_path / "out.szplt"
        assert tecelretyp([str(src), "--femixed", "-o", str(written)]) == 0
        with tecio.open(str(written), "r") as r:
            assert isinstance(r.zones[0], tecio.TecplotFEMixedZoneReader)

    def test_existing_femixed_zones_are_copied_unchanged(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A FEMIXED zone in the input is copied, beside zones that are retyped.

        Demonstrates:
        - The summary lists the zone as unchanged and a file with nothing else to do
          is not rewritten
        - In a file that also holds a retypable zone, the FEMIXED zone keeps its
          sections and node maps
        """
        x, y, z, shapes, maps = create_femixed_volume()
        volume = tmp_path / "volume.szplt"
        with tecio.open(str(volume), "w") as w:
            w.write_femixed_zone(
                data=[x, y, z],
                cell_shapes=shapes,
                node_maps=maps,
                variables=["x", "y", "z"],
            )
        untouched = tmp_path / "untouched.szplt"

        assert tecelretyp([str(volume), "-o", str(untouched)]) == 0

        assert "FEMIXED" in capsys.readouterr().out
        assert not untouched.exists()

        tets = tecio.expand_cells([[0, 1, 2, 3]], TET, BRICK)
        both = tmp_path / "both.szplt"
        with tecio.open(str(both), "w", variables=["x", "y", "z"]) as w:
            w.write_fe_zone(
                zone_type=BRICK, data=[c[:4] for c in (x, y, z)], node_map=tets
            )
            w.write_femixed_zone(data=[x, y, z], cell_shapes=shapes, node_maps=maps)
        retyped = tmp_path / "retyped.szplt"

        assert tecelretyp([str(both), "-o", str(retyped)]) == 0

        with tecio.open(str(retyped), "r") as r:
            assert r.zones[0].zone_type == TET
            femixed = r.zones[1]
            assert isinstance(femixed, tecio.TecplotFEMixedZoneReader)
            assert femixed.section_shapes == tuple(shapes)
            node_order = tecio.match_nodes(
                [x, y, z], femixed.get_array(["x", "y", "z"])
            )
            assert node_order is not None
            cell_order = tecio.match_cells(maps, femixed.node_maps, node_order)
            assert cell_order is not None
            for got, want in zip(
                reordered_node_maps(femixed.node_maps, node_order, cell_order),
                maps,
                strict=True,
            ):
                np.testing.assert_array_equal(got, want)

    def test_dry_run_estimates_and_suggests_commands(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Sizes of each option and the commands that give them follow the table.

        Demonstrates:
        - The table gives the connectivity size now, after retyping, after retyping and
          writing FEMIXED zones, and as poly zones (an estimate only)
        - Each suggested command shows what it saves and does, and the path is quoted
          so that it can be pasted into a shell
        - ``--zones`` is kept in a suggestion, and only commands that save anything for
          the listed zones are suggested
        """
        x, y, z, cells = create_fe_padded_brick()
        xyz = np.column_stack([x, y, z])
        src = tmp_path / "flow data.plt"
        # Solids need FEMIXED, tets only a retype and the hexahedron nothing
        with tecio.open(str(src), "w", variables=["x", "y", "z", "p", "c"]) as w:
            for title, node_map, nodes in (
                ("solids", cells, xyz),
                ("tets", cells[[0, 3]], xyz[:11]),
                ("hexahedron", cells[[2]], xyz[:8]),
            ):
                w.write_fe_zone(
                    zone_type=BRICK,
                    data=[
                        *nodes.T,
                        np.arange(len(nodes)) * 1.5,
                        np.arange(len(node_map)) + 10.0,
                    ],
                    node_map=node_map,
                    value_locations=[ValueLocation.NODAL] * 4
                    + [ValueLocation.CELL_CENTERED],
                    title=title,
                )

        assert tecelretyp(["--dry-run", str(src)]) == 0

        out = capsys.readouterr().out
        assert "Estimated size of the connectivity" in out
        rows = dict(
            re.findall(
                r"^  (Current|Retype|Retype \+ FEMIXED|Poly \(estimate only\))\s+(\d+ B)",
                out,
                flags=re.MULTILINE,
            )
        )
        # 4-byte entries. Current: 6 + 2 + 1 cells of 8 nodes. Retype: the tets drop to
        # 4 nodes. FEMIXED: the solids keep 2 x 4 + 8 + 2 x 5 + 6 nodes. Poly: faces
        # per cell of 48 B (tet), 84 B (hexahedron), 62 B (pyramid) and 66 B (prism)
        assert rows == {
            "Current": "288 B",
            "Retype": "256 B",
            "Retype + FEMIXED": "192 B",
            "Poly (estimate only)": "550 B",
        }
        assert "Suggested commands" in out
        assert f"tecelretyp '{src}'\n      -32 B" in out
        assert "retypes 1 zone\n" in out
        assert f"tecelretyp --femixed '{src}'\n      -96 B" in out
        assert "retypes 1 zone, writes 1 zone as FEMIXED" in out

        assert tecelretyp(["--dry-run", "-z", "0", str(src)]) == 0
        out = capsys.readouterr().out
        assert f"tecelretyp --femixed -z 0 '{src}'" in out
        assert f"tecelretyp -z 0 '{src}'" not in out

        assert tecelretyp(["--dry-run", "-z", "2", str(src)]) == 0
        assert "Suggested commands: none" in capsys.readouterr().out


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

    def test_dry_run_estimates_count_poly_zones(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """A polygon zone is counted at its face arrays, and is the same as poly output."""
        src = tmp_path / "in.plt"
        _write_with_aux_and_polygon(src)
        assert tecelretyp(["--dry-run", str(src)]) == 0
        out = capsys.readouterr().out
        # 3 upcast bricks (96 B) and 7 faces of 2 nodes, 2 elements (112 B) -> 208 B
        assert "Current" in out and "208 B" in out
        assert "Retype " in out and "160 B" in out
        assert "Poly (estimate only)" in out and "256 B" in out

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
