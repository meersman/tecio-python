#!/usr/bin/env python3
"""pytest tests for mixed finite-element (``FEMIXED``) zones.

FEMIXED zones are SZPLT only. This suite covers:

    1. Round trips through ``TecplotSzlWriter.write_mixed_zone`` and
       :class:`tecio.TecplotMixedZoneReader` (metadata, node maps, nodal and
       cell-centered data, sharing, aux data, append mode).
    2. Input validation, including the one case where the C library would crash
       rather than raise (fewer variables than the cell dimension).
    3. ``NotImplementedError`` from the PLT and DAT writers, which keep the writer
       interface uniform while declining the zone type.
    4. The CLI tools that copy zones, which must carry mixed zones through unchanged.
"""

# ruff: noqa: SIM117

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from create_test_data import create_fe_mixed

import tecio
from tecio import FeCellShape, TecplotMixedZoneReader, ValueLocation, ZoneType
from tecio.cli.tec2mat import main as tec2mat
from tecio.cli.tecaux import main as tecaux
from tecio.cli.tecdump import main as tecdump
from tecio.cli.tecelretyp import main as tecelretyp
from tecio.cli.tecextract import main as tecextract
from tecio.cli.tecfix import main as tecfix
from tecio.cli.tecmerge import main as tecmerge
from tecio.cli.teconvert import main as teconvert
from tecio.cli.tecscale import main as tecscale
from tecio.cli.tecslice import main as tecslice

# ======================================================================================
# Test data
# ======================================================================================

VOLUME_SHAPES = [
    FeCellShape.TETRAHEDRON,
    FeCellShape.PYRAMID,
    FeCellShape.PRISM,
    FeCellShape.HEXAHEDRON,
]


def _volume_mesh() -> tuple[list[np.ndarray], list[FeCellShape], list[np.ndarray]]:
    """One tet, pyramid, prism and hex, as ``[x, y, z]``, shapes and section maps.

    Returns:
        The coordinate arrays, the cell shape of each section and one node map per
        section.
    """
    x, y, z, node_map, nodes_per_cell = create_fe_mixed()
    ends = np.cumsum(nodes_per_cell)
    starts = ends - nodes_per_cell
    maps = [node_map[s:e].reshape(1, -1) for s, e in zip(starts, ends, strict=True)]
    return [x, y, z], list(VOLUME_SHAPES), maps


def _surface_mesh() -> tuple[list[np.ndarray], list[FeCellShape], list[np.ndarray]]:
    """Two triangles and one quad as ``[x, y]``, shapes and section maps.

    Returns:
        The coordinate arrays, the cell shape of each section and one node map per
        section.
    """
    x = np.array([0, 1, 2, 0, 1, 2], dtype=np.float64)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    tris = np.array([[0, 1, 4], [0, 4, 3]])
    quads = np.array([[1, 2, 5, 4]])
    return [x, y], [FeCellShape.TRIANGLE, FeCellShape.QUADRILATERAL], [tris, quads]


def _rows(arr: np.ndarray) -> np.ndarray:
    """Sort the rows of *arr*, to compare cell lists regardless of cell order.

    Returns:
        The rows of *arr* in lexicographic order.
    """
    return arr[np.lexsort(arr.T[::-1])]


def _write_volume(path: Path, **kwargs) -> None:
    coords, shapes, maps = _volume_mesh()
    pressure = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    kwargs.setdefault("variables", ["x", "y", "z", "p"])
    kwargs.setdefault(
        "value_locations", [ValueLocation.NODAL] * 3 + [ValueLocation.CELL_CENTERED]
    )
    with tecio.open(str(path), "w") as w:
        w.write_mixed_zone([*coords, pressure], shapes, maps, **kwargs)


@pytest.fixture
def volume_file(tmp_path: Path) -> Path:
    """A SZPLT file with one volume mixed zone.

    Returns:
        The file path.
    """
    path = tmp_path / "volume.szplt"
    _write_volume(path, title="mix", solution_time=1.5, strand_id=1, aux={"a": "b"})
    return path


# ======================================================================================
# Round trips
# ======================================================================================


class TestMixedRoundTrip:
    """Write with ``write_mixed_zone``, read back as a mixed zone reader."""

    def test_volume_metadata_and_connectivity(self, volume_file: Path) -> None:
        """Zone counts, section layout, node maps and scalar metadata round trip."""
        _, _, maps = _volume_mesh()
        with tecio.open(str(volume_file)) as r:
            zone = r.zones[0]
            assert isinstance(zone, TecplotMixedZoneReader)
            assert zone.zone_type == ZoneType.FEMIXED
            assert zone.title == "mix"
            assert zone.num_nodes == 18
            assert zone.num_elements == 4
            assert zone.num_sections == 4
            assert zone.section_shapes == tuple(VOLUME_SHAPES)
            assert zone.section_num_elements == (1, 1, 1, 1)
            assert zone.shared_connectivity is None
            assert zone.solution_time == 1.5
            assert zone.strand_id == 1
            assert dict(zone.auxdata.items()) == {"a": "b"}
            assert len(zone.node_maps) == 4
            for got, want in zip(zone.node_maps, maps, strict=True):
                assert got.dtype == np.int64
                np.testing.assert_array_equal(got, want)

    def test_node_maps_are_cached(self, volume_file: Path) -> None:
        """Repeated access returns the same arrays."""
        with tecio.open(str(volume_file)) as r:
            zone = r.zones[0]
            assert zone.node_maps is zone.node_maps

    def test_variables_and_cell_centered_data(self, volume_file: Path) -> None:
        """Nodal data has ``num_nodes`` values, cell-centered ``num_elements``."""
        coords, _, _ = _volume_mesh()
        with tecio.open(str(volume_file)) as r:
            zone = r.zones[0]
            for name, want in zip("xyz", coords, strict=True):
                np.testing.assert_array_equal(zone.variables[name].values, want)
            pressure = zone.variables["p"]
            assert pressure.value_location == ValueLocation.CELL_CENTERED
            np.testing.assert_array_equal(pressure.values, [1.0, 2.0, 3.0, 4.0])

    def test_surface_zone(self, tmp_path: Path) -> None:
        """Triangle and quad sections with two coordinate variables."""
        coords, shapes, maps = _surface_mesh()
        path = tmp_path / "surface.szplt"
        with tecio.open(str(path), "w") as w:
            w.write_mixed_zone(coords, shapes, maps, variables=["x", "y"])
        with tecio.open(str(path)) as r:
            zone = r.zones[0]
            assert zone.section_shapes == tuple(shapes)
            assert zone.section_num_elements == (2, 1)
            assert zone.num_nodes == 6
            assert zone.num_elements == 3
            assert zone.node_maps[0].shape == (2, 3)
            assert zone.node_maps[1].shape == (1, 4)
            for got, want in zip(zone.node_maps, maps, strict=True):
                np.testing.assert_array_equal(_rows(got), _rows(want))

    def test_many_cells_per_section(self, tmp_path: Path) -> None:
        """A larger zone keeps every cell's geometry, whatever order is stored."""
        rng = np.random.default_rng(0)
        num_nodes = 5000
        xyz = rng.random((3, num_nodes))
        shapes = [FeCellShape.TETRAHEDRON, FeCellShape.HEXAHEDRON]
        maps = [
            rng.integers(0, num_nodes, size=(3000, 4)),
            rng.integers(0, num_nodes, size=(1000, 8)),
        ]
        cell_values = rng.random(4000)
        path = tmp_path / "many.szplt"
        with tecio.open(str(path), "w") as w:
            w.write_mixed_zone(
                [*xyz, cell_values],
                shapes,
                maps,
                variables=["x", "y", "z", "c"],
                value_locations=[ValueLocation.NODAL] * 3
                + [ValueLocation.CELL_CENTERED],
            )
        with tecio.open(str(path)) as r:
            zone = r.zones[0]
            assert zone.section_num_elements == (3000, 1000)
            out_xyz = zone.get_array(["x", "y", "z"])
            out_c = zone.get_array("c")
            start = 0
            for want, got in zip(maps, zone.node_maps, strict=True):
                n = len(want)
                sig_in = np.concatenate([xyz[k][want] for k in range(3)], axis=1)
                sig_out = np.concatenate([out_xyz[k][got] for k in range(3)], axis=1)
                order_in = np.lexsort(sig_in.T[::-1])
                order_out = np.lexsort(sig_out.T[::-1])
                np.testing.assert_array_equal(sig_in[order_in], sig_out[order_out])
                # The cell-centered value must follow its cell
                np.testing.assert_array_equal(
                    cell_values[start : start + n][order_in],
                    out_c[start : start + n][order_out],
                )
                start += n

    def test_empty_section_allowed(self, tmp_path: Path) -> None:
        """A section with no cells is stored and read back with zero cells."""
        coords, shapes, maps = _surface_mesh()
        shapes = [*shapes, FeCellShape.TRIANGLE]
        maps = [*maps, np.zeros((0, 3), dtype=np.int64)]
        path = tmp_path / "empty.szplt"
        with tecio.open(str(path), "w") as w:
            w.write_mixed_zone(coords, shapes, maps, variables=["x", "y"])
        with tecio.open(str(path)) as r:
            zone = r.zones[0]
            assert zone.section_num_elements == (2, 1, 0)
            assert zone.node_maps[2].shape == (0, 3)

    def test_shape_names_values_and_flat_maps(self, tmp_path: Path) -> None:
        """Shapes may be given by name or value, and maps may be flat."""
        coords, _, maps = _surface_mesh()
        path = tmp_path / "forms.szplt"
        with tecio.open(str(path), "w") as w:
            w.write_mixed_zone(
                coords,
                ["triangle", FeCellShape.QUADRILATERAL.value],
                [maps[0].ravel(), maps[1].ravel()],
                variables=["x", "y"],
            )
        with tecio.open(str(path)) as r:
            zone = r.zones[0]
            assert zone.section_shapes == (
                FeCellShape.TRIANGLE,
                FeCellShape.QUADRILATERAL,
            )
            assert zone.node_maps[0].shape == (2, 3)

    def test_default_title_and_variable_names(self, tmp_path: Path) -> None:
        """Lazy open names the variables V1.. and the zone ``Mixed_Zone_<n>``."""
        coords, shapes, maps = _surface_mesh()
        path = tmp_path / "defaults.szplt"
        with tecio.open(str(path), "w") as w:
            w.write_mixed_zone(coords, shapes, maps)
            w.write_mixed_zone(coords, shapes, maps)
        with tecio.open(str(path)) as r:
            assert r.variables == ["V1", "V2"]
            assert [z.title for z in r.zones] == ["Mixed_Zone_0", "Mixed_Zone_1"]

    def test_writer_meta_records_sections(self, tmp_path: Path) -> None:
        """The writer's metadata record carries the section layout."""
        coords, shapes, maps = _surface_mesh()
        with tecio.open(str(tmp_path / "meta.szplt"), "w") as w:
            w.write_mixed_zone(coords, shapes, maps, variables=["x", "y"])
            meta = w.meta.zone(0)
        assert meta is not None
        assert meta.zone_type == ZoneType.FEMIXED
        assert meta.section_shapes == tuple(shapes)
        assert meta.section_num_elements == (2, 1)
        assert meta.num_elements == 3
        assert meta.num_nodes == 6

    def test_peek_and_zone_list_report_mixed_zone(self, volume_file: Path) -> None:
        """The summary helpers handle a mixed zone."""
        summary = tecio.peek(str(volume_file))
        zone = summary.zones[0]
        assert zone.zone_type == ZoneType.FEMIXED
        assert (zone.num_nodes, zone.num_elements) == (18, 4)


class TestMixedSharing:
    """Connectivity and variable sharing between mixed zones."""

    def test_connectivity_and_variable_sharing(self, tmp_path: Path) -> None:
        """A transient zone shares the grid and the node map of the first zone."""
        coords, shapes, maps = _volume_mesh()
        locations = [ValueLocation.CELL_CENTERED]
        path = tmp_path / "shared.szplt"
        with tecio.open(str(path), "w") as w:
            w.write_mixed_zone(
                [*coords, np.array([1.0, 2.0, 3.0, 4.0])],
                shapes,
                maps,
                variables=["x", "y", "z", "p"],
                value_locations=[ValueLocation.NODAL] * 3 + locations,
            )
            w.write_mixed_zone(
                [np.array([5.0, 6.0, 7.0, 8.0])],
                con_sharing=0,
                var_sharing={"x": 0, "y": 0, "z": 0},
                value_locations=locations,
                solution_time=1.0,
                strand_id=1,
            )
        with tecio.open(str(path)) as r:
            second = r.zones[1]
            assert isinstance(second, TecplotMixedZoneReader)
            assert second.shared_connectivity == 0
            assert second.section_shapes == tuple(shapes)
            assert second.num_nodes == 18
            for got, want in zip(second.node_maps, maps, strict=True):
                np.testing.assert_array_equal(got, want)
            assert second.variables["x"].shared_zone == 0
            np.testing.assert_array_equal(second.variables["p"].values, [5, 6, 7, 8])

    def test_con_sharing_source_must_be_mixed(self, tmp_path: Path) -> None:
        """Sharing connectivity from a zone that is not a mixed zone is an error."""
        x = np.array([0.0, 1.0, 0.0, 0.0])
        with tecio.open(str(tmp_path / "bad.szplt"), "w") as w:
            w.write_fe_zone(
                [x, x, x],
                ZoneType.FETETRAHEDRON,
                node_map=np.array([[0, 1, 2, 3]]),
                variables=["x", "y", "z"],
            )
            with pytest.raises(ValueError, match="not a mixed"):
                w.write_mixed_zone([x, x, x], con_sharing=0)


class TestMixedAppend:
    """Zone copying in append mode."""

    def test_append_copies_mixed_zone_and_adds_one(self, volume_file: Path) -> None:
        """Mode ``a`` re-emits the mixed zone, then a new one can follow."""
        coords, shapes, maps = _volume_mesh()
        with tecio.open(str(volume_file), "a") as a:
            a.write_mixed_zone(
                [np.array([9.0, 9.0, 9.0, 9.0])],
                con_sharing=0,
                var_sharing={"x": 0, "y": 0, "z": 0},
                value_locations=[ValueLocation.CELL_CENTERED],
                title="added",
            )
        with tecio.open(str(volume_file)) as r:
            assert [z.title for z in r.zones] == ["mix", "added"]
            first = r.zones[0]
            assert first.section_shapes == tuple(shapes)
            assert first.solution_time == 1.5
            assert dict(first.auxdata.items()) == {"a": "b"}
            for got, want in zip(first.node_maps, maps, strict=True):
                np.testing.assert_array_equal(got, want)
            np.testing.assert_array_equal(
                r.zones[0].variables["p"].values, [1, 2, 3, 4]
            )


# ======================================================================================
# Errors
# ======================================================================================


class TestMixedErrors:
    """Invalid input to ``write_mixed_zone``."""

    @staticmethod
    def _write(path: Path, *args, **kwargs) -> None:
        with tecio.open(str(path), "w") as w:
            w.write_mixed_zone(*args, **kwargs)

    def test_section_count_mismatch(self, tmp_path: Path) -> None:
        coords, shapes, maps = _surface_mesh()
        with pytest.raises(ValueError, match="2 sections but node_maps has 1"):
            self._write(tmp_path / "e.szplt", coords, shapes, maps[:1])

    def test_too_many_sections(self, tmp_path: Path) -> None:
        coords, _, maps = _surface_mesh()
        with pytest.raises(ValueError, match="1 to 16 sections"):
            self._write(
                tmp_path / "e.szplt",
                coords,
                [FeCellShape.TRIANGLE] * 17,
                [maps[0]] * 17,
            )

    def test_no_sections(self, tmp_path: Path) -> None:
        coords, _, _ = _surface_mesh()
        with pytest.raises(ValueError, match="1 to 16 sections"):
            self._write(tmp_path / "e.szplt", coords, [], [])

    def test_no_cells(self, tmp_path: Path) -> None:
        coords, _, _ = _surface_mesh()
        with pytest.raises(ValueError, match="at least one cell"):
            self._write(
                tmp_path / "e.szplt",
                coords,
                [FeCellShape.TRIANGLE],
                [np.zeros((0, 3), dtype=np.int64)],
            )

    def test_wrong_nodes_per_cell(self, tmp_path: Path) -> None:
        coords, _, maps = _surface_mesh()
        with pytest.raises(ValueError, match=r"\(num_cells, 4\)"):
            self._write(
                tmp_path / "e.szplt", coords, [FeCellShape.QUADRILATERAL], [maps[0]]
            )

    def test_non_integer_node_map(self, tmp_path: Path) -> None:
        coords, shapes, maps = _surface_mesh()
        with pytest.raises(ValueError, match="integers"):
            self._write(tmp_path / "e.szplt", coords, shapes, [maps[0] * 1.0, maps[1]])

    def test_negative_node_index(self, tmp_path: Path) -> None:
        coords, shapes, maps = _surface_mesh()
        with pytest.raises(ValueError, match="negative"):
            self._write(tmp_path / "e.szplt", coords, shapes, [maps[0] - 1, maps[1]])

    def test_unknown_shape_name(self, tmp_path: Path) -> None:
        coords, _, maps = _surface_mesh()
        with pytest.raises(ValueError, match="Unknown cell shape"):
            self._write(tmp_path / "e.szplt", coords, ["blob"], [maps[0]])

    def test_mixed_dimensions(self, tmp_path: Path) -> None:
        coords, _, _ = _volume_mesh()
        with pytest.raises(ValueError, match="same spatial dimension"):
            self._write(
                tmp_path / "e.szplt",
                coords,
                [FeCellShape.TRIANGLE, FeCellShape.TETRAHEDRON],
                [np.array([[0, 1, 2]]), np.array([[0, 1, 2, 3]])],
            )

    def test_fewer_variables_than_cell_dimension(self, tmp_path: Path) -> None:
        """The C library crashes here, so the writer must raise first."""
        coords, shapes, maps = _volume_mesh()
        with pytest.raises(ValueError, match="at least 3"):
            self._write(
                tmp_path / "e.szplt", coords[:2], shapes, maps, variables=["x", "y"]
            )

    def test_cell_centered_length_is_total_cells(self, tmp_path: Path) -> None:
        coords, shapes, maps = _volume_mesh()
        with pytest.raises(ValueError, match="CELL_CENTERED but has 3 values"):
            self._write(
                tmp_path / "e.szplt",
                [*coords, np.zeros(3)],
                shapes,
                maps,
                value_locations=[ValueLocation.NODAL] * 3
                + [ValueLocation.CELL_CENTERED],
            )

    def test_missing_connectivity(self, tmp_path: Path) -> None:
        coords, _, _ = _surface_mesh()
        with pytest.raises(ValueError, match="must be provided"):
            self._write(tmp_path / "e.szplt", coords)

    def test_connectivity_with_con_sharing(self, tmp_path: Path) -> None:
        coords, shapes, maps = _surface_mesh()
        with tecio.open(str(tmp_path / "e.szplt"), "w") as w:
            w.write_mixed_zone(coords, shapes, maps, variables=["x", "y"])
            with pytest.raises(ValueError, match="must be omitted"):
                w.write_mixed_zone([], shapes, maps, con_sharing=0)

    def test_con_sharing_unwritten_zone(self, tmp_path: Path) -> None:
        coords, _, _ = _surface_mesh()
        with pytest.raises(ValueError, match="has not been written"):
            self._write(tmp_path / "e.szplt", coords, con_sharing=3)

    def test_point_datapacking(self, tmp_path: Path) -> None:
        coords, shapes, maps = _surface_mesh()
        with pytest.raises(NotImplementedError, match="POINT"):
            self._write(tmp_path / "e.szplt", coords, shapes, maps, datapacking="POINT")


# ======================================================================================
# PLT and DAT cannot hold mixed zones
# ======================================================================================


@pytest.mark.parametrize("ext", ["plt", "dat"])
class TestMixedUnsupportedFormats:
    """PLT and DAT writers expose ``write_mixed_zone`` but decline the zone type."""

    def test_write_mixed_zone_not_implemented(
        self, ext: str, output_path: Callable
    ) -> None:
        coords, shapes, maps = _surface_mesh()
        with pytest.raises(NotImplementedError, match="FEMIXED"):
            with tecio.open(str(output_path(f"mixed.{ext}")), "w") as w:
                w.write_mixed_zone(coords, shapes, maps, variables=["x", "y"])

    def test_message_names_the_format_and_alternative(
        self, ext: str, output_path: Callable
    ) -> None:
        coords, shapes, maps = _surface_mesh()
        with pytest.raises(NotImplementedError) as info:
            with tecio.open(str(output_path(f"mixed.{ext}")), "w") as w:
                w.write_mixed_zone(coords, shapes, maps, variables=["x", "y"])
        message = str(info.value)
        assert ext.upper() in message
        assert ".szplt" in message

    def test_fe_zone_writer_does_not_accept_femixed(
        self, ext: str, output_path: Callable
    ) -> None:
        """``write_fe_zone`` is no way around it."""
        coords, _, maps = _surface_mesh()
        with pytest.raises(NotImplementedError):
            with tecio.open(str(output_path(f"mixed.{ext}")), "w") as w:
                w.write_fe_zone(
                    coords,
                    ZoneType.FEMIXED,
                    node_map=maps[0],
                    variables=["x", "y"],
                )


# ======================================================================================
# CLI tools
# ======================================================================================


def _assert_same_mixed_zone(path: Path, source: Path) -> None:
    """Assert that the first zone of *path* matches that of *source*."""
    with tecio.open(str(source)) as src, tecio.open(str(path)) as out:
        a, b = src.zones[0], out.zones[0]
        assert isinstance(b, TecplotMixedZoneReader)
        assert b.section_shapes == a.section_shapes
        assert b.section_num_elements == a.section_num_elements
        assert b.num_nodes == a.num_nodes
        for got, want in zip(b.node_maps, a.node_maps, strict=True):
            np.testing.assert_array_equal(got, want)


class TestMixedCli:
    """Tools that copy zones carry a mixed zone through unchanged."""

    def test_tecdump(self, volume_file: Path, capsys) -> None:
        assert tecdump([str(volume_file)]) == 0
        out = capsys.readouterr().out
        assert "FEMIXED" in out
        assert "Num Sections" in out
        assert "PYRAMID" in out

    @pytest.mark.parametrize(
        "tool,extra",
        [
            (tecextract, []),
            (tecfix, []),
            (tecscale, ["-v", "x", "-s", "2.0"]),
            (tecaux, ["-d", "Solver=Test"]),
            (tecslice, ["-i", "::2"]),
        ],
        ids=lambda v: getattr(v, "__module__", "").rsplit(".", 1)[-1] or None,
    )
    def test_copy_tools_preserve_mixed_zone(
        self, tool, extra: list[str], volume_file: Path, tmp_path: Path
    ) -> None:
        dst = tmp_path / "out.szplt"
        assert tool([*extra, "-o", str(dst), "--force", str(volume_file)]) == 0
        _assert_same_mixed_zone(dst, volume_file)
        with tecio.open(str(dst)) as r:
            np.testing.assert_array_equal(
                r.zones[0].variables["p"].values, [1, 2, 3, 4]
            )

    def test_tecelretyp_copies_mixed_zone_beside_retyped_zone(
        self, tmp_path: Path
    ) -> None:
        """A brick zone holding only tets is retyped, the mixed zone is copied."""
        coords, shapes, maps = _volume_mesh()
        src = tmp_path / "two_zones.szplt"
        tet = tecio.expand_cells(
            [[0, 1, 2, 3]], ZoneType.FETETRAHEDRON, ZoneType.FEBRICK
        )
        with tecio.open(str(src), "w") as w:
            w.write_fe_zone(
                [c[:4] for c in coords],
                ZoneType.FEBRICK,
                node_map=tet,
                variables=["x", "y", "z"],
            )
            w.write_mixed_zone(coords, shapes, maps)
        dst = tmp_path / "retyped.szplt"
        assert tecelretyp([str(src), "-o", str(dst)]) == 0
        with tecio.open(str(dst)) as r:
            assert r.zones[0].zone_type == ZoneType.FETETRAHEDRON
            mixed = r.zones[1]
            assert isinstance(mixed, TecplotMixedZoneReader)
            assert mixed.section_shapes == tuple(shapes)

    def test_tecelretyp_reports_mixed_zone_unchanged(
        self, volume_file: Path, tmp_path: Path, capsys
    ) -> None:
        dst = tmp_path / "out.szplt"
        assert tecelretyp([str(volume_file), "-o", str(dst)]) == 0
        assert "FEMIXED" in capsys.readouterr().out
        assert not dst.exists()

    def test_tecscale_scales_the_data(self, volume_file: Path, tmp_path: Path) -> None:
        dst = tmp_path / "scaled.szplt"
        assert tecscale(["-v", "x", "-s", "2.0", "-o", str(dst), str(volume_file)]) == 0
        with tecio.open(str(volume_file)) as src, tecio.open(str(dst)) as out:
            np.testing.assert_allclose(
                out.zones[0].variables["x"].values,
                2.0 * src.zones[0].variables["x"].values,
            )

    def test_tecmerge_two_files(self, volume_file: Path, tmp_path: Path) -> None:
        other = tmp_path / "other.szplt"
        _write_volume(other)
        dst = tmp_path / "merged.szplt"
        assert tecmerge(["-o", str(dst), str(volume_file), str(other)]) == 0
        with tecio.open(str(dst)) as r:
            assert r.num_zones == 2
            assert all(isinstance(z, TecplotMixedZoneReader) for z in r.zones)

    def test_teconvert_to_plt_fails_clearly(
        self, volume_file: Path, tmp_path: Path, capsys
    ) -> None:
        dst = tmp_path / "out.plt"
        assert teconvert(["-plt", "-o", str(dst), "--force", str(volume_file)]) == 1
        assert "FEMIXED" in capsys.readouterr().err

    def test_tec2mat_exports_sections(self, volume_file: Path, tmp_path: Path) -> None:
        sio = pytest.importorskip("scipy.io")
        dst = tmp_path / "out.mat"
        assert tec2mat(["-o", str(dst), str(volume_file)]) == 0
        zone = sio.loadmat(str(dst), squeeze_me=True, struct_as_record=False)["zone_1"]
        assert int(zone.num_sections) == 4
        assert list(zone.section_shapes) == [s.name for s in VOLUME_SHAPES]
        assert list(zone.section_num_elements) == [1, 1, 1, 1]
        _, _, maps = _volume_mesh()
        for k, want in enumerate(maps, start=1):
            # MATLAB output is 1-based
            np.testing.assert_array_equal(
                np.atleast_2d(getattr(zone, f"node_map_{k}")), want + 1
            )
