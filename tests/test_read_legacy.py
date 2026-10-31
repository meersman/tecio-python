#!/usr/bin/env python3
"""Hand-authored, byte-level v75 tests for paths no real sample exercises.

:mod:`test_read_legacy` covers everything a real v75/v107/v111 file can exercise. A
handful of v75 code paths have no real sample at all though: no ORDERED zone, no
BLOCK-packed zone, no variable-sharing zone, no malformed-input error path, and no
geometry/text/custom-label record has ever been seen in a real v75 file in this
project. Rather than ``# pragma: no cover`` all of them, this module hand-builds the
minimum valid byte sequence the v75 spec (``v75pltformat.txt``) calls for and exercises
the rest directly -- the same "hand-authored fixture" pattern :mod:`test_read` uses for
DAT's ``POINT`` format, for the same reason: no external fixture to keep in sync, and
exact values to assert against.

v75 geometry and text records still get ``# pragma: no cover`` rather than a synthetic
test: they are complex, multi-branch structures with no data-reading relevance (see the
general "no text/geometry support" limitation in :mod:`tecio._plt_read`'s own module
docstring).

The variable-sharing convention (1-based repeat numbers, previous-zone source) was
confirmed in the Tecplot GUI with the ``legacy_v75_repeat_*`` fixtures in
:mod:`test_read_legacy`; ``TestVariableSharing`` below combines it with connectivity
sharing, which no GUI-checked file covers.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

import tecio
from tecio import DataPacking, DataType, ZoneType
from tecio._plt_read import PltReadError

_BYTE_ORDER = ">"  # arbitrary; the reader auto-detects either order


def _s(text: str) -> bytes:
    """Encode a v75 string: one INT32 per character, null-terminated."""
    return b"".join(struct.pack(f"{_BYTE_ORDER}i", ord(c)) for c in text) + struct.pack(
        f"{_BYTE_ORDER}i", 0
    )


def _i(value: int) -> bytes:
    """Encode a single INT32."""
    return struct.pack(f"{_BYTE_ORDER}i", value)


def _f32(value: float) -> bytes:
    """Encode a single FLOAT32."""
    return struct.pack(f"{_BYTE_ORDER}f", value)


def _header_prelude(title: str, var_names: list[str]) -> bytes:
    """Magic, byte-order probe, title, and variable names."""
    out = b"#!TDV75 " + _i(1) + _s(title) + _i(len(var_names))
    for name in var_names:
        out += _s(name)
    return out


def _zone_header(name: str, fmt: int, imax: int, jmax: int, kmax: int) -> bytes:
    """One v75 zone-header record, including its own ``299.0`` marker."""
    return _f32(299.0) + _s(name) + _i(fmt) + _i(-1) + _i(imax) + _i(jmax) + _i(kmax)


def _zone_data_block(
    num_vars: int,
    repeat_var_nums: list[int],
    var_formats: list[int],
    var_blocks: list[bytes],
) -> bytes:
    """One v75 BLOCK/ORDERED data record (no connectivity): the common prefix."""
    out = _f32(299.0)
    out += _i(len(repeat_var_nums))
    for v in repeat_var_nums:
        out += _i(v)
    assert len(var_formats) == num_vars
    for f in var_formats:
        out += _i(f)
    for block in var_blocks:
        out += block
    return out


def _pack_values(rows: list[list[float]], *, point: bool, double: bool) -> bytes:
    """Pack ``rows[node][var]`` as one POINT-interleaved or BLOCK-per-variable run."""
    code = "d" if double else "f"
    n_vars = len(rows[0])
    order = (
        [(r, v) for r in range(len(rows)) for v in range(n_vars)]
        if point
        else [(r, v) for v in range(n_vars) for r in range(len(rows))]
    )
    return b"".join(struct.pack(f"{_BYTE_ORDER}{code}", rows[r][v]) for r, v in order)


def _zone_data_record(
    n_vars: int,
    rows: list[list[float]],
    connectivity: list[list[int]] | None,
    *,
    point: bool,
    double: bool,
    repeat: tuple[int, ...] = (),
    repeat_connectivity: bool = False,
) -> bytes:
    """One zone's data record: values, then optional connectivity.

    ``rows`` always holds every variable; the 1-based variable numbers in
    ``repeat`` are declared as repeated from the previous zone and their values
    are not written (in POINT packing they are dropped from every row).
    """
    data = _f32(299.0) + _i(len(repeat)) + b"".join(_i(v) for v in repeat)
    data += _i(2 if double else 1) * n_vars
    keep = [v for v in range(n_vars) if v + 1 not in repeat]
    stored = [[row[v] for v in keep] for row in rows]
    data += _pack_values(stored, point=point, double=double)
    if connectivity is not None:
        if repeat_connectivity:
            data += _i(1)  # connectivity repeated from the previous zone
        else:
            data += _i(0)
            data += b"".join(_i(n) for cell in connectivity for n in cell)
    return data


def _single_zone_file(
    var_names: list[str],
    zone_header: bytes,
    rows: list[list[float]],
    connectivity: list[list[int]] | None,
    *,
    point: bool,
    double: bool,
) -> bytes:
    """One zone, no sharing: prelude, header, data record, optional connectivity."""
    data = _header_prelude("Test", var_names) + zone_header + _f32(357.0)
    return data + _zone_data_record(
        len(var_names), rows, connectivity, point=point, double=double
    )


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# ======================================================================================
# Malformed input
# ======================================================================================


class TestMalformedV75:
    """Error paths: :class:`tecio._plt_read.PltReadError` on bad enum values."""

    def test_unrecognised_format_value_raises(self, tmp_path: Path) -> None:
        """An unrecognised ``Format`` value (not 0-3) raises cleanly."""
        data = _header_prelude("Test", ["X"])
        data += _zone_header("Z1", fmt=99, imax=1, jmax=1, kmax=1)
        data += _f32(357.0)  # EOHMARKER
        path = _write(tmp_path, "bad_format.bin", data)
        with pytest.raises(PltReadError, match="unrecognised v75 Format"):
            tecio.open(str(path), "r")

    def test_unrecognised_element_type_raises(self, tmp_path: Path) -> None:
        """An FE zone with an unrecognised ``KMax`` element-type code raises."""
        data = _header_prelude("Test", ["X"])
        # Format=2 (FEBLOCK): is_fe=True, so KMax is read as element type.
        data += _zone_header("Z1", fmt=2, imax=4, jmax=1, kmax=99)
        data += _f32(357.0)
        path = _write(tmp_path, "bad_element_type.bin", data)
        with pytest.raises(PltReadError, match="unrecognised v75 FE element type"):
            tecio.open(str(path), "r")

    def test_unexpected_header_marker_raises(self, tmp_path: Path) -> None:
        """A header marker that isn't Zone/Geom/Text/CustomLabel/EOH raises."""
        data = _header_prelude("Test", ["X"])
        data += _f32(123.0)  # not a recognised marker
        path = _write(tmp_path, "bad_marker.bin", data)
        with pytest.raises(PltReadError, match="Unexpected v75 header marker"):
            tecio.open(str(path), "r")

    def test_repeat_number_above_num_vars_raises(self, tmp_path: Path) -> None:
        """A repeated variable number above the variable count raises."""
        data = _header_prelude("Test", ["X"])
        data += _zone_header("Z1", fmt=0, imax=1, jmax=1, kmax=1)
        data += _zone_header("Z2", fmt=0, imax=1, jmax=1, kmax=1)
        data += _f32(357.0)
        data += _zone_data_block(1, [], [1], [_f32(1.0)])
        data += _zone_data_block(1, [2], [1], [])  # only 1 variable
        path = _write(tmp_path, "repeat_too_high.bin", data)
        with pytest.raises(PltReadError, match="out of range 1..1"):
            tecio.open(str(path), "r")

    def test_repeat_in_first_zone_raises(self, tmp_path: Path) -> None:
        """Repeating a variable in zone 0 has no previous zone to repeat from."""
        data = _header_prelude("Test", ["X"])
        data += _zone_header("Z1", fmt=0, imax=1, jmax=1, kmax=1)
        data += _f32(357.0)
        data += _zone_data_block(1, [1], [1], [])
        path = _write(tmp_path, "repeat_first_zone.bin", data)
        with pytest.raises(PltReadError, match="no previous zone"):
            tecio.open(str(path), "r")

    def test_data_section_marker_mismatch_raises(self, tmp_path: Path) -> None:
        """A data-section zone marker that doesn't match 299.0 raises."""
        data = _header_prelude("Test", ["X"])
        data += _zone_header("Z1", fmt=0, imax=1, jmax=1, kmax=1)  # ORDERED, BLOCK
        data += _f32(357.0)  # EOHMARKER
        data += _f32(1.0)  # wrong data-section marker (should be 299.0)
        path = _write(tmp_path, "bad_data_marker.bin", data)
        with pytest.raises(PltReadError, match="expected zone marker 299.0"):
            tecio.open(str(path), "r")


# ======================================================================================
# CustomLabel record (header-only, shared helper with every other version)
# ======================================================================================


class TestCustomLabelRecord:
    """v75's CustomLabel record uses the same shared skip helper as v107+."""

    def test_custom_label_with_zero_labels_is_skipped(self, tmp_path: Path) -> None:
        """A CustomLabel record with zero labels doesn't desync later parsing."""
        data = _header_prelude("Test", ["X"])
        data += _f32(599.0) + _i(0)  # CustomLabel marker, NumLabels=0
        data += _f32(357.0)  # EOHMARKER
        path = _write(tmp_path, "custom_label.bin", data)
        with tecio.open(str(path), "r") as r:
            assert r.title == "Test"
            assert len(r.zones) == 0


# ======================================================================================
# ORDERED zones and BLOCK packing (no real v75 sample has either)
# ======================================================================================


class TestOrderedAndBlockPacking:
    """Hand-built ORDERED+BLOCK zone: a literal 2x2x1 grid, two variables.

    Exercises the ``is_fe=False`` branch (``ZoneType.ORDERED``, IMax/JMax/KMax
    as literal dimensions, no connectivity record at all), BLOCK packing's
    contiguous-per-variable layout, and the ORDERED ``num_rows`` computation
    -- none of which any real v75 sample in this project has.
    """

    def test_ordered_block_zone_round_trips(self, tmp_path: Path) -> None:
        x_vals = [1.0, 2.0, 3.0, 4.0]
        y_vals = [10.0, 20.0, 30.0, 40.0]

        data = _header_prelude("Ordered", ["X", "Y"])
        data += _zone_header("Grid", fmt=0, imax=2, jmax=2, kmax=1)  # BLOCK, ORDERED
        data += _f32(357.0)
        data += _zone_data_block(
            num_vars=2,
            repeat_var_nums=[],
            var_formats=[1, 1],  # both float32
            var_blocks=[
                b"".join(_f32(v) for v in x_vals),
                b"".join(_f32(v) for v in y_vals),
            ],
        )
        path = _write(tmp_path, "ordered_block.bin", data)

        with tecio.open(str(path), "r") as r:
            assert len(r.zones) == 1
            zone = r.zones[0]
            assert zone.zone_type == ZoneType.ORDERED
            assert zone.datapacking == DataPacking.BLOCK
            assert zone.dimensions == (2, 2, 1)
            x = zone.variables["X"].get_values()
            y = zone.variables["Y"].get_values()
            assert x is not None
            assert y is not None
            # Full ORDERED reads reshape to (I, J, K) in Fortran order; ravel
            # the same way to get back the on-disk, column-major sequence.
            np.testing.assert_allclose(x.ravel(order="F"), x_vals)
            np.testing.assert_allclose(y.ravel(order="F"), y_vals)


# ======================================================================================
# Variable and connectivity sharing
# ======================================================================================


class TestVariableSharing:
    """Variable plus connectivity sharing in one hand-built file.

    The variable convention is GUI-confirmed (see this module's docstring).
    The connectivity flag is explicit in the spec, but no Tecplot-checked
    file combines it with sharing, so this one is hand-authored only.
    """

    def test_shared_variable_and_connectivity_resolve_to_previous_zone(
        self, tmp_path: Path
    ) -> None:
        # Zone A: a single FE quad (FEBLOCK), 4 points, 1 element.
        a_x = [0.0, 1.0, 1.0, 0.0]
        a_y = [0.0, 0.0, 1.0, 1.0]
        connectivity = [1, 2, 3, 4]  # 1-based, matches v75's on-disk convention

        data = _header_prelude("Sharing", ["X", "Y"])
        data += _zone_header("ZoneA", fmt=2, imax=4, jmax=1, kmax=1)  # FEBLOCK, quad
        data += _zone_header("ZoneB", fmt=2, imax=4, jmax=1, kmax=1)
        data += _f32(357.0)

        # Zone A data: both variables stored, connectivity stored (not shared).
        data += _zone_data_block(
            num_vars=2,
            repeat_var_nums=[],
            var_formats=[1, 1],
            var_blocks=[
                b"".join(_f32(v) for v in a_x),
                b"".join(_f32(v) for v in a_y),
            ],
        )
        data += _i(0)  # RepeatConnectivity=0: a connectivity block follows
        for idx in connectivity:
            data += _i(idx)

        # Zone B data: variable 0 (X) repeated (shared), variable 1 (Y) its
        # own; connectivity repeated ("1" = same as the previous zone).
        data += _f32(299.0)
        data += _i(1) + _i(1)  # NumRepeatVars=1, repeated var number 1 (X, 1-based)
        data += _i(1) + _i(1)  # both vars stored as float32 on disk
        b_y = [100.0, 200.0, 300.0, 400.0]
        data += b"".join(_f32(v) for v in b_y)  # only Y's block (X is repeated)
        data += _i(1)  # RepeatConnectivity=1: shared from the previous zone

        path = _write(tmp_path, "sharing.bin", data)

        with tecio.open(str(path), "r") as r:
            assert len(r.zones) == 2
            zone_a, zone_b = r.zones[0], r.zones[1]

            assert zone_b.variables["X"].shared_zone == 0
            x_a = zone_a.variables["X"].get_values()
            x_b = zone_b.variables["X"].get_values()
            y_b = zone_b.variables["Y"].get_values()
            assert x_a is not None
            assert x_b is not None
            assert y_b is not None
            np.testing.assert_allclose(x_b, x_a)
            np.testing.assert_allclose(x_b, a_x)
            np.testing.assert_allclose(y_b, b_y)

            assert zone_b.shared_connectivity == 0
            node_map_a = zone_a.node_map
            node_map_b = zone_b.node_map
            assert node_map_a is not None
            assert node_map_b is not None
            np.testing.assert_array_equal(node_map_b, node_map_a)
            np.testing.assert_array_equal(node_map_a, [[c - 1 for c in connectivity]])


# ======================================================================================
# FE element types (v75 has four KMax codes; others are degenerate bricks)
# ======================================================================================

# One hex, pyramid, prism and tetrahedron, each stored as an 8-node brick with
# repeated nodes (the Tecplot convention, see the Tecplot Data Format Guide). The
# reader must return them verbatim: 0-based, 8 columns, no de-duplication.
_DEGENERATE_BRICK_NODES = [
    # hex: unit cube
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    # pyramid: apex 9, square base 10..13
    (3.5, 0.5, 0.0),
    (3.0, 0.0, 1.0), (4.0, 0.0, 1.0), (4.0, 1.0, 1.0), (3.0, 1.0, 1.0),
    # prism: triangles 14..16 and 17..19
    (6.0, 0.0, 0.0), (7.0, 0.0, 0.0), (6.5, 1.0, 0.0),
    (6.0, 0.0, 1.0), (7.0, 0.0, 1.0), (6.5, 1.0, 1.0),
    # tetrahedron: triangle 20..22, apex 23
    (9.0, 0.0, 0.0), (10.0, 0.0, 0.0), (9.5, 1.0, 0.0), (9.5, 0.25, 1.0),
]  # fmt: skip
_DEGENERATE_BRICK_CELLS = [
    [1, 2, 3, 4, 5, 6, 7, 8],  # hex
    [9, 9, 9, 9, 10, 11, 12, 13],  # pyramid (the Data Format Guide's example)
    [14, 15, 16, 16, 17, 18, 19, 19],  # prism
    [20, 21, 22, 22, 23, 23, 23, 23],  # tetrahedron
]


def degenerate_brick_file(*, point: bool, double: bool) -> bytes:
    """A one-zone FEBRICK file holding the four cells above (X, Y, Z, P)."""
    rows = [[*xyz, float(i + 1)] for i, xyz in enumerate(_DEGENERATE_BRICK_NODES)]
    header = _zone_header(
        "Mixed", fmt=3 if point else 2, imax=len(rows), jmax=4, kmax=3
    )
    return _single_zone_file(
        ["X", "Y", "Z", "P"],
        header,
        rows,
        _DEGENERATE_BRICK_CELLS,
        point=point,
        double=double,
    )


class TestDegenerateBricks:
    """Pyramids, prisms and tetrahedra stored as repeated-node FEBRICK cells."""

    @pytest.mark.parametrize("point", [False, True], ids=["block", "point"])
    @pytest.mark.parametrize("double", [False, True], ids=["float", "double"])
    def test_cells_read_back_verbatim(
        self, tmp_path: Path, point: bool, double: bool
    ) -> None:
        path = _write(
            tmp_path, "bricks.bin", degenerate_brick_file(point=point, double=double)
        )
        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.zone_type == ZoneType.FEBRICK
            assert zone.datapacking == (
                DataPacking.POINT if point else DataPacking.BLOCK
            )
            assert zone.num_nodes == 23
            assert zone.num_elements == 4
            assert zone.node_map is not None
            np.testing.assert_array_equal(
                zone.node_map, np.array(_DEGENERATE_BRICK_CELLS) - 1
            )
            expected_type = DataType.DOUBLE if double else DataType.FLOAT
            for col, name in enumerate(["X", "Y", "Z"]):
                var = zone.variables[name]
                assert var.data_type == expected_type
                np.testing.assert_array_equal(
                    var.get_values(), [n[col] for n in _DEGENERATE_BRICK_NODES]
                )
            np.testing.assert_array_equal(
                zone.variables["P"].get_values(), np.arange(1, 24)
            )


# A Z-shaped line chain (elements deliberately not in node order) followed by a
# triangle zone, so that a wrong node count per line segment would misalign the
# second zone.
_CHAIN_ROWS = [[0.0, 0.0, 10.0], [1.0, 1.0, 20.0], [2.0, 0.0, 30.0], [3.0, 1.0, 40.0]]
_CHAIN_CELLS = [[1, 3], [3, 2], [2, 4]]
_TRI_ROWS = [[0.0, 2.0, 50.0], [1.0, 2.0, 60.0], [0.0, 3.0, 70.0]]


def lineseg_then_triangle_file(*, point: bool, double: bool) -> bytes:
    """Zone 1: FELINESEG chain (KMax 4). Zone 2: one FETRIANGLE."""
    fmt = 3 if point else 2
    data = _header_prelude("Lines", ["X", "Y", "T"])
    data += _zone_header("Chain", fmt=fmt, imax=4, jmax=3, kmax=4)
    data += _zone_header("Tri", fmt=fmt, imax=3, jmax=1, kmax=0)
    data += _f32(357.0)
    data += _zone_data_record(3, _CHAIN_ROWS, _CHAIN_CELLS, point=point, double=double)
    data += _zone_data_record(3, _TRI_ROWS, [[1, 2, 3]], point=point, double=double)
    return data


class TestLineSegments:
    """KMax 4 is FELINESEG (2 nodes per element): not in the spec, GUI-confirmed."""

    @pytest.mark.parametrize("point", [False, True], ids=["block", "point"])
    @pytest.mark.parametrize("double", [False, True], ids=["float", "double"])
    def test_line_chain_then_triangle(
        self, tmp_path: Path, point: bool, double: bool
    ) -> None:
        path = _write(
            tmp_path,
            "lines.bin",
            lineseg_then_triangle_file(point=point, double=double),
        )
        with tecio.open(str(path), "r") as r:
            chain, tri = r.zones[0], r.zones[1]
            assert chain.zone_type == ZoneType.FELINESEG
            assert chain.num_nodes == 4
            assert chain.num_elements == 3
            np.testing.assert_array_equal(chain.node_map, np.array(_CHAIN_CELLS) - 1)
            np.testing.assert_array_equal(
                chain.variables["T"].get_values(), [10, 20, 30, 40]
            )
            # The zone after the chain is only readable if 2 nodes per segment
            # was consumed from the chain's connectivity block.
            assert tri.zone_type == ZoneType.FETRIANGLE
            np.testing.assert_array_equal(tri.node_map, [[0, 1, 2]])
            np.testing.assert_array_equal(tri.variables["T"].get_values(), [50, 60, 70])

    @pytest.mark.parametrize("kmax", [5, 6])
    def test_pyramid_and_prism_codes_are_rejected(
        self, tmp_path: Path, kmax: int
    ) -> None:
        """Tecplot rejects KMax 5 and 6 ("Invalid element type for FE DataSet")."""
        header = _zone_header("Z", fmt=2, imax=5, jmax=1, kmax=kmax)
        data = _header_prelude("Test", ["X"]) + header + _f32(357.0)
        path = _write(tmp_path, f"k{kmax}.bin", data)
        with pytest.raises(PltReadError, match="stored as bricks"):
            tecio.open(str(path), "r")


class TestOtherV75ElementTypes:
    """Native triangle (KMax 0) and tetrahedron (KMax 2) codes, and ORDERED POINT."""

    @pytest.mark.parametrize("point", [False, True], ids=["block", "point"])
    def test_triangle_zone(self, tmp_path: Path, point: bool) -> None:
        rows = [[0.0, 0.0, 10.0], [1.0, 0.0, 20.0], [0.0, 1.0, 30.0], [1.0, 1.0, 40.0]]
        conn = [[1, 2, 3], [2, 4, 3]]
        header = _zone_header("Tri", fmt=3 if point else 2, imax=4, jmax=2, kmax=0)
        data = _single_zone_file(
            ["X", "Y", "T"], header, rows, conn, point=point, double=False
        )
        path = _write(tmp_path, "tri.bin", data)
        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.zone_type == ZoneType.FETRIANGLE
            np.testing.assert_array_equal(zone.node_map, np.array(conn) - 1)
            np.testing.assert_array_equal(
                zone.variables["T"].get_values(), [10, 20, 30, 40]
            )

    @pytest.mark.parametrize("point", [False, True], ids=["block", "point"])
    def test_tetrahedron_zone(self, tmp_path: Path, point: bool) -> None:
        rows = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        conn = [[1, 2, 3, 4]]
        header = _zone_header("Tet", fmt=3 if point else 2, imax=4, jmax=1, kmax=2)
        data = _single_zone_file(
            ["X", "Y", "Z"], header, rows, conn, point=point, double=True
        )
        path = _write(tmp_path, "tet.bin", data)
        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.zone_type == ZoneType.FETETRAHEDRON
            np.testing.assert_array_equal(zone.node_map, np.array(conn) - 1)
            np.testing.assert_array_equal(
                zone.variables["Z"].get_values(), [0, 0, 0, 1]
            )

    def test_ordered_point_zone(self, tmp_path: Path) -> None:
        rows = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
        header = _zone_header("Grid", fmt=1, imax=2, jmax=2, kmax=1)  # ORDERED, POINT
        data = _single_zone_file(
            ["X", "Y"], header, rows, None, point=True, double=False
        )
        path = _write(tmp_path, "ordered_point.bin", data)
        with tecio.open(str(path), "r") as r:
            zone = r.zones[0]
            assert zone.zone_type == ZoneType.ORDERED
            assert zone.datapacking == DataPacking.POINT
            x = zone.variables["X"].get_values()
            y = zone.variables["Y"].get_values()
            assert x is not None
            assert y is not None
            np.testing.assert_array_equal(x.ravel(order="F"), [1, 2, 3, 4])
            np.testing.assert_array_equal(y.ravel(order="F"), [10, 20, 30, 40])


# ======================================================================================
# Static grid shared down a chain of zones (the common time-series layout)
# ======================================================================================

_GRID_XY = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]
_GRID_CELLS = [[1, 2, 5, 4], [2, 3, 6, 5]]
_GRID_ZONES = 4


def static_grid_file(
    var_order: list[str], *, point: bool, share_connectivity: bool
) -> bytes:
    """Four quad zones: X and Y written once, repeated from the previous zone after.

    T is different in every zone: ``100 * zone + node`` (both 1-based). Zones 3
    and 4 repeat variables from a zone that itself repeats them (a chain).
    """
    fmt = 3 if point else 2
    data = _header_prelude("Static grid", var_order)
    for z in range(1, _GRID_ZONES + 1):
        data += _zone_header(f"Step{z}", fmt=fmt, imax=6, jmax=2, kmax=1)
    data += _f32(357.0)
    shared = tuple(i + 1 for i, name in enumerate(var_order) if name in ("X", "Y"))
    for z in range(1, _GRID_ZONES + 1):
        by_name = [
            {"X": x, "Y": y, "T": float(100 * z + n + 1)}
            for n, (x, y) in enumerate(_GRID_XY)
        ]
        rows = [[node[name] for name in var_order] for node in by_name]
        data += _zone_data_record(
            len(var_order),
            rows,
            _GRID_CELLS,
            point=point,
            double=False,
            repeat=shared if z > 1 else (),
            repeat_connectivity=share_connectivity and z > 1,
        )
    return data


class TestStaticGridChain:
    """Grid variables repeated down 4 zones: zone 4 repeats from 3, from 2, from 1."""

    @pytest.mark.parametrize(
        ("var_order", "point", "share_connectivity"),
        [
            (["X", "Y", "T"], False, False),
            (["X", "T", "Y"], True, False),
            (["X", "Y", "T"], True, True),
        ],
        ids=["block", "point-repeat-in-middle", "point-shared-connectivity"],
    )
    def test_chain_resolves_to_first_zone(
        self,
        tmp_path: Path,
        var_order: list[str],
        point: bool,
        share_connectivity: bool,
    ) -> None:
        data = static_grid_file(
            var_order, point=point, share_connectivity=share_connectivity
        )
        path = _write(tmp_path, "static_grid.bin", data)
        with tecio.open(str(path), "r") as r:
            assert len(r.zones) == _GRID_ZONES
            for i, zone in enumerate(r.zones):
                for col, name in enumerate(["X", "Y"]):
                    var = zone.variables[name]
                    np.testing.assert_array_equal(
                        var.get_values(), [xy[col] for xy in _GRID_XY]
                    )
                    # 0-based index of the previous zone; None in the first zone.
                    assert var.shared_zone == (i - 1 if i else None)
                t = zone.variables["T"]
                assert t.shared_zone is None
                np.testing.assert_array_equal(
                    t.get_values(), [100 * (i + 1) + n for n in range(1, 7)]
                )
                np.testing.assert_array_equal(zone.node_map, np.array(_GRID_CELLS) - 1)
                expected = i - 1 if (i and share_connectivity) else None
                assert zone.shared_connectivity == expected


# ======================================================================================
# Entry point
# ======================================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
