r"""Read the legacy Tecplot PLT v7.5 binary format (magic ``#!TDV75 ``).

v7.5 is a legacy Tecplot binary format that is still used by a handful of CFD solvers
(eg CFD++). It has no StrandID, SolutionTime, value location (cell-centered data),
face-neighbor, or aux-data support at all. Its zone
header uses a single "Format" field that packs together what later versions split into
two independent things: POINT vs. BLOCK data packing, and ORDERED vs. FE zone
topology. For FE zones, the element type (triangle/quad/tet/brick) is packed into the
zone's KMax field.

This module implements v7.5 header and data-section parsing against an archived data
formatting guide provided by Tecplot support, and tested against v7.5 solution files
written by CFD++ where the data arrays coiuld be cross-checked against the same dataset
re-exported through a v11.2 PLT writer using the Tecplot GUI:

Notes:
    * **Format field.** ``0`` = ORDERED, BLOCK packing. ``1`` = ORDERED, POINT
      packing. ``2`` = FE, BLOCK packing (FEBLOCK). ``3`` = FE, POINT packing
      (FEPOINT). POINT packing means all variables for node 0, then node 1, etc. See
      ``_ZoneMeta.is_point_format`` and
      ``TecplotPltVariableReader._get_point_format_values`` in :mod:`._plt_read`.
    * **KMax as element type.** For FE zones (Format 2 or 3), IMax = number of points,
      JMax = number of elements, and KMax is the element type: ``0`` = triangles, ``1``
      = quadrilaterals, ``2`` = tetrahedrons, ``3`` = bricks. For ORDERED zones (Format
      0 or 1), IMax/JMax/KMax are the literal ordered dimensions (same as later
      versions).
    * **Connectivity is 1-based.**
    * **Variable sharing ("repeat vars").** A zone header may declare that some
      of its variables are "repeated" rather than stored. I do not have a v7.5 sample that
      exercises this, so two details are inferred rather than confirmed: that the
      repeated variable numbers are 0-based, and that the source is always the
      immediately preceding zone (by analogy with the connectivity "repeat adjacency
      list from previous zone" flag below, which the spec does state explicitly). Treat
      this path as unverified against real bytes.
    * **Connectivity sharing.** A zone's data section carries an explicit INT32 boolean,
      "repeat adjacency list from previous zone": 1 means this zone's connectivity is
      identical to the previous zone's and nothing further is stored on disk; 0 means an
      ordinary INT32 * L * JMax connectivity block follows, L given by element type
      (3/4/4/8 for triangle/quad/tet/brick).

Not supported (no such field exists in v7.5):
    * Passive variables.
    * Cell-centered data.
    * StrandID / SolutionTime.
    * Face-neighbor connections of any kind.
    * Dataset, zone, or variable auxiliary data.

Format reference:
    A Tecplot provided binary format description for v7.5, plus the cross checks
    described above.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from ._constants import ValueLocation, ZoneType
from ._plt_read import (
    _DATATYPE_TO_NUMPY,
    _INT_TO_DATATYPE,
    _NODES_PER_ELEMENT,
    PltReadError,
    _read_float32,
    _read_int32,
    _read_string,
    _skip_string,
    _ZoneMeta,
)

if TYPE_CHECKING:
    import io

    from ._plt_read import _PltParser

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

_MARKER_ZONE: float = 299.0
_MARKER_GEOM: float = 399.0
_MARKER_TEXT: float = 499.0
_MARKER_CUSTOM_LABEL: float = 599.0
_MARKER_EOHMARKER: float = 357.0

# Format field: (is_fe, is_point_packing).
_FORMAT_IS_FE: dict[int, bool] = {0: False, 1: False, 2: True, 3: True}
_FORMAT_IS_POINT: dict[int, bool] = {0: False, 1: True, 2: False, 3: True}

# KMax element type, FE zones only.
_ELEMENT_TYPE: dict[int, ZoneType] = {
    0: ZoneType.FETRIANGLE,
    1: ZoneType.FEQUADRILATERAL,
    2: ZoneType.FETETRAHEDRON,
    3: ZoneType.FEBRICK,
    4: ZoneType.FELINESEG,
}


# --------------------------------------------------------------------------------------
# Header section
# --------------------------------------------------------------------------------------


def parse_v75_header(parser: _PltParser, fp: io.BufferedIOBase) -> None:
    """Parse a v75 header (magic and endian probe already consumed).

    Populates ``parser.title``, ``parser.var_names``, and ``parser.zones``
    directly, matching :meth:`_PltParser._parse_header`'s contract for every
    other version.

    Raises:
        PltReadError: On an unrecognised header marker. v75 has no aux-data or
            UserRec markers (they either postdate v75 or the format
            description gives no discoverable marker for them), so anything
            other than Zone/Geometry/Text/CustomLabel/EOHMARKER is treated as
            a malformed file rather than silently skipped.
    """
    byte_order = parser.byte_order

    parser.title = _read_string(fp, byte_order)
    num_vars = _read_int32(fp, byte_order)
    parser.var_names = [_read_string(fp, byte_order) for _ in range(num_vars)]

    while True:
        marker = _read_float32(fp, byte_order)

        if abs(marker - _MARKER_EOHMARKER) < 0.5:
            return
        elif abs(marker - _MARKER_ZONE) < 0.5:
            _parse_v75_zone_header(parser, fp)
        elif abs(marker - _MARKER_GEOM) < 0.5:  # pragma: no cover
            # No real or hand-authored v75 sample has a geometry record;
            # see _skip_v75_geometry's own docstring.
            _skip_v75_geometry(fp, byte_order)
        elif abs(marker - _MARKER_TEXT) < 0.5:  # pragma: no cover
            # No real or hand-authored v75 sample has a text record; see
            # _skip_v75_text's own docstring.
            _skip_v75_text(fp, byte_order)
        elif abs(marker - _MARKER_CUSTOM_LABEL) < 0.5:
            parser._skip_custom_label(fp)
        else:
            raise PltReadError(
                f"Unexpected v75 header marker {marker:.1f} at file offset "
                f"{fp.tell() - 4:#010x}."
            )


def _parse_v75_zone_header(parser: _PltParser, fp: io.BufferedIOBase) -> None:
    """Parse one v75 zone header record and append it to ``parser.zones``.

    Raises:
        PltReadError: If the data format or the FE element type is not recognised.
    """
    byte_order = parser.byte_order
    meta = _ZoneMeta()

    meta.title = _read_string(fp, byte_order)
    fmt = _read_int32(fp, byte_order)
    _read_int32(fp, byte_order)  # Zone Color, unused (-1 = let Tecplot decide)
    i_dim = _read_int32(fp, byte_order)
    j_dim = _read_int32(fp, byte_order)
    k_dim = _read_int32(fp, byte_order)

    is_fe = _FORMAT_IS_FE.get(fmt)
    if is_fe is None:
        raise PltReadError(
            f"Zone {meta.title!r}: unrecognised v75 Format value {fmt} "
            "(expected 0=BLOCK, 1=POINT, 2=FEBLOCK, or 3=FEPOINT)."
        )
    meta.is_point_format = _FORMAT_IS_POINT[fmt]

    if is_fe:
        element_type = _ELEMENT_TYPE.get(k_dim)
        if element_type is None:
            raise PltReadError(
                f"Zone {meta.title!r}: unrecognised v75 FE element type {k_dim} "
                "(expected 0=triangles, 1=quadrilaterals, 2=tetrahedrons, "
                "3=bricks, or 4=line segments). Pyramids and prisms are stored "
                "as bricks with repeated nodes."
            )
        meta.zone_type = element_type
        meta.num_nodes = i_dim
        meta.num_elements = j_dim
        meta.connectivity_is_one_based = True
    else:
        meta.zone_type = ZoneType.ORDERED
        meta.i_max, meta.j_max, meta.k_max = i_dim, j_dim, k_dim

    num_vars = len(parser.var_names)
    meta.value_locations = [ValueLocation.NODAL] * num_vars

    parser.zones.append(meta)


def _skip_v75_geometry(
    fp: io.BufferedIOBase, byte_order: str
) -> None:  # pragma: no cover
    """Skip a v75 geometry record (marker already consumed).

    Unlike the current-format geometry record, v75 has no DrawOrder field
    (right after Scope) or Clipping field (at the very end); both are absent
    from the format description. No real v75 sample exercises this record;
    implemented directly from the format description. Not hand-authored
    either: unlike the ORDERED/BLOCK/sharing cases in
    ``tests/test_read_legacy_synthetic.py``, a geometry record has no
    bearing on reading array data, so the cost of hand-building one
    correctly isn't worth it for a record this module only has to skip
    over, never interpret.
    """
    _read_int32(fp, byte_order)  # CoordSys
    _read_int32(fp, byte_order)  # Scope
    fp.read(24)  # X, Y, Z starting location (3 x FLOAT64)
    _read_int32(fp, byte_order)  # Zone (0 = all)
    _read_int32(fp, byte_order)  # Color
    _read_int32(fp, byte_order)  # FillColor
    _read_int32(fp, byte_order)  # IsFilled
    geom_type = _read_int32(fp, byte_order)
    _read_int32(fp, byte_order)  # LinePattern
    fp.read(16)  # PatternLength + LineThickness (2 x FLOAT64)
    _read_int32(fp, byte_order)  # NumEllipsePts
    _read_int32(fp, byte_order)  # ArrowheadStyle
    _read_int32(fp, byte_order)  # ArrowheadAttachment
    fp.read(16)  # ArrowheadSize + ArrowheadAngle (2 x FLOAT64)
    _skip_string(fp, byte_order)  # MacroFunctionCommand
    gtype = _read_int32(fp, byte_order)  # PolylineFieldDataType (1=float, 2=double)
    gtype_bytes = 4 if gtype == 1 else 8

    if geom_type in (0, 4):  # Line, 3DLine
        num_blocks = 3 if geom_type == 4 else 2  # 3DLine also has a Z-block
        num_polylines = _read_int32(fp, byte_order)
        for _ in range(num_polylines):
            num_pts = _read_int32(fp, byte_order)
            fp.read(num_pts * gtype_bytes * num_blocks)
    elif geom_type == 1:  # Rectangle
        fp.read(16)  # X, Y offset for far corner (2 x FLOAT64, always double)
    elif geom_type == 2:  # Circle
        fp.read(8)  # Radius (FLOAT64)
    elif geom_type == 3:  # Ellipse
        fp.read(16)  # X, Y radii (2 x FLOAT64)


def _skip_v75_text(fp: io.BufferedIOBase, byte_order: str) -> None:  # pragma: no cover
    """Skip a v75 text record (marker already consumed).

    Unlike the current-format text record, v75's starting location is X, Y
    only (no Z), and there is no Clipping field before the text string. No
    real v75 sample exercises this record; implemented directly from the
    format description. See ``_skip_v75_geometry``'s docstring for why this
    isn't hand-authored either.
    """
    _read_int32(fp, byte_order)  # CoordSys
    _read_int32(fp, byte_order)  # Scope
    fp.read(16)  # X, Y starting location (2 x FLOAT64)
    _read_int32(fp, byte_order)  # FontType
    _read_int32(fp, byte_order)  # CharHeightUnits
    fp.read(8)  # Height (FLOAT64)
    _read_int32(fp, byte_order)  # TextBoxType
    fp.read(16)  # Margin + MarginLinewidth (2 x FLOAT64)
    _read_int32(fp, byte_order)  # TextBoxOutlineColor
    _read_int32(fp, byte_order)  # TextBoxFillColor
    fp.read(16)  # Angle + LineSpacing (2 x FLOAT64)
    _read_int32(fp, byte_order)  # TextAnchor
    _read_int32(fp, byte_order)  # Zone (0 = all)
    _read_int32(fp, byte_order)  # Color
    _skip_string(fp, byte_order)  # MacroFunctionCommand
    _skip_string(fp, byte_order)  # The text itself


# --------------------------------------------------------------------------------------
# Data section
# --------------------------------------------------------------------------------------


def parse_v75_data_section(parser: _PltParser, fp: io.BufferedIOBase) -> None:
    """Parse the v75 data section (EOHMARKER already consumed).

    Iterates zones in declaration order, matching :meth:`_PltParser`'s
    contract for every other version.

    Raises:
        PltReadError: If a zone marker is wrong or a repeated variable reference is
            invalid.
    """
    byte_order = parser.byte_order
    num_vars = len(parser.var_names)

    for zone_idx, meta in enumerate(parser.zones):
        marker = _read_float32(fp, byte_order)
        if abs(marker - _MARKER_ZONE) >= 0.5:
            raise PltReadError(
                f"v75 data section zone {zone_idx}: expected zone marker 299.0, "
                f"got {marker:.2f}."
            )

        n_repeat = _read_int32(fp, byte_order)
        # See module docstring: numbers are 1-based, source is the previous zone.
        repeat_var_nums = [_read_int32(fp, byte_order) for _ in range(n_repeat)]
        raw_dtypes = [_read_int32(fp, byte_order) for _ in range(num_vars)]
        meta.var_data_types = [_INT_TO_DATATYPE[d] for d in raw_dtypes]
        meta.is_passive = [False] * num_vars
        meta.shared_zone = [-1] * num_vars
        if repeat_var_nums and zone_idx == 0:
            raise PltReadError(
                "v75 data section zone 0: repeated variables declared, but there "
                "is no previous zone to repeat from."
            )
        for var_num in repeat_var_nums:
            if not 1 <= var_num <= num_vars:
                raise PltReadError(
                    f"v75 data section zone {zone_idx}: repeated variable number "
                    f"{var_num} out of range 1..{num_vars}."
                )
            meta.shared_zone[var_num - 1] = zone_idx - 1

        active_vars = [v for v in range(num_vars) if meta.shared_zone[v] < 0]

        if meta.zone_type == ZoneType.ORDERED:
            num_rows = max(meta.i_max, 1) * max(meta.j_max, 1) * max(meta.k_max, 1)
        else:
            num_rows = meta.num_nodes

        if meta.is_point_format:
            _record_point_format_offsets(fp, meta, active_vars, num_rows, byte_order)
        else:
            _record_block_format_offsets(fp, meta, active_vars, num_rows)

        if meta.zone_type != ZoneType.ORDERED:
            _record_v75_connectivity(fp, meta, byte_order, zone_idx)


def _record_point_format_offsets(
    fp: io.BufferedIOBase,
    meta: _ZoneMeta,
    active_vars: list[int],
    num_rows: int,
    byte_order: str,
) -> None:
    """Record the interleaved row layout for a POINT/FEPOINT zone.

    Builds the structured dtype covering every active (non-repeated)
    variable, in file order, with byte order baked into each field so reads
    never need a separate byte-swap pass. Does not read the row block itself,
    :class:`~._plt_read.TecplotPltVariableReader` reads it lazily (and caches
    the whole block on first full read, see its ``_get_point_format_values``).
    """
    fields: list[tuple[str, str]] = []
    field_by_var: dict[int, str] = {}
    for var_idx in active_vars:
        dtype_str, itemsize = _DATATYPE_TO_NUMPY[meta.var_data_types[var_idx].value]
        field_name = f"v{var_idx}"
        fields.append((field_name, f"{byte_order}{dtype_str}"))
        field_by_var[var_idx] = field_name

    row_dtype = np.dtype(fields)
    meta.point_row_dtype = row_dtype
    meta.point_data_offset = fp.tell()
    meta.point_num_rows = num_rows
    meta.point_field_by_var = field_by_var

    for var_idx in active_vars:
        dtype_str, _ = _DATATYPE_TO_NUMPY[meta.var_data_types[var_idx].value]
        meta.var_offsets[var_idx] = (meta.point_data_offset, num_rows, dtype_str)

    fp.seek(row_dtype.itemsize * num_rows, os.SEEK_CUR)


def _record_block_format_offsets(
    fp: io.BufferedIOBase,
    meta: _ZoneMeta,
    active_vars: list[int],
    num_rows: int,
) -> None:
    """Record contiguous per-variable block offsets for a BLOCK/FEBLOCK zone.

    Same on-disk layout as every later version, seeks past each variable's
    block without reading it.
    """
    for var_idx in active_vars:
        dtype_str, itemsize = _DATATYPE_TO_NUMPY[meta.var_data_types[var_idx].value]
        meta.var_offsets[var_idx] = (fp.tell(), num_rows, dtype_str)
        fp.seek(num_rows * itemsize, os.SEEK_CUR)


def _record_v75_connectivity(
    fp: io.BufferedIOBase,
    meta: _ZoneMeta,
    byte_order: str,
    zone_idx: int,
) -> None:
    """Record the connectivity block (or shared-from-previous-zone marker).

    The "repeat adjacency list from previous zone" boolean is read
    unconditionally for every FE zone; connectivity data only follows when
    it's 0.
    """
    repeat_flag = _read_int32(fp, byte_order)
    if repeat_flag:
        meta.connectivity_shared_zone = zone_idx - 1
        return

    nodes_per_cell = _NODES_PER_ELEMENT[meta.zone_type]
    count = meta.num_elements * nodes_per_cell
    meta.connectivity_offset = fp.tell()
    meta.connectivity_count = count
    fp.seek(count * 4, os.SEEK_CUR)
