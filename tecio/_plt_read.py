r"""Read Tecplot PLT (``.plt``) binary files.

A pure-Python / NumPy reader for Tecplot PLT binary files, no C library involved.
Format versions v112 and v191 (current) are read natively by this module; legacy
versions v75, v107, and v111 are also supported, read-only (see :mod:`._plt_legacy`
for v75; v107/v111 are handled here as small deltas on the v112 parser).

Notes:
    * **No byte-swapping boilerplate.** The INT32 value of ``1`` written immediately
      after the magic number is used to detect endianness at open time. All subsequent
      reads use the correct NumPy byte-order prefix (``"<"`` or ``">"``).
    * **Eager metadata, lazy data.** Zone and variable metadata is parsed during
      ``TecplotPltReader.__init__``. Numeric data and node maps are read from disk only
      on first access, matching SZL's laziness even though PLT has no C library or file
      handle behind it.
    * **No text / geometry support.** Header records with markers 399.0 (geometry) and
      499.0 (text) are detected and skipped without parsing their contents.
    * **v112 and v191 zone headers.** Zone marker ``299.0`` -> v112 header; ``298.0`` ->
      v191 header. Both are fully supported.
    * **Legacy versions (read-only).** v75 (magic ``#!TDV75 ``) does not have fields for
      StrandID, SolutionTime, value location, face-neighbor, or aux-data. v107
      (``#!TDV107``) and v111 (``#!TDV111``) are structurally v112 with two deltas each:
      v107 has no file-level FileType field and no per-zone "face neighbors complete"
      flag; v111 has one extra INT32 value right after the zone type (the data-packing
      field, 0 = block) of unidentified purpose. Legacy files are read-only;
      :mod:`._plt_write` only ever writes the current format.

Limitations:
    * FEPOLYGON and FEPOLYHEDRON zones are read as
      :class:`~tecio._reader.TecplotPolyZoneReader` (v111 and v112). The v191 face-map
      layout is untested, and v107 files cannot contain poly zones.
    * Bit-packed data (DataType 6) is not supported and raises NotImplementedError.
    * Legacy files have no passive-variable.

Format reference:
    Tecplot 360 Data Format Guide, Binary PLT v112 / v191, for the current format.  An
    archived data format guide for v75 (see :mod:`._plt_legacy`).  v107/v111 deltas were
    derived empirically from real files written by Tecplot's 2006/2008 writers,
    cross-checked byte-for-byte against the same dataset written by later writers.
"""

from __future__ import annotations

import io
import os
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from ._constants import (
    DataPacking,
    DataType,
    FaceNeighborMode,
    FileType,
    ValueLocation,
    ZoneType,
)
from ._containers import ZoneList
from ._reader import (
    PolyArrayKey,
    TecplotAuxDataReader,
    TecplotFEZoneReader,
    TecplotOrderedZoneReader,
    TecplotPolyZoneReader,
    TecplotReader,
    TecplotVariableReader,
    TecplotZoneReader,
)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

# Magic strings for supported file versions.
_MAGIC_V75: bytes = b"#!TDV75 "
_MAGIC_V107: bytes = b"#!TDV107"
_MAGIC_V111: bytes = b"#!TDV111"
_MAGIC_V112: bytes = b"#!TDV112"
_MAGIC_V191: bytes = b"#!TDV191"
_LEGACY_MAGICS: frozenset[bytes] = frozenset({_MAGIC_V75, _MAGIC_V107, _MAGIC_V111})

# Floating-point sentinel values used in the header.
_MARKER_ZONE: float = 299.0  # v112 zone header
_MARKER_ZONE_V191: float = 298.0  # v191 zone header
_MARKER_GEOM: float = 399.0
_MARKER_TEXT: float = 499.0
_MARKER_CUSTOM_LABEL: float = 599.0
_MARKER_USER_REC: float = 699.0
_MARKER_DATASET_AUX: float = 799.0
_MARKER_VAR_AUX: float = 899.0
_MARKER_EOHMARKER: float = 357.0

# Nodes per element for simple FE zone types.
_NODES_PER_ELEMENT: dict[ZoneType, int] = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FETETRAHEDRON: 4,
    ZoneType.FEBRICK: 8,
}

_POLY_TYPES: frozenset[ZoneType] = frozenset({
    ZoneType.FEPOLYGON,
    ZoneType.FEPOLYHEDRON,
})

# Faces per element
_FACES_PER_ELEMENT: dict[ZoneType, int] = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FETETRAHEDRON: 4,
    ZoneType.FEBRICK: 6,
}

# Map from PLT DataType integer to (numpy dtype string, itemsize).
_DATATYPE_TO_NUMPY: dict[int, tuple[str, int]] = {
    1: ("f4", 4),  # FLOAT
    2: ("f8", 8),  # DOUBLE
    3: ("i4", 4),  # INT32 / LongInt
    4: ("i2", 2),  # INT16 / ShortInt
    5: ("u1", 1),  # BYTE
}

# Map from DataType integer to DataType enum.
_INT_TO_DATATYPE: dict[int, DataType] = {dt.value: dt for dt in DataType}

# Map from DataType enum to the canonical numpy dtype (no endian prefix).
_DATATYPE_DTYPE: dict[DataType, str] = {
    DataType.FLOAT: "f4",
    DataType.DOUBLE: "f8",
    DataType.INT32: "i4",
    DataType.INT16: "i2",
    DataType.BYTE: "u1",
}

# PLT format variable location integers → ValueLocation enum. The PLT spec uses 0 =
# Node, 1 = Cell Centered (format guide §IV.iv), which is the opposite of the
# ValueLocation enum (CELL_CENTERED=0, NODAL=1).
_PLT_VALUELOCATION_MAP: dict[int, ValueLocation] = {
    0: ValueLocation.NODAL,
    1: ValueLocation.CELL_CENTERED,
}


# --------------------------------------------------------------------------------------
# Custom exception
# --------------------------------------------------------------------------------------


class PltReadError(RuntimeError):
    """Raised when the PLT binary cannot be parsed."""


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------


def _read_int32(fp: io.BufferedIOBase, byte_order: str) -> int:
    """Read a single INT32 with the given byte-order prefix.

    Returns:
        int: Value decoded by byte order.

    Raises:
        PltReadError: If the file ends before the value.
    """
    raw = fp.read(4)
    if len(raw) < 4:
        raise PltReadError("Unexpected end of file reading INT32.")
    (v,) = struct.unpack(f"{byte_order}i", raw)
    return int(v)


def _read_int64(fp: io.BufferedIOBase, byte_order: str) -> int:
    """Read a single INT64 with the given byte-order prefix.

    Returns:
        int: Value decoded by byte order.

    Raises:
        PltReadError: If the file ends before the value.
    """
    raw = fp.read(8)
    if len(raw) < 8:
        raise PltReadError("Unexpected end of file reading INT64.")
    (v,) = struct.unpack(f"{byte_order}q", raw)
    return int(v)


def _read_float32(fp: io.BufferedIOBase, byte_order: str) -> float:
    """Read a single FLOAT32 with the given byte-order prefix.

    Returns:
        float: Value decoded by byte order.

    Raises:
        PltReadError: If the file ends before the value.
    """
    raw = fp.read(4)
    if len(raw) < 4:
        raise PltReadError("Unexpected end of file reading FLOAT32.")
    (v,) = struct.unpack(f"{byte_order}f", raw)
    return float(v)


def _read_float64(fp: io.BufferedIOBase, byte_order: str) -> float:
    """Read a single FLOAT64 with the given byte-order prefix.

    Returns:
        float: Value decoded by byte order.

    Raises:
        PltReadError: If the file ends before the value.
    """
    raw = fp.read(8)
    if len(raw) < 8:
        raise PltReadError("Unexpected end of file reading FLOAT64.")
    (v,) = struct.unpack(f"{byte_order}d", raw)
    return float(v)


def _read_string(fp: io.BufferedIOBase, byte_order: str) -> str:
    """Read a null-terminated string stored as INT32 code-points.

    The PLT format stores each character as a 4-byte integer (see Note 1 of the format
    spec). The string is terminated by a zero INT32.

    Returns:
        str: utf-8 decoded string.
    """
    chars: list[str] = []
    while True:
        raw = fp.read(4)
        if len(raw) < 4:
            break
        (code,) = struct.unpack(f"{byte_order}i", raw)
        if code == 0:
            break
        chars.append(chr(code))
    return "".join(chars)


def _skip_string(fp: io.BufferedIOBase, byte_order: str) -> None:
    """Read and discard a null-terminated INT32 string."""
    while True:
        raw = fp.read(4)
        if len(raw) < 4:
            break
        (code,) = struct.unpack(f"{byte_order}i", raw)
        if code == 0:
            break


def _peek_float32(fp: io.BufferedIOBase, byte_order: str) -> float:
    """Peek at the next 4 bytes as a FLOAT32 without advancing the file pointer.

    Returns:
        float: The value of the peeked bytes as float32.
    """
    pos = fp.tell()
    val = _read_float32(fp, byte_order)
    fp.seek(pos)
    return val


def _is_legacy_plt(file_path: str | os.PathLike) -> bool:
    """Return True if the file starts with a v75, v107, or v111 magic string.

    Reads only the first 8 bytes, so it is cheap and never raises on a malformed
    body. A missing or unreadable file raises :exc:`OSError` as usual.
    """
    with open(file_path, "rb") as fp:
        return fp.read(8) in _LEGACY_MAGICS


# --------------------------------------------------------------------------------------
# Internal storage dataclasses (populated during header parse)
# --------------------------------------------------------------------------------------


@dataclass
class _ZoneMeta:
    """All zone-header metadata plus data-section file offsets."""

    # Zone header fields
    title: str = ""
    zone_type: ZoneType = ZoneType.ORDERED
    strand_id: int = -1
    solution_time: float = 0.0

    # Ordered zone dimensions
    i_max: int = 0
    j_max: int = 0
    k_max: int = 0

    # FE zone counts
    num_nodes: int = 0
    num_elements: int = 0

    # Poly zone extras. num_boundary_faces is the public count; the header stores one
    # more when any exist (TecIO reserves entry 0 of the boundary tables), kept as
    # boundary_table_entries since it sizes the on-disk offsets table.
    num_faces: int = 0
    total_face_nodes: int = 0
    num_boundary_faces: int = 0
    boundary_table_entries: int = 0
    num_boundary_connections: int = 0
    # Face-map blocks on disk, {key: (file_offset, count, itemsize)}, empty for zones
    # that share connectivity.
    poly_sections: dict[str, tuple[int, int, int]] = field(default_factory=dict)

    # Variable metadata (per variable)
    var_data_types: list[DataType] = field(default_factory=list)
    is_passive: list[bool] = field(default_factory=list)
    shared_zone: list[int] = field(default_factory=list)  # -1 = not shared
    value_locations: list[ValueLocation] = field(default_factory=list)
    connectivity_shared_zone: int = -1
    has_raw_face_neighbors: bool = False
    # As declared by the header: whether/how many "miscellaneous" (explicit,
    # non-raw) face-neighbor connections this zone's data block has on disk.
    # Only used to decide whether to invoke _record_face_connections, never
    # read as the final result, that's num_face_connections below, since a
    # zone can have has_raw_face_neighbors instead, which doesn't touch this.
    declared_misc_face_connections: int = 0
    # Final, public-facing face-neighbor result, set by whichever mechanism
    # a zone actually uses (_record_face_connections or the _raw variant).
    num_face_connections: int = 0
    face_neighbor_mode: FaceNeighborMode | None = None
    face_connections_offset: int | None = None
    face_connections_flat_count: int = 0  # int32 values to read from that offset
    face_connections_eager: npt.NDArray[np.int64] | None = None  # ragged "many" modes

    # Zone-level aux data  {name: value}
    auxdata: dict[str, str] = field(default_factory=dict)

    # Header version: True → v191, False → v112
    is_v191: bool = False

    # File offsets filled in during data-section parse offsets[var_index] =
    # (file_offset, count, dtype_str)
    var_offsets: dict[int, tuple[int, int, str]] = field(default_factory=dict)

    # File offset of the connectivity block (None = no connectivity)
    connectivity_offset: int | None = None
    connectivity_count: int = 0  # total int32 values

    # -- Legacy (v75) on-disk quirks. False/empty/0 for every non-legacy file. --------
    # v75's Format field packs POINT/BLOCK packing together with ORDERED/FE zone
    # type; True means this zone's variable data is interleaved by node/element
    # rather than stored as one contiguous block per variable. See _plt_legacy.
    is_point_format: bool = False
    # v75 connectivity indices are 1-based on disk; every later version is 0-based on
    # disk, so 1 is subtracted when read. See TecplotPltFEZoneReader._load_node_map.
    connectivity_is_one_based: bool = False
    # Structured dtype for one interleaved row (point_field_by_var maps a 0-based
    # variable index to its field name), plus where the row block starts and how
    # many rows it holds. Only meaningful when is_point_format is True.
    point_row_dtype: np.dtype | None = None
    point_field_by_var: dict[int, str] = field(default_factory=dict)
    point_data_offset: int = 0
    point_num_rows: int = 0
    # Cache of the one-pass full-zone row read, shared by every variable in this
    # zone so reading all of them costs one disk pass rather than one per
    # variable. Populated on first full read; never invalidated, this dataclass
    # (and its cache) live only as long as the TecplotPltReader that owns them.
    point_format_cache: npt.NDArray[Any] | None = None


# --------------------------------------------------------------------------------------
class _PltParser:
    """Low-level PLT binary parser.

    Called once from :class:`Read.__init__`. After construction the following attributes
    are available:

        ===================   ===============================================
        Attribute             Description
        ===================   ===============================================
        ``title``             dataset title string
        ``file_type``         :class:`FileType` enum
        ``var_names``         list of variable name strings
        ``zones``             list of :class:`_ZoneMeta` objects
        ``dataset_auxdata``   ``{name: value}``
        ``var_auxdata``       ``{var_index_0based: {name: value}}``
        ``byte_order``         byte-order prefix for NumPy (``"<"`` or ``">"`)
        ===================   ===============================================
    """

    def __init__(self, file_path: str | os.PathLike) -> None:
        self._path = str(file_path)
        self.title: str = ""
        self.file_type: FileType = FileType.FULL
        self.var_names: list[str] = []
        self.zones: list[_ZoneMeta] = []
        self.dataset_auxdata: dict[str, str] = {}
        self.var_auxdata: dict[int, dict[str, str]] = {}
        self.byte_order: str = "<"  # byte order: "<" little, ">" big
        self._file_version: str = "v112"

        with open(self._path, "rb") as fp:
            self._parse_header(fp)
            self._parse_data_section(fp)

    # -- Header section ----------------------------------------------------------------

    def _parse_header(self, fp: io.BufferedIOBase) -> None:
        """Parse the PLT header section up to and including EOHMARKER.

        Raises:
            PltReadError: If an unexpected header marker is found.
        """
        self._detect_version_and_endian(fp)
        byte_order = self.byte_order

        if self._file_version == "v75":
            # v75's header, zone, and data-section layouts are all different enough to
            # move the data field order to its own module. The other legacy versions are
            # similar enough to v112 to handle with a couple conditional read statements
            # below.
            from ._plt_legacy import parse_v75_header

            parse_v75_header(self, fp)
            return

        # v107 has no file-level FileType field at all. v111 and later all have it.
        if self._file_version != "v107":
            self.file_type = FileType(_read_int32(fp, byte_order))
        else:
            self.file_type = FileType.FULL

        # Dataset title
        self.title = _read_string(fp, byte_order)

        # Variable names
        num_vars = _read_int32(fp, byte_order)
        self.var_names = [_read_string(fp, byte_order) for _ in range(num_vars)]

        # Zone / geometry / text / aux records — loop until EOHMARKER
        while True:
            marker = _read_float32(fp, byte_order)

            if abs(marker - _MARKER_EOHMARKER) < 0.5:
                # End of header
                return

            elif abs(marker - _MARKER_ZONE) < 0.5:
                self._parse_zone_header(fp, v191=False)

            elif abs(marker - _MARKER_ZONE_V191) < 0.5:
                self._parse_zone_header(fp, v191=True)

            elif abs(marker - _MARKER_GEOM) < 0.5:
                self._skip_geometry(fp)

            elif abs(marker - _MARKER_TEXT) < 0.5:
                self._skip_text(fp)

            elif abs(marker - _MARKER_CUSTOM_LABEL) < 0.5:
                self._skip_custom_label(fp)

            elif abs(marker - _MARKER_USER_REC) < 0.5:
                self._skip_user_rec(fp)

            elif abs(marker - _MARKER_DATASET_AUX) < 0.5:
                name = _read_string(fp, byte_order)
                _read_int32(fp, byte_order)  # value format (always 0 = string)
                value = _read_string(fp, byte_order)
                self.dataset_auxdata[name] = value

            elif abs(marker - _MARKER_VAR_AUX) < 0.5:
                var_num = _read_int32(fp, byte_order)  # 0-based variable number
                name = _read_string(fp, byte_order)
                _read_int32(fp, byte_order)  # value format
                value = _read_string(fp, byte_order)
                self.var_auxdata.setdefault(var_num, {})[name] = value

            else:
                # Unknown marker — should not happen in a well-formed file.
                raise PltReadError(
                    f"Unexpected header marker {marker:.1f} at file offset "
                    f"{fp.tell() - 4:#010x}."
                )

    def _detect_version_and_endian(self, fp: io.BufferedIOBase) -> None:
        """Read magic bytes and INT32=1 endian probe; set ``self.byte_order``.

        Raises:
            PltReadError: If the magic number is unrecognised or the byte-order probe is
                invalid.
        """
        magic = fp.read(8)
        if magic == _MAGIC_V75:
            self._file_version = "v75"
        elif magic == _MAGIC_V107:
            self._file_version = "v107"
        elif magic == _MAGIC_V111:
            self._file_version = "v111"
        elif magic == _MAGIC_V112:
            self._file_version = "v112"
        elif magic == _MAGIC_V191:
            self._file_version = "v191"
        else:
            raise PltReadError(
                f"Unrecognised PLT magic number: {magic!r}.  "
                "Expected one of b'#!TDV75 ', b'#!TDV107', b'#!TDV111', "
                "b'#!TDV112', b'#!TDV191'."
            )

        # Next 4 bytes encode the integer 1 in the writer's byte order.
        raw = fp.read(4)
        if len(raw) < 4:
            raise PltReadError("File too short — cannot read byte-order probe.")

        # Try little-endian first
        (le_val,) = struct.unpack("<i", raw)
        if le_val == 1:
            self.byte_order = "<"
            return

        (be_val,) = struct.unpack(">i", raw)
        if be_val == 1:
            self.byte_order = ">"
            return

        raise PltReadError(
            f"Byte-order probe value {raw!r} is neither 1 in little- nor big-endian."
        )

    def _parse_zone_header(self, fp: io.BufferedIOBase, *, v191: bool) -> None:
        """Parse a single zone header record and append to ``self.zones``.

        Raises:
            PltReadError: If the zone is a poly zone in a v107 file or a v111 POINT
                zone.
        """
        byte_order = self.byte_order
        num_vars = len(self.var_names)

        meta = _ZoneMeta(is_v191=v191)

        # Zone name
        meta.title = _read_string(fp, byte_order)

        # ParentZone (no longer used since 2020r1; still present in file)
        _read_int32(fp, byte_order)

        # StrandID
        meta.strand_id = _read_int32(fp, byte_order)

        # Solution time
        meta.solution_time = _read_float64(fp, byte_order)

        # Default zone colour (unused; set to -1)
        _read_int32(fp, byte_order)

        # ZoneType
        meta.zone_type = ZoneType(_read_int32(fp, byte_order))
        if self._file_version == "v107" and meta.zone_type in _POLY_TYPES:
            raise PltReadError("Poly zones are not supported in v107 files.")

        # Manual tests with v111 showed one extra INT32 value of 0 here. Its unknown what
        # the purpose is, but possible that data formats different that what was
        # manually tested could cause an unreported error here.
        if self._file_version == "v111" and _read_int32(fp, byte_order) != 0:
            raise PltReadError(
                "v111 zone with a nonzero data-packing field (POINT) is not supported."
            )

        # Variable locations
        # PLT spec: 0 = Node, 1 = Cell Centered — opposite of the input ValueLocation
        # enum (CELL_CENTERED=0, NODAL=1) to TecIO functions, so map through
        # _PLT_VALUELOCATION_MAP.
        specify_var_location = _read_int32(fp, byte_order)
        if specify_var_location == 1:
            locs = [_read_int32(fp, byte_order) for _ in range(num_vars)]
            meta.value_locations = [_PLT_VALUELOCATION_MAP[loc] for loc in locs]
        else:
            meta.value_locations = [ValueLocation.NODAL] * num_vars

        # Raw local 1-to-1 face neighbours supplied flag
        meta.has_raw_face_neighbors = bool(_read_int32(fp, byte_order))

        # Number of miscellaneous (non-raw) face-neighbour connections
        meta.declared_misc_face_connections = _read_int32(fp, byte_order)
        if meta.declared_misc_face_connections != 0:
            # face neighbour mode
            meta.face_neighbor_mode = FaceNeighborMode(_read_int32(fp, byte_order))
            zt = meta.zone_type
            fe_types = (
                ZoneType.FELINESEG,
                ZoneType.FETRIANGLE,
                ZoneType.FEQUADRILATERAL,
                ZoneType.FETETRAHEDRON,
                ZoneType.FEBRICK,
                ZoneType.FEPOLYGON,
                ZoneType.FEPOLYHEDRON,
            )
            # v107 has no "completely specified" flag here; every later version does.
            if zt in fe_types and self._file_version != "v107":
                _read_int32(fp, byte_order)

        # Dimensions / element counts
        if meta.zone_type == ZoneType.ORDERED:
            meta.i_max = _read_int32(fp, byte_order)
            meta.j_max = _read_int32(fp, byte_order)
            meta.k_max = _read_int32(fp, byte_order)
        else:
            meta.num_nodes = _read_int32(fp, byte_order)

            if meta.zone_type in (ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON):
                if v191:  # pragma: no cover
                    # v191 stores only the two counts, as INT64. It has no boundary
                    # counts in the zone header, so no boundary sections are read.
                    meta.num_faces = _read_int64(fp, byte_order)
                    meta.total_face_nodes = _read_int64(fp, byte_order)
                else:
                    meta.num_faces = _read_int32(fp, byte_order)
                    meta.total_face_nodes = _read_int32(fp, byte_order)
                    meta.boundary_table_entries = _read_int32(fp, byte_order)
                    meta.num_boundary_faces = max(meta.boundary_table_entries - 1, 0)
                    meta.num_boundary_connections = _read_int32(fp, byte_order)

            meta.num_elements = _read_int32(fp, byte_order)
            # ICellDim, JCellDim, KCellDim (for future use; always 0)
            _read_int32(fp, byte_order)
            _read_int32(fp, byte_order)
            _read_int32(fp, byte_order)

        # Zone-level auxiliary data
        while True:
            has_aux = _read_int32(fp, byte_order)
            if has_aux == 0:
                break
            name = _read_string(fp, byte_order)
            _read_int32(fp, byte_order)  # value format (always 0)
            value = _read_string(fp, byte_order)
            meta.auxdata[name] = value

        self.zones.append(meta)

    # -- Header skip helpers (geometry and text records) -------------------------------

    def _skip_geometry(self, fp: io.BufferedIOBase) -> None:
        """Skip a geometry record (marker already consumed)."""
        byte_order = self.byte_order
        _read_int32(fp, byte_order)  # CoordSys
        _read_int32(fp, byte_order)  # Scope
        _read_int32(fp, byte_order)  # DrawOrder
        # XYZ starting location (3 × FLOAT64)
        fp.read(24)
        _read_int32(fp, byte_order)  # Zone
        _read_int32(fp, byte_order)  # Color
        _read_int32(fp, byte_order)  # FillColor
        _read_int32(fp, byte_order)  # IsFilled
        geom_type = _read_int32(fp, byte_order)
        _read_int32(fp, byte_order)  # LinePattern
        fp.read(16)  # PatternLength + LineThickness (2 × FLOAT64)
        _read_int32(fp, byte_order)  # NumEllipsePts
        _read_int32(fp, byte_order)  # ArrowheadStyle
        _read_int32(fp, byte_order)  # ArrowheadAttachment
        fp.read(16)  # ArrowheadSize + ArrowheadAngle (2 × FLOAT64)
        _skip_string(fp, byte_order)  # MacroFunctionCommand
        gtype = _read_int32(fp, byte_order)  # PolylineFieldDataType (1=float, 2=double)
        gtype_bytes = 4 if gtype == 1 else 8
        _read_int32(fp, byte_order)  # Clipping

        if geom_type == 0:  # Line
            num_polylines = _read_int32(fp, byte_order)
            for _ in range(num_polylines):
                num_pts = _read_int32(fp, byte_order)
                # X-block, Y-block (and Z-block for Grid3D — we cannot
                # distinguish here, so we skip 2 blocks conservatively).
                fp.read(num_pts * gtype_bytes * 2)
        elif geom_type == 1:  # Rectangle
            fp.read(gtype_bytes * 2)
        elif geom_type == 2 or geom_type == 3:  # Square
            fp.read(gtype_bytes)
        elif geom_type == 4:  # Ellipse
            fp.read(gtype_bytes * 2)

    def _skip_text(self, fp: io.BufferedIOBase) -> None:
        """Skip a text record (marker already consumed)."""
        byte_order = self.byte_order
        _read_int32(fp, byte_order)  # CoordSys
        _read_int32(fp, byte_order)  # Scope
        fp.read(24)  # XYZ starting location (3 × FLOAT64)
        _read_int32(fp, byte_order)  # FontType
        _read_int32(fp, byte_order)  # CharHeightUnits
        fp.read(8)  # Height (FLOAT64)
        _read_int32(fp, byte_order)  # TextBoxType
        fp.read(16)  # Margin + MarginLinewidth (2 × FLOAT64)
        _read_int32(fp, byte_order)  # TextBoxOutlineColor
        _read_int32(fp, byte_order)  # TextBoxFillColor
        fp.read(16)  # Angle + LineSpacing (2 × FLOAT64)
        _read_int32(fp, byte_order)  # TextAnchor
        _read_int32(fp, byte_order)  # Zone
        _read_int32(fp, byte_order)  # Color
        _skip_string(fp, byte_order)  # MacroFunctionCommand
        _read_int32(fp, byte_order)  # Clipping
        _skip_string(fp, byte_order)  # The text itself

    def _skip_custom_label(self, fp: io.BufferedIOBase) -> None:
        """Skip a CustomLabel record (marker already consumed)."""
        byte_order = self.byte_order
        n_labels = _read_int32(fp, byte_order)
        for _ in range(n_labels):
            _skip_string(fp, byte_order)

    def _skip_user_rec(self, fp: io.BufferedIOBase) -> None:
        """Skip a UserRec record (marker already consumed)."""
        _skip_string(fp, self.byte_order)

    # -- Data section ------------------------------------------------------------------

    def _parse_data_section(self, fp: io.BufferedIOBase) -> None:
        """Parse the data section and record variable file offsets.

        The EOHMARKER has already been consumed by :meth:`_parse_header`. We iterate
        over zones in declaration order, matching each data block to the corresponding
        :class:`_ZoneMeta` in ``self.zones``.

        Raises:
            PltReadError: If a zone marker is not valid.
        """
        if self._file_version == "v75":
            from ._plt_legacy import parse_v75_data_section

            parse_v75_data_section(self, fp)
            return

        byte_order = self.byte_order
        num_vars = len(self.var_names)

        for zone_idx, meta in enumerate(self.zones):
            # Zone marker (299.0 or 298.0) — must match header.
            data_zone_marker = _read_float32(fp, byte_order)
            if not (
                abs(data_zone_marker - _MARKER_ZONE) < 0.5
                or abs(data_zone_marker - _MARKER_ZONE_V191) < 0.5
            ):
                raise PltReadError(
                    f"Data section zone {zone_idx}: expected zone marker "
                    f"299.0 or 298.0, got {data_zone_marker:.2f}."
                )

            # Per-variable data formats
            raw_data_types = [_read_int32(fp, byte_order) for _ in range(num_vars)]
            meta.var_data_types = [_INT_TO_DATATYPE[dt] for dt in raw_data_types]

            # Passive variables
            has_passive = _read_int32(fp, byte_order)
            if has_passive:
                meta.is_passive = [
                    bool(_read_int32(fp, byte_order)) for _ in range(num_vars)
                ]
            else:
                meta.is_passive = [False] * num_vars

            # Variable sharing
            has_var_sharing = _read_int32(fp, byte_order)
            if has_var_sharing:
                meta.shared_zone = [
                    _read_int32(fp, byte_order) for _ in range(num_vars)
                ]
            else:
                meta.shared_zone = [-1] * num_vars

            # Connectivity sharing (-1 = no sharing)
            meta.connectivity_shared_zone = _read_int32(fp, byte_order)

            # Min / max pairs for non-shared, non-passive variables
            for v in range(num_vars):
                if not meta.is_passive[v] and meta.shared_zone[v] < 0:
                    fp.read(16)  # min (FLOAT64) + max (FLOAT64)

            # Variable data blocks
            self._record_var_offsets(fp, meta, num_vars)

            # Connectivity block (FE non-ORDERED, non-shared)
            self._record_connectivity_offset(fp, meta)

    def _record_var_offsets(
        self,
        fp: io.BufferedIOBase,
        meta: _ZoneMeta,
        num_vars: int,
    ) -> None:
        """Record file offsets for each active variable data block.

        Data is stored in **block** order (all values for variable 0, then all values
        for variable 1, …). We seek rather than read so we do not load the data into
        memory.
        """
        for v in range(num_vars):
            if meta.is_passive[v] or meta.shared_zone[v] >= 0:
                # No data on disk for this variable.
                continue

            dt_int = meta.var_data_types[v].value
            if dt_int == 6:
                raise NotImplementedError(
                    "Bit-packed data (DataType 6) is not supported."
                )

            dtype_str, itemsize = _DATATYPE_TO_NUMPY[dt_int]
            count = self._var_value_count(meta, v)

            file_offset = fp.tell()
            meta.var_offsets[v] = (file_offset, count, dtype_str)

            # Seek past the data block without reading it.
            fp.seek(count * itemsize, os.SEEK_CUR)

    def _var_value_count(self, meta: _ZoneMeta, var_idx: int) -> int:
        """Return the number of values on disk for variable *var_idx*.

        For cell-centered variables in ORDERED zones the PLT format writes ``IMax * JMax
        * (KMax - 1)`` values on disk (see Note 5 of the format spec). The significant
        cell count is ``(I-1) * (J-1) * (K-1)``; the remainder are ghost (zero-padded)
        values that are trimmed in :meth:`TecplotPltVariableReader.get_values` after
        reshaping.

        For FE zones, cell-centered variables store exactly one value per element with
        no padding.
        """
        loc = meta.value_locations[var_idx]

        if meta.zone_type == ZoneType.ORDERED:
            i = max(meta.i_max, 1)
            j = max(meta.j_max, 1)
            k = max(meta.k_max, 1)
            if loc == ValueLocation.CELL_CENTERED:
                # PLT Note 5: IMax * JMax * (KMax - 1) values on disk. Only K is
                # reduced; I and J retain their full extent as ghost padding in the
                # Fortran-order layout.
                return i * j * max(k - 1, 1)
            return i * j * k
        else:
            # FE zones — no ghost padding.
            if loc == ValueLocation.CELL_CENTERED:
                return meta.num_elements
            return meta.num_nodes

    def _record_connectivity_offset(
        self,
        fp: io.BufferedIOBase,
        meta: _ZoneMeta,
    ) -> None:
        """Record the file offset of the connectivity block (if present)."""
        zt = meta.zone_type

        if zt == ZoneType.ORDERED:
            # Ordered zones can have miscellaneous face-neighbor connections
            if (
                meta.connectivity_shared_zone < 0
                and meta.declared_misc_face_connections > 0
            ):
                self._record_face_connections(fp, meta)
            return

        if zt in _POLY_TYPES:
            self._record_poly_face_map(fp, meta)
            return

        # Simple FE zones
        if meta.connectivity_shared_zone >= 0:
            # Connectivity is shared from another zone (nothing on disk)
            return

        nodes_per_cell = _NODES_PER_ELEMENT.get(zt)
        if nodes_per_cell is None:
            return

        count = meta.num_elements * nodes_per_cell
        meta.connectivity_offset = fp.tell()
        meta.connectivity_count = count

        # Seek past the connectivity data
        fp.seek(count * 4, os.SEEK_CUR)  # INT32 values

        # Raw local face neighbors (if supplied)
        if meta.has_raw_face_neighbors:
            faces_per_element = _FACES_PER_ELEMENT.get(zt, nodes_per_cell)
            self._record_face_connections_raw(fp, meta, faces_per_element)

        # Miscellaneous face-neighbor connections
        if meta.declared_misc_face_connections > 0:
            self._record_face_connections(fp, meta)

    def _record_face_connections(
        self,
        fp: io.BufferedIOBase,
        meta: _ZoneMeta,
    ) -> None:
        """Record/read this zone's miscellaneous face-neighbor connections.

        Must always fully advance past the block, or the file position would
        be wrong for whatever is read next, this isn't optional bookkeeping.
        ``LOCAL_ONE_TO_ONE``/``GLOBAL_ONE_TO_ONE`` have a byte length known in
        advance (a fixed 3 or 4 values per connection), so are recorded
        lazily here (offset + count), matching how connectivity itself is
        handled. The "many" modes are ragged, no closed-form length exists,
        correctness already requires reading through every record's own
        length field to find where the next one starts, so they're read and
        cached here rather than walked once now and read again later.
        """
        mode = meta.face_neighbor_mode
        one_to_one = (
            FaceNeighborMode.LOCAL_ONE_TO_ONE,
            FaceNeighborMode.GLOBAL_ONE_TO_ONE,
        )
        if mode in one_to_one:
            values_per = 3 if mode is FaceNeighborMode.LOCAL_ONE_TO_ONE else 4
            flat_count = meta.declared_misc_face_connections * values_per
            meta.face_connections_offset = fp.tell()
            meta.face_connections_flat_count = flat_count
            meta.num_face_connections = meta.declared_misc_face_connections
            fp.seek(flat_count * 4, os.SEEK_CUR)
            return

        # "Many" modes: ragged, walk record-by-record using each record's own
        # nz (the record's own 4th value) to find where the next one starts.
        # Plain fp.read()+frombuffer, not np.fromfile(fp, ...): this file
        # handle is shared with the rest of the metadata scan, which reads
        # via plain fp.read() throughout, mixing that with np.fromfile's own
        # internal buffering desyncs the file position.
        is_global = mode is FaceNeighborMode.GLOBAL_ONE_TO_MANY
        dt = np.dtype(f"{self.byte_order}i4")
        values: list[int] = []
        for _ in range(meta.declared_misc_face_connections):
            header = np.frombuffer(fp.read(16), dtype=dt)
            nz = int(header[3])
            rest_count = 2 * nz if is_global else nz
            rest = np.frombuffer(fp.read(rest_count * 4), dtype=dt)
            values.extend(header.tolist())
            values.extend(rest.tolist())
        meta.num_face_connections = meta.declared_misc_face_connections
        meta.face_connections_eager = np.array(values, dtype=np.int64)

    def _record_face_connections_raw(
        self,
        fp: io.BufferedIOBase,
        meta: _ZoneMeta,
        faces_per_element: int,
    ) -> None:
        """Read and convert the "raw" one-to-one face-neighbor block.

        Dense ``(num_elements, faces_per_element)`` array of 1-based
        neighbor element indices, 0 meaning boundary (no neighbor). Face
        index within each element is implicit, its position in the row.
        Converted to the same ``(cz1, fz, cz2)`` triple shape the explicit
        "miscellaneous" ``LOCAL_ONE_TO_ONE`` storage produces, so both
        on-disk mechanisms present identically through the public API, a
        caller shouldn't need to know or care which one a given file used.
        Cell numbers and the face ordinal are both converted to 0-based.

        Read here rather than recorded lazily by offset, unlike the
        "miscellaneous" one-to-one case: the real connection count isn't
        known until the boundary (zero) entries are filtered out, so cheap
        metadata isn't possible without reading the block anyway.
        """
        dt = np.dtype(f"{self.byte_order}i4")
        count = meta.num_elements * faces_per_element
        raw = np.frombuffer(fp.read(count * 4), dtype=dt).astype(np.int64)
        raw = raw.reshape(meta.num_elements, faces_per_element)

        # "No neighbor" is marked -1 (boundary face), not 0; 0 does appear,
        # rarely, as a separate, genuinely-unassigned-looking case, exclude
        # both, only strictly positive values are real neighbor references.
        elem_idx, face_idx = np.nonzero(raw > 0)
        if elem_idx.size == 0:
            return  # flag was set, but every entry is a boundary face

        cz1 = elem_idx
        fz = face_idx  # already a 0-based position within the row
        cz2 = raw[elem_idx, face_idx] - 1
        triples = np.stack([cz1, fz, cz2], axis=1).ravel()

        meta.face_neighbor_mode = FaceNeighborMode.LOCAL_ONE_TO_ONE
        meta.num_face_connections = int(elem_idx.size)
        meta.face_connections_eager = triples

    def _record_poly_face_map(self, fp: io.BufferedIOBase, meta: _ZoneMeta) -> None:
        """Record the offsets of the face-map blocks of a poly zone and skip them.

        Layout (all INT32 unless noted): polyhedra start with the face node offsets
        (``nf + 1``, or per-face counts in v191), then face nodes, left elements, and
        right elements. Polygons omit the first block, every face has 2 nodes. When
        boundary faces exist, the boundary connection offsets (one more entry than the
        stored table size), elements, and zones follow. Zones are INT16 in v111.
        """
        if meta.connectivity_shared_zone >= 0:
            return

        nf = meta.num_faces
        nbc = meta.num_boundary_connections
        sections = meta.poly_sections

        def take(key: str, count: int, itemsize: int = 4) -> None:
            sections[key] = (fp.tell(), count, itemsize)
            fp.seek(count * itemsize, os.SEEK_CUR)

        if meta.zone_type == ZoneType.FEPOLYHEDRON:
            if meta.is_v191:  # pragma: no cover
                take("face_node_counts", nf)
            else:
                take("face_node_offsets", nf + 1)
        take("face_nodes", meta.total_face_nodes)
        take("face_left_elements", nf)
        take("face_right_elements", nf)

        if meta.boundary_table_entries > 0:
            take("boundary_offsets", meta.boundary_table_entries + 1)
            take("boundary_connection_elements", nbc)
            take(
                "boundary_connection_zones",
                nbc,
                2 if self._file_version == "v111" else 4,
            )


# ======================================================================================
# TecplotPltAuxDataReader
# ======================================================================================


class TecplotPltAuxDataReader(TecplotAuxDataReader):
    """Auxiliary data for PLT files.

    Dataset-, zone-, and variable-level aux data are all parsed eagerly by
    :class:`_PltParser` into plain dicts, this just wraps one of them.
    """

    __slots__ = ("_raw",)
    _raw: dict[str, str]

    def __init__(self, data: dict[str, str]) -> None:
        super().__init__()
        object.__setattr__(self, "_raw", data)

    def _load_data(self) -> dict[str, str]:
        return self._raw


# ======================================================================================
# TecplotPltVariableReader
# ======================================================================================


class TecplotPltVariableReader(TecplotVariableReader):
    """Variable reader for PLT files.

    Metadata comes from the pre-parsed :class:`_ZoneMeta`, no disk I/O. Data values are
    read from disk only when :attr:`values`/:meth:`get_values` is called. ``is_enabled``
    is not overridden here, PLT has no dataset-level enabled flag, so the base class
    default (``not is_passive()``) applies.
    """

    __slots__ = (
        "_file_path",
        "_meta",
        "zone_index",
        "var_index",
        "_var_names",
        "_byte_order",
        "_all_metas",
    )
    _file_path: str | os.PathLike
    _meta: _ZoneMeta
    zone_index: int
    var_index: int
    _var_names: list[str]
    _byte_order: str
    _all_metas: list[_ZoneMeta] | None

    def __init__(
        self,
        file_path: str | os.PathLike,
        zone_meta: _ZoneMeta,
        zone_index: int,
        var_index: int,
        var_names: list[str],
        byte_order: str,
        all_metas: list[_ZoneMeta] | None = None,
    ) -> None:
        """Initialize a lazy variable reader.

        Args:
            file_path:  Path to the ``.plt`` file on disk.
            zone_meta:  Parsed metadata for the parent zone.
            zone_index: 0-based zone index (for API parity with SZL).
            var_index:  0-based variable index.
            var_names:  All variable names for the dataset.
            byte_order: NumPy byte-order prefix (``"<"`` or ``">"``).
            all_metas: All zones' parsed metadata, in file order (needed to resolve
                shared variables). Optional so the class remains constructible
                standalone; without it, shared variables cannot be resolved and read as
                None.
        """
        object.__setattr__(self, "_file_path", file_path)
        object.__setattr__(self, "_meta", zone_meta)
        object.__setattr__(self, "zone_index", zone_index)
        object.__setattr__(self, "var_index", var_index)
        object.__setattr__(self, "_var_names", var_names)
        object.__setattr__(self, "_byte_order", byte_order)
        object.__setattr__(self, "_all_metas", all_metas)

    def _resolve_data_meta(self) -> _ZoneMeta | None:
        """Return the :class:`_ZoneMeta` that owns this variable's data.

        For a variable this zone stores itself this is simply ``self._meta``.  For a
        shared variable, the chain of per-variable ``shared_zone`` references is
        followed to the owning zone (with a cycle/range guard against malformed files),
        the variable-level analogue of
        :meth:`TecplotPltFEZoneReader._resolve_connectivity_meta`. Returns None when the
        share cannot be resolved, e.g. when ``all_metas`` was not supplied at
        construction.
        """
        idx = self.var_index
        meta = self._meta
        if meta.shared_zone[idx] < 0:
            return meta
        if self._all_metas is None:
            return None
        seen: set[int] = set()
        while meta.shared_zone[idx] >= 0:
            src = meta.shared_zone[idx]  # zero-based file value
            if src in seen or src >= len(self._all_metas):
                return None  # cycle or out-of-range: malformed file
            seen.add(src)
            meta = self._all_metas[src]
        return meta

    # -- Metadata (read from zone_meta, no disk I/O) -----------------------------------

    @property
    def name(self) -> str:
        """Variable name string."""
        return self._var_names[self.var_index]

    @property
    def data_type(self) -> DataType:
        """Data type enum for this variable in this zone."""
        return self._meta.var_data_types[self.var_index]

    @property
    def value_location(self) -> ValueLocation:
        """Value location (NODAL or CELL_CENTERED)."""
        return self._meta.value_locations[self.var_index]

    def is_passive(self) -> bool:
        """True if this variable is passive in this zone.

        Returns:
            bool: ``True`` for a passive variable.
        """
        return self._meta.is_passive[self.var_index]

    @property
    def shared_zone(self) -> int | None:
        """Source zone index if this variable is shared, else None."""
        sz = self._meta.shared_zone[self.var_index]
        return sz if sz >= 0 else None

    @property
    def num_values(self) -> int:
        """Number of values stored for this variable.

        For a shared variable the count is read from the owning zone's data block.
        """
        meta = self._resolve_data_meta()
        if meta is None:
            return 0
        offset_info = meta.var_offsets.get(self.var_index)
        if offset_info is None:
            return 0
        _, count, _ = offset_info
        return count

    # -- Data access: read from disk only when called ----------------------------------

    def get_values(
        self,
        value_range: tuple[int | None, int | None] = (None, None),
    ) -> npt.NDArray[Any] | None:
        """Read variable data from disk.

        When this variable is shared from another zone (:attr:`shared_zone` is not
        None), the data is read from the owning zone's data block and returned as this
        variable's own. None is returned only for passive variables, which have no data
        anywhere in the file.

        Args:
            value_range: 1-based ``(start, end)``, half-open. ``(None, None)`` reads all
                values.

        Returns:
            NumPy array with the dtype matching :attr:`data_type`, or None if the
            variable is passive.

        Raises:
            ValueError: On invalid *value_range*.
        """
        if self.is_passive():
            return None

        # Resolve to the zone that actually owns this variable's data (self for unshared
        # variables, the share-chain source otherwise). The resolved meta describes the
        # on-disk block being read, so it also drives the count, dtype, and ordered-zone
        # reshape/ghost-trim logic below.
        meta = self._resolve_data_meta()
        if meta is None:
            return None

        offset_info = meta.var_offsets.get(self.var_index)
        if offset_info is None:
            return None

        file_offset, total_count, dtype_str = offset_info

        full_read = value_range == (None, None)
        if full_read:
            start_idx = 0  # zero-based
            count = total_count
        else:
            start_1, end_1 = value_range
            if start_1 is None or end_1 is None:
                raise ValueError("Both start and end indices must be specified.")
            if start_1 < 1 or start_1 > total_count:
                raise ValueError(
                    f"Start index {start_1} out of range [1, {total_count}]."
                )
            if end_1 > total_count or end_1 < start_1:
                raise ValueError(f"Invalid value range: ({start_1}, {end_1}).")
            start_idx = start_1 - 1
            count = end_1 - start_1

        if meta.is_point_format:
            return self._get_point_format_values(meta, start_idx, count, full_read)

        dt = np.dtype(f"{self._byte_order}{dtype_str}")
        itemsize = dt.itemsize

        with open(self._file_path, "rb") as fp:
            fp.seek(file_offset + start_idx * itemsize)
            data = np.fromfile(fp, dtype=dt, count=count)

        if data.dtype.byteorder not in ("=", "|", np.dtype(dt).str[0]):
            data = data.byteswap().view(data.dtype.newbyteorder())

        # Reshape for full reads of ordered zones, using the owning zone's dimensions
        # and value location (they describe the on-disk layout including cell-centered
        # ghost padding). Sharing requires matching dimensions, so these equal this
        # zone's own for well-formed files.
        if full_read and meta.zone_type == ZoneType.ORDERED:
            ni, nj, nk = meta.i_max, meta.j_max, meta.k_max
            if meta.value_locations[self.var_index] == ValueLocation.CELL_CENTERED:
                # On disk: i * j * (k-1) values in Fortran order. Ghost padding occupies
                # the last row in I and J after reshape, so reshape to (ni, nj, nk-1)
                # then slice to (ni-1, nj-1, nk-1) to discard the ghost values and
                # return only significant cells.
                data = data.reshape((ni, nj, max(nk - 1, 1)), order="F")
                data = data[: max(ni - 1, 1), : max(nj - 1, 1), :]
            else:
                shape = (ni, nj, nk)
                if data.size == shape[0] * shape[1] * shape[2]:
                    data = data.reshape(shape, order="F")

        return data

    def _get_point_format_values(
        self,
        meta: _ZoneMeta,
        start_idx: int,
        count: int,
        full_read: bool,
    ) -> npt.NDArray[Any]:
        """Read one variable's values out of an interleaved (POINT-format) row block.

        POINT-format zones (legacy v75 only) store one row per node/element (all
        active variables for node 0, then node 1, ...), so a single variable's values
        are strided across the file rather than contiguous. Every active variable in
        the row is read together using ``meta.point_row_dtype`` (a structured dtype
        built once at parse time, with the byte order already baked into each field so
        no separate byte-swap pass is needed), and this variable's column is then
        selected out.

        A full read (the common case, and the only one the point-format writer this
        package targets ever exercises) is cached on ``meta.point_format_cache`` so
        that reading every variable of a large POINT-format zone touches disk once
        rather than once per variable. A copy is always returned, never the cached
        array's own view, so a caller mutating the result cannot corrupt the shared
        cache or a later re-read of the same variable.

        Returns:
            NDArray: Copy of the variable values for the requested rows.
        """
        field_name = meta.point_field_by_var[self.var_index]

        if full_read:
            rows = meta.point_format_cache
            if rows is None:
                with open(self._file_path, "rb") as fp:
                    fp.seek(meta.point_data_offset)
                    rows = np.fromfile(
                        fp, dtype=meta.point_row_dtype, count=meta.point_num_rows
                    )
                meta.point_format_cache = rows
            return np.array(rows[field_name])

        assert meta.point_row_dtype is not None  # narrowed: is_point_format was True
        with open(self._file_path, "rb") as fp:
            fp.seek(meta.point_data_offset + start_idx * meta.point_row_dtype.itemsize)
            rows = np.fromfile(fp, dtype=meta.point_row_dtype, count=count)
        return np.array(rows[field_name])


# ======================================================================================
# TecplotPltOrderedZoneReader / TecplotPltFEZoneReader
# ======================================================================================


def _load_plt_variables(
    file_path: str | os.PathLike,
    meta: _ZoneMeta,
    zone_index: int,
    num_vars: int,
    var_names: list[str],
    byte_order: str,
    all_metas: list[_ZoneMeta] | None,
) -> list[TecplotVariableReader]:
    """Build this zone's variable readers. Shared by both PLT zone classes.

    Returns:
        list: One reader class per variable of the zone.
    """
    return [
        TecplotPltVariableReader(
            file_path=file_path,
            zone_meta=meta,
            zone_index=zone_index,
            var_index=i,
            var_names=var_names,
            byte_order=byte_order,
            all_metas=all_metas,
        )
        for i in range(num_vars)
    ]


def _resolve_connectivity_meta(
    meta: _ZoneMeta, all_metas: list[_ZoneMeta] | None
) -> _ZoneMeta | None:
    """Return the :class:`_ZoneMeta` that contains the *meta* connectivity.

    For a zone with its own connectivity this is *meta* itself. For a zone that shares
    connectivity, the chain of ``connectivity_shared_zone`` references is followed to
    the owning zone (with a cycle/range guard against malformed files). Returns None
    when the share cannot be resolved, e.g. when *all_metas* was not supplied.
    """
    if meta.connectivity_shared_zone < 0:
        return meta
    if all_metas is None:
        return None
    seen: set[int] = set()
    while meta.connectivity_shared_zone >= 0:
        src = meta.connectivity_shared_zone  # zero-based file value
        if src in seen or src >= len(all_metas):
            return None  # cycle or out-of-range: malformed file
        seen.add(src)
        meta = all_metas[src]
    return meta


class TecplotPltOrderedZoneReader(TecplotOrderedZoneReader):
    """Ordered (IJK) zone reader for PLT files.

    All metadata comes from the pre-parsed :class:`_ZoneMeta`, no disk I/O beyond what
    :class:`_PltParser` already did. The variable list and aux data stay lazily loaded
    on first access.
    """

    __slots__ = (
        "_file_path",
        "_meta",
        "_num_vars",
        "_var_names",
        "_byte_order",
        "_all_metas",
    )
    _file_path: str | os.PathLike
    _meta: _ZoneMeta
    _num_vars: int
    _var_names: list[str]
    _byte_order: str
    _all_metas: list[_ZoneMeta] | None

    def __init__(
        self,
        file_path: str | os.PathLike,
        meta: _ZoneMeta,
        zone_index: int,
        num_vars: int,
        var_names: list[str],
        byte_order: str,
        all_metas: list[_ZoneMeta] | None = None,
    ) -> None:
        super().__init__(
            zone_index=zone_index,
            title=meta.title,
            solution_time=meta.solution_time,
            strand_id=max(meta.strand_id + 1, 0),
            datapacking=(
                DataPacking.POINT if meta.is_point_format else DataPacking.BLOCK
            ),
            i=meta.i_max,
            j=meta.j_max,
            k=meta.k_max,
        )
        object.__setattr__(self, "_file_path", file_path)
        object.__setattr__(self, "_meta", meta)
        object.__setattr__(self, "_num_vars", num_vars)
        object.__setattr__(self, "_var_names", var_names)
        object.__setattr__(self, "_byte_order", byte_order)
        object.__setattr__(self, "_all_metas", all_metas)

    def _load_variables(self) -> list[TecplotVariableReader]:
        return _load_plt_variables(
            self._file_path,
            self._meta,
            self.zone_index,
            self._num_vars,
            self._var_names,
            self._byte_order,
            self._all_metas,
        )

    def _load_auxdata(self) -> TecplotAuxDataReader:
        return TecplotPltAuxDataReader(self._meta.auxdata)


class TecplotPltFEZoneReader(TecplotFEZoneReader):
    """Finite-element zone reader for PLT files.

    All metadata comes from the pre-parsed :class:`_ZoneMeta`, no disk I/O beyond what
    :class:`_PltParser` already did. The variable list, node map, and aux data stay
    lazily loaded on first access.
    """

    __slots__ = (
        "_file_path",
        "_meta",
        "_num_vars",
        "_var_names",
        "_byte_order",
        "_all_metas",
    )
    _file_path: str | os.PathLike
    _meta: _ZoneMeta
    _num_vars: int
    _var_names: list[str]
    _byte_order: str
    _all_metas: list[_ZoneMeta] | None

    def __init__(
        self,
        file_path: str | os.PathLike,
        meta: _ZoneMeta,
        zone_index: int,
        num_vars: int,
        var_names: list[str],
        byte_order: str,
        all_metas: list[_ZoneMeta] | None = None,
    ) -> None:
        super().__init__(
            zone_index=zone_index,
            title=meta.title,
            zone_type=meta.zone_type,
            solution_time=meta.solution_time,
            strand_id=max(meta.strand_id + 1, 0),
            datapacking=(
                DataPacking.POINT if meta.is_point_format else DataPacking.BLOCK
            ),
            num_nodes=meta.num_nodes,
            num_elements=meta.num_elements,
            shared_connectivity=(
                meta.connectivity_shared_zone
                if meta.connectivity_shared_zone >= 0
                else None
            ),
        )
        object.__setattr__(self, "_file_path", file_path)
        object.__setattr__(self, "_meta", meta)
        object.__setattr__(self, "_num_vars", num_vars)
        object.__setattr__(self, "_var_names", var_names)
        object.__setattr__(self, "_byte_order", byte_order)
        object.__setattr__(self, "_all_metas", all_metas)

    def _load_variables(self) -> list[TecplotVariableReader]:
        return _load_plt_variables(
            self._file_path,
            self._meta,
            self.zone_index,
            self._num_vars,
            self._var_names,
            self._byte_order,
            self._all_metas,
        )

    def _resolve_connectivity_meta(self) -> _ZoneMeta | None:
        """Return the :class:`_ZoneMeta` that owns this zone's connectivity."""
        return _resolve_connectivity_meta(self._meta, self._all_metas)

    def _load_node_map(self) -> npt.NDArray[np.int64] | None:
        """Read this zone's node connectivity.

        Returns:
            NDArray: Zero-based node map, or ``None`` if the zone has no stored
                connectivity.
        """
        meta = self._resolve_connectivity_meta()
        if meta is None or meta.connectivity_offset is None:
            return None

        count = meta.connectivity_count
        dt = np.dtype(f"{self._byte_order}i4")

        with open(self._file_path, "rb") as fp:
            fp.seek(meta.connectivity_offset)
            flat = np.fromfile(fp, dtype=dt, count=count)

        node_map = flat.reshape(self.num_elements, -1).astype(np.int64)
        if meta.connectivity_is_one_based:
            # v75 only: 1-based on disk, converted to the 0-based public convention.
            node_map -= 1
        return node_map

    def _load_face_neighbor_meta(
        self,
    ) -> tuple[FaceNeighborMode, int, bool | None] | None:
        meta = self._meta
        if meta.num_face_connections == 0 or meta.face_neighbor_mode is None:
            return None
        # PLT has no completeness concept: neither the classic API
        # (TECZNE142/TECZNEFEMIXED142) nor the new API (tecZoneCreateFE/
        # tecZoneCreateFEMixed) has a parameter for it. Always None here,
        # not a gap in this reader, the binary format genuinely can't
        # express the distinction.
        return (meta.face_neighbor_mode, meta.num_face_connections, None)

    def _load_face_connections(self) -> npt.NDArray[np.int64]:
        meta = self._meta
        if meta.face_connections_eager is not None:
            # Eager cache is either the "raw" mechanism (already converted to
            # the public convention when recorded) or a "many" mode record
            # from the offset-based mechanism below (not validated against
            # real data either way, left as-is rather than guessing).
            return meta.face_connections_eager

        assert meta.face_connections_offset is not None  # narrowed: one-to-one modes
        count = meta.face_connections_flat_count
        dt = np.dtype(f"{self._byte_order}i4")
        with open(self._file_path, "rb") as fp:
            fp.seek(meta.face_connections_offset)
            flat = np.fromfile(fp, dtype=dt, count=count)
        # Confirmed 0-based on disk against known-good data (cross-checked
        # triple-for-triple against the equivalent hand-written DAT file). Cell
        # references, remote-zone references, and the face ordinal (second
        # value of each record) are all already 0-based on disk here, matching
        # the public convention directly, so no conversion is needed.
        one_to_one = FaceNeighborMode.LOCAL_ONE_TO_ONE
        values_per = 3 if meta.face_neighbor_mode is one_to_one else 4
        records = flat.astype(np.int64).reshape(-1, values_per)
        return records.ravel()

    def _load_auxdata(self) -> TecplotAuxDataReader:
        return TecplotPltAuxDataReader(self._meta.auxdata)


class TecplotPltPolyZoneReader(TecplotPolyZoneReader):
    """Poly (FEPOLYGON / FEPOLYHEDRON) zone reader for PLT files.

    Metadata comes from the pre-parsed :class:`_ZoneMeta`. Each face-map array is read
    from disk on first access.
    """

    __slots__ = (
        "_file_path",
        "_meta",
        "_num_vars",
        "_var_names",
        "_byte_order",
        "_all_metas",
    )
    _file_path: str | os.PathLike
    _meta: _ZoneMeta
    _num_vars: int
    _var_names: list[str]
    _byte_order: str
    _all_metas: list[_ZoneMeta] | None

    def __init__(
        self,
        file_path: str | os.PathLike,
        meta: _ZoneMeta,
        zone_index: int,
        num_vars: int,
        var_names: list[str],
        byte_order: str,
        all_metas: list[_ZoneMeta] | None = None,
    ) -> None:
        super().__init__(
            zone_index=zone_index,
            title=meta.title,
            zone_type=meta.zone_type,
            solution_time=meta.solution_time,
            strand_id=max(meta.strand_id + 1, 0),
            datapacking=DataPacking.BLOCK,
            num_nodes=meta.num_nodes,
            num_elements=meta.num_elements,
            num_faces=meta.num_faces,
            num_boundary_faces=meta.num_boundary_faces,
            num_boundary_connections=meta.num_boundary_connections,
            shared_connectivity=(
                meta.connectivity_shared_zone
                if meta.connectivity_shared_zone >= 0
                else None
            ),
        )
        object.__setattr__(self, "_file_path", file_path)
        object.__setattr__(self, "_meta", meta)
        object.__setattr__(self, "_num_vars", num_vars)
        object.__setattr__(self, "_var_names", var_names)
        object.__setattr__(self, "_byte_order", byte_order)
        object.__setattr__(self, "_all_metas", all_metas)

    def _load_variables(self) -> list[TecplotVariableReader]:
        return _load_plt_variables(
            self._file_path,
            self._meta,
            self.zone_index,
            self._num_vars,
            self._var_names,
            self._byte_order,
            self._all_metas,
        )

    def _read_section(self, meta: _ZoneMeta, key: str) -> npt.NDArray[np.int64]:
        """Read one recorded face-map block as int64.

        Returns:
            NDArray: Flat array of node indices for section.
        """
        offset, count, itemsize = meta.poly_sections[key]
        dt = np.dtype(f"{self._byte_order}i{itemsize}")
        with open(self._file_path, "rb") as fp:
            fp.seek(offset)
            return np.fromfile(fp, dtype=dt, count=count).astype(np.int64)

    def _load_array(self, key: PolyArrayKey) -> npt.NDArray[np.int64]:
        meta = _resolve_connectivity_meta(self._meta, self._all_metas)
        if meta is None or not meta.poly_sections:
            raise PltReadError(
                f"Zone {self.zone_index}: shared connectivity could not be resolved."
            )
        if key == "face_node_counts":
            if "face_node_counts" in meta.poly_sections:  # pragma: no cover
                return self._read_section(meta, "face_node_counts")
            return np.diff(self._read_section(meta, "face_node_offsets"))
        if key == "boundary_connection_counts":
            return np.diff(self._read_section(meta, "boundary_offsets"))[1:]
        # The remaining arrays are stored 0-based, with -1 for "none" and -(b + 2)
        # for boundary face b, matching the public convention directly.
        return self._read_section(meta, key)

    def _load_auxdata(self) -> TecplotAuxDataReader:
        return TecplotPltAuxDataReader(self._meta.auxdata)


def _build_plt_zone(
    file_path: str | os.PathLike,
    meta: _ZoneMeta,
    zone_index: int,
    num_vars: int,
    var_names: list[str],
    byte_order: str,
    all_metas: list[_ZoneMeta] | None,
) -> TecplotZoneReader:
    """Construct the right concrete zone reader for one parsed zone.

    Returns:
        TecplotZoneReader: The zone reader class that matches the zone type.
    """
    if meta.zone_type == ZoneType.ORDERED:
        return TecplotPltOrderedZoneReader(
            file_path, meta, zone_index, num_vars, var_names, byte_order, all_metas
        )
    if meta.zone_type in _POLY_TYPES:
        return TecplotPltPolyZoneReader(
            file_path, meta, zone_index, num_vars, var_names, byte_order, all_metas
        )
    return TecplotPltFEZoneReader(
        file_path, meta, zone_index, num_vars, var_names, byte_order, all_metas
    )


# ======================================================================================
# TecplotPltReader
# ======================================================================================


class TecplotPltReader(TecplotReader):
    """Reader for Tecplot ``.plt`` binary files.

    Zone metadata is parsed eagerly at construction (:class:`_PltParser` reads the whole
    header section up front); variable data and node maps are read lazily from
    disk. Holds no open file handle between accesses, :meth:`close` is a no-op
    (inherited from the base class default).

    Args:
        file_name: Path to the ``.plt`` file.

    Raises:
        PltReadError: If the file cannot be parsed as a valid PLT binary.
    """

    __slots__ = (
        "_path",
        "_file_type",
        "_title",
        "_var_names",
        "_byte_order",
        "_zone_metas",
        "_dataset_aux_raw",
        "_raw_var_auxdata",
        "_zone_list",
        "_dataset_auxdata",
    )

    def __init__(self, file_name: str | os.PathLike) -> None:
        self._path = os.fspath(file_name)
        parser = _PltParser(self._path)

        self._file_type: FileType = parser.file_type
        self._title: str = parser.title
        self._var_names: list[str] = parser.var_names
        self._byte_order: str = parser.byte_order
        self._zone_metas: list[_ZoneMeta] = parser.zones
        self._dataset_aux_raw: dict[str, str] = parser.dataset_auxdata
        self._raw_var_auxdata: dict[int, dict[str, str]] = parser.var_auxdata
        self._zone_list: ZoneList[TecplotZoneReader] | None = None
        self._dataset_auxdata: TecplotAuxDataReader | None = None

    @property
    def path(self) -> str:
        """Source file path."""
        return self._path

    @property
    def file_type(self) -> FileType:
        """File type (FULL, GRID, or SOLUTION)."""
        return self._file_type

    @property
    def title(self) -> str:
        """Dataset title string."""
        return self._title

    @property
    def num_vars(self) -> int:
        """Number of variables in the dataset."""
        return len(self._var_names)

    @property
    def variables(self) -> list[str]:
        """Ordered list of variable name strings."""
        return list(self._var_names)

    @property
    def num_zones(self) -> int:
        """Number of zones in the file.

        Queried directly from the already-parsed metadata list. Cheaper than the base
        class's default of ``len(self.zones)``, which would build every zone's reader
        object first.
        """
        return len(self._zone_metas)

    @property
    def zones(self) -> ZoneList[TecplotZoneReader]:
        """Zones in this file, by 0-based index or slice."""
        if self._zone_list is None:
            self._zone_list = ZoneList([
                _build_plt_zone(
                    file_path=self._path,
                    meta=meta,
                    zone_index=i,
                    num_vars=self.num_vars,
                    var_names=self._var_names,
                    byte_order=self._byte_order,
                    all_metas=self._zone_metas,
                )
                for i, meta in enumerate(self._zone_metas)
            ])
        return self._zone_list

    @property
    def auxdata(self) -> TecplotAuxDataReader:
        """Dataset-level auxiliary data."""
        if self._dataset_auxdata is None:
            self._dataset_auxdata = TecplotPltAuxDataReader(self._dataset_aux_raw)
        return self._dataset_auxdata

    def _var_auxdata_at(self, var_index: int) -> TecplotAuxDataReader:
        return TecplotPltAuxDataReader(self._raw_var_auxdata.get(var_index, {}))
