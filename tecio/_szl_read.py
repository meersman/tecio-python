"""Read Tecplot SZL (``.szplt``) files via the TecIO C library.

Small scalar zone metadata (title, type, dimensions or node/element counts, solution
time, strand ID, shared connectivity) is queried once, at zone construction, and frozen
by :class:`~tecio._reader.TecplotOrderedZoneReader` /
:class:`~tecio._reader.TecplotFEZoneReader`. Variable metadata is queried live, on each
access, it's a single cheap C call either way. Variable data arrays and node maps are
read lazily, on first access, straight from the C library.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from . import libtecio
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
    TecplotAuxDataReader,
    TecplotFEZoneReader,
    TecplotOrderedZoneReader,
    TecplotReader,
    TecplotVariableReader,
    TecplotZoneReader,
    convert_face_connections_to_zero_based,
)

# ======================================================================================
# Auxiliary data
# ======================================================================================


class TecplotSzlDatasetAuxDataReader(TecplotAuxDataReader):
    """Dataset-level auxiliary data for an open SZL file."""

    __slots__ = ("_handle",)
    _handle: ctypes.c_void_p

    def __init__(self, handle: ctypes.c_void_p) -> None:
        super().__init__()
        object.__setattr__(self, "_handle", handle)

    def _load_data(self) -> dict[str, str]:
        data: dict[str, str] = {}
        num_items = libtecio.tec_data_set_aux_data_get_num_items(self._handle)
        for i in range(num_items):
            name, value = libtecio.tec_data_set_aux_data_get_item(self._handle, i + 1)
            data[name] = value
        return data


class TecplotSzlVarAuxDataReader(TecplotAuxDataReader):
    """Variable-level auxiliary data for one variable in an open SZL file."""

    __slots__ = ("_handle", "_var_index")
    _handle: ctypes.c_void_p
    _var_index: int

    def __init__(self, handle: ctypes.c_void_p, var_index: int) -> None:
        super().__init__()
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "_var_index", var_index)

    def _load_data(self) -> dict[str, str]:
        data: dict[str, str] = {}
        num_items = libtecio.tec_var_aux_data_get_num_items(
            self._handle, self._var_index
        )
        for i in range(num_items):
            name, value = libtecio.tec_var_aux_data_get_item(
                self._handle, self._var_index, i + 1
            )
            data[name] = value
        return data


class TecplotSzlZoneAuxDataReader(TecplotAuxDataReader):
    """Zone-level auxiliary data for one zone in an open SZL file."""

    __slots__ = ("_handle", "_zone_index")
    _handle: ctypes.c_void_p
    _zone_index: int

    def __init__(self, handle: ctypes.c_void_p, zone_index: int) -> None:
        super().__init__()
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "_zone_index", zone_index)

    def _load_data(self) -> dict[str, str]:
        data: dict[str, str] = {}
        num_items = libtecio.tec_zone_aux_data_get_num_items(
            self._handle, self._zone_index
        )
        for i in range(num_items):
            name, value = libtecio.tec_zone_aux_data_get_item(
                self._handle, self._zone_index, i + 1
            )
            data[name] = value
        return data


# ======================================================================================
# TecplotSzlVariableReader
# ======================================================================================


class TecplotSzlVariableReader(TecplotVariableReader):
    """Variable reader for SZL files.

    Every metadata property queries the C library directly on each access, one call is
    already cheap and callers don't typically re-read the same metadata field in a hot
    loop the way they do :attr:`values`.
    """

    __slots__ = ("_handle", "zone_index", "var_index")
    _handle: ctypes.c_void_p
    zone_index: int
    var_index: int

    def __init__(
        self, handle: ctypes.c_void_p, zone_index: int, var_index: int
    ) -> None:
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "zone_index", zone_index)
        object.__setattr__(self, "var_index", var_index)

    @property
    def name(self) -> str:
        """Variable name string."""
        return libtecio.tec_var_get_name(self._handle, self.var_index + 1)

    def is_enabled(self) -> bool:
        """True if the variable is enabled (dataset-level flag, queried live).

        Returns:
            bool: ``True`` for an enabled variable.
        """
        return libtecio.tec_var_is_enabled(self._handle, self.var_index + 1)

    @property
    def shared_zone(self) -> int | None:
        """Source zone index if shared, or None."""
        raw = libtecio.tec_zone_var_get_shared_zone(
            self._handle, self.zone_index + 1, self.var_index + 1
        )
        return raw - 1 if raw is not None else None

    def _data_zone(self) -> int:
        """Return the 0-based zone index that actually holds this variable's data."""
        shared = self.shared_zone
        return shared if shared is not None else self.zone_index

    @property
    def data_type(self) -> DataType:
        """Data type enum for this variable."""
        return libtecio.tec_zone_var_get_type(
            self._handle, self._data_zone() + 1, self.var_index + 1
        )

    @property
    def value_location(self) -> ValueLocation:
        """Value location (NODAL or CELL_CENTERED)."""
        return libtecio.tec_zone_var_get_value_location(
            self._handle, self._data_zone() + 1, self.var_index + 1
        )

    def is_passive(self) -> bool:
        """True if this variable is passive in this zone.

        Returns:
            bool: ``True`` for a passive variable.
        """
        return libtecio.tec_zone_var_is_passive(
            self._handle, self.zone_index + 1, self.var_index + 1
        )

    @property
    def num_values(self) -> int:
        """Number of values in the data array.

        Returns:
            int: Number of values.
        """
        return libtecio.tec_zone_var_get_num_values(
            self._handle, self._data_zone() + 1, self.var_index + 1
        )

    def get_values(
        self,
        value_range: tuple[int | None, int | None] = (None, None),
    ) -> npt.NDArray[Any] | None:
        """Get variable values with optional range specification.

        For ordered zones a full read returns the array shaped ``(I, J, K)`` for nodal
        variables, or ``(I-1, J-1, K-1)`` for cell-centered, so zone dimensions can be
        inferred directly from the array shape. Partial reads always return a flat 1-D
        array.

        Args:
            value_range: 1-based ``(start, end)``. ``(None, None)`` reads all values.

        Returns:
            NumPy array with the dtype matching :attr:`data_type`, reshaped for full
            reads of ordered zones, flat for FE zones and partial reads. A shared
            variable resolves to its source zone's array. None only if the variable is
            passive.

        Raises:
            ValueError: If only one of start/end is specified.
        """
        if self.is_passive():
            return None

        data_zone = self._data_zone() + 1
        data_type = self.data_type
        full_read = value_range == (None, None)

        if full_read:
            start_index = 1
            num_values = self.num_values
        else:
            start_index = value_range[0]
            end_index = value_range[1]

            if start_index is None or end_index is None:
                raise ValueError("Both start and end indices must be specified")

            num_values = end_index - start_index

            if start_index > self.num_values or start_index < 1:
                raise ValueError(
                    f"Start index {start_index} out of range [1, {self.num_values}]"
                )
            if num_values < 0 or end_index > self.num_values:
                raise ValueError(f"Invalid value range: ({start_index}, {end_index})")

        if data_type == DataType.FLOAT:
            arr = libtecio.tec_zone_var_get_float_values(
                self._handle, data_zone, self.var_index + 1, start_index, num_values
            )
        elif data_type == DataType.DOUBLE:
            arr = libtecio.tec_zone_var_get_double_values(
                self._handle, data_zone, self.var_index + 1, start_index, num_values
            )
        elif data_type == DataType.INT32:
            arr = libtecio.tec_zone_var_get_int32_values(
                self._handle, data_zone, self.var_index + 1, start_index, num_values
            )
        elif data_type == DataType.INT16:
            arr = libtecio.tec_zone_var_get_int16_values(
                self._handle, data_zone, self.var_index + 1, start_index, num_values
            )
        elif data_type == DataType.BYTE:
            arr = libtecio.tec_zone_var_get_uint8_values(
                self._handle, data_zone, self.var_index + 1, start_index, num_values
            )
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        # Reshape to (I, J, K) / (I-1, J-1, K-1) for full reads of ordered zones.
        if full_read and arr is not None:
            ni, nj, nk = libtecio.tec_zone_get_ijk(self._handle, self.zone_index + 1)
            zt = libtecio.tec_zone_get_type(self._handle, self.zone_index + 1)
            if zt == ZoneType.ORDERED:
                if self.value_location == ValueLocation.CELL_CENTERED:
                    shape = (max(ni - 1, 1), max(nj - 1, 1), max(nk - 1, 1))
                else:
                    shape = (ni, nj, nk)
                if arr.size == shape[0] * shape[1] * shape[2]:
                    arr = arr.reshape(shape, order="F")

        return arr


# ======================================================================================
# TecplotSzlOrderedZoneReader / TecplotSzlFEZoneReader
# ======================================================================================


def _load_szl_variables(
    handle: ctypes.c_void_p, zone_index: int, num_vars: int
) -> list[TecplotVariableReader]:
    """Build this zone's variable readers. Shared by both SZL zone classes.

    Returns:
        list: One reader per variable of the zone.
    """
    return [TecplotSzlVariableReader(handle, zone_index, i) for i in range(num_vars)]


def _szl_zone_is_enabled(handle: ctypes.c_void_p, zone_index: int) -> bool:
    """Query the C library's per-zone enabled flag. Shared by both SZL zone classes.

    Returns:
        bool: ``True`` if the zone is enabled.
    """
    return libtecio.tec_zone_is_enabled(handle, zone_index + 1)


class TecplotSzlOrderedZoneReader(TecplotOrderedZoneReader):
    """Ordered (IJK) zone reader for SZL files.

    Scalar metadata is queried from the C library once, at construction, and frozen by
    the base class. The variable list and aux data stay lazily loaded on first access.
    """

    __slots__ = ("_handle", "_num_vars")
    _handle: ctypes.c_void_p
    _num_vars: int

    def __init__(self, handle: ctypes.c_void_p, zone_index: int, num_vars: int) -> None:
        zone_num = zone_index + 1
        i, j, k = libtecio.tec_zone_get_ijk(handle, zone_num)
        super().__init__(
            zone_index=zone_index,
            title=libtecio.tec_zone_get_title(handle, zone_num),
            solution_time=libtecio.tec_zone_get_solution_time(handle, zone_num),
            strand_id=libtecio.tec_zone_get_strand_id(handle, zone_num),
            datapacking=DataPacking.BLOCK,
            i=i,
            j=j,
            k=k,
        )
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "_num_vars", num_vars)

    def is_enabled(self) -> bool:
        """True if the zone is enabled (queried live).

        SZL is the only format with a genuine per-zone enabled flag; PLT and
        DAT use the base class default of always True.

        Returns:
            bool: ``True`` for an enabled zone.
        """
        return _szl_zone_is_enabled(self._handle, self.zone_index)

    def _load_variables(self) -> list[TecplotVariableReader]:
        return _load_szl_variables(self._handle, self.zone_index, self._num_vars)

    def _load_auxdata(self) -> TecplotAuxDataReader:
        return TecplotSzlZoneAuxDataReader(self._handle, self.zone_index + 1)


class TecplotSzlFEZoneReader(TecplotFEZoneReader):
    """Finite-element zone reader for SZL files.

    Scalar metadata is queried from the C library once, at construction, and frozen by
    the base class. The variable list, node map, and aux data stay lazily loaded on
    first access.
    """

    __slots__ = ("_handle", "_num_vars")
    _handle: ctypes.c_void_p
    _num_vars: int

    def __init__(self, handle: ctypes.c_void_p, zone_index: int, num_vars: int) -> None:
        zone_num = zone_index + 1
        zone_type = ZoneType(libtecio.tec_zone_get_type(handle, zone_num))
        num_nodes, num_elements, _ = libtecio.tec_zone_get_ijk(handle, zone_num)
        raw_shared = libtecio.tec_zone_connectivity_get_shared_zone(handle, zone_num)
        shared_connectivity = raw_shared - 1 if raw_shared is not None else None
        super().__init__(
            zone_index=zone_index,
            title=libtecio.tec_zone_get_title(handle, zone_num),
            zone_type=zone_type,
            solution_time=libtecio.tec_zone_get_solution_time(handle, zone_num),
            strand_id=libtecio.tec_zone_get_strand_id(handle, zone_num),
            datapacking=DataPacking.BLOCK,
            num_nodes=num_nodes,
            num_elements=num_elements,
            shared_connectivity=shared_connectivity,
        )
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "_num_vars", num_vars)

    def is_enabled(self) -> bool:
        """True if the zone is enabled (queried live).

        SZL is the only format with a genuine per-zone enabled flag; PLT and DAT use the
        base class default of always True.

        Returns:
            bool: ``True`` for an enabled zone.
        """
        return _szl_zone_is_enabled(self._handle, self.zone_index)

    def _load_variables(self) -> list[TecplotVariableReader]:
        return _load_szl_variables(self._handle, self.zone_index, self._num_vars)

    def _connectivity_zone(self) -> int:
        """Return the 0-based zone index that holds this zone's connectivity."""
        shared = self.shared_connectivity
        return shared if shared is not None else self.zone_index

    def _load_node_map(self) -> npt.NDArray[np.int64]:
        connectivity_zone_num = self._connectivity_zone() + 1
        if libtecio.is_64bit(self._handle, self.zone_index + 1):
            raw = libtecio.tec_zone_node_map_get_64(
                self._handle,
                connectivity_zone_num,
                self.num_elements,
                self.nodes_per_cell,
            )
        else:
            raw = libtecio.tec_zone_node_map_get(
                self._handle,
                connectivity_zone_num,
                self.num_elements,
                self.nodes_per_cell,
            ).astype(np.int64)
        # Node numbers from the C library are 1-based; convert to 0-based indices
        # that point directly into the corresponding coordinate array.
        return raw - 1

    def _load_face_neighbor_meta(
        self,
    ) -> tuple[FaceNeighborMode, int, bool | None] | None:
        zone_num = self.zone_index + 1
        num_connections = libtecio.tec_zone_face_nbr_get_num_connections(
            self._handle, zone_num
        )
        if num_connections == 0:
            return None
        mode = libtecio.tec_zone_face_nbr_get_mode(self._handle, zone_num)
        # SZL has no completeness concept: no tecZoneFaceNbr* function reports
        # it, and none exists to write it either. Always None here, not a gap
        # in this reader, the C API genuinely has nothing to query.
        return (mode, num_connections, None)

    def _load_face_connections(self) -> npt.NDArray[np.int64]:
        zone_num = self.zone_index + 1
        num_values = libtecio.tec_zone_face_nbr_get_num_values(self._handle, zone_num)
        if libtecio.tec_zone_face_nbrs_are_64bit(self._handle, zone_num):
            flat = libtecio.tec_zone_face_nbr_get_connections_64(
                self._handle, zone_num, num_values
            )
        else:
            flat = libtecio.tec_zone_face_nbr_get_connections(
                self._handle, zone_num, num_values
            ).astype(np.int64)
        assert self.face_neighbor_mode is not None  # only called when connections exist
        return convert_face_connections_to_zero_based(flat, self.face_neighbor_mode)

    def _load_auxdata(self) -> TecplotAuxDataReader:
        return TecplotSzlZoneAuxDataReader(self._handle, self.zone_index + 1)


def _build_szl_zone(
    handle: ctypes.c_void_p, zone_index: int, num_vars: int
) -> TecplotZoneReader:
    """Construct the right concrete zone reader for zone *zone_index*.

    A single ``.szplt`` file commonly holds a mix of ORDERED and FE zones, so this
    queries the C library for the zone type first, cheap, one call, then dispatches to
    the matching class.

    Returns:
        TecplotZoneReader: The zone reader class that matches the zone type.

    Raises:
        NotImplementedError: For polygon and polyhedron zones, which only the PLT and
            DAT readers support.
    """
    zone_type = ZoneType(libtecio.tec_zone_get_type(handle, zone_index + 1))
    if zone_type == ZoneType.ORDERED:
        return TecplotSzlOrderedZoneReader(handle, zone_index, num_vars)
    if zone_type in (ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON):  # pragma: no cover
        raise NotImplementedError(
            f"Zone {zone_index} is {zone_type.name}; polygon and polyhedron zones "
            "are not yet supported in SZPLT files."
        )
    return TecplotSzlFEZoneReader(handle, zone_index, num_vars)


# ======================================================================================
# TecplotSzlReader
# ======================================================================================


class TecplotSzlReader(TecplotReader):
    """Reader for Tecplot ``.szplt`` files.

    Keeps a live C file handle for the reader's lifetime. Zones are constructed, and
    their scalar metadata resolved, on first access to :attr:`zone`, not at open time,
    so opening a file with many zones stays cheap if the caller only wants dataset-level
    information.

    Args:
        file_name: Path to the ``.szplt`` file.

    Raises:
        FileNotFoundError: If *file_name* does not exist.
    """

    __slots__ = ("_handle", "_path", "_zone_list", "_dataset_auxdata")

    def __init__(self, file_name: str) -> None:
        if not Path(file_name).exists():
            raise FileNotFoundError(f"No such file or directory: '{file_name}'")
        self._handle: ctypes.c_void_p | None = libtecio.tec_file_reader_open(file_name)
        self._path = str(file_name)
        self._zone_list: ZoneList[TecplotZoneReader] | None = None
        self._dataset_auxdata: TecplotAuxDataReader | None = None

    def _check_handle(self) -> ctypes.c_void_p:
        """Return the file handle, or raise if the reader has been closed.

        Raises:
            ValueError: If the reader has been closed.
        """
        if self._handle is None:
            raise ValueError(f"I/O operation on closed file: '{self._path}'")
        return self._handle

    @property
    def path(self) -> str:
        """Source file path."""
        return self._path

    @property
    def file_type(self) -> FileType:
        """File type enum (FULL, GRID, or SOLUTION)."""
        return libtecio.tec_file_get_type(self._check_handle())

    @property
    def title(self) -> str:
        """Dataset title string."""
        return libtecio.tec_data_set_get_title(self._check_handle())

    @property
    def num_vars(self) -> int:
        """Number of variables in the dataset.

        Queried directly. Cheaper than the base class's default of
        ``len(self.variables)``, which would need :attr:`zone` built first.
        """
        return libtecio.tec_data_set_get_num_vars(self._check_handle())

    @property
    def variables(self) -> list[str]:
        """Ordered list of variable name strings.

        Variable names are dataset-global in SZL, so this queries them
        directly rather than through any zone's variable list.
        """
        handle = self._check_handle()
        return [libtecio.tec_var_get_name(handle, i + 1) for i in range(self.num_vars)]

    @property
    def num_zones(self) -> int:
        """Number of zones in the file.

        Queried directly. Cheaper than the base class's default of
        ``len(self.zones)``, which would build every zone's metadata.
        """
        return libtecio.tec_data_set_get_num_zones(self._check_handle())

    @property
    def zones(self) -> ZoneList[TecplotZoneReader]:
        """Zones in this file, by 0-based index or slice."""
        if self._zone_list is None:
            handle = self._check_handle()
            self._zone_list = ZoneList([
                _build_szl_zone(handle, i, self.num_vars) for i in range(self.num_zones)
            ])
        return self._zone_list

    @property
    def auxdata(self) -> TecplotAuxDataReader:
        """Dataset-level auxiliary data."""
        if self._dataset_auxdata is None:
            self._dataset_auxdata = TecplotSzlDatasetAuxDataReader(self._check_handle())
        return self._dataset_auxdata

    def _var_auxdata_at(self, var_index: int) -> TecplotAuxDataReader:
        return TecplotSzlVarAuxDataReader(self._check_handle(), var_index + 1)

    def close(self) -> None:
        """Close the file reader handle."""
        if self._handle is not None:
            libtecio.tec_file_reader_close(self._handle)
            self._handle = None

    def __repr__(self) -> str:  # pragma: no cover
        if self._handle is None:
            name = self._path.replace("\\", "/").rsplit("/", 1)[-1]
            return f"{type(self).__name__}(path={name!r}, <closed>)"
        return super().__repr__()
