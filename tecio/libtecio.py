"""Python bindings for the TecIO C library.

Ctypes bindings for both the new SZL API (``tec_*``) and the classic PLT API
(``tec*142``), and a Python wrapper function for each C entry point.  The TecIO library
often uses integers with special meanings (zone types, data types, data locations). The
same values are used both for writing and for SZL reading and writing. Enum classes with
descriptive names are defined in :mod:`tecio._constants` and imported here.

Notes:
    - The ctypes bindings are defined conditionall if the symbol exists in the loaded
      TecIO shared object binary such that older versions not containing newer (often
      niche) functions such as :func:`~tecio.libtecio.tecznefemixed142` do not cause the
      entire `tecio` Python package to fail to load.
    - Only loading the available symbols also allows for the `tecio` Python package to
      be used without any TecIO shared object binary available as it is only needed for
      SZL read/write and PLT write. DAT read/write and PLT read have pure Python
      implementations and do not have external dependencies other than NumPy.
    - The load/bind/guard mechanism is defined in :mod:`tecio._utils`
    - After binding, a one-line summary is emitted via :mod:`warnings` (category
      :class:`TecioAvailabilityWarning`) if anything is disabled and is available as
      data via :data:`LIBRARY_LOAD_ERROR` and :data:`UNAVAILABLE_FUNCTIONS`.
"""

from __future__ import annotations

import ctypes
import functools
import warnings
from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _utils
from ._constants import (
    Boolean,
    DataPacking,
    DataType,
    Debug,
    FaceNeighborMode,
    FeCellShape,
    FileFormat,
    FileType,
    ValueLocation,
    VarStatus,
    ZoneType,
)

__all__ = [
    "TecioError",
    "TecioUnavailableError",
    "TecioAvailabilityWarning",
    "LIBRARY_LOAD_ERROR",
    "LIBRARY_PATH",
    "UNAVAILABLE_FUNCTIONS",
    "tec_file_reader_open",
    "tec_file_reader_close",
    "tec_file_get_type",
    "tec_data_set_get_title",
    "tec_data_set_get_num_vars",
    "tec_data_set_get_num_zones",
    "tec_zone_get_ijk",
    "tec_zone_get_title",
    "tec_zone_get_type",
    "tec_zone_get_num_sections",
    "tec_zone_get_section_metrics",
    "tec_zone_is_enabled",
    "tec_zone_get_solution_time",
    "tec_zone_get_strand_id",
    "is_64bit",
    "tec_zone_node_map_get_64",
    "tec_zone_node_map_get",
    "tec_zone_face_nbr_get_mode",
    "tec_zone_face_nbr_get_num_connections",
    "tec_zone_face_nbr_get_num_values",
    "tec_zone_face_nbrs_are_64bit",
    "tec_zone_face_nbr_get_connections",
    "tec_zone_face_nbr_get_connections_64",
    "tec_var_get_name",
    "tec_var_is_enabled",
    "tec_zone_var_get_type",
    "tec_zone_var_get_value_location",
    "tec_zone_var_is_passive",
    "tec_zone_var_get_shared_zone",
    "tec_zone_connectivity_get_shared_zone",
    "tec_zone_var_get_num_values",
    "tec_zone_var_get_float_values",
    "tec_zone_var_get_double_values",
    "tec_zone_var_get_int32_values",
    "tec_zone_var_get_int16_values",
    "tec_zone_var_get_uint8_values",
    "tec_data_set_aux_data_get_num_items",
    "tec_data_set_aux_data_get_item",
    "tec_var_aux_data_get_num_items",
    "tec_var_aux_data_get_item",
    "tec_zone_aux_data_get_num_items",
    "tec_zone_aux_data_get_item",
    "tec_file_writer_open",
    "tec_file_writer_close",
    "tec_file_writer_flush",
    "tec_zone_create_ijk",
    "tec_zone_create_fe",
    "tec_zone_create_fe_mixed",
    "tec_zone_set_unsteady_options",
    "tec_data_set_add_aux_data",
    "tec_var_add_aux_data",
    "tec_zone_add_aux_data",
    "tec_zone_var_write_double_values",
    "tec_zone_var_write_float_values",
    "tec_zone_var_write_int32_values",
    "tec_zone_var_write_int16_values",
    "tec_zone_var_write_uint8_values",
    "tec_zone_node_map_write32",
    "tec_zone_node_map_write64",
    "tec_zone_face_nbr_write_connections32",
    "tec_zone_face_nbr_write_connections64",
    "tecini142",
    "tecend142",
    "tecflush142",
    "tecfil142",
    "tecforeign142",
    "teczne142",
    "tecpolyzne142",
    "tecznefemixed142",
    "tecdat142",
    "tecnode142",
    "tecface142",
    "tecpolyface142",
    "tecpolybconn142",
    "tecauxstr142",
    "tecvauxstr142",
    "teczauxstr142",
    "tecusr142",
]


# ======================================================================================
# Error classes
# ======================================================================================


class TecioError(RuntimeError):
    """Exception for TecIO C library errors."""


class TecioUnavailableError(TecioError):
    """Raised when calling a function whose C symbol is unavailable.

    Either because no TecIO shared library could be located at all, or
    because the library that was found doesn't export this particular
    symbol (typically an older Tecplot version, predating that entry
    point).
    """


class TecioAvailabilityWarning(UserWarning):
    """Emitted once at import time if any C function could not be bound."""


_lib, _load_error, _library_path = _utils.load_library()

lib = _lib

# Exception raised while trying to locate/load the shared library, or None
# if it loaded successfully
LIBRARY_LOAD_ERROR: Exception | None = _load_error

# Path to the shared library that was loaded, or None if none could be
# located/loaded (see :data:`LIBRARY_LOAD_ERROR`).
LIBRARY_PATH: str | None = _library_path

# C function names (e.g. ``"tecFileReaderOpen"``) whose symbol could not be
# bound, either because no library loaded at all or because the loaded
# library doesn't export them
UNAVAILABLE_FUNCTIONS: set[str] = set()

# Bind one C function's ``restype``/``argtypes`` on this module's ``lib``, pre-applied
# against this module's own ``lib``/``UNAVAILABLE_FUNCTIONS`` so each of the ~75 call
# sites below only needs to supply ``name``, ``restype``, and ``argtypes``. See
# :func:`tecio._utils.bind`.
bind_ctypes = functools.partial(_utils.bind, lib=lib, unavailable=UNAVAILABLE_FUNCTIONS)

# Decorator guarding a wrapper function behind its required C symbol, pre-applied
# against this module's own state and :exc:`TecioUnavailableError`
requires_symbol = functools.partial(
    _utils.requires_symbol,
    unavailable=UNAVAILABLE_FUNCTIONS,
    load_error=LIBRARY_LOAD_ERROR,
    library_path=LIBRARY_PATH,
    exception_cls=TecioUnavailableError,
)


# ======================================================================================
# Local helpers
# ======================================================================================


def _prepare_array_for_ctypes(
    values: npt.ArrayLike, np_dtype, ctype
) -> tuple[Any, int, npt.NDArray]:
    """Convert an input array to contiguous numpy array and return a ctypes pointer.

    Args:
        values: array-like (list, tuple, numpy array)
        np_dtype: numpy dtype object or type (e.g. np.float32)
        ctype: corresponding ctypes scalar type (e.g. ctypes.c_float)

    Returns:
        ptr: ctypes pointer suitable for passing to the C API
        count: int number of elements
        backing_array: the numpy array object (returned to keep it alive)

    Note:
        Caller should keep the returned backing_array alive until the native call
        completes.

    Note:
        This function enforces dtype and C-contiguity.
    """
    arr = np.ascontiguousarray(values, dtype=np_dtype)
    count = int(arr.size)
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctype))
    return ptr, count, arr


def _to_int_value(value: int | Enum, enum_class: type[Enum] | None = None) -> int:
    """Convert Enum or int to int value.

    (Optional check against enum_class if provided.)
    """
    if isinstance(value, Enum):
        return value.value
    v = int(value)
    if enum_class is not None:
        try:
            enum_class(v)
        except ValueError as e:
            raise ValueError(
                f"Invalid value {v} for enum {enum_class.__name__}. "
                f"Valid values: {[e.value for e in enum_class]}"
            ) from e
    return v


def _process_sequence(seq: Sequence[int | Enum] | None) -> ctypes.Array | None:
    """Convert sequence of int/Enum to ctypes array, handling None."""
    if seq is None:
        return None
    values = [_to_int_value(v) for v in seq]
    return (ctypes.c_int32 * len(values))(*values)


def _decode(value: bytes | None) -> str:
    """Decode a ctypes c_char_p value to str, raising if unexpectedly null."""
    if value is None:
        raise ValueError("Unexpected null pointer returned from TecIO C library.")
    return value.decode("utf-8")


# ======================================================================================
# C library bindings
# - Set the input and output C types for each C function used in this library
# ======================================================================================

# --------------------------------------------------------------------------------------
# New SZL API bindings
# --------------------------------------------------------------------------------------

# Reading SZL files
bind_ctypes(
    name="tecFileReaderOpen",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ],
)
bind_ctypes(
    name="tecFileReaderClose",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_void_p),
    ],
)
bind_ctypes(
    name="tecFileGetType",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecDataSetGetTitle",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char_p),
    ],
)
bind_ctypes(
    name="tecDataSetGetNumVars",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecDataSetGetNumZones",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecDataSetAuxDataGetNumItems",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
    ],
)

# Reading SZL zones
bind_ctypes(
    name="tecZoneGetIJK",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ],
)
bind_ctypes(
    name="tecZoneGetTitle",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_char_p),
    ],
)
bind_ctypes(
    name="tecZoneGetType",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecZoneGetNumSections",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecZoneGetSectionMetrics",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,  # zone
        ctypes.c_int32,  # section
        ctypes.POINTER(ctypes.c_int32),  # cellShape
        ctypes.POINTER(ctypes.c_int32),  # gridOrder
        ctypes.POINTER(ctypes.c_int32),  # basisFunction
        ctypes.POINTER(ctypes.c_int64),  # numElemsInSection
        ctypes.POINTER(ctypes.c_int32),  # numNodesPerCell
    ],
)
bind_ctypes(
    name="tecZoneIsEnabled",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecZoneGetSolutionTime",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
    ],
)
bind_ctypes(
    name="tecZoneGetStrandID",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecZoneNodeMapIs64Bit",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecZoneNodeMapGet64",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
    ],
)
bind_ctypes(
    name="tecZoneNodeMapGet",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
    ],
)

# Reading SZL face neighbors
bind_ctypes(
    name="tecZoneFaceNbrGetMode",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecZoneFaceNbrGetNumConnections",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
    ],
)
bind_ctypes(
    name="tecZoneFaceNbrGetNumValues",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
    ],
)
bind_ctypes(
    name="tecZoneFaceNbrsAre64Bit",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecZoneFaceNbrGetConnections",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecZoneFaceNbrGetConnections64",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
    ],
)

# Reading SZL variable data
bind_ctypes(
    name="tecVarGetName",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_char_p),
    ],
)
bind_ctypes(
    name="tecVarIsEnabled",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecZoneVarGetType",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecZoneVarGetValueLocation",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecZoneVarIsPassive",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecZoneVarGetSharedZone",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecZoneConnectivityGetSharedZone",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecZoneVarGetNumValues",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecZoneVarGetFloatValues",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # File Handle
        ctypes.c_int32,  # Zone index
        ctypes.c_int32,  # Variable index
        ctypes.c_int64,  # Start index
        ctypes.c_int64,  # Number of values
        ctypes.POINTER(ctypes.c_float),  # Values
    ],
)
bind_ctypes(
    name="tecZoneVarGetDoubleValues",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # File Handle
        ctypes.c_int32,  # Zone index
        ctypes.c_int32,  # Variable index
        ctypes.c_int64,  # Start index
        ctypes.c_int64,  # Number of values
        ctypes.POINTER(ctypes.c_double),  # Values
    ],
)
bind_ctypes(
    name="tecZoneVarGetInt32Values",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # File Handle
        ctypes.c_int32,  # Zone index
        ctypes.c_int32,  # Variable index
        ctypes.c_int64,  # Start index
        ctypes.c_int64,  # Number of values
        ctypes.POINTER(ctypes.c_int32),  # Values
    ],
)
bind_ctypes(
    name="tecZoneVarGetInt16Values",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # File Handle
        ctypes.c_int32,  # Zone index
        ctypes.c_int32,  # Variable index
        ctypes.c_int64,  # Start index
        ctypes.c_int64,  # Number of values
        ctypes.POINTER(ctypes.c_int16),  # Values
    ],
)
bind_ctypes(
    name="tecZoneVarGetUInt8Values",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # File Handle
        ctypes.c_int32,  # Zone index
        ctypes.c_int32,  # Variable index
        ctypes.c_int64,  # Start index
        ctypes.c_int64,  # Number of values
        ctypes.POINTER(ctypes.c_uint8),  # Values
    ],
)

# Reading SZL aux data
bind_ctypes(
    name="tecDataSetAuxDataGetNumItems",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecDataSetAuxDataGetItem",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ],
)
bind_ctypes(
    name="tecVarAuxDataGetNumItems",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecVarAuxDataGetItem",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ],
)
bind_ctypes(
    name="tecZoneAuxDataGetNumItems",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ],
)
bind_ctypes(
    name="tecZoneAuxDataGetItem",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ],
)

# Output file initialization and file handling
bind_ctypes(
    name="tecFileWriterOpen",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_char_p,  # fileName
        ctypes.c_char_p,  # dataSetTitle
        ctypes.c_char_p,  # varNames (comma separated)
        ctypes.c_int32,  # useSZL (1)
        ctypes.c_int32,  # fileType
        ctypes.c_int32,  # reserved / options
        ctypes.c_void_p,  # gridFileHandle (optional)
        ctypes.POINTER(ctypes.c_void_p),  # out fileHandle
    ],
)
bind_ctypes(
    name="tecFileWriterClose",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_void_p),
    ],
)
bind_ctypes(
    name="tecFileWriterFlush",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # fileHandle
        ctypes.c_int32,  # numZonesToRetain
        ctypes.POINTER(ctypes.c_int32),  # zonesToRetain
    ],
)

# Write Zone Headers
bind_ctypes(
    name="tecZoneCreateIJK",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # file_handle
        ctypes.c_char_p,  # zoneTitle
        ctypes.c_int64,  # I
        ctypes.c_int64,  # J
        ctypes.c_int64,  # K
        ctypes.POINTER(ctypes.c_int32),  # varTypes
        ctypes.POINTER(ctypes.c_int32),  # shareVarFromZone
        ctypes.POINTER(ctypes.c_int32),  # valueLocations
        ctypes.POINTER(ctypes.c_int32),  # passiveVarList
        ctypes.c_int32,  # shareFaceNeighborsFromZone
        ctypes.c_int64,  # numFaceConnections
        ctypes.c_int32,  # faceNeighborMode
        ctypes.POINTER(ctypes.c_int32),  # out zone
    ],
)
bind_ctypes(
    name="tecZoneCreateFE",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # file_handle
        ctypes.c_char_p,  # zoneTitle
        ctypes.c_int32,  # ZoneType
        ctypes.c_int64,  # numNodes
        ctypes.c_int64,  # numCells
        ctypes.POINTER(ctypes.c_int32),  # varTypes
        ctypes.POINTER(ctypes.c_int32),  # shareVarFromZone
        ctypes.POINTER(ctypes.c_int32),  # valueLocations
        ctypes.POINTER(ctypes.c_int32),  # passiveVarList
        ctypes.c_int32,  # shareFaceNeighborsFromZone
        ctypes.c_int64,  # numFaceConnections
        ctypes.c_int32,  # faceNeighborMode
        ctypes.POINTER(ctypes.c_int32),  # out zone
    ],
)
bind_ctypes(
    name="tecZoneCreateFEMixed",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # file_handle
        ctypes.c_char_p,  # zoneTitle
        ctypes.c_int64,  # numNodes
        ctypes.c_int32,  # numSections
        ctypes.POINTER(ctypes.c_int32),  # cellShapePerSection
        ctypes.POINTER(ctypes.c_int32),  # gridOrderPerSection
        ctypes.POINTER(ctypes.c_int32),  # basisFnPerSection
        ctypes.POINTER(ctypes.c_int64),  # numElementsPerSection
        ctypes.POINTER(ctypes.c_int32),  # varTypes
        ctypes.POINTER(ctypes.c_int32),  # shareVarFromZone
        ctypes.POINTER(ctypes.c_int32),  # valueLocations
        ctypes.POINTER(ctypes.c_int32),  # passiveVarList
        ctypes.c_int32,  # shareConnectivityFromZone
        ctypes.c_int64,  # numFaceConnections
        ctypes.c_int32,  # faceNeighborMode
        ctypes.POINTER(ctypes.c_int32),  # out zone
    ],
)

# Optional fields
bind_ctypes(
    name="tecZoneSetUnsteadyOptions",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # file_handle
        ctypes.c_int32,  # zone
        ctypes.c_double,  # solutionTime
        ctypes.c_int32,  # strand
    ],
)
bind_ctypes(
    name="tecDataSetAddAuxData",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # fileHandle
        ctypes.c_char_p,  # name
        ctypes.c_char_p,  # value
    ],
)
bind_ctypes(
    name="tecVarAddAuxData",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # fileHandle
        ctypes.c_int32,  # varIndex (1-based)
        ctypes.c_char_p,  # name
        ctypes.c_char_p,  # value
    ],
)
bind_ctypes(
    name="tecZoneAddAuxData",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # fileHandle
        ctypes.c_int32,  # zoneIndex (1-based)
        ctypes.c_char_p,  # name
        ctypes.c_char_p,  # value
    ],
)

# Write variable value functions
bind_ctypes(
    name="tecZoneVarWriteDoubleValues",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # file handle
        ctypes.c_int32,  # zone index
        ctypes.c_int32,  # variable index
        ctypes.c_int32,  # partition index (0 for non-partitioned zones)
        ctypes.c_int64,  # number of values to write
        ctypes.POINTER(ctypes.c_double),  # pointer to values array
    ],
)
bind_ctypes(
    name="tecZoneVarWriteFloatValues",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # file handle
        ctypes.c_int32,  # zone index
        ctypes.c_int32,  # variable index
        ctypes.c_int32,  # partition index (0 for non-partitioned zones)
        ctypes.c_int64,  # number of values to write
        ctypes.POINTER(ctypes.c_float),  # pointer to values array
    ],
)
bind_ctypes(
    name="tecZoneVarWriteInt32Values",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # file handle
        ctypes.c_int32,  # zone index
        ctypes.c_int32,  # variable index
        ctypes.c_int32,  # partition index (0 for non-partitioned zones)
        ctypes.c_int64,  # number of values to write
        ctypes.POINTER(ctypes.c_int32),  # pointer to values array
    ],
)
bind_ctypes(
    name="tecZoneVarWriteInt16Values",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # file handle
        ctypes.c_int32,  # zone index
        ctypes.c_int32,  # variable index
        ctypes.c_int32,  # partition index (0 for non-partitioned zones)
        ctypes.c_int64,  # number of values to write
        ctypes.POINTER(ctypes.c_int16),  # pointer to values array
    ],
)
bind_ctypes(
    name="tecZoneVarWriteUInt8Values",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # file handle
        ctypes.c_int32,  # zone index
        ctypes.c_int32,  # variable index
        ctypes.c_int32,  # partition index (0 for non-partitioned zones)
        ctypes.c_int64,  # number of values to write
        ctypes.POINTER(ctypes.c_uint8),  # pointer to values array
    ],
)

# Write Zone Connectivity (FE zones only)
bind_ctypes(
    name="tecZoneNodeMapWrite32",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # fileHandle
        ctypes.c_int32,  # zone index(1-based)
        ctypes.c_int32,  # partition index (MPI)
        ctypes.c_int32,  # isOneBased (Boolean)
        ctypes.c_int64,  # nodeCount
        ctypes.POINTER(ctypes.c_int32),  # array of nodes
    ],
)
bind_ctypes(
    name="tecZoneNodeMapWrite64",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # fileHandle
        ctypes.c_int32,  # zone index(1-based)
        ctypes.c_int32,  # partition index (MPI)
        ctypes.c_int32,  # isOneBased (Boolean)
        ctypes.c_int64,  # nodeCount
        ctypes.POINTER(ctypes.c_int64),  # array of nodes
    ],
)
bind_ctypes(
    name="tecZoneFaceNbrWriteConnections32",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # fileHandle
        ctypes.c_int32,  # zone (1-based)
        ctypes.POINTER(ctypes.c_int32),  # faceNeighbors
    ],
)
bind_ctypes(
    name="tecZoneFaceNbrWriteConnections64",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_void_p,  # fileHandle
        ctypes.c_int32,  # zone (1-based)
        ctypes.POINTER(ctypes.c_int64),  # faceNeighbors
    ],
)

# --------------------------------------------------------------------------------------
# Classic API bindings
# --------------------------------------------------------------------------------------

# File initialization and finalization
bind_ctypes(
    name="tecini142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_char_p,  # Title
        ctypes.c_char_p,  # Variables
        ctypes.c_char_p,  # FName
        ctypes.c_char_p,  # ScratchDir
        ctypes.POINTER(ctypes.c_int32),  # FileFormat (0=PLT, 1=SZPLT)
        ctypes.POINTER(ctypes.c_int32),  # FileType (0=FULL, 1=GRID, 2=SOLUTION)
        ctypes.POINTER(ctypes.c_int32),  # Debug
        ctypes.POINTER(ctypes.c_int32),  # VIsDouble
    ],
)
bind_ctypes(
    name="tecend142",
    restype=ctypes.c_int32,
    argtypes=[],
)
bind_ctypes(
    name="tecflush142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_int32),  # NumZonesToRetain
        ctypes.POINTER(ctypes.c_int32),  # ZonesToRetain
    ],
)
bind_ctypes(
    name="tecfil142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_int32),  # OutputFileHandle
    ],
)
bind_ctypes(
    name="tecforeign142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_int32),  # OutputForeignByteOrder
    ],
)

# Zone creation
bind_ctypes(
    name="teczne142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_char_p,  # ZoneTitle
        ctypes.POINTER(ctypes.c_int32),  # ZoneType
        ctypes.POINTER(ctypes.c_int32),  # IMx (or NumNodes for FE)
        ctypes.POINTER(ctypes.c_int32),  # JMx (or NumElements for FE)
        ctypes.POINTER(ctypes.c_int32),  # KMx
        ctypes.POINTER(ctypes.c_int32),  # ICellMax
        ctypes.POINTER(ctypes.c_int32),  # JCellMax
        ctypes.POINTER(ctypes.c_int32),  # KCellMax
        ctypes.POINTER(ctypes.c_double),  # SolutionTime
        ctypes.POINTER(ctypes.c_int32),  # StrandID
        ctypes.POINTER(ctypes.c_int32),  # ParentZone
        ctypes.POINTER(ctypes.c_int32),  # IsBlock (1=Block, 0=Point)
        ctypes.POINTER(ctypes.c_int32),  # NumFaceConnections
        ctypes.POINTER(ctypes.c_int32),  # FaceNeighborMode
        ctypes.POINTER(ctypes.c_int32),  # TotalNumFaceNodes (for poly zones)
        ctypes.POINTER(ctypes.c_int32),  # NumConnectedBoundaryFaces (for poly)
        ctypes.POINTER(ctypes.c_int32),  # TotalNumBoundaryConnections (for poly)
        ctypes.POINTER(ctypes.c_int32),  # PassiveVarList
        ctypes.POINTER(ctypes.c_int32),  # ValueLocation
        ctypes.POINTER(ctypes.c_int32),  # ShareVarFromZone
        ctypes.POINTER(ctypes.c_int32),  # ShareConnectivityFromZone
    ],
)
bind_ctypes(
    name="tecpolyzne142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_char_p,  # ZoneTitle
        ctypes.POINTER(ctypes.c_int32),  # ZoneType (FEPOLYGON or FEPOLYHEDRON)
        ctypes.POINTER(ctypes.c_int32),  # NumNodes
        ctypes.POINTER(ctypes.c_int32),  # NumElements
        ctypes.POINTER(ctypes.c_int64),  # NumFaces
        ctypes.POINTER(ctypes.c_int64),  # TotalNumFaceNodes
        ctypes.POINTER(ctypes.c_double),  # SolutionTime
        ctypes.POINTER(ctypes.c_int32),  # StrandID
        ctypes.POINTER(ctypes.c_int32),  # ParentZone
        ctypes.POINTER(ctypes.c_int32),  # NumConnectedBoundaryFaces
        ctypes.POINTER(ctypes.c_int32),  # TotalNumBoundaryConnections
        ctypes.POINTER(ctypes.c_int32),  # PassiveVarList
        ctypes.POINTER(ctypes.c_int32),  # ValueLocation
        ctypes.POINTER(ctypes.c_int32),  # ShareVarFromZone
        ctypes.POINTER(ctypes.c_int32),  # ShareConnectivityFromZone
    ],
)
bind_ctypes(
    name="tecznefemixed142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_char_p,  # ZoneTitle
        ctypes.POINTER(ctypes.c_int64),  # NumNodes
        ctypes.POINTER(ctypes.c_int32),  # NumSections
        ctypes.POINTER(ctypes.c_int32),  # CellShapePerSection
        ctypes.POINTER(ctypes.c_int32),  # GridOrderPerSection
        ctypes.POINTER(ctypes.c_int32),  # BasisFnPerSection
        ctypes.POINTER(ctypes.c_int64),  # NumElementsPerSection
        ctypes.POINTER(ctypes.c_double),  # SolutionTime
        ctypes.POINTER(ctypes.c_int32),  # StrandID
        ctypes.POINTER(ctypes.c_int32),  # NumFaceConnections
        ctypes.POINTER(ctypes.c_int32),  # FaceNeighborMode
        ctypes.POINTER(ctypes.c_int32),  # PassiveVarList
        ctypes.POINTER(ctypes.c_int32),  # ValueLocation
        ctypes.POINTER(ctypes.c_int32),  # ShareVarFromZone
        ctypes.POINTER(ctypes.c_int32),  # ShareConnectivityFromZone
    ],
)

# Partitioned zone creation
bind_ctypes(
    name="tecijkptn142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_int32),  # PartitionOwnerZone
        ctypes.POINTER(ctypes.c_int32),  # IMin
        ctypes.POINTER(ctypes.c_int32),  # JMin
        ctypes.POINTER(ctypes.c_int32),  # KMin
        ctypes.POINTER(ctypes.c_int32),  # IMax
        ctypes.POINTER(ctypes.c_int32),  # JMax
        ctypes.POINTER(ctypes.c_int32),  # KMax
    ],
)
bind_ctypes(
    name="tecfeptn142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_int32),  # PartitionOwnerZone
        ctypes.POINTER(ctypes.c_int32),  # NumNodes
        ctypes.POINTER(ctypes.c_int32),  # NumElements
    ],
)
bind_ctypes(
    name="tecfemixedptn142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_int32),  # PartitionOwnerZone
        ctypes.POINTER(ctypes.c_int32),  # NumNodes
        ctypes.POINTER(ctypes.c_int32),  # NumElements
        ctypes.POINTER(ctypes.c_int32),  # NumNodesPerElement
    ],
)

# Data writing
bind_ctypes(
    name="tecdat142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_int32),  # N (number of values)
        ctypes.c_void_p,  # FieldData (void pointer for flexibility)
        ctypes.POINTER(ctypes.c_int32),  # IsDouble (1=double, 0=float)
    ],
)

# Connectivity writing
bind_ctypes(
    name="tecnod142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_int32),  # NData (connectivity array)
    ],
)
bind_ctypes(
    name="tecnode142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_int32),  # N (number of values)
        ctypes.POINTER(ctypes.c_int32),  # NData (connectivity array)
    ],
)
bind_ctypes(
    name="tecznemap142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_int32),  # N (number of values)
        ctypes.POINTER(ctypes.c_int32),  # NodeMap
    ],
)
bind_ctypes(
    name="tecface142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_int32),  # FaceConnections
    ],
)
bind_ctypes(
    name="tecpolyface142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_int32),  # NumFaces
        ctypes.POINTER(ctypes.c_int32),  # FaceNodeCounts
        ctypes.POINTER(ctypes.c_int32),  # FaceNodes
        ctypes.POINTER(ctypes.c_int32),  # FaceLeftElems
        ctypes.POINTER(ctypes.c_int32),  # FaceRightElems
    ],
)
bind_ctypes(
    name="tecpolybconn142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_int32),  # NumBoundaryFaces
        ctypes.POINTER(ctypes.c_int32),  # BoundaryConnectionCounts
        ctypes.POINTER(ctypes.c_int32),  # BoundaryConnectionElems
        ctypes.POINTER(ctypes.c_int32),  # BoundaryConnectionZones
    ],
)

# Auxiliary data
bind_ctypes(
    name="tecauxstr142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_char_p,  # Name
        ctypes.c_char_p,  # Value
    ],
)

bind_ctypes(
    name="tecvauxstr142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_int32),  # Var (1-based variable index)
        ctypes.c_char_p,  # Name
        ctypes.c_char_p,  # Value
    ],
)

bind_ctypes(
    name="teczauxstr142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_char_p,  # Name
        ctypes.c_char_p,  # Value
    ],
)

# MPI initialization (for parallel I/O)
bind_ctypes(
    name="tecmpiinit142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.POINTER(ctypes.c_int32),  # Communicator
        ctypes.POINTER(ctypes.c_int32),  # MainRank
    ],
)

# User-defined data (custom records)
bind_ctypes(
    name="tecusr142",
    restype=ctypes.c_int32,
    argtypes=[
        ctypes.c_char_p,  # UserRec
    ],
)

if UNAVAILABLE_FUNCTIONS:  # pragma: no cover - environment-dependent
    if LIBRARY_LOAD_ERROR is not None:
        _reason = f"no TecIO shared library could be loaded ({LIBRARY_LOAD_ERROR})"
    else:
        _reason = f"the loaded library ({LIBRARY_PATH}) doesn't export them"
    warnings.warn(
        f"tecio.libtecio: {len(UNAVAILABLE_FUNCTIONS)} function(s) unavailable, "
        f"{_reason}. Calling one of these raises TecioUnavailableError; see "
        "tecio.libtecio.UNAVAILABLE_FUNCTIONS for the full list. DAT reading and "
        "writing is unaffected, it does not use this library. Affected: "
        + ", ".join(sorted(UNAVAILABLE_FUNCTIONS)),
        TecioAvailabilityWarning,
        stacklevel=2,
    )


# ======================================================================================
# Wrappers for C functions
# - 1-to-1 python functions to format python inputs/ouputs to C TecIO functions
# - "New" API functions (tecXxxXxx) are wrapped by equivaltly named tec_xxx_xxx
#   python funcitons
# - "Classic" API functions (TECXXXX142) wrapped by equivaltly named tecxxx142 python
#   funcitons
# - Scope for this library is limited to IO for data related file records (geometry and
#   text records are ignored)
# - Some MPI funcion python wrappers are included, but not yet fully implemented
# - Wherever data arrays are output, conversion to numpy arrays is handled in the
#   wrapper function
# ======================================================================================

# --------------------------------------------------------------------------------------
# New read API (SZL/.szplt):
# - tec_file_reader_open returns an explicit file handle that is passed to every
#   subsequent call, making the target file unambiguous
# - Multiple files can be read simultaneously by holding multiple handles at once
# - Fucntions are available to query dataset/zone/variable metadata before reading any
#   data, and data can be read in any order by referencing the file handle +
#   zone/variable index
# - General organization of functions:
#   - tec_file_* functions operate on the file
#   - tec_data_set_* functions query whole dataset metadata
#   - tec_zone_* functions query zone level metadata (type, dimensions, title, solution
#     time, strand ID, node map, aux data)
#   - tec_var_* functions query variable-level metadata (name, aux data)
#   - tec_zone_var_* functions read variable values for a given zone and variable index
# --------------------------------------------------------------------------------------


# -- Reading SZL files -----------------------------------------------------------------


@requires_symbol("tecFileReaderOpen")
def tec_file_reader_open(file_name: str) -> ctypes.c_void_p:
    """Open an SZL file for reading.

    Args:
        file_name (str): Path to the ``.szplt`` file.

    Returns:
        Opaque file handle for subsequent TecIO calls.

    Raises:
        TecioError: If the file cannot be opened.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    handle = ctypes.c_void_p(0)

    ret = lib.tecFileReaderOpen(
        ctypes.c_char_p(bytes(file_name, encoding="UTF-8")),
        ctypes.byref(handle),
    )
    if ret != 0:
        raise TecioError(
            f"SzlFile Initialization Error: file_name={file_name}, return_code={ret}"
        )

    return handle


@requires_symbol("tecFileReaderClose")
def tec_file_reader_close(handle: ctypes.c_void_p) -> None:
    """Close an SZL file reader handle and release its resources.

    Args:
        handle: File handle from :func:`tec_file_reader_open`.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ret = lib.tecFileReaderClose(ctypes.byref(handle))
    if ret != 0:
        raise TecioError(
            f"tecFileReaderClose Error: handle={handle}, return_code={ret}"
        )


@requires_symbol("tecFileGetType")
def tec_file_get_type(handle: ctypes.c_void_p) -> FileType:
    """Get the file type for an opened SZL file.

    Args:
        handle (ctypes.c_void_p): File handle from :func:`tec_file_reader_open`.

    Returns:
        :class:`FileType` enum (FULL, GRID, or SOLUTION).

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    file_type = ctypes.c_int32(0)

    ret = lib.tecFileGetType(handle, ctypes.byref(file_type))
    if ret != 0:
        raise TecioError(f"Error getting file type: handle:{handle}, return_code={ret}")

    return FileType(file_type.value)


@requires_symbol("tecDataSetGetTitle")
def tec_data_set_get_title(handle: ctypes.c_void_p) -> str:
    """Read the dataset title string.

    Args:
        handle (ctypes.c_void_p): File handle.

    Returns:
        UTF-8 decoded dataset title.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    title = ctypes.c_char_p(0)

    ret = lib.tecDataSetGetTitle(handle, ctypes.byref(title))
    if ret != 0:
        raise TecioError(
            f"Error getting data set title: handle={handle}, return_code={ret}"
        )

    return _decode(title.value)


@requires_symbol("tecDataSetGetNumVars")
def tec_data_set_get_num_vars(handle: ctypes.c_void_p) -> int:
    """Query the number of variables in the dataset.

    Args:
        handle (ctypes.c_void_p): File handle.

    Returns:
        Number of variables.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    num_vars = ctypes.c_int32(0)

    ret = lib.tecDataSetGetNumVars(handle, ctypes.byref(num_vars))
    if ret != 0:
        raise TecioError(
            f"Error getting number of variables: handle={handle}, return_code={ret}"
        )

    return num_vars.value


@requires_symbol("tecDataSetGetNumZones")
def tec_data_set_get_num_zones(handle: ctypes.c_void_p) -> int:
    """Query the number of zones in the dataset.

    Args:
        handle (ctypes.c_void_p): File handle.

    Returns:
        Number of zones.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    num_zones = ctypes.c_int32(0)

    ret = lib.tecDataSetGetNumZones(handle, ctypes.byref(num_zones))
    if ret != 0:
        raise TecioError(
            f"Error getting number of zones: handle={handle}, return_code={ret}"
        )

    return num_zones.value


# -- Reading SZL zones -----------------------------------------------------------------


@requires_symbol("tecZoneGetIJK")
def tec_zone_get_ijk(handle: ctypes.c_void_p, zone_index: int) -> tuple[int, int, int]:
    """Get zone dimensions (ORDERED) or node/element counts (FE).

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        ``(I, J, K)`` for ORDERED zones, or ``(num_nodes, num_elements, 0)``
        for FE zones.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    imax = ctypes.c_int64(0)
    jmax = ctypes.c_int64(0)
    kmax = ctypes.c_int64(0)

    ret = lib.tecZoneGetIJK(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.byref(imax),
        ctypes.byref(jmax),
        ctypes.byref(kmax),
    )
    if ret != 0:
        raise TecioError(
            f"Error getting zone data indices: : handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return imax.value, jmax.value, kmax.value


@requires_symbol("tecZoneGetTitle")
def tec_zone_get_title(handle: ctypes.c_void_p, zone_index: int) -> str:
    """Read the title for a zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        Zone title string.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    zone_title = ctypes.c_char_p(0)

    ret = lib.tecZoneGetTitle(
        handle, ctypes.c_int32(zone_index), ctypes.byref(zone_title)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneGetTitle Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return _decode(zone_title.value)


@requires_symbol("tecZoneGetType")
def tec_zone_get_type(handle: ctypes.c_void_p, zone_index: int) -> ZoneType:
    """Query the zone type.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        :class:`ZoneType` enum value.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    zone_type = ctypes.c_int32(0)

    ret = lib.tecZoneGetType(
        handle, ctypes.c_int32(zone_index), ctypes.byref(zone_type)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneGetType Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return ZoneType(zone_type.value)


@requires_symbol("tecZoneGetNumSections")
def tec_zone_get_num_sections(handle: ctypes.c_void_p, zone_index: int) -> int:
    """Get the number of sections in a mixed finite-element zone.

    Only meaningful for zones of :attr:`ZoneType.FEMIXED`.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        Number of sections (1-16) in the zone.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    num_sections = ctypes.c_int32(0)

    ret = lib.tecZoneGetNumSections(
        handle, ctypes.c_int32(zone_index), ctypes.byref(num_sections)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneGetNumSections Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return num_sections.value


@requires_symbol("tecZoneGetSectionMetrics")
def tec_zone_get_section_metrics(
    handle: ctypes.c_void_p, zone_index: int, section_index: int
) -> tuple[FeCellShape, int, int, int, int]:
    """Get metrics for one section of a mixed finite-element zone.

    Only meaningful for zones of :attr:`ZoneType.FEMIXED`; use
    :func:`tec_zone_get_num_sections` first to get the section count.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        section_index (int): 1-based section index.

    Returns:
        ``(cell_shape, grid_order, basis_function, num_elements,
        num_nodes_per_cell)`` for the requested section.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    cell_shape = ctypes.c_int32(0)
    grid_order = ctypes.c_int32(0)
    basis_function = ctypes.c_int32(0)
    num_elements = ctypes.c_int64(0)
    num_nodes_per_cell = ctypes.c_int32(0)

    ret = lib.tecZoneGetSectionMetrics(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(section_index),
        ctypes.byref(cell_shape),
        ctypes.byref(grid_order),
        ctypes.byref(basis_function),
        ctypes.byref(num_elements),
        ctypes.byref(num_nodes_per_cell),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneGetSectionMetrics Error: handle={handle}, "
            f"zone_index={zone_index}, section_index={section_index}, "
            f"return_code={ret}"
        )

    return (
        FeCellShape(cell_shape.value),
        grid_order.value,
        basis_function.value,
        num_elements.value,
        num_nodes_per_cell.value,
    )


@requires_symbol("tecZoneIsEnabled")
def tec_zone_is_enabled(handle: ctypes.c_void_p, zone_index: int) -> bool:
    """Check whether a zone is enabled.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        True if the zone is enabled.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    is_enabled = ctypes.c_int32(0)

    ret = lib.tecZoneIsEnabled(
        handle, ctypes.c_int32(zone_index), ctypes.byref(is_enabled)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneIsEnabled Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return bool(is_enabled.value)


@requires_symbol("tecZoneGetSolutionTime")
def tec_zone_get_solution_time(handle: ctypes.c_void_p, zone_index: int) -> float:
    """Read the solution time for a zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        Solution time (double precision).

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    solution_time = ctypes.c_double(0)

    ret = lib.tecZoneGetSolutionTime(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.byref(solution_time),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneGetSolutionTime Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return solution_time.value


@requires_symbol("tecZoneGetStrandID")
def tec_zone_get_strand_id(handle: ctypes.c_void_p, zone_index: int) -> int:
    """Get the strand ID for a zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        Strand ID integer.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    strand_id = ctypes.c_int32(0)

    ret = lib.tecZoneGetStrandID(
        handle, ctypes.c_int32(zone_index), ctypes.byref(strand_id)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneGetStrandID Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return strand_id.value


@requires_symbol("tecZoneNodeMapIs64Bit")
def is_64bit(handle: ctypes.c_void_p, zone_index: int) -> bool:
    """Check whether a zone's node-map uses 64-bit indices.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        True if 64-bit, False if 32-bit.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    is64bit = ctypes.c_int32(0)
    ret = lib.tecZoneNodeMapIs64Bit(
        handle, ctypes.c_int32(zone_index), ctypes.byref(is64bit)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneNodeMapIs64Bit Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return bool(is64bit.value)


@requires_symbol("tecZoneNodeMapGet64")
def tec_zone_node_map_get_64(
    handle: ctypes.c_void_p,
    zone_index: int,
    num_elements: int,
    nodes_per_cell: int,
) -> npt.NDArray[np.int64]:
    """Read a 64-bit node map for an FE zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        num_elements (int): Number of elements to read.
        nodes_per_cell (int): Nodes per element.

    Returns:
        Array of shape ``(num_elements, nodes_per_cell)`` with dtype int64.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    size_of_array = num_elements * nodes_per_cell
    nodemap = (ctypes.c_int64 * size_of_array)()

    ret = lib.tecZoneNodeMapGet64(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int64(1),
        ctypes.c_int64(num_elements),
        ctypes.cast(
            nodemap,
            ctypes.POINTER(ctypes.c_int64),
        ),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneNodeMapGet64 Error: handle={handle}, "
            f"zone_index={zone_index}, num_elements={num_elements}, "
            f"nodes_per_cell={nodes_per_cell}, return_code={ret}"
        )

    return np.ctypeslib.as_array(nodemap).reshape(num_elements, nodes_per_cell)


@requires_symbol("tecZoneNodeMapGet")
def tec_zone_node_map_get(
    handle: ctypes.c_void_p,
    zone_index: int,
    num_elements: int,
    nodes_per_cell: int,
) -> npt.NDArray[np.int32]:
    """Read a 32-bit node map for an FE zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        num_elements (int): Number of elements to read.
        nodes_per_cell (int): Nodes per element.

    Returns:
        Array of shape ``(num_elements, nodes_per_cell)`` with dtype int32.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    size_of_array = num_elements * nodes_per_cell
    nodemap = (ctypes.c_int32 * size_of_array)()

    ret = lib.tecZoneNodeMapGet(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int64(1),
        ctypes.c_int64(num_elements),
        ctypes.cast(
            nodemap,
            ctypes.POINTER(ctypes.c_int32),
        ),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneNodeMapGet Error: handle={handle}, "
            f"zone_index={zone_index}, num_elements={num_elements}, "
            f"nodes_per_cell={nodes_per_cell}, return_code={ret}"
        )

    return np.ctypeslib.as_array(nodemap).reshape(num_elements, nodes_per_cell)


@requires_symbol("tecZoneFaceNbrGetMode")
def tec_zone_face_nbr_get_mode(
    handle: ctypes.c_void_p, zone_index: int
) -> FaceNeighborMode:
    """Get the face-neighbor mode for a zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        :class:`FaceNeighborMode` enum value.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    mode = ctypes.c_int32(0)

    ret = lib.tecZoneFaceNbrGetMode(
        handle, ctypes.c_int32(zone_index), ctypes.byref(mode)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneFaceNbrGetMode Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return FaceNeighborMode(mode.value)


@requires_symbol("tecZoneFaceNbrGetNumConnections")
def tec_zone_face_nbr_get_num_connections(
    handle: ctypes.c_void_p, zone_index: int
) -> int:
    """Get the number of face-neighbor connections for a zone.

    This is a count of distinct connections, not a flat value count, see
    :func:`tec_zone_face_nbr_get_num_values` for the latter. How many raw
    values make up one connection depends on
    :func:`tec_zone_face_nbr_get_mode` (e.g. 3 per connection for
    ``LOCAL_ONE_TO_ONE``).

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        Number of face-neighbor connections, 0 if the zone has none.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    num_connections = ctypes.c_int64(0)

    ret = lib.tecZoneFaceNbrGetNumConnections(
        handle, ctypes.c_int32(zone_index), ctypes.byref(num_connections)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneFaceNbrGetNumConnections Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return num_connections.value


@requires_symbol("tecZoneFaceNbrGetNumValues")
def tec_zone_face_nbr_get_num_values(handle: ctypes.c_void_p, zone_index: int) -> int:
    """Get the number of face-neighbor connection values for a zone.

    This is the flat length of the array returned by
    :func:`tec_zone_face_nbr_get_connections` (or its 64-bit counterpart),
    not a count of connections, the number of values per connection depends
    on :func:`tec_zone_face_nbr_get_mode`.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        Number of face-neighbor connection values, 0 if the zone has none.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    num_values = ctypes.c_int64(0)

    ret = lib.tecZoneFaceNbrGetNumValues(
        handle, ctypes.c_int32(zone_index), ctypes.byref(num_values)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneFaceNbrGetNumValues Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return num_values.value


@requires_symbol("tecZoneFaceNbrsAre64Bit")
def tec_zone_face_nbrs_are_64bit(handle: ctypes.c_void_p, zone_index: int) -> bool:
    """Check whether a zone's face-neighbor connections use 64-bit indices.

    Determines whether to call :func:`tec_zone_face_nbr_get_connections` or
    :func:`tec_zone_face_nbr_get_connections64`.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        True if connections are 64-bit, False if 32-bit.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    is_64bit = ctypes.c_int32(0)

    ret = lib.tecZoneFaceNbrsAre64Bit(
        handle, ctypes.c_int32(zone_index), ctypes.byref(is_64bit)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneFaceNbrsAre64Bit Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return bool(is_64bit.value)


@requires_symbol("tecZoneFaceNbrGetConnections")
def tec_zone_face_nbr_get_connections(
    handle: ctypes.c_void_p, zone_index: int, num_values: int
) -> npt.NDArray[np.int32]:
    """Read 32-bit face-neighbor connections for a zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        num_values (int): Flat value count, from
            :func:`tec_zone_face_nbr_get_num_values`.

    Returns:
        Flat array of face-neighbor connection values, dtype int32. Group
        into individual connections according to
        :func:`tec_zone_face_nbr_get_mode`.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    connections = (ctypes.c_int32 * num_values)()

    ret = lib.tecZoneFaceNbrGetConnections(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.cast(connections, ctypes.POINTER(ctypes.c_int32)),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneFaceNbrGetConnections Error: handle={handle}, "
            f"zone_index={zone_index}, num_values={num_values}, "
            f"return_code={ret}"
        )

    return np.ctypeslib.as_array(connections)


@requires_symbol("tecZoneFaceNbrGetConnections64")
def tec_zone_face_nbr_get_connections_64(
    handle: ctypes.c_void_p, zone_index: int, num_values: int
) -> npt.NDArray[np.int64]:
    """Read 64-bit face-neighbor connections for a zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        num_values (int): Flat value count, from
            :func:`tec_zone_face_nbr_get_num_values`.

    Returns:
        Flat array of face-neighbor connection values, dtype int64. Group
        into individual connections according to
        :func:`tec_zone_face_nbr_get_mode`.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    connections = (ctypes.c_int64 * num_values)()

    ret = lib.tecZoneFaceNbrGetConnections64(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.cast(connections, ctypes.POINTER(ctypes.c_int64)),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneFaceNbrGetConnections64 Error: handle={handle}, "
            f"zone_index={zone_index}, num_values={num_values}, "
            f"return_code={ret}"
        )

    return np.ctypeslib.as_array(connections)


# -- Reading SZL variable data ---------------------------------------------------------


@requires_symbol("tecVarGetName")
def tec_var_get_name(handle: ctypes.c_void_p, var_index: int) -> str:
    """Get a variable name by index.

    Args:
        handle (ctypes.c_void_p): File handle.
        var_index (int): 1-based variable index.

    Returns:
        Variable name string.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    var_name = ctypes.c_char_p(0)

    ret = lib.tecVarGetName(handle, ctypes.c_int32(var_index), ctypes.byref(var_name))
    if ret != 0:
        raise TecioError(
            f"tecVarGetName Error: handle={handle}, "
            f"var_index={var_index}, return_code={ret}"
        )

    return _decode(var_name.value)


@requires_symbol("tecVarIsEnabled")
def tec_var_is_enabled(handle: ctypes.c_void_p, var_index: int) -> bool:
    """Check whether a variable is enabled.

    Args:
        handle (ctypes.c_void_p): File handle.
        var_index (int): 1-based variable index.

    Returns:
        True if the variable is enabled.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    is_enabled = ctypes.c_int32(0)

    ret = lib.tecVarIsEnabled(
        handle, ctypes.c_int32(var_index), ctypes.byref(is_enabled)
    )
    if ret != 0:
        raise TecioError(
            f"tecVarIsEnabled Error: handle={handle}, "
            f"var_index={var_index}, return_code={ret}"
        )

    return bool(is_enabled.value)


@requires_symbol("tecZoneVarGetType")
def tec_zone_var_get_type(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> DataType:
    """Get the data type for a variable in a zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.

    Returns:
        :class:`DataType` enum value.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    var_type = ctypes.c_int32(0)

    ret = lib.tecZoneVarGetType(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(var_type),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetType Error: handle={handle}, zone_index={zone_index}, "
            f"var_index={var_index}, return_code={ret}"
        )

    return DataType(var_type.value)


@requires_symbol("tecZoneVarGetValueLocation")
def tec_zone_var_get_value_location(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> ValueLocation:
    """Get the value location for a zone variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.

    Returns:
        :class:`ValueLocation` enum (NODAL or CELL_CENTERED).

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    value_location = ctypes.c_int32(0)

    ret = lib.tecZoneVarGetValueLocation(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(value_location),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetValueLocation Error: handle={handle}, "
            f"zone_index={zone_index}, var_index={var_index}, return_code={ret}"
        )

    return ValueLocation(value_location.value)


@requires_symbol("tecZoneVarIsPassive")
def tec_zone_var_is_passive(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> bool:
    """Check whether a zone variable is passive.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.

    Returns:
        True if the variable is passive in this zone.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    is_passive = ctypes.c_int32(0)

    ret = lib.tecZoneVarIsPassive(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(is_passive),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarIsPassive Error: handle={handle}, zone_index={zone_index}, "
            f"var_index={var_index}, return_code={ret}"
        )

    return bool(is_passive.value)


@requires_symbol("tecZoneVarGetSharedZone")
def tec_zone_var_get_shared_zone(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> int | None:
    """Get the shared zone index for a variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.

    Returns:
        Shared zone index, or None if not shared.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    shared_zone = ctypes.c_int32(0)

    ret = lib.tecZoneVarGetSharedZone(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(shared_zone),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetSharedZone Error: handle={handle}, "
            f"zone_index={zone_index}, var_index={var_index}, return_code={ret}"
        )

    return shared_zone.value if shared_zone.value != 0 else None


@requires_symbol("tecZoneConnectivityGetSharedZone")
def tec_zone_connectivity_get_shared_zone(
    handle: ctypes.c_void_p, zone_index: int
) -> int | None:
    """Get the shared zone index for unstructured mesh connectivity data.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        Shared zone index, or None if not shared.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    shared_zone = ctypes.c_int32(0)

    ret = lib.tecZoneConnectivityGetSharedZone(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.byref(shared_zone),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneConnectivityGetSharedZone Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return shared_zone.value if shared_zone.value != 0 else None


@requires_symbol("tecZoneVarGetNumValues")
def tec_zone_var_get_num_values(
    handle: ctypes.c_void_p, zone_index: int, var_index: int
) -> int:
    """Query the number of values for a zone variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.

    Returns:
        Number of values.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    num_values = ctypes.c_int32(0)

    ret = lib.tecZoneVarGetNumValues(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.byref(num_values),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetNumValues Error: handle={handle}, "
            f"zone_index={zone_index}, var_index={var_index}, return_code={ret}"
        )

    return num_values.value


@requires_symbol("tecZoneVarGetFloatValues")
def tec_zone_var_get_float_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.float32]:
    """Read float32 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.
        start_index (int): 1-based start position.
        num_values (int): Number of values to read.

    Returns:
        NumPy float32 array of length *num_values*.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    values = (ctypes.c_float * num_values)()

    ret = lib.tecZoneVarGetFloatValues(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.c_int64(start_index),
        ctypes.c_int64(num_values),
        ctypes.cast(
            values,
            ctypes.POINTER(ctypes.c_float),
        ),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetFloatValues Error: handle={handle}, "
            f"zone_index={zone_index}, var_index={var_index}, "
            f"start_index={start_index}, num_values={num_values}, return_code={ret}"
        )

    return np.ctypeslib.as_array(values)


@requires_symbol("tecZoneVarGetDoubleValues")
def tec_zone_var_get_double_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.float64]:
    """Read float64 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.
        start_index (int): 1-based start position.
        num_values (int): Number of values to read.

    Returns:
        NumPy float64 array of length *num_values*.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    values = (ctypes.c_double * num_values)()

    ret = lib.tecZoneVarGetDoubleValues(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.c_int64(start_index),
        ctypes.c_int64(num_values),
        ctypes.cast(
            values,
            ctypes.POINTER(ctypes.c_double),
        ),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetDoubleValues Error: handle={handle}, "
            f"zone_index={zone_index}, var_index={var_index}, "
            f"start_index={start_index}, num_values={num_values}, return_code={ret}"
        )

    return np.ctypeslib.as_array(values)


@requires_symbol("tecZoneVarGetInt32Values")
def tec_zone_var_get_int32_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.int32]:
    """Read int32 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.
        start_index (int): 1-based start position.
        num_values (int): Number of values to read.

    Returns:
        NumPy int32 array of length *num_values*.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    values = (ctypes.c_int32 * num_values)()

    ret = lib.tecZoneVarGetInt32Values(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.c_int64(start_index),
        ctypes.c_int64(num_values),
        ctypes.cast(
            values,
            ctypes.POINTER(ctypes.c_int32),
        ),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetInt32Values Error: handle={handle}, "
            f"zone_index={zone_index}, var_index={var_index}, "
            f"start_index={start_index}, num_values={num_values}, return_code={ret}"
        )

    return np.ctypeslib.as_array(values)


@requires_symbol("tecZoneVarGetInt16Values")
def tec_zone_var_get_int16_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.int16]:
    """Read int16 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.
        start_index (int): 1-based start position.
        num_values (int): Number of values to read.

    Returns:
        NumPy int16 array of length *num_values*.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    values = (ctypes.c_int16 * num_values)()

    ret = lib.tecZoneVarGetInt16Values(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.c_int64(start_index),
        ctypes.c_int64(num_values),
        ctypes.cast(
            values,
            ctypes.POINTER(ctypes.c_int16),
        ),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetInt16Values Error: handle={handle}, "
            f"zone_index={zone_index}, var_index={var_index}, "
            f"start_index={start_index}, num_values={num_values}, return_code={ret}"
        )

    return np.ctypeslib.as_array(values)


@requires_symbol("tecZoneVarGetUInt8Values")
def tec_zone_var_get_uint8_values(
    handle: ctypes.c_void_p,
    zone_index: int,
    var_index: int,
    start_index: int,
    num_values: int,
) -> npt.NDArray[np.uint8]:
    """Read uint8 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        var_index (int): 1-based variable index.
        start_index (int): 1-based start position.
        num_values (int): Number of values to read.

    Returns:
        NumPy uint8 array of length *num_values*.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    values = (ctypes.c_uint8 * num_values)()

    ret = lib.tecZoneVarGetUInt8Values(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(var_index),
        ctypes.c_int64(start_index),
        ctypes.c_int64(num_values),
        ctypes.cast(
            values,
            ctypes.POINTER(ctypes.c_uint8),
        ),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarGetUInt8Values Error: handle={handle}, "
            f"zone_index={zone_index}, var_index={var_index}, "
            f"start_index={start_index}, num_values={num_values}, return_code={ret}"
        )

    return np.ctypeslib.as_array(values)


# -- Reading SZL aux data --------------------------------------------------------------


@requires_symbol("tecDataSetAuxDataGetNumItems")
def tec_data_set_aux_data_get_num_items(handle: ctypes.c_void_p) -> int:
    """Get the number of dataset-level auxiliary data items.

    Args:
        handle (ctypes.c_void_p): File handle.

    Returns:
        Number of auxiliary data items.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    num_auxdata_items = ctypes.c_int32(0)

    ret = lib.tecDataSetAuxDataGetNumItems(handle, ctypes.byref(num_auxdata_items))
    if ret != 0:
        raise TecioError(
            f"tecDataSetAuxDataGetNumItems Error: handle={handle}, return_code={ret}"
        )

    return num_auxdata_items.value


@requires_symbol("tecDataSetAuxDataGetItem")
def tec_data_set_aux_data_get_item(
    handle: ctypes.c_void_p, item_index: int
) -> tuple[str, str]:
    """Read a dataset-level auxiliary data item.

    Args:
        handle (ctypes.c_void_p): File handle.
        item_index (int): 1-based item index.

    Returns:
        ``(name, value)`` string tuple.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    name = ctypes.c_char_p(0)
    value = ctypes.c_char_p(0)

    ret = lib.tecDataSetAuxDataGetItem(
        handle,
        ctypes.c_int32(item_index),
        ctypes.byref(name),
        ctypes.byref(value),
    )
    if ret != 0:
        raise TecioError(
            f"tecDataSetAuxDataGetItem Error: handle={handle}, "
            f"item_index={item_index}, return_code={ret}"
        )

    return _decode(name.value), _decode(value.value)


@requires_symbol("tecVarAuxDataGetNumItems")
def tec_var_aux_data_get_num_items(handle: ctypes.c_void_p, var_index: int) -> int:
    """Get the number of auxiliary data items for a variable.

    Args:
        handle (ctypes.c_void_p): File handle.
        var_index (int): 1-based variable index.

    Returns:
        Number of auxiliary data items.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    num_items = ctypes.c_int32(0)

    ret = lib.tecVarAuxDataGetNumItems(
        handle, ctypes.c_int32(var_index), ctypes.byref(num_items)
    )
    if ret != 0:
        raise TecioError(
            f"tecVarAuxDataGetNumItems Error: handle={handle}, "
            f"var_index={var_index}, return_code={ret}"
        )

    return num_items.value


@requires_symbol("tecVarAuxDataGetItem")
def tec_var_aux_data_get_item(
    handle: ctypes.c_void_p, var_index: int, item_index: int
) -> tuple[str, str]:
    """Read a variable-level auxiliary data item.

    Args:
        handle (ctypes.c_void_p): File handle.
        var_index (int): 1-based variable index.
        item_index (int): 1-based item index.

    Returns:
        ``(name, value)`` string tuple.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    name = ctypes.c_char_p(0)
    value = ctypes.c_char_p(0)

    ret = lib.tecVarAuxDataGetItem(
        handle,
        ctypes.c_int32(var_index),
        ctypes.c_int32(item_index),
        ctypes.byref(name),
        ctypes.byref(value),
    )
    if ret != 0:
        raise TecioError(
            f"tecVarAuxDataGetItem Error: handle={handle}, var_index={var_index}, "
            f"item_index={item_index}, return_code={ret}"
        )

    return _decode(name.value), _decode(value.value)


@requires_symbol("tecZoneAuxDataGetNumItems")
def tec_zone_aux_data_get_num_items(handle: ctypes.c_void_p, zone_index: int) -> int:
    """Get the number of auxiliary data items for a zone.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.

    Returns:
        Number of auxiliary data items.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    num_items = ctypes.c_int32(0)

    ret = lib.tecZoneAuxDataGetNumItems(
        handle, ctypes.c_int32(zone_index), ctypes.byref(num_items)
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneAuxDataGetNumItems Error: handle={handle}, "
            f"zone_index={zone_index}, return_code={ret}"
        )

    return num_items.value


@requires_symbol("tecZoneAuxDataGetItem")
def tec_zone_aux_data_get_item(
    handle: ctypes.c_void_p, zone_index: int, item_index: int
) -> tuple[str, str]:
    """Read a zone-level auxiliary data item.

    Args:
        handle (ctypes.c_void_p): File handle.
        zone_index (int): 1-based zone index.
        item_index (int): 1-based item index.

    Returns:
        ``(name, value)`` string tuple.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    name = ctypes.c_char_p(0)
    value = ctypes.c_char_p(0)

    ret = lib.tecZoneAuxDataGetItem(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_int32(item_index),
        ctypes.byref(name),
        ctypes.byref(value),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneAuxDataGetItem Error: handle={handle}, zone_index={zone_index}, "
            f"item_index={item_index}, return_code={ret}"
        )

    return _decode(name.value), _decode(value.value)


# --------------------------------------------------------------------------------------
# New write API (SZL/.szplt):
# - tec_file_writer_open returns an explicit file handle that is passed to every
#   subsequent call, making the target file unambiguous
# - Multiple files can be written simultaneously by holding multiple handles at once
#   tec_zone_create_ijk / tec_zone_create_fe append a new zone record and return a
#   1-based zone index
# - Variable data, node maps, aux data, and unsteady options are written by referencing
#   the file handle + zone/variable index, so they can be written in any order after
#   zone creation
# - tec_file_writer_close finalizes and flushes the file
# --------------------------------------------------------------------------------------


# -- Initialization and File Handling --------------------------------------------------


@requires_symbol("tecFileWriterOpen")
def tec_file_writer_open(
    filename: str,
    variables: Sequence[str],
    title: str = "Untitled",
    file_type: FileType = FileType.FULL,
    use_szl: int = 1,
    grid_file_handle: ctypes.c_void_p | None = None,
) -> ctypes.c_void_p:
    """Open a writer handle for creating SZL files.

    Args:
        filename (str): Output file path.
        variables (Sequence[str]): Variable name list.
        title (str): Dataset title.
        file_type (FileType): File type enum.
        use_szl (int): SZL flag (1 to use SZL format).
        grid_file_handle (ctypes.c_void_p | None): Optional handle to an
            existing grid file for solution-only output.

    Returns:
        Opaque writer handle for subsequent write calls.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    if not isinstance(file_type, FileType):
        raise TypeError("file_type must be a libtecio.FileType enum")

    assert all(len(v) <= 128 for v in variables), "Variables limited to 128 characters"

    varstring = ", ".join(variables)
    handle = ctypes.c_void_p()

    ret = lib.tecFileWriterOpen(
        ctypes.c_char_p(filename.encode("utf-8")),
        ctypes.c_char_p(title.encode("utf-8")),
        ctypes.c_char_p(varstring.encode("utf-8")),
        ctypes.c_int32(use_szl),
        ctypes.c_int32(file_type.value),
        ctypes.c_int32(0),
        grid_file_handle if grid_file_handle is not None else None,
        ctypes.byref(handle),
    )
    if ret != 0:
        raise TecioError(
            f"tecFileWriterOpen Error: file_name={filename!r}, title={title!r}, "
            f"variables={variables!r}, file_type={file_type!r}, return_code={ret}"
        )
    return handle


@requires_symbol("tecFileWriterClose")
def tec_file_writer_close(handle: ctypes.c_void_p) -> None:
    """Close a writer handle and finalise the output file.

    Args:
        handle (ctypes.c_void_p): Writer handle.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ret = lib.tecFileWriterClose(ctypes.byref(handle))
    if ret != 0:
        raise TecioError(
            f"tecFileWriterClose Error: handle={handle}, return_code={ret}"
        )


@requires_symbol("tecFileWriterFlush")
def tec_file_writer_flush(
    handle: ctypes.c_void_p,
    num_zones_to_retain: int = 0,
    zones_to_retain: Sequence[int] | None = None,
) -> None:
    """Flush written zone data to a temporary intermediate file.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        num_zones_to_retain (int): Number of zones to keep in memory.
        zones_to_retain (Sequence[int] | None): 1-based zone indices to retain.

    Raises:
        TecioError: On C library error.

    Important:
        SZL Only!

    Note:
        Used to reduce memory usage for large files. All zone data written so far, other
        than any zones listed in ``zones_to_retain``, is written out to a temporary file
        on disk and released from memory.

    Note:
        Retained zones can still be modified.

    Note:
        Temporary files created by flushing are merged into the final output file when
        :func:`tec_file_writer_close` is called.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    zones_ptr = None
    if zones_to_retain is not None and len(zones_to_retain) > 0:
        zones_array = (ctypes.c_int32 * len(zones_to_retain))(*zones_to_retain)
        zones_ptr = ctypes.cast(zones_array, ctypes.POINTER(ctypes.c_int32))
    else:
        zones_ptr = ctypes.POINTER(ctypes.c_int32)()

    ret = lib.tecFileWriterFlush(
        handle,
        ctypes.c_int32(num_zones_to_retain),
        zones_ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecFileWriterFlush Error: handle={handle}, "
            f"num_zones_to_retain={num_zones_to_retain}, "
            f"zones_to_retain={zones_to_retain}, "
            f"return_code={ret}"
        )


# -- Write Zone Headers ----------------------------------------------------------------


@requires_symbol("tecZoneCreateIJK")
def tec_zone_create_ijk(
    handle: ctypes.c_void_p,
    zone_title: str,
    imax: int,
    jmax: int,
    kmax: int,
    var_types: Sequence[DataType | int],
    var_sharing: Sequence[int] | None = None,
    value_locations: Sequence[int | ValueLocation] | None = None,
    pas_vars: Sequence[VarStatus | bool | int] | None = None,
    face_nbr_sharing: int = 0,
    num_face_cons: int = 0,
    face_nbr_mode: FaceNeighborMode | int = FaceNeighborMode.LOCAL_ONE_TO_ONE,
) -> int:
    """Create an ordered IJK zone for writing.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone_title (str): Zone title.
        imax (int): I dimension.
        jmax (int): J dimension.
        kmax (int): K dimension.
        var_types (Sequence[DataType | int]): Per-variable data types.
        var_sharing (Sequence[int] | None): Optional per-variable sharing source
            zones. Must be same length as var_types if provided. None/null means all no
            variable sharing.
        value_locations (Sequence[int | ValueLocation] | None): Optional per-variable
            data locations.
        pas_vars (Sequence[bool | int] | None): Optional per-variable passive flags or
            1/True if passive else 0/False. Must be same length as var_types if
            provided. None/null means all variables are active.
        face_nbr_sharing (int): Optional face-neighbor sharing source zone.
        num_face_cons (int): Optional number of face connections.
        face_nbr_mode (FaceNeighborMode | int): Optional face-neighbor mode.

    Returns:
        1-based zone index of the created zone.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    zone_out = ctypes.c_int32()

    # Create C array for varable types
    var_types_ptr = (ctypes.c_int32 * len(var_types))(*[
        _to_int_value(v, DataType) for v in var_types
    ])

    # Create C array for variable sharing
    var_sharing_ptr = None
    if var_sharing is not None:
        var_sharing_ptr = (ctypes.c_int32 * len(var_sharing))(*list(var_sharing))

    # Create C array for value locations
    value_locations_ptr = None
    if value_locations is not None:
        value_locations_ptr = (ctypes.c_int32 * len(value_locations))(*[
            _to_int_value(v, ValueLocation) for v in value_locations
        ])

    # Create C array for passive variable flags
    pas_vars_ptr = None
    if pas_vars is not None:
        pas_vars_ptr = (ctypes.c_int32 * len(pas_vars))(*[
            _to_int_value(v, Boolean) for v in pas_vars
        ])

    ret = lib.tecZoneCreateIJK(
        handle,
        ctypes.c_char_p(zone_title.encode("utf-8")),
        ctypes.c_int64(imax),
        ctypes.c_int64(jmax),
        ctypes.c_int64(kmax),
        var_types_ptr,
        var_sharing_ptr,
        value_locations_ptr,
        pas_vars_ptr,
        ctypes.c_int32(face_nbr_sharing),  # face neighbor sharing source zone
        ctypes.c_int64(num_face_cons),  # number of face connections
        ctypes.c_int32(
            _to_int_value(face_nbr_mode, FaceNeighborMode)
        ),  # face neighbor mode
        ctypes.byref(zone_out),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneCreateIJK Error: zone_title={zone_title!r}, "
            f"imax={imax}, jmax={jmax}, kmax={kmax}, "
            f"var_types_len={len(var_types) if var_types is not None else 0}, "
            f"return_code={ret}"
        )
    return zone_out.value


@requires_symbol("tecZoneCreateFE")
def tec_zone_create_fe(
    handle: ctypes.c_void_p,
    zone_title: str,
    zone_type: int | ZoneType,
    num_nodes: int,
    num_cells: int,
    var_types: Sequence[DataType | int],
    var_sharing: Sequence[int] | None = None,
    value_locations: Sequence[int | ValueLocation] | None = None,
    pas_vars: Sequence[bool | int] | None = None,
    con_sharing: int = 0,
    num_face_cons: int = 0,
    face_nbr_mode: FaceNeighborMode | int = FaceNeighborMode.LOCAL_ONE_TO_ONE,
) -> int:
    """Create a finite-element zone for writing.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone_title (str): Zone title.
        zone_type (int | ZoneType): FE zone type.
        num_nodes (int): Number of nodes.
        num_cells (int): Number of cells/elements.
        var_types (Sequence[DataType | int]): Optional per-variable data types.
        var_sharing (Sequence[int] | None): Optional per-variable sharing source
            zones. Must be same length as var_types if provided. None/null means all no
            variable sharing.
        value_locations (Sequence[int | ValueLocation] | None): Optional per-variable
            data locations.
        pas_vars (Sequence[bool | int] | None): Optional per-variable passive flags or
            1/True if passive else 0/False. Must be same length as var_types if
            provided. None/null means all variables are active.
        con_sharing (int): Optional connectivity sharing source zone (0 =
            none). Connectivity and/or face neighbors cannot be shared when the face
            neighbor mode is set to Global. Connectivity cannot be shared between
            cell-based and face-based finite element zones.
        num_face_cons (int): Optional number of face connections.
        face_nbr_mode (FaceNeighborMode | int): Optional face-neighbor mode.

    Returns:
        1-based zone index of the created zone.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    zone_out = ctypes.c_int32()

    # Create C array for varable types
    var_types_ptr = (ctypes.c_int32 * len(var_types))(*[
        _to_int_value(v, DataType) for v in var_types
    ])

    # Create C array for variable sharing
    var_sharing_ptr = None
    if var_sharing is not None:
        var_sharing_ptr = (ctypes.c_int32 * len(var_sharing))(*list(var_sharing))

    # Create C array for value locations
    value_locations_ptr = None
    if value_locations is not None:
        value_locations_ptr = (ctypes.c_int32 * len(value_locations))(*[
            _to_int_value(v, ValueLocation) for v in value_locations
        ])

    # Create C array for passive variable flags
    pas_vars_ptr = None
    if pas_vars is not None:
        pas_vars_ptr = (ctypes.c_int32 * len(pas_vars))(*[
            _to_int_value(v, Boolean) for v in pas_vars
        ])

    ret = lib.tecZoneCreateFE(
        handle,
        ctypes.c_char_p(zone_title.encode("utf-8")),
        ctypes.c_int32(_to_int_value(zone_type)),
        ctypes.c_int64(num_nodes),
        ctypes.c_int64(num_cells),
        var_types_ptr,
        var_sharing_ptr,
        value_locations_ptr,
        pas_vars_ptr,
        ctypes.c_int32(con_sharing),  # connectivity sharing
        ctypes.c_int64(num_face_cons),  # num face connections
        ctypes.c_int32(
            _to_int_value(face_nbr_mode, FaceNeighborMode)
        ),  # face neighbor mode
        ctypes.byref(zone_out),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneCreateFE Error: zone_title={zone_title!r}, "
            f"ZoneType={zone_type!r}, NODES={num_nodes}, CELLS={num_cells}, "
            f"var_types_len={len(var_types) if var_types is not None else 0}, "
            f"return_code={ret}"
        )
    return zone_out.value


@requires_symbol("tecZoneCreateFEMixed")
def tec_zone_create_fe_mixed(
    handle: ctypes.c_void_p,
    zone_title: str,
    num_nodes: int,
    cell_shapes_per_section: Sequence[int | FeCellShape],
    num_elements_per_section: Sequence[int],
    var_types: Sequence[DataType | int],
    grid_order_per_section: Sequence[int] | None = None,
    var_sharing: Sequence[int] | None = None,
    value_locations: Sequence[int | ValueLocation] | None = None,
    pas_vars: Sequence[bool | int] | None = None,
    con_sharing: int = 0,
    num_face_cons: int = 0,
    face_nbr_mode: FaceNeighborMode | int = FaceNeighborMode.LOCAL_ONE_TO_ONE,
) -> int:
    """Create a mixed finite-element zone for writing.

    A mixed-element zone groups cells into 1-16 sections; every cell within
    one section shares the same shape and grid order. All sections in a
    zone must share the same spatial dimensionality (all line, all surface,
    or all volume cell types), not a mix of them.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone_title (str): Zone title.
        num_nodes (int): Total number of nodes for the zone.
        cell_shapes_per_section (Sequence[int | FeCellShape]): Cell shape for
            each section. Length determines the number of sections (1-16).
        num_elements_per_section (Sequence[int]): Number of elements in each
            section. Must be the same length as cell_shapes_per_section.
        var_types (Sequence[DataType | int]): Per-variable data types.
        grid_order_per_section (Sequence[int] | None): Grid order (1-4) for
            each section. None defaults to linear (1) for every section.
        var_sharing (Sequence[int] | None): Optional per-variable sharing
            source zones. Must be same length as var_types if provided.
            None means no variable sharing.
        value_locations (Sequence[int | ValueLocation] | None): Optional
            per-variable data locations.
        pas_vars (Sequence[bool | int] | None): Optional per-variable
            passive flags. Must be same length as var_types if provided.
            None means all variables are active.
        con_sharing (int): Optional connectivity sharing source zone (0 =
            none).
        num_face_cons (int): Optional number of face connections.
        face_nbr_mode (FaceNeighborMode | int): Optional face-neighbor mode.

    Returns:
        1-based zone index of the created zone.

    Raises:
        TecioError: On C library error.
        ValueError: If cell_shapes_per_section and num_elements_per_section
            (or grid_order_per_section, if provided) have mismatched
            lengths, or there are not between 1 and 16 sections.

    Note:
        The basis function per section is always 0 (the only value the C
        library currently accepts), so it isn't exposed as a parameter here.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    num_sections = len(cell_shapes_per_section)
    if not 1 <= num_sections <= 16:
        raise ValueError(
            f"tecZoneCreateFEMixed requires 1-16 sections, got {num_sections}"
        )
    if len(num_elements_per_section) != num_sections:
        raise ValueError(
            "cell_shapes_per_section and num_elements_per_section must be "
            f"the same length ({num_sections} != "
            f"{len(num_elements_per_section)})"
        )
    if grid_order_per_section is None:
        grid_order_per_section = [1] * num_sections
    elif len(grid_order_per_section) != num_sections:
        raise ValueError(
            "grid_order_per_section must be the same length as "
            f"cell_shapes_per_section ({num_sections} != "
            f"{len(grid_order_per_section)})"
        )

    zone_out = ctypes.c_int32()

    cell_shapes_ptr = (ctypes.c_int32 * num_sections)(*[
        _to_int_value(v, FeCellShape) for v in cell_shapes_per_section
    ])
    grid_order_ptr = (ctypes.c_int32 * num_sections)(*list(grid_order_per_section))
    basis_fn_ptr = (ctypes.c_int32 * num_sections)(*([0] * num_sections))
    num_elements_ptr = (ctypes.c_int64 * num_sections)(*list(num_elements_per_section))

    # Create C array for variable types
    var_types_ptr = (ctypes.c_int32 * len(var_types))(*[
        _to_int_value(v, DataType) for v in var_types
    ])

    # Create C array for variable sharing
    var_sharing_ptr = None
    if var_sharing is not None:
        var_sharing_ptr = (ctypes.c_int32 * len(var_sharing))(*list(var_sharing))

    # Create C array for value locations
    value_locations_ptr = None
    if value_locations is not None:
        value_locations_ptr = (ctypes.c_int32 * len(value_locations))(*[
            _to_int_value(v, ValueLocation) for v in value_locations
        ])

    # Create C array for passive variable flags
    pas_vars_ptr = None
    if pas_vars is not None:
        pas_vars_ptr = (ctypes.c_int32 * len(pas_vars))(*[
            _to_int_value(v, Boolean) for v in pas_vars
        ])

    ret = lib.tecZoneCreateFEMixed(
        handle,
        ctypes.c_char_p(zone_title.encode("utf-8")),
        ctypes.c_int64(num_nodes),
        ctypes.c_int32(num_sections),
        cell_shapes_ptr,
        grid_order_ptr,
        basis_fn_ptr,
        num_elements_ptr,
        var_types_ptr,
        var_sharing_ptr,
        value_locations_ptr,
        pas_vars_ptr,
        ctypes.c_int32(con_sharing),
        ctypes.c_int64(num_face_cons),
        ctypes.c_int32(_to_int_value(face_nbr_mode, FaceNeighborMode)),
        ctypes.byref(zone_out),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneCreateFEMixed Error: zone_title={zone_title!r}, "
            f"NODES={num_nodes}, SECTIONS={num_sections}, return_code={ret}"
        )
    return zone_out.value


# -- Optional fields -------------------------------------------------------------------


@requires_symbol("tecZoneSetUnsteadyOptions")
def tec_zone_set_unsteady_options(
    handle: ctypes.c_void_p, zone: int, strand: int = 0, solution_time: float = 0.0
) -> None:
    """Set time/strand metadata for a zone.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        strand (int): Strand ID.
        solution_time (float): Solution time.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ret = lib.tecZoneSetUnsteadyOptions(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_double(solution_time),
        ctypes.c_int32(strand),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneSetUnsteadyOptions Error: zone={zone}, strand={strand}, "
            f"solution_time={solution_time}, return_code={ret}"
        )


@requires_symbol("tecDataSetAddAuxData")
def tec_data_set_add_aux_data(
    handle: ctypes.c_void_p,
    name: str,
    value: str,
) -> None:
    """Add a dataset-level auxiliary data record.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        name (str): Auxiliary data name.
        value (str): Auxiliary data value.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ret = lib.tecDataSetAddAuxData(
        handle,
        ctypes.c_char_p(name.encode("utf-8")),
        ctypes.c_char_p(value.encode("utf-8")),
    )
    if ret != 0:
        raise TecioError(
            f"tecDataSetAddAuxData Error: name={name!r}, value={value!r}, "
            f"return_code={ret}"
        )


@requires_symbol("tecVarAddAuxData")
def tec_var_add_aux_data(
    handle: ctypes.c_void_p,
    var_index: int,
    name: str,
    value: str,
) -> None:
    """Add a variable-level auxiliary data record.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        var_index (int): 1-based variable index.
        name (str): Auxiliary data name.
        value (str): Auxiliary data value.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ret = lib.tecVarAddAuxData(
        handle,
        ctypes.c_int32(var_index),
        ctypes.c_char_p(name.encode("utf-8")),
        ctypes.c_char_p(value.encode("utf-8")),
    )
    if ret != 0:
        raise TecioError(
            f"tecVarAddAuxData Error: var_index={var_index}, name={name!r}, "
            f"value={value!r}, return_code={ret}"
        )


@requires_symbol("tecZoneAddAuxData")
def tec_zone_add_aux_data(
    handle: ctypes.c_void_p,
    zone_index: int,
    name: str,
    value: str,
) -> None:
    """Add a zone-level auxiliary data record.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone_index (int): 1-based zone index.
        name (str): Auxiliary data name.
        value (str): Auxiliary data value.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ret = lib.tecZoneAddAuxData(
        handle,
        ctypes.c_int32(zone_index),
        ctypes.c_char_p(name.encode("utf-8")),
        ctypes.c_char_p(value.encode("utf-8")),
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneAddAuxData Error: zone_index={zone_index}, name={name!r}, "
            f"value={value!r}, return_code={ret}"
        )


# -- Write variable value functions ----------------------------------------------------


@requires_symbol("tecZoneVarWriteDoubleValues")
def tec_zone_var_write_double_values(
    handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
    """Write float64 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        var (int): 1-based variable index.
        values (npt.ArrayLike): Array of float64 values.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ptr, count, _backing = _prepare_array_for_ctypes(
        values, np.float64, ctypes.c_double
    )

    ret = lib.tecZoneVarWriteDoubleValues(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_int32(var),
        ctypes.c_int32(0),
        ctypes.c_int64(count),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarWriteDoubleValues Error: zone={zone}, var={var}, "
            f"count={count}, return_code={ret}"
        )


@requires_symbol("tecZoneVarWriteFloatValues")
def tec_zone_var_write_float_values(
    handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
    """Write float32 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        var (int): 1-based variable index.
        values (npt.ArrayLike): Array of float32 values.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ptr, count, _backing = _prepare_array_for_ctypes(values, np.float32, ctypes.c_float)

    ret = lib.tecZoneVarWriteFloatValues(
        handle,
        ctypes.c_int32(zone),  # zone index (1-based)
        ctypes.c_int32(var),  # variable index (1-based)
        ctypes.c_int32(0),  # partition index (MPI)
        ctypes.c_int64(count),  # number of values
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarWriteFloatValues Error: zone={zone}, var={var}, "
            f"count={count}, return_code={ret}"
        )


@requires_symbol("tecZoneVarWriteInt32Values")
def tec_zone_var_write_int32_values(
    handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
    """Write int32 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        var (int): 1-based variable index.
        values (npt.ArrayLike): Array of int32 values.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ptr, count, _backing = _prepare_array_for_ctypes(values, np.int32, ctypes.c_int32)

    ret = lib.tecZoneVarWriteInt32Values(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_int32(var),
        ctypes.c_int32(0),
        ctypes.c_int64(count),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarWriteInt32Values Error: zone={zone}, var={var}, "
            f"count={count}, return_code={ret}"
        )


@requires_symbol("tecZoneVarWriteInt16Values")
def tec_zone_var_write_int16_values(
    handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
    """Write int16 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        var (int): 1-based variable index.
        values (npt.ArrayLike): Array of int16 values.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ptr, count, _backing = _prepare_array_for_ctypes(values, np.int16, ctypes.c_int16)

    ret = lib.tecZoneVarWriteInt16Values(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_int32(var),
        ctypes.c_int32(0),
        ctypes.c_int64(count),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarWriteInt16Values Error: zone={zone}, var={var}, "
            f"count={count}, return_code={ret}"
        )


@requires_symbol("tecZoneVarWriteUInt8Values")
def tec_zone_var_write_uint8_values(
    handle: ctypes.c_void_p, zone: int, var: int, values: npt.ArrayLike
) -> None:
    """Write uint8 values for a zone variable.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        var (int): 1-based variable index.
        values (npt.ArrayLike): Array of uint8 values.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ptr, count, _backing = _prepare_array_for_ctypes(values, np.uint8, ctypes.c_uint8)

    ret = lib.tecZoneVarWriteUInt8Values(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_int32(var),
        ctypes.c_int32(0),
        ctypes.c_int64(count),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneVarWriteUInt8Values Error: zone={zone}, var={var}, "
            f"count={count}, return_code={ret}"
        )


# -- Write Zone Connectivity (FE zones only) -------------------------------------------


@requires_symbol("tecZoneNodeMapWrite32")
def tec_zone_node_map_write32(
    handle: ctypes.c_void_p,
    zone: int,
    nodes: npt.ArrayLike,
    partition: int = 0,
    is_one_based: bool | int = True,
) -> None:
    """Write 32-bit node map entries for an FE zone.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        nodes (npt.ArrayLike): Array of int32 node indices.
        partition (int): Partition index (0 for non-partitioned).
        is_one_based (bool | int): Whether indices are 1-based.

    Raises:
        TecioError: On C library error.

    Note:
        Use for zones with fewer than ~2 billion node map entries.

    Note:
        Can be called multiple times per zone; total entries must match zone
        definition.

    Note:
        Node indices are 1-based by convention in Tecplot.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ptr, count, _backing = _prepare_array_for_ctypes(nodes, np.int32, ctypes.c_int32)

    ret = lib.tecZoneNodeMapWrite32(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_int32(partition),
        ctypes.c_int32(_to_int_value(is_one_based, Boolean)),
        ctypes.c_int64(count),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneNodeMapWrite32 Error: zone={zone}, partition={partition}, "
            f"count={count}, return_code={ret}"
        )


@requires_symbol("tecZoneNodeMapWrite64")
def tec_zone_node_map_write64(
    handle: ctypes.c_void_p,
    zone: int,
    nodes: npt.ArrayLike,
    partition: int = 0,
    is_one_based: bool | int = True,
) -> None:
    """Write 64-bit node map entries for an FE zone.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        nodes (npt.ArrayLike): Array of int64 node indices.
        partition (int): Partition index (0 for non-partitioned).
        is_one_based (bool | int): Whether indices are 1-based.

    Raises:
        TecioError: On C library error.

    Note:
        Use for zones with fewer than ~2 billion node map entries.

    Note:
        Can be called multiple times per zone; total entries must match zone
        definition.

    Note:
        Node indices are 1-based by convention in Tecplot.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ptr, count, _backing = _prepare_array_for_ctypes(nodes, np.int64, ctypes.c_int64)

    ret = lib.tecZoneNodeMapWrite64(
        handle,
        ctypes.c_int32(zone),
        ctypes.c_int32(partition),
        ctypes.c_int32(_to_int_value(is_one_based, Boolean)),
        ctypes.c_int64(count),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneNodeMapWrite64 Error: zone={zone}, partition={partition}, "
            f"count={count}, return_code={ret}"
        )


@requires_symbol("tecZoneFaceNbrWriteConnections32")
def tec_zone_face_nbr_write_connections32(
    handle: ctypes.c_void_p,
    zone: int,
    face_neighbors: npt.ArrayLike,
) -> None:
    """Write 32-bit face-neighbor connections for an FE zone.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        face_neighbors (npt.ArrayLike): Array of int32 face-neighbor indices.

    Raises:
        TecioError: On C library error.

    Note:
        ``num_face_cons`` and ``face_nbr_mode`` must be set when creating the zone via
        :func:`tec_zone_create_ijk` or :func:`tec_zone_create_fe`.

    Note:
        Use for zones with fewer than ~2 billion face neighbor entries.

    Note:
        Can be called multiple times. Total entries must match num_face_cons
        declared at zone creation.

    Note:
        Face neighbors have expensive performance implications. Use face neighbors
        only to manually specify connections that are not defined via the connectivity
        list.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ptr, count, _backing = _prepare_array_for_ctypes(
        face_neighbors, np.int32, ctypes.c_int32
    )
    ret = lib.tecZoneFaceNbrWriteConnections32(
        handle,
        ctypes.c_int32(zone),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneFaceNbrWriteConnections32 Error: zone={zone}, "
            f"count={count}, return_code={ret}"
        )


@requires_symbol("tecZoneFaceNbrWriteConnections64")
def tec_zone_face_nbr_write_connections64(
    handle: ctypes.c_void_p,
    zone: int,
    face_neighbors: npt.ArrayLike,
) -> None:
    """Write 64-bit face-neighbor connections for an FE zone.

    Args:
        handle (ctypes.c_void_p): Writer handle.
        zone (int): 1-based zone index.
        face_neighbors (npt.ArrayLike): Array of int64 face-neighbor indices.

    Raises:
        TecioError: On C library error.

    Note:
        ``num_face_cons`` and ``face_nbr_mode`` must be set when creating the zone via
        :func:`tec_zone_create_ijk` or :func:`tec_zone_create_fe`.

    Note:
        Use for zones with more than ~2 billion face neighbor entries.

    Note:
        Can be called multiple times. Total entries must match num_face_cons
        declared at zone creation.

    Note:
        Face neighbors have expensive performance implications. Use face neighbors
        only to manually specify connections that are not defined via the connectivity
        list.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ptr, count, _backing = _prepare_array_for_ctypes(
        face_neighbors, np.int64, ctypes.c_int64
    )
    ret = lib.tecZoneFaceNbrWriteConnections64(
        handle,
        ctypes.c_int32(zone),
        ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecZoneFaceNbrWriteConnections64 Error: zone={zone}, "
            f"count={count}, return_code={ret}"
        )


# --------------------------------------------------------------------------------------
# Classic API (PLT/.plt AND SZL/.szplt):
# - No handle is returned — the library maintains a single implicit global file context
# - Only one file can be active at a time; tecfil142 must be called to switch between
#   files if writing multiple files simultainously
# - tecini142 initializes the file and sets the global context; all subsequent calls
#   implicitly target this file
# - Zone records (teczne142), data (tecdat142), and node maps (tecnode142) must be
#   written strictly in order — each zone's header followed immediately by its data
#   before the next zone is declared
# - tecend142 finalizes and closes the active file
# --------------------------------------------------------------------------------------


# -- File initialization and finalization ----------------------------------------------


@requires_symbol("tecini142")
def tecini142(
    filename: str,
    variables: Sequence[str],
    title: str = "Untitled",
    scratch_dir: str = ".",
    file_format: int | FileFormat = FileFormat.PLT,
    file_type: int | FileType = FileType.FULL,
    debug: int | Debug = Debug.FALSE,
    vis_double: int | DataType = DataType.DOUBLE,
) -> None:
    """Initialise a Tecplot data file (classic API).

    Args:
        filename (str): Output file path. Supports PLT and SZPLT extensions.
        variables (Sequence[str]): Variable name list.
        title (str): Dataset title.
        scratch_dir (str): Scratch directory for temporary files.
        file_format (int | FileFormat): FileFormat.PLT (0) or FileFormat.SZPLT (1)
        file_type (int | FileType): FileType.FULL (0), GRID (1), or SOLUTION (2)
        debug (int | Debug): Debug.FALSE (0) or Debug.TRUE (1)
        vis_double (int | DataType): DataType.DOUBLE (1) or DataType.FLOAT (0). PLT
            files do not suport integer zone variables. Use the new API functions to
            create integer zone variables for SZPLT files.

    Raises:
        TecioError: On C library error.

    Note:
        Must be called before any zone or data operations

    Note:
        Call :func:`tecend142()` to finalize the file
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    if not isinstance(file_type, FileType):
        raise TypeError("file_type must be a libtecio.FileType enum")

    assert all(len(v) <= 128 for v in variables), "Variables limited to 128 characters"

    varstring = ", ".join(variables)
    vis_double_c = ctypes.c_int32(
        1 if _to_int_value(vis_double) == DataType.DOUBLE.value else 0
    )
    ret = lib.tecini142(
        ctypes.c_char_p(title.encode("utf-8")),
        ctypes.c_char_p(varstring.encode("utf-8")),
        ctypes.c_char_p(filename.encode("utf-8")),
        ctypes.c_char_p(scratch_dir.encode("utf-8")),
        ctypes.c_int32(_to_int_value(file_format)),
        ctypes.c_int32(_to_int_value(file_type)),
        ctypes.c_int32(_to_int_value(debug)),
        vis_double_c,
    )
    if ret != 0:
        raise TecioError(f"tecini142 Error: filename={filename!r}, return_code={ret}")


@requires_symbol("tecend142")
def tecend142() -> None:
    """Finalise and close the active Tecplot data file.

    Raises:
        TecioError: On C library error.

    Note:
        Must be called after all data has been written

    Note:
        Flushes all pending data and closes the file
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ret = lib.tecend142()
    if ret != 0:
        raise TecioError(f"tecend142 Error: return_code={ret}")


@requires_symbol("tecflush142")
def tecflush142(
    num_zones_to_retain: int = 0,
    zones_to_retain: Sequence[int] | None = None,
) -> None:
    """Flush data to disk, optionally retaining zones in memory (SZL only).

    Args:
        num_zones_to_retain (int): Number of zones to keep in memory.
        zones_to_retain (Sequence[int] | None): 1-based zone indices to retain.

    Raises:
        TecioError: On C library error.

    Important:
        SZL Only!

    Note:
        Used to reduce memory usage for large files.

    Note:
        Retained zones can still be modified.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    num_zones_c = ctypes.c_int32(num_zones_to_retain)

    zones_ptr = None
    if zones_to_retain is not None and len(zones_to_retain) > 0:
        zones_array = (ctypes.c_int32 * len(zones_to_retain))(*zones_to_retain)
        zones_ptr = ctypes.cast(zones_array, ctypes.POINTER(ctypes.c_int32))
    else:
        zones_ptr = ctypes.POINTER(ctypes.c_int32)()

    ret = lib.tecflush142(
        ctypes.byref(num_zones_c),
        zones_ptr,
    )
    if ret != 0:
        raise TecioError(f"tecflush142 Error: return_code={ret}")


@requires_symbol("tecfil142")
def tecfil142() -> int:
    """Get the file handle for the current output file.

    Returns:
        File handle integer.

    Raises:
        TecioError: On C library error.

    Note:
        Used for advanced file operations.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    output_file_handle = ctypes.c_int32(0)

    ret = lib.tecfil142(ctypes.byref(output_file_handle))
    if ret != 0:
        raise TecioError(f"tecfil142 Error: return_code={ret}")

    return output_file_handle.value


@requires_symbol("tecforeign142")
def tecforeign142(output_foreign_byte_order: int) -> None:
    """Set foreign byte order for output.

    Args:
        output_foreign_byte_order (int): 0 = native, 1 = foreign.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    foreign_c = ctypes.c_int32(output_foreign_byte_order)

    ret = lib.tecforeign142(ctypes.byref(foreign_c))
    if ret != 0:
        raise TecioError(f"tecforeign142 Error: return_code={ret}")


# -- Zone creation ---------------------------------------------------------------------


@requires_symbol("teczne142")
def teczne142(
    zone_title: str,
    zone_type: int | ZoneType,
    imax: int,
    jmax: int,
    kmax: int,
    var_sharing: Sequence[int] | None = None,
    value_locations: Sequence[int | ValueLocation] | None = None,
    pas_vars: Sequence[int | VarStatus] | None = None,
    con_sharing: int = 0,
    strand: int = 0,
    solution_time: float = 0.0,
    num_face_connections: int = 0,
    face_nbr_mode: int | FaceNeighborMode = FaceNeighborMode.LOCAL_ONE_TO_ONE,
    total_num_face_nodes: int = 0,
    num_connected_boundary_faces: int = 0,
    total_num_boundary_connections: int = 0,
) -> None:
    """Create a new zone in the active file (classic API).

    Args:
        zone_title (str): Zone title.
        zone_type (int | ZoneType): Zone type enum or integer (0=ORDERED, 1=FELINESEG,
            etc.).
        imax (int): I dimension (or num nodes for FE).
        jmax (int): J dimension (or num elements for FE).
        kmax (int): K dimension or NumFaces for FEPOLYGON and FEPOLYHEDRON. Not used for
            all other finite element zone types.
        var_sharing (Sequence[int] | None): List of zone indices to share variables from
            (0: No sharing, Null: no variables are shared from other zones)
        value_locations (Sequence[int | ValueLocation] | None): Per-variable
            data locations.
        pas_vars (Sequence[int | VarStatus] | None): List of VarStatus enums or 0/1 for
            passive variables. Must be same length as var_sharing if provided. None/null
            means all variables are active.
        con_sharing (int): Connectivity sharing source zone number.
        strand (int): Strand ID for transient data (0 for static data, positive integer
            for transient data).
        solution_time (float): Solution time.
        num_face_connections (int): Number of face connections (for cell-based FE zones
            only. The number of face connections that will be passed to ``tecface142``).
        face_nbr_mode (int | FaceNeighborMode): FaceNeighborMode enum. Used for
            cell-based FE and ordered zones only. Type of face connection that will be
            passed in routine ``tecface142``.
        total_num_face_nodes (int): Total face nodes (poly zones).
        num_connected_boundary_faces (int): Boundary faces (poly zones).
        total_num_boundary_connections (int): Boundary connections (poly zones).

    Raises:
        TecioError: On C library error.

    Hint:
        For ORDERED zones: imx, jmx, kmx are dimensions.

    Note:
        For FE zones: imax=NumNodes, jmax=NumElements, kmax=0 (unless higher order
        element)
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    # Create C array for passive variable flags
    pas_vars_ptr = None
    if pas_vars is not None:
        pas_vars_ptr = (ctypes.c_int32 * len(pas_vars))(*[
            _to_int_value(v, Boolean) for v in pas_vars
        ])

    # Create C array for value locations
    value_locations_ptr = None
    if value_locations is not None:
        value_locations_ptr = (ctypes.c_int32 * len(value_locations))(*[
            _to_int_value(v, ValueLocation) for v in value_locations
        ])

    # Create C array for variable sharing
    var_sharing_ptr = None
    if var_sharing is not None:
        var_sharing_ptr = (ctypes.c_int32 * len(var_sharing))(*list(var_sharing))

    ret = lib.teczne142(
        ctypes.c_char_p(zone_title.encode("utf-8")),
        ctypes.c_int32(_to_int_value(zone_type)),
        ctypes.c_int32(imax),
        ctypes.c_int32(jmax),
        ctypes.c_int32(kmax),
        ctypes.c_int32(0),  # I-cell dimension (Reserved for future use. Set to zero)
        ctypes.c_int32(0),  # J-cell dimension (Reserved for future use. Set to zero)
        ctypes.c_int32(0),  # K-cell dimension (Reserved for future use. Set to zero)
        ctypes.c_double(solution_time),
        ctypes.c_int32(strand),
        ctypes.c_int32(0),  # Deprecated. Enter 0 for this value
        ctypes.c_int32(_to_int_value(DataPacking.BLOCK)),  # Deprecated. Always set to 1
        ctypes.c_int32(num_face_connections),
        ctypes.c_int32(_to_int_value(face_nbr_mode)),
        ctypes.c_int32(total_num_face_nodes),
        ctypes.c_int32(num_connected_boundary_faces),
        ctypes.c_int32(total_num_boundary_connections),
        pas_vars_ptr,
        value_locations_ptr,
        var_sharing_ptr,
        ctypes.c_int32(con_sharing),
    )
    if ret != 0:
        raise TecioError(
            f"teczne142 Error: zone_title={zone_title!r}, return_code={ret}"
        )


@requires_symbol("tecpolyzne142")
def tecpolyzne142(
    zone_title: str,
    zone_type: int | ZoneType,
    num_nodes: int,
    num_faces: int,
    num_elements: int,
    var_sharing: Sequence[int] | None = None,
    value_locations: Sequence[int | ValueLocation] | None = None,
    pas_vars: Sequence[int | VarStatus] | None = None,
    con_sharing: int = 0,
    strand: int = 0,
    solution_time: float = 0.0,
    total_num_face_nodes: int = 0,
    num_connected_boundary_faces: int = 0,
    total_num_boundary_connections: int = 0,
) -> None:
    """Create a polygonal or polyhedral zone (classic API).

    Args:
        zone_title (str): Zone title.
        zone_type (int | ZoneType): ``ZoneType.FEPOLYGON`` or ``ZoneType.FEPOLYHEDRON``.
        num_nodes (int): Number of nodes.
        num_faces (int): Number of faces.
        num_elements (int): Number of elements.
        var_sharing (Sequence[int] | None): List of zone indices to share variables from
            (0: No sharing, Null: no variables are shared from other zones). Must be
            same length as value_locations and pas_vars if provided.
        value_locations (Sequence[int | ValueLocation] | None): Per-variable
            data locations.
        pas_vars (Sequence[int | VarStatus] | None): ist of VarStatus enums or 0/1 for
            passive variables. Must be same length as value_locations and var_sharing if
            provided. None/null means all variables are active.
        con_sharing (int): Connectivity sharing source zone.
        strand (int): Strand ID for transient data (0 for static data, positive integer
            for transient data).
        solution_time (float): Solution time for transient data.
        total_num_face_nodes (int): Total face nodes.
        num_connected_boundary_faces (int): Boundary faces.
        total_num_boundary_connections (int): Boundary connections.

    Raises:
        TecioError: On C library error.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    # Create C array for passive variable flags
    pas_vars_ptr = None
    if pas_vars is not None:
        pas_vars_ptr = (ctypes.c_int32 * len(pas_vars))(*[
            _to_int_value(v, Boolean) for v in pas_vars
        ])

    # Create C array for value locations
    value_locations_ptr = None
    if value_locations is not None:
        value_locations_ptr = (ctypes.c_int32 * len(value_locations))(*[
            _to_int_value(v, ValueLocation) for v in value_locations
        ])

    # Create C array for variable sharing
    var_sharing_ptr = None
    if var_sharing is not None:
        var_sharing_ptr = (ctypes.c_int32 * len(var_sharing))(*list(var_sharing))

    ret = lib.tecpolyzne142(
        ctypes.c_char_p(zone_title.encode("utf-8")),
        ctypes.c_int32(_to_int_value(zone_type)),
        ctypes.c_int32(num_nodes),
        ctypes.c_int32(num_elements),
        ctypes.c_int64(num_faces),
        ctypes.c_int64(total_num_face_nodes),
        ctypes.c_double(solution_time),
        ctypes.c_int32(strand),
        ctypes.c_int32(0),  # Parent zone deprecated.
        ctypes.c_int32(num_connected_boundary_faces),
        ctypes.c_int32(total_num_boundary_connections),
        pas_vars_ptr,
        value_locations_ptr,
        var_sharing_ptr,
        ctypes.c_int32(con_sharing),
    )
    if ret != 0:
        raise TecioError(
            f"tecpolyzne142 Error: zone_title={zone_title!r}, return_code={ret}"
        )


@requires_symbol("tecznefemixed142")
def tecznefemixed142(
    zone_title: str,
    num_nodes: int,
    cell_shapes_per_section: Sequence[int | FeCellShape],
    num_elements_per_section: Sequence[int],
    grid_order_per_section: Sequence[int] | None = None,
    solution_time: float = 0.0,
    strand_id: int = 0,
    num_face_connections: int = 0,
    face_neighbor_mode: int | FaceNeighborMode = FaceNeighborMode.LOCAL_ONE_TO_ONE,
    passive_var_list: Sequence[int | VarStatus] | None = None,
    value_location: Sequence[int | ValueLocation] | None = None,
    share_var_from_zone: Sequence[int] | None = None,
    share_connectivity_from_zone: int = 0,
) -> None:
    """Create a mixed finite-element zone (classic API).

    A mixed-element zone groups cells into 1-16 sections; every cell within
    one section shares the same shape and grid order. All sections in a
    zone must share the same spatial dimensionality (all line, all surface,
    or all volume cell types), not a mix of them.

    Args:
        zone_title (str): Zone title.
        num_nodes (int): Total number of nodes for the zone.
        cell_shapes_per_section (Sequence[int | FeCellShape]): Cell shape for
            each section. Length determines the number of sections (1-16).
        num_elements_per_section (Sequence[int]): Number of elements in each
            section. Must be the same length as cell_shapes_per_section.
        grid_order_per_section (Sequence[int] | None): Grid order (1-4) for
            each section. None defaults to linear (1) for every section.
        solution_time (float): Solution time.
        strand_id (int): Strand ID for transient data (0 for static data,
            positive integer for transient data).
        num_face_connections (int): Number of face connections that will be
            passed to ``tecface142``.
        face_neighbor_mode (int | FaceNeighborMode): Face-neighbor mode.
        passive_var_list (Sequence[int | VarStatus] | None): List of
            VarStatus enums or 0/1 for passive variables. None means all
            variables are active.
        value_location (Sequence[int | ValueLocation] | None): Per-variable
            data locations. None means all variables are nodal.
        share_var_from_zone (Sequence[int] | None): List of zone indices to
            share variables from. None means no variable sharing.
        share_connectivity_from_zone (int): Zone index to share connectivity
            from (0 = none).

    Raises:
        TecioError: On C library error.
        ValueError: If cell_shapes_per_section and num_elements_per_section
            (or grid_order_per_section, if provided) have mismatched
            lengths, or there are not between 1 and 16 sections.

    Note:
        The basis function per section is always 0 (the only value the C
        library currently accepts), so it isn't exposed as a parameter here.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    num_sections = len(cell_shapes_per_section)
    if not 1 <= num_sections <= 16:
        raise ValueError(f"tecznefemixed142 requires 1-16 sections, got {num_sections}")
    if len(num_elements_per_section) != num_sections:
        raise ValueError(
            "cell_shapes_per_section and num_elements_per_section must be "
            f"the same length ({num_sections} != "
            f"{len(num_elements_per_section)})"
        )
    if grid_order_per_section is None:
        grid_order_per_section = [1] * num_sections
    elif len(grid_order_per_section) != num_sections:
        raise ValueError(
            "grid_order_per_section must be the same length as "
            f"cell_shapes_per_section ({num_sections} != "
            f"{len(grid_order_per_section)})"
        )

    cell_shapes_ptr = (ctypes.c_int32 * num_sections)(*[
        _to_int_value(v, FeCellShape) for v in cell_shapes_per_section
    ])
    grid_order_ptr = (ctypes.c_int32 * num_sections)(*list(grid_order_per_section))
    basis_fn_ptr = (ctypes.c_int32 * num_sections)(*([0] * num_sections))
    num_elements_ptr = (ctypes.c_int64 * num_sections)(*list(num_elements_per_section))

    # Create C array for passive variable flags
    pas_vars_ptr = None
    if passive_var_list is not None:
        pas_vars_ptr = (ctypes.c_int32 * len(passive_var_list))(*[
            _to_int_value(v, Boolean) for v in passive_var_list
        ])

    # Create C array for value locations
    value_loc_ptr = None
    if value_location is not None:
        value_loc_ptr = (ctypes.c_int32 * len(value_location))(*[
            _to_int_value(v, ValueLocation) for v in value_location
        ])

    # Create C array for variable sharing
    share_var_ptr = None
    if share_var_from_zone is not None:
        share_var_ptr = (ctypes.c_int32 * len(share_var_from_zone))(
            *list(share_var_from_zone)
        )

    ret = lib.tecznefemixed142(
        ctypes.c_char_p(zone_title.encode("utf-8")),
        ctypes.c_int64(num_nodes),
        ctypes.c_int32(num_sections),
        cell_shapes_ptr,
        grid_order_ptr,
        basis_fn_ptr,
        num_elements_ptr,
        ctypes.c_double(solution_time),
        ctypes.c_int32(strand_id),
        ctypes.c_int32(num_face_connections),
        ctypes.c_int32(_to_int_value(face_neighbor_mode)),
        pas_vars_ptr,
        value_loc_ptr,
        share_var_ptr,
        ctypes.c_int32(share_connectivity_from_zone),
    )
    if ret != 0:
        raise TecioError(
            f"tecznefemixed142 Error: zone_title={zone_title!r}, return_code={ret}"
        )


# -- Data writing ----------------------------------------------------------------------


@requires_symbol("tecdat142")
def tecdat142(
    field_data: npt.ArrayLike,
    is_double: bool = True,
) -> None:
    """Write field data to the current zone (classic API).

    Args:
        field_data (npt.ArrayLike): Array of field values.
        is_double (bool): True for double precision, False for single.

    Raises:
        TecioError: On C library error.

    Important:
        Data must be written in the order specified by dataset variables and zone
        definition

    Hint:
        If variable is shared or passive, skip writing data for that variable

    Note:
        For ORDERED zones: data should be ordered with the I dimension varying
        fastest, then J, then K

    Note:
        For FE zones: data should be ordered by node index for nodal variables, and by
        element index for cell-centered variables
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    # Convert to appropriate numpy array
    if is_double:
        arr = np.ascontiguousarray(field_data, dtype=np.float64)
        data_ptr = arr.ctypes.data_as(ctypes.c_void_p)
        is_double_c = ctypes.c_int32(1)
    else:
        arr = np.ascontiguousarray(field_data, dtype=np.float32)
        data_ptr = arr.ctypes.data_as(ctypes.c_void_p)
        is_double_c = ctypes.c_int32(0)

    ret = lib.tecdat142(
        ctypes.c_int32(arr.size),
        data_ptr,
        is_double_c,
    )
    if ret != 0:
        raise TecioError(
            f"tecdat142 Error: n={arr.size}, is_double={is_double}, return_code={ret}"
        )


# -- Connectivity writing --------------------------------------------------------------
@requires_symbol("tecnode142")
def tecnode142(nodes: npt.ArrayLike) -> None:
    """Write node connectivity for an FE zone (classic API).

    Args:
        nodes (npt.ArrayLike): Array of 1-based node indices.

    Raises:
        TecioError: On C library error.

    Important:
        Node indices are 1-based (Fortran convention)

    Hint:
        - For FEBRICK elements: 8 nodes per element.
        - For FETETRAHEDRON elements: 4 nodes per element.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    nodes_array = np.ascontiguousarray(nodes, dtype=np.int32)
    n = ctypes.c_int32(nodes_array.size)
    nodes_ptr = nodes_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    ret = lib.tecnode142(
        ctypes.byref(n),
        nodes_ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecnode142 Error: n_nodes={nodes_array.size}, return_code={ret}"
        )


@requires_symbol("tecface142")
def tecface142(face_connections: npt.ArrayLike) -> None:
    """Write face-neighbor connections (classic API).

    Args:
        face_connections (npt.ArrayLike): Array of face connection data.

    Raises:
        TecioError: On C library error.

    Note:
        Used to specify face-to-face connectivity between zones.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    face_conn_array = np.ascontiguousarray(face_connections, dtype=np.int32)
    face_conn_ptr = face_conn_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    ret = lib.tecface142(face_conn_ptr)
    if ret != 0:
        raise TecioError(f"tecface142 Error: return_code={ret}")


@requires_symbol("tecpolyface142")
def tecpolyface142(
    face_node_counts: npt.ArrayLike | None,
    face_nodes: npt.ArrayLike,
    face_left_elems: npt.ArrayLike,
    face_right_elems: npt.ArrayLike,
    num_faces: int | None = None,
) -> None:
    """Write face data for polygonal/polyhedral zones (classic API).

    Args:
        face_node_counts (npt.ArrayLike | None): Number of nodes per face, or ``None``
            for a polygonal zone (every face has two nodes).
        face_nodes (npt.ArrayLike): Concatenated face node indices.
        face_left_elems (npt.ArrayLike): Left element per face.
        face_right_elems (npt.ArrayLike): Right element per face.
        num_faces (int | None): Number of faces, required when ``face_node_counts`` is
            ``None``.

    Raises:
        TecioError: On C library error.
        ValueError: If ``face_node_counts`` and ``num_faces`` are both ``None``.

    Note:
        ``face_node_counts`` has one entry per face.

    Note:
        ``face_nodes`` is a concatenated list of all face node indices.

    Hint:
        Element indices are 1-based; 0 indicates boundary.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    face_nodes_array = np.ascontiguousarray(face_nodes, dtype=np.int32)
    face_left_elems_array = np.ascontiguousarray(face_left_elems, dtype=np.int32)
    face_right_elems_array = np.ascontiguousarray(face_right_elems, dtype=np.int32)

    face_node_counts_ptr = None
    if face_node_counts is None:
        if num_faces is None:
            raise ValueError("num_faces is required when face_node_counts is None")
    else:
        face_node_counts_array = np.ascontiguousarray(face_node_counts, dtype=np.int32)
        num_faces = face_node_counts_array.size
        face_node_counts_ptr = face_node_counts_array.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int32)
        )
    num_faces_c = ctypes.c_int32(num_faces)
    face_nodes_ptr = face_nodes_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    face_left_elems_ptr = face_left_elems_array.ctypes.data_as(
        ctypes.POINTER(ctypes.c_int32)
    )
    face_right_elems_ptr = face_right_elems_array.ctypes.data_as(
        ctypes.POINTER(ctypes.c_int32)
    )

    ret = lib.tecpolyface142(
        ctypes.byref(num_faces_c),
        face_node_counts_ptr,
        face_nodes_ptr,
        face_left_elems_ptr,
        face_right_elems_ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecpolyface142 Error: num_faces={num_faces}, return_code={ret}"
        )


@requires_symbol("tecpolybconn142")
def tecpolybconn142(
    boundary_connection_counts: npt.ArrayLike,
    boundary_connection_elems: npt.ArrayLike,
    boundary_connection_zones: npt.ArrayLike,
) -> None:
    """Write boundary connections for poly zones (classic API).

    Args:
        boundary_connection_counts (npt.ArrayLike): Connections per boundary face.
        boundary_connection_elems (npt.ArrayLike): 1-based element index of every
            connection, concatenated over the boundary faces.
        boundary_connection_zones (npt.ArrayLike): 1-based zone number of every
            connection, same length as ``boundary_connection_elems``. Required: TecIO
            crashes the process on a null array. A zone must already exist (the
            current zone counts), so a connection cannot point to a later zone.

    Raises:
        TecioError: On C library error.
        ValueError: If the array lengths are inconsistent. TecIO reads the arrays
            without a bounds check, so a short array would be read past its end.

    Note:
        Used to specify connectivity to neighboring zones at boundaries.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    bconn_counts_array = np.ascontiguousarray(
        boundary_connection_counts, dtype=np.int32
    )
    bconn_elems_array = np.ascontiguousarray(boundary_connection_elems, dtype=np.int32)
    total = int(bconn_counts_array.sum())
    if bconn_elems_array.size != total:
        raise ValueError(
            f"boundary_connection_elems has {bconn_elems_array.size} entries but "
            f"boundary_connection_counts sums to {total}."
        )

    num_boundary_faces = ctypes.c_int32(bconn_counts_array.size)
    bconn_counts_ptr = bconn_counts_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    bconn_elems_ptr = bconn_elems_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    # TecIO reads the zone numbers as INT32, not INT16 as the Fortran documentation
    # states: an int16 array is read as pairs and yields garbage zone numbers.
    bconn_zones_array = np.ascontiguousarray(boundary_connection_zones, dtype=np.int32)
    if bconn_zones_array.size != total:
        raise ValueError(
            f"boundary_connection_zones has {bconn_zones_array.size} entries but "
            f"boundary_connection_counts sums to {total}."
        )
    bconn_zones_ptr = bconn_zones_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    ret = lib.tecpolybconn142(
        ctypes.byref(num_boundary_faces),
        bconn_counts_ptr,
        bconn_elems_ptr,
        bconn_zones_ptr,
    )
    if ret != 0:
        raise TecioError(
            f"tecpolybconn142 Error: num_boundary_faces={bconn_counts_array.size}, "
            f"return_code={ret}"
        )


# -- Auxiliary data --------------------------------------------------------------------


@requires_symbol("tecauxstr142")
def tecauxstr142(name: str, value: str) -> None:
    """Add dataset-level auxiliary data (classic API).

    Args:
        name (str): Auxiliary data name.
        value (str): Auxiliary data value.

    Raises:
        TecioError: On C library error.

    Note:
        Must be called after ``tecini142`` but before first ``teczne142``.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ret = lib.tecauxstr142(
        ctypes.c_char_p(name.encode("utf-8")),
        ctypes.c_char_p(value.encode("utf-8")),
    )
    if ret != 0:
        raise TecioError(
            f"tecauxstr142 Error: name={name!r}, value={value!r}, return_code={ret}"
        )


@requires_symbol("tecvauxstr142")
def tecvauxstr142(var: int, name: str, value: str) -> None:
    """Add variable-level auxiliary data (classic API).

    Args:
        var (int): 1-based variable index.
        name (str): Auxiliary data name.
        value (str): Auxiliary data value.

    Raises:
        TecioError: On C library error.

    Note:
        Must be called after ``tecini142`` but before first ``teczne142``.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    var_c = ctypes.c_int32(var)

    ret = lib.tecvauxstr142(
        ctypes.byref(var_c),
        ctypes.c_char_p(name.encode("utf-8")),
        ctypes.c_char_p(value.encode("utf-8")),
    )
    if ret != 0:
        raise TecioError(
            f"tecvauxstr142 Error: var={var}, name={name!r}, value={value!r}, "
            f"return_code={ret}"
        )


@requires_symbol("teczauxstr142")
def teczauxstr142(name: str, value: str) -> None:
    """Add zone-level auxiliary data (classic API).

    Args:
        name (str): Auxiliary data name.
        value (str): Auxiliary data value.

    Raises:
        TecioError: On C library error.

    Note:
        Must be called after ``teczne142`` for the current zone.
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ret = lib.teczauxstr142(
        ctypes.c_char_p(name.encode("utf-8")),
        ctypes.c_char_p(value.encode("utf-8")),
    )
    if ret != 0:
        raise TecioError(
            f"teczauxstr142 Error: name={name!r}, value={value!r}, return_code={ret}"
        )


# -- User-defined data (custom records) ------------------------------------------------


@requires_symbol("tecusr142")
def tecusr142(user_rec: str) -> None:
    """Write a user-defined data record (classic API).

    Args:
        user_rec (str): User-defined data string.

    Raises:
        TecioError: On C library error.

    Note:
        Used to write custom data records into the file.
        Data is preserved but not interpreted by Tecplot
    """
    assert lib is not None  # narrowed: @requires_symbol already checked
    ret = lib.tecusr142(ctypes.c_char_p(user_rec.encode("utf-8")))
    if ret != 0:
        raise TecioError(f"tecusr142 Error: user_rec={user_rec!r}, return_code={ret}")
