"""Write Tecplot SZL (``.szplt``) files via the TecIO C library.

Supports lazy-open (deferred until first zone write), buffered auxiliary
data, and automatic dtype inference from NumPy arrays.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
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
from ._meta import ZoneMeta
from ._writer import (
    PassiveVarsSpec,
    PreparedFEZone,
    PreparedOrderedZone,
    TecplotWriter,
    VarSharingSpec,
    normalize_precision,
    prepare_fe_zone,
    prepare_ordered_zone,
    var_sharing_for_libtecio,
    zone_for_libtecio,
)

# --------------------------------------------------------------------------------------
# Module-level constants
# --------------------------------------------------------------------------------------

# FE zone types that use tec_zone_create_fe
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})


# ======================================================================================
# Helpers
# ======================================================================================


def _infer_data_type(dt: DataType | np.dtype) -> DataType:
    """Return the closest C-supported DataType for a DataType or NumPy dtype."""
    # Mapping of NumPy type categories to closest DataType
    closest_dtype_map = {
        np.dtype(np.float64): DataType.DOUBLE,
        np.dtype(np.float32): DataType.FLOAT,
        np.dtype(np.float16): DataType.FLOAT,  # promote half → float
        np.dtype(np.int64): DataType.INT32,  # promote 64-bit → INT32
        np.dtype(np.int32): DataType.INT32,
        np.dtype(np.int16): DataType.INT16,
        np.dtype(np.int8): DataType.BYTE,  # small ints → BYTE
        np.dtype(np.uint8): DataType.BYTE,
    }
    if isinstance(dt, DataType):
        return dt

    dt_np = np.dtype(dt)

    for key in closest_dtype_map:
        if dt_np == key:
            return closest_dtype_map[key]

    if np.issubdtype(dt_np, np.floating):
        return DataType.FLOAT if dt_np.itemsize <= 4 else DataType.DOUBLE

    if np.issubdtype(dt_np, np.signedinteger):
        return DataType.INT16 if dt_np.itemsize <= 2 else DataType.INT32

    if np.issubdtype(dt_np, np.unsignedinteger):
        return DataType.BYTE

    raise ValueError(f"Unsupported dtype: {dt_np}")


# ASCII/enum-name aliases for the two precision options
_STR_TO_PRECISION: dict[str, DataType] = {
    "single": DataType.FLOAT,
    "float": DataType.FLOAT,
    "double": DataType.DOUBLE,
}


def _resolve_written_type(inferred: DataType, precision: DataType | None) -> DataType:
    """Return the :class:`DataType` actually written for one variable.

    *precision* overrides *inferred* only when *inferred* is itself a floating-point
    type (FLOAT or DOUBLE). Integer-inferred variables (INT32/INT16/BYTE) always keep
    their own inferred type, unaffected by *precision*. A variable holding a meaningful
    integer (a CPU number, an index, a count) should never be silently coerced by a
    setting that's conceptually about floating-point precision.
    """
    if precision is None:
        return inferred
    if inferred in (DataType.FLOAT, DataType.DOUBLE):
        return precision
    return inferred


# ======================================================================================
# Local functions
# ======================================================================================


def _write_data(
    handle: ctypes.c_void_p,
    zone_num: int,
    var_num: int,
    data: npt.ArrayLike,
    dt: np.dtype | DataType | None = None,
) -> None:
    """Write a single variable's data array to an SZL file.

    Infers the data type from the array dtype and dispatches to the
    appropriate C write function. Arrays are ravelled in Fortran order.

    Args:
        handle (ctypes.c_void_p): C library file handle.
        zone_num (int): 1-based zone index.
        var_num (int): 1-based variable index.
        data (npt.ArrayLike): Array of values to write.
        dt (np.dtype | DataType | None): Optional explicit data type override.

    Output defaults:
        1. Inferrs data_typeype for nummpy arrays
        2. Defaults to double precision for array-like (list, tuple, etc)
        3. Optionally casts to input DataType or numpy dtype.
        4. Assumes data is in the correct shape and order (column major / Fortran order)
    """
    # Mappings between C-supported data types and numpy dtypes
    dtype_to_datatype: dict[np.dtype, DataType] = {
        np.dtype(np.float64): DataType.DOUBLE,
        np.dtype(np.float32): DataType.FLOAT,
        np.dtype(np.int32): DataType.INT32,
        np.dtype(np.int16): DataType.INT16,
        np.dtype(np.uint8): DataType.BYTE,
    }
    datatype_to_dtype: dict[DataType, np.dtype] = {
        DataType.DOUBLE: np.dtype(np.float64),
        DataType.FLOAT: np.dtype(np.float32),
        DataType.INT32: np.dtype(np.int32),
        DataType.INT16: np.dtype(np.int16),
        DataType.BYTE: np.dtype(np.uint8),
    }

    if dt is not None:
        data_type = _infer_data_type(dt)
        arr = np.ascontiguousarray(data, dtype=datatype_to_dtype[data_type]).ravel(
            order="F"
        )
    else:
        arr = np.ascontiguousarray(data).ravel(order="F")
        data_type = dtype_to_datatype[arr.dtype]

    if data_type == DataType.DOUBLE:
        libtecio.tec_zone_var_write_double_values(handle, zone_num, var_num, arr)
    elif data_type == DataType.FLOAT:
        libtecio.tec_zone_var_write_float_values(handle, zone_num, var_num, arr)
    elif data_type == DataType.INT32:
        libtecio.tec_zone_var_write_int32_values(handle, zone_num, var_num, arr)
    elif data_type == DataType.INT16:
        libtecio.tec_zone_var_write_int16_values(handle, zone_num, var_num, arr)
    elif data_type == DataType.BYTE:
        libtecio.tec_zone_var_write_uint8_values(handle, zone_num, var_num, arr)
    else:
        raise ValueError(f"Unsupported DataType: {data_type!r}")


def _write_connectivity(
    handle: ctypes.c_void_p,
    zone_num: int,
    node_map: npt.ArrayLike,
    face_neighbors: npt.ArrayLike | None = None,
) -> None:
    """Write FE zone connectivity: node map and optional face-neighbor connections.

    Integer width (32 or 64 bit) is chosen automatically from the maximum index value in
    each array.

    Args:
        handle (ctypes.c_void_p): C library file handle.
        zone_num (int): 1-based zone index.
        node_map (npt.ArrayLike): Connectivity array, 1-based node indices.
        face_neighbors (npt.ArrayLike | None): Optional face-neighbor connection array.

    Note:
        Both arrays are written using the minimum integer width capable of representing
        the maximum index value present in each array. No copy is made for C-contiguous
        input arrays — ravel(order="C") returns a flat view of the node_map or
        face_neighbors without making a copy

    Note:
        Node and face-neighbor integer widths are chosen independently based on the
        maximum value in each respective array.
    """
    # Node map
    node_map_flat = np.ascontiguousarray(node_map).ravel(order="C")
    if node_map_flat.max() > np.iinfo(np.int32).max:
        libtecio.tec_zone_node_map_write64(handle, zone_num, node_map_flat)
    else:
        libtecio.tec_zone_node_map_write32(handle, zone_num, node_map_flat)

    # Face neighbors (optional)
    if face_neighbors is not None:
        face_nbr_flat = np.ascontiguousarray(face_neighbors).ravel(order="C")
        if face_nbr_flat.max() > np.iinfo(np.int32).max:
            libtecio.tec_zone_face_nbr_write_connections64(
                handle, zone_num, face_nbr_flat
            )
        else:
            libtecio.tec_zone_face_nbr_write_connections32(
                handle, zone_num, face_nbr_flat
            )


def _write_zone_aux_data(
    handle: ctypes.c_void_p, aux: dict[int, dict[str, Any]]
) -> None:
    """Write zone-level auxiliary data.

    Args:
        handle (ctypes.c_void_p): C library file handle.
        aux (dict[int, dict[str, Any]]): Mapping of
            ``{zone_index: {name: value}}``.

    Note:
        Aux data should be structured as {zone_idx: {name, value}}
    """
    for zone_idx, subdict in aux.items():
        for name, value in subdict.items():
            libtecio.tec_zone_add_aux_data(handle, zone_idx, str(name), str(value))


def _write_variable_aux_data(
    handle: ctypes.c_void_p, aux: dict[int, dict[str, Any]]
) -> None:
    """Write variable-level auxiliary data.

    Args:
        handle (ctypes.c_void_p): C library file handle.
        aux (dict[int, dict[str, Any]]): Mapping of ``{var_index: {name: value}}``.

    Hint:
        Aux data should be structured as ``{var_idx: {name, value}}``
    """
    for var_idx, subdict in aux.items():
        for name, value in subdict.items():
            libtecio.tec_var_add_aux_data(handle, var_idx, str(name), str(value))


def _write_dataset_aux_data(handle: ctypes.c_void_p, aux: dict[str, Any]) -> None:
    """Write dataset-level auxiliary data.

    Args:
        handle (ctypes.c_void_p): C library file handle.
        aux (dict[str, Any]): Mapping of ``{name: value}``.

    Hint:
        Aux data should be structured as ``{var_idx: {name, value}}``
    """
    for name, value in aux.items():
        libtecio.tec_data_set_add_aux_data(handle, str(name), str(value))


def _write_aux_data(handle: ctypes.c_void_p, aux: dict[str, dict[Any, Any]]) -> None:
    """Write a combined auxiliary data dictionary to the file.

    Args:
        handle (ctypes.c_void_p): C library file handle.
        aux (dict[str, dict[Any]]): Dict with keys ``"AUXDATA"``, ``"AUXVAR"``,
            ``"AUXZONE"``, each containing the appropriate nested structure.

    Example:
        {
            "AUXDATASET":
                {name1: value1}
                {name2: value2}
            "AUXVAR": {
                        1:
                           {name1: value1}
                        2:
                           {name1: value1}
                    },
            "AUXZONE": {
                        1:
                            {name1: value1}
                            {name2: value2}
                    }
        }
    """
    for auxtype, auxdict in aux.items():
        if auxtype.lower() == "auxdata":
            for name, value in auxdict.items():
                libtecio.tec_data_set_add_aux_data(handle, str(name), str(value))
        elif auxtype.lower() == "auxvar":
            for var, subdict in auxdict.items():
                for name, value in subdict.items():
                    libtecio.tec_var_add_aux_data(handle, var, name, value)
        elif auxtype.lower() == "auxzone":
            for zone, subdict in auxdict.items():
                for name, value in subdict.items():
                    libtecio.tec_zone_add_aux_data(handle, zone, name, value)


# ======================================================================================


# ======================================================================================
# TecplotSzlWriter
# ======================================================================================


class TecplotSzlWriter(TecplotWriter):
    """Write Tecplot SZL (``.szplt``) files with a lazy-open file handle.

    Supports lazy-open: if *variables* is ``None`` at construction, the file is created
    on the first zone write. Auxiliary data is buffered and flushed automatically before
    the first zone.

    The tecio library requires a list of variables when the output file is
    opened. However if writing data on the fly, it may be beneficial to store file
    outputs until the first zone is passed to the writer. Then file header will be
    garanteed to be consistent with the first zone variables.

    For the SZL API, file contents can be written out of order after creating zones.

    Args:
        path:         Output file path.
        title:        Dataset title.
        variables:    Variable name list. ``None`` defers file creation.
        file_type:    File type enum (FULL, GRID, or SOLUTION).
        precision:    Optional whole file floating point precision override
                      (:attr:`DataType.FLOAT`/``"single"`` or
                      :attr:`DataType.DOUBLE`/``"double"``). Defaults to ``None``: each
                      variable's type is inferred automatically from its own array's
                      dtype.

    Attributes:
        precision:    Whole-file floating-point override, or ``None`` for automatic
                      per-variable inference.
        handle:       Raw C file handle from ``tec_file_writer_open``, or ``None``
                      before the file has been opened (or after :meth:`close`).
        current_var:  0-based index of the variable most recently passed to a
                      libtecio write call. Set only while a zone's variable data is
                      being written, and only ever read by user code inspecting the
                      writer, not consumed internally; useful for diagnosing which
                      variable was in progress if a write fails partway through a
                      zone.
    """

    def __init__(
        self,
        path: str,
        title: str = "untitled",
        variables: list[str] | None = None,
        file_type: FileType = FileType.FULL,
        *,
        precision: DataType | str | None = None,
    ) -> None:
        """Store minimum necessary info until first zone is ready to write.

        Raises:
            ValueError: If *precision* is neither ``None`` nor
                        :attr:`DataType.FLOAT`/:attr:`DataType.DOUBLE` (or a recognized
                        string alias for one of them).
        """
        # Set before super().__init__(), which calls self._open() when
        # variables is already known (eager open), and _open() needs this.
        self.precision: DataType | None = normalize_precision(
            precision, allow_none=True
        )
        self.current_var = -1
        self.handle: ctypes.c_void_p | None = None
        super().__init__(path, title, variables, file_type)

    @property
    def _file_format(self) -> str:
        return "szplt"

    # -- Validation checks and errror handling -----------------------------------------

    def _check_handle(self) -> ctypes.c_void_p:
        """Return file handle catching errors if the writer has already been closed.

        This ensures the that each libtecio call will execute or return an appropriate
        ValueError.
        """
        if self.handle is None:
            raise RuntimeError(f"I/O operation on closed file: '{self.path}'")
        else:
            return self.handle

    # -- File lifecycle ----------------------------------------------------------------

    def _open(self, var_names: list[str]) -> None:
        """Open the file handle. Called exactly once on the first zone."""
        self.variables = var_names
        self.handle = libtecio.tec_file_writer_open(
            filename=self.path,
            variables=self.variables,
            title=self.title,
            file_type=self.file_type,
            use_szl=1,
        )
        self._meta.set_variables(self.variables)

    def close(self) -> None:
        """Finalise and flush the file (safe to call more than once)."""
        if self.handle is not None:
            self.flush_aux()
            libtecio.tec_file_writer_close(self.handle)
            self.handle = None

    # -- Aux data: only the per-item write differs from the shared base ----------------

    def _write_dataset_aux_item(self, name: str, value: str) -> None:
        libtecio.tec_data_set_add_aux_data(self._check_handle(), name, value)

    def _write_var_aux_item(
        self, one_based_var_index: int, name: str, value: str
    ) -> None:
        libtecio.tec_var_add_aux_data(
            self._check_handle(), one_based_var_index, name, value
        )

    # -- Structured zone writer --------------------------------------------------------

    def write_ordered_zone(
        self,
        data: Sequence[npt.ArrayLike],
        *,
        title: str | None = None,
        variables: list[str] | None = None,
        value_locations: Sequence[ValueLocation] | None = None,
        passive_vars: PassiveVarsSpec | None = None,
        var_sharing: VarSharingSpec | None = None,
        solution_time: float = 0.0,
        strand_id: int = 0,
        aux: dict[str, Any] | None = None,
        datapacking: DataPacking | str = DataPacking.BLOCK,
        flush: bool = False,
    ) -> None:
        """Write a complete IJK-ordered zone.

        Dimensions are inferred from the first array's shape. Arrays may be 1-D, 2-D, or
        3-D; missing trailing dimensions default to 1.

        Args:
            data (Sequence[npt.ArrayLike]):
                One NumPy array per active (non-passive, non-shared) dataset variable,
                in dataset order. Array shape is used to infer ``imax``, ``jmax``, and
                ``kmax``; Fortran (column-major) order is assumed. Pass ``None`` to
                write a zone header only.
            title (str):
                Zone title. Defaults to ``"IJK_Zone_{index}"``, this zone's 0-based
                index.
            variables (list[str]):
                Variable name list. Required on the first call when the file has not
                been opened yet (lazy-open path); ignored once the file is already
                initialised. Default to ``[V1, V2, V3, ...]`` if not provided in open or
                zone call.
            value_locations (Sequence[ValueLocation]):
                Per-active-variable :class:`~tecio.ValueLocation`.  Defaults to all
                ``NODAL``.
            passive_vars (list | tuple | set | None):
                Variables to mark passive: a list, tuple, or set of variable names, or
                of 0-based indices, e.g. ``{"x"}`` or ``[3]``. Defaults to all active
                (``passive_vars=None``).
            var_sharing (Mapping[int | str, int]):
                Variables to share from another zone: a ``{variable: source zone}``
                mapping, name or 0-based index to 0-based zone index, e.g. ``{"x": 0,
                "y": 0}``. Defaults to no sharing.
            solution_time (float):
                Solution time for transient data. Default to ``0.0`` if not defined.
            strand_id (int):
                Strand ID for transient data. Use ``0`` for static zones. Default to
                ``0`` (static) if not defined.
            aux (dict[str, str | float | int]):
                Zone-level auxiliary data as ``{name: value}`` string pairs.
            datapacking (DataPacking | str):
                In :attr:`~tecio.DataPacking.POINT` format, the values for all variables
                are given for the first point, then the second point, and so on. In
                :attr:`~tecio.DataPacking.BLOCK` format, all of the values for the first
                variable are given in a block, then all of the values for the second
                variable, then all of the values for the third, and so
                forth. :attr:`~tecio.DataPacking.BLOCK` format must be used for
                cell-centered data and polyhedral zones
                (:attr:`~tecio.ZoneType.FEPOLYGON`/:attr:`~tecio.ZoneType.FEPOLYHEDRAL`),
                as well as for all binary data. `POINT` and `BLOCK` keyword strings also
                accepted. Defaults to :attr:`~tecio.DataPacking.BLOCK`.
            flush (bool):
                If ``True``, flush this zone and all previous data to a temporary
                intermediate file immediately after writing, releasing data from
                memory. Defaults to ``False``. Useful when memory is a concern, but adds
                the overhead of a disk write. Temporary files are merged back into the
                final output file when :meth:`close` is called.

        Raises:
           NotImplementedError:
               If *datapacking* is :attr:`~tecio.libtecio.DataPacking.POINT`.
            TypeError:
               If *passive_vars*/*var_sharing* isn't a list/tuple/set/mapping as
               appropriate, or mixes variable names and indices.
            ValueError:
               If I/O operation attempted on closed or None file handle, or a variable
               is named in both *passive_vars* and *var_sharing*.
            RuntimeError:
               If attempting to write variable aux before variables are defined.
            KeyError:
               A *passive_vars*/*var_sharing* name doesn't match any dataset variable.
            IndexError:
               A *passive_vars*/*var_sharing* index is out of range.

        Note:
            If the file is already open, ``data`` and ``variables`` may be omitted to
            write a zone header only. If the file has not been opened yet, ``variables``
            must be provided on this call.

        Note:
            Data arrays are written as DOUBLE precision by default. To write other
            types, cast the NumPy arrays before calling (e.g.
            ``arr.astype(np.float32)``).

        Note:
            Separate grid files (where all variables are cell-centred) are not handled
            automatically; use the low-level API for that case.
        """
        # Validate inputs
        if isinstance(datapacking, str):
            try:
                datapacking = DataPacking[datapacking.upper()]
            except KeyError:
                raise ValueError(
                    f"datapacking={datapacking!r} is not a recognised value; "
                    "use DataPacking.BLOCK or the string 'BLOCK'."
                ) from None
        if datapacking != DataPacking.BLOCK:
            raise NotImplementedError(
                "DATAPACKING=POINT is an ASCII-only layout and is not supported "
                "by the SZL binary format. Use DataPacking.BLOCK (the default) "
                "or write to a .dat file instead."
            )

        # Convert input data to NumPy arrays
        arrays: list[npt.NDArray] = [np.asarray(arr) for arr in data]

        # Set default title if none provided
        if title is None:
            title = f"IJK_Zone_{self.current_zone + 1}"

        # Set default variable names if none provided. Only relevant if file lazily
        # loaded and first zone
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(arrays) + 1)]

        # Open the file if lazily loaded and flush buffered aux data
        if self.handle is None:
            self._open(variables)
        if self.current_zone == -1:
            self.flush_aux()

        # Infer per-variable data types from array dtypes
        variable_types = [
            _resolve_written_type(_infer_data_type(arr.dtype), self.precision)
            for arr in arrays
        ]

        prepared: PreparedOrderedZone | None = prepare_ordered_zone(
            arrays,
            variable_types,
            value_locations=value_locations,
            passive_vars=passive_vars,
            var_sharing=var_sharing,
            dataset_variables=self._check_variables(),
            meta=self._meta,
        )
        if prepared is None:
            return
        passive_flags = prepared.passive_vars
        share_idx = prepared.var_sharing
        active_var_idx = prepared.active_var_idx
        imax, jmax, kmax = prepared.imax, prepared.jmax, prepared.kmax
        value_locations_global = prepared.value_locations_global
        variable_types_global = prepared.variable_types_global

        # Write zone header. The C library assigns and returns a 1-based zone
        # number; converted to 0-based for the public current_zone attribute.
        self.current_zone = (
            libtecio.tec_zone_create_ijk(
                self._check_handle(),
                title,
                imax,
                jmax,
                kmax,
                var_types=variable_types_global,
                value_locations=value_locations_global,
                var_sharing=var_sharing_for_libtecio(share_idx),
                pas_vars=passive_flags,
            )
            - 1
        )
        zone_num = self.current_zone + 1

        # Unsteady options
        if strand_id != 0 or solution_time != 0.0:
            libtecio.tec_zone_set_unsteady_options(
                handle=self._check_handle(),
                zone=zone_num,
                strand=strand_id,
                solution_time=solution_time,
            )

        # Write aux data
        if aux is not None:
            _write_zone_aux_data(self._check_handle(), {zone_num: aux})

        # Write active data only
        for var_idx, arr, dtype in zip(
            active_var_idx, arrays, variable_types, strict=True
        ):
            self.current_var = var_idx
            _write_data(
                self._check_handle(),
                zone_num=zone_num,
                var_num=var_idx + 1,
                data=arr,
                dt=dtype,
            )

        # Flush zone data, releasing memory
        if flush:
            libtecio.tec_file_writer_flush(self._check_handle())

        # Finally set zone metadata after successfully completing TecIO calls
        self._meta.record_zone(
            ZoneMeta(
                index=self.current_zone,
                title=title,
                zone_type=ZoneType.ORDERED,
                solution_time=solution_time,
                strand_id=strand_id,
                num_aux_items=len(aux) if aux else 0,
                dimensions=(imax, jmax, kmax),
                value_locations=tuple(value_locations_global),
                passive_vars=tuple(bool(p) for p in passive_flags),
                shared_vars=tuple(share_idx),
                data_types=tuple(variable_types_global),
            )
        )

    # -- Unstructured zone writer ------------------------------------------------------

    def write_fe_zone(
        self,
        data: Sequence[npt.ArrayLike],
        zone_type: ZoneType,
        node_map: npt.ArrayLike | None = None,
        *,
        title: str | None = None,
        variables: list[str] | None = None,
        value_locations: Sequence[ValueLocation] | None = None,
        passive_vars: PassiveVarsSpec | None = None,
        var_sharing: VarSharingSpec | None = None,
        con_sharing: int | None = None,
        face_neighbors: npt.ArrayLike | None = None,
        face_neighbor_mode: FaceNeighborMode | None = None,
        solution_time: float = 0.0,
        strand_id: int = 0,
        aux: dict[str, Any] | None = None,
        datapacking: DataPacking | str = DataPacking.BLOCK,
        flush: bool = False,
    ) -> None:
        """Write a complete finite-element zone.

        Node and cell counts are inferred from *node_map*, or if *node_map* is omitted
        from the zone referenced by *con_sharing*. The 32- or 64-bit write path is
        chosen automatically from the max index.

        Args:
            data (Sequence[npt.ArrayLike]):
                Sequence of 1-D arrays, one per active (non-passive, non-shared) dataset
                variable, in dataset order. NODAL arrays must have length ``num_nodes``;
                CELL_CENTERED arrays must have length ``num_cells``.  ``num_nodes`` and
                ``num_cells`` are inferred from ``node_map`` (or from the
                ``con_sharing`` source zone when ``node_map`` is omitted).
            zone_type (ZoneType):
                FE zone type from the ZoneType enum. Must be one of the types in
                ``_FE_SIMPLE``.
            node_map (npt.ArrayLike):
                Integer array of shape ``(num_cells, nodes_per_cell)`` containing
                0-based node indices. Required unless ``con_sharing`` is set, in which
                case the connectivity are inherited from the source zone instead.
            title (str):
                Zone title string. Defaults to ``"FE_Zone_{index}"``, this zone's
                0-based index.
            variables (list[str]):
                Variable name list. Required only when the file has not been opened yet
                (lazy-open path). Ignored on subsequent zones once the file is already
                initialised. Default to ``[V1, V2, V3, ...]`` if not provided in open or
                zone call.
            value_locations (Sequence[ValueLocation]):
                Per-active-variable :class:`~tecio.ValueLocation`. Defaults to all
                ``NODAL``.
            passive_vars (list | tuple | set | None):
                Variables to mark passive: a list, tuple, or set of variable names, or
                of 0-based indices, e.g. ``{"x"}`` or ``[3]``. Defaults to all active
                (``passive_vars=None``).
            var_sharing (Mapping[int | str, int]):
                Variables to share from another zone: a ``{variable: source zone}``
                mapping, name or 0-based index to 0-based zone index. Defaults to no
                sharing (None or an empty mapping). Cross-checked against ``node_map`` /
                ``con_sharing`` for a consistent node/cell count.
            con_sharing (int):
                Optional zone index that the connectivity is shared from. ``None``
                indicates no sharing (this zone owns its connectivity). The first
                zone in a dataset must own its connectivity. Connectivity cannot be
                shared when face neighbor mode is set to global. Connectivity cannot be
                shared between cell-based and face-based finite element zones.
            face_neighbors (npt.ArrayLike):
                Optional face-neighbor connectivity, always a flat array (whatever shape
                you pass is flattened before writing). The number of values per
                connection depends on ``face_neighbor_mode``, a fixed 3 or 4 for the
                one-to-one modes, a variable, self-describing count per record for the
                "many" modes. The connection count is always derived from this array,
                never a separate parameter.
            face_neighbor_mode (FaceNeighborMode):
                Default to None meaning no face-neighbor data. If ``face_neighbors`` is
                supplied without defining a face neighbor mode, it is treated as
                LOCAL_ONE_TO_ONE. Cannot be defined without also supplying a
                ``face_neighbors`` array.
            solution_time (float):
                Solution time for transient data. Default to ``0.0`` if not defined.
            strand_id (int):
                Strand ID for transient data. Use ``0`` for static zones. Default to
                ``0`` (static) if not defined.
            aux (dict[str, str | float | int]):
                Zone-level auxiliary data as ``{name: value}`` string pairs.
            datapacking (DataPacking | str):
                In :attr:`~tecio.DataPacking.POINT` format, the values for all variables
                are given for the first point, then the second point, and so on. In
                :attr:`~tecio.DataPacking.BLOCK` format, all of the values for the first
                variable are given in a block, then all of the values for the second
                variable, then all of the values for the third, and so
                forth. :attr:`~tecio.DataPacking.BLOCK` format must be used for
                cell-centered data and polyhedral zones
                (:attr:`~tecio.ZoneType.FEPOLYGON`/:attr:`~tecio.ZoneType.FEPOLYHEDRAL`),
                as well as for all binary data. `POINT` and `BLOCK` keyword strings also
                accepted. Defaults to :attr:`~tecio.DataPacking.BLOCK`.
            flush:
                If ``True``, flush this zone and all previous data to a temporary
                intermediate file immediately after writing, releasing data from
                memory. Defaults to ``False``. Useful when memory is a concern, but adds
                the overhead of a disk write. Temporary files are merged back into the
                final output file when :meth:`close` is called.

        Raises:
            NotImplementedError:
                If *datapacking* is :attr:`~tecio.libtecio.DataPacking.POINT`.
            TypeError:
                If *passive_vars*/*var_sharing* isn't a list/tuple/set/mapping as
                appropriate, or mixes variable names and indices.
            ValueError:
                For FEPOLYGON or FEPOLYHEDRON (not supported in SZPLT files, see
                :meth:`write_poly_zone`); on variable count or array length mismatch;
                if ``node_map`` is omitted
                without ``con_sharing``; if ``var_sharing``/``con_sharing`` reference a
                zone with no recorded node/cell count, or one whose count disagrees with
                this zone's; or a variable named in both *passive_vars* and
                *var_sharing*.
            ValueError:
                If I/O operation attempted on closed or None file handle.
            RuntimeError:
                If attempting to write variable aux before variables are defined.
            KeyError:
                A *passive_vars*/*var_sharing* name doesn't match any dataset variable.
            IndexError:
                A *passive_vars*/*var_sharing* index is out of range.

        Note:
            FE variable arrays are 1-D and node-ordered. ``write_data`` handles dtype
            inference and F-order ravel internally, but 1-D arrays are unaffected by
            memory order.
        """
        # Validate input
        if isinstance(datapacking, str):
            try:
                datapacking = DataPacking[datapacking.upper()]
            except KeyError:
                raise ValueError(
                    f"datapacking={datapacking!r} is not a recognised value; "
                    "use DataPacking.BLOCK or the string 'BLOCK'."
                ) from None
        if datapacking != DataPacking.BLOCK:
            raise NotImplementedError(
                "DATAPACKING=POINT is an ASCII-only layout and is not supported "
                "by the SZL binary format. Use DataPacking.BLOCK (the default) "
                "or write to a .dat file instead."
            )
        if zone_type in (ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON):
            raise ValueError(
                f"{zone_type.name} zones are face-based, use write_poly_zone "
                "(which SZPLT files do not support)."
            )
        if zone_type not in _FE_SIMPLE:
            raise NotImplementedError(
                f"Zone type {zone_type.name!r} is not supported by szl file formats."
            )

        # Convert input data to NumPy arrays
        arrays: list[npt.NDArray] = [np.asarray(arr) for arr in data]

        # Set default title if none provided
        if title is None:
            title = f"FE_Zone_{self.current_zone + 1}"

        # Default variable names (only used on lazy-open first-zone call)
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(data) + 1)]

        # Open the file if lazily loaded and flush buffered aux data
        if self.handle is None:
            self._open(variables)
        if self.current_zone == -1:
            self.flush_aux()

        # Infer per-variable data types from array dtypes
        variable_types = [
            _resolve_written_type(_infer_data_type(arr.dtype), self.precision)
            for arr in arrays
        ]

        prepared: PreparedFEZone | None = prepare_fe_zone(
            arrays,
            variable_types,
            zone_type,
            node_map=node_map,
            value_locations=value_locations,
            passive_vars=passive_vars,
            var_sharing=var_sharing,
            con_sharing=con_sharing,
            face_neighbors=face_neighbors,
            face_neighbor_mode=face_neighbor_mode,
            dataset_variables=self._check_variables(),
            meta=self._meta,
        )
        if prepared is None:
            return
        passive_flags = prepared.passive_vars
        share_idx = prepared.var_sharing
        con_sharing = prepared.con_sharing
        active_var_idx = prepared.active_var_idx
        num_nodes = prepared.num_nodes
        num_cells = prepared.num_cells
        value_locations_global = prepared.value_locations_global
        variable_types_global = prepared.variable_types_global
        face_neighbors_arr = prepared.face_neighbors_arr
        face_neighbor_mode = prepared.face_neighbor_mode
        num_face_cons = prepared.num_face_connections

        if face_neighbors_arr is not None:
            raise NotImplementedError(
                "Writing face-neighbor connections to SZL (.szplt) output is not "
                "currently supported. Face-neighbor writing works correctly for PLT "
                "and DAT; use one of those formats if you need it."
            )

        # Write zone header. The C library assigns and returns a 1-based zone
        # number; converted to 0-based for the public current_zone attribute.
        self.current_zone = (
            libtecio.tec_zone_create_fe(
                self._check_handle(),
                title,
                zone_type,
                num_nodes,
                num_cells,
                var_types=variable_types_global,
                value_locations=value_locations_global,
                pas_vars=passive_flags,
                var_sharing=var_sharing_for_libtecio(share_idx),
                con_sharing=zone_for_libtecio(con_sharing),
                num_face_cons=num_face_cons,
                face_nbr_mode=face_neighbor_mode or FaceNeighborMode.LOCAL_ONE_TO_ONE,
            )
            - 1
        )
        zone_num = self.current_zone + 1

        # Unsteady options
        if strand_id != 0 or solution_time != 0.0:
            libtecio.tec_zone_set_unsteady_options(
                handle=self._check_handle(),
                zone=zone_num,
                strand=strand_id,
                solution_time=solution_time,
            )

        # Write zone-level aux data
        if aux is not None:
            _write_zone_aux_data(self._check_handle(), {zone_num: aux})

        # Write active data only
        for var_idx, arr, dtype in zip(
            active_var_idx, arrays, variable_types, strict=True
        ):
            self.current_var = var_idx
            _write_data(
                self._check_handle(),
                zone_num=zone_num,
                var_num=var_idx + 1,
                data=arr,
                dt=dtype,
            )

        # Write connectivity (if not shared). Node numbers are 0-based in
        # Python; the C library expects 1-based node numbers.
        if con_sharing is None:
            assert node_map is not None
            _write_connectivity(
                self._check_handle(),
                zone_num,
                np.asarray(node_map) + 1,
                face_neighbors_arr,
            )

        # Flush zone data, releasing memory
        if flush:
            libtecio.tec_file_writer_flush(self._check_handle())

        # Finally set zone metadata after successfully completing TecIO calls
        self._meta.record_zone(
            ZoneMeta(
                index=self.current_zone,
                title=title,
                zone_type=zone_type,
                solution_time=solution_time,
                strand_id=strand_id,
                num_aux_items=len(aux) if aux else 0,
                num_nodes=num_nodes,
                num_elements=num_cells,
                value_locations=tuple(value_locations_global),
                passive_vars=tuple(bool(p) for p in passive_flags),
                shared_vars=tuple(share_idx),
                data_types=tuple(variable_types_global),
                face_neighbor_mode=(
                    face_neighbor_mode if face_neighbors_arr is not None else None
                ),
                num_face_connections=num_face_cons or None,
            )
        )

    def write_poly_zone(
        self,
        data: Sequence[npt.ArrayLike],
        zone_type: ZoneType,
        *,
        face_nodes: npt.ArrayLike | None = None,
        face_node_counts: npt.ArrayLike | None = None,
        face_left_elements: npt.ArrayLike | None = None,
        face_right_elements: npt.ArrayLike | None = None,
        boundary_connection_counts: npt.ArrayLike | None = None,
        boundary_connection_elements: npt.ArrayLike | None = None,
        boundary_connection_zones: npt.ArrayLike | None = None,
        title: str | None = None,
        variables: list[str] | None = None,
        value_locations: Sequence[ValueLocation] | None = None,
        passive_vars: PassiveVarsSpec | None = None,
        var_sharing: VarSharingSpec | None = None,
        con_sharing: int | None = None,
        solution_time: float = 0.0,
        strand_id: int = 0,
        aux: dict[str, Any] | None = None,
        datapacking: DataPacking | str = DataPacking.BLOCK,
    ) -> None:
        """Not supported: SZPLT files cannot hold polygon or polyhedron zones.

        Raises:
            NotImplementedError: Always. Write to a ``.plt`` or ``.dat`` file instead,
                see :meth:`TecplotPltWriter.write_poly_zone`.
        """
        raise NotImplementedError(
            "Polygon and polyhedron zones are not yet supported in SZPLT files by the "
            "TecIO library. Write a .plt or .dat file instead."
        )
