"""Write Tecplot PLT (``.plt``) binary files using the classic TecIO API.

Note:
    PLT's classic API sets output precision once for the *entire file* via ``VIsDouble``
    at ``tecini142`` time.  ``tecdat142``'s own ``IsDouble`` argument does not control
    per-variable output precision, and there is no per-variable type array in
    ``teczne142`` either. Every variable in a PLT file is written at the single,
    file-wide precision set by :attr:`TecplotPltWriter.precision`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from . import libtecio
from ._constants import (
    DataPacking,
    DataType,
    FaceNeighborMode,
    FileFormat,
    FileType,
    ValueLocation,
    ZoneType,
)
from ._meta import ZoneMeta
from ._writer import (
    PassiveVarsSpec,
    PreparedFEZone,
    PreparedOrderedZone,
    PreparedPolyZone,
    TecplotWriter,
    VarSharingSpec,
    convert_face_connections_to_one_based,
    normalize_precision,
    prepare_fe_zone,
    prepare_ordered_zone,
    prepare_poly_zone,
    var_sharing_for_libtecio,
    zone_for_libtecio,
)

# -------------------------------------------------------------------------------------
# Module-level constants
# -------------------------------------------------------------------------------------

# FE zone types supported by :meth:`Write.write_fe_zone`.
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})
# =====================================================================================
# Helpers
# =====================================================================================


def _infer_data_type(dt: DataType | np.dtype) -> DataType:
    """Return a C-supported :class:`DataType` for a :class:`DataType` or NumPy dtype.

    Maps NumPy dtypes to the closest supported Tecplot data type. For PLT format only
    FLOAT and DOUBLE are supported; all integer types are upcast to DOUBLE.  precision
    is promoted to ``FLOAT``; 64-bit integers are promoted to ``INT32``; ``int8`` and
    ``uint8`` map to ``BYTE``.

    Raises:
        ValueError: If the dtype is not supported.
    """
    closest: dict[np.dtype, DataType] = {
        np.dtype(np.float64): DataType.DOUBLE,
        np.dtype(np.float32): DataType.FLOAT,
        np.dtype(np.float16): DataType.FLOAT,
        np.dtype(np.int64): DataType.INT32,
        np.dtype(np.int32): DataType.INT32,
        np.dtype(np.int16): DataType.INT16,
        np.dtype(np.int8): DataType.BYTE,
        np.dtype(np.uint8): DataType.BYTE,
    }

    if isinstance(dt, DataType):
        return dt

    dt_np = np.dtype(dt)

    for key, val in closest.items():
        if dt_np == key:
            return val

    if np.issubdtype(dt_np, np.floating):
        return DataType.FLOAT if dt_np.itemsize <= 4 else DataType.DOUBLE

    if np.issubdtype(dt_np, np.signedinteger):
        return DataType.INT16 if dt_np.itemsize <= 2 else DataType.INT32

    if np.issubdtype(dt_np, np.unsignedinteger):
        return DataType.BYTE

    raise ValueError(f"Unsupported dtype: {dt_np}")


# =====================================================================================
# Local functions
# =====================================================================================


def _write_data(data: npt.ArrayLike, dtype: np.dtype | DataType | None = None) -> None:
    """Write a single variable's data using ``tecdat142``.

    ``tecdat142`` only accepts ``float32`` or ``float64``; integer types are
    upcast to ``float64``. The array is ravelled in Fortran (column-major)
    order before writing, which matches Tecplot's expected BLOCK layout.

    Args:
        data: Array-like of values to write.
        dtype: Intended :class:`DataType` for the variable.

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

    # Convert input array-like data to NumPy arrays
    arr = np.asarray(data)

    data_type = (
        _infer_data_type(dtype) if dtype is not None else dtype_to_datatype[arr.dtype]
    )

    # tecdat142 already returns a contiguous array so no need to cast before calling
    if data_type in (DataType.FLOAT, DataType.INT16, DataType.BYTE):
        # All fit within float32's precision
        libtecio.tecdat142(arr.ravel(order="F"), is_double=False)
    else:
        # INT32 max value (~2.1e9) exceeds float32's integer ceiling (2**24 ~= 16.7e6),
        # so write as a double
        libtecio.tecdat142(arr.ravel(order="F"), is_double=True)


def _write_connectivity(
    node_map: npt.ArrayLike,
    face_neighbors: npt.ArrayLike | None = None,
) -> None:
    """Write FE connectivity: node map and optional face-neighbour data.

    Args:
        node_map:       Integer array of shape ``(num_cells, nodes_per_cell)``
                        with 1-based node indices.
        face_neighbors: Optional face-neighbour connection array.
    """
    nodes_flat = np.ascontiguousarray(node_map, dtype=np.int32).ravel(order="C")
    libtecio.tecnode142(nodes_flat)

    if face_neighbors is not None:
        face_flat = np.ascontiguousarray(face_neighbors, dtype=np.int32).ravel(
            order="C"
        )
        libtecio.tecface142(face_flat)


# =====================================================================================
# Write class
# =====================================================================================


# ======================================================================================
# TecplotPltWriter
# ======================================================================================


class TecplotPltWriter(TecplotWriter):
    """Write Tecplot PLT (``.plt``) files using the classic TecIO API.

    The classic TecIO API maintains a single implicit global file context;
    only one PLT file may be open at a time. This class wraps that
    procedural API behind the same interface as :class:`~tecio.TecplotSzlWriter`.

    Dataset- and variable-level auxiliary data can be set at any time before
    the first zone is written. They are buffered and flushed automatically
    when the first zone header is created. Zone-level auxiliary data is
    passed directly to each zone-writing method.

    Like :class:`~tecio.TecplotSzlWriter`, the file can be opened *eagerly*
    (when ``variables`` is supplied to ``__init__``) or *lazily* (deferred
    until the first :meth:`write_ordered_zone` or :meth:`write_fe_zone` call, at
    which point ``variables`` must be provided to that call).

    Args:
        path:         Output file path (should end in ``.plt``).
        title:        Dataset title string.
        variables:    Variable name list. If provided the file is opened
                      immediately. If ``None``, the file is opened on the first
                      zone write.
        file_type:    :class:`~libtecio.FileType` enum. Defaults to
                      :attr:`~libtecio.FileType.FULL`.
        precision:    Whole-file storage precision (:attr:`DataType.FLOAT`/``"single"``
                      or :attr:`DataType.DOUBLE`/``"double"``). Unlike SZL, PLT has no
                      per-variable type, so this always resolves to a concrete value;
                      ``None`` is not accepted. Defaults to :attr:`DataType.DOUBLE`.

    Attributes:
        precision:    Whole-file storage precision (:attr:`DataType.FLOAT` or
                      :attr:`DataType.DOUBLE`). Maps directly to ``VIsDouble`` in
                      ``tecini142``.
        _opened:      True once ``tecini142`` has been called (the file has been
                      opened, eagerly or lazily), False before that and after
                      :meth:`close`. Governs whether :meth:`close` calls
                      ``tecend142``, and whether the next zone write needs to open
                      the file first.

    Caution:
        Unlike the SZL writer, zone data must be written strictly in order: header →
        variable data → connectivity. This is enforced internally by the write methods.

    Examples:
        Define file header fields on open.

        >>> with tecio.TecplotPltWriter("out.plt", variables=["X", "Y", "P"]) as w:
        ...     w.write_ordered_zone(data=[x, y, p], title="Zone 1")

        If writer handle is opened with just the file name, the variable name list can
        be provided with the first zone written.

        >>> with tecio.TecplotPltWriter("out.plt") as w:
        ...     w.write_ordered_zone(
        ...         data=[x, y, p],
        ...         variables=["X", "Y", "P"],
        ...         title="Zone 1",
        ...     )
    """

    def __init__(
        self,
        path: str,
        title: str = "untitled",
        variables: list[str] | None = None,
        file_type: FileType = FileType.FULL,
        *,
        precision: DataType | str = DataType.DOUBLE,
    ) -> None:
        """Store configuration; open the file immediately if *variables* given.

        Raises:
            ValueError: If *precision* is not :attr:`DataType.FLOAT` /
                        :attr:`DataType.DOUBLE` (or a recognized string alias for one of
                        them).
        """
        # Set before super().__init__(), which calls self._open() when
        # variables is already known (eager open), and _open() needs this.
        self.precision: DataType = cast(
            DataType, normalize_precision(precision, allow_none=False)
        )
        # Track whether the file has been opened so we know whether to call
        # tecini142 inside open().
        self._opened: bool = False
        super().__init__(path, title, variables, file_type)

    @property
    def _file_format(self) -> str:
        return "plt"

    # -- File lifecycle ---------------------------------------------------------------

    def _open(self, var_names: list[str]) -> None:
        """Call ``tecini142`` and record the variable list.

        This is called at most once per instance — either from ``__init__``
        (eager open) or from the first zone-writing method (lazy open).

        Args:
            var_names: Ordered list of variable name strings.
        """
        self.variables = var_names
        libtecio.tecini142(
            filename=self.path,
            variables=self.variables,
            title=self.title,
            file_format=FileFormat.PLT,
            file_type=self.file_type,
            vis_double=self.precision,
        )
        self._opened = True
        self._meta.set_variables(self.variables)

    def close(self) -> None:
        """Finalize and close the PLT file (safe to call more than once).

        Flushes any buffered aux data first (in case it was added after the
        first zone, and so never reached the automatic pre-first-zone
        flush, e.g. ``add_auxdataset_dict`` called after a zone is already
        written), then calls ``tecend142`` if the file was opened.
        """
        if self._opened:
            self.flush_aux()
            libtecio.tecend142()
            self._opened = False

    # -- Aux data: only the per-item write differs from the shared base ----------------

    def _write_dataset_aux_item(self, name: str, value: str) -> None:
        libtecio.tecauxstr142(name, value)

    def _write_var_aux_item(
        self, one_based_var_index: int, name: str, value: str
    ) -> None:
        libtecio.tecvauxstr142(one_based_var_index, name, value)

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
    ) -> None:
        """Write a complete IJK-ordered zone.

        The zone header (``teczne142``), variable data (``tecdat142``), and optional
        zone aux data (``teczauxstr142``) are all written in strict sequence, as
        required by the classic TecIO API.

        The zone dimensions ``imax × jmax × kmax`` are inferred from the shape of the
        first nodal data array.  1-D arrays produce a ``(N, 1, 1)`` zone; 2-D arrays
        produce an ``(I, J, 1)`` zone.

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
                "by the PLT binary format. Use DataPacking.BLOCK (the default) "
                "or write to a .dat file instead."
            )

        # Convert input data to NumPy arrays
        arrays: list[npt.NDArray] = [np.asarray(arr) for arr in data]

        # Default title
        if title is None:
            title = f"IJK_Zone_{self.current_zone + 1}"

        # Default variable names (only used on lazy-open first-zone call)
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(arrays) + 1)]

        # Open the file if lazily loaded and flush buffered aux data
        if not self._opened:
            self._open(variables)
        if self.current_zone == -1:
            self.flush_aux()

        # Per-active-variable data types inferred from array dtypes (currently overrided
        # to precision)
        variable_types = [self.precision] * len(arrays)

        # Default value locations — all nodal
        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * len(arrays)

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

        # Write zone header
        libtecio.teczne142(
            zone_title=title,
            zone_type=ZoneType.ORDERED,
            imax=imax,
            jmax=jmax,
            kmax=kmax,
            value_locations=value_locations_global,
            pas_vars=passive_flags,
            var_sharing=(
                var_sharing_for_libtecio(share_idx)
                if any(s is not None for s in share_idx)
                else None
            ),
            con_sharing=0,
            strand=strand_id,
            solution_time=solution_time,
        )
        self.current_zone += 1

        # Zone-level aux data must be written immediately after teczne142 and
        # before the first tecdat142 call.
        if aux is not None:
            for name, value in aux.items():
                libtecio.teczauxstr142(str(name), str(value))

        # Write active variable data in dataset-variable order
        for var_idx, arr, dtype in zip(
            active_var_idx, arrays, variable_types, strict=True
        ):
            self.current_var = var_idx
            _write_data(arr, dtype)

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
    ) -> None:
        """Write a complete finite-element zone.

        Node and cell counts are inferred from *node_map*, or if *node_map* is omitted,
        from the zone referenced by *con_sharing*.

        The full sequence required by the classic TecIO API is issued internally:

        1. :func:`~tecio.libtecio.teczne142` — zone header.
        2. :func:`~tecio.libtecio.teczauxstr142` — zone-level aux data (if any), before
           data.
        3. :func:`~tecio.libtecio.tecdat142` — one call per active variable.
        4. :func:`~tecio.libtecio.tecnode142` — node map (skipped when connectivity is
           shared via ``con_sharing``).
        5. :func:`~tecio.libtecio.tecface142` — face-neighbour connections (if
           provided).

        ``FEPOLYGON`` and ``FEPOLYHEDRON`` zones are face-based, write them with
        :meth:`write_poly_zone`.

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

        Raises:
            NotImplementedError:
                If *datapacking* is :attr:`~tecio.libtecio.DataPacking.POINT`.
            TypeError:
                If *passive_vars*/*var_sharing* isn't a list/tuple/set/mapping as
                appropriate, or mixes variable names and indices.
            ValueError:
                For FEPOLYGON or FEPOLYHEDRON (use :meth:`write_poly_zone`); on
                variable count or array length mismatch; if ``node_map`` is omitted
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
                "by the PLT binary format. Use DataPacking.BLOCK (the default) "
                "or write to a .dat file instead."
            )
        if zone_type in (ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON):
            raise ValueError(
                f"{zone_type.name} zones are face-based, use write_poly_zone."
            )
        if zone_type not in _FE_SIMPLE:
            raise NotImplementedError(
                f"Zone type {zone_type.name!r} is not supported by write_fe_zone."
            )

        # Convert input data to NumPy arrays
        arrays: list[npt.NDArray] = [np.asarray(arr) for arr in data]

        # Default title
        if title is None:
            title = f"FE_Zone_{self.current_zone + 1}"

        # Default variable names (only used on lazy-open first-zone call)
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(arrays) + 1)]

        # Open the file if lazily loaded and flush buffered aux data
        if not self._opened:
            self._open(variables)
        if self.current_zone == -1:
            self.flush_aux()

        # Infer per-variable data types from array dtypes (currently overrided to
        # precision)
        variable_types = [self.precision] * len(arrays)

        # Set default value locations
        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * len(arrays)

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
        face_neighbor_mode = prepared.face_neighbor_mode
        num_face_cons = prepared.num_face_connections
        # PLT's classic API only supports 32-bit face-neighbor writes, so
        # the dtype cast (not forced by the shared preparation step, which
        # leaves it to each format) happens here, right before dispatch.
        face_neighbors_arr: npt.NDArray | None = None
        if prepared.face_neighbors_arr is not None:
            face_neighbors_arr = prepared.face_neighbors_arr.astype(np.int32)

        # Write zone header.
        # imax = num_nodes, jmax = num_cells, kmax = 0 for simple FE zones.
        libtecio.teczne142(
            zone_title=title,
            zone_type=zone_type,
            imax=num_nodes,
            jmax=num_cells,
            kmax=0,
            value_locations=value_locations_global,
            pas_vars=passive_flags,
            var_sharing=(
                var_sharing_for_libtecio(share_idx)
                if any(s is not None for s in share_idx)
                else None
            ),
            con_sharing=zone_for_libtecio(con_sharing),
            strand=strand_id,
            solution_time=solution_time,
            num_face_connections=num_face_cons,
            face_nbr_mode=face_neighbor_mode or FaceNeighborMode.LOCAL_ONE_TO_ONE,
        )
        self.current_zone += 1

        # Zone-level aux data must be immediately after teczne142, before
        # any tecdat142 calls.
        if aux is not None:
            for name, value in aux.items():
                libtecio.teczauxstr142(str(name), str(value))

        # Write active variable data in dataset-variable order
        for var_idx, arr, dtype in zip(
            active_var_idx, arrays, variable_types, strict=True
        ):
            self.current_var = var_idx
            _write_data(arr, dtype)

        # Write connectivity (if not shared). Must come after all data arrays.
        if con_sharing is None:
            assert node_map is not None
            face_flat = None
            if face_neighbors_arr is not None:
                assert face_neighbor_mode is not None
                face_flat = convert_face_connections_to_one_based(
                    face_neighbors_arr, face_neighbor_mode
                )
            _write_connectivity(np.asarray(node_map) + 1, face_flat)

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
        """Write a complete FEPOLYGON or FEPOLYHEDRON (face-based) zone.

        Node and element counts are inferred from the face map, or from the
        *con_sharing* source zone when it is omitted. Every index is 0-based. The
        arguments match what :class:`~tecio.TecplotPolyZoneReader` returns, so a zone
        read from a file can be written back unchanged.

        The classic TecIO sequence is issued internally: ``tecpolyzne142`` (header),
        ``teczauxstr142`` (aux data), ``tecdat142`` (one call per active variable),
        then ``tecpolyface142`` and, with boundary connections, ``tecpolybconn142``.
        Connectivity is skipped when shared via *con_sharing*.

        Args:
            data (Sequence[npt.ArrayLike]):
                Sequence of 1-D arrays, one per active (non-passive, non-shared) dataset
                variable, in dataset order. NODAL arrays need ``num_nodes`` values,
                CELL_CENTERED arrays ``num_elements``, both inferred from the face map.
            zone_type (ZoneType):
                :attr:`~tecio.ZoneType.FEPOLYGON` or
                :attr:`~tecio.ZoneType.FEPOLYHEDRON`.
            face_nodes (npt.ArrayLike):
                Flat 0-based node indices of all faces, in face order. Two per face
                for a polygon zone. Required unless ``con_sharing`` is set.
            face_node_counts (npt.ArrayLike):
                Number of nodes of each face. Required for FEPOLYHEDRON. Optional for
                FEPOLYGON, where every face has 2 nodes.
            face_left_elements (npt.ArrayLike):
                Element on the left of each face: a 0-based element index, ``-1`` for
                no element, or ``-(b + 2)`` for boundary face ``b`` (0-based).
            face_right_elements (npt.ArrayLike):
                Element on the right of each face, same convention as
                ``face_left_elements``.
            boundary_connection_counts (npt.ArrayLike):
                Number of connections of each boundary face. Boundary faces describe
                neighbors in other zones, or faces with more than one neighbor.
            boundary_connection_elements (npt.ArrayLike):
                Flat 0-based neighbor element of every boundary connection.
            boundary_connection_zones (npt.ArrayLike):
                Flat 0-based zone of every neighbor element. May only reference a zone
                already written or this zone.
            title (str):
                Zone title string. Defaults to ``"Poly_Zone_{index}"``, with this
                zone's 0-based index.
            variables (list[str]):
                Variable name list, only used when the file has not been opened yet
                (lazy-open path). Defaults to ``[V1, V2, V3, ...]``.
            value_locations (Sequence[ValueLocation]):
                Per-active-variable :class:`~tecio.ValueLocation`. Defaults to all
                ``NODAL``.
            passive_vars (list | tuple | set | None):
                Variables to mark passive: a list, tuple, or set of variable names, or
                of 0-based indices. Defaults to all active.
            var_sharing (Mapping[int | str, int]):
                Variables to share from another zone: a ``{variable: source zone}``
                mapping, name or 0-based index to 0-based zone index.
            con_sharing (int):
                Zone index to share the face map from, which must be a zone of the
                same type without boundary faces (TecIO does not allow sharing those).
                ``None`` means this zone owns its connectivity.
            solution_time (float):
                Solution time for transient data. Defaults to ``0.0``.
            strand_id (int):
                Strand ID for transient data, ``0`` for static zones.
            aux (dict[str, str | float | int]):
                Zone-level auxiliary data as ``{name: value}`` pairs.
            datapacking (DataPacking | str):
                Must be :attr:`~tecio.DataPacking.BLOCK`, the only layout of the PLT
                binary format.

        Raises:
            TypeError:
                If *passive_vars*/*var_sharing* isn't a list/tuple/set/mapping as
                appropriate, or mixes variable names and indices.
            ValueError:
                If *datapacking* is :attr:`~tecio.DataPacking.POINT`. For a
                *zone_type* other than FEPOLYGON or FEPOLYHEDRON (use
                :meth:`write_fe_zone`); on a missing, inconsistent, or out of range
                face map, boundary connection, or variable array.
            KeyError:
                A *passive_vars*/*var_sharing* name doesn't match any dataset variable.
            IndexError:
                A *passive_vars*/*var_sharing* index is out of range.
        """
        if isinstance(datapacking, str):
            try:
                datapacking = DataPacking[datapacking.upper()]
            except KeyError:
                raise ValueError(
                    f"datapacking={datapacking!r} is not a recognised value; "
                    "use DataPacking.BLOCK or the string 'BLOCK'."
                ) from None
        if datapacking != DataPacking.BLOCK:
            raise ValueError(
                "Polygon and polyhedron zones must use BLOCK data packing."
            )

        arrays: list[npt.NDArray] = [np.asarray(arr) for arr in data]

        if title is None:
            title = f"Poly_Zone_{self.current_zone + 1}"
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(arrays) + 1)]

        # Open the file if lazily loaded and flush buffered aux data
        if not self._opened:
            self._open(variables)
        if self.current_zone == -1:
            self.flush_aux()

        variable_types = [self.precision] * len(arrays)

        prepared: PreparedPolyZone | None = prepare_poly_zone(
            arrays,
            variable_types,
            zone_type,
            face_nodes=face_nodes,
            face_node_counts=face_node_counts,
            face_left_elements=face_left_elements,
            face_right_elements=face_right_elements,
            boundary_connection_counts=boundary_connection_counts,
            boundary_connection_elements=boundary_connection_elements,
            boundary_connection_zones=boundary_connection_zones,
            value_locations=value_locations,
            passive_vars=passive_vars,
            var_sharing=var_sharing,
            con_sharing=con_sharing,
            dataset_variables=self._check_variables(),
            meta=self._meta,
        )
        if prepared is None:
            return
        conn = prepared.connectivity
        share_idx = prepared.var_sharing

        libtecio.tecpolyzne142(
            zone_title=title,
            zone_type=zone_type,
            num_nodes=conn.num_nodes,
            num_faces=conn.num_faces,
            num_elements=conn.num_cells,
            var_sharing=(
                var_sharing_for_libtecio(share_idx)
                if any(s is not None for s in share_idx)
                else None
            ),
            value_locations=prepared.value_locations_global,
            pas_vars=prepared.passive_vars,
            con_sharing=zone_for_libtecio(prepared.con_sharing),
            strand=strand_id,
            solution_time=solution_time,
            total_num_face_nodes=conn.total_face_nodes,
            num_connected_boundary_faces=conn.num_boundary_faces,
            total_num_boundary_connections=conn.num_boundary_connections,
        )
        self.current_zone += 1

        # Zone-level aux data must be immediately after the zone header, before data.
        if aux is not None:
            for name, value in aux.items():
                libtecio.teczauxstr142(str(name), str(value))

        for var_idx, arr, dtype in zip(
            prepared.active_var_idx, arrays, variable_types, strict=True
        ):
            self.current_var = var_idx
            _write_data(arr, dtype)

        # Face map (if not shared), after all data arrays. TecIO is 1-based with 0 for
        # no element and -k for the k-th boundary face, which is the 0-based API
        # convention shifted by one for every array.
        if prepared.con_sharing is None:
            assert conn.face_nodes is not None
            assert (
                conn.face_left_elements is not None
                and conn.face_right_elements is not None
            )
            # Polygon: counts are NULL, every face is a two node edge.
            libtecio.tecpolyface142(
                conn.face_node_counts,
                conn.face_nodes.astype(np.int32) + 1,
                conn.face_left_elements.astype(np.int32) + 1,
                conn.face_right_elements.astype(np.int32) + 1,
                num_faces=conn.num_faces,
            )
            if conn.num_boundary_faces:
                assert conn.boundary_connection_counts is not None
                assert conn.boundary_connection_elements is not None
                assert conn.boundary_connection_zones is not None
                libtecio.tecpolybconn142(
                    conn.boundary_connection_counts,
                    conn.boundary_connection_elements.astype(np.int32) + 1,
                    conn.boundary_connection_zones.astype(np.int32) + 1,
                )

        # Finally set zone metadata after successfully completing TecIO calls
        self._meta.record_zone(
            ZoneMeta(
                index=self.current_zone,
                title=title,
                zone_type=zone_type,
                solution_time=solution_time,
                strand_id=strand_id,
                num_aux_items=len(aux) if aux else 0,
                num_nodes=conn.num_nodes,
                num_elements=conn.num_cells,
                num_faces=conn.num_faces,
                total_face_nodes=conn.total_face_nodes,
                num_boundary_faces=conn.num_boundary_faces,
                num_boundary_connections=conn.num_boundary_connections,
                value_locations=tuple(prepared.value_locations_global),
                passive_vars=tuple(prepared.passive_vars),
                shared_vars=tuple(share_idx),
                data_types=tuple(prepared.variable_types_global),
            )
        )
