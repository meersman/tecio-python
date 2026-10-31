r"""Write Tecplot ASCII DAT (``.dat`` / ``.tec``) files.

Supported ``DATAPACKING`` modes for writing:
    * ``BLOCK`` -- one contiguous value block per variable (default).
    * ``POINT`` -- one row of all active nodal variable values per node,
      followed by a separate row-per-cell section for cell-centred variables.
      Pass ``datapacking="POINT"`` to :meth:`~TecplotDatWriter.write_ordered_zone`
      or :meth:`~TecplotDatWriter.write_fe_zone`.
    * Poly zones (:meth:`~TecplotDatWriter.write_poly_zone`) are always ``BLOCK``.
"""

from __future__ import annotations

import contextlib

# Standard library
import io
import os
from collections.abc import Sequence
from typing import Any, cast

# Third-party
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
)

# -------------------------------------------------------------------------------------
# Module-level constants
# -------------------------------------------------------------------------------------

# FE zone types fully supported for reading and writing
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})

# Zone types whose connectivity is face-based
_FE_POLY: frozenset[ZoneType] = frozenset({
    ZoneType.FEPOLYGON,
    ZoneType.FEPOLYHEDRON,
})

# Nodes per element for each supported simple FE type
_NODES_PER_ELEM: dict[ZoneType, int] = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FETETRAHEDRON: 4,
    ZoneType.FEBRICK: 8,
}

# ASCII keyword to ZoneType
_STR_TO_ZONETYPE: dict[str, ZoneType] = {
    "ordered": ZoneType.ORDERED,
    "felineseg": ZoneType.FELINESEG,
    "fetriangle": ZoneType.FETRIANGLE,
    "fequadrilateral": ZoneType.FEQUADRILATERAL,
    "fetetrahedron": ZoneType.FETETRAHEDRON,
    "febrick": ZoneType.FEBRICK,
    "fepolygon": ZoneType.FEPOLYGON,
    "fepolyhedron": ZoneType.FEPOLYHEDRON,
}

# ASCII keyword to FileType
_STR_TO_FILETYPE: dict[str, FileType] = {
    "full": FileType.FULL,
    "grid": FileType.GRID,
    "solution": FileType.SOLUTION,
}

# ZoneType  ASCII keyword
_ZONETYPE_STR: dict[ZoneType, str] = {
    ZoneType.ORDERED: "Ordered",
    ZoneType.FELINESEG: "FELineSeg",
    ZoneType.FETRIANGLE: "FETriangle",
    ZoneType.FEQUADRILATERAL: "FEQuadrilateral",
    ZoneType.FETETRAHEDRON: "FETetrahedron",
    ZoneType.FEBRICK: "FEBrick",
    ZoneType.FEPOLYGON: "FEPolygon",
    ZoneType.FEPOLYHEDRON: "FEPolyhedron",
}

# FileType to ASCII keyword
_FILETYPE_STR: dict[FileType, str] = {
    FileType.GRID: "GRID",
    FileType.SOLUTION: "SOLUTION",
}

# FaceNeighborMode to ASCII keyword
_FACENEIGHBORMODE_STR: dict[FaceNeighborMode, str] = {
    FaceNeighborMode.LOCAL_ONE_TO_ONE: "LOCALONETOONE",
    FaceNeighborMode.LOCAL_ONE_TO_MANY: "LOCALONETOMANY",
    FaceNeighborMode.GLOBAL_ONE_TO_ONE: "GLOBALONETOONE",
    FaceNeighborMode.GLOBAL_ONE_TO_MANY: "GLOBALONETOMANY",
}

# DataType to NumPy dtype string
_DT_TO_DTYPE: dict[DataType, str] = {
    DataType.FLOAT: "f4",
    DataType.DOUBLE: "f8",
    DataType.INT32: "i4",
    DataType.INT16: "i2",
    DataType.BYTE: "u1",
}

# DataType to ASCII DT= keyword
_DATATYPE_STR: dict[DataType, str] = {
    DataType.FLOAT: "SINGLE",
    DataType.DOUBLE: "DOUBLE",
    DataType.INT32: "LONGINT",
    DataType.INT16: "SHORTINT",
    DataType.BYTE: "BYTE",
}

# ASCII DT= keyword (and common aliases) to DataType, case-insensitive lookup
_STR_TO_DATATYPE: dict[str, DataType] = {
    "single": DataType.FLOAT,
    "float": DataType.FLOAT,
    "double": DataType.DOUBLE,
    "longint": DataType.INT32,
    "shortint": DataType.INT16,
    "byte": DataType.BYTE,
}

# Significant digits that guarantee a lossless round-trip for each floating precision
# (matches the IEEE 754 theoretical bounds)
_SIG_DIGITS_FOR_PRECISION: dict[DataType, int] = {
    DataType.FLOAT: 9,
    DataType.DOUBLE: 17,
}

# Values per line for Write data blocks
_VALUES_PER_LINE: int = 5

#: Indentation applied to every zone-header line after the first
_INDENT: str = "  "

# Indentation applied to data and connectivity lines
_DATA_INDENT: str = _INDENT + "  "

#: Separator written between adjacent values on a data line
_VALUE_SEP: str = "  "


# =====================================================================================
# Helpers
# =====================================================================================


def _quote(s: str) -> str:
    """Wrap *s* in double-quotes, escaping embedded double-quotes."""
    return '"' + str(s).replace('"', '\\"') + '"'


def _unquote(s: str) -> str:
    r"""Remove surrounding double-quotes and unescape internal ``\\"``."""
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1].replace('\\"', '"')
    return s


def _strip_comment(line: str) -> str:
    """Remove a Tecplot ``#`` comment and trailing whitespace."""
    idx = line.find("#")
    return line[:idx].rstrip() if idx >= 0 else line.rstrip()


def _infer_data_type(arr: npt.NDArray) -> DataType:
    """Return the most appropriate :class:`DataType` for *arr*'s dtype."""
    dtype = arr.dtype
    if dtype.kind == "f":
        return DataType.DOUBLE if dtype.itemsize >= 8 else DataType.FLOAT
    if dtype.kind in ("i", "u"):
        if dtype.itemsize >= 4:
            return DataType.INT32
        if dtype.itemsize == 2:
            return DataType.INT16
        return DataType.BYTE
    return DataType.FLOAT


def _resolve_written_type(inferred: DataType, precision: DataType) -> DataType:
    """Return the :class:`DataType` actually written for one variable.

    *precision* overrides *inferred* only when *inferred* is itself a floating-point
    type (FLOAT or DOUBLE). Integer-inferred variables (INT32/INT16/BYTE) always keep
    their own inferred type, unaffected by *precision*. A variable holding a meaningful
    integer (a CPU number, an index, a count) should never be silently coerced by a
    setting that's conceptually about floating-point precision.
    """
    if inferred in (DataType.FLOAT, DataType.DOUBLE):
        return precision
    return inferred


def _make_float_fmt(sig_digits: int) -> str:
    """Return a ``format()``-compatible scientific-notation format string."""
    return f" .{max(sig_digits - 1, 0)}e"


def _stage_float_array(buf: io.StringIO, arr: npt.NDArray, fmt: str) -> None:
    """Write a 1-D float array to *buf* in scientific notation.

    Values are written ``_VALUES_PER_LINE`` per line, each line indented by
    ``_DATA_INDENT`` and the values separated by ``_VALUE_SEP`` so that columns align.
    """
    flat = np.asarray(arr).ravel()
    vpl = _VALUES_PER_LINE
    for start in range(0, flat.size, vpl):
        chunk = flat[start : start + vpl]
        buf.write(
            _DATA_INDENT + _VALUE_SEP.join(format(float(v), fmt) for v in chunk) + "\n"
        )


def _stage_point_rows(
    buf: io.StringIO,
    cols: list[npt.NDArray],
    fmt: str,
) -> None:
    """Write *cols* as ``DATAPACKING=POINT`` rows into *buf*.

    Each element of *cols* is the flat value array for one variable. One tab-separated
    row is written per point (node or cell): all variable values for that point appear
    on the same line. All column arrays must have the same length. Does nothing when
    *cols* is empty.

    Example:
        >>> _stage_point_rows(buf, [x_flat, y_flat, p_flat], ".8e")
    """
    if not cols:
        return
    # Stack into a (n_points, n_vars) matrix then write row by row.
    matrix = np.column_stack(cols) if len(cols) > 1 else cols[0].reshape(-1, 1)
    for row in matrix:
        buf.write(
            _DATA_INDENT + _VALUE_SEP.join(format(float(v), fmt) for v in row) + "\n"
        )


def _stage_connectivity_row(buf: io.StringIO, row: npt.NDArray) -> None:
    """Write one element's node indices to *buf* as space-separated ints."""
    buf.write(_DATA_INDENT + " ".join(str(int(n)) for n in row) + "\n")


def _stage_int_block(
    buf: io.StringIO, label: str, arr: npt.NDArray, per_line: int = 10
) -> None:
    """Write a flat integer array to *buf* under a ``# label`` comment line.

    Tecplot writes the face-map sections of poly zones this way, ``per_line`` values
    per line, and the reader skips the comment.
    """
    buf.write(f"# {label}\n")
    flat = arr.ravel()
    for start in range(0, flat.size, per_line):
        chunk = flat[start : start + per_line].tolist()
        buf.write(_DATA_INDENT + " ".join(map(str, chunk)) + "\n")


# ======================================================================================
# TecplotDatWriter
# ======================================================================================


class TecplotDatWriter(TecplotWriter):
    r"""Write Tecplot ASCII (``.dat``/``.tec``) files.

    The public interface matches :class:`~tecio.TecplotSzlWriter` and
    :class:`~tecio.TecplotPltWriter`.

    Args:
        path:         Destination file path.
        title:        Dataset title. Defaults to ``"untitled"``.
        variables:    Variable name list. ``None`` defers file creation until the first
                      zone-writing call (lazy open).
        file_type:    :class:`~libtecio.FileType` enum. Defaults to
                      :attr:`~libtecio.FileType.FULL`.
        precision:    Uniform floating point precision for the whole file.
                      :attr:`DataType.FLOAT` (default; ``"single"``) or
                      :attr:`DataType.DOUBLE` (``"double"``). Also accepts those
                      strings directly (``"single"``/``"double"``, or ``"float"`` for
                      the enum's own name). Applies only to floating-point variables: a
                      ``float64`` array is downcast to ``SINGLE`` under the default, but
                      integer variables (LONGINT/SHORTINT/BYTE) always keep their own
                      inferred type in the zone's ``DT=`` declaration regardless of
                      *precision*. Also sets the significant digit count used for every
                      variable uniformly (9 for FLOAT, 17 for DOUBLE).

    Attributes:
        precision:    Uniform floating-point precision for the file. Governs the
                      ASCII significant-digit count and which of FLOAT/DOUBLE is
                      declared for float-inferred variables; integer-inferred
                      variables keep their own type regardless (see *precision*
                      above).
        _fp:          The open text file handle, or ``None`` before the file has
                      been opened (or after :meth:`close`).
        _opened:      True once the file has been opened (eagerly or lazily) and
                      its header written, False before that and after
                      :meth:`close`.
        _float_fmt:   ``format()``-compatible scientific-notation format string
                      derived from *precision* (9 significant digits for FLOAT, 17
                      for DOUBLE), used to print every floating-point value in the
                      file.
    """

    def __init__(
        self,
        path: str,
        title: str = "untitled",
        variables: list[str] | None = None,
        file_type: FileType = FileType.FULL,
        *,
        precision: DataType | str = DataType.FLOAT,
    ) -> None:
        """Store configuration; open the file immediately if *variables* given.

        Raises:
            ValueError: If *precision* is not :attr:`DataType.FLOAT` /
                        :attr:`DataType.DOUBLE` (or a recognized string alias for one of
                        them).
        """
        # Set before super().__init__(), which calls self._open() when
        # variables is already known (eager open), and _open() needs these.
        self.precision: DataType = cast(
            DataType, normalize_precision(precision, allow_none=False)
        )
        self._fp: io.TextIOWrapper | None = None
        self._opened: bool = False
        self._float_fmt: str = _make_float_fmt(
            _SIG_DIGITS_FOR_PRECISION[self.precision]
        )
        super().__init__(path, title, variables, file_type)

    @property
    def _file_format(self) -> str:
        return "dat"

    # -- Validation checks and errror handling -----------------------------------------

    def _check_handle(self) -> io.TextIOWrapper:
        """Return the file handle, raising if the file has been closed."""
        if self._fp is None:
            raise ValueError(f"I/O operation on closed file: '{self.path}'")
        return self._fp

    # -- File lifecycle ---------------------------------------------------------------

    def _open(self, var_names: list[str]) -> None:
        """Open the output file and write the dataset header.

        Raises:
            ValueError: If ``var_names`` is empty.
        """
        if not var_names:
            raise ValueError("Write requires at least one variable name.")
        self.variables = var_names
        self._fp = open(self.path, "w", encoding="utf-8", newline="\n")  # noqa: SIM115
        self._write_file_header()
        self._opened = True
        self._meta.set_variables(self.variables)

    def close(self) -> None:
        """Flush and close the output file (safe to call more than once).

        Flushes any buffered aux data first (in case it was added after the
        first zone, and so never reached the automatic pre-first-zone
        flush, e.g. ``add_auxdataset_dict`` called after a zone is already
        written); DAT's ``DATASETAUXDATA``/``VARAUXDATA`` keywords are valid
        anywhere in the file, including after the last zone, so this is
        always a legal position to write them.
        """
        if self._fp is not None:
            self.flush_aux()
            self._fp.flush()
            self._fp.close()
            self._fp = None
            self._opened = False

    # -- Aux data: only the per-item write differs from the shared base ----------------

    def _write_dataset_aux_item(self, name: str, value: str) -> None:
        self._check_handle().write(f"DATASETAUXDATA {name}={_quote(value)}\n")

    def _write_var_aux_item(
        self, one_based_var_index: int, name: str, value: str
    ) -> None:
        self._check_handle().write(
            f"VARAUXDATA {one_based_var_index} {name}={_quote(value)}\n"
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
    ) -> None:
        """Write a complete IJK-ordered zone.

        Zone dimensions are inferred from the shape of the first NODAL array.

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
        if isinstance(datapacking, str):
            try:
                datapacking = DataPacking[datapacking.upper()]
            except KeyError:
                raise ValueError(
                    f"datapacking={datapacking!r} is not supported; "
                    "use DataPacking.BLOCK, DataPacking.POINT, or their "
                    "string equivalents."
                ) from None
        new_zone_index = self.current_zone + 1
        if title is None:
            title = f"IJK_Zone_{new_zone_index}"
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(data) + 1)]

        # Open the file if lazily loaded and flush buffered aux data
        if not self._opened:
            self._open(variables)
        if self.current_zone == -1:
            self.flush_aux()

        # Convert input data to NumPy arrays
        arrays: list[npt.NDArray] = [np.asarray(arr) for arr in data]

        variable_types = [
            _resolve_written_type(_infer_data_type(arr), self.precision)
            for arr in arrays
        ]

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
            on_error=self._handle_zone_error,
        )
        if prepared is None:
            return
        passive_flags = prepared.passive_vars
        share_idx = prepared.var_sharing
        imax, jmax, kmax = prepared.imax, prepared.jmax, prepared.kmax
        value_locations_global = prepared.value_locations_global
        variable_types_global = prepared.variable_types_global

        buf = io.StringIO()
        self._stage_zone_header(
            buf,
            title,
            ZoneType.ORDERED,
            imax=imax,
            jmax=jmax,
            kmax=kmax,
            solution_time=solution_time,
            strand_id=strand_id,
            passive_vars=passive_flags,
            var_sharing=share_idx,
            value_locations_global=value_locations_global,
            variable_types_global=variable_types_global,
            aux=aux,
            datapacking=datapacking,
        )

        if datapacking == DataPacking.POINT:
            # Nodal section: one row per node, all nodal vars per row.
            nodal_cols = [
                np.asarray(arr, dtype=_DT_TO_DTYPE[dt]).ravel(order="F")
                for arr, dt, loc in zip(
                    data, variable_types, value_locations, strict=True
                )
                if loc == ValueLocation.NODAL
            ]
            _stage_point_rows(buf, nodal_cols, self._float_fmt)
            # Cell-centred section: one row per cell, all CC vars per row.
            cc_cols = [
                np.asarray(arr, dtype=_DT_TO_DTYPE[dt]).ravel(order="F")
                for arr, dt, loc in zip(
                    data, variable_types, value_locations, strict=True
                )
                if loc == ValueLocation.CELL_CENTERED
            ]
            _stage_point_rows(buf, cc_cols, self._float_fmt)
        else:
            for arr, dt in zip(data, variable_types, strict=False):
                cast = np.asarray(arr, dtype=_DT_TO_DTYPE[dt]).ravel(order="F")
                _stage_float_array(buf, cast, self._float_fmt)

        self._check_handle().write(buf.getvalue())
        self.current_zone = new_zone_index

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
        face_neighbors_complete: bool | None = None,
        solution_time: float = 0.0,
        strand_id: int = 0,
        aux: dict[str, Any] | None = None,
        datapacking: DataPacking | str = DataPacking.BLOCK,
    ) -> None:
        """Write a complete finite-element zone.

        Node and cell counts are inferred from *node_map*, or if *node_map* is omitted
        from the zone referenced by *con_sharing*.

        Warning:
            ``FEPOLYGON`` and ``FEPOLYHEDRON`` are face-based and raise
            :exc:`ValueError`, write them with :meth:`write_poly_zone`.

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
            face_neighbors_complete (bool):
                Whether ``face_neighbors`` is the entire adjacency picture (``True``),
                Tecplot shouldn't auto-detect anything further, or a supplement to
                auto-detected conformal adjacency (``False``). ``None`` (the default)
                omits the ``FEFACENEIGHBORSCOMPLETE`` keyword entirely, deferring to the
                Tecplot default. ASCII-only: neither the classic nor new C API can
                express this, so it has no equivalent on
                :class:`~tecio.TecplotSzlWriter` or :class:`~tecio.TecplotPltWriter`.
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
        if isinstance(datapacking, str):
            try:
                datapacking = DataPacking[datapacking.upper()]
            except KeyError:
                raise ValueError(
                    f"datapacking={datapacking!r} is not supported; "
                    "use DataPacking.BLOCK, DataPacking.POINT, or their "
                    "string equivalents."
                ) from None
        if zone_type in _FE_POLY:
            raise ValueError(
                f"{zone_type.name} zones are face-based, use write_poly_zone."
            )
        if zone_type not in _FE_SIMPLE:
            raise NotImplementedError(
                f"Zone type {zone_type!r} is not supported by write_fe_zone."
            )

        new_zone_index = self.current_zone + 1
        if title is None:
            title = f"FE_Zone_{new_zone_index}"
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(data) + 1)]

        # Open the file if lazily loaded and flush buffered aux data
        if not self._opened:
            self._open(variables)
        if self.current_zone == -1:
            self.flush_aux()

        variable_types = [
            _resolve_written_type(_infer_data_type(np.asarray(arr)), self.precision)
            for arr in data
        ]

        # Convert input data to NumPy arrays
        arrays: list[npt.NDArray] = [np.asarray(arr) for arr in data]

        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * len(data)

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
            on_error=self._handle_zone_error,
        )
        if prepared is None:
            return
        passive_flags = prepared.passive_vars
        share_idx = prepared.var_sharing
        con_sharing = prepared.con_sharing
        num_nodes = prepared.num_nodes
        num_cells = prepared.num_cells
        value_locations_global = prepared.value_locations_global
        variable_types_global = prepared.variable_types_global
        # DAT writes face-neighbor rows via the same int-index-based
        # walk/reshape as node_map (see below), so the dtype cast (not
        # forced by the shared preparation step, which leaves it to each
        # format) happens here, matching what DAT always used before.
        face_neighbors_arr: npt.NDArray | None = None
        if prepared.face_neighbors_arr is not None:
            face_neighbors_arr = prepared.face_neighbors_arr.astype(np.intp)
        face_neighbor_mode = prepared.face_neighbor_mode
        num_face_connections = prepared.num_face_connections

        buf = io.StringIO()
        self._stage_zone_header(
            buf,
            title,
            zone_type,
            num_nodes=num_nodes,
            num_elements=num_cells,
            solution_time=solution_time,
            strand_id=strand_id,
            passive_vars=passive_flags,
            var_sharing=share_idx,
            value_locations_global=value_locations_global,
            variable_types_global=variable_types_global,
            con_sharing=con_sharing,
            aux=aux,
            datapacking=datapacking,
            num_face_connections=num_face_connections,
            face_neighbor_mode=face_neighbor_mode or FaceNeighborMode.LOCAL_ONE_TO_ONE,
            face_neighbors_complete=face_neighbors_complete,
        )

        if datapacking == DataPacking.POINT:
            # Nodal section: one row per node, all nodal vars per row.
            nodal_cols = [
                np.asarray(arr, dtype=_DT_TO_DTYPE[dt]).ravel()
                for arr, dt, loc in zip(
                    data, variable_types, value_locations, strict=True
                )
                if loc == ValueLocation.NODAL
            ]
            _stage_point_rows(buf, nodal_cols, self._float_fmt)
            # Cell-centred section: one row per cell, all CC vars per row.
            cc_cols = [
                np.asarray(arr, dtype=_DT_TO_DTYPE[dt]).ravel()
                for arr, dt, loc in zip(
                    data, variable_types, value_locations, strict=True
                )
                if loc == ValueLocation.CELL_CENTERED
            ]
            _stage_point_rows(buf, cc_cols, self._float_fmt)
        else:
            for arr, dt in zip(data, variable_types, strict=False):
                cast = np.asarray(arr, dtype=_DT_TO_DTYPE[dt]).ravel()
                _stage_float_array(buf, cast, self._float_fmt)

        # Write connectivity (if not shared). Node numbers are 0-based in Python;
        # ASCII text always uses 1-based node numbers.
        if con_sharing is None:
            assert node_map is not None
            conn = (
                np.asarray(node_map, dtype=np.intp).reshape(
                    num_cells, _NODES_PER_ELEM[zone_type]
                )
                + 1
            )
            for row in conn:
                _stage_connectivity_row(buf, row)

        # Write face-neighbor connections (if provided). Cell and remote-zone
        # references are 0-based in Python; ASCII text is always fully 1-based.
        if face_neighbors_arr is not None:
            assert face_neighbor_mode is not None  # narrowed above
            flat = convert_face_connections_to_one_based(
                face_neighbors_arr, face_neighbor_mode
            )
            one_to_one = (
                FaceNeighborMode.LOCAL_ONE_TO_ONE,
                FaceNeighborMode.GLOBAL_ONE_TO_ONE,
            )
            if face_neighbor_mode in one_to_one:
                values_per = (
                    3 if face_neighbor_mode is FaceNeighborMode.LOCAL_ONE_TO_ONE else 4
                )
                for row in flat.reshape(num_face_connections, values_per):
                    _stage_connectivity_row(buf, row)
            else:
                # "Many" modes: ragged, each record's own 4th value (nz)
                # gives the count of additional neighbor references that
                # follow it, matching the reader's own walk exactly.
                is_global = face_neighbor_mode is FaceNeighborMode.GLOBAL_ONE_TO_MANY
                i = 0
                for _ in range(num_face_connections):
                    nz = int(flat[i + 3])
                    record_len = 4 + (2 * nz if is_global else nz)
                    _stage_connectivity_row(buf, flat[i : i + record_len])
                    i += record_len

        self._check_handle().write(buf.getvalue())
        self.current_zone = new_zone_index

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
                num_face_connections=num_face_connections or None,
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
        *con_sharing* source zone when it is omitted. Every index is 0-based in
        Python and shifted to the 1-based ASCII convention here (0 for no element,
        ``-k`` for the k-th boundary face).

        The face map follows the variable blocks as integer sections introduced by the
        same ``#`` comment lines Tecplot writes: ``# node count per face`` (polyhedra
        only), ``# face nodes``, ``# left elements``, ``# right elements``, and with
        boundary connections ``# boundary connection counts``, ``# boundary connection
        elements``, and ``# boundary connection zones``. Nothing is written for a zone
        that shares connectivity.

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
                Must be :attr:`~tecio.DataPacking.BLOCK`. Tecplot refuses ``POINT``
                for poly zones, whether or not they have cell-centered variables.

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
                    f"datapacking={datapacking!r} is not supported; "
                    "use DataPacking.BLOCK or its string equivalent."
                ) from None
        if datapacking != DataPacking.BLOCK:
            raise ValueError(
                "Polygon and polyhedron zones must use BLOCK data packing."
            )

        new_zone_index = self.current_zone + 1
        if title is None:
            title = f"Poly_Zone_{new_zone_index}"
        if variables is None:
            variables = [f"V{i}" for i in range(1, len(data) + 1)]

        # Open the file if lazily loaded and flush buffered aux data
        if not self._opened:
            self._open(variables)
        if self.current_zone == -1:
            self.flush_aux()

        arrays: list[npt.NDArray] = [np.asarray(arr) for arr in data]
        variable_types = [
            _resolve_written_type(_infer_data_type(arr), self.precision)
            for arr in arrays
        ]

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
            on_error=self._handle_zone_error,
        )
        if prepared is None:
            return
        conn = prepared.connectivity

        buf = io.StringIO()
        self._stage_zone_header(
            buf,
            title,
            zone_type,
            num_nodes=conn.num_nodes,
            num_elements=conn.num_cells,
            solution_time=solution_time,
            strand_id=strand_id,
            passive_vars=prepared.passive_vars,
            var_sharing=prepared.var_sharing,
            value_locations_global=prepared.value_locations_global,
            variable_types_global=prepared.variable_types_global,
            con_sharing=prepared.con_sharing,
            aux=aux,
            datapacking=datapacking,
            num_faces=conn.num_faces,
            total_face_nodes=conn.total_face_nodes,
            num_boundary_faces=conn.num_boundary_faces,
            num_boundary_connections=conn.num_boundary_connections,
        )

        for arr, dt in zip(arrays, variable_types, strict=True):
            _stage_float_array(
                buf, arr.astype(_DT_TO_DTYPE[dt], copy=False), self._float_fmt
            )

        # Face map (if not shared). ASCII is the 0-based API convention plus one for
        # every array: 0 for no element and -k for the k-th boundary face.
        if prepared.con_sharing is None:
            assert conn.face_nodes is not None
            assert (
                conn.face_left_elements is not None
                and conn.face_right_elements is not None
            )
            if zone_type is ZoneType.FEPOLYHEDRON:
                assert conn.face_node_counts is not None
                _stage_int_block(buf, "node count per face", conn.face_node_counts)
            _stage_int_block(buf, "face nodes", conn.face_nodes.astype(np.int64) + 1)
            _stage_int_block(
                buf, "left elements", conn.face_left_elements.astype(np.int64) + 1
            )
            _stage_int_block(
                buf, "right elements", conn.face_right_elements.astype(np.int64) + 1
            )
            if conn.num_boundary_faces:
                assert conn.boundary_connection_counts is not None
                assert conn.boundary_connection_elements is not None
                assert conn.boundary_connection_zones is not None
                _stage_int_block(
                    buf, "boundary connection counts", conn.boundary_connection_counts
                )
                _stage_int_block(
                    buf,
                    "boundary connection elements",
                    conn.boundary_connection_elements.astype(np.int64) + 1,
                )
                _stage_int_block(
                    buf,
                    "boundary connection zones",
                    conn.boundary_connection_zones.astype(np.int64) + 1,
                )

        self._check_handle().write(buf.getvalue())
        self.current_zone = new_zone_index

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
                shared_vars=tuple(prepared.var_sharing),
                data_types=tuple(prepared.variable_types_global),
            )
        )

    # -- Private helpers --------------------------------------------------------------

    def _handle_zone_error(self) -> None:
        """Delete the output file if no zone has been committed yet."""
        if self.current_zone == -1:
            if self._fp is not None:
                self._fp.close()
                self._fp = None
                self._opened = False
            with contextlib.suppress(OSError):
                os.remove(self.path)

    def _write_file_header(self) -> None:
        """Write TITLE, optional FILETYPE, and VARIABLES lines."""
        fp = self._check_handle()
        fp.write(f"TITLE     = {_quote(self.title)}\n")
        if self.file_type in _FILETYPE_STR:
            fp.write(f"FILETYPE  = {_FILETYPE_STR[self.file_type]}\n")
        var_lines = "\n".join(_quote(v) for v in self._check_variables())
        fp.write(f"VARIABLES = {var_lines}\n")

    def _stage_zone_header(
        self,
        buf: io.StringIO,
        title: str,
        zone_type: ZoneType,
        imax: int = 0,
        jmax: int = 0,
        kmax: int = 0,
        num_nodes: int = 0,
        num_elements: int = 0,
        solution_time: float = 0.0,
        strand_id: int = 0,
        passive_vars: Sequence[bool | int] | None = None,
        var_sharing: Sequence[int | None] | None = None,
        value_locations_global: list[ValueLocation] | None = None,
        variable_types_global: list[DataType] | None = None,
        con_sharing: int | None = None,
        aux: dict[str, Any] | None = None,
        datapacking: DataPacking = DataPacking.BLOCK,
        num_face_connections: int = 0,
        face_neighbor_mode: FaceNeighborMode = FaceNeighborMode.LOCAL_ONE_TO_ONE,
        face_neighbors_complete: bool | None = None,
        num_faces: int = 0,
        total_face_nodes: int = 0,
        num_boundary_faces: int = 0,
        num_boundary_connections: int = 0,
    ) -> None:
        """Write a ``ZONE`` header block into the staging buffer *buf*."""
        zt_str = _ZONETYPE_STR[zone_type]

        # First line is never indented; every subsequent header line is indented.
        buf.write(f"ZONE T={_quote(title)}\n")
        buf.write(
            f"{_INDENT}STRANDID={strand_id}, SOLUTIONTIME={float(solution_time)}\n"
        )

        if zone_type == ZoneType.ORDERED:
            buf.write(f"{_INDENT}I={imax}, J={jmax}, K={kmax}, ZONETYPE={zt_str}\n")
        elif zone_type in _FE_POLY:
            buf.write(
                f"{_INDENT}Nodes={num_nodes}, Faces={num_faces}, "
                f"Elements={num_elements}, ZONETYPE={zt_str}\n"
            )
        else:
            buf.write(
                f"{_INDENT}Nodes={num_nodes}, Elements={num_elements}, "
                f"ZONETYPE={zt_str}\n"
            )

        buf.write(f"{_INDENT}DATAPACKING={datapacking.name}\n")

        if zone_type in _FE_POLY:
            # A polygon face always has 2 nodes, so only polyhedra record the total.
            face_nodes_str = (
                f"TotalNumFaceNodes={total_face_nodes}, "
                if zone_type is ZoneType.FEPOLYHEDRON
                else ""
            )
            buf.write(
                f"{_INDENT}{face_nodes_str}"
                f"NumConnectedBoundaryFaces={num_boundary_faces}, "
                f"TotalNumBoundaryConnections={num_boundary_connections}\n"
            )

        if num_face_connections > 0:
            buf.write(f"{_INDENT}FACENEIGHBORCONNECTIONS={num_face_connections}\n")
            buf.write(
                f"{_INDENT}FACENEIGHBORMODE="
                f"{_FACENEIGHBORMODE_STR[face_neighbor_mode]}\n"
            )
            if face_neighbors_complete is not None:
                complete_str = "YES" if face_neighbors_complete else "NO"
                buf.write(f"{_INDENT}FEFACENEIGHBORSCOMPLETE={complete_str}\n")

        if variable_types_global:
            dt_str = " ".join(_DATATYPE_STR[dt] for dt in variable_types_global)
            buf.write(f"{_INDENT}DT=({dt_str})\n")

        if value_locations_global:
            cc = [
                i + 1
                for i, loc in enumerate(value_locations_global)
                if loc == ValueLocation.CELL_CENTERED
            ]
            if cc:
                buf.write(
                    f"{_INDENT}VARLOCATION=([{','.join(str(i) for i in cc)}]"
                    "=CELLCENTERED)\n"
                )

        if passive_vars:
            pidx = [str(i + 1) for i, f in enumerate(passive_vars) if f]
            if pidx:
                buf.write(f"{_INDENT}PASSIVEVARLIST=[{','.join(pidx)}]\n")

        if var_sharing and any(z is not None for z in var_sharing):
            entries = [
                f"[{i + 1}]={z + 1}" for i, z in enumerate(var_sharing) if z is not None
            ]
            if entries:
                buf.write(f"{_INDENT}VARSHARELIST=({','.join(entries)})\n")

        if con_sharing is not None:
            buf.write(f"{_INDENT}CONNECTIVITYSHAREZONE={con_sharing + 1}\n")

        if aux:
            for name, value in aux.items():
                buf.write(f"{_INDENT}AUXDATA {name}={_quote(value)}\n")
