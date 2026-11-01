"""Entry point for tecio such that IO capability can be called from a single function.

Supported modes mirror Python's built-in :func:`open`:

    ======  ============================================================================
    Mode    Behaviour
    ======  ============================================================================
    ``r``   Open existing file for reading. Raises :exc:`FileNotFoundError` if the file
            does not exist.
    ``w``   Open file for writing. Overwrites any existing file.
    ``x``   Exclusive creation — open for writing only if the file does **not** already
            exist. Raises :exc:`FileExistsError` otherwise.
    ``a``   Append — stream all zones from the existing file into a new temporary file,
            then continue writing new zones into that same file. On
            :meth:`~AppendWrite.close` the temporary file atomically replaces the
            original path. Returns an :class:`AppendWrite` object whose write API is
            identical to the format's ``Write`` class.
    ``a+``  Append-read — same streaming copy as ``a``, but the returned
            :class:`AppendReadWrite` object also exposes the full ``Read`` API populated
            from the *original* file so that existing zone data can be inspected while
            new zones are written.
    ======  ============================================================================

Note:
    ``"a"`` and ``"a+"`` work by reading the source file in full and re-writing it
    zone-by-zone into a temporary sibling file, then leaving the ``Write`` handle open
    for the caller to add further zones. On close the temporary file is atomically
    renamed over the original path (POSIX) or replaced (Windows). This is the only safe
    approach because PLT and SZL are sequential binary formats with no in-place editing
    capability.

Note:
    Legacy PLT files (v75, v107, v111) are read-only, so ``"a"`` and ``"a+"`` raise
    :exc:`ValueError` for them.
"""

# ruff: noqa: A001 N806

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Literal, NamedTuple, overload

import numpy.typing as npt

from ._constants import FileType, ValueLocation, ZoneType
from ._containers import ZoneList
from ._dat_read import TecplotDatReader, _peek_dat_metadata
from ._dat_write import TecplotDatWriter
from ._plt_read import TecplotPltReader, _is_legacy_plt
from ._plt_write import TecplotPltWriter
from ._reader import (
    TecplotAuxDataReader,
    TecplotFEZoneReader,
    TecplotOrderedZoneReader,
    TecplotPolyZoneReader,
    TecplotReader,
    TecplotZoneReader,
)
from ._szl_read import TecplotSzlReader
from ._szl_write import TecplotSzlWriter
from ._writer import TecplotWriter, poly_face_kwargs

# --------------------------------------------------------------------------------------
# Dispatch table
# --------------------------------------------------------------------------------------

_HANDLERS: dict[str, dict[str, Any]] = {
    ".szplt": {
        "r": TecplotSzlReader,
        "w": TecplotSzlWriter,
        "x": None,  # filled below after class definitions
        "a": None,
        "a+": None,
    },
    ".plt": {
        "r": TecplotPltReader,
        "w": TecplotPltWriter,
        "x": None,
        "a": None,
        "a+": None,
    },
    ".bin": {
        "r": TecplotPltReader,
        "w": TecplotPltWriter,
        "x": None,
        "a": None,
        "a+": None,
    },
    ".bbin": {
        "r": TecplotPltReader,
        "w": TecplotPltWriter,
        "x": None,
        "a": None,
        "a+": None,
    },
    ".bbin8": {
        "r": TecplotPltReader,
        "w": TecplotPltWriter,
        "x": None,
        "a": None,
        "a+": None,
    },
    ".dat": {
        "r": TecplotDatReader,
        "w": TecplotDatWriter,
        "x": None,
        "a": None,
        "a+": None,
    },
    ".tec": {
        "r": TecplotDatReader,
        "w": TecplotDatWriter,
        "x": None,
        "a": None,
        "a+": None,
    },
}

# Extensions handled by the PLT reader, which may hold a legacy (read-only) version.
_PLT_EXTS: frozenset[str] = frozenset({".plt", ".bin", ".bbin", ".bbin8"})

# FE zone types that the Write API supports for zone copying.
_FE_SIMPLE: frozenset[ZoneType] = frozenset({
    ZoneType.FELINESEG,
    ZoneType.FETRIANGLE,
    ZoneType.FEQUADRILATERAL,
    ZoneType.FETETRAHEDRON,
    ZoneType.FEBRICK,
})


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------


def _reader_for_ext(ext: str) -> type[TecplotReader]:
    """Return the Read class for *ext*."""
    return _HANDLERS[ext]["r"]


def _writer_for_ext(ext: str) -> type:
    """Return the Write class for *ext*."""
    return _HANDLERS[ext]["w"]


def _copy_zones(reader: TecplotReader, writer: TecplotWriter) -> None:
    """Stream all zones from *reader* into the open *writer*.

    Each zone is copied variable-by-variable at its original data type and value
    location. Both variable sharing (``var_sharing``) and FE connectivity sharing
    (``con_sharing``) are preserved: a zone that shares its connectivity is
    re-emitted as a share rather than a duplicated node map.

    Args:
        reader: An already-opened ``Read`` instance.
        writer: An already-opened ``Write`` instance (file handle live).

    Raises:
        NotImplementedError: If a zone is neither ordered, finite-element, nor poly.

    """
    for zone in reader.zones:
        # Collect per-variable metadata and data arrays
        data: list[npt.NDArray] = []
        value_locations: list[ValueLocation] = []
        passive_vars: set[str] = set()
        var_sharing: dict[str, int] = {}

        for var in zone.variables:
            sv = var.shared_zone  # None, or the 0-based source-zone index shared from
            if var.is_passive():
                passive_vars.add(var.name)
            elif sv is not None:
                var_sharing[var.name] = sv
            else:
                data.append(var.values)  # ty: ignore[invalid-argument-type]
                value_locations.append(var.value_location)

        common_kw: dict[str, Any] = {
            "title": zone.title,
            "value_locations": value_locations,
            "passive_vars": passive_vars,
            "var_sharing": var_sharing,
            "solution_time": zone.solution_time,
            "strand_id": zone.strand_id,
            "aux": dict(zone.auxdata.items()) or None,
        }

        if isinstance(zone, TecplotOrderedZoneReader):
            writer.write_ordered_zone(data=data, **common_kw)
        elif isinstance(zone, TecplotPolyZoneReader):
            con_sharing = zone.shared_connectivity
            writer.write_poly_zone(
                zone_type=zone.zone_type,
                data=data,
                con_sharing=con_sharing,
                **({} if con_sharing is not None else poly_face_kwargs(zone)),
                **common_kw,
            )
        elif isinstance(zone, TecplotFEZoneReader):
            # Forward connectivity sharing: when shared, pass ``con_sharing`` and omit
            # the node map so the writer derives the node/cell counts from the source
            # zone; otherwise write this zone's own node map.
            con_sharing = zone.shared_connectivity  # None, or 0-based source-zone index
            writer.write_fe_zone(
                zone_type=zone.zone_type,
                data=data,
                node_map=None if con_sharing is not None else zone.node_map,
                con_sharing=con_sharing,
                **common_kw,
            )
        else:
            raise NotImplementedError(
                f"Zone '{zone.title}' is neither an ordered, classic FE, nor poly "
                "zone, copying it is not supported by the Write API yet."
            )


# --------------------------------------------------------------------------------------
# AppendWrite — returned by open(..., "a")
# --------------------------------------------------------------------------------------


class AppendWrite:
    """Write handle returned by :func:`tecio.open` with ``mode='a'``.

    Appends new zones to an existing Tecplot file. Internally the existing file is
    stream-copied zone-by-zone into a temporary sibling file, and the write handle is
    left open past all copied zones so new zones can be appended. On :meth:`close` (or
    context-manager exit) the temporary file is atomically renamed over the original
    path — a POSIX rename on Unix, :func:`Path.replace` on Windows.

    The copy step preserves the original title, variable list, file type, all zone data,
    all zone metadata (strand, solution time, aux data), and variable sharing.
    Connectivity is copied for FE and poly zones.

    Do not instantiate this class directly — use :func:`tecio.open`.

    Example:
        >>> with tecio.open("flow.szplt", "a") as tec:
        ...     print(tec.variables)  # variable list from the existing file
        ...     print(tec.current_zone)  # number of zones already copied
        ...     tec.write_ordered_zone(
        ...         data=[x_new, y_new, p_new],
        ...         solution_time=10.0,
        ...         strand_id=1,
        ...     )
        >>> # flow.szplt now contains the original zones plus the new one

    """

    def __init__(
        self,
        original_path: str | os.PathLike,
        tmp_path: str | os.PathLike,
        writer: TecplotWriter,
    ) -> None:
        self._original = Path(original_path)
        self._tmp = Path(tmp_path)
        self._writer = writer
        self._closed = False

    # -- Context manager ---------------------------------------------------------------

    def __enter__(self) -> AppendWrite:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

    # -- Write delegation --------------------------------------------------------------

    def write_ordered_zone(self, *args: Any, **kwargs: Any) -> None:  # noqa: D417
        """Append a structured IJK-ordered zone.

        Delegates to the underlying format writer's ``write_ordered_zone``
        (e.g. :meth:`tecio.TecplotWriter.write_ordered_zone`). All parameters from that
        method are accepted here with one exception: ``variables`` is not meaningful
        because the variable list is fixed at open time from the existing file.

        Args:
            data (Sequence[ndarray]): One NumPy array per **active** variable
                (non-passive, non-shared). Shape ``(imax[, jmax[, kmax]])`` is inferred
                automatically. Fortran (column-major) order is assumed.
            title (str | None): Zone title. Defaults to ``"IJK_Zone_{n}"``.
            value_locations (Sequence[ValueLocation] | None): Per-variable
                :class:`~tecio.libtecio.ValueLocation` for active variables. Defaults
                to all :attr:`~tecio.libtecio.ValueLocation.NODAL`.
            passive_vars (list | tuple | set | None): Variables to mark passive: a list,
                tuple, or set of variable names, or of 0-based indices. Defaults to all
                active.
            var_sharing (dict[str | int, int] | None): Variables to share from another
                zone: a ``{variable: source zone}`` mapping (name or 0-based index to
                0-based zone index, counting all zones including those copied from the
                original file). Defaults to no sharing. Sharing grid coordinates from
                zone 0 is a common pattern for transient data to avoid duplicating large
                arrays.
            solution_time (float): Solution time for transient data. Defaults to
                ``0.0``.
            strand_id (int): Strand ID grouping related time steps. Zones with the same
                strand ID animate together in Tecplot 360. Defaults to ``0`` (static).
            aux (dict[str, str] | None): Zone-level auxiliary data as ``{name: value}``
                pairs. Defaults to ``None``.

        Example:
            Append a time step, sharing the grid from zone 0:

            >>> tec.write_ordered_zone(
            ...     data=[p_new],  # only the non-shared variable
            ...     var_sharing={"x": 0, "y": 0},
            ...     solution_time=5.0,
            ...     strand_id=1,
            ... )

        """
        self._writer.write_ordered_zone(*args, **kwargs)

    def write_fe_zone(self, *args: Any, **kwargs: Any) -> None:  # noqa: D417
        """Append an unstructured finite-element zone.

        Delegates to the underlying format writer's ``write_fe_zone``
        (e.g. :meth:`tecio.TecplotWriter.write_fe_zone`). The ``variables`` parameter is
        not meaningful here — the variable list is fixed from the existing file.
        FEPOLYGON and FEPOLYHEDRON are face-based, use :meth:`write_poly_zone`.

        Args:
            zone_type (ZoneType): Element topology. One of:
                :attr:`~tecio.libtecio.ZoneType.FELINESEG`,
                :attr:`~tecio.libtecio.ZoneType.FETRIANGLE`,
                :attr:`~tecio.libtecio.ZoneType.FEQUADRILATERAL`,
                :attr:`~tecio.libtecio.ZoneType.FETETRAHEDRON`, or
                :attr:`~tecio.libtecio.ZoneType.FEBRICK`.
            data (Sequence[ndarray]): One 1-D array per active variable. NODAL arrays
                must have length ``num_nodes``; CELL_CENTERED arrays must have length
                ``num_cells``. Both are inferred from ``node_map``.
            node_map (array-like | None): Integer array of shape ``(num_cells,
                nodes_per_cell)`` with 0-based node indices. ``num_nodes =
                node_map.max() + 1``. The 32- or 64-bit write path is chosen
                automatically.
            title (str | None): Zone title. Defaults to ``"FE_Zone_{n}"``.
            value_locations (Sequence[ValueLocation] | None): Per-variable value
                location for active variables. Defaults to all
                :attr:`~tecio.libtecio.ValueLocation.NODAL`.
            passive_vars (list | tuple | set | None): Variables to mark passive: a list,
                tuple, or set of variable names, or of 0-based indices. Defaults to all
                active.
            var_sharing (dict[str | int, int] | None): Variables to share from another
                zone: a ``{variable: source zone}`` mapping (name or 0-based index to
                0-based zone index). Defaults to no sharing.
            con_sharing (int | None): Zone index to share connectivity from (0-based).
                ``None`` = write connectivity directly. Cannot be used with GLOBAL
                face-neighbor modes. Defaults to ``None``.
            face_neighbors (array-like | None): Face-neighbor connectivity. When
                provided, ``num_face_cons`` is set automatically.
            face_nbr_mode (FaceNeighborMode | None): Face-neighbor mode. Defaults to
                :attr:`~tecio.libtecio.FaceNeighborMode.LOCAL_ONE_TO_ONE`. Ignored when
                ``face_neighbors`` is ``None``.
            solution_time (float): Solution time. Defaults to ``0.0``.
            strand_id (int): Strand ID. Defaults to ``0``.
            aux (dict[str, str] | None): Zone-level auxiliary data.

        Raises:
            ValueError: If ``zone_type`` is FEPOLYGON or FEPOLYHEDRON.
        """
        self._writer.write_fe_zone(*args, **kwargs)

    def write_poly_zone(self, *args: Any, **kwargs: Any) -> None:  # noqa: D417
        """Append a polygon or polyhedron (face-based) zone.

        Delegates to the underlying format writer's ``write_poly_zone``
        (e.g. :meth:`tecio.TecplotPltWriter.write_poly_zone`). The ``variables``
        parameter is not meaningful here, the variable list is fixed from the existing
        file. All indices are 0-based.

        Args:
            zone_type (ZoneType): :attr:`~tecio.libtecio.ZoneType.FEPOLYGON` or
                :attr:`~tecio.libtecio.ZoneType.FEPOLYHEDRON`.
            data (Sequence[ndarray]): One 1-D array per active variable. NODAL arrays
                need ``num_nodes`` values, CELL_CENTERED arrays ``num_elements``, both
                inferred from the face map.
            face_nodes (array-like | None): Flat node indices of all faces.
            face_node_counts (array-like | None): Nodes per face. Required for
                polyhedra, optional for polygons (always 2).
            face_left_elements (array-like | None): Element left of each face, ``-1``
                for none, ``-(b + 2)`` for boundary face ``b``.
            face_right_elements (array-like | None): Element right of each face.
            boundary_connection_counts (array-like | None): Connections per boundary
                face.
            boundary_connection_elements (array-like | None): Flat neighbor elements.
            boundary_connection_zones (array-like | None): Flat zone of each neighbor.
            title (str | None): Zone title. Defaults to ``"Poly_Zone_{n}"``.
            value_locations (Sequence[ValueLocation] | None): Per-variable value
                location for active variables. Defaults to all NODAL.
            passive_vars (list | tuple | set | None): Variables to mark passive.
            var_sharing (dict[str | int, int] | None): ``{variable: source zone}``.
            con_sharing (int | None): Zone to share the face map from (0-based).
            solution_time (float): Solution time. Defaults to ``0.0``.
            strand_id (int): Strand ID. Defaults to ``0``.
            aux (dict[str, str] | None): Zone-level auxiliary data.

        Raises:
            NotImplementedError: For SZPLT files, which cannot hold poly zones.
            ValueError: On an inconsistent or out of range face map.
        """
        self._writer.write_poly_zone(*args, **kwargs)

    def add_auxdataset_dict(self, auxdict: dict[str, Any]) -> None:
        """Buffer dataset-level auxiliary data."""
        self._writer.add_auxdataset_dict(auxdict)

    def add_auxvar_dict(self, auxdict: dict[int, dict[str, Any]]) -> None:
        """Buffer variable-level auxiliary data."""
        self._writer.add_auxvar_dict(auxdict)

    def flush_aux(self) -> None:
        """Flush buffered auxiliary data to disk."""
        self._writer.flush_aux()

    # Expose title / variables / current_zone so callers can inspect state.
    @property
    def title(self) -> str:
        """Dataset title of the output file."""
        return self._writer.title

    @property
    def variables(self) -> list[str] | None:
        """Variable name list of the output file."""
        return self._writer.variables

    @property
    def current_zone(self) -> int:
        """0-based index of the most recently written zone, -1 before any write."""
        return self._writer.current_zone

    # -- Close and finalise ------------------------------------------------------------

    def close(self) -> None:
        """Finalise the temporary file and replace the original.

        Safe to call more than once.
        """
        if self._closed:
            return
        self._closed = True
        self._writer.close()
        # Atomic replace: on POSIX this is a rename; on Windows replace() also works
        # across drives by falling back to copy+delete.
        self._tmp.replace(self._original)


# --------------------------------------------------------------------------------------
# AppendReadWrite — returned by open(..., "a+")
# --------------------------------------------------------------------------------------


class AppendReadWrite(AppendWrite):
    """Read + write handle returned by :func:`tecio.open` with ``mode='a+'``.

    Inherits all write methods from :class:`AppendWrite` and additionally exposes the
    full read interface populated from the **original** file. The read data reflects
    the file as it existed before any new zones were appended — zones written during the
    current session are not visible via :attr:`zone` until the file is closed and
    re-opened.

    Useful when new zones depend on existing zone data — for example, computing a
    time-average over all previous time steps and writing it as an additional zone.

    Do not instantiate this class directly — use :func:`tecio.open`.

    Example:
        >>> with tecio.open("flow.szplt", "a+") as tec:
        ...     # Read from all zones present before this session
        ...     times = [tec.zones[i].solution_time for i in range(tec.num_zones)]
        ...     p_avg = (
        ...         sum(
        ...             tec.zones[i].variables["p"].values for i in range(tec.num_zones)
        ...         )
        ...         / tec.num_zones
        ...     )
        ...
        ...     # Append a new zone using the computed average
        ...     x = tec.zones[0].variables["x"].values
        ...     y = tec.zones[0].variables["y"].values
        ...     tec.write_ordered_zone(
        ...         data=[x, y, p_avg],
        ...         title="Time-average",
        ...         solution_time=max(times) + 1.0,
        ...         strand_id=2,
        ...     )

    """

    def __init__(
        self,
        original_path: str | os.PathLike,
        tmp_path: str | os.PathLike,
        writer: TecplotWriter,
        reader: TecplotReader,
    ) -> None:
        super().__init__(original_path, tmp_path, writer)
        self._reader = reader

    # -- Read delegation ---------------------------------------------------------------

    @property
    def file_type(self) -> FileType:
        """File type of the *original* file (FULL / GRID / SOLUTION)."""
        return self._reader.file_type

    @property
    def num_vars(self) -> int:
        """Number of variables in the *original* file."""
        return self._reader.num_vars

    @property
    def num_zones(self) -> int:
        """Number of zones in the *original* file (before appending)."""
        return self._reader.num_zones

    @property
    def zones(self) -> ZoneList[TecplotZoneReader]:
        """Zone list from the *original* file."""
        return self._reader.zones

    @property
    def auxdata(self) -> TecplotAuxDataReader:
        """Dataset-level auxiliary data from the *original* file."""
        return self._reader.auxdata

    @property
    def num_auxdata_items(self) -> int:
        """Number of dataset-level aux data items in the *original* file."""
        return self._reader.num_auxdata_items

    @property
    def var_auxdata(self) -> list[TecplotAuxDataReader]:
        """Per-variable auxiliary data list from the *original* file."""
        return self._reader.var_auxdata

    def get_var_auxdata(self, var_index: int) -> TecplotAuxDataReader:
        """Return variable aux data for *var_index* (0-based)."""
        return self._reader.get_var_auxdata(var_index)

    def get_zone_auxdata(self, zone_index: int) -> TecplotAuxDataReader:
        """Return zone aux data for *zone_index* (0-based)."""
        return self._reader.get_zone_auxdata(zone_index)


# --------------------------------------------------------------------------------------
# _open_append — shared implementation for "a" and "a+"
# --------------------------------------------------------------------------------------


def _open_append(
    path: str | os.PathLike,
    ext: str,
    read_write: bool,
    **writer_kwargs: Any,
) -> AppendWrite | AppendReadWrite:
    """Stream-copy *path* into a temp file and return an open write handle.

    Args:
        path:            Original file path.
        ext:             Lowercase file extension (e.g. ``".szplt"``).
        read_write:      If ``True`` return :class:`AppendReadWrite`; otherwise
                         :class:`AppendWrite`.
        **writer_kwargs: Extra keyword arguments forwarded to the ``Write`` constructor
                         (e.g. ``title``, ``file_type``).

    Returns:
        An :class:`AppendWrite` or :class:`AppendReadWrite` instance with the ``Write``
        handle already past all original zones and ready for new zones to be written.

    Raises:
        FileNotFoundError: If *path* does not exist (nothing to append to).
        ValueError: If *path* is a legacy PLT file (v75, v107, or v111), which is
            read-only.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(
            f"Cannot append to '{src}': file does not exist.  "
            "Use mode='w' to create a new file."
        )
    if ext in _PLT_EXTS and _is_legacy_plt(src):
        raise ValueError(
            f"Cannot append to '{src}': legacy PLT versions (v75, v107, v111) are "
            "read-only. Open with mode='r', or convert to a current format first."
        )

    ReadCls = _reader_for_ext(ext)
    WriteCls = _writer_for_ext(ext)

    # Open the original file for reading.
    reader: TecplotReader = ReadCls(str(src))

    # Resolve writer kwargs from the source file when not explicitly supplied.
    title: str = writer_kwargs.pop("title", reader.title)
    file_type: FileType = writer_kwargs.pop("file_type", reader.file_type)
    variables: list[str] = writer_kwargs.pop("variables", reader.variables)

    # Create a sibling temporary file so the rename at close stays on the same
    # filesystem (required for an atomic os.rename on most OSes).
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=src.parent,
        prefix=f".{src.stem}_append_",
        suffix=src.suffix,
    )
    os.close(tmp_fd)  # WriteCls will open it independently
    tmp_path = Path(tmp_path_str)

    try:
        writer = WriteCls(
            str(tmp_path),
            title=title,
            variables=variables,
            file_type=file_type,
            **writer_kwargs,
        )
        # Replay all existing zones into the new file.
        _copy_zones(reader, writer)
    except Exception:
        # Close writer and clean up the temp file if anything goes wrong during the
        # copy.
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        tmp_path.unlink(missing_ok=True)
        raise

    if read_write:
        return AppendReadWrite(
            original_path=src,
            tmp_path=tmp_path,
            writer=writer,
            reader=reader,
        )
    return AppendWrite(
        original_path=src,
        tmp_path=tmp_path,
        writer=writer,
    )


# --------------------------------------------------------------------------------------
# _open_exclusive — implementation for "x"
# --------------------------------------------------------------------------------------


def _open_exclusive(
    path: str | os.PathLike,
    ext: str,
    **writer_kwargs: Any,
) -> TecplotWriter:
    """Open *path* for writing only if it does not already exist.

    Args:
        path: Target file path.
        ext:  Lowercase file extension.
        **writer_kwargs: Forwarded to the ``Write`` constructor.

    Returns:
        A ``Write`` instance, identical to ``open(path, 'w')``.

    Raises:
        FileExistsError: If *path* already exists.
    """
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"Cannot create '{target}': file already exists.  "
            "Use mode='w' to overwrite or mode='a' to append."
        )
    WriteCls = _writer_for_ext(ext)
    return WriteCls(str(target), **writer_kwargs)


# ======================================================================================
# Public open()
# ======================================================================================


# fmt: off
@overload
def open(
    path: str | os.PathLike,
    mode: Literal["r"],
    **kwargs: Any,
) -> TecplotReader: ...
@overload
def open(
    path: str | os.PathLike,
    mode: Literal["w"] | Literal["x"],
    **kwargs: Any,
) -> TecplotWriter: ...
@overload
def open(
    path: str | os.PathLike,
    mode: Literal["a"],
    **kwargs: Any,
) -> AppendWrite: ...
@overload
def open(
    path: str | os.PathLike,
    mode: Literal["a+"],
    **kwargs: Any,
) -> AppendReadWrite: ...
# fmt: on


def open(
    path: str | os.PathLike,
    mode: str = "r",
    **kwargs: Any,
) -> TecplotReader | TecplotWriter | AppendWrite | AppendReadWrite:
    """Open a Tecplot file for reading, writing, or appending.

    Selects the correct format handler from the file extension and returns the
    appropriate reader or writer. All handles are context managers.

    Args:
        path (str | os.PathLike): File path. Extension determines the format handler:

            - ``.szplt`` → :mod:`tecio.szl`
            - ``.plt`` / ``.bin`` / ``.bbin`` / ``.bbin8`` → :mod:`tecio.plt`
            - ``.dat`` / ``.tec`` → :mod:`tecio.dat`

        mode (str): Access mode — mirrors Python's built-in :func:`open`:

            - ``'r'`` -- Open Tecplot file for reading.
            - ``'w'`` -- Open Tecplot file for writing, truncating the file first.
            - ``'x'`` -- Open Tecplot file for exclusive creation, failing if the file
              already exists.
            - ``'a'`` -- Append new zones to an existing file. The file is
              stream-copied to a temporary file; on close the temporary atomically
              replaces the original. Raises :exc:`FileNotFoundError` if the file does
              not exist (use ``'w'`` to create a new file).
            - ``'a+'`` -- Same as ``'a'`` but the returned handle also exposes the full
              read API populated from the original file.

            Legacy PLT files (v75, v107, v111) can only be opened with ``'r'``;
            ``'a'`` and ``'a+'`` raise :exc:`ValueError` for them.

        **kwargs: Forwarded to the underlying handler constructor. The accepted keyword
            arguments differ by mode — see the tables below.

    Keyword Args (modes ``'r'``, ``'w'``, ``'x'``):
        Forwarded directly to the format's ``Read`` or ``Write`` constructor.

        For **reading** — no keyword arguments are accepted; all metadata is read from
        the file.

        For **writing** (``'w'`` and ``'x'``):

        .. list-table::
           :header-rows: 1
           :widths: 20 15 65

           * - Keyword
             - Default
             - Description
           * - ``title``
             - ``""``
             - Dataset title string embedded in the file header.
           * - ``variables``
             - ``None``
             - Variable name list. For SZL and PLT this may be deferred to the first
               :meth:`~tecio.TecplotWriter.write_ordered_zone`,
               :meth:`~tecio.TecplotWriter.write_fe_zone`, or
               :meth:`~tecio.TecplotWriter.write_poly_zone` call. Required at open time
               for DAT.
           * - ``file_type``
             - ``FileType.FULL``
             - One of :attr:`~tecio.libtecio.FileType.FULL`,
               :attr:`~tecio.libtecio.FileType.GRID`, or
               :attr:`~tecio.libtecio.FileType.SOLUTION`.

    Keyword Args (modes ``'a'``, ``'a+'``):
        ``title``, ``variables``, and ``file_type`` default to the values read from the
        existing file. In most cases you should **not** pass these. Providing them will
        override the source file when written out.

    Returns:
        - ``'r'`` → :class:`~tecio.TecplotSzlReader`, :class:`~tecio.TecplotPltReader`,
          or :class:`~tecio.TecplotDatReader` (all share the
          :class:`~tecio.TecplotReader` interface)
        - ``'w'`` / ``'x'`` → :class:`~tecio.TecplotSzlWriter`,
          :class:`~tecio.TecplotPltWriter`, or :class:`~tecio.TecplotDatWriter` (all
          share the :class:`~tecio.TecplotWriter` interface)
        - ``'a'`` → :class:`~tecio.AppendWrite`
        - ``'a+'`` → :class:`~tecio.AppendReadWrite`

    Raises:
        ValueError: Unsupported file extension, unrecognised mode, or ``'a'``/``'a+'``
            on a legacy PLT file (v75, v107, v111).
        FileNotFoundError: ``'r'``, ``'a'``, or ``'a+'`` on a missing file.
        FileExistsError: ``'x'`` on an existing file.

    Examples:
        Read a file:

        >>> with tecio.open("flow.szplt") as tec:
        ...     x = tec.zones[0].variables["x"].values

        Write a new file, deferring variable names to the first zone:

        >>> with tecio.open("out.szplt", "w", title="Run 1") as tec:
        ...     tec.write_ordered_zone(data=[x, y, p], variables=["x", "y", "p"])

        Append a new zone to an existing file:

        >>> with tecio.open("out.szplt", "a") as tec:
        ...     tec.write_ordered_zone(data=[x2, y2, p2], solution_time=2.0)

        Append and read in the same session:

        >>> with tecio.open("out.szplt", "a+") as tec:
        ...     prev_p = tec.zones[-1].variables["p"].values
        ...     tec.write_ordered_zone(data=[x2, y2, prev_p * 0.9])
    """
    ext = Path(path).suffix.lower()

    if ext not in _HANDLERS:
        raise ValueError(
            f"Unsupported file extension: '{ext}'. Supported: {sorted(_HANDLERS)}"
        )

    if mode == "r":
        ReadCls = _HANDLERS[ext]["r"]
        return ReadCls(str(path), **kwargs)

    elif mode == "w":
        WriteCls = _HANDLERS[ext]["w"]
        return WriteCls(str(path), **kwargs)

    elif mode == "x":
        return _open_exclusive(path, ext, **kwargs)

    elif mode == "a":
        return _open_append(path, ext, read_write=False, **kwargs)

    elif mode == "a+":
        return _open_append(path, ext, read_write=True, **kwargs)

    else:
        raise ValueError(
            f"Unrecognised mode '{mode}'. Supported modes: 'r', 'w', 'x', 'a', 'a+'"
        )


# --------------------------------------------------------------------------------------
# Lightweight metadata queries
# --------------------------------------------------------------------------------------


def _validated_ext(path: str | os.PathLike) -> str:
    """Return *path*'s lowercased extension, or raise if unsupported.

    Raises:
        ValueError: If the extension is not supported.

    Example:
        >>> _validated_ext("flow.szplt")
        '.szplt'
    """
    ext = Path(path).suffix.lower()
    if ext not in _HANDLERS:
        raise ValueError(
            f"Unsupported file extension: '{ext}'. Supported: {sorted(_HANDLERS)}"
        )
    return ext


class ZoneSummary(NamedTuple):
    """Per-zone metadata from a lightweight :func:`peek` query.

    ``num_nodes``/``num_elements`` are ``None`` only for a DAT ordered zone whose header
    omits I/J/K entirely (rare; relies on the reader inferring the point count from the
    data itself, which a lightweight query never does). Every other case, and every
    SZL/PLT zone, always has both.
    """

    title: str
    zone_type: ZoneType
    num_nodes: int | None
    num_elements: int | None


class FileSummary(NamedTuple):
    """Whole-file metadata from :func:`peek`, with no variable data touched."""

    title: str
    file_type: FileType
    variable_names: list[str]
    zones: list[ZoneSummary]


def get_variable_list(path: str | os.PathLike) -> list[str]:
    """Return a file's variable names without reading any variable data.

    For SZL and PLT this is already what a normal :func:`open` does, both formats
    resolve dataset-level metadata (including variable names) up front without touching
    data arrays, so this is a thin convenience wrapper for those two. DAT is different:
    opening it normally always reads and numerically converts every value in the file,
    so this function instead scans just the file's header text, several orders of
    magnitude cheaper for a file with large data blocks.

    Args:
        path: Path to a ``.szplt``, ``.plt``/``.bin``, or ``.dat``/``.tec`` file.

    Returns:
        Variable names, in file order.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is not recognized.

    Example:
        >>> get_variable_list("flow.szplt")
        ['x', 'y', 'z', 'pressure']
    """
    ext = _validated_ext(path)
    if ext in (".dat", ".tec"):
        return _peek_dat_metadata(path).variable_names
    with _HANDLERS[ext]["r"](str(path)) as r:
        return list(r.variables)


def get_zone_list(path: str | os.PathLike) -> list[str]:
    """Return a file's zone titles without reading any variable data.

    Same rationale as :func:`get_variable_list`: already cheap for SZL and PLT
    (constructing a zone reader only resolves scalar metadata, never variable data),
    so this is a thin wrapper for those; for DAT, uses a dedicated scan that skips
    every data and connectivity block entirely.

    Args:
        path: Path to a ``.szplt``, ``.plt``/``.bin``, or ``.dat``/``.tec`` file.

    Returns:
        Zone titles, in file order. An unnamed zone contributes an empty string, not
        a placeholder like ``"Zone 1"``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is not recognized.

    Example:
        >>> get_zone_list("flow.szplt")
        ['FluidVolume', 'WingSurface']
    """
    ext = _validated_ext(path)
    if ext in (".dat", ".tec"):
        return [z.title for z in _peek_dat_metadata(path).zones]
    with _HANDLERS[ext]["r"](str(path)) as r:
        return [zone.title for zone in r.zones]


def get_num_zones(path: str | os.PathLike) -> int:
    """Return a file's zone count without reading any variable data.

    Cheaper still than :func:`get_zone_list`: for SZL, a single scalar C-library
    call with no zone objects constructed at all; for PLT and DAT, a count against
    already-parsed/scanned header metadata.

    Args:
        path: Path to a ``.szplt``, ``.plt``/``.bin``, or ``.dat``/``.tec`` file.

    Returns:
        Number of zones in the file.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is not recognized.

    Example:
        >>> get_num_zones("flow.szplt")
        2
    """
    ext = _validated_ext(path)
    if ext in (".dat", ".tec"):
        return len(_peek_dat_metadata(path).zones)
    with _HANDLERS[ext]["r"](str(path)) as r:
        return r.num_zones


def get_num_variables(path: str | os.PathLike) -> int:
    """Return a file's variable count without reading any variable data.

    Cheaper still than :func:`get_variable_list`: for SZL, a single scalar
    C-library call.

    Args:
        path: Path to a ``.szplt``, ``.plt``/``.bin``, or ``.dat``/``.tec`` file.

    Returns:
        Number of variables in the dataset.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is not recognized.

    Example:
        >>> get_num_variables("flow.szplt")
        4
    """
    ext = _validated_ext(path)
    if ext in (".dat", ".tec"):
        return len(_peek_dat_metadata(path).variable_names)
    with _HANDLERS[ext]["r"](str(path)) as r:
        return r.num_vars


def get_title(path: str | os.PathLike) -> str:
    """Return a file's dataset title without reading any variable data.

    Args:
        path: Path to a ``.szplt``, ``.plt``/``.bin``, or ``.dat``/``.tec`` file.

    Returns:
        Dataset title string.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is not recognized.

    Example:
        >>> get_title("flow.szplt")
        'Onera M6 wing'
    """
    ext = _validated_ext(path)
    if ext in (".dat", ".tec"):
        return _peek_dat_metadata(path).title
    with _HANDLERS[ext]["r"](str(path)) as r:
        return r.title


def get_file_type(path: str | os.PathLike) -> FileType:
    """Return a file's :class:`~tecio.FileType` without reading any variable data.

    Args:
        path: Path to a ``.szplt``, ``.plt``/``.bin``, or ``.dat``/``.tec`` file.

    Returns:
        The file type (``FULL``, ``GRID``, or ``SOLUTION``).

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is not recognized.

    Example:
        >>> get_file_type("flow.szplt")
        <FileType.FULL: 0>
    """
    ext = _validated_ext(path)
    if ext in (".dat", ".tec"):
        return _peek_dat_metadata(path).file_type
    with _HANDLERS[ext]["r"](str(path)) as r:
        return r.file_type


def peek(path: str | os.PathLike) -> FileSummary:
    """Return a file's title, file type, variable names, and per-zone summaries.

    A single combined query covering everything :func:`get_variable_list`,
    :func:`get_zone_list`, :func:`get_title`, and :func:`get_file_type` return, plus
    each zone's type and size. For DAT, calling several of those functions
    separately on the same file means scanning it that many times over, this
    function scans once no matter how much of the summary is needed.

    Args:
        path: Path to a ``.szplt``, ``.plt``/``.bin``, or ``.dat``/``.tec`` file.

    Returns:
        Whole-file and per-zone metadata. See :class:`FileSummary` and
        :class:`ZoneSummary`.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file extension is not recognized.

    Example:
        >>> summary = peek("flow.szplt")
        >>> [z.title for z in summary.zones]
        ['FluidVolume', 'WingSurface']
    """
    ext = _validated_ext(path)
    if ext in (".dat", ".tec"):
        meta = _peek_dat_metadata(path)
        zones = [
            ZoneSummary(z.title, z.zone_type, z.num_nodes, z.num_elements)
            for z in meta.zones
        ]
        return FileSummary(meta.title, meta.file_type, meta.variable_names, zones)
    with _HANDLERS[ext]["r"](str(path)) as r:
        zones = [
            ZoneSummary(zone.title, zone.zone_type, zone.num_nodes, zone.num_elements)
            for zone in r.zones
        ]
        return FileSummary(r.title, r.file_type, list(r.variables), zones)
