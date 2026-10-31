"""Shared, format-independent base classes for Tecplot readers.

Every format's reader (SZL, PLT, DAT) exposes the same interface: :class:`TecplotReader`
(the open file/dataset), :class:`TecplotZoneReader` (one zone, split into
:class:`TecplotOrderedZoneReader`, :class:`TecplotFEZoneReader`, and
:class:`TecplotPolyZoneReader` since the topologies don't share properties like
dimensions or connectivity),
:class:`TecplotVariableReader` (one variable within a zone, not split by value location
or zone topology, neither changes which properties exist, only the shape
``get_values()`` returns, which the method already computes from
``zone_type``/``value_location``), and :class:`TecplotAuxDataReader` (a read-only
auxiliary-data mapping). This module defines that interface once so the three formats
can no longer drift apart, and so downstream code (the ParaView plugin, in particular)
can be written against ``TecplotZoneReader``/``TecplotVariableReader`` instead of
``Any``.

Design notes:
    * Instances are immutable: attribute assignment raises :exc:`AttributeError`.
      Internal caches for lazily loaded data (a zone's variable list, node map, and
      aux data) are populated through ``object.__setattr__``, bypassing the
      blocked ``__setattr__`` on purpose, the same trick :mod:`~tecio._meta`
      already uses via ``frozen=True`` dataclasses, applied by hand here because
      these classes also need lazy loading, which a plain frozen dataclass can't
      combine with ``__slots__``.
    * :class:`TecplotZoneReader` and its subclasses split fields into two groups.
      Small, cheap scalar metadata (title, zone type, dimensions or node/element
      counts, solution time, strand ID) is resolved once at construction and
      frozen, since even a per-value C call is cheap and doing it once beats
      re-querying on every access. Data that can be large (the variable list, an
      FE node map that may hold millions of entries) or that a caller may never
      touch (aux data) stays lazily loaded on first access, exactly matching each
      format's existing behaviour.
    * Unlike :mod:`~tecio._meta`, this module imports :mod:`~tecio._constants`
      at runtime rather than only under ``TYPE_CHECKING``. ``_meta`` only ever
      needs the enums as type annotations (erased at runtime by ``from __future__
      import annotations``); ``nodes_per_cell`` below needs the actual
      ``ZoneType`` members to key a lookup table, so a runtime import is
      unavoidable. ``_constants`` is a dependency-free leaf module (no C library,
      no other ``tecio`` submodule), so this introduces no cycle regardless.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from typing import Any, Final, Literal, TypeAlias, overload

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
from ._containers import VariableList, ZoneList, select_variable_arrays
from ._utils import public_dir

# Nodes per element for the standard finite-element zone types. ORDERED zones are
# computed from active dimensions instead, see TecplotZoneReader.nodes_per_cell.
_NODES_PER_ELEM: dict[ZoneType, int] = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FETETRAHEDRON: 4,
    ZoneType.FEBRICK: 8,
}


def reshape_face_connections(
    flat: npt.NDArray[np.int64],
    mode: FaceNeighborMode,
    num_connections: int,
) -> npt.NDArray[np.int64] | list[npt.NDArray[np.int64]]:
    """Reshape flat face-neighbor connection values per ``FaceNeighborMode``.

    Shared across formats, any reader whose C/binary layer returns face
    connections as a flat array can call this to get the structured form
    :attr:`TecplotFEZoneReader.face_connections` documents.

    ``LOCAL_ONE_TO_ONE`` (3 values/connection: ``cz1, fz, cz2``) and
    ``GLOBAL_ONE_TO_ONE`` (4 values/connection: ``cz, fz, ZZ, CZ``) are
    uniform, reshaped directly into a 2D array of shape
    ``(num_connections, 3 or 4)``.

    ``LOCAL_ONE_TO_MANY``/``GLOBAL_ONE_TO_MANY`` are ragged: each record
    starts with a fixed 4-value header (``cz, fz, oz, nz``), where ``nz``
    (the record's own 4th value) gives the count of additional neighbor
    references that follow, ``nz`` cell indices for local, ``2*nz``
    zone/cell index pairs for global. Since no single rectangular shape
    fits, each record is returned as its own 1D array, walking the flat
    buffer using each record's own ``nz`` to find where the next one
    starts. This ragged case is rare in practice for the classic FE element
    types this reader targets, "many" typically comes from polygon zones
    instead.

    Args:
        flat: The raw flat array as read from the file/library.
        mode: This zone's face-neighbor mode.
        num_connections: Number of connections, from a dedicated count
            (e.g. ``tecZoneFaceNbrGetNumConnections``), not inferred from
            *flat*'s length, which isn't reliable for the ragged modes.

    Returns:
        A ``(num_connections, 3 or 4)`` array for the one-to-one modes, or
        a list of ``num_connections`` 1D arrays for the one-to-many modes.
    """
    if mode is FaceNeighborMode.LOCAL_ONE_TO_ONE:
        return flat.reshape(num_connections, 3)
    if mode is FaceNeighborMode.GLOBAL_ONE_TO_ONE:
        return flat.reshape(num_connections, 4)

    is_global = mode is FaceNeighborMode.GLOBAL_ONE_TO_MANY
    connections: list[npt.NDArray[np.int64]] = []
    i = 0
    for _ in range(num_connections):
        nz = int(flat[i + 3])
        record_len = 4 + (2 * nz if is_global else nz)
        connections.append(flat[i : i + record_len])
        i += record_len
    return connections


def convert_face_connections_to_zero_based(
    flat: npt.NDArray[np.int64], mode: FaceNeighborMode
) -> npt.NDArray[np.int64]:
    """Convert a flat, fully 1-based face-neighbor array to the public convention.

    Shared by any reader whose source is inherently 1-based throughout (SZL's live C
    calls, DAT's ASCII text), unlike PLT's binary layout which needs its own,
    different conversion.

    For the one-to-one modes, every value in a record (cell, face ordinal, and for
    the global mode, remote zone and remote cell) is an index, so the whole array
    shifts uniformly. The "many" modes are ragged and mix in two non-index fields
    (``oz``, the face obscuration flag, and ``nz``, the trailing-reference count)
    that must stay untouched, so each record is converted individually instead.

    Args:
        flat: The raw flat array, exactly as returned by the source, fully 1-based.
        mode: This zone's face-neighbor mode.

    Returns:
        A new flat array with cell/face/remote-zone fields converted, safe to
        reshape with :func:`reshape_face_connections`.
    """
    one_to_one = (FaceNeighborMode.LOCAL_ONE_TO_ONE, FaceNeighborMode.GLOBAL_ONE_TO_ONE)
    if mode in one_to_one:
        return flat - 1

    flat = flat.copy()
    is_global = mode is FaceNeighborMode.GLOBAL_ONE_TO_MANY
    i = 0
    while i < flat.size:
        nz = int(flat[i + 3])
        rest_count = 2 * nz if is_global else nz
        flat[i] -= 1  # local cell number
        flat[i + 1] -= 1  # face ordinal
        # flat[i + 2] (oz, the obscuration flag) and nz itself are not indices.
        flat[i + 4 : i + 4 + rest_count] -= 1  # trailing cell/zone references
        i += 4 + rest_count
    return flat


def _immutable_setattr(instance: object, name: str, value: object) -> None:
    """Raise, used as every reader class's ``__setattr__``."""
    raise AttributeError(f"{type(instance).__name__} instances are immutable")


def _immutable_delattr(instance: object, name: str) -> None:
    """Raise, used as every reader class's ``__delattr__``."""
    raise AttributeError(f"{type(instance).__name__} instances are immutable")


# ======================================================================================
# TecplotAuxDataReader
# ======================================================================================


class TecplotAuxDataReader(Mapping[str, str], ABC):
    """Read-only view of one Tecplot auxiliary-data mapping.

    Backs dataset-, zone-, and variable-level auxiliary data for all three formats.
    Subclasses supply the underlying ``{name: value}`` mapping through
    :meth:`_load_data`, called at most once, on first access, and cached for the
    life of the object.

    Implements :class:`collections.abc.Mapping`, so ``get``, ``keys``, ``values``,
    ``items``, ``__contains__``, and ``in`` all work without being redefined here.

    Note:
        A concrete subclass must call ``super().__init__()`` first, then store
        any private fields it needs (the data source to lazily load from) with
        ``object.__setattr__``, the same contract as :class:`TecplotZoneReader`
        and :class:`TecplotVariableReader`.
    """

    __slots__ = ("_cache",)
    _cache: dict[str, str] | None

    def __init__(self) -> None:
        object.__setattr__(self, "_cache", None)

    __setattr__ = _immutable_setattr
    __delattr__ = _immutable_delattr

    @abstractmethod
    def _load_data(self) -> dict[str, str]:
        """Return the underlying mapping. Called at most once, lazily."""

    @property
    def _data(self) -> dict[str, str]:
        data = self._cache
        if data is None:
            data = self._load_data()
            object.__setattr__(self, "_cache", data)
        return data

    def __getitem__(self, key: str) -> str:
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}({self._data!r})"

    # Mapping already provides get/keys/values/items via __getitem__ + __iter__ +
    # __len__, but typed overrides are kept here since the base implementations
    # return generic view types rather than the str-specific ones callers expect.

    def keys(self) -> KeysView[str]:
        """Return the auxiliary data names."""
        return self._data.keys()

    def values(self) -> ValuesView[str]:
        """Return the auxiliary data values."""
        return self._data.values()

    def items(self) -> ItemsView[str, str]:
        """Return the ``(name, value)`` pairs."""
        return self._data.items()

    def as_int(self, key: str, default: int | None = None) -> int | None:
        """Return the value for *key* as :class:`int`, or *default* on failure."""
        try:
            return int(self._data[key])
        except (KeyError, ValueError):
            return default

    def as_float(self, key: str, default: float | None = None) -> float | None:
        """Return the value for *key* as :class:`float`, or *default* on failure."""
        try:
            return float(self._data[key])
        except (KeyError, ValueError):
            return default

    def as_bool(self, key: str, default: bool | None = None) -> bool | None:
        """Return the value for *key* as :class:`bool`, or *default* on failure.

        Recognizes ``true``/``t``/``yes``/``y``/``1`` as True and
        ``false``/``f``/``no``/``n``/``0`` as False, case-insensitively.
        """
        try:
            value = self._data[key].strip().lower()
        except KeyError:
            return default
        if value in ("true", "t", "yes", "y", "1"):
            return True
        if value in ("false", "f", "no", "n", "0"):
            return False
        return default


# ======================================================================================
# TecplotVariableReader
# ======================================================================================


class TecplotVariableReader(ABC):
    """Read-only handle to one variable's metadata and data within a zone.

    Every metadata field is resolved by the concrete subclass on each access (a
    single libtecio call, or a lookup in already-parsed metadata), cheap enough
    in every format that this base does not add its own caching layer on top.
    Only :attr:`values` reads actual array data, and only when called.

    Note:
        ``__setattr__`` is blocked here, which also applies inside a subclass's
        own ``__init__``. A concrete subclass must store its private fields with
        ``object.__setattr__(self, "_name", value)`` rather than plain
        assignment, and should declare its own ``__slots__`` for them.
    """

    __slots__ = ()

    __setattr__ = _immutable_setattr
    __delattr__ = _immutable_delattr

    def __repr__(self) -> str:  # pragma: no cover
        parts = [repr(self.name)]
        if self.is_passive():
            parts.append("passive")
        elif self.shared_zone is not None:
            parts.append(f"shared(zone={self.shared_zone})")
        else:
            parts.append(f"dtype={self.data_type.name}")
            values = self.values
            if values is not None:
                parts.append(f"shape={values.shape}")
            if self.value_location == ValueLocation.CELL_CENTERED:
                parts.append("CELL_CENTERED")
        return f"{type(self).__name__}({', '.join(parts)})"

    # -- Abstract: one C call / metadata lookup each, format-specific ------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Variable name string."""

    @property
    @abstractmethod
    def data_type(self) -> DataType:
        """Stored data type."""

    @property
    @abstractmethod
    def value_location(self) -> ValueLocation:
        """NODAL or CELL_CENTERED."""

    @abstractmethod
    def is_passive(self) -> bool:
        """True if this variable has no data in this zone."""

    @property
    @abstractmethod
    def shared_zone(self) -> int | None:
        """0-based source zone index if shared, else None."""

    @property
    @abstractmethod
    def num_values(self) -> int:
        """Number of stored values, 0 for a passive variable."""

    @abstractmethod
    def get_values(
        self,
        value_range: tuple[int | None, int | None] = (None, None),
    ) -> npt.NDArray[Any] | None:
        """Read values from disk or memory.

        Args:
            value_range: 1-based ``(start, end)``, half-open. ``(None, None)``
                reads all values.

        Returns:
            The data array, or ``None`` if the variable is passive. A shared
            variable resolves to its source zone's array.

        Raises:
            ValueError: If only one of ``start``/``end`` is given.
        """

    # -- Shared ------------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """True unless the variable is passive.

        SZL overrides this with the dataset-level enabled flag it actually
        stores; PLT and DAT have no such flag and use this default.
        """
        return not self.is_passive()

    @property
    def values(self) -> npt.NDArray[Any] | None:
        """All values as a NumPy array, or None if passive."""
        return self.get_values()


# ======================================================================================
# TecplotZoneReader
# ======================================================================================


class TecplotZoneReader(ABC):
    """Read-only handle to one zone's metadata in an open Tecplot file.

    This is the common base for :class:`TecplotOrderedZoneReader` and
    :class:`TecplotFEZoneReader`. Only properties that mean the same thing for
    every zone topology live here, dimensions, node maps, and everything else
    that's specific to one topology live on the corresponding subclass instead,
    so accessing e.g. ``.node_map`` on an ordered zone is a plain
    :exc:`AttributeError` rather than a silent ``None``.

    Scalar metadata (title, zone type, solution time, strand ID, datapacking)
    is resolved once at construction and frozen. The variable list and aux
    data stay lazily loaded on first access, since either can be large enough
    that eagerly building them for every zone in a file would be wasteful.

    Args:
        zone_index: 0-based zone index within the dataset.
        title: Zone title.
        zone_type: The zone's :class:`~tecio.libtecio.ZoneType`.
        solution_time: Solution time, 0.0 for static data.
        strand_id: Strand ID, 0 for static data.
        datapacking: On-disk value layout. SZL and every PLT version from
            v107 onward always supply :attr:`~tecio.libtecio.DataPacking.BLOCK`,
            those formats have no packing distinction on disk; the field
            exists purely so code can treat every zone uniformly regardless
            of source format. The legacy PLT v75 format is the one exception:
            it genuinely stores some zones interleaved by node/element
            (POINT/FEPOINT), reported here as
            :attr:`~tecio.libtecio.DataPacking.POINT`.
    """

    __slots__ = (
        "_zone_index",
        "_title",
        "_zone_type",
        "_solution_time",
        "_strand_id",
        "_datapacking",
        "_variable_cache",
        "_auxdata_cache",
    )
    _zone_index: int
    _title: str
    _zone_type: ZoneType
    _solution_time: float
    _strand_id: int
    _datapacking: DataPacking
    _variable_cache: VariableList[TecplotVariableReader] | None
    _auxdata_cache: TecplotAuxDataReader | None

    def __init__(
        self,
        zone_index: int,
        title: str,
        zone_type: ZoneType,
        solution_time: float,
        strand_id: int,
        datapacking: DataPacking,
    ) -> None:
        object.__setattr__(self, "_zone_index", zone_index)
        object.__setattr__(self, "_title", title)
        object.__setattr__(self, "_zone_type", zone_type)
        object.__setattr__(self, "_solution_time", solution_time)
        object.__setattr__(self, "_strand_id", strand_id)
        object.__setattr__(self, "_datapacking", datapacking)
        object.__setattr__(self, "_variable_cache", None)
        object.__setattr__(self, "_auxdata_cache", None)

    __setattr__ = _immutable_setattr
    __delattr__ = _immutable_delattr

    def __dir__(self) -> list[str]:  # pragma: no cover
        """List only the public attributes and methods."""
        return public_dir(self)

    def __repr__(self) -> str:  # pragma: no cover
        title = self.title
        if len(title) > 30:
            title = title[:29] + "\u2026"
        extra = f", aux={len(self.auxdata)}" if len(self.auxdata) else ""
        cls = type(self).__name__
        return f"{cls}({title!r}, {self.zone_type.name}, {self._repr_size()}{extra})"

    @abstractmethod
    def _repr_size(self) -> str:
        """Return the shape portion of __repr__, e.g. ``'I=4, J=3, K=1'``."""

    # -- Frozen scalar metadata --------------------------------------------------------

    @property
    def zone_index(self) -> int:
        """0-based zone index within the dataset."""
        return self._zone_index

    @property
    def title(self) -> str:
        """Zone title string."""
        return self._title

    @property
    def zone_type(self) -> ZoneType:
        """The zone's :class:`~tecio.libtecio.ZoneType`."""
        return self._zone_type

    @property
    def solution_time(self) -> float:
        """Solution time, 0.0 for static data."""
        return self._solution_time

    @property
    def strand_id(self) -> int:
        """Strand ID, 0 for static data."""
        return self._strand_id

    @property
    def datapacking(self) -> DataPacking:
        """On-disk value layout for this zone."""
        return self._datapacking

    def is_enabled(self) -> bool:
        """True unless the zone reports itself disabled.

        Only SZL can report False here; PLT and DAT have no such flag.
        """
        return True

    # -- Lazy: variable list, aux data ------------------------------------------------

    @property
    def variables(self) -> VariableList[TecplotVariableReader]:
        """Variables in this zone, by 0-based index or exact name."""
        variables = self._variable_cache
        if variables is None:
            variables = VariableList(self._load_variables())
            object.__setattr__(self, "_variable_cache", variables)
        return variables

    @abstractmethod
    def _load_variables(self) -> list[TecplotVariableReader]:
        """Construct this zone's variable readers. Called at most once, lazily."""

    @property
    def auxdata(self) -> TecplotAuxDataReader:
        """Zone-level auxiliary data."""
        aux = self._auxdata_cache
        if aux is None:
            aux = self._load_auxdata()
            object.__setattr__(self, "_auxdata_cache", aux)
        return aux

    @abstractmethod
    def _load_auxdata(self) -> TecplotAuxDataReader:
        """Construct this zone's aux-data reader. Called at most once, lazily."""

    # -- Fully generic -----------------------------------------------------------------

    @overload
    def get_array(self, key: int | str) -> npt.NDArray[Any] | None: ...
    @overload
    def get_array(self, key: list[str]) -> tuple[npt.NDArray[Any] | None, ...]: ...

    def get_array(
        self, key: int | str | list[str]
    ) -> npt.NDArray[Any] | None | tuple[npt.NDArray[Any] | None, ...]:
        """Return variable data array(s) for this zone.

        A single key (0-based index or exact name) returns one array. A list of
        exact names returns a tuple of arrays in the order given::

            p = zone.get_array("p")
            x, y, z = zone.get_array(["x", "y", "z"])

        Returns:
            One array (or ``None`` only if the variable is passive) for a scalar
            key; a tuple of such arrays for a list of names.

        Raises:
            KeyError: If a name does not exist.
            IndexError: If an index is out of range.
        """
        return select_variable_arrays(self.variables, key)


# ======================================================================================
# TecplotOrderedZoneReader
# ======================================================================================


class TecplotOrderedZoneReader(TecplotZoneReader):
    """Read-only handle to one ORDERED (IJK-structured) zone.

    Args:
        zone_index: 0-based zone index within the dataset.
        title: Zone title.
        solution_time: Solution time, 0.0 for static data.
        strand_id: Strand ID, 0 for static data.
        datapacking: On-disk value layout.
        i: Nodal I dimension.
        j: Nodal J dimension.
        k: Nodal K dimension.
    """

    __slots__ = ("_i", "_j", "_k")
    _i: int
    _j: int
    _k: int

    def __init__(
        self,
        zone_index: int,
        title: str,
        solution_time: float,
        strand_id: int,
        datapacking: DataPacking,
        i: int,
        j: int,
        k: int,
    ) -> None:
        super().__init__(
            zone_index=zone_index,
            title=title,
            zone_type=ZoneType.ORDERED,
            solution_time=solution_time,
            strand_id=strand_id,
            datapacking=datapacking,
        )
        object.__setattr__(self, "_i", i)
        object.__setattr__(self, "_j", j)
        object.__setattr__(self, "_k", k)

    def _repr_size(self) -> str:
        return f"I={self._i}, J={self._j}, K={self._k}"

    @property
    def I(self) -> int:  # noqa: E743 N802
        """Nodal I dimension."""
        return self._i

    @property
    def J(self) -> int:  # noqa: N802
        """Nodal J dimension."""
        return self._j

    @property
    def K(self) -> int:  # noqa: N802
        """Nodal K dimension."""
        return self._k

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """``(I, J, K)`` dimensions."""
        return (self._i, self._j, self._k)

    @property
    def num_nodes(self) -> int:
        """Number of nodes/points, ``I * J * K``."""
        return self._i * self._j * self._k

    @property
    def num_elements(self) -> int:
        """Number of cells."""
        # Normalize to proper dimension
        if self._k == 1 and self._j == 1:
            # 1D case
            return self._i - 1
        elif self._k == 1 and self._j > 1:
            # 2D case
            return (self._i - 1) * (self._j - 1)
        else:
            # 3D case
            return (self._i - 1) * (self._j - 1) * (self._k - 1)


# ======================================================================================
# TecplotFEZoneReader
# ======================================================================================


class TecplotFEZoneReader(TecplotZoneReader):
    """Read-only handle to one finite-element zone.

    Covers the classic element zone types (FELINESEG through FEBRICK). Face-based
    FEPOLYGON and FEPOLYHEDRON zones are read by :class:`TecplotPolyZoneReader`.

    Args:
        zone_index: 0-based zone index within the dataset.
        title: Zone title.
        zone_type: The zone's :class:`~tecio.libtecio.ZoneType`.
        solution_time: Solution time, 0.0 for static data.
        strand_id: Strand ID, 0 for static data.
        datapacking: On-disk value layout.
        num_nodes: Number of nodes/points.
        num_elements: Number of elements/cells.
        shared_connectivity: 0-based source zone index if this zone's
            connectivity is shared, else None.
    """

    __slots__ = (
        "_num_nodes",
        "_num_elements",
        "_shared_connectivity",
        "_node_map_cache",
        "_node_map_loaded",
        "_face_neighbor_meta_cache",
        "_face_neighbor_meta_loaded",
        "_face_connections_cache",
        "_face_connections_loaded",
    )
    _num_nodes: int
    _num_elements: int
    _shared_connectivity: int | None
    _node_map_cache: npt.NDArray[np.int64] | None
    _node_map_loaded: bool
    _face_neighbor_meta_cache: tuple[FaceNeighborMode, int, bool | None] | None
    _face_neighbor_meta_loaded: bool
    _face_connections_cache: npt.NDArray[np.int64] | None
    _face_connections_loaded: bool

    def __init__(
        self,
        zone_index: int,
        title: str,
        zone_type: ZoneType,
        solution_time: float,
        strand_id: int,
        datapacking: DataPacking,
        num_nodes: int,
        num_elements: int,
        shared_connectivity: int | None,
    ) -> None:
        super().__init__(
            zone_index=zone_index,
            title=title,
            zone_type=zone_type,
            solution_time=solution_time,
            strand_id=strand_id,
            datapacking=datapacking,
        )
        object.__setattr__(self, "_num_nodes", num_nodes)
        object.__setattr__(self, "_num_elements", num_elements)
        object.__setattr__(self, "_shared_connectivity", shared_connectivity)
        object.__setattr__(self, "_node_map_cache", None)
        object.__setattr__(self, "_node_map_loaded", False)
        object.__setattr__(self, "_face_neighbor_meta_cache", None)
        object.__setattr__(self, "_face_neighbor_meta_loaded", False)
        object.__setattr__(self, "_face_connections_cache", None)
        object.__setattr__(self, "_face_connections_loaded", False)

    def _repr_size(self) -> str:
        return f"N={self._num_nodes}, E={self._num_elements}"

    @property
    def num_nodes(self) -> int:
        """Number of nodes/points."""
        return self._num_nodes

    @property
    def num_elements(self) -> int:
        """Number of elements/cells."""
        return self._num_elements

    @property
    def shared_connectivity(self) -> int | None:
        """0-based source zone index if connectivity is shared, else None."""
        return self._shared_connectivity

    @property
    def nodes_per_cell(self) -> int:
        """Nodes per cell, fixed by zone type.

        Raises:
            ValueError: For a zone type without a fixed nodes-per-cell count (not
                expected here, :class:`TecplotFEZoneReader` is only ever constructed for
                the classic FE types, but kept as a defensive check rather than an
                unchecked KeyError).
        """
        zt = self.zone_type
        if zt in _NODES_PER_ELEM:
            return _NODES_PER_ELEM[zt]
        raise ValueError(  # pragma: no cover
            f"ZoneType {zt} does not have a fixed nodes-per-cell count."
        )

    @property
    def node_map(self) -> npt.NDArray[np.int64] | None:
        """Node connectivity array ``(num_elements, nodes_per_cell)``.

        None only if a format cannot resolve connectivity for this zone, e.g. when a
        shared-connectivity source zone is missing.
        """
        if not self._node_map_loaded:
            node_map = self._load_node_map()
            object.__setattr__(self, "_node_map_cache", node_map)
            object.__setattr__(self, "_node_map_loaded", True)
            return node_map
        return self._node_map_cache

    @abstractmethod
    def _load_node_map(self) -> npt.NDArray[np.int64] | None:
        """Read this zone's node connectivity. Called at most once, lazily."""

    def _resolve_face_neighbor_meta(
        self,
    ) -> tuple[FaceNeighborMode, int, bool | None] | None:
        if not self._face_neighbor_meta_loaded:
            result = self._load_face_neighbor_meta()
            object.__setattr__(self, "_face_neighbor_meta_cache", result)
            object.__setattr__(self, "_face_neighbor_meta_loaded", True)
            return result
        return self._face_neighbor_meta_cache

    @property
    def face_neighbor_mode(self) -> FaceNeighborMode | None:
        """This zone's face-neighbor connection mode.

        None if the zone has no face-neighbor connections, or if this format
        doesn't (yet) support reading them, unstructured connectivity alone
        already implies node adjacency; face neighbors only exist to state
        adjacency explicitly for non-conformal or multi-zone-connected
        meshes, which most FE zones don't need. Cheap: reading this never
        loads the connections themselves, only the small metadata it's
        paired with, so it's safe to check on every zone without cost.
        """
        result = self._resolve_face_neighbor_meta()
        return result[0] if result is not None else None

    @property
    def num_face_connections(self) -> int | None:
        """Number of face-neighbor connections in this zone, else None.

        Not derivable from :meth:`get_face_connections`'s result without
        loading it, and for the "many" modes, not derivable from its flat
        length at all, a variable number of neighbor cells per connection
        means the array's length isn't a fixed multiple of this count.
        Cheap either way: this never loads the connections themselves.
        """
        result = self._resolve_face_neighbor_meta()
        return result[1] if result is not None else None

    @property
    def face_neighbors_complete(self) -> bool | None:
        """Whether this zone's explicit face-neighbor data is exhaustive.

        Tecplot can auto-detect ordinary conformal adjacency from the node
        map alone; explicit face-neighbor data exists for what auto-detection
        can't find (non-conformal interfaces, cross-zone adjacency). ``True``
        means the explicit data is the entire adjacency picture, nothing
        else to auto-detect. ``False`` means it's a supplement, Tecplot
        still auto-detects the rest. ``None`` means either this zone has no
        face-neighbor data at all (see :attr:`face_neighbor_mode`), or this
        format has no way to represent the distinction: only DAT's ASCII
        format actually carries this flag (``FEFACENEIGHBORSCOMPLETE``);
        neither the classic nor new C APIs have an equivalent, so PLT and
        SZL always report ``None`` here even when face-neighbor data exists.
        """
        result = self._resolve_face_neighbor_meta()
        return result[2] if result is not None else None

    def _load_face_neighbor_meta(
        self,
    ) -> tuple[FaceNeighborMode, int, bool | None] | None:
        """Read this zone's face-neighbor mode, count, and completeness.

        Called at most once, lazily, and never triggers loading the
        connections themselves. Default: None (not implemented for this
        format). A format that overrides this to return real data must
        also override :meth:`_load_face_connections`.
        """
        return None

    def get_face_connections(
        self, reshape: bool = False
    ) -> npt.NDArray[np.int64] | None:
        """This zone's face-neighbor connections, else None.

        Loaded and cached on first access (flat form), independently of
        :attr:`face_neighbor_mode`/:attr:`num_face_connections`, reading
        those never triggers this.

        Args:
            reshape: If True, and :attr:`face_neighbor_mode` is
                ``LOCAL_ONE_TO_ONE`` or ``GLOBAL_ONE_TO_ONE``, reshape into
                a 2D array of shape ``(num_face_connections, 3 or 4)``
                instead of the raw flat array. Raises for the "many" modes,
                a variable number of neighbor cells per connection means
                there's no single rectangular shape to reshape into.

        Returns:
            Flat array (dtype int64) if *reshape* is False (the default),
            or a 2D array if *reshape* is True and the mode supports it.
            None if the zone has no face-neighbor connections.

        Raises:
            ValueError: If *reshape* is True and :attr:`face_neighbor_mode`
                is ``LOCAL_ONE_TO_MANY`` or ``GLOBAL_ONE_TO_MANY``. Call
                without *reshape*, or use :func:`reshape_face_connections`
                directly, which handles the "many" modes too, returning a
                list of per-connection arrays instead of raising.
        """
        if self.num_face_connections is None:
            return None
        if not self._face_connections_loaded:
            connections = self._load_face_connections()
            object.__setattr__(self, "_face_connections_cache", connections)
            object.__setattr__(self, "_face_connections_loaded", True)
        else:
            connections = self._face_connections_cache
        assert connections is not None  # narrowed: num_face_connections was not None
        if not reshape:
            return connections

        mode = self.face_neighbor_mode
        assert mode is not None  # narrowed: num_face_connections was not None above
        if mode not in (
            FaceNeighborMode.LOCAL_ONE_TO_ONE,
            FaceNeighborMode.GLOBAL_ONE_TO_ONE,
        ):
            raise ValueError(
                "get_face_connections(reshape=True) requires a one-to-one "
                f"FaceNeighborMode, this zone's mode is {mode.name}, which "
                "is ragged (a variable number of neighbor cells per "
                "connection) and has no single rectangular shape. Call "
                "without reshape to get the flat array, or use "
                "reshape_face_connections() directly, which returns a "
                "list of per-connection arrays for the many modes instead "
                "of raising."
            )
        result = reshape_face_connections(connections, mode, self.num_face_connections)
        assert isinstance(result, np.ndarray)  # narrowed: one-to-one modes above
        return result

    def _load_face_connections(self) -> npt.NDArray[np.int64]:
        """Read this zone's flat face-neighbor connection values.

        Called at most once, lazily, and only when
        :attr:`num_face_connections` is already known to be non-None, so a
        format overriding :meth:`_load_face_neighbor_meta` to return real
        data must also override this; the default raises to make a missing
        override obvious rather than silently returning nothing.
        """
        raise NotImplementedError(
            f"{type(self).__name__} reports face-neighbor metadata but "
            "does not implement _load_face_connections."
        )


# ======================================================================================
# TecplotPolyZoneReader
# ======================================================================================

PolyArrayKey: TypeAlias = Literal[
    "face_node_counts",
    "face_nodes",
    "face_left_elements",
    "face_right_elements",
    "boundary_connection_counts",
    "boundary_connection_elements",
    "boundary_connection_zones",
]

# Reference value for "no element" in ``face_left_elements`` and
# ``face_right_elements``. Values at or below ``BOUNDARY_REF_BASE`` reference boundary
# face ``BOUNDARY_REF_BASE - value``.
NO_ELEMENT: Final = -1
BOUNDARY_REF_BASE: Final = -2


def counts_to_offsets(counts: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Return the ``len(counts) + 1`` running offsets for per-item *counts*."""
    offsets = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets


class TecplotPolyZoneReader(TecplotZoneReader):
    """Read-only handle to one FEPOLYGON or FEPOLYHEDRON (face-based) zone.

    A poly zone stores faces instead of per-element node lists. Every face lists its
    nodes and the elements on its left and right. All indices are 0-based. The face
    arrays are flat and are returned as stored in the file, never split per element.

    ``face_left_elements`` and ``face_right_elements`` hold, for each face:

    * ``>= 0``: the 0-based element index.
    * ``-1`` (:data:`NO_ELEMENT`): no element on that side (a domain boundary).
    * ``<= -2``: a boundary face, an entry in the boundary connection arrays, used
      for neighbors in other zones or faces with several neighbors. The boundary
      face index is ``-2 - value`` (:data:`BOUNDARY_REF_BASE`).

    Boundary face ``b`` owns ``boundary_connection_counts[b]`` consecutive entries of
    ``boundary_connection_elements`` and ``boundary_connection_zones``, which hold the
    0-based neighbor element and the 0-based zone containing it.

    Face-neighbor connections (:class:`~tecio.FaceNeighborMode`) do not apply to poly
    zones, the left and right elements play that role.

    For a FEPOLYGON zone each face is an edge with exactly 2 nodes. Its
    ``face_node_counts`` is synthesized (all 2) and ``face_nodes`` has
    ``2 * num_faces`` entries.

    Args:
        zone_index: 0-based zone index within the dataset.
        title: Zone title.
        zone_type: FEPOLYGON or FEPOLYHEDRON.
        solution_time: Solution time, 0.0 for static data.
        strand_id: Strand ID, 0 for static data.
        datapacking: On-disk value layout.
        num_nodes: Number of nodes/points.
        num_elements: Number of elements/cells.
        num_faces: Number of faces.
        num_boundary_faces: Number of boundary faces.
        num_boundary_connections: Total number of boundary connections.
        shared_connectivity: 0-based source zone index if this zone's connectivity
            is shared, else None.
    """

    __slots__ = (
        "_num_nodes",
        "_num_elements",
        "_num_faces",
        "_num_boundary_faces",
        "_num_boundary_connections",
        "_shared_connectivity",
        "_array_cache",
    )
    _num_nodes: int
    _num_elements: int
    _num_faces: int
    _num_boundary_faces: int
    _num_boundary_connections: int
    _shared_connectivity: int | None
    _array_cache: dict[str, npt.NDArray[np.int64]]

    def __init__(
        self,
        zone_index: int,
        title: str,
        zone_type: ZoneType,
        solution_time: float,
        strand_id: int,
        datapacking: DataPacking,
        num_nodes: int,
        num_elements: int,
        num_faces: int,
        num_boundary_faces: int,
        num_boundary_connections: int,
        shared_connectivity: int | None,
    ) -> None:
        if zone_type not in (ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON):
            raise ValueError(f"Not a poly zone type: {zone_type!r}")
        super().__init__(
            zone_index=zone_index,
            title=title,
            zone_type=zone_type,
            solution_time=solution_time,
            strand_id=strand_id,
            datapacking=datapacking,
        )
        object.__setattr__(self, "_num_nodes", num_nodes)
        object.__setattr__(self, "_num_elements", num_elements)
        object.__setattr__(self, "_num_faces", num_faces)
        object.__setattr__(self, "_num_boundary_faces", num_boundary_faces)
        object.__setattr__(self, "_num_boundary_connections", num_boundary_connections)
        object.__setattr__(self, "_shared_connectivity", shared_connectivity)
        object.__setattr__(self, "_array_cache", {})

    def _repr_size(self) -> str:
        return f"N={self._num_nodes}, F={self._num_faces}, E={self._num_elements}"

    @property
    def num_nodes(self) -> int:
        """Number of nodes/points."""
        return self._num_nodes

    @property
    def num_elements(self) -> int:
        """Number of elements/cells."""
        return self._num_elements

    @property
    def num_faces(self) -> int:
        """Number of faces."""
        return self._num_faces

    @property
    def num_boundary_faces(self) -> int:
        """Number of boundary faces."""
        return self._num_boundary_faces

    @property
    def num_boundary_connections(self) -> int:
        """Total number of boundary connections over all boundary faces."""
        return self._num_boundary_connections

    @property
    def shared_connectivity(self) -> int | None:
        """0-based source zone index if connectivity is shared, else None."""
        return self._shared_connectivity

    def _face_map_digest(self) -> str:
        """Return a hash identifying this zone's grid, for matching grids across zones.

        The face map is read without filling the array cache, so scanning many zones
        does not keep their arrays in memory.
        """
        h = hashlib.blake2b(digest_size=16)
        h.update(
            f"{self.zone_type.name}:{self._num_nodes}:{self._num_elements}:"
            f"{self._num_faces}".encode()
        )
        keys: tuple[PolyArrayKey, ...] = (
            "face_nodes",
            "face_left_elements",
            "face_right_elements",
        )
        if self.zone_type is ZoneType.FEPOLYHEDRON:
            keys = ("face_node_counts", *keys)
        for key in keys:
            arr = np.ascontiguousarray(self._load_array(key))
            h.update(arr.data)
        return h.hexdigest()

    # -- Lazy face-map arrays ----------------------------------------------------------

    def _array(self, key: PolyArrayKey) -> npt.NDArray[np.int64]:
        """Return the cached array for *key*, loading it on first use."""
        cache = self._array_cache
        arr = cache.get(key)
        if arr is None:
            arr = self._load_array(key)
            cache[key] = arr
        return arr

    @abstractmethod
    def _load_array(self, key: PolyArrayKey) -> npt.NDArray[np.int64]:
        """Read one face-map array, 0-based, as int64. Called at most once per key.

        Never called for ``face_node_counts`` of a FEPOLYGON zone, which is
        synthesized by the base class.
        """

    @property
    def face_node_counts(self) -> npt.NDArray[np.int64]:
        """Number of nodes of each face, shape ``(num_faces,)``."""
        if self.zone_type is ZoneType.FEPOLYGON:
            cache = self._array_cache
            arr = cache.get("face_node_counts")
            if arr is None:
                arr = np.full(self._num_faces, 2, dtype=np.int64)
                cache["face_node_counts"] = arr
            return arr
        return self._array("face_node_counts")

    @property
    def face_node_offsets(self) -> npt.NDArray[np.int64]:
        """Running offsets into :attr:`face_nodes`, shape ``(num_faces + 1,)``.

        The nodes of face ``f`` are ``face_nodes[offsets[f]:offsets[f + 1]]``.
        """
        arr = self._array_cache.get("face_node_offsets")
        if arr is None:
            arr = counts_to_offsets(self.face_node_counts)
            self._array_cache["face_node_offsets"] = arr
        return arr

    @property
    def face_nodes(self) -> npt.NDArray[np.int64]:
        """Flat 0-based node indices of all faces, in face order."""
        return self._array("face_nodes")

    @property
    def face_left_elements(self) -> npt.NDArray[np.int64]:
        """Element on the left of each face, shape ``(num_faces,)``, see class doc."""
        return self._array("face_left_elements")

    @property
    def face_right_elements(self) -> npt.NDArray[np.int64]:
        """Element on the right of each face, shape ``(num_faces,)``, see class doc."""
        return self._array("face_right_elements")

    @property
    def boundary_connection_counts(self) -> npt.NDArray[np.int64]:
        """Connections of each boundary face, shape ``(num_boundary_faces,)``."""
        if self._num_boundary_faces == 0:
            return np.zeros(0, dtype=np.int64)
        return self._array("boundary_connection_counts")

    @property
    def boundary_connection_offsets(self) -> npt.NDArray[np.int64]:
        """Running offsets into the boundary connection arrays.

        Shape ``(num_boundary_faces + 1,)``.
        """
        arr = self._array_cache.get("boundary_connection_offsets")
        if arr is None:
            arr = counts_to_offsets(self.boundary_connection_counts)
            self._array_cache["boundary_connection_offsets"] = arr
        return arr

    @property
    def boundary_connection_elements(self) -> npt.NDArray[np.int64]:
        """0-based neighbor element of every boundary connection, flat."""
        if self._num_boundary_connections == 0:
            return np.zeros(0, dtype=np.int64)
        return self._array("boundary_connection_elements")

    @property
    def boundary_connection_zones(self) -> npt.NDArray[np.int64]:
        """0-based zone of the neighbor of every boundary connection, flat."""
        if self._num_boundary_connections == 0:
            return np.zeros(0, dtype=np.int64)
        return self._array("boundary_connection_zones")


# ======================================================================================
# TecplotReader
# ======================================================================================


class TecplotReader(ABC):
    """Shared interface for all Tecplot file readers (SZL, PLT, DAT).

    Concrete subclasses (:class:`TecplotSzlReader`, ...) differ in how a file is opened
    and parsed, SZL keeps a live C file handle and queries it on demand, PLT and DAT
    parse eagerly and hold no handle, but expose an identical interface once open, so
    code (including the ParaView plugin) can be written against this base without caring
    which format produced a given file.
    """

    __slots__ = ()

    @abstractmethod
    def __init__(self, path: str) -> None:
        """Open *path* for reading.

        Declared here so that ``ReaderCls(path)`` type-checks against
        ``type[TecplotReader]``, e.g. in :func:`~tecio._io.open`'s dispatch table,
        without needing a cast. Each format's actual constructor may accept a broader
        path type (PLT also takes ``os.PathLike``); ``str`` is the narrowest common
        signature, matching how every call site in this package already invokes it.
        """

    # -- Abstract: resolved differently per format -------------------------------------

    @property
    @abstractmethod
    def path(self) -> str:
        """Source file path."""

    @property
    @abstractmethod
    def file_type(self) -> FileType:
        """FULL, GRID, or SOLUTION."""

    @property
    @abstractmethod
    def title(self) -> str:
        """Dataset title string."""

    @property
    @abstractmethod
    def variables(self) -> list[str]:
        """Ordered list of variable name strings."""

    @property
    @abstractmethod
    def zones(self) -> ZoneList[TecplotZoneReader]:
        """Zones in this file, by 0-based index or slice."""

    @property
    @abstractmethod
    def auxdata(self) -> TecplotAuxDataReader:
        """Dataset-level auxiliary data."""

    @abstractmethod
    def _var_auxdata_at(self, var_index: int) -> TecplotAuxDataReader:
        """Return aux data for variable *var_index* (0-based, unchecked)."""

    def close(self) -> None:  # noqa: B027
        """Release any open resources. No-op unless overridden.

        SZL overrides this to close its C file handle; PLT and DAT hold no handle
        between accesses and use this default.
        """

    # -- Shared ------------------------------------------------------------------------

    def __enter__(self) -> TecplotReader:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

    @property
    def num_vars(self) -> int:
        """Number of variables in the dataset."""
        return len(self.variables)

    @property
    def num_zones(self) -> int:
        """Number of zones in the file."""
        return len(self.zones)

    @property
    def num_auxdata_items(self) -> int:
        """Number of dataset-level auxiliary data items."""
        return len(self.auxdata)

    @property
    def var_auxdata(self) -> list[TecplotAuxDataReader]:
        """Per-variable auxiliary data, 0-based.

        Derived from :meth:`get_var_auxdata`, a subclass never implements this
        separately.
        """
        return [self.get_var_auxdata(i) for i in range(self.num_vars)]

    def get_var_auxdata(self, var_index: int) -> TecplotAuxDataReader:
        """Return auxiliary data for variable *var_index*.

        Raises:
            IndexError: If *var_index* is out of range.
        """
        if var_index < 0 or var_index >= self.num_vars:
            raise IndexError(
                f"Variable index {var_index} out of range [0, {self.num_vars - 1}]"
            )
        return self._var_auxdata_at(var_index)

    def get_zone_auxdata(self, zone_index: int) -> TecplotAuxDataReader:
        """Return auxiliary data for zone *zone_index*.

        Raises:
            IndexError: If *zone_index* is out of range.
        """
        if zone_index < 0 or zone_index >= self.num_zones:
            raise IndexError(
                f"Zone index {zone_index} out of range [0, {self.num_zones - 1}]"
            )
        return self.zones[zone_index].auxdata

    def __repr__(self) -> str:  # pragma: no cover
        cls = type(self).__name__
        name = self.path.replace("\\", "/").rsplit("/", 1)[-1]
        try:
            parts = [f"path={name!r}"]
            title = self.title
            if title:
                if len(title) > 40:
                    title = title[:25] + "\u2026"
                parts.append(f"title={title!r}")
            parts += [
                f"file_type={self.file_type.name}",
                f"zones={self.num_zones}",
                f"variables={self.num_vars}",
            ]
            if self.num_auxdata_items:
                parts.append(f"aux={self.num_auxdata_items}")
            return f"{cls}({', '.join(parts)})"
        except Exception:
            return f"{cls}(path={name!r}, <unavailable>)"
