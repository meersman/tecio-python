r"""Structural metadata records shared by the Tecplot file writers.

The writer for each output format -- SZL (``.szplt``), PLT (``.plt``), and ASCII DAT
(``.dat``) -- keeps a running, in-memory description of what it has committed to disk:
the dataset header, auxiliary-data counts, and one record per zone. Defining that
description once here lets all three writers share a single representation instead of
each maintaining ad-hoc bookkeeping.

The records mirror the file- and zone-level fields of the Tecplot data format
(``TECINI142`` for the file header and ``TECZNE142`` for each zone): dataset title, file
type, variable names, and the per-variable passive, value-location, and share-from
arrays, together with per-zone dimensions and aux-item counts.

Design notes:
    * Only lightweight descriptors are stored -- shapes, enums, and small integers --
      never the variable data arrays, so the record stays cheap in memory even for files
      with many zones.
    * ``slots=True`` removes the per-instance ``__dict__``, and immutable tuple fields
      keep the per-variable lists compact. A :class:`ZoneMeta` is a write-once snapshot
      and is therefore ``frozen``; :class:`WriterMeta` is mutable because it grows as
      zones are written.
    * Enum types are imported only under :data:`typing.TYPE_CHECKING`. With ``from
      __future__ import annotations`` the annotations are never evaluated at runtime,
      so no import of :mod:`tecio._constants` happens at all unless a type checker is
      running. Not that it would matter either way, ``_constants`` is a dependency-free
      leaf module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._constants import (
        DataType,
        FaceNeighborMode,
        FileType,
        ValueLocation,
        ZoneType,
    )


# =====================================================================================
# Zone-level record
# =====================================================================================


@dataclass(slots=True, frozen=True)
class ZoneMeta:
    """Lightweight, write-once description of a single written zone.

    Per-variable fields (:attr:`value_locations`, :attr:`passive_vars`,
    :attr:`shared_vars`, :attr:`data_types`) span every dataset variable and correspond
    to the ``ValueLocation``, ``PassiveVarList``, ``ShareVarFromZone``, and
    variable-type arrays of a ``TECZNE142`` zone header.

    Attributes:
        index:                    0-based index of this zone within the dataset.
        title:                    Zone title.
        zone_type:                The zone's :class:`~tecio.libtecio.ZoneType`.
        solution_time:            Solution time (``0.0`` for static zones).
        strand_id:                Strand ID (``0`` for static zones).
        num_aux_items:            Number of zone-level auxiliary items written.
        dimensions:               Nodal ``(imax, jmax, kmax)`` for ORDERED zones, else
                                  ``None``.
        num_nodes:                Node count for FE zones, else ``None``.
        num_elements:             Element count for FE zones, else ``None``.
        face_neighbor_mode:       The zone's :class:`~tecio.FaceNeighborMode`, else
                                  ``None`` if it has no face-neighbor connections.
        num_face_connections:     Number of face-neighbor connections written, else
                                  ``None`` if it has none.
        num_faces:                Face count for poly zones, else ``None``.
        total_face_nodes:         Total face-node entries for poly zones, else ``None``.
        num_boundary_faces:       Boundary face count for poly zones, else ``None``.
        num_boundary_connections: Total boundary connections for poly zones, else
                                  ``None``.
        value_locations:          Per-variable value location, full dataset length.
        passive_vars:             Per-variable passive flags, full dataset length.
        shared_vars:              Per-variable share-from zone index, or ``None`` for
                                  not shared, full dataset length.
        data_types:               Per-variable data type, full dataset length.
    """

    index: int
    title: str
    zone_type: ZoneType
    solution_time: float = 0.0
    strand_id: int = 0
    num_aux_items: int = 0
    # Ordered zones carry IJK dimensions; FE zones carry node/element counts.
    dimensions: tuple[int, int, int] | None = None
    num_nodes: int | None = None
    num_elements: int | None = None
    face_neighbor_mode: FaceNeighborMode | None = None
    num_face_connections: int | None = None
    # Poly zones (FEPOLYGON/FEPOLYHEDRON) also record their face-map sizes.
    num_faces: int | None = None
    total_face_nodes: int | None = None
    num_boundary_faces: int | None = None
    num_boundary_connections: int | None = None
    # Per-variable descriptors (length == dataset variable count).
    value_locations: tuple[ValueLocation, ...] = ()
    passive_vars: tuple[bool, ...] = ()
    shared_vars: tuple[int | None, ...] = ()
    data_types: tuple[DataType, ...] = ()

    @property
    def nodal_shape(self) -> tuple[int, int, int] | None:
        """Nodal ``(imax, jmax, kmax)`` for ORDERED zones, else ``None``."""
        return self.dimensions

    @property
    def num_cells(self) -> int | None:
        """Cell count of any zone type, or ``None`` if not recorded."""
        if self.num_elements is not None:
            return self.num_elements
        shape = self.cell_shape
        if shape is None:
            return None
        return shape[0] * shape[1] * shape[2]

    @property
    def cell_shape(self) -> tuple[int, ...] | None:
        """Cell-centred ``(imax-1, jmax-1, kmax-1)`` (floored at 1) or ``None``."""
        if self.dimensions is None:
            return None
        return tuple(max(n - 1, 1) for n in self.dimensions)


# =====================================================================================
# Dataset-level record
# =====================================================================================


@dataclass(slots=True)
class WriterMeta:
    """Running record of everything a writer has committed to a file.

    Populated incrementally as the file header, auxiliary data, and zones are
    written, so it always reflects the current on-disk state. It is the single
    source of truth for cross-zone validation (for example, resolving the shape
    of a shared variable) and is suitable for summarising the writer state.

    Attributes:
        path:                  Output file path.
        title:                 Dataset title.
        file_type:             :class:`~tecio.libtecio.FileType` of the output.
        file_format:           Format tag, e.g. ``"szplt"``, ``"plt"``, or
                               ``"dat"``.
        variables:             Variable name list, or ``None`` before the file
                               header has been written (lazy-open).
        num_dataset_aux_items: Count of dataset-level aux items written.
        num_var_aux_items:     Total variable-level aux items written.
        zones:                 Mapping of 0-based zone index to :class:`ZoneMeta`,
                               in write order.
    """

    path: str
    title: str
    file_type: FileType
    file_format: str
    variables: list[str] | None = None
    num_dataset_aux_items: int = 0
    num_var_aux_items: int = 0
    zones: dict[int, ZoneMeta] = field(default_factory=dict)

    # -- Derived quantities -----------------------------------------------------------

    @property
    def num_vars(self) -> int:
        """Number of dataset variables, or ``0`` before the header is written."""
        return len(self.variables) if self.variables is not None else 0

    @property
    def num_zones(self) -> int:
        """Number of zones written so far."""
        return len(self.zones)

    # -- Update methods (called by the writer as it commits data) ---------------------

    def set_variables(self, names: list[str]) -> None:
        """Record the dataset variable names once the header is written."""
        self.variables = names

    def note_dataset_aux(self, count: int) -> None:
        """Accumulate the number of dataset-level aux items written."""
        self.num_dataset_aux_items += count

    def note_var_aux(self, count: int) -> None:
        """Accumulate the number of variable-level aux items written."""
        self.num_var_aux_items += count

    def record_zone(self, zone: ZoneMeta) -> None:
        """Register a fully written zone by its 0-based index."""
        self.zones[zone.index] = zone

    # -- Retrieval helpers ------------------------------------------------------------

    def zone(self, index: int) -> ZoneMeta | None:
        """Return the :class:`ZoneMeta` for *index*, or ``None`` if unknown."""
        return self.zones.get(index)
