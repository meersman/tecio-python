"""Shared, format-independent base class for Tecplot writers.

Every format's writer (SZL, PLT, DAT) exposes the same lifecycle: construct with a path
and optional variable list (eager or lazy open), buffer auxiliary data with
:meth:`~TecplotWriter.add_auxdataset_dict`/ :meth:`~TecplotWriter.add_auxvar_dict`,
write zones with ``write_ordered_zone``/ ``write_fe_zone``/ ``write_poly_zone``, and
close (directly or via context manager). This module defines the parts of that
lifecycle that are identical across formats once, so the three writers can no longer
drift apart the way the readers had.

Notes:
    * Unlike the reader hierarchy, writer instances are not made immutable.  A writer is
      inherently a mutable, sequentially-appended-to object (``current_zone`` grows, aux
      buffers fill and drain, a file handle opens and closes), there's no "already-read,
      now frozen" state to protect the way there was for reader Zone/Variable objects.
    * :meth:`~TecplotWriter.flush_aux` centralizes the aux-data buffer draining and key
      resolution (a 0-based int or a variable name), which was previously duplicated
      nearly verbatim in all three formats. Each format only implements the two small
      hooks that actually differ, writing one dataset-level item and one variable-level
      item.
    * ``_open``, ``close``, and the ``write_*_zone`` methods stay abstract: how a file
      is opened/closed and how a zone is actually written are genuinely
      format-specific (a live C handle for SZL, a global implicit context for PLT's
      classic API, a plain text file for DAT). Their signatures here are the common
      shape for documentation; Python doesn't enforce exact signature matching on
      ``abstractmethod``, so a format may add its own extra keyword-only parameters
      (SZL's ``flush``, for subzone flushing mid-write) without conflict.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
from ._meta import WriterMeta
from ._utils import public_dir

if TYPE_CHECKING:
    from ._reader import TecplotPolyZoneReader


class EmptyZoneWarning(UserWarning):
    """Emitted when a zone has no active variables to write.

    Raised when every dataset variable is either passive or shared for this zone, such
    that there is no data to write. No zone is created, rather than raising an error, so
    the writer stays open and usable for subsequent zones.
    """


# passive_vars/var_sharing each name only the variables that deviate from the default
# (active, not shared)
PassiveVarsSpec = Collection[int | str]
VarSharingSpec = Mapping[int | str, int]

_STR_TO_PRECISION: dict[str, DataType] = {
    "single": DataType.FLOAT,
    "float": DataType.FLOAT,
    "double": DataType.DOUBLE,
}

# Faces per cell
_FACES_PER_ELEM: dict[ZoneType, int] = {
    ZoneType.FELINESEG: 2,
    ZoneType.FETRIANGLE: 3,
    ZoneType.FEQUADRILATERAL: 4,
    ZoneType.FETETRAHEDRON: 4,
    ZoneType.FEBRICK: 6,
}


def count_face_connections(flat: npt.NDArray, mode: FaceNeighborMode) -> int:
    """Count connections in a flat face-neighbor array for the given mode.

    Shared across formats. Always derived from *flat* itself, never asked for
    separately: unlike reading (where a dedicated count avoids loading a potentially
    large array just to know its size), on write the caller already has the full array
    in memory, so deriving costs nothing extra, and it makes a mismatched explicit
    count, silently producing a corrupted file, structurally impossible.

    ``LOCAL_ONE_TO_ONE``/``GLOBAL_ONE_TO_ONE`` connections are a fixed 3 or 4 values
    each, so the count is *len(flat)* divided by that. The "many" modes are ragged: each
    record starts with a 4-value header whose own 4th value (``nz``) gives the count of
    additional neighbor references that follow it, so counting means walking record by
    record until the array is exhausted, the same walk
    :func:`~tecio.reshape_face_connections` does to parse them, just counting steps
    instead of collecting values.

    Raises:
        ValueError: If *flat*'s length doesn't cleanly divide for the one-to-one modes,
            or ends mid-record for the "many" modes, either way, this is a reliable
            signal of malformed input.
    """
    n = len(flat)
    if mode in (FaceNeighborMode.LOCAL_ONE_TO_ONE, FaceNeighborMode.GLOBAL_ONE_TO_ONE):
        values_per = 3 if mode is FaceNeighborMode.LOCAL_ONE_TO_ONE else 4
        if n % values_per != 0:
            raise ValueError(
                f"face_neighbors has {n} values, not a multiple of "
                f"{values_per} as required for {mode.name}."
            )
        return n // values_per

    is_global = mode is FaceNeighborMode.GLOBAL_ONE_TO_MANY
    i = 0
    count = 0
    while i < n:
        if i + 4 > n:
            raise ValueError(
                f"face_neighbors is truncated: a new record starts at "
                f"index {i} but fewer than 4 values remain ({n - i})."
            )
        nz = int(flat[i + 3])
        record_len = 4 + (2 * nz if is_global else nz)
        if i + record_len > n:
            raise ValueError(
                f"face_neighbors is truncated: the record starting at "
                f"index {i} needs {record_len} values but only "
                f"{n - i} remain."
            )
        i += record_len
        count += 1
    return count


def validate_face_neighbor_sharing(
    con_sharing: int | None, face_neighbor_mode: FaceNeighborMode | None
) -> None:
    """Raise if a global face-neighbor mode is combined with con_sharing.

    Shared across formats. Per the classic API's own documented constraint:
    connectivity, and any face-neighbor data associated with it, cannot be shared
    between zones when the face-neighbor mode is global, only local modes support
    sharing (implicitly, along with connectivity itself, via con_sharing, there's no
    separate face-neighbor-sharing mechanism).
    """
    global_modes = (
        FaceNeighborMode.GLOBAL_ONE_TO_ONE,
        FaceNeighborMode.GLOBAL_ONE_TO_MANY,
    )
    if con_sharing is not None and face_neighbor_mode in global_modes:
        raise ValueError(
            f"face_neighbor_mode={face_neighbor_mode.name} cannot be "
            "combined with con_sharing: connectivity (and any face-"
            "neighbor data with it) cannot be shared between zones when "
            "the face-neighbor mode is global."
        )


def validate_face_neighbors(
    flat: npt.NDArray,
    mode: FaceNeighborMode,
    zone_type: ZoneType,
    num_cells: int,
) -> None:
    """Validate face-neighbor connections against this zone's own shape.

    Shared across formats. Structural compatibility only: every local cell reference is
    a valid 0-based cell index for this zone, and every face index is a valid 0-based
    face ordinal for this zone's cell type. Doesn't independently verify the
    *geometric* adjacency itself (that two cells stated as sharing a face genuinely
    share the node set that face implies), that would mean recomputing adjacency from
    node_map ourselves, duplicating the very auto-detection Tecplot already does; this
    only rules out references that are structurally impossible regardless of geometry.

    Remote references (the zone/cell pairs in the global modes) aren't checked, they
    refer to a different zone's own cell count, which isn't known here.

    Raises:
        ValueError: If any local cell reference is out of range [0, num_cells - 1], or
            any face index is out of range [0, faces_per_cell - 1] for *zone_type*.
    """
    faces_per_cell = _FACES_PER_ELEM.get(zone_type)
    if faces_per_cell is None:
        return  # poly/mixed zones: no fixed face count, not this function's job

    def check_cell(cz: int, pos: int) -> None:
        if not 0 <= cz < num_cells:
            raise ValueError(
                f"face_neighbors[{pos}]: cell {cz} is out of range "
                f"[0, {num_cells - 1}] for this zone."
            )

    def check_face(fz: int, pos: int) -> None:
        if not 0 <= fz < faces_per_cell:
            raise ValueError(
                f"face_neighbors[{pos}]: face {fz} is out of range "
                f"[0, {faces_per_cell - 1}] for {zone_type.name} "
                f"({faces_per_cell} faces per cell)."
            )

    if mode is FaceNeighborMode.LOCAL_ONE_TO_ONE:
        for i in range(0, len(flat), 3):
            check_cell(int(flat[i]), i)
            check_face(int(flat[i + 1]), i + 1)
            check_cell(int(flat[i + 2]), i + 2)
    elif mode is FaceNeighborMode.GLOBAL_ONE_TO_ONE:
        for i in range(0, len(flat), 4):
            check_cell(int(flat[i]), i)
            check_face(int(flat[i + 1]), i + 1)
    else:
        is_global = mode is FaceNeighborMode.GLOBAL_ONE_TO_MANY
        i = 0
        n = len(flat)
        while i < n:
            cz, fz, nz = int(flat[i]), int(flat[i + 1]), int(flat[i + 3])
            check_cell(cz, i)
            check_face(fz, i + 1)
            if is_global:
                record_len = 4 + 2 * nz
            else:
                record_len = 4 + nz
                for j in range(nz):
                    check_cell(int(flat[i + 4 + j]), i + 4 + j)
            i += record_len


def convert_face_connections_to_one_based(
    flat: npt.NDArray, mode: FaceNeighborMode
) -> npt.NDArray:
    """Convert a flat, public 0-based face-neighbor array to fully 1-based.

    Inverse of :func:`~tecio._reader.convert_face_connections_to_zero_based`, shared
    by any writer whose destination is inherently 1-based throughout (SZL's live C
    calls, DAT's ASCII text).

    For the one-to-one modes, every value in a record (cell, face ordinal, and for
    the global mode, remote zone and remote cell) is an index, so the whole array
    shifts uniformly. The "many" modes are ragged and mix in two non-index fields
    (``oz``, the face obscuration flag, and ``nz``, the trailing-reference count)
    that must stay untouched, so each record is converted individually instead.

    Args:
        flat: The raw flat array, exactly as validated for writing, 0-based.
        mode: This zone's face-neighbor mode.

    Returns:
        A new flat array with cell/face/remote-zone fields converted to 1-based.
    """
    one_to_one = (FaceNeighborMode.LOCAL_ONE_TO_ONE, FaceNeighborMode.GLOBAL_ONE_TO_ONE)
    if mode in one_to_one:
        return flat + 1

    flat = flat.copy()
    is_global = mode is FaceNeighborMode.GLOBAL_ONE_TO_MANY
    i = 0
    while i < flat.size:
        nz = int(flat[i + 3])
        rest_count = 2 * nz if is_global else nz
        flat[i] += 1  # local cell number
        flat[i + 1] += 1  # face ordinal
        # flat[i + 2] (oz, the obscuration flag) and nz itself are not indices.
        flat[i + 4 : i + 4 + rest_count] += 1  # trailing cell/zone references
        i += 4 + rest_count
    return flat


def var_sharing_for_libtecio(share_idx: Sequence[int | None]) -> list[int]:
    """Convert a public 0-based ``var_sharing`` array to the TecIO C API's form.

    Both the classic and SZL C APIs use ``0`` to mean "not shared" and a 1-based
    zone number otherwise, so ``None`` becomes ``0`` and every real zone index is
    shifted by one.
    """
    return [0 if s is None else s + 1 for s in share_idx]


def zone_for_libtecio(zone: int | None) -> int:
    """Convert a public 0-based zone reference to the TecIO C API's form.

    ``None`` (no sharing) becomes ``0``; a real 0-based zone index is shifted to
    the C API's 1-based zone number.
    """
    return 0 if zone is None else zone + 1


# -- Variable key resolution -----------------------------------------------------------


def resolve_variable_key(key: int | str, dataset_variables: Sequence[str]) -> int:
    """Return the 0-based dataset variable index for *key*.

    Shared by aux-data key resolution (:meth:`~TecplotWriter._resolve_var_index`) and
    the sparse forms of *passive_vars*/*var_sharing* below, so a variable is always
    identified the same way everywhere in the writer API: either its exact name, or its
    0-based dataset index.

    Args:
        key: A variable name (exact match) or a 0-based index.
        dataset_variables: The full dataset variable name list.

    Raises:
        TypeError: *key* is neither ``int`` nor ``str`` (``bool`` included, since it is
            a subtype of ``int`` but never a meaningful key).
        IndexError: An index is out of range.
        KeyError: A name doesn't match any dataset variable.
    """
    if isinstance(key, bool):
        raise TypeError(
            f"Variable key must be a name (str) or 0-based index (int), got {key!r}"
        )
    if isinstance(key, int):
        if key not in range(len(dataset_variables)):
            raise IndexError(
                f"Variable index {key} out of bounds [0, {len(dataset_variables) - 1}]"
            )
        return key
    if isinstance(key, str):
        try:
            return list(dataset_variables).index(key)
        except ValueError:
            raise KeyError(
                f"Variable {key!r} not found in {list(dataset_variables)}"
            ) from None
    raise TypeError(
        f"Variable key must be a name (str) or 0-based index (int), got {key!r}"
    )


def _check_uniform_key_type(keys: Iterable[int | str], label: str) -> None:
    """Raise if *keys* names variables by both name and 0-based index.

    A single passive_vars/var_sharing specification must address variables one way or
    the other. Mixing them introduces ambiguity and, for var_sharing, would let one
    entry silently overwrite the other.

    Args:
        keys: The keys to check (bool values are ignored here; they are rejected with a
            more specific error by :func:`resolve_variable_key`).
        label: Parameter name to use in the error message.

    Raises:
        TypeError: *keys* contains at least one name and at least one index.
    """
    has_name = any(isinstance(k, str) for k in keys)
    has_index = any(isinstance(k, int) and not isinstance(k, bool) for k in keys)
    if has_name and has_index:
        raise TypeError(
            f"{label} must use variable names or 0-based indices, not a mix of both."
        )


def resolve_passive_vars(
    passive_keys: Collection[int | str], dataset_variables: Sequence[str]
) -> list[bool]:
    """Expand a list-like set of passive variable keys to an expanded flag list.

    Args:
        passive_keys: Variable names, or 0-based indices, to mark passive. Must be all
            names or all indices.
        dataset_variables: The full dataset variable name list.

    Raises:
        TypeError: *passive_keys* mixes names and indices.
    """
    _check_uniform_key_type(passive_keys, "passive_vars")
    passive = [False] * len(dataset_variables)
    for key in passive_keys:
        passive[resolve_variable_key(key, dataset_variables)] = True
    return passive


def resolve_var_sharing(
    sharing_map: Mapping[int | str, int],
    dataset_variables: Sequence[str],
    *,
    passive_vars: Sequence[bool],
) -> list[int | None]:
    """Expand a sparse ``{variable: source zone}`` mapping to a dense list.

    Args:
        sharing_map: Maps a variable name, or 0-based index, to the 0-based zone index
            to share its data from. Keys must be all names or all indices.
        dataset_variables: The full dataset variable name list.
        passive_vars: Already-resolved dense passive_vars list, checked against so no
            variable ends up both passive and shared.

    Raises:
        TypeError: *sharing_map* mixes names and indices.
        ValueError: A key is already marked passive in *passive_vars*.
    """
    _check_uniform_key_type(sharing_map.keys(), "var_sharing")
    sharing: list[int | None] = [None] * len(dataset_variables)
    for key, zone in sharing_map.items():
        var_idx = resolve_variable_key(key, dataset_variables)
        if passive_vars[var_idx]:
            raise ValueError(
                f"Variable {dataset_variables[var_idx]!r} is both passive "
                "and shared; a variable cannot be both."
            )
        sharing[var_idx] = int(zone)
    return sharing


def _normalize_passive_and_sharing(
    passive_vars: PassiveVarsSpec | None,
    var_sharing: VarSharingSpec | None,
    dataset_variables: Sequence[str],
) -> tuple[list[bool], list[int | None]]:
    """Resolve sparse *passive_vars*/*var_sharing* to dense internal arrays.

    Each names only the variables that deviate from the default (active, not shared);
    ``None`` or an empty collection means all-active/no-sharing.  *passive_vars* accepts
    any list, tuple, or set of names, or of indices, but not a mix of both within the
    same call (see :func:`_check_uniform_key_type`). This is expanded to dataset length
    arrays here for input to TecIO functions. The dense sharing array uses ``None`` for
    "not shared", since ``0`` is a real (0-based) zone index and can no longer double
    as a sentinel.

    Raises:
        TypeError: *passive_vars* is given but isn't a list/tuple/set, or *var_sharing*
            is given but isn't a mapping, or either mixes variable names and indices.
            The old dense form, a sequence with one entry per dataset variable, is no
            longer accepted; specify only the variables that are passive or shared
            instead.
        ValueError: A variable is named in both *passive_vars* and *var_sharing*.
        KeyError: A name doesn't match any dataset variable.
        IndexError: An index is out of range.
    """
    if isinstance(passive_vars, (str, bytes)) or (
        passive_vars is not None
        and not isinstance(passive_vars, (AbstractSet, Sequence))
    ):
        raise TypeError(
            "passive_vars must be a list, tuple, or set of variable names, or of "
            f"0-based indices, e.g. {{'unused'}} or [1]; got "
            f"{type(passive_vars).__name__}."
        )
    if var_sharing is not None and not isinstance(var_sharing, Mapping):
        raise TypeError(
            "var_sharing must be a {variable: source zone} mapping, e.g. "
            f"{{'x': 0}}; got {type(var_sharing).__name__}."
        )

    passive = (
        resolve_passive_vars(passive_vars, dataset_variables)
        if passive_vars
        else [False] * len(dataset_variables)
    )
    sharing: list[int | None] = (
        resolve_var_sharing(var_sharing, dataset_variables, passive_vars=passive)
        if var_sharing
        else [None] * len(dataset_variables)
    )
    return passive, sharing


@dataclass
class PreparedOrderedZone:
    """Validated, normalized inputs for writing one ordered (IJK) zone.

    Everything a writer needs to proceed straight to its own format-specific zone
    creation dispatch call, already validated and shape-consistent.
    """

    value_locations: list[ValueLocation]
    passive_vars: list[bool]
    var_sharing: list[int | None]
    active_var_idx: list[int]  # 0-based dataset indices being written now
    imax: int
    jmax: int
    kmax: int
    value_locations_global: list[ValueLocation]
    variable_types_global: list[DataType]


def prepare_ordered_zone(
    arrays: Sequence[npt.NDArray],
    variable_types: Sequence[DataType],
    *,
    value_locations: Sequence[ValueLocation] | None,
    passive_vars: PassiveVarsSpec | None,
    var_sharing: VarSharingSpec | None,
    dataset_variables: Sequence[str],
    meta: WriterMeta,
    on_error: Callable[[], None] | None = None,
) -> PreparedOrderedZone | None:
    """Validate and normalize inputs for writing an ordered (IJK) zone.

    Shared across formats: derives imax/jmax/kmax from the supplied arrays' shapes,
    resolves value_locations/passive_vars/var_sharing defaults, validates the
    active-variable count and every array's shape (local and shared) against a single
    reference shape, and builds the dataset-level (not just active-variable)
    value_locations/ variable_types arrays every format's zone-creation call needs.

    Args:
        arrays:            NumPy arrays already converted from the caller's raw data,
                           one per active (non-passive, non-shared) variable.
        variable_types:    Per-active-variable data type, one per array in
                           *arrays*. Computed by the caller: type-inference strategy is
                           currently format-specific (SZL/DAT infer from each array's
                           own dtype, PLT always uses its configured precision), so it
                           isn't derived here.
        value_locations:   Per-active-variable value location, or None for all-NODAL.
        passive_vars:      Variables to mark passive: a list, tuple, or set of variable
                           names, or of 0-based indices (not a mix of both), e.g.
                           ``{"x"}`` or ``[1]``. None means all active.
        var_sharing:       Variables to share from another zone: a ``{variable: source
                           zone}`` mapping, name or 0-based index to 0-based zone index,
                           e.g. ``{"x": 0}``. None or an empty mapping means no sharing.
        dataset_variables: The full dataset variable name list, already established
                           (after the writer's own lazy-open handling).
        meta:              The writer's own zone metadata, for resolving shared
                           variables' source zones.
        on_error:          Optional callback invoked once, before any ValueError is
                           raised, for a format needing extra cleanup on failure (DAT's
                           partial-file deletion; SZL/PLT need none, the C library
                           handles it, so they pass None).

    Returns:
        A :class:`PreparedOrderedZone` with everything else needed to proceed to the
        format-specific zone-creation call, or ``None`` if every dataset variable
        is passive or shared, leaving nothing to write. In the ``None`` case an
        :class:`EmptyZoneWarning` is emitted and the caller should return without
        creating a zone; ``on_error`` is not invoked since nothing has gone wrong.

    Raises:
        TypeError: *passive_vars*/*var_sharing* isn't a list/tuple/set/mapping as
            appropriate, or mixes variable names and indices.
        ValueError: On any inconsistency: wrong active-variable count, shape mismatch
            (local or shared), a shared variable referencing a zone that doesn't exist
            yet or isn't ORDERED, or a variable named in both *passive_vars* and
            *var_sharing*.
        KeyError: A *passive_vars*/*var_sharing* name doesn't match any dataset
            variable.
        IndexError: A *passive_vars*/*var_sharing* index is out of range.

    """
    try:
        if value_locations is None:
            value_locations = [ValueLocation.NODAL] * len(arrays)
        passive_flags, share_idx = _normalize_passive_and_sharing(
            passive_vars, var_sharing, dataset_variables
        )

        if len(arrays) != len(dataset_variables):
            expected_vars = sum(
                1
                for is_passive, share_zone in zip(passive_flags, share_idx, strict=True)
                if not is_passive and share_zone is None
            )
            if expected_vars == 0:
                warnings.warn(
                    "Zone has no active variables to write (every variable is "
                    "passive or shared); skipping this zone.",
                    EmptyZoneWarning,
                    stacklevel=3,
                )
                return None
            if len(arrays) == 0:
                raise ValueError(
                    f"No data arrays provided. Expected {expected_vars} "
                    "active variable arrays based on passive_vars and "
                    "var_sharing."
                )
            if len(arrays) != expected_vars:
                raise ValueError(
                    f"Expected {expected_vars} data arrays, got {len(arrays)}."
                )

        active_var_idx = [
            var_idx
            for var_idx, (is_passive, sharing_zone_idx) in enumerate(
                zip(passive_flags, share_idx, strict=True)
            )
            if (not is_passive) and (sharing_zone_idx is None)
        ]
        active_local_idx = {var_idx: i for i, var_idx in enumerate(active_var_idx)}

        # Determine validation reference shape. NODAL local or shared variable arrays
        # give shape dimensions directly; CELL_CENTERED is only used as a fallback
        # (ambiguous for a degenerate axis).
        nodal_shape: tuple[int, ...] | None = None
        ndims: int | None = None
        cell_fallback: tuple[int, ...] | None = None
        cell_fallback_ndims: int | None = None

        for var_idx in range(len(dataset_variables)):
            if passive_flags[var_idx]:
                continue
            src = share_idx[var_idx]
            if src is not None:
                if nodal_shape is None:
                    src_zone = meta.zone(src)
                    if src_zone is None or src_zone.dimensions is None:
                        raise ValueError(
                            f"Variable {var_idx} shares from zone {src}, "
                            "which has not been written yet, or is not an "
                            "ORDERED zone."
                        )
                    nodal_shape = src_zone.dimensions
                continue

            arr = arrays[active_local_idx[var_idx]]
            loc = value_locations[active_local_idx[var_idx]]
            arr_ndims = arr.ndim
            if arr_ndims not in (1, 2, 3):
                raise ValueError(
                    f"Arrays must be 1D, 2D, or 3D; got {arr_ndims}-D "
                    "array. For time-dependent data, write each time step "
                    "as a separate zone."
                )
            shape = arr.shape + (1,) * (3 - arr_ndims)
            if loc == ValueLocation.NODAL:
                if nodal_shape is None:
                    nodal_shape, ndims = shape, arr_ndims
            elif cell_fallback is None:
                cell_fallback, cell_fallback_ndims = shape, arr_ndims

        if nodal_shape is None:
            if cell_fallback is not None:
                nodal_shape = tuple(n + 1 for n in cell_fallback)
                ndims = cell_fallback_ndims
            else:
                raise ValueError("Could not determine zone dimensions.")

        cell_shape = tuple(max(n - 1, 1) for n in nodal_shape)
        imax, jmax, kmax = nodal_shape

        # Validate every non-passive dataset variable (local and shared) against the
        # reference shape.
        for var_idx in range(len(dataset_variables)):
            if passive_flags[var_idx]:
                continue
            src = share_idx[var_idx]
            if src is not None:
                src_zone = meta.zone(src)
                if src_zone is None or src_zone.dimensions is None:
                    raise ValueError(
                        f"Variable {var_idx} shares from zone {src}, which "
                        "has not been written yet, or is not an ORDERED "
                        "zone."
                    )
                if src_zone.dimensions != nodal_shape:
                    raise ValueError(
                        f"Variable {var_idx} shares from zone {src} with "
                        f"dimensions {src_zone.dimensions}, which does not "
                        f"match this zone's dimensions {nodal_shape}."
                    )
                continue

            i = active_local_idx[var_idx]
            arr, loc = arrays[i], value_locations[i]
            if ndims is not None and arr.ndim != ndims:
                raise ValueError(f"Array {i} is {arr.ndim}D, expected {ndims}D.")
            shape = arr.shape + (1,) * (3 - arr.ndim)
            if (loc == ValueLocation.NODAL) and (shape != nodal_shape):
                raise ValueError(
                    f"Array {i} is NODAL but has shape {shape}, expected {nodal_shape}."
                )
            if (loc == ValueLocation.CELL_CENTERED) and (shape != cell_shape):
                raise ValueError(
                    f"Array {i} is CELL_CENTERED but has shape {shape}, "
                    f"expected {cell_shape}."
                )

        variable_types_global = [DataType.DOUBLE] * len(dataset_variables)
        value_locations_global = [ValueLocation.NODAL] * len(dataset_variables)
        for local_idx, var_idx in enumerate(active_var_idx):
            variable_types_global[var_idx] = variable_types[local_idx]
            value_locations_global[var_idx] = value_locations[local_idx]

    except ValueError:
        if on_error is not None:
            on_error()
        raise

    return PreparedOrderedZone(
        value_locations=list(value_locations),
        passive_vars=[bool(p) for p in passive_flags],
        var_sharing=list(share_idx),
        active_var_idx=active_var_idx,
        imax=imax,
        jmax=jmax,
        kmax=kmax,
        value_locations_global=value_locations_global,
        variable_types_global=variable_types_global,
    )


@dataclass
class _ZoneVariables:
    """Per-variable state shared by every unstructured zone preparation."""

    value_locations: list[ValueLocation]
    passive_flags: list[bool]
    share_idx: list[int | None]
    active_var_idx: list[int]
    value_locations_global: list[ValueLocation]
    variable_types_global: list[DataType]


def _prepare_zone_variables(
    arrays: Sequence[npt.NDArray],
    variable_types: Sequence[DataType],
    *,
    value_locations: Sequence[ValueLocation] | None,
    passive_vars: PassiveVarsSpec | None,
    var_sharing: VarSharingSpec | None,
    dataset_variables: Sequence[str],
    num_nodes: int,
    num_cells: int,
    meta: WriterMeta,
    stacklevel: int,
) -> _ZoneVariables | None:
    """Resolve and validate the variable arguments of an unstructured zone.

    Resolves defaults, validates the active-variable count and every array's size
    (local and shared) against *num_nodes*/*num_cells*, and builds the dataset-level
    value location and type arrays every format's zone-creation call needs.

    Returns:
        The resolved variable state, or ``None`` after emitting an
        :class:`EmptyZoneWarning` (attributed *stacklevel* frames up) when every
        dataset variable is passive or shared.

    Raises:
        ValueError: On any inconsistency, see :func:`prepare_fe_zone`.
    """
    if value_locations is None:
        value_locations = [ValueLocation.NODAL] * len(arrays)
    passive_flags, share_idx = _normalize_passive_and_sharing(
        passive_vars, var_sharing, dataset_variables
    )

    if len(arrays) != len(dataset_variables):
        expected_vars = sum(
            1
            for is_passive, sharing_zone_idx in zip(
                passive_flags, share_idx, strict=True
            )
            if not is_passive and sharing_zone_idx is None
        )
        if expected_vars == 0:
            warnings.warn(
                "Zone has no active variables to write (every variable is "
                "passive or shared); skipping this zone.",
                EmptyZoneWarning,
                stacklevel=stacklevel,
            )
            return None
        if len(arrays) == 0:
            raise ValueError(
                "No data arrays provided for active variables. "
                f"Expected {expected_vars} active variables based on "
                "passive_vars and var_sharing settings."
            )
        if len(arrays) != expected_vars:
            raise ValueError(
                f"Expected {expected_vars} data arrays, got {len(arrays)}."
            )

    active_var_idx = [
        var_idx
        for var_idx, (is_passive, sharing_zone_idx) in enumerate(
            zip(passive_flags, share_idx, strict=True)
        )
        if (not is_passive) and (sharing_zone_idx is None)
    ]

    # Shared variable data shape validation
    for var_idx, src in enumerate(share_idx):
        if src is None:
            continue
        src_zone = meta.zone(src)
        if src_zone is None:
            raise ValueError(
                f"Variable {var_idx} shares from zone {src}, which has "
                "not been written yet."
            )
        src_loc = (
            src_zone.value_locations[var_idx]
            if var_idx < len(src_zone.value_locations)
            else ValueLocation.NODAL
        )
        if src_loc == ValueLocation.CELL_CENTERED:
            if src_zone.num_elements != num_cells:
                raise ValueError(
                    f"Variable {var_idx} shares from zone {src} with "
                    f"{src_zone.num_elements} cells, which does not "
                    f"match this zone's cell count of {num_cells}."
                )
        elif src_zone.num_nodes != num_nodes:
            raise ValueError(
                f"Variable {var_idx} shares from zone {src} with "
                f"{src_zone.num_nodes} nodes, which does not match "
                f"this zone's node count of {num_nodes}."
            )

    # Local variable data shape validation
    for i, (arr, loc) in enumerate(zip(arrays, value_locations, strict=True)):
        if (loc == ValueLocation.NODAL) and (arr.size != num_nodes):
            raise ValueError(
                f"Array {i} is NODAL but has {arr.size} values, expected {num_nodes}."
            )
        elif (loc == ValueLocation.CELL_CENTERED) and (arr.size != num_cells):
            raise ValueError(
                f"Array {i} is CELL_CENTERED but has {arr.size} "
                f"values, expected {num_cells}."
            )

    variable_types_global = [DataType.DOUBLE] * len(dataset_variables)
    value_locations_global = [ValueLocation.NODAL] * len(dataset_variables)
    for local_idx, var_idx in enumerate(active_var_idx):
        variable_types_global[var_idx] = variable_types[local_idx]
        value_locations_global[var_idx] = value_locations[local_idx]
    return _ZoneVariables(
        value_locations=list(value_locations),
        passive_flags=[bool(p) for p in passive_flags],
        share_idx=list(share_idx),
        active_var_idx=active_var_idx,
        value_locations_global=value_locations_global,
        variable_types_global=variable_types_global,
    )


@dataclass
class PreparedFEZone:
    """Validated, normalized inputs for writing one finite-element zone.

    Everything a writer needs to proceed straight to its own format-specific
    zone-creation dispatch call, already validated and shape-consistent.
    """

    value_locations: list[ValueLocation]
    passive_vars: list[bool]
    var_sharing: list[int | None]
    con_sharing: int | None
    active_var_idx: list[int]  # 0-based dataset indices being written now
    num_nodes: int
    num_cells: int
    value_locations_global: list[ValueLocation]
    variable_types_global: list[DataType]
    face_neighbors_arr: npt.NDArray | None
    face_neighbor_mode: FaceNeighborMode | None
    num_face_connections: int


def prepare_fe_zone(
    arrays: Sequence[npt.NDArray],
    variable_types: Sequence[DataType],
    zone_type: ZoneType,
    *,
    node_map: npt.ArrayLike | None,
    value_locations: Sequence[ValueLocation] | None,
    passive_vars: PassiveVarsSpec | None,
    var_sharing: VarSharingSpec | None,
    con_sharing: int | None,
    face_neighbors: npt.ArrayLike | None,
    face_neighbor_mode: FaceNeighborMode | None,
    dataset_variables: Sequence[str],
    meta: WriterMeta,
    on_error: Callable[[], None] | None = None,
) -> PreparedFEZone | None:
    """Validate and normalize inputs for writing a finite-element zone.

    Shared across formats: derives num_nodes/num_cells from node_map (or the con_sharing
    source zone), resolves value_locations/passive_vars/ var_sharing defaults, validates
    the active-variable count and every array's size (local and shared) against those
    counts, validates and derives face-neighbor state (mode, count, structural
    compatibility), and builds the dataset-level value_locations/variable_types arrays
    every format's zone-creation call needs.

    Args:
        arrays: NumPy arrays already converted from the caller's raw data, one per
            active (non-passive, non-shared) variable.
        variable_types: Per-active-variable data type, one per array in
            *arrays*. Computed by the caller (see
            :func:`prepare_ordered_zone` for why this isn't derived here).
        zone_type: This zone's :class:`~tecio.ZoneType`.
        node_map: Connectivity array, or None if *con_sharing* is set.
        value_locations: Per-active-variable value location, or None for all-NODAL.
        passive_vars: Variables to mark passive: a list, tuple, or set of variable
            names, or of 0-based indices (not a mix of both), e.g. ``{"x"}`` or
            ``[1]``. None means all active.
        var_sharing: Variables to share from another zone: a ``{variable: source zone}``
            mapping, name or 0-based index to 0-based zone index, e.g. ``{"x": 0}``.
            None or an empty mapping means no sharing.
        con_sharing: Source zone index to share connectivity from, or None for no
            sharing.
        face_neighbors: Optional flat face-neighbor connectivity array.  Left in its
            natural dtype (never forced to a particular integer width here): a writer
            needing a specific dtype for its own dispatch call (PLT's int32-only classic
            API, for instance) casts the returned array itself.
        face_neighbor_mode: Required if *face_neighbors* is given, invalid if it isn't;
            see :func:`validate_face_neighbor_sharing`.
        dataset_variables: The full dataset variable name list.
        meta: The writer's own zone metadata, for resolving shared connectivity/variable
            source zones.
        on_error: Optional callback invoked once, before any ValueError is raised, for a
            format needing extra cleanup on failure (DAT's partial-file deletion;
            SZL/PLT need none, the C library handles it, so they pass None).

    Returns:
        A :class:`PreparedFEZone` with everything else needed to proceed to the
        format-specific zone-creation call, or ``None`` if every dataset variable is
        passive or shared, leaving nothing to write. In the ``None`` case an
        :class:`EmptyZoneWarning` is emitted and the caller should return without
        creating a zone; ``on_error`` is not invoked since nothing has gone wrong.

    Raises:
        TypeError: *passive_vars*/*var_sharing* isn't a list/tuple/set/mapping as
            appropriate, or mixes variable names and indices.
        ValueError: On any inconsistency: wrong active-variable count, array-size
            mismatch (local or shared), con_sharing/var_sharing referencing a zone that
            doesn't exist yet or isn't FE, any face-neighbor validation failure, or a
            variable named in both *passive_vars* and *var_sharing*.
        KeyError: A *passive_vars*/*var_sharing* name doesn't match any dataset
            variable.
        IndexError: A *passive_vars*/*var_sharing* index is out of range.
    """
    try:
        if node_map is not None:
            node_map_arr = np.asarray(node_map)
            num_cells = node_map_arr.shape[0]
            num_nodes = int(node_map_arr.max()) + 1
        elif con_sharing is not None:
            src_zone = meta.zone(con_sharing)
            if (
                src_zone is None
                or src_zone.num_nodes is None
                or src_zone.num_elements is None
            ):
                raise ValueError(
                    f"con_sharing={con_sharing} references a zone that has "
                    "not been written yet, or is not a finite-element zone."
                )
            num_nodes = src_zone.num_nodes
            num_cells = src_zone.num_elements
        else:
            raise ValueError(
                "node_map must be provided unless connectivity is shared "
                "from another zone via con_sharing."
            )

        zone_vars = _prepare_zone_variables(
            arrays,
            variable_types,
            value_locations=value_locations,
            passive_vars=passive_vars,
            var_sharing=var_sharing,
            dataset_variables=dataset_variables,
            num_nodes=num_nodes,
            num_cells=num_cells,
            meta=meta,
            stacklevel=4,
        )
        if zone_vars is None:
            return None
        value_locations = zone_vars.value_locations
        passive_flags = zone_vars.passive_flags
        share_idx = zone_vars.share_idx
        active_var_idx = zone_vars.active_var_idx
        value_locations_global = zone_vars.value_locations_global
        variable_types_global = zone_vars.variable_types_global

        # Face-neighbor connections
        if face_neighbors is not None and face_neighbor_mode is None:
            face_neighbor_mode = FaceNeighborMode.LOCAL_ONE_TO_ONE
        elif face_neighbors is None and face_neighbor_mode is not None:
            raise ValueError(
                "face_neighbor_mode was given without face_neighbors; "
                "that combination has nothing to apply the mode to."
            )
        validate_face_neighbor_sharing(con_sharing, face_neighbor_mode)
        face_neighbors_arr: npt.NDArray | None = None
        num_face_connections = 0
        if face_neighbors is not None and con_sharing is None:
            # Face-neighbor data is inherited from the same source zone
            assert face_neighbor_mode is not None  # narrowed above
            face_neighbors_arr = np.ascontiguousarray(face_neighbors).ravel(order="C")
            num_face_connections = count_face_connections(
                face_neighbors_arr, face_neighbor_mode
            )
            validate_face_neighbors(
                face_neighbors_arr, face_neighbor_mode, zone_type, num_cells
            )

    except ValueError:
        if on_error is not None:
            on_error()
        raise

    return PreparedFEZone(
        value_locations=list(value_locations),
        passive_vars=[bool(p) for p in passive_flags],
        var_sharing=list(share_idx),
        con_sharing=con_sharing,
        active_var_idx=active_var_idx,
        num_nodes=num_nodes,
        num_cells=num_cells,
        value_locations_global=value_locations_global,
        variable_types_global=variable_types_global,
        face_neighbors_arr=face_neighbors_arr,
        face_neighbor_mode=(
            face_neighbor_mode if face_neighbors_arr is not None else None
        ),
        num_face_connections=num_face_connections,
    )


@dataclass
class _PolyConnectivity:
    """Validated face map of a poly zone plus the counts derived from it."""

    num_nodes: int
    num_cells: int
    num_faces: int
    total_face_nodes: int
    num_boundary_faces: int
    num_boundary_connections: int
    face_node_counts: npt.NDArray | None
    face_nodes: npt.NDArray | None
    face_left_elements: npt.NDArray | None
    face_right_elements: npt.NDArray | None
    boundary_connection_counts: npt.NDArray | None
    boundary_connection_elements: npt.NDArray | None
    boundary_connection_zones: npt.NDArray | None


@dataclass
class PreparedPolyZone:
    """Validated, normalized inputs for writing one poly (face-based) zone.

    The face arrays are 0-based exactly as passed in (``-1`` for no element,
    ``-(b + 2)`` for boundary face ``b``) and keep their integer dtype. A writer
    converts to its own convention and dtype right before dispatch. When connectivity
    is shared (:attr:`con_sharing` is not None) the array fields are None and the
    counts come from the source zone.
    """

    value_locations: list[ValueLocation]
    passive_vars: list[bool]
    var_sharing: list[int | None]
    con_sharing: int | None
    active_var_idx: list[int]  # 0-based dataset indices being written now
    value_locations_global: list[ValueLocation]
    variable_types_global: list[DataType]
    connectivity: _PolyConnectivity


def _as_index_array(value: npt.ArrayLike, name: str) -> npt.NDArray:
    """Return *value* as a flat integer array, without copying when possible."""
    arr = np.asarray(value)
    if arr.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array, got dtype {arr.dtype}.")
    return arr.ravel()


def _check_boundary_arrays(
    counts: npt.ArrayLike | None,
    elements: npt.ArrayLike | None,
    zones: npt.ArrayLike | None,
    *,
    num_cells: int,
    meta: WriterMeta,
) -> tuple[npt.NDArray | None, npt.NDArray | None, npt.NDArray | None]:
    """Validate the boundary connection arrays against the zones they reference.

    The Tecplot libraries do not bounds-check these, a bad zone or element index
    corrupts the file or crashes, so they are checked here.
    """
    given = [a is not None for a in (counts, elements, zones)]
    if not any(given):
        return None, None, None
    if not all(given):
        raise ValueError(
            "boundary_connection_counts, boundary_connection_elements and "
            "boundary_connection_zones must be given together."
        )
    assert counts is not None and elements is not None and zones is not None
    n_counts = _as_index_array(counts, "boundary_connection_counts")
    n_elems = _as_index_array(elements, "boundary_connection_elements")
    n_zones = _as_index_array(zones, "boundary_connection_zones")
    if n_counts.size and n_counts.min() < 1:
        raise ValueError("boundary_connection_counts entries must be positive.")
    if int(n_counts.sum()) != n_elems.size:
        raise ValueError(
            f"boundary_connection_counts sums to {int(n_counts.sum())}, but "
            f"boundary_connection_elements has {n_elems.size} entries."
        )
    if n_zones.size != n_elems.size:
        raise ValueError(
            f"boundary_connection_zones has {n_zones.size} entries, expected "
            f"{n_elems.size}."
        )
    if n_elems.size == 0:
        return n_counts, n_elems, n_zones

    new_index = meta.num_zones
    if n_elems.min() < 0:
        raise ValueError("boundary_connection_elements must be 0-based (>= 0).")
    if n_zones.min() < 0 or int(n_zones.max()) > new_index:
        raise ValueError(
            "boundary_connection_zones must be 0-based and reference a zone already "
            f"written or this zone (0 to {new_index})."
        )
    for zone in np.unique(n_zones).tolist():
        if zone == new_index:
            cells: int | None = num_cells
        else:
            zone_meta = meta.zone(zone)
            cells = zone_meta.num_cells if zone_meta is not None else None
        if cells is not None and int(n_elems[n_zones == zone].max()) >= cells:
            raise ValueError(
                "boundary_connection_elements references an element past the "
                f"{cells} elements of zone {zone}."
            )
    return n_counts, n_elems, n_zones


def _check_poly_connectivity(
    zone_type: ZoneType,
    face_nodes: npt.ArrayLike | None,
    face_node_counts: npt.ArrayLike | None,
    face_left_elements: npt.ArrayLike | None,
    face_right_elements: npt.ArrayLike | None,
    boundary_connection_counts: npt.ArrayLike | None,
    boundary_connection_elements: npt.ArrayLike | None,
    boundary_connection_zones: npt.ArrayLike | None,
    meta: WriterMeta,
) -> _PolyConnectivity:
    """Validate the face map of a poly zone that owns its connectivity."""
    if face_nodes is None or face_left_elements is None or face_right_elements is None:
        raise ValueError(
            "face_nodes, face_left_elements and face_right_elements are required "
            "unless connectivity is shared from another zone via con_sharing."
        )
    nodes = _as_index_array(face_nodes, "face_nodes")
    left = _as_index_array(face_left_elements, "face_left_elements")
    right = _as_index_array(face_right_elements, "face_right_elements")
    num_faces = left.size
    if num_faces == 0:
        raise ValueError("A poly zone needs at least one face.")
    if right.size != num_faces:
        raise ValueError(
            f"face_right_elements has {right.size} entries, expected one per face "
            f"({num_faces}, the length of face_left_elements)."
        )

    counts: npt.NDArray | None = None
    if zone_type is ZoneType.FEPOLYHEDRON:
        if face_node_counts is None:
            raise ValueError("face_node_counts is required for FEPOLYHEDRON zones.")
        counts = _as_index_array(face_node_counts, "face_node_counts")
        if counts.size != num_faces:
            raise ValueError(
                f"face_node_counts has {counts.size} entries, expected {num_faces}."
            )
        if counts.min() < 1:
            raise ValueError("face_node_counts entries must be positive.")
        total_face_nodes = int(counts.sum())
    else:
        # Every polygon face is an edge. Counts are implied, but accept them when
        # they agree so a reader's arrays can be passed straight back in.
        total_face_nodes = 2 * num_faces
        if face_node_counts is not None:
            given = _as_index_array(face_node_counts, "face_node_counts")
            if given.size != num_faces or (given != 2).any():
                raise ValueError(
                    "FEPOLYGON faces always have 2 nodes: omit face_node_counts or "
                    "pass 2 for every face."
                )
    if nodes.size != total_face_nodes:
        raise ValueError(
            f"face_nodes has {nodes.size} entries, expected {total_face_nodes}."
        )
    if nodes.min() < 0:
        raise ValueError("face_nodes must be 0-based node indices (no negatives).")

    top = max(int(left.max()), int(right.max()))
    if top < 0:
        raise ValueError(
            "No face references an element in face_left_elements/face_right_elements."
        )
    num_cells = top + 1

    bcounts, belems, bzones = _check_boundary_arrays(
        boundary_connection_counts,
        boundary_connection_elements,
        boundary_connection_zones,
        num_cells=num_cells,
        meta=meta,
    )
    num_boundary_faces = 0 if bcounts is None else bcounts.size
    # Boundary references in the left/right element arrays are -(b + 2) for boundary
    # face b.
    lowest = min(int(left.min()), int(right.min()))
    if lowest < -1 and -2 - lowest >= num_boundary_faces:
        raise ValueError(
            "face_left_elements/face_right_elements reference boundary face "
            f"{-2 - lowest}, but only {num_boundary_faces} boundary faces are defined."
        )
    return _PolyConnectivity(
        num_nodes=int(nodes.max()) + 1,
        num_cells=num_cells,
        num_faces=num_faces,
        total_face_nodes=total_face_nodes,
        num_boundary_faces=num_boundary_faces,
        num_boundary_connections=0 if belems is None else belems.size,
        face_node_counts=counts,
        face_nodes=nodes,
        face_left_elements=left,
        face_right_elements=right,
        boundary_connection_counts=bcounts,
        boundary_connection_elements=belems,
        boundary_connection_zones=bzones,
    )


def _shared_poly_connectivity(
    zone_type: ZoneType, con_sharing: int, meta: WriterMeta
) -> _PolyConnectivity:
    """Take the face-map sizes of a zone sharing connectivity from *con_sharing*."""
    src = meta.zone(con_sharing)
    if src is None or src.num_faces is None or src.num_nodes is None:
        raise ValueError(
            f"con_sharing={con_sharing} references a zone that has not been written "
            "yet, or is not a poly zone."
        )
    if src.zone_type is not zone_type:
        raise ValueError(
            f"con_sharing={con_sharing} is a {src.zone_type.name} zone, which cannot "
            f"share connectivity with a {zone_type.name} zone."
        )
    if src.num_boundary_faces:
        raise ValueError(
            f"con_sharing={con_sharing} has boundary faces, and a poly zone with "
            "connected boundary faces cannot share its connectivity."
        )
    assert src.num_elements is not None  # poly zones always record it
    return _PolyConnectivity(
        num_nodes=src.num_nodes,
        num_cells=src.num_elements,
        num_faces=src.num_faces,
        total_face_nodes=src.total_face_nodes or 0,
        num_boundary_faces=src.num_boundary_faces or 0,
        num_boundary_connections=src.num_boundary_connections or 0,
        face_node_counts=None,
        face_nodes=None,
        face_left_elements=None,
        face_right_elements=None,
        boundary_connection_counts=None,
        boundary_connection_elements=None,
        boundary_connection_zones=None,
    )


def prepare_poly_zone(
    arrays: Sequence[npt.NDArray],
    variable_types: Sequence[DataType],
    zone_type: ZoneType,
    *,
    face_nodes: npt.ArrayLike | None,
    face_node_counts: npt.ArrayLike | None,
    face_left_elements: npt.ArrayLike | None,
    face_right_elements: npt.ArrayLike | None,
    boundary_connection_counts: npt.ArrayLike | None,
    boundary_connection_elements: npt.ArrayLike | None,
    boundary_connection_zones: npt.ArrayLike | None,
    value_locations: Sequence[ValueLocation] | None,
    passive_vars: PassiveVarsSpec | None,
    var_sharing: VarSharingSpec | None,
    con_sharing: int | None,
    dataset_variables: Sequence[str],
    meta: WriterMeta,
    on_error: Callable[[], None] | None = None,
) -> PreparedPolyZone | None:
    """Validate and normalize inputs for writing a poly (face-based) zone.

    Shared across formats. Validates the face map (sizes, index ranges, boundary
    references, and the zones and elements that boundary connections point to), derives
    the node and element counts from it (or from the *con_sharing* source zone), and
    resolves the variable arguments exactly as :func:`prepare_fe_zone` does.

    Args:
        arrays: NumPy arrays for the active (non-passive, non-shared) variables.
        variable_types: Per-active-variable data type, one per array in *arrays*.
        zone_type: FEPOLYGON or FEPOLYHEDRON.
        face_nodes: Flat 0-based node indices of all faces.
        face_node_counts: Nodes per face, required for FEPOLYHEDRON. Optional for
            FEPOLYGON, where every face has 2 nodes.
        face_left_elements: Element on the left of each face (0-based, -1 for none,
            -(b + 2) for boundary face b).
        face_right_elements: Same for the right side.
        boundary_connection_counts: Connections per boundary face.
        boundary_connection_elements: Flat 0-based neighbor elements.
        boundary_connection_zones: Flat 0-based zone of each neighbor element.
        value_locations: Per-active-variable value location, or None for all-NODAL.
        passive_vars: Variables to mark passive, see :func:`prepare_fe_zone`.
        var_sharing: Variables to share from another zone, see :func:`prepare_fe_zone`.
        con_sharing: Source zone index to share connectivity from, or None. The source
            must be a poly zone of the same type without boundary faces.
        dataset_variables: The full dataset variable name list.
        meta: The writer's own zone metadata.
        on_error: Optional callback invoked once before any ValueError is raised.

    Returns:
        A :class:`PreparedPolyZone`, or ``None`` after an :class:`EmptyZoneWarning`
        when every dataset variable is passive or shared.

    Raises:
        ValueError: On any inconsistency in the face map, the variable arguments, or a
            *zone_type* that is not FEPOLYGON or FEPOLYHEDRON.
        TypeError: *passive_vars*/*var_sharing* has the wrong type.
        KeyError: A *passive_vars*/*var_sharing* name is unknown.
        IndexError: A *passive_vars*/*var_sharing* index is out of range.
    """
    try:
        if zone_type not in (ZoneType.FEPOLYGON, ZoneType.FEPOLYHEDRON):
            raise ValueError(
                f"write_poly_zone needs FEPOLYGON or FEPOLYHEDRON, got "
                f"{zone_type.name}; use write_fe_zone for other finite-element zones."
            )
        if con_sharing is not None:
            given = (
                face_nodes,
                face_node_counts,
                face_left_elements,
                face_right_elements,
                boundary_connection_counts,
                boundary_connection_elements,
                boundary_connection_zones,
            )
            if any(a is not None for a in given):
                raise ValueError(
                    "Face map arrays cannot be given together with con_sharing, the "
                    "connectivity comes from the source zone."
                )
            conn = _shared_poly_connectivity(zone_type, con_sharing, meta)
        else:
            conn = _check_poly_connectivity(
                zone_type,
                face_nodes,
                face_node_counts,
                face_left_elements,
                face_right_elements,
                boundary_connection_counts,
                boundary_connection_elements,
                boundary_connection_zones,
                meta,
            )
        zone_vars = _prepare_zone_variables(
            arrays,
            variable_types,
            value_locations=value_locations,
            passive_vars=passive_vars,
            var_sharing=var_sharing,
            dataset_variables=dataset_variables,
            num_nodes=conn.num_nodes,
            num_cells=conn.num_cells,
            meta=meta,
            stacklevel=4,
        )
        if zone_vars is None:
            return None
    except ValueError:
        if on_error is not None:
            on_error()
        raise

    return PreparedPolyZone(
        value_locations=zone_vars.value_locations,
        passive_vars=zone_vars.passive_flags,
        var_sharing=zone_vars.share_idx,
        con_sharing=con_sharing,
        active_var_idx=zone_vars.active_var_idx,
        value_locations_global=zone_vars.value_locations_global,
        variable_types_global=zone_vars.variable_types_global,
        connectivity=conn,
    )


def normalize_precision(
    precision: DataType | str | None, *, allow_none: bool
) -> DataType | None:
    """Return the :class:`~tecio.libtecio.DataType` for *precision*.

    Args:
        precision: ``None``, the enum directly, or a case-insensitive string
            (``"single"``/``"float"``/``"double"``).
        allow_none: Whether ``None`` is a valid result. SZL: yes, per-variable data type
            is inferred automatically from each array, with no file-wide override to
            normalize. PLT and DAT: no, ``precision`` must resolve to a concrete
            FLOAT/DOUBLE, but for different reasons and with different scope:

            * PLT's classic API has one ``VIsDouble`` flag for the entire file (set once
              at ``tecini142``); every variable, including integer-valued data, is
              declared and stored at that single type. There is no per-variable type in
              PLT's zone header at all.
            * DAT's ``precision`` only decides two things: the ASCII significant-digit
              count used to format floating values, and which of FLOAT/DOUBLE is
              declared for variables whose own array is itself float-typed. A variable
              inferred as an integer type keeps its own INT32/INT16/BYTE in the zone's
              ``DT=`` declaration regardless of ``precision`` (see
              :func:`~tecio._dat_write._resolve_written_type`), since that declaration
              is what Tecplot uses to allocate memory on read, not the printed digit
              count. A value can be written as ``1.000000000e0`` and still be declared
              (and read back) as an integer.

    Raises:
        ValueError: If *precision* is ``None`` and *allow_none* is False, or if it's
            neither ``None`` nor FLOAT/DOUBLE (or a recognized string alias for one of
            them).
    """
    if precision is None:
        if allow_none:
            return None
        raise ValueError(
            "precision=None is not supported by this format; use "
            "DataType.FLOAT/DataType.DOUBLE (or 'single'/'double')."
        )
    if isinstance(precision, str):
        try:
            precision = _STR_TO_PRECISION[precision.strip().lower()]
        except KeyError:
            raise ValueError(
                f"precision={precision!r} is not recognized; use 'single' or "
                "'double' (or DataType.FLOAT / DataType.DOUBLE)."
            ) from None
    if precision not in (DataType.FLOAT, DataType.DOUBLE):
        raise ValueError(
            f"precision={precision!r} is not supported; precision only "
            "applies to floating-point variables -- use DataType.FLOAT, "
            "DataType.DOUBLE, or None."
        )
    return precision


def poly_face_kwargs(
    zone: TecplotPolyZoneReader, zone_map: Mapping[int, int] | None = None
) -> dict[str, Any]:
    """Return the ``write_poly_zone`` face arguments that reproduce *zone*.

    The arrays are passed as the reader returns them (0-based). Boundary connection
    arrays are included only when the zone has boundary faces. Callers that preserve
    connectivity sharing should skip this and pass ``con_sharing``.

    Args:
        zone: Source poly zone.
        zone_map: Source zone index to output zone index, for tools that renumber or
            drop zones. The zones in ``boundary_connection_zones`` are renumbered with
            it. A connection to a zone that is not in the map is dropped, and a
            boundary face left with no connections becomes an ordinary face. ``None``
            keeps the source numbering.
    """
    left = zone.face_left_elements
    right = zone.face_right_elements
    kwargs: dict[str, Any] = {
        "face_nodes": zone.face_nodes,
        "face_node_counts": zone.face_node_counts,
        "face_left_elements": left,
        "face_right_elements": right,
    }
    if not zone.num_boundary_faces:
        return kwargs

    counts = zone.boundary_connection_counts
    elements = zone.boundary_connection_elements
    zones = zone.boundary_connection_zones
    if zone_map is not None:
        left, right, counts, elements, zones = _remap_boundary_connections(
            zone, zone_map
        )
        kwargs["face_left_elements"] = left
        kwargs["face_right_elements"] = right
    if counts.size:
        kwargs["boundary_connection_counts"] = counts
        kwargs["boundary_connection_elements"] = elements
        kwargs["boundary_connection_zones"] = zones
    return kwargs


def _remap_boundary_connections(
    zone: TecplotPolyZoneReader, zone_map: Mapping[int, int]
) -> tuple[npt.NDArray, ...]:
    """Renumber the zones a poly zone is connected to, see :func:`poly_face_kwargs`.

    Returns:
        ``(face_left_elements, face_right_elements, counts, elements, zones)``.

    """
    left = zone.face_left_elements
    right = zone.face_right_elements
    counts = zone.boundary_connection_counts
    elements = zone.boundary_connection_elements
    zones = zone.boundary_connection_zones

    lookup = np.full(int(zones.max()) + 1, -1, dtype=np.int64)
    for source, output in zone_map.items():
        if 0 <= source < lookup.size:
            lookup[source] = output
    new_zones = lookup[zones]
    keep = new_zones >= 0
    if keep.all():
        return left, right, counts, elements, new_zones

    # Drop the connections to zones that are gone, then renumber the boundary faces
    # that still have connections. A reference to boundary face b is -(b + 2).
    owner = np.repeat(np.arange(counts.size), counts)
    new_counts = np.bincount(owner[keep], minlength=counts.size)
    alive = new_counts > 0
    new_id = np.cumsum(alive) - 1
    warnings.warn(
        f"Zone '{zone.title}': dropped {int((~keep).sum())} boundary connection(s) "
        f"to zones that are not in the output, and {int((~alive).sum())} boundary "
        "face(s) with no connection left became ordinary faces.",
        stacklevel=2,
    )

    def renumber(refs: npt.NDArray) -> npt.NDArray:
        refs = refs.copy()
        is_boundary = refs <= -2
        old = -2 - refs[is_boundary]
        refs[is_boundary] = np.where(alive[old], -2 - new_id[old], -1)
        return refs

    return (
        renumber(left),
        renumber(right),
        new_counts[alive],
        elements[keep],
        new_zones[keep],
    )


class TecplotWriter(ABC):
    """Shared interface and lifecycle for all Tecplot file writers.

    Concrete subclasses (:class:`~tecio.TecplotSzlWriter`, ...) differ in how a file is
    actually opened, closed, and written to, but share the same aux-data buffering,
    variable-list handling, and context-manager lifecycle, so application code can be
    written against this base without caring which format is being produced.

    Args:
        path: Output file path.
        title: Dataset title.
        variables: Variable name list. ``None`` defers file creation until the first
            zone-writing call (lazy open).
        file_type: File type enum (FULL, GRID, or SOLUTION).

    Attributes:
        path: Output file path.
        title: Dataset title string.
        variables: Variable name list, or ``None`` if the file has not been opened yet.
        file_type: File type (FULL, GRID, or SOLUTION).
        current_zone: 0-based index of the most recently written zone. ``-1`` before
            any zone has been written; updated only after a zone-writing method
            successfully completes.
        auxdataset: Buffered dataset-level auxiliary data, flushed before the first
            zone.
        auxvar: Buffered variable-level auxiliary data, flushed before the first zone.
    """

    def __init__(
        self,
        path: str,
        title: str,
        variables: list[str] | None,
        file_type: FileType,
    ) -> None:
        self.path: str = str(path)
        self.title: str = title
        self.variables: list[str] | None = variables
        self.file_type: FileType = file_type
        self.current_zone: int = -1
        self.auxdataset: dict[str, str] = {}
        self.auxvar: dict[int, dict[str, str]] = {}
        self._meta = WriterMeta(
            path=self.path,
            title=self.title,
            file_type=self.file_type,
            file_format=self._file_format,
        )
        if self.variables is not None:
            self._open(self.variables)

    @property
    @abstractmethod
    def _file_format(self) -> str:
        """Format tag for :class:`~tecio._meta.WriterMeta`, e.g. ``'szplt'``."""

    def __dir__(self) -> list[str]:  # pragma: no cover
        """List only the public attributes and methods."""
        return public_dir(self)

    # -- Context manager ---------------------------------------------------------------

    def __enter__(self) -> TecplotWriter:
        """Support ``with`` statement, returns *self*."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the file on context-manager exit.

        The file is closed regardless of whether an exception was raised in the ``with``
        block. If closing itself raises, that secondary exception is only re-raised when
        the ``with`` block completed without error; otherwise the original exception
        takes precedence.
        """
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

    # -- Validation --------------------------------------------------------------------

    def _check_variables(self) -> list[str]:
        """Return the variable list, raising if the file has not been opened yet."""
        if self.variables is None:
            raise RuntimeError(
                "Attempted to access variable name list before they were set. "
                "Ensure variables are set on initialization or zone write."
            )
        return self.variables

    @property
    def meta(self) -> WriterMeta:
        """Read-only record of everything written to this file so far."""
        return self._meta

    # -- File lifecycle: format-specific -----------------------------------------------

    @abstractmethod
    def _open(self, var_names: list[str]) -> None:
        """Open the file/write the header. Called at most once per instance."""

    @abstractmethod
    def close(self) -> None:
        """Finalize and close the file. Safe to call more than once."""

    # -- Aux data: buffering and key resolution are shared, the actual write isn't -----

    def add_auxdataset_dict(self, auxdict: dict[str, Any]) -> None:
        """Buffer dataset-level auxiliary data from a dictionary."""
        self.auxdataset.update(auxdict)

    def add_auxvar_dict(self, auxdict: dict[int, dict[str, Any]]) -> None:
        """Buffer variable-level auxiliary data from a dictionary."""
        self.auxvar.update(auxdict)

    def _resolve_var_index(self, key: int | str) -> int:
        """Return the 0-based variable index for an aux-data key.

        Delegates to :func:`resolve_variable_key`, the same resolver the sparse forms of
        *passive_vars*/*var_sharing* use, so a variable is identified the same way
        everywhere in the writer API.

        Args:
            key: A 0-based variable index, or an exact variable name.

        Raises:
            IndexError: If a 0-based index key is out of range.
            KeyError: If a name key doesn't match any variable.
            TypeError: If *key* is neither ``int`` nor ``str``.
        """
        return resolve_variable_key(key, self._check_variables())

    def flush_aux(self) -> None:
        """Write buffered dataset- and variable-level aux data to the file.

        Called automatically before the first zone is written. Only needed directly if
        you want to flush explicitly, e.g. before checking :attr:`meta`.
        """
        for name, value in self.auxdataset.items():
            self._write_dataset_aux_item(str(name), str(value))

        for key, subdict in self.auxvar.items():
            var_idx = self._resolve_var_index(key)
            for name, value in subdict.items():
                self._write_var_aux_item(var_idx + 1, str(name), str(value))

        # Record counts, then clear buffers -- each item is written exactly once.
        self._meta.note_dataset_aux(len(self.auxdataset))
        self._meta.note_var_aux(sum(len(subdict) for subdict in self.auxvar.values()))
        self.auxdataset.clear()
        self.auxvar.clear()

    @abstractmethod
    def _write_dataset_aux_item(self, name: str, value: str) -> None:
        """Write one dataset-level aux item. Called only from :meth:`flush_aux`."""

    @abstractmethod
    def _write_var_aux_item(
        self, one_based_var_index: int, name: str, value: str
    ) -> None:
        """Write one variable-level aux item. Called only from :meth:`flush_aux`."""

    # -- Zone writers: fully format-specific ------------------------------------------

    @abstractmethod
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
        """Write a complete IJK-ordered zone. See the concrete subclass for details."""

    @abstractmethod
    def write_fe_zone(
        self,
        data: Sequence[npt.ArrayLike],
        zone_type: ZoneType,
        *,
        node_map: npt.ArrayLike | None = None,
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
    ) -> None:
        """Write a complete finite-element zone."""

    @abstractmethod
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
        """Write a complete FEPOLYGON or FEPOLYHEDRON zone."""
