"""ParaView reader plugin for Tecplot 360 data files, built on ``tecio``.

This plugin closes three gaps in ParaView's built-in Tecplot reader: it cannot read
``.szplt``, it errors out on some ASCII zone header fields, and does not read auxiliary
data as VTK :class:`~vtkmodules.vtkCommonDataModel. vtkFieldData`.

Output structure
    Each file is read into one ``vtkMultiBlockDataSet``, one block per zone, named after
    the zone's title (or ``"Zone <n>"`` if untitled):

    * ``ORDERED`` zones become ``vtkStructuredGrid`` blocks.
    * Classic finite-element zones (line, triangle, quadrilateral, tetrahedron, brick)
      become ``vtkUnstructuredGrid`` blocks.
    * ``FEPOLYGON`` zones become ``vtkUnstructuredGrid`` blocks of ``VTK_POLYGON``
      cells, and ``FEPOLYHEDRON`` zones become blocks of ``VTK_POLYHEDRON`` cells. The
      face-based connectivity is converted per element: polygon edges are chained into
      the element's node loop, and polyhedron faces are listed for each of the two
      elements that share them (reversed for the right element) so every face points
      out of its cell. Faces whose neighbor is another zone (boundary connections) are
      treated as plain boundary faces.
    * ``FEMIXED`` zones are not yet supported and are skipped with a printed warning.
    * A zone whose aux data marks it a non-``Wall`` boundary (see "Zone visibility"
      below) still becomes a block, but its entry in the ``Zones`` property defaults
      to unchecked, matching Tecplot's own default display convention.

    Auxiliary data has no first-class equivalent in VTK, so it is mapped onto the
    nearest matching :class:`vtkFieldData`, each item as a one-element array named after
    the aux-data key -- numeric (int64/float64) when the value parses cleanly as a
    number, else a :class:`vtkStringArray`. Tecplot itself treats aux data as usable
    scalars in its own equations, and a numeric VTK array is the equivalent for
    ParaView: it works directly in the Python Calculator, Calculator filter, and
    numeric-oriented annotation filters, not just as inert display text.

    * Dataset-level aux data -> the root ``vtkMultiBlockDataSet``'s field data.
    * Variable-level aux data -> also the root field data, keyed ``"<variable
      name>::<aux key>"`` (there is no per-array metadata slot that ParaView's UI
      surfaces to users).
    * Zone-level aux data -> that zone's own block's field data.

Coordinates
    Tecplot data files do not tag which variables are spatial coordinates (that is
    ordinarily a plot/frame setting, not a file-format concept), so this reader resolves
    each axis in priority order:

    1. The ``XArray``/``YArray``/``ZArray`` properties, if set to a real variable name.
    2. The dataset aux data items ``Common.XVar``/``Common.YVar``/``Common.ZVar``, per
       the Tecplot data format guide -- each holds the 1-based dataset variable number
       to use for that axis, when the file itself specifies one.
    3. A name match against some spelling of ``x``/``y``/``z`` (case-insensitively,
       e.g. ``"X"``, ``"x-coordinate"``).
    4. For X and Y only, positional fallback to the first and second dataset variables,
       since almost every Tecplot CFD file leads with its coordinates. Z has no
       positional fallback beyond the third variable and defaults to all-zero (2-D
       data) when nothing is found by steps 1-3.

Vector components
    If the dataset aux data specifies ``Common.UVar``/``Common.VVar``/``Common.WVar``
    (or the ``UArray``/``VArray``/``WArray`` properties are set manually), this reader
    assembles those component variables into one 3-component array per zone so glyphs,
    streamlines, and vector-magnitude coloring work without a separate Calculator
    step. Unlike coordinates, there's no name-based guessing here (solver vector
    naming varies too much to guess reliably), so nothing is built at all unless at
    least one of U/V/W actually resolves; a resolved-but-missing component (2-D flow,
    or passive in a particular zone) is treated as zero rather than blocking the
    other components. The array is named ``"Velocity"`` if the dataset aux item
    ``Common.VectorVarsAreVelocity`` is truthy, else ``"Momentum"``.

Zone visibility
    Tecplot sets default visiblilty for zones with auxdata that marks them a boundary
    that isn't a wall (``Common.IsBoundaryZone`` true and ``Common.BoundaryCondition``
    not ``"Wall"``).

Known limitations (first pass)
    * ``FEMIXED`` zones are skipped. Polygon and polyhedron zones need a ``tecio``
      that reads them (PLT and DAT files only; SZPLT files cannot hold them yet), and
      are skipped with a printed warning when it does not.
    * A polygon element is assumed to have its left face neighbors on the left of each
      directed edge (counterclockwise). If an element's edges do not chain into one
      closed loop, the loop is rebuilt without relying on edge direction.
    * SOLUTION-only files (grid coordinates stored in a separate, unopened GRID file)
      have no coordinate data to build geometry from and are not handled specially; they
      will fail coordinate resolution unless the variables happen to still be present.
    * Boundary connections between zones, custom labels, geometry/text annotations,
      and Tecplot's classic-PLT auxiliary record types outside dataset/zone/variable
      aux data are not handled by ``tecio`` data readers.

Indexing
    ``tecio``'s Python API is 0-based, so zone titles that are missing fall back to
    ``"Zone <zone_index>"`` counting from 0, and the ``Zones`` property keys are
    ``"<zone_index>: <title>"``. File-level values that Tecplot defines as 1-based
    (the ``Common.XVar``/``YVar``/``ZVar``/``UVar``/``VVar``/``WVar`` aux items) are
    still read as 1-based variable numbers.

Requirements
    The ``tecio`` package must be importable from the same Python interpreter ParaView
    uses to run this plugin.
"""

from __future__ import annotations

import math
import os
import sys
import weakref
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from paraview.util.vtkAlgorithm import smdomain, smhint, smproperty, smproxy
from vtkmodules.util import numpy_support
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonCore import (
    vtkDataArraySelection,
    vtkPoints,
    vtkStringArray,
    vtkUnsignedCharArray,
)
from vtkmodules.vtkCommonDataModel import (
    VTK_EMPTY_CELL,
    VTK_HEXAHEDRON,
    VTK_LINE,
    VTK_POLYGON,
    VTK_POLYHEDRON,
    VTK_QUAD,
    VTK_TETRA,
    VTK_TRIANGLE,
    vtkCellArray,
    vtkCompositeDataSet,
    vtkFieldData,
    vtkMultiBlockDataSet,
    vtkStructuredGrid,
    vtkUnstructuredGrid,
)

# Because ParaView's bundled Python (vtkpython) has its own write-protected
# site-packages tecio cannot directly be installed into the pvpython environment,
# create a TECIO_PYTHONPATH environment variable to point to the local install
# (e.g. "/Users/you/tecio-for-paraview").
_TECIO_FALLBACK_PATH = os.environ.get("TECIO_PYTHONPATH", "")
if _TECIO_FALLBACK_PATH and _TECIO_FALLBACK_PATH not in sys.path:
    sys.path.insert(0, _TECIO_FALLBACK_PATH)

try:
    import tecio
    from tecio import (
        TecplotFEZoneReader,
        TecplotOrderedZoneReader,
        ValueLocation,
        ZoneType,
    )

    # Polygon and polyhedron zone support is newer than the rest of the reader API.
    TecplotPolyZoneReader = getattr(tecio, "TecplotPolyZoneReader", None)
    _TECIO_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment-dependent  # noqa: BLE001
    tecio = None  # ty: ignore[invalid-assignment]
    TecplotFEZoneReader = TecplotOrderedZoneReader = None  # ty: ignore[invalid-assignment]
    TecplotPolyZoneReader = None  # ty: ignore[invalid-assignment]
    ValueLocation = ZoneType = None  # ty: ignore[invalid-assignment]
    _TECIO_IMPORT_ERROR = exc

if TYPE_CHECKING:
    from tecio import TecplotFEZoneReader as _TecioFEZone
    from tecio import TecplotOrderedZoneReader as _TecioOrderedZone
    from tecio import TecplotReader as _TecioReader
    from tecio import TecplotZoneReader as _TecioZone

# --------------------------------------------------------------------------------------
# Module-level constants and small helpers
# --------------------------------------------------------------------------------------

_EXTENSIONS = "szplt plt bin dat tec"
_FILE_DESCRIPTION = "Tecplot 360 Data Files"

# Maps a supported classic finite-element ZoneType to its VTK cell type
_VTK_CELL_TYPE: dict[Any, int] = (
    {
        ZoneType.FELINESEG: VTK_LINE,
        ZoneType.FETRIANGLE: VTK_TRIANGLE,
        ZoneType.FEQUADRILATERAL: VTK_QUAD,
        ZoneType.FETETRAHEDRON: VTK_TETRA,
        ZoneType.FEBRICK: VTK_HEXAHEDRON,
    }
    if tecio is not None
    else {}
)

# Exact-match (case-insensitive, trimmed) spellings of each coordinate axis
_AXIS_SYNONYMS: dict[str, frozenset[str]] = {
    "x": frozenset({
        "x",
        "x-coordinate",
        "x coordinate",
        "coordinate x",
        "coord-x",
        "coordx",
        "xcoord",
        "x_coord",
        "xgrid",
        "x-grid",
        "x_grid",
    }),
    "y": frozenset({
        "y",
        "y-coordinate",
        "y coordinate",
        "coordinate y",
        "coord-y",
        "coordy",
        "ycoord",
        "y_coord",
        "ygrid",
        "y-grid",
        "y_grid",
    }),
    "z": frozenset({
        "z",
        "z-coordinate",
        "z coordinate",
        "coordinate z",
        "coord-z",
        "coordz",
        "zcoord",
        "z_coord",
        "zgrid",
        "z-grid",
        "z_grid",
    }),
}


# Case-insensitive truthy spellings Tecplot recognizes for Common.IsBoundaryZone
_BOUNDARY_ZONE_TRUE_VALUES = frozenset({"yes", "y", "true", "t", "on"})


def _is_deactivated_boundary_zone(zone: _TecioZone) -> bool:
    """Return True if *zone* should default to hidden, per Tecplot's own convention.

    Mirrors Tecplot 360's documented behavior for the ``Common.IsBoundaryZone`` /
    ``Common.BoundaryCondition`` zone aux data pair: a zone marked a boundary is
    deactivated (hidden) by default unless its boundary condition is exactly
    ``"Wall"``. Only the default *initial* state is affected -- the zone still loads
    as a block either way, and its checkbox in the ``Zones`` property can always be
    re-enabled.
    """
    aux = zone.auxdata
    is_boundary = aux.get("Common.IsBoundaryZone", "").strip().lower()
    if is_boundary not in _BOUNDARY_ZONE_TRUE_VALUES:
        return False
    return aux.get("Common.BoundaryCondition", "") != "Wall"


class _UnsupportedZoneError(Exception):
    """Raised internally when a zone's geometry can't be represented in VTK.

    Caught by :meth:`TecplotReader.RequestData`, which prints a warning and skips the
    offending zone rather than failing the whole read.
    """


def _modified_callback(target: VTKPythonAlgorithmBase):
    """Return a callback that calls ``target.Modified()`` while it's alive.

    Wires a :class:`vtkDataArraySelection`'s ``ModifiedEvent`` back to the owning reader
    so that toggling an array/zone checkbox in the ParaView UI correctly marks the
    pipeline as needing re-execution. A weak reference avoids keeping the reader alive
    solely because the observer closure refers to it.
    """
    ref = weakref.ref(target)

    def _callback(*_args: Any) -> None:
        obj = ref()
        if obj is not None:
            obj.Modified()

    return _callback


def _autodetect_axis(names: list[str], axis: str) -> str | None:
    """Return the variable name in *names* that looks like *axis*, or ``None``."""
    synonyms = _AXIS_SYNONYMS[axis]
    for name in names:
        if name.strip().lower() in synonyms:
            return name
    return None


def _resolve_common_var(reader: _TecioReader, key: str) -> str | None:
    """Resolve a ``Common.<Something>Var`` dataset aux item to a variable name.

    Per the Tecplot data format guide, items such as ``Common.XVar`` and
    ``Common.UVar`` hold a 1-based dataset variable number when present. Returns
    ``None`` if *key* is absent from the dataset aux data, isn't a valid integer, or
    is out of range for the dataset -- the caller decides what to fall back to.

    Returns:
        str: Variable name for the item, or ``None`` if it is absent or invalid.
    """
    index = reader.auxdata.as_int(key)
    if index is None:
        return None
    if not 1 <= index <= reader.num_vars:
        print(
            f"[TecplotTecioReader] {key} = {index} is out of range "
            f"[1, {reader.num_vars}]; ignoring."
        )
        return None
    return reader.variables[index - 1]


def _coerce_aux_value(value: str) -> int | float | str:
    """Best-effort convert an aux-data string to ``int`` or ``float``.

    Tecplot itself treats aux data as usable scalars in its own equations, so a value
    that parses cleanly as a number is returned as that type; anything else is left as
    the original string. ``int()`` is tried before ``float()`` so an integer-looking
    value (e.g. the variable number in ``Common.XVar``) becomes a integer rather than a
    ``1.0``-style float.

    Returns:
        str: The value as an ``int`` or ``float`` if it parses as one, else the string.
    """
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _add_aux_field_data(
    field_data: vtkFieldData,
    items: Any,
    key_fn: Any = None,
) -> None:
    """Attach each ``(name, value)`` pair in *items* to *field_data*.

    Each pair becomes a one-element array on *field_data*: a numeric int64/float64
    array (see :func:`_coerce_aux_value`) when the value parses cleanly as a number,
    else a :class:`vtkStringArray`. This is the closest practical VTK equivalent to
    Tecplot's auxiliary data -- there is no first-class "aux data" concept in VTK, but
    field data is visible in ParaView's Information panel and Spreadsheet View, and a
    numeric array additionally works directly with numeric-oriented tools such as the
    Python Calculator and Annotate Attribute Data filter.

    Args:
        field_data: Target field data (root dataset, block, etc.).
        items: Iterable of ``(key, value)`` string pairs, e.g. a dict's ``.items()``.
        key_fn: Optional ``key -> array name`` transform, e.g. to prefix variable-level
            aux data with the variable name.
    """
    for key, value in items:
        name = key_fn(key) if key_fn is not None else key
        coerced = _coerce_aux_value(value)

        if isinstance(coerced, str):
            arr = vtkStringArray()
            arr.SetName(name)
            arr.SetNumberOfValues(1)
            arr.SetValue(0, coerced)
        else:
            dtype = np.int64 if isinstance(coerced, int) else np.float64
            arr = numpy_support.numpy_to_vtk(
                np.array([coerced], dtype=dtype), deep=True
            )
            arr.SetName(name)

        field_data.AddArray(arr)


# --------------------------------------------------------------------------------------
# Face-based connectivity (FEPOLYGON / FEPOLYHEDRON) -> VTK cells
# --------------------------------------------------------------------------------------
#
# A poly zone stores faces, each with a left and right element index (all 0-based,
# -1 for none, <= -2 for a connection to another zone). VTK instead wants every cell to
# list its own nodes (polygon) or faces (polyhedron), so the face map is inverted below
# with array operations only, no per-element Python loops.


def _vtk_id_array(values: npt.ArrayLike) -> Any:
    return numpy_support.numpy_to_vtkIdTypeArray(
        np.ascontiguousarray(values, dtype=np.int64), deep=True
    )


def _vtk_cell_array(
    offsets: npt.ArrayLike, connectivity: npt.ArrayLike
) -> vtkCellArray:
    cells = vtkCellArray()
    cells.SetData(_vtk_id_array(offsets), _vtk_id_array(connectivity))
    return cells


def _cell_types(counts: npt.NDArray[np.int64], cell_type: int) -> Any:
    """Per-cell type array: *cell_type*, or an empty cell for an element with no faces.

    Returns:
        VTK array holding one cell type per element.
    """
    types = np.where(counts > 0, cell_type, VTK_EMPTY_CELL).astype(np.uint8)
    vtk_types = vtkUnsignedCharArray()
    vtk_types.DeepCopy(numpy_support.numpy_to_vtk(types, deep=True))
    return vtk_types


def _chain_undirected(a: npt.NDArray, b: npt.NDArray) -> npt.NDArray[np.int64]:
    """Order the edges ``(a[i], b[i])`` of one polygon into a node loop.

    Used only for the rare element whose edges do not chain by direction. Falls back
    to the edges' own order if they do not form one simple loop.

    Returns:
        NDArray: Node indices of the loop, in order.
    """
    ends = list(zip(a.tolist(), b.tolist(), strict=True))
    at_node: dict[int, list[int]] = {}
    for i, (u, v) in enumerate(ends):
        at_node.setdefault(u, []).append(i)
        at_node.setdefault(v, []).append(i)
    used = [False] * len(ends)
    loop: list[int] = []
    node = ends[0][0]
    for _ in ends:
        edge = next((i for i in at_node.get(node, ()) if not used[i]), None)
        if edge is None:
            break
        used[edge] = True
        loop.append(node)
        u, v = ends[edge]
        node = v if u == node else u
    if len(loop) != len(ends):
        return np.asarray(a, dtype=np.int64).copy()
    return np.asarray(loop, dtype=np.int64)


def _polygon_loops(
    face_nodes: npt.ArrayLike,
    face_left: npt.ArrayLike,
    face_right: npt.ArrayLike,
    num_elements: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], int]:
    """Chain a polygon zone's edges into one node loop per element.

    The left element sees an edge in its stored direction (node 0 to node 1) and the
    right element sees it reversed, so each element's edges link head to tail. Elements
    are handled together: step ``k`` places the ``k``-th node of every element still
    unfinished.

    Args:
        face_nodes: Two nodes per face, flat or shaped ``(num_faces, 2)``.
        face_left: Left element per face (``< 0`` for none).
        face_right: Right element per face (``< 0`` for none or another zone).
        num_elements: Number of elements in the zone.

    Returns:
        ``(offsets, nodes, num_rebuilt)``: CSR offsets (length ``num_elements + 1``)
        and the concatenated loops, plus how many elements needed the direction-free
        rebuild.
    """
    edges = np.asarray(face_nodes, dtype=np.int64).reshape(-1, 2)
    left = np.asarray(face_left, dtype=np.int64)
    right = np.asarray(face_right, dtype=np.int64)
    use_left, use_right = left >= 0, right >= 0
    src = np.concatenate((edges[use_left, 0], edges[use_right, 1]))
    dst = np.concatenate((edges[use_left, 1], edges[use_right, 0]))
    elem = np.concatenate((left[use_left], right[use_right]))
    order = np.argsort(elem, kind="stable")
    src, dst, elem = src[order], dst[order], elem[order]

    counts = np.bincount(elem, minlength=num_elements)
    offsets = np.zeros(num_elements + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    if not src.size:
        return offsets, src, 0

    # Edge i is followed by the edge of the same element that starts where i ends.
    width = int(max(src.max(), dst.max())) + 1
    sort_key = np.argsort(elem * width + src, kind="stable")
    sorted_keys = (elem * width + src)[sort_key]
    wanted = elem * width + dst
    pos = np.minimum(np.searchsorted(sorted_keys, wanted), len(sorted_keys) - 1)
    linked = sorted_keys[pos] == wanted
    following = sort_key[pos]

    nodes = np.empty_like(src)
    bad = np.zeros(num_elements, dtype=bool)
    ids = np.flatnonzero(counts > 0)
    first = offsets[ids]
    cur = first.copy()
    k = 0
    while ids.size:
        nodes[offsets[ids] + k] = src[cur]
        bad[ids[~linked[cur]]] = True
        cur = following[cur]
        k += 1
        done = counts[ids] == k
        closed = cur == first
        bad[ids[done != closed]] = True  # closed too early, or never closed
        keep = ~done
        ids, cur, first = ids[keep], cur[keep], first[keep]

    rebuilt = np.flatnonzero(bad)
    for e in rebuilt:
        lo, hi = offsets[e], offsets[e + 1]
        nodes[lo:hi] = _chain_undirected(src[lo:hi], dst[lo:hi])
    return offsets, nodes, len(rebuilt)


def _element_faces(
    face_nodes: npt.ArrayLike,
    face_node_counts: npt.ArrayLike,
    face_left: npt.ArrayLike,
    face_right: npt.ArrayLike,
    num_elements: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """List the faces of every polyhedral element, each pointing out of its element.

    A face is outward for its left element as stored and for its right element
    reversed, so a shared face appears twice, once per element.

    Returns:
        ``(faces_per_element, node_counts, nodes)``: the face count of each element,
        the node count of each listed face, and all listed faces' nodes concatenated,
        in element order.
    """
    nodes = np.asarray(face_nodes, dtype=np.int64)
    counts = np.asarray(face_node_counts, dtype=np.int64)
    left = np.asarray(face_left, dtype=np.int64)
    right = np.asarray(face_right, dtype=np.int64)
    starts = np.zeros(len(counts), dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])

    use_left, use_right = left >= 0, right >= 0
    face = np.concatenate((np.flatnonzero(use_left), np.flatnonzero(use_right)))
    elem = np.concatenate((left[use_left], right[use_right]))
    reverse = np.concatenate((
        np.zeros(use_left.sum(), dtype=bool),
        np.ones(use_right.sum(), dtype=bool),
    ))
    order = np.argsort(elem, kind="stable")
    face, elem, reverse = face[order], elem[order], reverse[order]

    node_counts = counts[face]
    owner = np.repeat(np.arange(len(face)), node_counts)
    local = np.arange(owner.size) - np.repeat(
        np.cumsum(node_counts) - node_counts, node_counts
    )
    source = np.where(
        reverse[owner],
        starts[face][owner] + node_counts[owner] - 1 - local,
        starts[face][owner] + local,
    )
    faces_per_element = np.bincount(elem, minlength=num_elements)
    return faces_per_element, node_counts, nodes[source]


def _set_polyhedral_cells(
    grid: vtkUnstructuredGrid,
    face_nodes: npt.ArrayLike,
    face_node_counts: npt.ArrayLike,
    face_left: npt.ArrayLike,
    face_right: npt.ArrayLike,
    num_elements: int,
    *,
    legacy_face_stream: bool = False,
) -> None:
    """Fill *grid* with one ``VTK_POLYHEDRON`` cell per element of a face map.

    VTK 9.2+ takes the faces as ``vtkCellArray`` objects; older versions take a
    ``vtkIdTypeArray`` face stream (``[nfaces, n0, ids..., n1, ids..., ...]`` per cell).
    The first form is tried unless *legacy_face_stream* is set (used by tests).
    Elements without faces become empty cells so cell data stays aligned.
    """
    per_element, node_counts, nodes = _element_faces(
        face_nodes, face_node_counts, face_left, face_right, num_elements
    )
    cell_types = _cell_types(per_element, VTK_POLYHEDRON)

    # Unique points of each cell, which VTK stores alongside the faces.
    owner = np.repeat(np.repeat(np.arange(num_elements), per_element), node_counts)
    width = int(nodes.max()) + 1 if nodes.size else 1
    unique = np.unique(owner * width + nodes)
    point_counts = np.bincount(unique // width, minlength=num_elements)
    point_offsets = np.zeros(num_elements + 1, dtype=np.int64)
    np.cumsum(point_counts, out=point_offsets[1:])
    cells = _vtk_cell_array(point_offsets, unique % width)

    face_offsets = np.zeros(len(node_counts) + 1, dtype=np.int64)
    np.cumsum(node_counts, out=face_offsets[1:])
    cell_face_offsets = np.zeros(num_elements + 1, dtype=np.int64)
    np.cumsum(per_element, out=cell_face_offsets[1:])

    if not legacy_face_stream and hasattr(grid, "SetPolyhedralCells"):
        try:
            grid.SetPolyhedralCells(
                cell_types,
                cells,
                _vtk_cell_array(cell_face_offsets, np.arange(len(node_counts))),
                _vtk_cell_array(face_offsets, nodes),
            )
        except TypeError:
            pass
        else:
            return

    # Face stream: per cell [nfaces], then per face [count, ids...].
    face_cell = np.repeat(np.arange(num_elements), per_element)
    token_end = np.zeros(len(node_counts) + 1, dtype=np.int64)
    np.cumsum(node_counts + 1, out=token_end[1:])
    cell_len = 1 + np.diff(token_end[cell_face_offsets])
    cell_start = np.cumsum(cell_len) - cell_len
    stream = np.empty(int(cell_len.sum()), dtype=np.int64)
    stream[cell_start] = per_element
    face_pos = (
        cell_start[face_cell]
        + 1
        + token_end[:-1]
        - token_end[cell_face_offsets[:-1]][face_cell]
    )
    stream[face_pos] = node_counts
    node_face = np.repeat(np.arange(len(node_counts)), node_counts)
    node_pos = (
        face_pos[node_face] + 1 + np.arange(nodes.size) - face_offsets[:-1][node_face]
    )
    stream[node_pos] = nodes
    locations = np.where(per_element > 0, cell_start, -1)
    grid.SetCells(cell_types, cells, _vtk_id_array(locations), _vtk_id_array(stream))


# ======================================================================================
# Reader
# ======================================================================================


@smproxy.reader(
    name="TecplotTecioReader",
    label="Tecplot 360 Reader (TecIO)",
    extensions=_EXTENSIONS,
    file_description=_FILE_DESCRIPTION,
)
class TecplotReader(VTKPythonAlgorithmBase):
    """Reads Tecplot SZL/PLT/DAT files into a ``vtkMultiBlockDataSet``.

    One block per zone; see the module docstring for the full mapping, including how
    auxiliary data becomes field data and how coordinate variables are chosen.
    """

    def __init__(self) -> None:
        """Initialize from VTK object."""
        super().__init__(
            nInputPorts=0, nOutputPorts=1, outputType="vtkMultiBlockDataSet"
        )

        self._filename: str | None = None
        self._reader: _TecioReader | None = None
        self._reader_path: str | None = None
        self._reader_mtime: float | None = None

        self._x_override: str | None = None
        self._y_override: str | None = None
        self._z_override: str | None = None
        self._u_override: str | None = None
        self._v_override: str | None = None
        self._w_override: str | None = None

        self._array_selection = vtkDataArraySelection()
        self._array_selection.AddObserver("ModifiedEvent", _modified_callback(self))

        # Keyed "<0-based zone index>: <title>" -> whether that zone/block loads.
        self._zone_selection = vtkDataArraySelection()
        self._zone_selection.AddObserver("ModifiedEvent", _modified_callback(self))
        self._zone_keys: dict[int, str] = {}

    def __del__(self) -> None:  # pragma: no cover
        """Best-effort cleanup."""
        try:
            self._close_reader()
        except Exception:  # noqa: BLE001 - never raise from __del__
            pass

    # -- File handle management -------------------------------------------------------

    def _close_reader(self) -> None:
        """Release the cached reader handle, if any."""
        if self._reader is not None:
            close = getattr(self._reader, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - closing must not fail the pipeline
                    pass
        self._reader = None
        self._reader_path = None
        self._reader_mtime = None

    def _get_reader(self) -> _TecioReader | None:
        """Return a cached, open ``tecio`` reader for the current file.

        Reopens only when the path or on-disk modification time changes, so the
        frequent, metadata-only pipeline callbacks ParaView issues (array lists, time
        values, ...) don't repeatedly pay the cost of reopening the file.

        Returns:
            An open ``tecio`` reader, or ``None`` if no file has been set.

        Raises:
            RuntimeError: If the ``tecio`` package could not be imported.
            FileNotFoundError: If the configured file can't be found/read.
        """
        if self._filename is None:
            return None
        if tecio is None:
            raise RuntimeError(
                "The 'tecio' package is not importable in ParaView's Python "
                f"environment ({_TECIO_IMPORT_ERROR!r}). Install tecio (and "
                "make sure the TecIO shared library can be located, e.g. via "
                "the TECIO_LIB environment variable) in the Python "
                "interpreter ParaView uses, then reload this plugin."
            )

        try:
            mtime = os.path.getmtime(self._filename)
        except OSError as exc:
            raise FileNotFoundError(
                f"Cannot read Tecplot file: {self._filename}"
            ) from exc

        if (
            self._reader is not None
            and self._reader_path == self._filename
            and self._reader_mtime == mtime
        ):
            return self._reader

        self._close_reader()
        self._reader = tecio.open(self._filename, "r")
        self._reader_path = self._filename
        self._reader_mtime = mtime
        self._sync_selections(self._reader)
        return self._reader

    def _sync_selections(self, reader: _TecioReader) -> None:
        """(Re)populate the array and zone selections for a newly opened file."""
        self._array_selection.RemoveAllArrays()
        for name in reader.variables:
            self._array_selection.AddArray(name)

        self._zone_selection.RemoveAllArrays()
        self._zone_keys.clear()
        for zone in reader.zones:
            title = zone.title or f"Zone {zone.zone_index}"
            key = f"{zone.zone_index}: {title}"
            self._zone_keys[zone.zone_index] = key
            default_enabled = not _is_deactivated_boundary_zone(zone)
            self._zone_selection.AddArray(key, default_enabled)

    # -- FileName --------------------------------------------------------------------

    @smproperty.stringvector(name="FileName")
    @smdomain.filelist()
    @smhint.filechooser(extensions=_EXTENSIONS, file_description=_FILE_DESCRIPTION)
    def SetFileName(self, name: str | None) -> None:
        """Set the path to the Tecplot file.

        Accepts ``.szplt``/``.plt``/``.bin``/``.dat``/``.tec``.
        """
        name = name or None
        if self._filename != name:
            self._filename = name
            self._close_reader()
            self.Modified()

    # -- Array / zone selection --------------------------------------------------------

    @smproperty.dataarrayselection(name="Arrays")
    def GetDataArraySelection(self) -> vtkDataArraySelection:
        """Which dataset variables load as point/cell data arrays.

        Returns:
            vtkDataArraySelection: Selection object for the dataset variables.
        """
        return self._array_selection

    @smproperty.dataarrayselection(name="Zones")
    def GetZoneSelection(self) -> vtkDataArraySelection:
        """Which zones load as blocks -- useful to skip zones in large files.

        Returns:
            vtkDataArraySelection: Selection object for the zones.
        """
        return self._zone_selection

    # -- Coordinate variable overrides ------------------------------------------------

    @smproperty.stringvector(name="VariableInfo", information_only="1")
    def GetVariableInfo(self) -> list[str]:
        """Choices offered by the X/Y/Z array dropdowns.

        Eg ``"(auto)"`` plus every variable.

        Returns:
            list: ``"(auto)"`` followed by the dataset variable names.
        """
        try:
            reader = self._get_reader()
        except Exception:  # noqa: BLE001 - domain refresh must not raise into the UI
            reader = None
        names = list(reader.variables) if reader is not None else []
        return ["(auto)", *names]

    def _set_axis_override(self, axis: str, value: str) -> None:
        resolved = None if value in ("", "(auto)") else value
        attr = f"_{axis}_override"
        if getattr(self, attr) != resolved:
            setattr(self, attr, resolved)
            self.Modified()

    @smproperty.stringvector(
        name="XArray", number_of_elements="1", default_values=["(auto)"]
    )
    @smdomain.xml("""
        <StringListDomain name="list">
          <RequiredProperties>
            <Property name="VariableInfo" function="StringInfo"/>
          </RequiredProperties>
        </StringListDomain>
    """)
    def SetXArray(self, value: str) -> None:
        """Override auto-detection of the X-coordinate variable."""
        self._set_axis_override("x", value)

    @smproperty.stringvector(
        name="YArray", number_of_elements="1", default_values=["(auto)"]
    )
    @smdomain.xml("""
        <StringListDomain name="list">
          <RequiredProperties>
            <Property name="VariableInfo" function="StringInfo"/>
          </RequiredProperties>
        </StringListDomain>
    """)
    def SetYArray(self, value: str) -> None:
        """Override auto-detection of the Y-coordinate variable."""
        self._set_axis_override("y", value)

    @smproperty.stringvector(
        name="ZArray", number_of_elements="1", default_values=["(auto)"]
    )
    @smdomain.xml("""
        <StringListDomain name="list">
          <RequiredProperties>
            <Property name="VariableInfo" function="StringInfo"/>
          </RequiredProperties>
        </StringListDomain>
    """)
    def SetZArray(self, value: str) -> None:
        """Override auto-detection of the Z-coordinate variable (else 2-D: Z=0)."""
        self._set_axis_override("z", value)

    @smproperty.stringvector(
        name="UArray", number_of_elements="1", default_values=["(auto)"]
    )
    @smdomain.xml("""
        <StringListDomain name="list">
          <RequiredProperties>
            <Property name="VariableInfo" function="StringInfo"/>
          </RequiredProperties>
        </StringListDomain>
    """)
    def SetUArray(self, value: str) -> None:
        """Override the vector-array U component (else ``Common.UVar``, if set)."""
        self._set_axis_override("u", value)

    @smproperty.stringvector(
        name="VArray", number_of_elements="1", default_values=["(auto)"]
    )
    @smdomain.xml("""
        <StringListDomain name="list">
          <RequiredProperties>
            <Property name="VariableInfo" function="StringInfo"/>
          </RequiredProperties>
        </StringListDomain>
    """)
    def SetVArray(self, value: str) -> None:
        """Override the vector-array V component (else ``Common.VVar``, if set)."""
        self._set_axis_override("v", value)

    @smproperty.stringvector(
        name="WArray", number_of_elements="1", default_values=["(auto)"]
    )
    @smdomain.xml("""
        <StringListDomain name="list">
          <RequiredProperties>
            <Property name="VariableInfo" function="StringInfo"/>
          </RequiredProperties>
        </StringListDomain>
    """)
    def SetWArray(self, value: str) -> None:
        """Override the vector-array W component (else ``Common.WVar``, if set)."""
        self._set_axis_override("w", value)

    def _resolve_vector_component_names(
        self, reader: _TecioReader
    ) -> tuple[str | None, str | None, str | None]:
        """Pick the dataset variables to use as the U, V, W vector components.

        Unlike coordinates, there's no name-based or positional fallback here. Tecplot
        vector-variable names (velocity or momentum components) vary too much across
        solvers to guess reliably. A component resolves only via a manual override
        (``UArray``/``VArray``/``WArray``) or the file's own
        ``Common.UVar``/``Common.VVar``/``Common.WVar`` dataset aux data; anything left
        unresolved stays ``None`` and is treated as zero when the vector is assembled,
        so a 2-D (U, V only) flow still gets a valid, glyph-able vector.

        Returns:
            tuple: Tuple of variable names for ``(u, v, w)``, with ``None`` for an
                unresolved component.
        """
        names = list(reader.variables)

        def override(axis: str) -> str | None:
            value = getattr(self, f"_{axis}_override")
            return value if value in names else None

        u = override("u") or _resolve_common_var(reader, "Common.UVar")
        v = override("v") or _resolve_common_var(reader, "Common.VVar")
        w = override("w") or _resolve_common_var(reader, "Common.WVar")
        return u, v, w

    @staticmethod
    def _resolve_vector_array_name(reader: _TecioReader) -> str:
        """Return "Velocity" or "Momentum" per Common.VectorVarsAreVelocity.

        Per the Tecplot data format guide: if that dataset aux item is truthy, the
        U/V/W vector is velocity; otherwise Tecplot's own documented default is to
        assume momentum, which this mirrors.
        """
        is_velocity = reader.auxdata.as_bool("Common.VectorVarsAreVelocity")
        return "Velocity" if is_velocity else "Momentum"

    def _resolve_coordinate_names(
        self, reader: _TecioReader
    ) -> tuple[str | None, str | None, str | None]:
        """Pick the dataset variables to use as X, Y, and Z point coordinates.

        Priority order: manual override, then the ``Common.XVar``/``YVar``/``ZVar``
        dataset aux items if the file specifies them, then a name match (see
        :func:`_autodetect_axis`), then -- for X and Y only -- positional fallback to
        the first and second dataset variables, since almost every Tecplot CFD file
        leads with its coordinates. Z has no positional fallback beyond the third
        variable and defaults to all-zero (2-D data) when nothing is found.

        Returns:
            tuple: Tuple of variable names for ``(x, y, z)``, with ``None`` for an
                unresolved axis.
        """
        names = list(reader.variables)

        def override(axis: str) -> str | None:
            value = getattr(self, f"_{axis}_override")
            return value if value in names else None

        x = (
            override("x")
            or _resolve_common_var(reader, "Common.XVar")
            or _autodetect_axis(names, "x")
        )
        y = (
            override("y")
            or _resolve_common_var(reader, "Common.YVar")
            or _autodetect_axis(names, "y")
        )
        z = (
            override("z")
            or _resolve_common_var(reader, "Common.ZVar")
            or _autodetect_axis(names, "z")
        )

        if x is None and names:
            x = names[0]
        if y is None and len(names) > 1:
            y = names[1]
        if z is None and len(names) > 2 and names[2] not in (x, y):
            z = names[2]

        return x, y, z

    # -- Time -----------------------------------------------------------------------

    def _compute_timesteps(self, reader: _TecioReader) -> list[float]:
        """Sorted, distinct solution times across all non-static zones.

        Zones with ``strand_id == 0`` are the standard Tecplot convention for
        static/always-present geometry, so they're excluded from the time axis and
        instead included at every requested time (see :meth:`RequestData`).

        Returns:
            list: Sorted list of solution times in ascending order.
        """
        times = {zone.solution_time for zone in reader.zones if zone.strand_id != 0}
        return sorted(times)

    @smproperty.doublevector(
        name="TimestepValues", information_only="1", si_class="vtkSITimeStepsProperty"
    )
    def GetTimestepValues(self) -> list[float]:
        """Publish the dataset's discrete solution times to ParaView's time UI.

        Returns:
            list: Solution times, or an empty list if no file is loaded.
        """
        try:
            reader = self._get_reader()
        except Exception:  # noqa: BLE001 - domain refresh must not raise into the UI
            return []
        return self._compute_timesteps(reader) if reader is not None else []

    def _get_update_time(self, out_info: Any) -> float | None:
        """Resolve the pipeline's requested time to one of the dataset's own times.

        Returns:
            float: The matching dataset time, or ``None`` if there is no reader or no
                time steps.
        """
        reader = self._reader
        if reader is None:
            return None
        timesteps = self._compute_timesteps(reader)
        if not timesteps:
            return None

        executive = self.GetExecutive()
        if out_info.Has(executive.UPDATE_TIME_STEP()):
            requested = out_info.Get(executive.UPDATE_TIME_STEP())
            result = timesteps[0]
            for t in timesteps:
                if t <= requested + 1e-9:
                    result = t
                else:
                    break
            return result
        return timesteps[0]

    # -- Pipeline: RequestInformation --------------------------------------------------

    def RequestInformation(self, request, inInfoVec, outInfoVec) -> int:
        """Publish the dataset's time steps, if any, before ``RequestData`` runs.

        Returns:
            int: ``1``, the VTK pipeline value for success.
        """
        try:
            reader = self._get_reader()
        except Exception as exc:  # noqa: BLE001 - report and produce empty output
            print(f"[TecplotTecioReader] {exc}")
            return 1
        if reader is None:
            return 1

        executive = self.GetExecutive()
        out_info = outInfoVec.GetInformationObject(0)
        out_info.Remove(executive.TIME_STEPS())
        out_info.Remove(executive.TIME_RANGE())

        timesteps = self._compute_timesteps(reader)
        if timesteps:
            for t in timesteps:
                out_info.Append(executive.TIME_STEPS(), t)
            out_info.Append(executive.TIME_RANGE(), timesteps[0])
            out_info.Append(executive.TIME_RANGE(), timesteps[-1])
        return 1

    # -- Pipeline: RequestData ---------------------------------------------------------

    def RequestData(self, request, inInfoVec, outInfoVec) -> int:
        """Build the ``vtkMultiBlockDataSet`` output for the requested time step.

        Returns:
            int: ``1``, the VTK pipeline value for success.
        """
        output = vtkMultiBlockDataSet.GetData(outInfoVec, 0)

        try:
            reader = self._get_reader()
        except Exception as exc:  # noqa: BLE001 - report and produce empty output
            print(f"[TecplotTecioReader] {exc}")
            return 1
        if reader is None:
            return 1

        requested_time = self._get_update_time(outInfoVec.GetInformationObject(0))
        x_name, y_name, z_name = self._resolve_coordinate_names(reader)
        u_name, v_name, w_name = self._resolve_vector_component_names(reader)
        vector_array_name = self._resolve_vector_array_name(reader)

        _add_aux_field_data(output.GetFieldData(), reader.auxdata.items())
        self._add_variable_aux_field_data(output.GetFieldData(), reader)

        block_index = 0
        for zone in reader.zones:
            key = self._zone_keys.get(zone.zone_index)
            if key is not None and not self._zone_selection.ArrayIsEnabled(key):
                continue
            if (
                requested_time is not None
                and zone.strand_id != 0
                and not math.isclose(zone.solution_time, requested_time, abs_tol=1e-9)
            ):
                continue

            try:
                dataset = self._build_zone_dataset(
                    zone,
                    x_name,
                    y_name,
                    z_name,
                    u_name,
                    v_name,
                    w_name,
                    vector_array_name,
                )
            except _UnsupportedZoneError as exc:
                print(
                    f"[TecplotTecioReader] Skipping zone {zone.zone_index} "
                    f"({zone.title!r}): {exc}"
                )
                continue
            except Exception as exc:  # noqa: BLE001 - one bad zone shouldn't fail the read
                print(
                    f"[TecplotTecioReader] Error reading zone {zone.zone_index} "
                    f"({zone.title!r}), skipping: {exc}"
                )
                continue

            output.SetBlock(block_index, dataset)
            name = zone.title or f"Zone {zone.zone_index}"
            output.GetMetaData(block_index).Set(vtkCompositeDataSet.NAME(), name)
            block_index += 1

        if requested_time is not None:
            output.GetInformation().Set(output.DATA_TIME_STEP(), requested_time)
        return 1

    def _add_variable_aux_field_data(
        self, field_data: vtkFieldData, reader: _TecioReader
    ) -> None:
        """Attach variable-level aux data to *field_data*, prefixed by variable name."""
        names = reader.variables
        for i in range(reader.num_vars):
            aux = reader.get_var_auxdata(i)
            if not len(aux):
                continue
            var_name = names[i]
            _add_aux_field_data(
                field_data, aux.items(), key_fn=lambda k, v=var_name: f"{v}::{k}"
            )

    # -- Zone -> VTK dataset conversion ------------------------------------------------

    def _build_zone_dataset(
        self,
        zone: _TecioZone,
        x_name: str | None,
        y_name: str | None,
        z_name: str | None,
        u_name: str | None,
        v_name: str | None,
        w_name: str | None,
        vector_array_name: str,
    ) -> vtkStructuredGrid | vtkUnstructuredGrid:
        """Convert one Tecplot zone into a VTK type.

        ORDERED zones become a ``vtkStructuredGrid``. Finite-element, polygon, and
        polyhedron zones become a ``vtkUnstructuredGrid``.

        Returns:
            The structured or unstructured grid class for the zone.

        Raises:
            _UnsupportedZoneError: If the zone type is not supported by this reader.
        """
        # By the time any zone is being converted, RequestData has already gone
        # through _get_reader(), which raises RuntimeError if tecio failed to
        # import, so these are never actually None here. Asserted rather than
        # re-raised so ty can narrow the isinstance checks below.
        assert TecplotOrderedZoneReader is not None
        assert TecplotFEZoneReader is not None

        if isinstance(zone, TecplotOrderedZoneReader):
            dataset: vtkStructuredGrid | vtkUnstructuredGrid = (
                self._build_structured_grid(zone, x_name, y_name, z_name)
            )
        elif TecplotPolyZoneReader is not None and isinstance(
            zone, TecplotPolyZoneReader
        ):
            dataset = self._build_poly_grid(zone, x_name, y_name, z_name)
        elif isinstance(zone, TecplotFEZoneReader):
            cell_type = _VTK_CELL_TYPE.get(zone.zone_type)
            if cell_type is None:
                raise _UnsupportedZoneError(
                    f"zone type {zone.zone_type.name} is not yet supported by this "
                    "reader (only ORDERED, classic FE zones -- line, triangle, "
                    "quadrilateral, tetrahedron, brick -- polygon, and polyhedron "
                    "zones are converted; mixed zones are not)"
                )
            dataset = self._build_unstructured_grid(
                zone, cell_type, x_name, y_name, z_name
            )
        else:
            raise _UnsupportedZoneError(
                f"zone type {zone.zone_type.name} is not supported by this reader"
            )

        _add_aux_field_data(dataset.GetFieldData(), zone.auxdata.items())
        self._add_variable_arrays(dataset, zone)
        self._add_vector_array(dataset, zone, vector_array_name, u_name, v_name, w_name)
        return dataset

    def _build_structured_grid(
        self,
        zone: _TecioOrderedZone,
        x_name: str | None,
        y_name: str | None,
        z_name: str | None,
    ) -> vtkStructuredGrid:
        """Build a ``vtkStructuredGrid`` for an ORDERED zone.

        Returns:
            vtkStructuredGrid: Structured grid with the zone points.
        """
        ni, nj, nk = zone.dimensions
        shape = (ni, nj, nk)
        x = self._fetch_ordered_coordinate(zone, x_name, shape)
        y = self._fetch_ordered_coordinate(zone, y_name, shape)
        z = self._fetch_ordered_coordinate(zone, z_name, shape)

        # Tecplot's IJK arrays come back Fortran-ordered (I fastest); raveling the same
        # way lines the flat point list up with VTK's own I-fastest point ordering, so
        # no per-point reindexing is needed.
        points = np.empty((ni * nj * nk, 3), dtype=np.float64)
        points[:, 0] = x.ravel(order="F")
        points[:, 1] = y.ravel(order="F")
        points[:, 2] = z.ravel(order="F")

        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_support.numpy_to_vtk(points, deep=True))

        grid = vtkStructuredGrid()
        grid.SetDimensions(ni, nj, nk)
        grid.SetPoints(vtk_points)
        return grid

    @staticmethod
    def _fetch_ordered_coordinate(
        zone: _TecioOrderedZone, name: str | None, shape: tuple[int, int, int]
    ) -> npt.NDArray[np.float64]:
        """Return an ``(I, J, K)`` nodal coordinate array for *name*, or zeros.

        Raises:
            _UnsupportedZoneError: If the variable is passive or is not nodal.
        """
        if name is None:
            return np.zeros(shape, dtype=np.float64)
        values = zone.get_array(name)
        if values is None:
            raise _UnsupportedZoneError(
                f"coordinate variable '{name}' has no data in this zone (it is passive)"
            )
        if values.shape != shape:
            raise _UnsupportedZoneError(
                f"coordinate variable '{name}' has shape {values.shape}, expected "
                f"nodal shape {shape} -- is it cell-centered?"
            )
        return values.astype(np.float64, copy=False)

    def _build_unstructured_grid(
        self,
        zone: _TecioFEZone,
        cell_type: int,
        x_name: str | None,
        y_name: str | None,
        z_name: str | None,
    ) -> vtkUnstructuredGrid:
        """Build a ``vtkUnstructuredGrid`` for a classic finite-element zone.

        Returns:
            vtkUnstructuredGrid: Unstructured grid with the zone points and cells.

        Raises:
            _UnsupportedZoneError: If the zone has no connectivity.
        """
        vtk_points = self._build_flat_points(zone, x_name, y_name, z_name)

        node_map = zone.node_map
        if node_map is None or node_map.size == 0:
            raise _UnsupportedZoneError("zone has no connectivity (empty node map)")

        n_cells, nodes_per_cell = node_map.shape
        # tecio node maps are 0-based, same as VTK point ids.
        connectivity = node_map.astype(np.int64, copy=False).ravel()
        offsets = np.arange(
            0, (n_cells + 1) * nodes_per_cell, nodes_per_cell, dtype=np.int64
        )

        cell_array = vtkCellArray()
        cell_array.SetData(
            numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
            numpy_support.numpy_to_vtkIdTypeArray(connectivity, deep=True),
        )

        grid = vtkUnstructuredGrid()
        grid.SetPoints(vtk_points)
        grid.SetCells(cell_type, cell_array)
        return grid

    def _build_flat_points(
        self,
        zone: _TecioFEZone,
        x_name: str | None,
        y_name: str | None,
        z_name: str | None,
    ) -> vtkPoints:
        """Return the points of a node-list zone (FE, polygon, or polyhedron)."""
        n_nodes = zone.num_nodes
        x = self._fetch_flat_coordinate(zone, x_name, n_nodes)
        y = self._fetch_flat_coordinate(zone, y_name, n_nodes)
        z = self._fetch_flat_coordinate(zone, z_name, n_nodes)

        points = np.column_stack((x, y, z)).astype(np.float64, copy=False)
        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_support.numpy_to_vtk(points, deep=True))
        return vtk_points

    def _build_poly_grid(
        self,
        zone: Any,
        x_name: str | None,
        y_name: str | None,
        z_name: str | None,
    ) -> vtkUnstructuredGrid:
        """Build a ``vtkUnstructuredGrid`` for a FEPOLYGON or FEPOLYHEDRON zone.

        Returns:
            vtkUnstructuredGrid: Unstructured grid with the zone points and polygon or
                polyhedron cells.

        Raises:
            _UnsupportedZoneError: If the zone has no face connectivity.
        """
        vtk_points = self._build_flat_points(zone, x_name, y_name, z_name)
        num_elements = int(zone.num_elements)
        face_nodes, face_left, face_right = (
            zone.face_nodes,
            zone.face_left,
            zone.face_right,
        )
        if face_nodes is None or len(face_nodes) == 0:
            raise _UnsupportedZoneError("zone has no face connectivity")

        grid = vtkUnstructuredGrid()
        grid.SetPoints(vtk_points)
        if zone.zone_type == ZoneType.FEPOLYGON:
            offsets, nodes, rebuilt = _polygon_loops(
                face_nodes, face_left, face_right, num_elements
            )
            if rebuilt:
                print(
                    f"[TecplotTecioReader] Zone {zone.zone_index} ({zone.title!r}): "
                    f"{rebuilt} polygon(s) did not chain by edge direction and were "
                    "rebuilt without it."
                )
            counts = np.diff(offsets)
            grid.SetCells(
                _cell_types(counts, VTK_POLYGON), _vtk_cell_array(offsets, nodes)
            )
        else:
            _set_polyhedral_cells(
                grid,
                face_nodes,
                zone.face_node_counts,
                face_left,
                face_right,
                num_elements,
            )
        return grid

    @staticmethod
    def _fetch_flat_coordinate(
        zone: _TecioFEZone, name: str | None, n_nodes: int
    ) -> npt.NDArray[np.float64]:
        """Return a flat, length-``n_nodes`` coordinate array for *name*, or zeros.

        Raises:
            _UnsupportedZoneError: If the variable is passive or does not have one value
                per node.
        """
        if name is None:
            return np.zeros(n_nodes, dtype=np.float64)
        values = zone.get_array(name)
        if values is None:
            raise _UnsupportedZoneError(
                f"coordinate variable '{name}' has no data in this zone (it is passive)"
            )
        if values.size != n_nodes:
            raise _UnsupportedZoneError(
                f"coordinate variable '{name}' has {values.size} values, expected "
                f"{n_nodes} nodal values"
            )
        return values.astype(np.float64, copy=False)

    def _add_variable_arrays(
        self, dataset: vtkStructuredGrid | vtkUnstructuredGrid, zone: _TecioZone
    ) -> None:
        """Attach every selected, active variable to point or cell data.

        Node-located (``NODAL``) variables become point data; cell-located
        (``CELL_CENTERED``) variables become cell data. A variable whose array length
        doesn't match the dataset's point/cell count is skipped rather than raising, so
        one bad variable doesn't take an otherwise-good zone down with it.
        """
        n_points = dataset.GetNumberOfPoints()
        n_cells = dataset.GetNumberOfCells()

        for var in zone.variables:
            if not self._array_selection.ArrayIsEnabled(var.name):
                continue
            values = var.values
            if values is None:  # passive, or genuinely no data
                continue

            flat = values.ravel(order="F") if values.ndim == 3 else np.ravel(values)
            vtk_array = numpy_support.numpy_to_vtk(
                np.ascontiguousarray(flat), deep=True
            )
            vtk_array.SetName(var.name)

            if var.value_location == ValueLocation.NODAL:
                if flat.size != n_points:
                    continue
                dataset.GetPointData().AddArray(vtk_array)
            else:
                if flat.size != n_cells:
                    continue
                dataset.GetCellData().AddArray(vtk_array)

    def _add_vector_array(
        self,
        dataset: vtkStructuredGrid | vtkUnstructuredGrid,
        zone: _TecioZone,
        array_name: str,
        u_name: str | None,
        v_name: str | None,
        w_name: str | None,
    ) -> None:
        """Assemble the U/V/W components into one 3-component vector array.

        Skipped entirely if none of U/V/W resolved for the dataset. A component that did
        resolve for the dataset but is passive or absent in *this specific* zone is
        treated as zero for that zone, same as an axis that never resolved at all. All
        contributing components must share one value location (``NODAL`` or
        ``CELL_CENTERED``) and array length; a mismatch skips vector assembly for this
        zone with a printed warning, since the rest of the zone is still good.
        """
        if u_name is None and v_name is None and w_name is None:
            return

        location: ValueLocation | None = None
        flat_components: list[npt.NDArray[np.float64] | None] = []
        for name in (u_name, v_name, w_name):
            if name is None:
                flat_components.append(None)
                continue
            var = zone.variables[name]
            values = var.values
            if values is None:  # passive, or absent in this specific zone
                flat_components.append(None)
                continue

            flat = values.ravel(order="F") if values.ndim == 3 else np.ravel(values)
            if location is None:
                location = var.value_location
            elif var.value_location != location:
                print(
                    f"[TecplotTecioReader] Zone {zone.zone_index} ({zone.title!r}): "
                    f"'{name}' has a different value location than the other "
                    "vector components; skipping the vector for this zone."
                )
                return
            flat_components.append(flat)

        if location is None:
            return  # every resolved component was passive/absent in this zone

        n_tuples = (
            dataset.GetNumberOfPoints()
            if location == ValueLocation.NODAL
            else dataset.GetNumberOfCells()
        )
        zeros = np.zeros(n_tuples, dtype=np.float64)
        columns = []
        for flat in flat_components:
            if flat is None:
                columns.append(zeros)
            elif flat.size != n_tuples:
                print(
                    f"[TecplotTecioReader] Zone {zone.zone_index} ({zone.title!r}): "
                    f"a vector component has {flat.size} values, expected "
                    f"{n_tuples}; skipping the vector for this zone."
                )
                return
            else:
                columns.append(flat.astype(np.float64, copy=False))

        vector = np.ascontiguousarray(np.column_stack(columns))
        vtk_array = numpy_support.numpy_to_vtk(vector, deep=True)
        vtk_array.SetName(array_name)

        if location == ValueLocation.NODAL:
            dataset.GetPointData().AddArray(vtk_array)
        else:
            dataset.GetCellData().AddArray(vtk_array)
