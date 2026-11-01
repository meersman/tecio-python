"""ParaView writer plugin for Tecplot 360 data files, built on ``tecio``.

The counterpart to ``TecplotTecioReader``: writes whatever is on the input port out as
a Tecplot SZL/PLT/DAT file, so a pipeline (or a filter's output) can be exported without
going through ParaView's own, more limited Tecplot writer. As with the reader, the
output format is chosen by the saved file's extension.

Output structure
    The input may be a single ``vtkDataSet`` or a ``vtkMultiBlockDataSet``; every leaf
    block becomes one Tecplot zone, named after the block's composite-data name (or
    ``"Zone <n>"``, counting from 0, if unnamed). Geometry is mapped per block:

    * ``vtkStructuredGrid``, ``vtkRectilinearGrid``, and ``vtkImageData`` become
      ``ORDERED`` (IJK) zones.
    * ``vtkPolyData`` and ``vtkUnstructuredGrid`` (and any other dataset exposing
      explicit points and cells) become finite-element zones, one per VTK cell type
      present: line, triangle, quadrilateral, tetrahedron, and hexahedron cells give
      classic ``FELINESEG``/``FETRIANGLE``/``FEQUADRILATERAL``/``FETETRAHEDRON``/
      ``FEBRICK`` zones, ``VTK_POLYGON`` cells give an ``FEPOLYGON`` zone, and
      ``VTK_POLYHEDRON`` cells give an ``FEPOLYHEDRON`` zone. A block mixing cell types
      (e.g. a surface with both triangles and polygons) is split into one zone per
      type, each zone keeping only the points its own cells reference.
    * Any other cell type (triangle strip, vertex, wedge, pyramid, ...) is not
      representable by these zone types and is skipped with a printed warning -- the
      rest of the block is still written.
    * Polygon and polyhedron zones are stored face by face, each face with the one or
      two elements on its sides, so the faces are matched up between neighboring
      cells. Polyhedron faces are turned to point out of their cell (using the sign of
      its volume), and polygons are turned counterclockwise when all the points have
      the same ``z``. Polygon and polyhedron zones can be saved to ``.plt`` and
      ``.dat`` files, not yet ``.szplt``.

Coordinates
    The first three dataset variables are always named ``x``, ``y``, ``z`` and
    carry each zone's point coordinates (``z`` is all-zero for 2-D geometry). The
    dataset aux items ``Common.XVar``/``YVar``/``ZVar`` are set to ``1``/``2``/``3``
    so this plugin's own reader (and Tecplot 360 itself) auto-detect them on reopen.

Variables
    Every point-data and cell-data array across every block contributes to one
    dataset-wide variable list (in first-seen order, after the three coordinates); a
    block missing a given array has that variable marked passive for its own zone(s).
    A multi-component array (a vector or tensor) has no scalar equivalent in classic
    Tecplot zones, so each component becomes its own scalar variable: 2- and
    3-component arrays use ``"<name>_X"``/``"_Y"``/``"_Z"`` suffixes, matching how
    ``TecplotTecioReader`` reassembles ``Common.UVar``/``VVar``/``WVar`` into one
    vector array; wider arrays fall back to ``"<name>_0"``, ``"<name>_1"``, etc. An
    array named exactly ``"Velocity"`` or ``"Momentum"`` additionally gets
    ``Common.UVar``/``VVar``/``WVar`` (and, for ``"Velocity"``,
    ``Common.VectorVarsAreVelocity``) pointing at its decomposed components, so it
    round-trips back into one vector array when reopened through this plugin's reader.

Auxiliary data
    The input's field data has no first-class Tecplot equivalent, so it is mapped the
    same way ``TecplotTecioReader`` reads it back, in reverse: the root dataset's
    field data becomes dataset-level aux data (a ``"<variable name>::<key>"`` array
    name is split back into variable-level aux for that variable, if the name
    resolves), and each block's own field data becomes that zone's aux data. Only the
    first value of a multi-valued field-data array is kept, since Tecplot aux items
    are single strings.

Known limitations (first pass)
    * No coordinate, connectivity, or variable sharing is attempted between zones.
      Zone-to-zone sharing (as used elsewhere in this project, e.g. ``tecmerge``'s grid
      sharing) would reduce output size for such pipelines, but is not implemented here.
    * This plugin writes one static snapshot of its input. It does not loop over the
      pipeline's time steps itself; ParaView's own "write a file per time step"
      animation/file-series mechanism, where supported by the installed ParaView
      version, is the way to export a transient pipeline through this writer.
    * Triangle-strip, vertex, wedge, and pyramid cells are skipped rather than
      approximated. A polygon edge or polyhedron face shared by more than two cells
      cannot be stored, and the cells of that type in that block are skipped.
    * Points not referenced by any kept cell are dropped from their zone rather than
      written as isolated, cell-less nodes (classic Tecplot FE zones have no concept
      of an unused node).

Indexing
    ``tecio``'s Python API is 0-based, so an unnamed block is titled ``"Zone <n>"``
    counting from 0 and variable-level aux data is keyed by 0-based variable index.
    Values that Tecplot defines as 1-based in the file (``Common.XVar``/``YVar``/
    ``ZVar``/``UVar``/``VVar``/``WVar``) are still written as 1-based variable numbers.

Requirements
    The ``tecio`` package must be importable from the same Python interpreter ParaView
    uses to run this plugin.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import numpy.typing as npt
from paraview.util.vtkAlgorithm import smdomain, smproperty, smproxy
from vtkmodules.util import numpy_support
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonCore import vtkDataArray, vtkIdList, vtkStringArray
from vtkmodules.vtkCommonDataModel import (
    VTK_HEXAHEDRON,
    VTK_LINE,
    VTK_POLYGON,
    VTK_POLYHEDRON,
    VTK_QUAD,
    VTK_TETRA,
    VTK_TRIANGLE,
    vtkCompositeDataSet,
    vtkDataObject,
    vtkDataSet,
    vtkFieldData,
    vtkImageData,
    vtkMultiBlockDataSet,
    vtkPointSet,
    vtkRectilinearGrid,
    vtkStructuredGrid,
    vtkUnstructuredGrid,
)
from vtkmodules.vtkFiltersCore import vtkAppendFilter

# Same escape hatch as TecplotTecioReader: ParaView's bundled Python can't have tecio
# installed into its own write-protected site-packages, so point at a local install
# via TECIO_PYTHONPATH (e.g. "/Users/you/tecio-for-paraview") if needed.
_TECIO_FALLBACK_PATH = os.environ.get("TECIO_PYTHONPATH", "")
if _TECIO_FALLBACK_PATH and _TECIO_FALLBACK_PATH not in sys.path:
    sys.path.insert(0, _TECIO_FALLBACK_PATH)

try:
    import tecio
    from tecio import FileType, ValueLocation, ZoneType

    _TECIO_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment-dependent  # noqa: BLE001
    tecio = None  # ty: ignore[invalid-assignment]
    FileType = ValueLocation = ZoneType = None  # ty: ignore[invalid-assignment]
    _TECIO_IMPORT_ERROR = exc

# --------------------------------------------------------------------------------------
# Module-level constants and small helpers
# --------------------------------------------------------------------------------------

_EXTENSIONS = "szplt plt bin dat tec"
_FILE_DESCRIPTION = "Tecplot 360 Data Files"

# Maps a VTK cell type to the Tecplot zone type it becomes. Fixed-size, single-shape
# cells become classic finite-element zones, polygons and polyhedra become face-based
# zones; anything else is skipped.
_ZONE_TYPE_FOR_VTK_CELL: dict[int, Any] = (
    {
        VTK_LINE: ZoneType.FELINESEG,
        VTK_TRIANGLE: ZoneType.FETRIANGLE,
        VTK_QUAD: ZoneType.FEQUADRILATERAL,
        VTK_TETRA: ZoneType.FETETRAHEDRON,
        VTK_HEXAHEDRON: ZoneType.FEBRICK,
        VTK_POLYGON: ZoneType.FEPOLYGON,
        VTK_POLYHEDRON: ZoneType.FEPOLYHEDRON,
    }
    if tecio is not None
    else {}
)

# Number of nodes per cell of the classic types
_NODES_PER_CELL: dict[int, int] = {
    VTK_LINE: 2,
    VTK_TRIANGLE: 3,
    VTK_QUAD: 4,
    VTK_TETRA: 4,
    VTK_HEXAHEDRON: 8,
}

_COMPONENT_SUFFIXES: dict[int, tuple[str, ...]] = {2: ("X", "Y"), 3: ("X", "Y", "Z")}

# Arrays with one of these names are additionally advertised as a Common.*Var vector,
# matching how TecplotTecioReader names the vector it assembles on read.
_VECTOR_ARRAY_NAMES = frozenset({"Velocity", "Momentum"})


def _component_name(base_name: str, num_components: int, component: int) -> str:
    """Return the scalar variable name for one component of a multi-component array.

    Classic Tecplot zones only carry scalar variables, so each component of a vector
    or tensor array becomes its own variable. 2- and 3-component arrays use the
    conventional ``_X``/``_Y``/``_Z`` suffixes; wider arrays fall back to a numeric
    ``_<i>`` suffix.
    """
    if num_components == 1:
        return base_name
    suffixes = _COMPONENT_SUFFIXES.get(num_components)
    suffix = suffixes[component] if suffixes is not None else str(component)
    return f"{base_name}_{suffix}"


def _coerce_field_value(arr: Any) -> str | None:
    """Return the first value of a one-column field-data array, as a string.

    Mirrors ``TecplotTecioReader``'s ``_coerce_aux_value`` in reverse: a
    :class:`vtkStringArray` yields its own string, a numeric array yields ``int`` or
    ``float`` formatting depending on its dtype. Returns ``None`` for an empty array.
    """
    if isinstance(arr, vtkStringArray):
        return arr.GetValue(0) if arr.GetNumberOfValues() >= 1 else None

    values = numpy_support.vtk_to_numpy(arr)
    if values.size == 0:
        return None
    value = values.reshape(values.shape[0], -1)[0, 0]
    if np.issubdtype(values.dtype, np.integer):
        return str(int(value))
    return str(float(value))


def _field_data_to_aux(field_data: vtkFieldData) -> dict[str, str]:
    """Convert a ``vtkFieldData``'s arrays into a Tecplot-style ``{name: value}`` dict.

    Only the first value of each array is kept, since Tecplot aux items are single
    strings; an unnamed array is skipped.

    Returns:
        dict: Mapping of array name to its first value as a string.
    """
    aux: dict[str, str] = {}
    for i in range(field_data.GetNumberOfArrays()):
        arr = field_data.GetAbstractArray(i)
        name = arr.GetName()
        if not name:
            continue
        value = _coerce_field_value(arr)
        if value is not None:
            aux[name] = value
    return aux


def _split_variable_aux(
    root_aux: dict[str, str], var_names: list[str]
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Split root field data's aux dict into dataset-level and per-variable aux.

    ``TecplotTecioReader`` writes variable-level aux back to the root field data as
    ``"<variable name>::<key>"``; this reverses that only for names whose prefix
    matches a real variable, so ordinary dataset aux (which may happen to contain
    "::") is left alone.

    Returns:
        tuple: Tuple of the dataset-level aux dict and the per-variable aux dicts, keyed by
            variable name.
    """
    dataset_aux: dict[str, str] = {}
    var_aux: dict[str, dict[str, str]] = {}
    var_name_set = set(var_names)
    for key, value in root_aux.items():
        prefix, sep, rest = key.partition("::")
        if sep and prefix in var_name_set:
            var_aux.setdefault(prefix, {})[rest] = value
        else:
            dataset_aux[key] = value
    return dataset_aux, var_aux


# --------------------------------------------------------------------------------------
# Per-zone intermediate representation
# --------------------------------------------------------------------------------------


class _ZoneBuild:
    """Everything needed to emit one Tecplot zone via the ``tecio`` Write API.

    A plain class rather than a dataclass: ParaView executes a plugin file before
    registering it in ``sys.modules``, and ``dataclasses`` looks the defining module up
    there, so a ``@dataclass`` makes the whole plugin fail to load.

    Attributes:
        title: Zone title.
        zone_type: ``ZoneType`` of a finite-element or face-based zone, or ``None``
            for an ORDERED zone.
        dims: ``(imax, jmax, kmax)``, ORDERED zones only.
        node_map: 0-based node map, classic finite-element zones only.
        face_map: Keyword arguments for ``write_poly_zone`` that describe the faces,
            polygon and polyhedron zones only.
        arrays: Variable name -> ``(values, value_location)``.
        aux: Zone-level auxiliary data.
    """

    def __init__(
        self,
        title: str,
        zone_type: Any | None,
        dims: tuple[int, int, int] | None = None,
        node_map: npt.NDArray[np.int64] | None = None,
        face_map: dict[str, npt.NDArray[np.int64]] | None = None,
    ) -> None:
        self.title = title
        self.zone_type = zone_type
        self.dims = dims
        self.node_map = node_map
        self.face_map = face_map
        self.arrays: dict[str, tuple[npt.NDArray, Any]] = {}
        self.aux: dict[str, str] = {}


def _decompose_array(arr: Any, flatten: Any) -> list[tuple[str, npt.NDArray]] | None:
    """Return ``[(component_name, flat_values), ...]`` for one VTK data array.

    *flatten* converts the array's natural VTK layout into the flat, per-node or
    per-cell order the caller needs (a no-op ``vtk_to_numpy`` for unstructured data,
    an IJK reshape for structured data). Returns ``None`` for an unnamed array.
    """
    name = arr.GetName()
    if not name:
        return None
    values = numpy_support.vtk_to_numpy(arr)
    num_components = 1 if values.ndim == 1 else values.shape[1]
    return [
        (
            _component_name(name, num_components, c),
            flatten(values if num_components == 1 else values[:, c]),
        )
        for c in range(num_components)
    ]


def _structured_coordinates(
    dataset: vtkImageData | vtkRectilinearGrid | vtkStructuredGrid,
    dims: tuple[int, int, int],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return per-axis ``(ni, nj, nk)`` nodal coordinate arrays for a structured block."""
    ni, nj, nk = dims

    if isinstance(dataset, vtkStructuredGrid):
        points = numpy_support.vtk_to_numpy(dataset.GetPoints().GetData())
        x = points[:, 0].reshape(dims, order="F")
        y = points[:, 1].reshape(dims, order="F")
        z = points[:, 2].reshape(dims, order="F")
        return (
            x.astype(np.float64, copy=False),
            y.astype(np.float64, copy=False),
            z.astype(np.float64, copy=False),
        )

    if isinstance(dataset, vtkRectilinearGrid):
        xs = numpy_support.vtk_to_numpy(dataset.GetXCoordinates())
        ys = numpy_support.vtk_to_numpy(dataset.GetYCoordinates())
        zs = numpy_support.vtk_to_numpy(dataset.GetZCoordinates())
    else:  # vtkImageData
        ox, oy, oz = dataset.GetOrigin()
        sx, sy, sz = dataset.GetSpacing()
        xs = ox + sx * np.arange(ni, dtype=np.float64)
        ys = oy + sy * np.arange(nj, dtype=np.float64)
        zs = oz + sz * np.arange(nk, dtype=np.float64)

    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    return (
        x.astype(np.float64, copy=False),
        y.astype(np.float64, copy=False),
        z.astype(np.float64, copy=False),
    )


def _structured_dims(
    dataset: vtkImageData | vtkRectilinearGrid | vtkStructuredGrid,
) -> tuple[int, int, int]:
    """Return ``(imax, jmax, kmax)``, via the output-argument overload.

    Some VTK Python binding versions only expose ``GetDimensions(dims: list)`` (no
    zero-argument, tuple-returning overload) for these classes, so the output-argument
    form is used unconditionally for portability.
    """
    dims = [0, 0, 0]
    dataset.GetDimensions(dims)
    return (dims[0], dims[1], dims[2])


def _build_ordered_zone(
    title: str, dataset: vtkImageData | vtkRectilinearGrid | vtkStructuredGrid
) -> _ZoneBuild:
    """Build an ORDERED zone from a structured VTK dataset.

    Returns:
        The ordered zone description.
    """
    dims = _structured_dims(dataset)
    x, y, z = _structured_coordinates(dataset, dims)
    cell_dims = tuple(max(n - 1, 1) for n in dims)

    zone = _ZoneBuild(title, None, dims=dims)
    zone.arrays["x"] = (x, ValueLocation.NODAL)
    zone.arrays["y"] = (y, ValueLocation.NODAL)
    zone.arrays["z"] = (z, ValueLocation.NODAL)

    def nodal(values: npt.NDArray) -> npt.NDArray:
        return values.reshape(dims, order="F")

    def cell_centered(values: npt.NDArray) -> npt.NDArray:
        return values.reshape(cell_dims, order="F")

    point_data, cell_data = dataset.GetPointData(), dataset.GetCellData()
    for i in range(point_data.GetNumberOfArrays()):
        for name, values in _decompose_array(point_data.GetArray(i), nodal) or []:
            zone.arrays[name] = (values, ValueLocation.NODAL)
    for i in range(cell_data.GetNumberOfArrays()):
        for name, values in (
            _decompose_array(cell_data.GetArray(i), cell_centered) or []
        ):
            zone.arrays[name] = (values, ValueLocation.CELL_CENTERED)

    zone.aux = _field_data_to_aux(dataset.GetFieldData())
    return zone


def _as_unstructured_grid(dataset: vtkPointSet) -> vtkUnstructuredGrid:
    """Return *dataset* as a ``vtkUnstructuredGrid`` (converted if it is not one)."""
    if isinstance(dataset, vtkUnstructuredGrid):
        return dataset
    append = vtkAppendFilter()  # keeps every point and the cell order
    append.AddInputData(dataset)
    append.Update()
    return append.GetOutput()


def _cell_type_array(grid: vtkUnstructuredGrid) -> npt.NDArray[np.int64]:
    """Return the VTK cell type of every cell of *grid*."""
    types: Any = None
    get_types = getattr(grid, "GetCellTypes", None)  # VTK 9.6+, no arguments
    if get_types is not None:
        try:
            types = get_types()
        except TypeError:
            types = None
    if not isinstance(types, vtkDataArray):
        types = grid.GetCellTypesArray()
    if types is None:  # pragma: no cover - defensive, depends on VTK version
        return np.array(
            [grid.GetCellType(i) for i in range(grid.GetNumberOfCells())],
            dtype=np.int64,
        )
    return numpy_support.vtk_to_numpy(types).astype(np.int64, copy=False)


def _cell_connectivity(
    grid: vtkUnstructuredGrid,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return the CSR ``(offsets, point_ids)`` of *grid*'s cells."""
    cells = grid.GetCells()
    offsets = numpy_support.vtk_to_numpy(cells.GetOffsetsArray())
    connectivity = numpy_support.vtk_to_numpy(cells.GetConnectivityArray())
    return offsets.astype(np.int64, copy=False), connectivity.astype(
        np.int64, copy=False
    )


def _gather_ranges(
    starts: npt.NDArray[np.int64], counts: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Return ``concat(arange(s, s + c) for s, c in zip(starts, counts))``."""
    total = int(counts.sum())
    first = np.cumsum(counts) - counts
    return np.arange(total) - np.repeat(first - starts, counts)


def _reverse_ranges(
    starts: npt.NDArray[np.int64], counts: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Return ``concat(arange(s + c - 1, s - 1, -1))`` for each ``(s, c)``."""
    total = int(counts.sum())
    first = np.cumsum(counts) - counts
    return np.repeat(starts + counts - 1 + first, counts) - np.arange(total)


def _csr_offsets(counts: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets


def _next_in_loop(offsets: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Return, for each node of CSR loops, the position of the next node in its loop.

    Returns:
        NDArray: Position of the next node in the loop for every node.
    """
    following = np.arange(offsets[-1]) + 1
    nonempty = offsets[1:] > offsets[:-1]
    following[offsets[1:][nonempty] - 1] = offsets[:-1][nonempty]
    return following


def _polygon_face_map(
    offsets: npt.NDArray[np.int64],
    nodes: npt.NDArray[np.int64],
    points: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.int64]]:
    """Build the edge-based face map of a polygon zone from its node loops.

    Every polygon edge shared by two cells becomes one face whose left element is the
    first cell that uses it and whose right element is the other cell. When all points
    share one ``z`` the loops are first turned counterclockwise, which makes the left
    element really lie on the left of the edge's direction; otherwise each loop keeps
    the winding it has.

    Args:
        offsets: CSR offsets of each cell's loop in *nodes*.
        nodes: Concatenated 0-based node loops.
        points: ``(n, 3)`` coordinates indexed by node.

    Returns:
        dict:``face_nodes`` (flat, two per face), ``face_left_elements``, and
            ``face_right_elements`` (``-1`` for none), in terms of the node ids in
            *nodes*.

    Raises:
        ValueError: If an edge is used by more than two cells.
    """
    counts = np.diff(offsets)
    cell = np.repeat(np.arange(len(counts)), counts)
    if len(points) and np.ptp(points[:, 2]) == 0.0:
        x, y = points[nodes, 0], points[nodes, 1]
        following = _next_in_loop(offsets)
        area = np.bincount(cell, weights=x * y[following] - x[following] * y)
        flip = (area < 0)[cell]
        index = np.where(
            flip, _reverse_ranges(offsets[:-1], counts), np.arange(nodes.size)
        )
        nodes = nodes[index]

    following = _next_in_loop(offsets)
    a, b = nodes, nodes[following]
    width = int(nodes.max()) + 1 if nodes.size else 1
    key = np.minimum(a, b) * width + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    starts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
    sizes = np.diff(np.r_[starts, len(order)])
    if sizes.size and sizes.max() > 2:
        raise ValueError("a polygon edge is shared by more than two cells")

    left_edge = order[starts]
    right_edge = np.where(sizes == 2, order[np.minimum(starts + 1, len(order) - 1)], -1)
    arrange = np.argsort(left_edge, kind="stable")
    left_edge, right_edge = left_edge[arrange], right_edge[arrange]
    return {
        "face_nodes": np.column_stack((a[left_edge], b[left_edge])).ravel(),
        "face_left_elements": cell[left_edge],
        "face_right_elements": np.where(
            right_edge >= 0, cell[np.maximum(right_edge, 0)], -1
        ),
    }


def _polyhedron_cell_faces(
    grid: vtkUnstructuredGrid, cell_ids: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return the faces of the polyhedral cells *cell_ids* of *grid*.

    Returns:
        ``(faces_per_cell, nodes_per_face, nodes)`` with the faces of each cell listed
        in cell order and each face's nodes concatenated in ``nodes``.
    """
    faces = getattr(grid, "GetPolyhedronFaces", lambda: None)()
    locations = getattr(grid, "GetPolyhedronFaceLocations", lambda: None)()
    if faces is not None and locations is not None and locations.GetNumberOfCells():
        face_off = numpy_support.vtk_to_numpy(faces.GetOffsetsArray()).astype(np.int64)
        face_conn = numpy_support.vtk_to_numpy(faces.GetConnectivityArray()).astype(
            np.int64
        )
        loc_off = numpy_support.vtk_to_numpy(locations.GetOffsetsArray()).astype(
            np.int64
        )
        loc_conn = numpy_support.vtk_to_numpy(locations.GetConnectivityArray()).astype(
            np.int64
        )
        per_cell = loc_off[cell_ids + 1] - loc_off[cell_ids]
        face_ids = loc_conn[_gather_ranges(loc_off[cell_ids], per_cell)]
        per_face = face_off[face_ids + 1] - face_off[face_ids]
        return (
            per_cell,
            per_face,
            face_conn[_gather_ranges(face_off[face_ids], per_face)],
        )

    # Older VTK: one face stream, [nfaces, n0, ids..., n1, ids..., ...], per cell.
    per_cell = np.zeros(len(cell_ids), dtype=np.int64)
    per_face_list: list[int] = []
    node_list: list[int] = []
    stream = vtkIdList()
    for i, cell_id in enumerate(cell_ids.tolist()):
        grid.GetFaceStream(cell_id, stream)
        ids = [stream.GetId(j) for j in range(stream.GetNumberOfIds())]
        per_cell[i] = ids[0]
        pos = 1
        for _ in range(ids[0]):
            n = ids[pos]
            per_face_list.append(n)
            node_list.extend(ids[pos + 1 : pos + 1 + n])
            pos += n + 1
    return (
        per_cell,
        np.asarray(per_face_list, dtype=np.int64),
        np.asarray(node_list, dtype=np.int64),
    )


def _polyhedron_face_map(
    per_cell: npt.NDArray[np.int64],
    per_face: npt.NDArray[np.int64],
    nodes: npt.NDArray[np.int64],
    points: npt.NDArray[np.float64],
) -> dict[str, npt.NDArray[np.int64]]:
    """Build the face map of a polyhedron zone from each cell's own face list.

    A face listed by two cells is stored once, with the first cell as its left element
    (the face's right-hand-rule normal points out of the left element) and the second
    as its right element. A cell whose faces enclose a negative volume is listed
    inside out in VTK, and is turned around first so every face points outward.

    Args:
        per_cell: Number of faces of each cell.
        per_face: Number of nodes of each listed face.
        nodes: Concatenated 0-based nodes of the listed faces.
        points: ``(n, 3)`` coordinates indexed by node.

    Returns:
        dict: ``face_nodes`` (flat), ``face_node_counts``, ``face_left_elements`` and
            ``face_right_elements`` (``-1`` for none), in terms of the node ids in
            *nodes*.

    Raises:
        ValueError: If a face is listed by more than two cells.
    """
    cell = np.repeat(np.arange(len(per_cell)), per_cell)
    face_start = _csr_offsets(per_face)[:-1]

    # Signed volume of each cell from its faces, by the divergence theorem.
    owner = np.repeat(np.arange(len(per_face)), per_face)
    following = _next_in_loop(_csr_offsets(per_face))
    p = points[nodes] - points[nodes].mean(axis=0)
    area = 0.5 * np.add.reduceat(np.cross(p, p[following]), face_start, axis=0)
    volume = np.bincount(cell, weights=np.einsum("ij,ij->i", p[face_start], area) / 3.0)
    flip = (volume < 0)[cell]
    if flip.any():
        index = np.where(
            flip[owner],
            _reverse_ranges(face_start, per_face),
            np.arange(nodes.size),
        )
        nodes = nodes[index]

    # Match each face with the same node set listed by another cell.
    left = np.empty(0, dtype=np.int64)
    right = np.empty(0, dtype=np.int64)
    for size in np.unique(per_face):
        listed = np.flatnonzero(per_face == size)
        rows = nodes[face_start[listed][:, None] + np.arange(size)]
        rows.sort(axis=1)
        keys = (
            np
            .ascontiguousarray(rows)
            .view(np.dtype((np.void, rows.dtype.itemsize * int(size))))
            .ravel()
        )
        _, inverse, repeats = np.unique(keys, return_inverse=True, return_counts=True)
        if repeats.max() > 2:
            raise ValueError("a polyhedron face is shared by more than two cells")
        order = np.argsort(inverse, kind="stable")
        group = np.flatnonzero(np.r_[True, np.diff(inverse[order]) != 0])
        first = order[group]
        second = np.where(
            repeats == 2, order[np.minimum(group + 1, len(order) - 1)], -1
        )
        left = np.concatenate((left, listed[first]))
        right = np.concatenate((
            right,
            np.where(second >= 0, listed[np.maximum(second, 0)], -1),
        ))

    arrange = np.argsort(left, kind="stable")
    left, right = left[arrange], right[arrange]
    counts = per_face[left]
    return {
        "face_nodes": nodes[_gather_ranges(face_start[left], counts)],
        "face_node_counts": counts,
        "face_left_elements": cell[left],
        "face_right_elements": np.where(right >= 0, cell[np.maximum(right, 0)], -1),
    }


def _build_fe_zones(title: str, dataset: vtkPointSet) -> list[_ZoneBuild]:
    """Build one finite-element or face-based zone per supported cell type in *dataset*.

    A block containing only one supported cell type yields a single zone named
    *title*; a block mixing types (e.g. triangles and polygons on one surface) yields
    one zone per type, suffixed with the Tecplot zone type name so each is
    identifiable. Unsupported cell types (triangle strip, vertex, wedge, pyramid,
    ...) are skipped with a printed warning.

    Returns:
        list: The zones, one per supported cell type.
    """
    grid = _as_unstructured_grid(dataset)
    types = _cell_type_array(grid)
    offsets, connectivity = _cell_connectivity(grid)
    points = numpy_support.vtk_to_numpy(grid.GetPoints().GetData())
    point_data, cell_data = grid.GetPointData(), grid.GetCellData()
    field_aux = _field_data_to_aux(dataset.GetFieldData())

    present, first_seen = np.unique(types, return_index=True)
    present = present[np.argsort(first_seen)]  # zones in order of first appearance
    supported = [int(t) for t in present if int(t) in _ZONE_TYPE_FOR_VTK_CELL]
    for vtk_type in sorted(int(t) for t in present if int(t) not in supported):
        print(
            f"[TecplotTecioWriter] '{title}': skipping {int((types == vtk_type).sum())} "
            f"cell(s) of unsupported VTK cell type {vtk_type} (only line, triangle, "
            "quadrilateral, tetrahedron, hexahedron, polygon, and polyhedron cells "
            "become Tecplot zones)."
        )

    zones: list[_ZoneBuild] = []
    multiple = len(supported) > 1
    for vtk_cell_type in supported:
        zone_type = _ZONE_TYPE_FOR_VTK_CELL[vtk_cell_type]
        zone_title = f"{title} ({zone_type.name})" if multiple else title
        cell_idx = np.flatnonzero(types == vtk_cell_type)
        starts = offsets[cell_idx]
        sizes = offsets[cell_idx + 1] - starts

        node_map: npt.NDArray[np.int64] | None = None
        face_map: dict[str, npt.NDArray[np.int64]] | None = None
        try:
            if vtk_cell_type == VTK_POLYHEDRON:
                per_cell, per_face, face_nodes = _polyhedron_cell_faces(grid, cell_idx)
                used = np.unique(face_nodes)
                face_map = _polyhedron_face_map(per_cell, per_face, face_nodes, points)
            elif vtk_cell_type == VTK_POLYGON:
                loops = connectivity[_gather_ranges(starts, sizes)]
                used = np.unique(loops)
                face_map = _polygon_face_map(_csr_offsets(sizes), loops, points)
            else:
                global_ids = connectivity[
                    starts[:, None] + np.arange(_NODES_PER_CELL[vtk_cell_type])
                ]
                used = np.unique(global_ids)
                node_map = np.searchsorted(used, global_ids)
        except ValueError as exc:
            print(
                f"[TecplotTecioWriter] '{title}': skipping {zone_type.name} cells: {exc}."
            )
            continue
        if face_map is not None:
            face_map["face_nodes"] = np.searchsorted(used, face_map["face_nodes"])

        zone = _ZoneBuild(
            title=zone_title, zone_type=zone_type, node_map=node_map, face_map=face_map
        )
        local_points = points[used]
        zone.arrays["x"] = (local_points[:, 0].copy(), ValueLocation.NODAL)
        zone.arrays["y"] = (local_points[:, 1].copy(), ValueLocation.NODAL)
        zone.arrays["z"] = (local_points[:, 2].copy(), ValueLocation.NODAL)

        def nodal(values: npt.NDArray, _ids: npt.NDArray = used) -> npt.NDArray:
            return values[_ids]

        def cell_centered(
            values: npt.NDArray, _idx: npt.NDArray = cell_idx
        ) -> npt.NDArray:
            return values[_idx]

        for i in range(point_data.GetNumberOfArrays()):
            for name, values in _decompose_array(point_data.GetArray(i), nodal) or []:
                zone.arrays[name] = (values, ValueLocation.NODAL)
        for i in range(cell_data.GetNumberOfArrays()):
            for name, values in (
                _decompose_array(cell_data.GetArray(i), cell_centered) or []
            ):
                zone.arrays[name] = (values, ValueLocation.CELL_CENTERED)

        zone.aux = dict(field_aux)
        zones.append(zone)
    return zones


def _build_zones(name: str, dataset: vtkDataSet) -> list[_ZoneBuild]:
    """Build the Tecplot zone(s) for one input block, or ``[]`` if unsupported.

    Returns:
        list: The zones for the block, empty if the dataset type is unsupported.
    """
    if isinstance(dataset, (vtkImageData, vtkRectilinearGrid, vtkStructuredGrid)):
        return [_build_ordered_zone(name, dataset)]
    if isinstance(dataset, vtkPointSet):
        return _build_fe_zones(name, dataset)
    print(
        f"[TecplotTecioWriter] '{name}': dataset type {type(dataset).__name__} is not "
        "supported (expected a structured grid, polydata, or unstructured grid); "
        "skipping."
    )
    return []


def _iter_blocks(data: vtkDataObject) -> list[tuple[str, vtkDataSet]]:
    """Return ``(title, dataset)`` for every non-empty leaf block of *data*.

    *data* may be a plain ``vtkDataSet`` (treated as a single block) or a
    (possibly nested) ``vtkMultiBlockDataSet``, whose leaves are visited via a
    composite-data iterator and named after their composite-data name, or
    ``"Zone <n>"`` (``n`` counting leaves from 0) if unnamed.
    """
    if isinstance(data, vtkMultiBlockDataSet):
        blocks: list[tuple[str, vtkDataSet]] = []
        it = data.NewIterator()
        it.UnRegister(None)
        it.VisitOnlyLeavesOn()
        it.InitTraversal()
        n = 0
        while not it.IsDoneWithTraversal():
            block = it.GetCurrentDataObject()
            if isinstance(block, vtkDataSet) and block.GetNumberOfPoints() > 0:
                meta = it.GetCurrentMetaData()
                name = (
                    meta.Get(vtkCompositeDataSet.NAME())
                    if meta is not None and meta.Has(vtkCompositeDataSet.NAME())
                    else f"Zone {n}"
                )
                blocks.append((name, block))
            n += 1
            it.GoToNextItem()
        return blocks
    if isinstance(data, vtkDataSet) and data.GetNumberOfPoints() > 0:
        return [("Zone 0", data)]
    return []


def _assemble_global_variables(
    zones: list[_ZoneBuild],
) -> list[str]:
    """Return the dataset-wide variable list: ``x``, ``y``, ``z``, then first-seen order."""
    names = ["x", "y", "z"]
    seen = set(names)
    for zone in zones:
        for name in zone.arrays:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _add_vector_aux(dataset_aux: dict[str, str], var_names: list[str]) -> None:
    """Advertise ``Common.UVar``/``VVar``/``WVar`` for a ``Velocity``/``Momentum`` array.

    Lets this plugin's own reader (and Tecplot 360) reassemble the decomposed ``_X``/
    ``_Y``/``_Z`` components back into one vector array on reopen. Silently does
    nothing if no array with a recognized name was decomposed.
    """
    name_to_index = {name: i for i, name in enumerate(var_names)}
    for array_name in _VECTOR_ARRAY_NAMES:
        components = [f"{array_name}_{axis}" for axis in ("X", "Y", "Z")]
        if components[0] not in name_to_index:
            continue
        for key, comp in zip(
            ("Common.UVar", "Common.VVar", "Common.WVar"), components, strict=False
        ):
            if comp in name_to_index:
                dataset_aux[key] = str(name_to_index[comp] + 1)  # 1-based, per format
        if array_name == "Velocity":
            dataset_aux["Common.VectorVarsAreVelocity"] = "1"


# ======================================================================================
# Writer
# ======================================================================================


@smproxy.writer(
    name="TecplotTecioWriter",
    label="Tecplot 360 Writer (TecIO)",
    extensions=_EXTENSIONS,
    file_description=_FILE_DESCRIPTION,
    support_reload=False,
)
@smproperty.input(name="Input", port_index=0)
@smdomain.datatype(
    dataTypes=["vtkDataSet", "vtkMultiBlockDataSet"], composite_data_supported=True
)
class TecplotWriter(VTKPythonAlgorithmBase):
    """Writes the input's current data as one Tecplot SZL/PLT/DAT file.

    One zone per leaf block (or a single zone for a non-composite input); see the
    module docstring for the full mapping, including variable decomposition and
    auxiliary data.
    """

    def __init__(self) -> None:
        """Initialize from VTK object."""
        super().__init__(nInputPorts=1, nOutputPorts=0, inputType="vtkDataObject")
        self._filename: str | None = None

    # -- FileName ------------------------------------------------------------------

    @smproperty.stringvector(name="FileName", panel_visibility="never")
    @smdomain.filelist()
    def SetFileName(self, name: str | None) -> None:
        """Set the output path. Accepts ``.szplt``/``.plt``/``.bin``/``.dat``/``.tec``."""
        name = name or None
        if self._filename != name:
            self._filename = name
            self.Modified()

    # -- Pipeline ---------------------------------------------------------------------

    def Write(self) -> None:
        """Run the pipeline so :meth:`RequestData` writes the file.

        ParaView's writer proxy calls this method when "Save Data" is used. A writer
        written with ``VTKPythonAlgorithmBase`` has no output port to pull on, so
        without it nothing is ever executed.
        """
        self.Modified()
        self.Update()

    def RequestData(self, request: Any, inInfoVec: Any, outInfoVec: Any) -> int:
        """Write the current input to :attr:`_filename`.

        Returns:
            int: ``1`` on success, ``0`` if nothing was written (the reason is printed).
        """
        if tecio is None:
            print(
                "[TecplotTecioWriter] The 'tecio' package is not importable in "
                f"ParaView's Python environment ({_TECIO_IMPORT_ERROR!r}). Install "
                "tecio (and make sure the TecIO shared library can be located, e.g. "
                "via the TECIO_LIB environment variable) in the Python interpreter "
                "ParaView uses, then reload this plugin."
            )
            return 0
        if not self._filename:
            print("[TecplotTecioWriter] No output FileName set; nothing written.")
            return 0

        input0 = self.GetInputData(inInfoVec, 0, 0)
        blocks = _iter_blocks(input0)
        if not blocks:
            print(
                f"[TecplotTecioWriter] '{self._filename}': input has no data to write."
            )
            return 0

        zones: list[_ZoneBuild] = []
        for name, dataset in blocks:
            zones.extend(_build_zones(name, dataset))
        if not zones:
            print(
                f"[TecplotTecioWriter] '{self._filename}': no zone could be built from "
                "the input (see warnings above)."
            )
            return 0
        if self._filename.lower().endswith(".szplt") and any(
            zone.face_map is not None for zone in zones
        ):
            print(
                f"[TecplotTecioWriter] '{self._filename}': polygon and polyhedron "
                "zones cannot be written to SZPLT files yet. Save as .plt or .dat "
                "instead."
            )
            return 0

        var_names = _assemble_global_variables(zones)
        root_dataset_aux, root_var_aux = _split_variable_aux(
            _field_data_to_aux(input0.GetFieldData()), var_names
        )
        root_dataset_aux.setdefault("Common.XVar", "1")
        root_dataset_aux.setdefault("Common.YVar", "2")
        root_dataset_aux.setdefault("Common.ZVar", "3")
        _add_vector_aux(root_dataset_aux, var_names)

        try:
            with tecio.open(
                self._filename,
                "w",
                title="ParaView export",
                variables=var_names,
                file_type=FileType.FULL,
            ) as writer:
                if root_dataset_aux:
                    writer.add_auxdataset_dict(root_dataset_aux)
                if root_var_aux:
                    name_to_index = {name: i for i, name in enumerate(var_names)}
                    writer.add_auxvar_dict({
                        name_to_index[name]: aux
                        for name, aux in root_var_aux.items()
                        if name in name_to_index
                    })

                for zone in zones:
                    passive_vars = {n for n in var_names if n not in zone.arrays}
                    data = [zone.arrays[n][0] for n in var_names if n in zone.arrays]
                    value_locations = [
                        zone.arrays[n][1] for n in var_names if n in zone.arrays
                    ]
                    common_kw: dict[str, Any] = {
                        "title": zone.title,
                        "value_locations": value_locations,
                        "passive_vars": passive_vars,
                        "aux": zone.aux or None,
                    }
                    if zone.zone_type is None:
                        writer.write_ordered_zone(data=data, **common_kw)
                    elif zone.face_map is not None:
                        writer.write_poly_zone(
                            data=data,
                            zone_type=zone.zone_type,
                            **zone.face_map,
                            **common_kw,
                        )
                    else:
                        writer.write_fe_zone(
                            data=data,
                            zone_type=zone.zone_type,
                            node_map=zone.node_map,
                            **common_kw,
                        )
        except Exception as exc:  # noqa: BLE001 - report and leave no half-written file
            print(f"[TecplotTecioWriter] Error writing '{self._filename}': {exc}")
            try:
                os.unlink(self._filename)
            except OSError:
                pass
            return 0

        return 1
