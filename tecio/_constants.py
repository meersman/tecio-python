"""Tecplot meaningful integers.

The TecIO library often uses integers with special meanings (zone types, data types,
data locations). The same values are used both for writing (``tec*142`` functions) and
for SZL reading and writing (``tec_*`` functions). Where available, the equivalent
keyword used in Tecplot ASCII files is exposed as a class property, returning the
corresponding int value.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "Boolean",
    "DataPacking",
    "DataType",
    "Debug",
    "FaceNeighborMode",
    "FeCellShape",
    "FileFormat",
    "FileType",
    "ValueLocation",
    "VarStatus",
    "ZoneType",
]


class FileFormat(Enum):
    """Binary data format selector.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``PLT``
         - ``0``
         - Classic PLT binary format.
       * - ``SZPLT``
         - ``1``
         - SZL subzone-loadable format.
    """

    PLT = 0
    SZPLT = 1


class FileType(Enum):
    """Tecplot file type.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``FULL``
         - ``0``
         - Contains both grid and solution data.
       * - ``GRID``
         - ``1``
         - Grid coordinates only.
       * - ``SOLUTION``
         - ``2``
         - Solution variables only.
    """

    FULL = 0
    GRID = 1
    SOLUTION = 2


class ZoneType(Enum):
    """Tecplot zone type.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``ORDERED``
         - ``0``
         - Structured IJK grid.
       * - ``FELINESEG``
         - ``1``
         - Finite-element line segments.
       * - ``FETRIANGLE``
         - ``2``
         - Finite-element triangles.
       * - ``FEQUADRILATERAL``
         - ``3``
         - Finite-element quadrilaterals.
       * - ``FETETRAHEDRON``
         - ``4``
         - Finite-element tetrahedra.
       * - ``FEBRICK``
         - ``5``
         - Finite-element hexahedra.
       * - ``FEPOLYGON``
         - ``6``
         - Finite-element polygons (face-based).
       * - ``FEPOLYHEDRON``
         - ``7``
         - Finite-element polyhedra (face-based).
       * - ``FEMIXED``
         - ``8``
         - Mixed finite-element types.
    """

    ORDERED = 0
    FELINESEG = 1
    FETRIANGLE = 2
    FEQUADRILATERAL = 3
    FETETRAHEDRON = 4
    FEBRICK = 5
    FEPOLYGON = 6
    FEPOLYHEDRON = 7
    FEMIXED = 8


class FeCellShape(Enum):
    """Unstructured cell shape category.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``BAR``
         - ``0``
         - 2D two-node element.
       * - ``TRIANGLE``
         - ``1``
         - 2D three-node element.
       * - ``QUADRILATERAL``
         - ``2``
         - 2D four-node element.
       * - ``TETRAHEDRON``
         - ``3``
         - 3D four-node element.
       * - ``HEXAHEDRON``
         - ``4``
         - 3D eight-node element.
       * - ``PYRAMID``
         - ``5``
         - 3D five-node element.
       * - ``PRISM``
         - ``6``
         - 3D six-node element.
    """

    BAR = 0
    TRIANGLE = 1
    QUADRILATERAL = 2
    TETRAHEDRON = 3
    HEXAHEDRON = 4
    PYRAMID = 5
    PRISM = 6


class FaceNeighborMode(Enum):
    """Boundary face-sharing mode between zones.

    .. list-table::
       :header-rows: 1
       :widths: 35 10 55

       * - Attribute
         - Value
         - Description
       * - ``LOCAL_ONE_TO_ONE``
         - ``0``
         - Each face has at most one local neighbor.
       * - ``LOCAL_ONE_TO_MANY``
         - ``1``
         - Each face may have multiple local neighbors (hanging nodes).
       * - ``GLOBAL_ONE_TO_ONE``
         - ``2``
         - Each face has at most one neighbor in any zone.
       * - ``GLOBAL_ONE_TO_MANY``
         - ``3``
         - Each face may have multiple neighbors in any zone (hanging nodes).
    """

    LOCAL_ONE_TO_ONE = 0
    LOCAL_ONE_TO_MANY = 1
    GLOBAL_ONE_TO_ONE = 2
    GLOBAL_ONE_TO_MANY = 3


class ValueLocation(Enum):
    """Data value location within a cell.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``CELL_CENTERED``
         - ``0``
         - Values stored at cell centres.
       * - ``NODAL``
         - ``1``
         - Values stored at grid nodes.
    """

    CELL_CENTERED = 0
    NODAL = 1


class DataPacking(Enum):
    """Zone data packing order for ASCII (``.dat``) files.

    Controls whether data is laid out variable-by-variable or point-by-point
    in the ASCII file. The ``DATAPACKING`` keyword in a zone header takes one
    of these two values.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``POINT``
         - ``0``
         - One row per node/cell containing all variable values.
       * - ``BLOCK``
         - ``1``
         - One contiguous block per variable containing all node/cell values.
           Tecplot default; faster for variable-at-a-time access patterns.
    """

    POINT = 0
    BLOCK = 1


class DataType(Enum):
    """On-disk storage type for variable data.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``FLOAT``
         - ``1``
         - 32-bit IEEE floating point.
       * - ``DOUBLE``
         - ``2``
         - 64-bit IEEE floating point.
       * - ``INT32``
         - ``3``
         - 32-bit signed integer.
       * - ``INT16``
         - ``4``
         - 16-bit signed integer.
       * - ``BYTE``
         - ``5``
         - 8-bit unsigned integer.
    """

    FLOAT = 1
    DOUBLE = 2
    INT32 = 3
    INT16 = 4
    BYTE = 5


class VarStatus(Enum):
    """Variable active/passive flag.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``ACTIVE``
         - ``0``
         - Variable has data in this zone.
       * - ``PASSIVE``
         - ``1``
         - Variable has no data in this zone.
    """

    ACTIVE = 0
    PASSIVE = 1


class Boolean(Enum):
    """Boolean flag for C function arguments.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``FALSE``
         - ``0``
         - Logical false.
       * - ``TRUE``
         - ``1``
         - Logical true.
    """

    FALSE = 0
    TRUE = 1


class Debug(Enum):
    """Debug flag for C function arguments.

    .. list-table::
       :header-rows: 1
       :widths: 25 10 65

       * - Attribute
         - Value
         - Description
       * - ``FALSE``
         - ``0``
         - Debug output disabled.
       * - ``TRUE``
         - ``1``
         - Debug output enabled.
    """

    FALSE = 0
    TRUE = 1
