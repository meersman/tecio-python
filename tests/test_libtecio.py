#!/usr/bin/env python3
"""pytest tests for :mod:`tecio.libtecio` C-library bindings.

Two test classes mirror the two APIs:

* :class:`TestNewApi`     — SZL writer via explicit file handles.
                            Tests use mixed DataTypes (FLOAT, DOUBLE, INT32,
                            INT16, BYTE) to cover all write-function paths.
* :class:`TestClassicApi` — Classic PLT writer via global file context.
                            Tests use FLOAT (is_double=False) and DOUBLE
                            (is_double=True) since tecdat142 only supports
                            those two precisions.

Run directly:

    $ python tests/test_libtecio.py -v

Keep output files for Tecplot inspection:

    $ python tests/test_libtecio.py -v --keep-files
"""

# ruff: noqa: E501

import sys
from collections.abc import Callable

import numpy as np
import pytest
from create_test_data import (
    create_fe_brick,
    create_fe_lineseg,
    create_fe_prism,
    create_fe_pyramid,
    create_fe_quad,
    create_fe_tet,
    create_fe_tri,
    create_fe_two_bricks,
    create_ordered,
    create_poly_octagon,
)

from tecio import (
    DataType,
    FaceNeighborMode,
    FileFormat,
    FileType,
    ValueLocation,
    ZoneType,
    libtecio,
)

# ===========================================================================
# New API (SZL/.szplt) — explicit file handles, all DataTypes supported
# ===========================================================================


class TestNewApi:
    """New SZL writer API.

    Key pattern::

        handle = tec_file_writer_open(path, variables=[...])
        izone = tec_zone_create_ijk(handle, title, i, j, k, var_types=[...])
        tec_zone_var_write_float_values(handle, izone, var_num, array)
        ...
        tec_file_writer_close(handle)

    Multiple files can be open simultaneously because each has its own handle.
    """

    def test_tec_zone_create_ijk(self, output_path: Callable) -> None:
        """ORDERED zone with mixed DataTypes.

        Demonstrates:
        - ``tec_file_writer_open``: creates an SZL file and returns a handle
        - ``tec_zone_create_ijk``: registers an ordered zone; returns 1-based zone index
        - ``var_types`` with mixed precision in the same zone:
          FLOAT (var 1, 3), DOUBLE (var 2), INT32 (var 4)
        - Matching write functions: ``write_float_values`` / ``write_double_values``
          / ``write_int32_values`` — the function must match the declared DataType
        - Fortran-order ravel: ``x.ravel(order="F")`` is required for IJK arrays
          so that I varies fastest (Tecplot convention)
        - ``tec_file_writer_close``: finalises and flushes the file
        """
        i, j, k = 3, 4, 5
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)  # FLOAT
        y = y.astype(np.float64)  # DOUBLE
        z = z.astype(np.float32)  # FLOAT
        c = (np.sin(2 * np.pi * x) * 1000).astype(np.int32)  # INT32

        path = output_path("test_tec_zone_create_ijk.szplt")
        handle = libtecio.tec_file_writer_open(
            str(path), variables=["x", "y", "z", "c"]
        )
        izone = libtecio.tec_zone_create_ijk(
            handle,
            "test_ordered_ijk",
            i,
            j,
            k,
            var_types=[DataType.FLOAT, DataType.DOUBLE, DataType.FLOAT, DataType.INT32],
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x.ravel(order="F"))
        libtecio.tec_zone_var_write_double_values(handle, izone, 2, y.ravel(order="F"))
        libtecio.tec_zone_var_write_float_values(handle, izone, 3, z.ravel(order="F"))
        libtecio.tec_zone_var_write_int32_values(handle, izone, 4, c.ravel(order="F"))
        libtecio.tec_file_writer_close(handle)

        assert izone == 1
        assert path.exists()

    def test_tec_zone_create_fe_lineseg(self, output_path: Callable) -> None:
        """FELINESEG zone — two-node line segment elements.

        Demonstrates:
        - ``tec_zone_create_fe``: registers an FE zone; takes num_nodes and
          num_elements separately (unlike ordered zones which take I, J, K)
        - ``ZoneType.FELINESEG``: 2 nodes per element, 2-D line mesh
        - ``tec_zone_node_map_write32``: writes the 1-based connectivity array
          after all variable data; node_map shape is (num_elements, 2)
        - FLOAT + DOUBLE mixed in the same FE zone
        - FE arrays are 1-D and do not need Fortran ravel (node-indexed)
        """
        x, y, nodes = create_fe_lineseg()
        x = x.astype(np.float32)  # FLOAT
        y = y.astype(np.float32)  # FLOAT
        c = np.sin(2 * np.pi * x).astype(np.float64)  # DOUBLE

        path = output_path("test_tec_zone_create_fe_lineseg.szplt")
        handle = libtecio.tec_file_writer_open(str(path), variables=["x", "y", "c"])
        izone = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_lineseg",
            ZoneType.FELINESEG,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT, DataType.FLOAT, DataType.DOUBLE],
            value_locations=[ValueLocation.NODAL] * 3,
        )
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x)
        libtecio.tec_zone_var_write_float_values(handle, izone, 2, y)
        libtecio.tec_zone_var_write_double_values(handle, izone, 3, c)
        libtecio.tec_zone_node_map_write32(handle, izone, (nodes + 1).ravel())
        libtecio.tec_file_writer_close(handle)

        assert izone == 1
        assert path.exists()

    def test_tec_zone_create_fe_tri(self, output_path: Callable) -> None:
        """FETRIANGLE zone — three-node triangular elements.

        Demonstrates:
        - ``ZoneType.FETRIANGLE``: 3 nodes per element, 2-D surface mesh
        - INT32 scalar variable: cast via ``(field * 1000).astype(np.int32)``
        - ``tec_zone_var_write_int32_values``: the third distinct write path
        - node_map shape: (num_elements, 3) — rows are per-element, C-order
        """
        x, y, nodes = create_fe_tri()
        x = x.astype(np.float32)  # FLOAT
        y = y.astype(np.float64)  # DOUBLE
        c = (np.sin(2 * np.pi * x) * 1000).astype(np.int32)  # INT32

        path = output_path("test_tec_zone_create_fe_tri.szplt")
        handle = libtecio.tec_file_writer_open(str(path), variables=["x", "y", "c"])
        izone = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_triangle",
            ZoneType.FETRIANGLE,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT, DataType.DOUBLE, DataType.INT32],
            value_locations=[ValueLocation.NODAL] * 3,
        )
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x)
        libtecio.tec_zone_var_write_double_values(handle, izone, 2, y)
        libtecio.tec_zone_var_write_int32_values(handle, izone, 3, c)
        libtecio.tec_zone_node_map_write32(handle, izone, (nodes + 1).ravel())
        libtecio.tec_file_writer_close(handle)

        assert izone == 1
        assert path.exists()

    def test_tec_zone_create_fe_quad(self, output_path: Callable) -> None:
        """FEQUADRILATERAL zone — four-node quadrilateral elements.

        Demonstrates:
        - ``ZoneType.FEQUADRILATERAL``: 4 nodes per element, 2-D surface mesh
        - DOUBLE coordinates (float64 x/y) with FLOAT scalar — opposite of
          the usual CFD convention, testing the reverse ordering
        - node_map shape: (num_elements, 4)
        """
        x, y, nodes = create_fe_quad()
        x = x.astype(np.float64)  # DOUBLE
        y = y.astype(np.float64)  # DOUBLE
        c = np.sin(2 * np.pi * x).astype(np.float32)  # FLOAT

        path = output_path("test_tec_zone_create_fe_quad.szplt")
        handle = libtecio.tec_file_writer_open(str(path), variables=["x", "y", "c"])
        izone = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_quad",
            ZoneType.FEQUADRILATERAL,
            len(x),
            len(nodes),
            var_types=[DataType.DOUBLE, DataType.DOUBLE, DataType.FLOAT],
            value_locations=[ValueLocation.NODAL] * 3,
        )
        libtecio.tec_zone_var_write_double_values(handle, izone, 1, x)
        libtecio.tec_zone_var_write_double_values(handle, izone, 2, y)
        libtecio.tec_zone_var_write_float_values(handle, izone, 3, c)
        libtecio.tec_zone_node_map_write32(handle, izone, (nodes + 1).ravel())
        libtecio.tec_file_writer_close(handle)

        assert izone == 1
        assert path.exists()

    def test_tec_zone_create_fe_tet(self, output_path: Callable) -> None:
        """FETETRAHEDRON zone — four-node tetrahedral elements (3-D mesh).

        Demonstrates:
        - ``ZoneType.FETETRAHEDRON``: 4 nodes per element, 3-D volume mesh
        - Three coordinate arrays (x, y, z) plus one solution variable
        - FLOAT for all spatial coords, DOUBLE for the solution field —
          typical CFD convention (compact grid, high-precision solution)
        - node_map shape: (num_elements, 4); ravel to 1-D before writing
        """
        x, y, z, nodes = create_fe_tet()
        x = x.astype(np.float32)  # FLOAT
        y = y.astype(np.float32)  # FLOAT
        z = z.astype(np.float32)  # FLOAT
        c = np.sin(2 * np.pi * x).astype(np.float64)  # DOUBLE

        path = output_path("test_tec_zone_create_fe_tet.szplt")
        handle = libtecio.tec_file_writer_open(
            str(path), variables=["x", "y", "z", "c"]
        )
        izone = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_tet",
            ZoneType.FETETRAHEDRON,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT, DataType.FLOAT, DataType.FLOAT, DataType.DOUBLE],
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x)
        libtecio.tec_zone_var_write_float_values(handle, izone, 2, y)
        libtecio.tec_zone_var_write_float_values(handle, izone, 3, z)
        libtecio.tec_zone_var_write_double_values(handle, izone, 4, c)
        libtecio.tec_zone_node_map_write32(handle, izone, (nodes + 1).ravel())
        libtecio.tec_file_writer_close(handle)

        assert izone == 1
        assert path.exists()

    def test_tec_zone_create_fe_pyramid(self, output_path: Callable) -> None:
        """Pyramid element as a degenerate FEBRICK — INT16 scalar.

        Demonstrates:
        - Pyramid representation: use ``ZoneType.FEBRICK`` with nodes 5-8
          all pointing to the single apex node (collapsed brick)
        - ``DataType.INT16``: compact short-integer field, range [-32768, 32767]
        - ``tec_zone_var_write_int16_values``: fourth distinct write path
        - Cast pattern: ``(field * 100).astype(np.int16)``
        - Degenerate elements are the standard Tecplot approach for pyramids
          and prisms when using the simple FE zone types
        """
        x, y, z, nodes = create_fe_pyramid()
        x = x.astype(np.float32)  # FLOAT
        y = y.astype(np.float64)  # DOUBLE
        z = z.astype(np.float32)  # FLOAT
        c = (np.sin(2 * np.pi * x) * 100).astype(np.int16)  # INT16

        path = output_path("test_tec_zone_create_fe_pyramid.szplt")
        handle = libtecio.tec_file_writer_open(
            str(path), variables=["x", "y", "z", "c"]
        )
        izone = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_pyramid",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT, DataType.DOUBLE, DataType.FLOAT, DataType.INT16],
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x)
        libtecio.tec_zone_var_write_double_values(handle, izone, 2, y)
        libtecio.tec_zone_var_write_float_values(handle, izone, 3, z)
        libtecio.tec_zone_var_write_int16_values(handle, izone, 4, c)
        libtecio.tec_zone_node_map_write32(handle, izone, (nodes + 1).ravel())
        libtecio.tec_file_writer_close(handle)

        assert izone == 1
        assert path.exists()

    def test_tec_zone_create_fe_prism(self, output_path: Callable) -> None:
        """Triangular prism as a degenerate FEBRICK — BYTE (uint8) scalar.

        Demonstrates:
        - Prism representation: ``ZoneType.FEBRICK`` with repeated edge nodes
          (bottom tri 1,2,3,3 / top tri 4,5,6,6)
        - ``DataType.BYTE`` (uint8): ultra-compact, range [0, 255]
        - ``tec_zone_var_write_uint8_values``: fifth and final write path
        - Cast pattern: ``((field + 1.0) * 127).astype(np.uint8)`` maps
          the [-1, 1] float range to [0, 254] bytes
        - Useful for per-node color indices or boolean masks
        """
        x, y, z, nodes = create_fe_prism()
        x = x.astype(np.float32)  # FLOAT
        y = y.astype(np.float64)  # DOUBLE
        z = z.astype(np.float32)  # FLOAT
        c = ((np.sin(2 * np.pi * x) + 1.0) * 127).astype(np.uint8)  # BYTE

        path = output_path("test_tec_zone_create_fe_prism.szplt")
        handle = libtecio.tec_file_writer_open(
            str(path), variables=["x", "y", "z", "c"]
        )
        izone = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_prism",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT, DataType.DOUBLE, DataType.FLOAT, DataType.BYTE],
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x)
        libtecio.tec_zone_var_write_double_values(handle, izone, 2, y)
        libtecio.tec_zone_var_write_float_values(handle, izone, 3, z)
        libtecio.tec_zone_var_write_uint8_values(handle, izone, 4, c)
        libtecio.tec_zone_node_map_write32(handle, izone, (nodes + 1).ravel())
        libtecio.tec_file_writer_close(handle)

        assert izone == 1
        assert path.exists()

    def test_tec_zone_create_fe_brick(self, output_path: Callable) -> None:
        """FEBRICK zone — standard 8-node hexahedral elements.

        Demonstrates:
        - ``ZoneType.FEBRICK``: 8 nodes per element, standard 3-D volume mesh
        - FLOAT x/y, DOUBLE z, INT32 scalar — three different types in one zone
        - node_map shape: (num_elements, 8); Tecplot brick node ordering
          follows the standard hex ordering (four bottom, four top)
        """
        x, y, z, faces, nodes = create_fe_brick()
        x = x.astype(np.float32)  # FLOAT
        y = y.astype(np.float32)  # FLOAT
        z = z.astype(np.float64)  # DOUBLE
        c = (np.sin(2 * np.pi * x) * 1000).astype(np.int32)  # INT32

        path = output_path("test_tec_zone_create_fe_brick.szplt")
        handle = libtecio.tec_file_writer_open(
            str(path), variables=["x", "y", "z", "c"]
        )
        izone = libtecio.tec_zone_create_fe(
            handle,
            "test_fe_brick",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT, DataType.FLOAT, DataType.DOUBLE, DataType.INT32],
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x)
        libtecio.tec_zone_var_write_float_values(handle, izone, 2, y)
        libtecio.tec_zone_var_write_double_values(handle, izone, 3, z)
        libtecio.tec_zone_var_write_int32_values(handle, izone, 4, c)
        libtecio.tec_zone_node_map_write32(handle, izone, (nodes + 1).ravel())
        libtecio.tec_file_writer_close(handle)

        assert izone == 1
        assert path.exists()

    def test_tec_zone_face_nbr_write_connections(self, output_path: Callable) -> None:
        """Two adjacent FEBRICK cells with face-neighbor connectivity.

        Demonstrates:
        - ``num_face_cons``: declared at zone creation, must match the number
          of face-neighbor records written via ``tec_zone_face_nbr_write_connections32``
        - ``FaceNeighborMode.LOCAL_ONE_TO_ONE``: each face has at most one
          neighbor within the same zone
        - ``tec_zone_face_nbr_write_connections32``: called after node map,
          writes the (cell, face, neighbor_cell) triplet array
        - Cell-centered INT16 variable (one value per element, not per node):
          requires ``ValueLocation.CELL_CENTERED`` in value_locations
        - Face neighbors help Tecplot render smooth surfaces at element boundaries
        """
        x, y, z, nodes, face_neighbors = create_fe_two_bricks()
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        c = np.array([100, 200], dtype=np.int16)  # CELL_CENTERED INT16

        path = output_path("test_tec_zone_face_nbr.szplt")
        handle = libtecio.tec_file_writer_open(
            str(path), variables=["x", "y", "z", "c"]
        )
        izone = libtecio.tec_zone_create_fe(
            handle,
            "test_two_bricks",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            var_types=[DataType.FLOAT, DataType.FLOAT, DataType.FLOAT, DataType.INT16],
            value_locations=[
                ValueLocation.NODAL,
                ValueLocation.NODAL,
                ValueLocation.NODAL,
                ValueLocation.CELL_CENTERED,
            ],
            num_face_cons=len(face_neighbors),
            face_nbr_mode=FaceNeighborMode.LOCAL_ONE_TO_ONE,
        )
        libtecio.tec_zone_var_write_float_values(handle, izone, 1, x)
        libtecio.tec_zone_var_write_float_values(handle, izone, 2, y)
        libtecio.tec_zone_var_write_float_values(handle, izone, 3, z)
        libtecio.tec_zone_var_write_int16_values(handle, izone, 4, c)
        libtecio.tec_zone_node_map_write32(handle, izone, (nodes + 1).ravel())
        # Cell indices and the face ordinal are all 1-based for this raw API.
        face_neighbors_1based = face_neighbors + 1
        libtecio.tec_zone_face_nbr_write_connections32(
            handle, izone, face_neighbors_1based.ravel()
        )
        libtecio.tec_file_writer_close(handle)

        assert izone == 1
        assert path.exists()

    def test_tec_zone_aux_data_and_unsteady(self, output_path: Callable) -> None:
        """Dataset, variable, and zone auxiliary data with 50 unsteady zones.

        Demonstrates:
        - ``tec_data_set_add_aux_data``: dataset-level key/value metadata,
          called before any zone is created
        - ``tec_var_add_aux_data``: variable-level metadata (units, description),
          called before any zone is created; references 1-based variable index
        - ``tec_zone_add_aux_data``: zone-level metadata, called after
          ``tec_zone_create_ijk`` using the returned zone index
        - ``tec_zone_set_unsteady_options``: assigns solution_time and strand_id
          to a zone for transient animation in Tecplot 360
        - Multiple zones in a loop: izone increments from 1 to N automatically
        - FLOAT x/z, DOUBLE y/c — mixed precision in a transient dataset
        """
        i, j, k = 3, 4, 5
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float64)
        z = z.astype(np.float32)
        solution_times = np.linspace(0.0, 2 * np.pi, 50)

        path = output_path("test_tec_aux_data_unsteady.szplt")
        handle = libtecio.tec_file_writer_open(
            str(path), variables=["x", "y", "z", "c"]
        )

        libtecio.tec_data_set_add_aux_data(handle, "Author", "test_tecio_lite")
        libtecio.tec_data_set_add_aux_data(handle, "CreatedBy", "libtecio")
        libtecio.tec_data_set_add_aux_data(handle, "Version", "1.0")
        libtecio.tec_var_add_aux_data(handle, 1, "Units", "meters")
        libtecio.tec_var_add_aux_data(handle, 2, "Units", "meters")
        libtecio.tec_var_add_aux_data(handle, 3, "Units", "meters")
        libtecio.tec_var_add_aux_data(handle, 4, "Units", "dimensionless")
        libtecio.tec_var_add_aux_data(handle, 4, "Description", "sin*cos field")

        last_zone = None
        for num, t in enumerate(solution_times):
            c = (np.sin(2 * np.pi * x + t) * np.cos(2 * np.pi * y + t)).astype(
                np.float64
            )
            izone = libtecio.tec_zone_create_ijk(
                handle,
                f"test_aux_unsteady_{num + 1}",
                i,
                j,
                k,
                var_types=[
                    DataType.FLOAT,
                    DataType.DOUBLE,
                    DataType.FLOAT,
                    DataType.DOUBLE,
                ],
                value_locations=[ValueLocation.NODAL] * 4,
            )
            libtecio.tec_zone_add_aux_data(handle, izone, "MeshType", "structured")
            libtecio.tec_zone_add_aux_data(handle, izone, "IJK", f"{i}x{j}x{k}")
            libtecio.tec_zone_set_unsteady_options(
                handle, izone, solution_time=t, strand=1
            )
            libtecio.tec_zone_var_write_float_values(
                handle, izone, 1, x.ravel(order="F")
            )
            libtecio.tec_zone_var_write_double_values(
                handle, izone, 2, y.ravel(order="F")
            )
            libtecio.tec_zone_var_write_float_values(
                handle, izone, 3, z.ravel(order="F")
            )
            libtecio.tec_zone_var_write_double_values(
                handle, izone, 4, c.ravel(order="F")
            )
            last_zone = izone

        libtecio.tec_file_writer_close(handle)

        assert last_zone == len(solution_times)
        assert path.exists()


# ===========================================================================
# Classic API (PLT/.plt) — global context, FLOAT and DOUBLE only
# ===========================================================================


class TestClassicApi:
    """Classic TecIO API.

    Key pattern::

        tecini142(path, variables=[...])  # opens global file context
        teczne142(title, zone_type, ...)  # declares zone; must come first
        tecdat142(array, is_double=False)  # writes one variable at a time
        tecnode142(node_map)  # writes connectivity (FE only)
        tecend142()  # finalises and closes the file

    Only one file can be open at a time. Zone header → data → connectivity
    must be strictly ordered before declaring the next zone.
    """

    def test_plt_tec_zone_create_ijk(self, output_path: Callable) -> None:
        """PLT ORDERED zone — float32 x/z, float64 y/scalar.

        Demonstrates:
        - ``tecini142``: opens the global PLT file context; call once per file
        - ``teczne142``: declares an ORDERED zone with I/J/K dimensions
        - ``tecdat142``: writes one variable per call in dataset-variable order
          - ``is_double=False`` → float32 (FLOAT) stored on disk
          - ``is_double=True``  → float64 (DOUBLE) stored on disk
        - ``tecend142``: flushes and closes; required to produce a valid file
        - Fortran ravel: ``x.ravel(order="F")`` is required for IJK arrays
        """
        i, j, k = 3, 4, 5
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float64)
        z = z.astype(np.float32)
        c = (np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)).astype(np.float64)
        path = output_path("test_plt_zone_create_ijk.plt")

        libtecio.tecini142(str(path), variables=["x", "y", "z", "c"])
        libtecio.teczne142(
            "test_ordered_ijk",
            ZoneType.ORDERED,
            i,
            j,
            k,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tecdat142(x.ravel(order="F"), is_double=False)
        libtecio.tecdat142(y.ravel(order="F"), is_double=True)
        libtecio.tecdat142(z.ravel(order="F"), is_double=False)
        libtecio.tecdat142(c.ravel(order="F"), is_double=True)
        libtecio.tecend142()

        assert path.exists()

    def test_plt_tec_zone_create_fe_lineseg(self, output_path: Callable) -> None:
        """PLT FELINESEG — float32 x/y, float64 scalar.

        Demonstrates:
        - FE zone declaration: ``imax=num_nodes``, ``jmax=num_elements``,
          ``kmax=0`` (unused for simple FE zones)
        - ``tecnode142``: writes the 1-based node map after all tecdat142 calls;
          order is header → data → connectivity
        - ``file_format=FileFormat.PLT``: explicit PLT format selection
        """
        x, y, nodes = create_fe_lineseg()
        c = np.sin(2 * np.pi * x).astype(np.float64)
        path = output_path("test_plt_zone_create_fe_lineseg.plt")

        libtecio.tecini142(
            str(path),
            variables=["x", "y", "c"],
            file_format=FileFormat.PLT,
            file_type=FileType.FULL,
        )
        libtecio.teczne142(
            "test_fe_lineseg",
            ZoneType.FELINESEG,
            len(x),
            len(nodes),
            0,
            value_locations=[ValueLocation.NODAL] * 3,
        )
        libtecio.tecdat142(x.ravel(), is_double=False)
        libtecio.tecdat142(y.ravel(), is_double=False)
        libtecio.tecdat142(c.ravel(), is_double=True)
        libtecio.tecnode142((nodes + 1).ravel())
        libtecio.tecend142()

        assert path.exists()

    def test_plt_tec_zone_create_fe_tri(self, output_path: Callable) -> None:
        """PLT FETRIANGLE — float64 x/y, float32 scalar.

        Demonstrates:
        - ``ZoneType.FETRIANGLE`` in the classic API
        - Reversing the typical precision: DOUBLE coordinates, FLOAT solution
        - FE arrays are 1-D; no Fortran ravel needed
        """
        x, y, nodes = create_fe_tri()
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        c = np.sin(2 * np.pi * x).astype(np.float32)
        path = output_path("test_plt_zone_create_fe_tri.plt")

        libtecio.tecini142(str(path), variables=["x", "y", "c"])
        libtecio.teczne142(
            "test_fe_tri",
            ZoneType.FETRIANGLE,
            len(x),
            len(nodes),
            0,
            value_locations=[ValueLocation.NODAL] * 3,
        )
        libtecio.tecdat142(x.ravel(), is_double=True)
        libtecio.tecdat142(y.ravel(), is_double=True)
        libtecio.tecdat142(c.ravel(), is_double=False)
        libtecio.tecnode142((nodes + 1).ravel())
        libtecio.tecend142()

        assert path.exists()

    def test_plt_tec_zone_create_fe_quad(self, output_path: Callable) -> None:
        """PLT FEQUADRILATERAL — float32 x/y, float64 scalar.

        Demonstrates:
        - ``ZoneType.FEQUADRILATERAL`` in the classic API
        - Standard precision pattern: FLOAT grid, DOUBLE solution
        """
        x, y, nodes = create_fe_quad()
        c = np.sin(2 * np.pi * x).astype(np.float64)
        path = output_path("test_plt_zone_create_fe_quad.plt")

        libtecio.tecini142(str(path), variables=["x", "y", "c"])
        libtecio.teczne142(
            "test_fe_quad",
            ZoneType.FEQUADRILATERAL,
            len(x),
            len(nodes),
            0,
            value_locations=[ValueLocation.NODAL] * 3,
        )
        libtecio.tecdat142(x.ravel(), is_double=False)
        libtecio.tecdat142(y.ravel(), is_double=False)
        libtecio.tecdat142(c.ravel(), is_double=True)
        libtecio.tecnode142((nodes + 1).ravel())
        libtecio.tecend142()

        assert path.exists()

    def test_plt_tec_zone_create_fe_tet(self, output_path: Callable) -> None:
        """PLT FETETRAHEDRON — float32 x/y/z, float64 scalar.

        Demonstrates:
        - ``ZoneType.FETETRAHEDRON`` in the classic API with 3-D coordinates
        - Four ``tecdat142`` calls in dataset-variable order before ``tecnode142``
        - Compact grid (float32) + high-precision solution (float64)
        """
        x, y, z, nodes = create_fe_tet()
        c = np.sin(2 * np.pi * x).astype(np.float64)
        path = output_path("test_plt_zone_create_fe_tet.plt")

        libtecio.tecini142(str(path), variables=["x", "y", "z", "c"])
        libtecio.teczne142(
            "test_fe_tet",
            ZoneType.FETETRAHEDRON,
            len(x),
            len(nodes),
            0,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tecdat142(x.ravel(), is_double=False)
        libtecio.tecdat142(y.ravel(), is_double=False)
        libtecio.tecdat142(z.ravel(), is_double=False)
        libtecio.tecdat142(c.ravel(), is_double=True)
        libtecio.tecnode142((nodes + 1).ravel())
        libtecio.tecend142()

        assert path.exists()

    def test_plt_tec_zone_create_fe_pyramid(self, output_path: Callable) -> None:
        """PLT pyramid as degenerate FEBRICK — float64 all variables.

        Demonstrates:
        - Pyramid as FEBRICK: apex node repeated for nodes 5-8
        - All float64 (is_double=True for every tecdat142 call) — maximum
          precision, useful when mixed float32/float64 isn't needed
        """
        x, y, z, nodes = create_fe_pyramid()
        c = np.sin(2 * np.pi * x).astype(np.float64)
        path = output_path("test_plt_zone_create_fe_pyramid.plt")

        libtecio.tecini142(str(path), variables=["x", "y", "z", "c"])
        libtecio.teczne142(
            "test_fe_pyramid",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            0,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tecdat142(x.ravel(), is_double=True)
        libtecio.tecdat142(y.ravel(), is_double=True)
        libtecio.tecdat142(z.ravel(), is_double=True)
        libtecio.tecdat142(c.ravel(), is_double=True)
        libtecio.tecnode142((nodes + 1).ravel())
        libtecio.tecend142()

        assert path.exists()

    def test_plt_tec_zone_create_fe_prism(self, output_path: Callable) -> None:
        """PLT prism as degenerate FEBRICK — float32 all variables.

        Demonstrates:
        - Triangular prism as FEBRICK: edge nodes repeated (1,2,3,3 / 4,5,6,6)
        - All float32 (is_double=False for every tecdat142 call) — minimum
          precision, smallest file size, suitable for visualization-only data
        """
        x, y, z, nodes = create_fe_prism()
        c = np.sin(2 * np.pi * x).astype(np.float32)
        path = output_path("test_plt_zone_create_fe_prism.plt")

        libtecio.tecini142(str(path), variables=["x", "y", "z", "c"])
        libtecio.teczne142(
            "test_fe_prism",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            0,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tecdat142(x.ravel(), is_double=False)
        libtecio.tecdat142(y.ravel(), is_double=False)
        libtecio.tecdat142(z.ravel(), is_double=False)
        libtecio.tecdat142(c.ravel(), is_double=False)
        libtecio.tecnode142((nodes + 1).ravel())
        libtecio.tecend142()

        assert path.exists()

    def test_plt_tec_zone_create_fe_brick(self, output_path: Callable) -> None:
        """PLT FEBRICK — float32 x/y/z, float64 scalar.

        Demonstrates:
        - Standard 8-node hex element in the classic API
        - Mixed precision: float32 grid coordinates, float64 solution field
        """
        x, y, z, faces, nodes = create_fe_brick()
        c = np.sin(2 * np.pi * x).astype(np.float64)
        path = output_path("test_plt_zone_create_fe_brick.plt")

        libtecio.tecini142(str(path), variables=["x", "y", "z", "c"])
        libtecio.teczne142(
            "test_fe_brick",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            0,
            value_locations=[ValueLocation.NODAL] * 4,
        )
        libtecio.tecdat142(x.ravel(), is_double=False)
        libtecio.tecdat142(y.ravel(), is_double=False)
        libtecio.tecdat142(z.ravel(), is_double=False)
        libtecio.tecdat142(c.ravel(), is_double=True)
        libtecio.tecnode142((nodes + 1).ravel())
        libtecio.tecend142()

        assert path.exists()

    def test_plt_tec_zone_face_nbr_write_connections(
        self, output_path: Callable
    ) -> None:
        """PLT face-neighbor connections — float32 all variables.

        Demonstrates:
        - ``num_face_connections``: declared in ``teczne142``; must match the
          number of records written by ``tecface142``
        - ``tecface142``: called after ``tecnode142``; writes the face-neighbor
          array (cell, face, neighbor_cell triplets)
        - Full write order: ``teczne142`` → ``tecdat142`` × N → ``tecnode142``
          → ``tecface142`` → (next zone or tecend142)
        """
        x, y, z, nodes, face_neighbors = create_fe_two_bricks()
        c = np.sin(2 * np.pi * x).astype(np.float32)
        path = output_path("test_plt_zone_face_nbr.plt")

        libtecio.tecini142(str(path), variables=["x", "y", "z", "c"])
        libtecio.teczne142(
            "test_two_bricks",
            ZoneType.FEBRICK,
            len(x),
            len(nodes),
            0,
            value_locations=[ValueLocation.NODAL] * 4,
            num_face_connections=2,
            face_nbr_mode=FaceNeighborMode.LOCAL_ONE_TO_ONE,
        )
        libtecio.tecdat142(x.ravel(), is_double=False)
        libtecio.tecdat142(y.ravel(), is_double=False)
        libtecio.tecdat142(z.ravel(), is_double=False)
        libtecio.tecdat142(c.ravel(), is_double=False)
        libtecio.tecnode142((nodes + 1).ravel())
        # Cell indices and the face ordinal are all 1-based for this raw API.
        face_neighbors_1based = face_neighbors + 1
        libtecio.tecface142(face_neighbors_1based.ravel())
        libtecio.tecend142()

        assert path.exists()

    def test_plt_tec_zone_aux_data_and_unsteady(self, output_path: Callable) -> None:
        """PLT aux data and unsteady options — float32 x/z, float64 y/scalar.

        Demonstrates:
        - ``tecauxstr142``: dataset-level aux data; must be called after
          ``tecini142`` and before the first ``teczne142``
        - ``tecvauxstr142``: variable-level aux data; also before first zone;
          references 1-based variable index
        - ``teczauxstr142``: zone-level aux data; must be called immediately
          after ``teczne142`` and before any ``tecdat142`` calls for that zone
        - ``strand`` and ``solution_time`` in ``teczne142``: transient dataset
          metadata embedded directly in the zone header (no separate call)
        - Classic API strict ordering: dataset aux → variable aux → zone header
          → zone aux → zone data → connectivity → next zone
        """
        i, j, k = 3, 4, 5
        x, y, z = create_ordered((i, j, k))
        x = x.astype(np.float32)
        y = y.astype(np.float64)
        z = z.astype(np.float32)
        solution_times = np.linspace(0.0, 2 * np.pi, 50)
        path = output_path("test_plt_aux_data_unsteady.plt")

        libtecio.tecini142(str(path), variables=["x", "y", "z", "c"])
        libtecio.tecauxstr142("Author", "test_tecio_lite")
        libtecio.tecauxstr142("CreatedBy", "libtecio")
        libtecio.tecauxstr142("Version", "1.0")
        libtecio.tecvauxstr142(1, "Units", "meters")
        libtecio.tecvauxstr142(2, "Units", "meters")
        libtecio.tecvauxstr142(3, "Units", "meters")
        libtecio.tecvauxstr142(4, "Units", "dimensionless")
        libtecio.tecvauxstr142(4, "Description", "sin*cos field")

        for num, t in enumerate(solution_times):
            c = (np.sin(2 * np.pi * x + t) * np.cos(2 * np.pi * y + t)).astype(
                np.float64
            )
            libtecio.teczne142(
                f"test_aux_unsteady_{num + 1}",
                ZoneType.ORDERED,
                i,
                j,
                k,
                value_locations=[ValueLocation.NODAL] * 4,
                strand=1,
                solution_time=t,
            )
            libtecio.teczauxstr142("MeshType", "structured")
            libtecio.teczauxstr142("IJK", f"{i}x{j}x{k}")
            libtecio.tecdat142(x.ravel(order="F"), is_double=False)
            libtecio.tecdat142(y.ravel(order="F"), is_double=True)
            libtecio.tecdat142(z.ravel(order="F"), is_double=False)
            libtecio.tecdat142(c.ravel(order="F"), is_double=True)

        libtecio.tecend142()

        assert path.exists()

    def test_plt_tec_zone_create_fe_polygon(self, output_path: Callable) -> None:
        """PLT FEPOLYGON zone — polygon elements via face-based connectivity.

        Demonstrates:
        - ``tecpolyzne142``: special zone creation for polygon and polyhedral zones;
          requires ``num_faces`` and ``total_num_face_nodes`` in addition to
          the standard node and element counts
        - ``tecpolyface142``: writes the face connectivity instead of ``tecnode142``;
          takes per-face node counts, concatenated node lists, and left/right element
          indices (0 = boundary face, positive = 1-based element index)
        - Cell-centered scalar: ``ValueLocation.CELL_CENTERED`` for a variable
          with one value per element
        - FEPOLYGON is the only way to represent truly arbitrary 2-D polygons;
          use this instead of FEQUADRILATERAL when elements have variable node counts
        """
        x, y, faces, left_elems, right_elems = create_poly_octagon()
        c = np.array([8], dtype=np.float32)
        path = output_path("test_plt_fe_polygon.plt")

        num_nodes = len(x)
        num_elements = 1
        num_faces = len(faces)
        total_face_nodes = len(faces.ravel())
        face_node_counts = np.full(num_faces, 2)
        faces_1based = faces + 1
        left_elems_1based = left_elems + 1
        right_elems_1based = right_elems + 1
        # left_elems = np.full(num_faces, 0)
        # right_elems = np.full(num_faces, 1)

        libtecio.tecini142(str(path), variables=["x", "y", "c"])
        libtecio.tecpolyzne142(
            zone_title="test_fe_polygon",
            zone_type=ZoneType.FEPOLYGON,
            num_nodes=num_nodes,
            num_faces=num_faces,
            num_elements=num_elements,
            total_num_face_nodes=total_face_nodes,
            value_locations=[
                ValueLocation.NODAL,
                ValueLocation.NODAL,
                ValueLocation.CELL_CENTERED,
            ],
        )
        libtecio.tecdat142(x.ravel(), is_double=False)
        libtecio.tecdat142(y.ravel(), is_double=False)
        libtecio.tecdat142(c.ravel(), is_double=False)
        libtecio.tecpolyface142(
            face_node_counts,
            faces_1based.ravel(order="C"),
            left_elems_1based,
            right_elems_1based,
        )
        libtecio.tecend142()

        assert path.exists()

    def test_plt_tecpolybconn142_rejects_inconsistent_lengths(
        self, output_path: Callable
    ) -> None:
        """``tecpolybconn142`` checks array lengths before TecIO reads them unchecked.

        TecIO does no bounds check on the connection arrays, and a null zone array
        crashes the process, so the wrapper requires zones and validates every length.
        """
        path = output_path("test_plt_polybconn_lengths.plt")
        libtecio.tecini142(
            str(path), variables=["x", "y"], scratch_dir=str(path.parent)
        )
        try:
            with pytest.raises(ValueError, match="elems"):
                libtecio.tecpolybconn142([2], [1], [1])
            with pytest.raises(ValueError, match="zones"):
                libtecio.tecpolybconn142([2], [1, 1], [1])
        finally:
            libtecio.tecend142()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
