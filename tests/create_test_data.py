"""Library of functions that create various data types for tecio tests."""

# ruff: noqa: F403, F405, N802

import numpy as np
import numpy.typing as npt

# ======================================================================================
# Functions to create all supported Tecplot data formats
# ======================================================================================


def scalar_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None = None,
) -> np.ndarray:
    """Return a simple sin-cos scalar field over the supplied coordinate arrays."""
    if z is not None:
        return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) * (1.0 + 0.1 * z)
    return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


def create_ordered(
    ijk: tuple[int, ...],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Create ordered coordinates.

    Example layout for (3, 4, 1) - single XY plane viewed from above:

    j=3   9 -- 10 -- 11
          |     |     |
    j=2   6 --- 7 --- 8
          |     |     |
    j=1   3 --- 4 --- 5
          |     |     |
    j=0   0 --- 1 --- 2
         i=0  i=1   i=2

    Node ordering: i (x) varies fastest, then j (y), then k (z).
    """
    x_ = np.linspace(0.0, ijk[0], ijk[0], dtype=np.float32)
    y_ = np.linspace(0.0, ijk[1], ijk[1], dtype=np.float32)
    z_ = np.linspace(0.0, ijk[2], ijk[2], dtype=np.float32)
    x, y, z = np.meshgrid(x_, y_, z_, indexing="ij")
    return x, y, z


def create_FE_lineseg() -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]
]:
    """Create coordinates and nodemap for FELINESEG zone type.

    Node Map:
    [0] -- <C0> -- [1] -- <C1> -- [2]
    """
    # xy coords
    points = np.array([
        [0, 0],  # point A
        [0.5, 1],  # point B
        [1, 0],  # Point C
    ])
    x = points[:, 0]
    y = points[:, 1]
    # Nodemap
    nodes = np.array([
        [0, 1],  # line seg AB
        [1, 2],  # line seg BC
    ])
    return x, y, nodes


def create_FE_tri() -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]
]:
    r"""Create coordinates and nodemap for FELTRIANGLE zone type.

    Node Map: Two triangles sharing edge 1-2:
    2 --- 3  Cell 0: 0-1-2
    |\ C1 |  Cell 1: 1-3-2
    | \   |
    |C0 \ |
    0 --- 1
    """
    points = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ])
    x = points[:, 0]
    y = points[:, 1]
    # Nodemap
    nodes = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ])
    return x, y, nodes


def create_FE_quad() -> tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]
]:
    """Create coordinates and nodemap for FEQUADRILATERAL zone type.

    Node Map: Two quads sharing edge 1-4:
    3 --- 4 --- 5  Cell 0: 0-1-4-3
    | C0  | C1  |  Cell 1: 1-2-5-4
    0 --- 1 --- 2
    """
    # Can create polygon from the same two FE tri cells above
    points = np.array([
        [0, 0],
        [0.5, 0],
        [1, 0],
        [0, 0.3],
        [0.5, 0.6],
        [1, 1],
    ])
    x = points[:, 0]
    y = points[:, 1]
    # Nodemap
    nodes = np.array([
        [0, 1, 4, 3],
        [1, 2, 5, 4],
    ])
    return x, y, nodes


def create_FE_tet() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
]:
    r"""Create coordinates and nodemap for FETETRAHEDRON zone type.

    Node Map: Two tetrahedra sharing base face 0-1-2, apices above (3)
    and below (4):

        3        <- apex tet 1 (z > 0)
       /|\
      / | \
     /  |  \
    0---+---1   <- shared base triangle
     \  |  /
      \ | /
       \|/
        4        <- apex tet 2 (z < 0)

    Cell 0: [0, 1, 2, 3]
    Cell 1: [0, 2, 1, 4]
    """
    points = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.5, 1.0, 0.0],  # 2
        [0.5, 0.5, 1.0],  # 3 - apex of tet 1
        [0.5, 0.5, -1.0],  # 4 - apex of tet 2
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # Each row: base triangle (CCW from outside) + apex
    nodes = np.array([
        [0, 1, 2, 3],  # tet 1 - apex above
        [0, 2, 1, 4],  # tet 2 - apex below
    ])
    return x, y, z, nodes


def create_FE_pyramid() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
]:
    """Create coordinates and nodemap for a pyramid using FEBRICK zone type.

    Node: Tecplot represents pyramids as degenerate bricks where nodes 4,5,6,7 are all
    the apex node repeated.

    """
    points = np.array([
        [0.0, 0.0, 0.0],  # 0 - base
        [1.0, 0.0, 0.0],  # 1 - base
        [1.0, 1.0, 0.0],  # 2 - base
        [0.0, 1.0, 0.0],  # 3 - base
        [0.5, 0.5, 1.0],  # 4 - apex
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # Repeat apex node (4) for the top 4 nodes of the brick
    nodes = np.array([
        [0, 1, 2, 3, 4, 4, 4, 4],
    ])
    return x, y, z, nodes


def create_FE_prism() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
]:
    """Create coordinates and nodemap for a triangular prism using FEBRICK.

    Note: Represented as degenerate brick: bottom tri nodes 0,1,2 paired with
    top tri nodes 3,4,5 — each tri edge node repeated to fill 8-node brick.

    """
    points = np.array([
        [0.0, 0.0, 0.0],  # 0 - bottom tri
        [1.0, 0.0, 0.0],  # 1 - bottom tri
        [0.5, 1.0, 0.0],  # 2 - bottom tri
        [0.0, 0.0, 1.0],  # 3 - top tri
        [1.0, 0.0, 1.0],  # 4 - top tri
        [0.5, 1.0, 1.0],  # 5 - top tri
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # Bottom tri: 0,1,2,2 / Top tri: 3,4,5,5
    nodes = np.array([
        [0, 1, 2, 2, 3, 4, 5, 5],
    ])
    return x, y, z, nodes


def create_FE_brick() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Create coordinates and nodemap for FEBRICK zone type.

    Node Map:
      7 ----- 6
     /|      /|
    4 ----- 5 |
    | 3 ----|-2
    |/      |/
    0 ----- 1

    Faces:
    f0: n0-n4-n7-n3 left face (Imin)
    f1: n1-n2-n6-n5 right face (Imax)
    f2: n0-n1-n5-n5 front face (Jmin)
    f3: n2-n3-n7-n6 back face (Jmax)
    f4: n0-n1-n2-n3 bottom face (Kmin)
    f5: n4-n5-n6-n7 top face (Kmax)
    """
    points = np.array([
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [1, 1, 1],  # 6
        [0, 1, 1],  # 7
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    # Nodemap
    faces = np.array([
        [0, 1, 2, 3],  # Face 1
        [0, 3, 7, 4],  # Face 2
        [4, 7, 6, 5],  # Face 3
        [1, 5, 6, 2],  # Face 4
        [5, 1, 0, 4],  # Face 5
        [2, 6, 7, 3],  # Face 6
    ])
    nodes = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    return x, y, z, faces, nodes


def create_FE_two_bricks() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Two bricks sharing a face, with explicit face neighbor connectivity.

    Node Map:
      7 ----- 6 ---- 11
     /|      /|      /|
    4 ----- 5 ---- 10 |
    | 3 ----|-2 ----|-8
    |/      |/      |/
    0 ----- 1 ----- 9

    Notes:
    - For the first FEBRICK cell shown above the faces are numbered
    f0: n0-n4-n7-n3 left face (Imin)
    f1: n1-n2-n6-n5 right face (Imax)
    f2: n0-n1-n5-n5 front face (Jmin)
    f3: n2-n3-n7-n6 back face (Jmax)
    f4: n0-n1-n2-n3 bottom face (Kmin)
    f5: n4-n5-n6-n7 top face (Kmax)
    """
    points = np.array([
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [1, 1, 1],  # 6
        [0, 1, 1],  # 7
        [2, 0, 0],  # 8
        [2, 1, 0],  # 9
        [2, 0, 1],  # 10
        [2, 1, 1],  # 11
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    nodes = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],  # cell 0
        [1, 8, 9, 2, 5, 10, 11, 6],  # cell 1
    ])
    # 6 faces per cell: left, right, front, back, bottom, top. cz1/cz2 and fz
    # (the face ordinal) are all.
    face_neighbors = np.array([
        [0, 1, 1],  # cell 0: right face (f1) neighbors cell 1 (cz1, fz, cz2)
        [1, 0, 0],  # cell 1: left face (f0) neighbors cell 0
    ])
    return x, y, z, nodes, face_neighbors


def create_FE_mixed() -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Create a mixed FE zone with one tet, one pyramid, one prism, one hex.

    All cells share a common base region, each demonstrating a different
    3D element type. Returns x, y, z coordinates, concatenated node map,
    and num_nodes_per_element array (one entry per cell).

    Cell layout:
        Cell 0: Tet       (4 nodes)
        Cell 1: Pyramid   (5 nodes)
        Cell 2: Prism     (6 nodes)
        Cell 3: Hex       (8 nodes)
    """
    points = np.array(
        [
            # Base layer (z=0)
            [0.0, 0.0, 0.0],  #  0
            [1.0, 0.0, 0.0],  #  1
            [1.0, 1.0, 0.0],  #  2
            [0.0, 1.0, 0.0],  #  3
            [2.0, 0.0, 0.0],  #  4
            [2.0, 1.0, 0.0],  #  5
            [3.0, 0.0, 0.0],  #  6
            [3.0, 1.0, 0.0],  #  7
            [4.0, 0.0, 0.0],  #  8
            [4.0, 1.0, 0.0],  #  9
            # Mid layer (z=1)
            [2.0, 0.0, 1.0],  # 10
            [2.0, 1.0, 1.0],  # 11
            [3.0, 0.0, 1.0],  # 12
            [3.0, 1.0, 1.0],  # 13
            [4.0, 0.0, 1.0],  # 14
            [4.0, 1.0, 1.0],  # 15
            # Apex nodes
            [0.5, 0.5, 1.0],  # 16 - tet apex
            [1.5, 0.5, 1.0],  # 17 - pyramid apex
        ],
        dtype=np.float32,
    )

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Cell 0: Tet - base triangle 0,1,2 + apex 16
    # Cell 1: Pyramid - base quad 1,4,5,2 + apex 17
    # Cell 2: Prism - bottom tri 4,6,5 + top tri 10,12,11
    # Cell 3: Hex - bottom quad 6,8,9,7 + top quad 12,14,15,13
    # fmt: off
    node_map = np.array(
        [
            0, 1, 2, 16,  # tet: 4 nodes
            1, 4, 5, 2, 17,  # pyramid: 5 nodes
            4, 6, 5, 10, 12, 11,  # prism: 6 nodes
            6, 8, 9, 7, 12, 14, 15, 13,  # hex: 8 nodes
        ],
    )
    # fmt: on

    # One entry per cell: number of nodes in that cell
    num_nodes_per_element = np.array([4, 5, 6, 8])

    return x, y, z, node_map, num_nodes_per_element


# ======================================================================================
# Face-based (FEPOLYGON / FEPOLYHEDRON) zones
#
# Notesl:
# - Nodes and elements start at 0, -1 in the left/right arrays means no element, and -(b +
#   2) references connected boundary face b of the boundary connection arrays. A polygon
#   face (edge) runs from its first to its second node with the left element on its
#   left. A polyhedron face lists its nodes so the right-hand rule normal points out of
#   the left element.
# The functions return:
#   polygons:    x, y, face_nodes (num_faces, 2), face_left, face_right
#   polyhedra:   x, y, z, face_nodes (flat), face_node_counts, face_left, face_right
# The "coarse" boundary connection zones also return bconn_counts, bconn_elems and
# bconn_zones, which hold the elements and zones that every connected boundary face
# touches.
# ======================================================================================


def _split_faces(
    faces: list[tuple[list[int], int, int]],
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Split ``(nodes, left, right)`` rows into face_nodes, counts, left and right."""
    face_nodes = np.concatenate([f[0] for f in faces])
    counts = np.array([len(f[0]) for f in faces])
    left = np.array([f[1] for f in faces])
    right = np.array([f[2] for f in faces])
    return face_nodes, counts, left, right


def create_poly_octagon() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    r"""Create coordinates and face map for a single octagon FEPOLYGON cell.

    Nodes, faces (F) and element (E), counterclockwise:

              F4
           5 ---- 4
       F5 /        \ F3
         6          3
      F6 |    E0    | F2
         7          2
       F7 \        / F1
           0 ---- 1
              F0

    Face i runs from node i to node i + 1 (F7 closes the loop from node 7 to node 0).
    The cell is on the left of every face and nothing is on the right.
    """
    points = np.array([
        [0.25, 0],  # 0
        [0.75, 0],  # 1
        [1, 0.25],  # 2
        [1, 0.75],  # 3
        [0.75, 1],  # 4
        [0.25, 1],  # 5
        [0, 0.75],  # 6
        [0, 0.25],  # 7
    ])
    x = points[:, 0]
    y = points[:, 1]
    faces = np.array([
        # node 0, node 1, left, right
        [0, 1, 0, -1],  # F0
        [1, 2, 0, -1],  # F1
        [2, 3, 0, -1],  # F2
        [3, 4, 0, -1],  # F3
        [4, 5, 0, -1],  # F4
        [5, 6, 0, -1],  # F5
        [6, 7, 0, -1],  # F6
        [7, 0, 0, -1],  # F7
    ])
    return x, y, faces[:, :2], faces[:, 2], faces[:, 3]


def create_poly_two_cells() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    r"""Create coordinates and face map for a pentagon and a triangle FEPOLYGON.

    The pentagon E0 and the triangle E1 share face F1 (not to scale):
            3
       F3 /    \ F2
        /        \      F6
      4            2 ------ 5
      \     E0     |       /
       \           | E1  /
     F4 \       F1 |   / F5
         \         | /
          0 ------ 1
             F0
    The pentagon has faces F0 to F4 and the triangle F1, F5 and F6. F1 is the only face
    with an element on both sides.
    """
    points = np.array([
        [0, 0],  # 0
        [1, 0],  # 1
        [1, 1],  # 2
        [0.5, 1.5],  # 3
        [-0.5, 1],  # 4
        [2, 1],  # 5
    ])
    x = points[:, 0]
    y = points[:, 1]
    faces = np.array([
        # node 0, node 1, left, right
        [0, 1, 0, -1],  # F0
        [1, 2, 0, 1],  # F1
        [2, 3, 0, -1],  # F2
        [3, 4, 0, -1],  # F3
        [4, 0, 0, -1],  # F4
        [1, 5, 1, -1],  # F5
        [5, 2, 1, -1],  # F6
    ])
    return x, y, faces[:, :2], faces[:, 2], faces[:, 3]


def create_poly_hanging_ladder() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Create coordinates and face map for FEPOLYGON cells with hanging nodes.

    Three columns hold 1, 2 and 4 cells, each cell half as tall as the one before it.
    A cell touches two cells of the next column and carries the node between them, so
    E0, E1 and E2 are pentagons (hanging nodes 3, 6 and 8) and E3 to E6 are quads:

     1 ----------- 4 ----------- 9 ----------- 14
     |             |             |    E6       |
     |             |    E2       8 ----------- 13
     |             |             |    E5       |
     |    E0       3 ----------- 7 ----------- 12
     |             |             |    E4       |
     |             |    E1       6 ----------- 11
     |             |             |    E3       |
     0 ----------- 2 ----------- 5 ----------- 10

    Column k spans x = k to k + 1 and the domain is 4 tall. Faces are listed cell by
    cell, so each face belongs to the first cell that uses it.
    """
    points = np.array([
        [0, 0],  # 0
        [0, 4],  # 1
        [1, 0],  # 2
        [1, 2],  # 3
        [1, 4],  # 4
        [2, 0],  # 5
        [2, 1],  # 6
        [2, 2],  # 7
        [2, 3],  # 8
        [2, 4],  # 9
        [3, 0],  # 10
        [3, 1],  # 11
        [3, 2],  # 12
        [3, 3],  # 13
        [3, 4],  # 14
    ])
    x = points[:, 0]
    y = points[:, 1]
    faces = np.array([
        # node 0, node 1, left, right
        [0, 2, 0, -1],  # F0
        [2, 3, 0, 1],  # F1
        [3, 4, 0, 2],  # F2
        [4, 1, 0, -1],  # F3
        [1, 0, 0, -1],  # F4
        [2, 5, 1, -1],  # F5
        [5, 6, 1, 3],  # F6
        [6, 7, 1, 4],  # F7
        [7, 3, 1, 2],  # F8
        [7, 8, 2, 5],  # F9
        [8, 9, 2, 6],  # F10
        [9, 4, 2, -1],  # F11
        [5, 10, 3, -1],  # F12
        [10, 11, 3, -1],  # F13
        [11, 6, 3, 4],  # F14
        [11, 12, 4, -1],  # F15
        [12, 7, 4, 5],  # F16
        [12, 13, 5, -1],  # F17
        [13, 8, 5, 6],  # F18
        [13, 14, 6, -1],  # F19
        [14, 9, 6, -1],  # F20
    ])
    return x, y, faces[:, :2], faces[:, 2], faces[:, 3]


def create_poly_golden_spiral() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Create coordinates and face map for squares laid out as a golden spiral.

    Five squares with sides 1, 1, 2, 3 and 5 tile a rectangle. Each square is attached
    to the long side of the rectangle built so far and turns the spiral by 90 degrees.
    The larger squares collect hanging nodes where they touch several smaller ones, so
    E0 and E1 have 4 faces, E2 and E3 have 5 and E4 has 6 (nodes 6 and 3 on its top):

     2 ----------------- 5 ----------- 11
     |                   |             |
     |                   |    E2       |
     |       E3          |             |
     |                   4 ----- 7 --- 10
     |                   |  E0  |  E1  |
     1 ----------------- 3 ----- 6 ---- 9
     |                                 |
     |                                 |
     |                                 |
     |              E4                 |
     |                                 |
     |                                 |
     |                                 |
     0 ------------------------------- 8

    Square k has area side(k)**2 with sides 1, 1, 2, 3, 5. Faces are listed cell by
    cell, so each face belongs to the first cell that uses it.
    """
    points = np.array([
        [-3, -5],  # 0
        [-3, 0],  # 1
        [-3, 3],  # 2
        [0, 0],  # 3
        [0, 1],  # 4
        [0, 3],  # 5
        [1, 0],  # 6
        [1, 1],  # 7
        [2, -5],  # 8
        [2, 0],  # 9
        [2, 1],  # 10
        [2, 3],  # 11
    ])
    x = points[:, 0]
    y = points[:, 1]
    faces = np.array([
        # node 0, node 1, left, right
        [3, 6, 0, 4],  # F0
        [6, 7, 0, 1],  # F1
        [7, 4, 0, 2],  # F2
        [4, 3, 0, 3],  # F3
        [6, 9, 1, 4],  # F4
        [9, 10, 1, -1],  # F5
        [10, 7, 1, 2],  # F6
        [10, 11, 2, -1],  # F7
        [11, 5, 2, -1],  # F8
        [5, 4, 2, 3],  # F9
        [1, 3, 3, 4],  # F10
        [5, 2, 3, -1],  # F11
        [2, 1, 3, -1],  # F12
        [0, 8, 4, -1],  # F13
        [8, 9, 4, -1],  # F14
        [1, 0, 4, -1],  # F15
    ])
    return x, y, faces[:, :2], faces[:, 2], faces[:, 3]


def create_poly_bconn_fine() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Create zone 0 of a pair of FEPOLYGON zones joined by boundary connections.

    Four fine cells stacked in x = 1 to 2. The coarse zone (see
    :func:`create_poly_bconn_coarse`) lies against their left side, x = 1:

     4 ----------- 9
     |     E3      |
     3 ----------- 8
     |     E2      |
     2 ----------- 7
     |     E1      |
     1 ----------- 6
     |     E0      |
     0 ----------- 5

    The left faces (F3, F6, F9 and F12) have no element on their right because the
    neighbors are in another zone, which only the coarse zone references.
    """
    points = np.array([
        [1, 0],  # 0
        [1, 1],  # 1
        [1, 2],  # 2
        [1, 3],  # 3
        [1, 4],  # 4
        [2, 0],  # 5
        [2, 1],  # 6
        [2, 2],  # 7
        [2, 3],  # 8
        [2, 4],  # 9
    ])
    x = points[:, 0]
    y = points[:, 1]
    faces = np.array([
        # node 0, node 1, left, right
        [0, 5, 0, -1],  # F0
        [5, 6, 0, -1],  # F1
        [6, 1, 0, 1],  # F2
        [1, 0, 0, -1],  # F3
        [6, 7, 1, -1],  # F4
        [7, 2, 1, 2],  # F5
        [2, 1, 1, -1],  # F6
        [7, 8, 2, -1],  # F7
        [8, 3, 2, 3],  # F8
        [3, 2, 2, -1],  # F9
        [8, 9, 3, -1],  # F10
        [9, 4, 3, -1],  # F11
        [4, 3, 3, -1],  # F12
    ])
    return x, y, faces[:, :2], faces[:, 2], faces[:, 3]


def create_poly_bconn_coarse() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Create zone 1 of a pair of FEPOLYGON zones joined by boundary connections.

    Two coarse cells in x = 0 to 1. Each right face (F1 and F4) is one face against two
    fine cells of zone 0 (see :func:`create_poly_bconn_fine`), so its right element is
    a connected boundary face. TecIO only accepts connections to a zone that already
    exists, so this zone is written second.

     2 ----------- 5            Right element   Connected to zone 0
     |             |            F1: -2          elements 0 and 1
     |     E1      |            F4: -3          elements 2 and 3
     |             |
     1 ----------- 4
     |             |
     |     E0      |
     |             |
     0 ----------- 3

    Returns the usual polygon arrays followed by bconn_counts, bconn_elems and
    bconn_zones.
    """
    points = np.array([
        [0, 0],  # 0
        [0, 2],  # 1
        [0, 4],  # 2
        [1, 0],  # 3
        [1, 2],  # 4
        [1, 4],  # 5
    ])
    x = points[:, 0]
    y = points[:, 1]
    faces = np.array([
        # node 0, node 1, left, right
        [0, 3, 0, -1],  # F0
        [3, 4, 0, -2],  # F1, connected boundary face 0
        [4, 1, 0, 1],  # F2
        [1, 0, 0, -1],  # F3
        [4, 5, 1, -3],  # F4, connected boundary face 1
        [5, 2, 1, -1],  # F5
        [2, 1, 1, -1],  # F6
    ])
    bconn_counts = np.array([2, 2])  # connections of each connected boundary face
    bconn_elems = np.array([0, 1, 2, 3])  # connected element of each connection
    bconn_zones = np.array([0, 0, 0, 0])  # connected zone of each connection
    return (
        x,
        y,
        faces[:, :2],
        faces[:, 2],
        faces[:, 3],
        bconn_counts,
        bconn_elems,
        bconn_zones,
    )


def create_polyhedron_cube() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Create coordinates and face map for a single cube FEPOLYHEDRON cell.

    Node ordering is the same as for a FEBRICK (x right, y back, z up):

        6 ------- 7
       /|        /|
      4 ------- 5 |
      | 2 ------|-3
      |/        |/
      0 ------- 1

    The six faces are F0 -x, F1 +x, F2 -y, F3 +y, F4 -z and F5 +z. The cell E0 is on
    the left of every face and nothing is on the right.
    """
    points = np.array([
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [0, 1, 0],  # 2
        [1, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [0, 1, 1],  # 6
        [1, 1, 1],  # 7
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    faces = [
        # nodes, left, right
        ([0, 4, 6, 2], 0, -1),  # F0
        ([1, 3, 7, 5], 0, -1),  # F1
        ([0, 1, 5, 4], 0, -1),  # F2
        ([2, 6, 7, 3], 0, -1),  # F3
        ([0, 2, 3, 1], 0, -1),  # F4
        ([4, 5, 7, 6], 0, -1),  # F5
    ]
    return x, y, z, *_split_faces(faces)


def create_polyhedron_cube_pair() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Create coordinates and face map for two cubes sharing a face.

    E1 sits on top of E0 and they share F5, the only face with an element on both sides
    (x right, y back, z up):

        10 ------ 11
        /|        /|
       8 ------- 9 |
       | |       | |        E1
       | 6 ------|-7        F5 (shared, nodes 4, 5, 6, 7)
       |/|       |/|
       4 ------- 5 |
       | |       | |        E0
       | 2 ------|-3
       |/        |/
       0 ------- 1

    The faces of each cube are numbered -x, +x, -y, +y, -z, +z and the shared face is
    the +z face of E0 and the -z face of E1, so E1 has no -z face of its own.
    """
    points = np.array([
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [0, 1, 0],  # 2
        [1, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [0, 1, 1],  # 6
        [1, 1, 1],  # 7
        [0, 0, 2],  # 8
        [1, 0, 2],  # 9
        [0, 1, 2],  # 10
        [1, 1, 2],  # 11
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    faces = [
        # nodes, left, right
        ([0, 4, 6, 2], 0, -1),  # F0, E0 -x
        ([1, 3, 7, 5], 0, -1),  # F1, E0 +x
        ([0, 1, 5, 4], 0, -1),  # F2, E0 -y
        ([2, 6, 7, 3], 0, -1),  # F3, E0 +y
        ([0, 2, 3, 1], 0, -1),  # F4, E0 -z
        ([4, 5, 7, 6], 0, 1),  # F5, E0 +z and E1 -z
        ([4, 8, 10, 6], 1, -1),  # F6, E1 -x
        ([5, 7, 11, 9], 1, -1),  # F7, E1 +x
        ([4, 5, 9, 8], 1, -1),  # F8, E1 -y
        ([6, 10, 11, 7], 1, -1),  # F9, E1 +y
        ([8, 9, 11, 10], 1, -1),  # F10, E1 +z
    ]
    return x, y, z, *_split_faces(faces)


def create_polyhedron_mixed_faces() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    r"""Create coordinates and face map for polyhedra with 3, 4 and 6 node faces.

    E0 is a unit cube, E1 a pyramid on its top face and E2 a triangular prism on its +x
    face. The hexagonal prism E3 stands apart. The cube has the same nodes as in
    :func:`create_polyhedron_cube`. Side view along y (back nodes hide behind the
    front ones), with x right and z up:
              8                           8
             / \                         / \
        F9 /     \ F7               F9 /     \ F7
         /  E1/F6  \                 /  E1/F8  \
        4 --------- 5 .             7 --------- 6 .
        |    F1     |   \  F13      |    F1     |   \  F13
        |           |     \         |           |     \
     F5 |  E0/F2    |E2/F10 9    F5 |  E0/F4    |E2/F11 10
        |        F3 |     /         |        F3 |     /
        |           |   / F12       |           |   / F12
        0 --------- 1 '             3 --------- 2 '
             F0                           F0
        (Front, y = 0)                (Rear, y = 1)

    The pyramid base (F1) and the prism side (F3) are the only shared faces. The
    hexagonal prism is seen from above with its bottom ring 11 to 16 (z = 0) and top
    ring 17 to 22 (z = 1), so node n + 6 is above node n:
               F17                      F17
           13 ----- 12              19 ----- 18
      F18 /           \ F16    F18 /           \ F16
         /             \          /             \
       14    E3/F14     11      20    E3/F15     17
         \             /          \             /
      F19 \           / F21    F19 \           / F21
           15 ----- 16              21 ----- 22
               F20                      F20
         (Bottom, z = 0)            (Top, z = 1)
    Faces: F0 to F5 cube, F6 to F9 pyramid, F10 to F13 prism, F14 to F21 hexagonal
    prism (F14 bottom, F15 top, then six sides).
    """
    h = np.sqrt(3) / 4  # half the hexagon height
    points = np.array([
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [1, 1, 1],  # 6
        [0, 1, 1],  # 7
        [0.5, 0.5, 1.8],  # 8
        [1.8, 0, 0.5],  # 9
        [1.8, 1, 0.5],  # 10
        [4.5, 0.5, 0],  # 11
        [4.25, 0.5 + h, 0],  # 12
        [3.75, 0.5 + h, 0],  # 13
        [3.5, 0.5, 0],  # 14
        [3.75, 0.5 - h, 0],  # 15
        [4.25, 0.5 - h, 0],  # 16
        [4.5, 0.5, 1],  # 17
        [4.25, 0.5 + h, 1],  # 18
        [3.75, 0.5 + h, 1],  # 19
        [3.5, 0.5, 1],  # 20
        [3.75, 0.5 - h, 1],  # 21
        [4.25, 0.5 - h, 1],  # 22
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    faces = [
        # nodes, left, right
        ([0, 3, 2, 1], 0, -1),  # F0, cube -z
        ([4, 5, 6, 7], 0, 1),  # F1, cube +z and pyramid base
        ([0, 1, 5, 4], 0, -1),  # F2, cube -y
        ([1, 2, 6, 5], 0, 2),  # F3, cube +x and prism side
        ([2, 3, 7, 6], 0, -1),  # F4, cube +y
        ([0, 4, 7, 3], 0, -1),  # F5, cube -x
        ([4, 5, 8], 1, -1),  # F6, pyramid -y side
        ([5, 6, 8], 1, -1),  # F7, pyramid +x side
        ([6, 7, 8], 1, -1),  # F8, pyramid +y side
        ([4, 8, 7], 1, -1),  # F9, pyramid -x side
        ([1, 9, 5], 2, -1),  # F10, prism -y end
        ([2, 6, 10], 2, -1),  # F11, prism +y end
        ([1, 2, 10, 9], 2, -1),  # F12, prism bottom slope
        ([5, 9, 10, 6], 2, -1),  # F13, prism top slope
        ([11, 16, 15, 14, 13, 12], 3, -1),  # F14, hexagon bottom
        ([17, 18, 19, 20, 21, 22], 3, -1),  # F15, hexagon top
        ([11, 12, 18, 17], 3, -1),  # F16
        ([12, 13, 19, 18], 3, -1),  # F17
        ([13, 14, 20, 19], 3, -1),  # F18
        ([14, 15, 21, 20], 3, -1),  # F19
        ([15, 16, 22, 21], 3, -1),  # F20
        ([11, 17, 22, 16], 3, -1),  # F21
    ]
    return x, y, z, *_split_faces(faces)


def create_polyhedron_hanging_columns() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Create coordinates and face map for one coarse cube against four fine cubes.

    The coarse cube E0 spans x = 0 to 2, y = 0 to 2 and z = 0 to 2, with nodes 0 to 7
    numbered like the cube in :func:`create_polyhedron_cube`. The fine cubes E1 to E4
    are 1 by 1 by 1 and fill x = 2 to 3, so the +x side of E0 is split into four faces
    (F1 to F4) with hanging nodes 8 to 12. The two node planes seen from +x (y right,
    z up):

      x = 2                x = 3
      5 ---- 12 --- 7      19 --- 20 --- 21
      |  E3  |  E4  |      |  E3  |  E4  |
      9 ---- 10 --- 11     16 --- 17 --- 18
      |  E1  |  E2  |      |  E1  |  E2  |
      1 ---- 8 ---- 3      13 --- 14 --- 15

    Faces are listed cell by cell, so each face belongs to the first cell that uses it.
    E0 has 9 faces: F0 and F5 to F8 on the outside and F1 to F4 shared with E1 to E4.
    """
    points = np.array([
        [0, 0, 0],  # 0
        [2, 0, 0],  # 1
        [0, 2, 0],  # 2
        [2, 2, 0],  # 3
        [0, 0, 2],  # 4
        [2, 0, 2],  # 5
        [0, 2, 2],  # 6
        [2, 2, 2],  # 7
        [2, 1, 0],  # 8
        [2, 0, 1],  # 9
        [2, 1, 1],  # 10
        [2, 2, 1],  # 11
        [2, 1, 2],  # 12
        [3, 0, 0],  # 13
        [3, 1, 0],  # 14
        [3, 2, 0],  # 15
        [3, 0, 1],  # 16
        [3, 1, 1],  # 17
        [3, 2, 1],  # 18
        [3, 0, 2],  # 19
        [3, 1, 2],  # 20
        [3, 2, 2],  # 21
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    faces = [
        # nodes, left, right
        ([0, 4, 6, 2], 0, -1),  # F0, E0 -x
        ([1, 8, 10, 9], 0, 1),  # F1, E0 +x against E1
        ([3, 11, 10, 8], 0, 2),  # F2, E0 +x against E2
        ([5, 9, 10, 12], 0, 3),  # F3, E0 +x against E3
        ([7, 12, 10, 11], 0, 4),  # F4, E0 +x against E4
        ([0, 1, 5, 4], 0, -1),  # F5, E0 -y
        ([2, 6, 7, 3], 0, -1),  # F6, E0 +y
        ([0, 2, 3, 1], 0, -1),  # F7, E0 -z
        ([4, 5, 7, 6], 0, -1),  # F8, E0 +z
        ([13, 14, 17, 16], 1, -1),  # F9, E1 +x
        ([1, 13, 16, 9], 1, -1),  # F10, E1 -y
        ([8, 10, 17, 14], 1, 2),  # F11, E1 +y and E2 -y
        ([1, 8, 14, 13], 1, -1),  # F12, E1 -z
        ([9, 16, 17, 10], 1, 3),  # F13, E1 +z and E3 -z
        ([14, 15, 18, 17], 2, -1),  # F14, E2 +x
        ([3, 11, 18, 15], 2, -1),  # F15, E2 +y
        ([3, 15, 14, 8], 2, -1),  # F16, E2 -z
        ([10, 17, 18, 11], 2, 4),  # F17, E2 +z and E4 -z
        ([16, 17, 20, 19], 3, -1),  # F18, E3 +x
        ([5, 9, 16, 19], 3, -1),  # F19, E3 -y
        ([10, 12, 20, 17], 3, 4),  # F20, E3 +y and E4 -y
        ([5, 19, 20, 12], 3, -1),  # F21, E3 +z
        ([17, 18, 21, 20], 4, -1),  # F22, E4 +x
        ([7, 21, 18, 11], 4, -1),  # F23, E4 +y
        ([7, 12, 20, 21], 4, -1),  # F24, E4 +z
    ]
    return x, y, z, *_split_faces(faces)


def create_polyhedron_bconn_fine() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Create zone 0 of a pair of FEPOLYHEDRON zones joined by boundary connections.

    A 2 by 2 block of unit cubes in x = 2 to 3, with the coarse zone (see
    :func:`create_polyhedron_bconn_coarse`) against the x = 2 side. The two node
    planes seen from +x (y right, z up):

      x = 2                x = 3
      6 ---- 7 ---- 8      15 --- 16 --- 17
      |  E2  |  E3  |      |  E2  |  E3  |
      3 ---- 4 ---- 5      12 --- 13 --- 14
      |  E0  |  E1  |      |  E0  |  E1  |
      0 ---- 1 ---- 2      9 ---- 10 --- 11

    The -x faces (F0, F6, F11 and F16) have no element on their right because their
    neighbor is in another zone, which only the coarse zone references.
    """
    points = np.array([
        [2, 0, 0],  # 0
        [2, 1, 0],  # 1
        [2, 2, 0],  # 2
        [2, 0, 1],  # 3
        [2, 1, 1],  # 4
        [2, 2, 1],  # 5
        [2, 0, 2],  # 6
        [2, 1, 2],  # 7
        [2, 2, 2],  # 8
        [3, 0, 0],  # 9
        [3, 1, 0],  # 10
        [3, 2, 0],  # 11
        [3, 0, 1],  # 12
        [3, 1, 1],  # 13
        [3, 2, 1],  # 14
        [3, 0, 2],  # 15
        [3, 1, 2],  # 16
        [3, 2, 2],  # 17
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    faces = [
        # nodes, left, right
        ([0, 3, 4, 1], 0, -1),  # F0, E0 -x
        ([9, 10, 13, 12], 0, -1),  # F1, E0 +x
        ([0, 9, 12, 3], 0, -1),  # F2, E0 -y
        ([1, 4, 13, 10], 0, 1),  # F3, E0 +y and E1 -y
        ([0, 1, 10, 9], 0, -1),  # F4, E0 -z
        ([3, 12, 13, 4], 0, 2),  # F5, E0 +z and E2 -z
        ([1, 4, 5, 2], 1, -1),  # F6, E1 -x
        ([10, 11, 14, 13], 1, -1),  # F7, E1 +x
        ([2, 5, 14, 11], 1, -1),  # F8, E1 +y
        ([1, 2, 11, 10], 1, -1),  # F9, E1 -z
        ([4, 13, 14, 5], 1, 3),  # F10, E1 +z and E3 -z
        ([3, 6, 7, 4], 2, -1),  # F11, E2 -x
        ([12, 13, 16, 15], 2, -1),  # F12, E2 +x
        ([3, 12, 15, 6], 2, -1),  # F13, E2 -y
        ([4, 7, 16, 13], 2, 3),  # F14, E2 +y and E3 -y
        ([6, 15, 16, 7], 2, -1),  # F15, E2 +z
        ([4, 7, 8, 5], 3, -1),  # F16, E3 -x
        ([13, 14, 17, 16], 3, -1),  # F17, E3 +x
        ([5, 8, 17, 14], 3, -1),  # F18, E3 +y
        ([7, 16, 17, 8], 3, -1),  # F19, E3 +z
    ]
    return x, y, z, *_split_faces(faces)


def create_polyhedron_bconn_coarse() -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
]:
    """Create zone 1 of a pair of FEPOLYHEDRON zones joined by boundary connections.

    One cube spanning x, y, z = 0 to 2 with nodes numbered like a FEBRICK (see
    :func:`create_polyhedron_cube`):

        6 ------- 7
       /|        /|
      4 ------- 5 |
      | 2 ------|-3
      |/        |/
      0 ------- 1

    The +x face F1 (nodes 1, 3, 7, 5) is a single face against all four cubes of zone 0
    (see :func:`create_polyhedron_bconn_fine`), so its right element is connected
    boundary face 0, written as -2. TecIO only accepts connections to a zone that
    already exists, so this zone is written second.

    Returns the usual polyhedron arrays followed by bconn_counts, bconn_elems and
    bconn_zones.
    """
    points = np.array([
        [0, 0, 0],  # 0
        [2, 0, 0],  # 1
        [0, 2, 0],  # 2
        [2, 2, 0],  # 3
        [0, 0, 2],  # 4
        [2, 0, 2],  # 5
        [0, 2, 2],  # 6
        [2, 2, 2],  # 7
    ])
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    faces = [
        # nodes, left, right
        ([0, 4, 6, 2], 0, -1),  # F0, -x
        ([1, 3, 7, 5], 0, -2),  # F1, +x, connected boundary face 0
        ([0, 1, 5, 4], 0, -1),  # F2, -y
        ([2, 6, 7, 3], 0, -1),  # F3, +y
        ([0, 2, 3, 1], 0, -1),  # F4, -z
        ([4, 5, 7, 6], 0, -1),  # F5, +z
    ]
    bconn_counts = np.array([4])  # connections of each connected boundary face
    bconn_elems = np.array([0, 1, 2, 3])  # connected element of each connection
    bconn_zones = np.array([0, 0, 0, 0])  # connected zone of each connection
    return (
        x,
        y,
        z,
        *_split_faces(faces),
        bconn_counts,
        bconn_elems,
        bconn_zones,
    )
