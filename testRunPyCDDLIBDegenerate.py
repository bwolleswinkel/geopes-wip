"""Script to test whether pycddlib library can handle degenerate cases without errors"""

import numpy as np
from numpy.typing import NDArray
import cdd


# ------ FUNCTIONS ------


def enum_facets(verts: NDArray) -> NDArray:
    """Enumerate the facets of a polytope defined by its vertices using pycddlib.
    
    Parameters
    ----------
    verts : NDArray
        An (n, k) array of n vertices in k-dimensional space.
        
    Returns
    -------
    Ab : NDArray
        An (m, n + 1) array of m facet normals.
    
    """
    # Create cdd matrix from vertices
    mat = cdd.matrix_from_array(np.column_stack((np.ones(verts.shape[1]), verts.T)), rep_type=cdd.RepType.GENERATOR)
    poly = cdd.polyhedron_from_matrix(mat)
    
    # Get inequalities (facets)
    hmat = np.array(cdd.copy_inequalities(poly).array).astype(float)
    Ab = np.column_stack((-hmat[:, 1:], hmat[:, 0]))  # Convert to standard Ax <= b format

    return Ab


def enum_verts(Ab: NDArray) -> tuple[NDArray, NDArray]:
    """Enumerate the vertices and rays of a polytope defined by its facets using pycddlib.
    
    Parameters
    ----------
    Ab : NDArray
        An (m, n + 1) array of m facet normals.
        
    Returns
    -------
    verts : NDArray
        An (n, k) array of n vertices in k-dimensional space.
    rays : NDArray
        An (n, l) array of n rays in k-dimensional space.
    
    """
    # Create cdd matrix from inequalities
    cdd_input = np.column_stack((Ab[:, -1], -Ab[:, :-1]))
    
    mat = cdd.matrix_from_array(cdd_input, rep_type=cdd.RepType.INEQUALITY)
    poly = cdd.polyhedron_from_matrix(mat)
    
    # Get generators (vertices)
    gmat = np.array(cdd.copy_generators(poly).array).astype(float)
    
    # Separate vertices and rays
    vertex_mask = gmat[:, 0] == 1.0
    ray_mask = gmat[:, 0] == 0.0
    
    vertices = gmat[vertex_mask, 1:] if np.any(vertex_mask) else np.empty((0, gmat.shape[1]-1))
    rays = gmat[ray_mask, 1:] if np.any(ray_mask) else np.empty((0, gmat.shape[1]-1))
    
    # Format vertices and rays
    if len(vertices) > 0:
        verts = vertices.T  # Transpose to get (k, n) format
    else:
        verts = np.empty((gmat.shape[1]-1, 0))  # Empty array with proper dimensions
        
    if len(rays) > 0:
        rays_formatted = rays.T  # Transpose to get (k, l) format
    else:
        rays_formatted = np.empty((gmat.shape[1]-1, 0))  # Empty array with proper dimensions

    return verts, rays_formatted


# ------ SCRIPT ------


def main() -> None:
    """Main function to test degenerate cases."""
    
    V_single_point = np.array([[2.0], [2.0]])  # Single point [2, 2] in 2D space
    
    Ab_single_point = enum_facets(V_single_point)
    print(f"Facets for single point:\n{Ab_single_point + 0}\n")  # NOTE: Incorrect/insufficient facets

    V_empty = np.empty((2, 0))  # No vertices in 2D space

    try:
        Ab_empty = enum_facets(V_empty)
        print(f"Facets for empty polytope:\n{Ab_empty + 0}\n")
    except RuntimeError as e:
        print(f"RuntimeError: {e}\n")

    V_line = np.array([[1, 1], [2, 2]])  # Line segment from [1, 1] to [2, 2] in 2D space

    Ab_line = enum_facets(V_line)
    print(f"Facets for line segment:\n{Ab_line + 0}\n")  # NOTE: Incorrect/insufficient facets

    A_unbounded_single_vert_origin, b_unbounded_single_vert_origin = np.array([[-1, 0], [0, -1]]), np.array([0, 0])  # Unbounded polyhedron (first quadrant) in 2D space 

    V_unbounded_single_vert_origin, rays_unbounded_single_vert_origin = enum_verts(np.column_stack((A_unbounded_single_vert_origin, b_unbounded_single_vert_origin)))
    print(f"Vertices for unbounded polyhedron:\n{V_unbounded_single_vert_origin + 0}")  # NOTE: Incorrect return type; only returns the rays, not the vertex at [0, 0]
    print(f"Rays for unbounded polyhedron:\n{rays_unbounded_single_vert_origin + 0}\n")

    A_unbounded_single_vert_not_origin, b_unbounded_single_vert_not_origin = np.array([[-1, 0], [0, -1]]), np.array([-1, -1])  # Unbounded polyhedron [x_1 >= 1, x_2 >= 1] in 2D space 

    V_unbounded_single_vert_not_origin, rays_unbounded_single_vert_not_origin = enum_verts(np.column_stack((A_unbounded_single_vert_not_origin, b_unbounded_single_vert_not_origin)))
    print(f"Vertices for unbounded polyhedron:\n{V_unbounded_single_vert_not_origin + 0}")  # NOTE: Now seems to correctly return the point [1, 1]
    print(f"Rays for unbounded polyhedron:\n{rays_unbounded_single_vert_not_origin + 0}\n")

    A_unbounded_two_verts, b_unbounded_two_verts = np.array([[-1, 0], [0, -1], [1, 0]]), np.array([0, 0, 1])  # Unbounded polyhedron [1 >= x_1 >= 0, x_2 >= 0] in 2D space 

    V_unbounded_two_verts, rays_unbounded_two_verts = enum_verts(np.column_stack((A_unbounded_two_verts, b_unbounded_two_verts)))
    print(f"Vertices for unbounded polyhedron:\n{V_unbounded_two_verts + 0}")
    print(f"Rays for unbounded polyhedron:\n{rays_unbounded_two_verts + 0}\n")  # NOTE: Now only seems to return one ray (so which vertex does it associate the ray with?); should be two, identical rays

    A_unbounded_line, b_unbounded_line = np.array([[1, -1], [-1, 1]]), np.array([0, 0])  # Unbounded polyhedron (line x_1 = x_2) in 2D space

    V_unbounded_line, rays_unbounded_line = enum_verts(np.column_stack((A_unbounded_line, b_unbounded_line)))
    print(f"Vertices for unbounded polyhedron:\n{V_unbounded_line + 0}")  # NOTE: Returns no vertices, wondering if it should return one degenerate vertex (i.e., [0, 0])?
    print(f"Rays for unbounded polyhedron:\n{rays_unbounded_line + 0}\n")  # NOTE: Returns only one ray [1, 1], should return two rays [1, 1] and [-1, -1]

    A_unbounded_line_offset, b_unbounded_line_offset = np.array([[1, -1], [-1, 1]]), np.array([1, -1])  # Unbounded polyhedron (line x_1 - 1 = x_2) in 2D space

    V_unbounded_line_offset, rays_unbounded_line_offset = enum_verts(np.column_stack((A_unbounded_line_offset, b_unbounded_line_offset)))
    print(f"Vertices for unbounded polyhedron:\n{V_unbounded_line_offset + 0}")  # NOTE: Correctly return the point [1, 0], which lies on the line
    print(f"Rays for unbounded polyhedron:\n{rays_unbounded_line_offset + 0}\n")  # NOTE: Returns only one ray [1, 1], should return two rays [1, 1] and [-1, -1]

if __name__ == "__main__":
    main()