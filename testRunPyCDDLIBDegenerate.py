"""Script to test whether pycddlib library can handle degenerate cases without errors"""

import numpy as np
from numpy.typing import NDArray
import cdd


# ------ FUNCTIONS ------


# FROM: GitHub Copilot Claude Sonnet 4.5 | 2026/02/08
def enum_facets(verts: NDArray, rays: NDArray | None = None) -> tuple[NDArray, NDArray]:
    """Enumerate the facets of a polytope defined by its vertices using pycddlib.
    
    Parameters
    ----------
    verts : NDArray
        A (k, n) array of k vertices in n-dimensional space.
    rays : NDArray | None
        A (k_rays, n) array of k_rays rays in n-dimensional space.
        
    Returns
    -------
    Ab : NDArray
        An (m, n + 1) array of m facet normals.
    Ab_eq : NDArray
        An (m_eq, n + 1) array of m_eq facet normals corresponding to equalities.
    """
    # Create cdd matrix from vertices and the rays (if any)
    if rays is not None and rays.size > 0:
        if verts.size > 0:
            # Both vertices and rays present
            generators = np.vstack((verts, rays))
            ray_flags = np.concatenate((np.ones(verts.shape[0]), np.zeros(rays.shape[0])))
        else:
            # Only rays, no vertices
            generators = rays
            ray_flags = np.zeros(rays.shape[0])
        mat = cdd.matrix_from_array(np.column_stack((ray_flags, generators)), rep_type=cdd.RepType.GENERATOR)
    else:
        # Only vertices, no rays
        mat = cdd.matrix_from_array(np.column_stack((np.ones(verts.shape[0]), verts)), rep_type=cdd.RepType.GENERATOR)
    poly = cdd.polyhedron_from_matrix(mat)

    ineq = cdd.copy_inequalities(poly)
    hmat = np.array(ineq.array).astype(float)
    lin_set = ineq.lin_set
    eq_mask = np.zeros(hmat.shape[0], dtype=bool)
    for idx in lin_set:
        eq_mask[idx] = True
    h_ineq = hmat[~eq_mask] if np.any(~eq_mask) else np.empty((0, hmat.shape[1]))
    h_eq = hmat[eq_mask] if np.any(eq_mask) else np.empty((0, hmat.shape[1]))
    Ab = np.column_stack((-h_ineq[:, 1:], h_ineq[:, 0])) if h_ineq.size else np.empty((0, hmat.shape[1]))
    if Ab.size:
        A, b = Ab[:, :-1], Ab[:, -1]
        trivial = np.isclose(A, 0).all(axis=1) & ((b >= 0) | np.isclose(b, 0))
        Ab = Ab[~trivial]
    Ab_eq = np.column_stack((-h_eq[:, 1:], h_eq[:, 0])) if h_eq.size else np.empty((0, hmat.shape[1]))
    return Ab, Ab_eq


# FROM: GitHub Copilot Claude Sonnet 4.5 | 2026/02/08
def enum_verts(Ab: NDArray, Ab_eq: NDArray, add_origin_empty_verts: bool = True) -> tuple[NDArray, NDArray]:
    """Enumerate the vertices and rays of a polytope defined by its facets using pycddlib.
    
    Parameters
    ----------
    Ab : NDArray
        An (m, n + 1) array of m facet normals.
    Ab_eq : NDArray
        An (m_eq, n + 1) array of m_eq facet normals corresponding to equalities.
    add_origin_empty_verts : bool, optional
        If True, adds the origin as an explicit vertex when no vertices are found but rays exist.
        This ensures the Minkowski-Weyl representation has a non-empty vertex set.
        Default is True.

    Returns
    -------
    verts : NDArray
        A (k, n) array of k vertices in n-dimensional space.
    rays : NDArray
        A (k_rays, n) array of k_rays rays in n-dimensional space.
    
    """
    # Determine the number of columns (dimension) from Ab or Ab_eq
    if Ab.ndim >= 2 and Ab.shape[0] >= 0:  # Ab has shape information
        n_cols_ab = Ab.shape[1]
    else:
        n_cols_ab = None
    
    if Ab_eq.ndim >= 2 and Ab_eq.shape[0] >= 0:  # Ab_eq has shape information
        n_cols_eq = Ab_eq.shape[1]
    else:
        n_cols_eq = None
    
    # Convert to cdd format (b, -A)
    cdd_ineq = np.column_stack((Ab[:, -1], -Ab[:, :-1])) if Ab.shape[0] > 0 else np.empty((0, n_cols_ab if n_cols_ab else (n_cols_eq if n_cols_eq else 0)))
    cdd_eq = np.column_stack((Ab_eq[:, -1], -Ab_eq[:, :-1])) if Ab_eq.shape[0] > 0 else np.empty((0, n_cols_eq if n_cols_eq else (n_cols_ab if n_cols_ab else 0)))
    
    # Combine inequalities and equalities
    if cdd_ineq.shape[0] > 0 and cdd_eq.shape[0] > 0:
        cdd_input = np.vstack((cdd_ineq, cdd_eq))
    elif cdd_ineq.shape[0] > 0:
        cdd_input = cdd_ineq
    elif cdd_eq.shape[0] > 0:
        cdd_input = cdd_eq
    else:
        # No constraints at all: use the empty cdd_ineq or cdd_eq which already has dimension info
        # Prefer cdd_ineq since it usually has the dimension from Ab
        cdd_input = cdd_ineq if cdd_ineq.shape[1] > 0 else cdd_eq
    
    mat = cdd.matrix_from_array(cdd_input, rep_type=cdd.RepType.INEQUALITY)
    
    # Mark equality rows in lin_set
    if cdd_eq.shape[0] > 0:
        mat.lin_set = set(range(cdd_ineq.shape[0], cdd_ineq.shape[0] + cdd_eq.shape[0]))
    
    poly = cdd.polyhedron_from_matrix(mat)
    
    # Get generators (vertices and rays)
    gen_obj = cdd.copy_generators(poly)
    gmat = np.array(gen_obj.array).astype(float)
    lin_set = gen_obj.lin_set  # Indices of bidirectional lines
    
    # Check if polytope is empty (infeasible constraints)
    # Note: When there are NO constraints, gmat may be empty, which is valid (represents entire space)
    # Only raise error if constraints exist but result in infeasibility
    if (gmat.ndim == 1 or gmat.shape[0] == 0) and cdd_input.shape[0] > 0:
        # Determine dimension from input for error message
        n = cdd_input.shape[1] - 1 if cdd_input.shape[1] > 0 else 0
        raise ValueError("Polytope is empty (infeasible constraints)")
    
    # Handle case where polytope has no constraints (entire space)
    if gmat.ndim == 1 or gmat.shape[0] == 0:
        n = cdd_input.shape[1] - 1 if cdd_input.shape[1] > 0 else 0
        # Unconstrained space: origin + rays in all fundamental directions
        vertices = np.zeros((1, n))
        # Create rays along all axes (positive and negative directions)
        rays = np.vstack((np.eye(n), -np.eye(n)))  # 2n rays: +/- each axis
        return vertices, rays
    vertex_mask = gmat[:, 0] == 1.0
    ray_mask = gmat[:, 0] == 0.0
    
    vertices = gmat[vertex_mask, 1:] if np.any(vertex_mask) else np.empty((0, gmat.shape[1]-1))
    rays_and_lines = gmat[ray_mask, 1:] if np.any(ray_mask) else np.empty((0, gmat.shape[1]-1))
    
    # Expand bidirectional lines into pairs of opposite rays
    if lin_set:
        ray_list = []
        ray_indices = np.where(ray_mask)[0]
        
        for i, ray_idx in enumerate(ray_indices):
            if ray_idx in lin_set:
                # This is a bidirectional line - add both directions
                ray_list.append(rays_and_lines[i])
                ray_list.append(-rays_and_lines[i])
            else:
                # Regular ray
                ray_list.append(rays_and_lines[i])
        
        rays = np.array(ray_list) if ray_list else np.empty((0, gmat.shape[1]-1))
    else:
        rays = rays_and_lines
    
    # Add origin as vertex if no vertices found but rays exist
    # FROM: GitHub Copilot Claude Sonnet 4.5 | 2026/02/08
    if add_origin_empty_verts and vertices.shape[0] == 0 and rays.shape[0] > 0:
        n = gmat.shape[1] - 1
        vertices = np.zeros((1, n))
    
    return vertices, rays


# ------ SCRIPT ------


def main() -> None:
    # NOTE: It seems that the solution to the empty vertices case is to add the origin as an explicit vertex when no vertices are found but rays exist (using `enum_verts(add_origin_empty_verts=True)`). This seems robust, in that it 'works' for the cases when needed, and doesn't break any other cases.

    # ---- Single Point [2, 2] ----
    V_single_point = np.array([[2, 2]])  # Single point [2, 2] in 2D space (k=1, n=2)
    
    Ab_single_point, Ab_eq_single_point = enum_facets(V_single_point)
    print("---- Single Point [2, 2] ---- (V-repr)")
    print(f"Facets:\n{Ab_single_point + 0}\t\t\t(m = {Ab_single_point.shape[0]})")
    print(f"Facets (eq):\n{Ab_eq_single_point + 0}\t\t(m_eq = {Ab_eq_single_point.shape[0]})\n")
    V_single_point_reconstructed, rays_single_point_reconstructed = enum_verts(Ab_single_point, Ab_eq_single_point)
    print(f"Reconstructed vertices:\n{V_single_point_reconstructed + 0}\t\t(k = {V_single_point_reconstructed.shape[0]})")
    print(f"Reconstructed rays:\n{rays_single_point_reconstructed + 0}\t\t\t(k_rays = {rays_single_point_reconstructed.shape[0]})\n")

    # ---- Empty Polytope {} ----
    V_empty = np.empty((0, 2))  # No vertices in 2D space (k=0, n=2)

    try:
        Ab_empty, Ab_eq_empty = enum_facets(V_empty)
        print("---- Empty Polytope {} ---- (V-repr)")
        print(f"Facets:\n{Ab_empty + 0}\t\t\t(m = {Ab_empty.shape[0]})\n")
        print(f"Facets (eq):\n{Ab_eq_empty + 0}\t\t(m_eq = {Ab_eq_empty.shape[0]})\n")
    except RuntimeError as e:
        print("---- Empty Polytope {} ---- (V-repr)")
        print(f"RuntimeError: {e}\n")

    # ---- Empty Polytope (infeasible) ----
    A_empty_inf, b_empty_inf = np.array([[1, 0], [-1, 0]]), np.array([-1, -1])  # Infeasible polytope (empty) in 2D space

    try:
        V_empty_inf, rays_empty_inf = enum_verts(np.column_stack((A_empty_inf, b_empty_inf)), np.empty((0, 2)))
        print("---- Empty Polytope (infeasible) ---- (H-repr)")
        print(f"Vertices:\n{V_empty_inf + 0}\t\t\t(k = {V_empty_inf.shape[0]})\n")
        print(f"Rays:\n{rays_empty_inf + 0}\t\t(k_rays = {rays_empty_inf.shape[0]})\n")
    except ValueError as e:
        print("---- Empty Polytope (infeasible) ---- (H-repr)")
        print(f"RuntimeError: {e}\n")

    # ---- Line Segment [1, 1] -> [2, 2] ----
    V_line = np.array([[1, 1], [2, 2]])  # Line segment from [1, 1] to [2, 2] in 2D space (k=2, n=2)

    Ab_line, Ab_eq_line = enum_facets(V_line)
    print("---- Line Segment [1, 1] -> [2, 2] ---- (V-repr)")
    print(f"Facets :\n{Ab_line + 0}\t\t(m = {Ab_line.shape[0]})")
    print(f"Facets (eq):\n{Ab_eq_line + 0}\t\t(m_eq = {Ab_eq_line.shape[0]})\n")
    V_line_reconstructed, rays_line_reconstructed = enum_verts(Ab_line, Ab_eq_line)
    print(f"Reconstructed vertices:\n{V_line_reconstructed + 0}\t\t(k = {V_line_reconstructed.shape[0]})")
    print(f"Reconstructed rays:\n{rays_line_reconstructed + 0}\t\t\t(k_rays = {rays_line_reconstructed.shape[0]})\n")

    # ---- Unbounded Polyhedron (first quadrant) ----
    A_unbounded_single_vert_origin, b_unbounded_single_vert_origin = np.array([[-1, 0], [0, -1]]), np.array([0, 0])  # Unbounded polyhedron (first quadrant) in 2D space 

    V_unbounded_single_vert_origin, rays_unbounded_single_vert_origin = enum_verts(np.column_stack((A_unbounded_single_vert_origin, b_unbounded_single_vert_origin)), np.empty((0, 3)))
    print("---- Unbounded Polyhedron (first quadrant) ---- (H-repr)")
    print(f"Vertices:\n\033[35m{V_unbounded_single_vert_origin + 0}\t\t\t(k = {V_unbounded_single_vert_origin.shape[0]}) | Should be [0, 0], k = 1\033[0m")
    print(f"Rays:\n{rays_unbounded_single_vert_origin + 0}\t\t(k_rays = {rays_unbounded_single_vert_origin.shape[0]})\n")

    Ab_unbounded_single_vert_origin_reconstructed, Ab_eq_unbounded_single_vert_origin_reconstructed = enum_facets(V_unbounded_single_vert_origin, rays_unbounded_single_vert_origin)
    print(f"Reconstructed facets:\n{Ab_unbounded_single_vert_origin_reconstructed + 0}\t\t(m = {Ab_unbounded_single_vert_origin_reconstructed.shape[0]})\n")
    print(f"Reconstructed facets (eq):\n{Ab_eq_unbounded_single_vert_origin_reconstructed + 0}\t\t\t(m_eq = {Ab_eq_unbounded_single_vert_origin_reconstructed.shape[0]})\n")

    # ---- Unbounded Polyhedron [x_1 >= 1, x_2 >= 1] ----
    A_unbounded_single_vert_not_origin, b_unbounded_single_vert_not_origin = np.array([[-1, 0], [0, -1]]), np.array([-1, -1])  # Unbounded polyhedron [x_1 >= 1, x_2 >= 1] in 2D space 

    V_unbounded_single_vert_not_origin, rays_unbounded_single_vert_not_origin = enum_verts(np.column_stack((A_unbounded_single_vert_not_origin, b_unbounded_single_vert_not_origin)), np.empty((0, 3)))
    print("---- Unbounded Polyhedron [x_1 >= 1, x_2 >= 1] ---- (H-repr)")
    print(f"Vertices:\n{V_unbounded_single_vert_not_origin + 0}\t\t(k = {V_unbounded_single_vert_not_origin.shape[0]})")
    print(f"Rays:\n{rays_unbounded_single_vert_not_origin + 0}\t\t(k_rays = {rays_unbounded_single_vert_not_origin.shape[0]})\n")

    Ab_unbounded_single_vert_not_origin_reconstructed, Ab_eq_unbounded_single_vert_not_origin_reconstructed = enum_facets(V_unbounded_single_vert_not_origin, rays_unbounded_single_vert_not_origin)
    print(f"Reconstructed facets:\n{Ab_unbounded_single_vert_not_origin_reconstructed + 0}\t\t(m = {Ab_unbounded_single_vert_not_origin_reconstructed.shape[0]})\n")
    print(f"Reconstructed facets (eq):\n{Ab_eq_unbounded_single_vert_not_origin_reconstructed + 0}\t\t\t(m_eq = {Ab_eq_unbounded_single_vert_not_origin_reconstructed.shape[0]})\n")

    # ---- Unbounded Polyhedron [1 >= x_1 >= 0, x_2 >= 0] ----
    A_unbounded_two_verts, b_unbounded_two_verts = np.array([[-1, 0], [0, -1], [1, 0]]), np.array([0, 0, 1])  # Unbounded polyhedron [1 >= x_1 >= 0, x_2 >= 0] in 2D space 

    V_unbounded_two_verts, rays_unbounded_two_verts = enum_verts(np.column_stack((A_unbounded_two_verts, b_unbounded_two_verts)), np.empty((0, 3)))
    print("---- Unbounded Polyhedron [1 >= x_1 >= 0, x_2 >= 0] ---- (H-repr)")
    print(f"Vertices:\n{V_unbounded_two_verts + 0}\t\t(k = {V_unbounded_two_verts.shape[0]})")
    print(f"Rays:\n{rays_unbounded_two_verts + 0}\t\t(k_rays = {rays_unbounded_two_verts.shape[0]})\n")

    A_unbounded_two_verts_reconstructed, b_unbounded_two_verts_reconstructed = enum_facets(V_unbounded_two_verts, rays_unbounded_two_verts)
    print(f"Reconstructed facets:\n{A_unbounded_two_verts_reconstructed + 0}\t\t(m = {A_unbounded_two_verts_reconstructed.shape[0]})\n")
    print(f"Reconstructed facets (eq):\n{b_unbounded_two_verts_reconstructed + 0}\t\t\t(m_eq = {b_unbounded_two_verts_reconstructed.shape[0]})\n")

    # ---- Line x_1 = x_2 ----
    A_unbounded_line, b_unbounded_line = np.array([[1, -1], [-1, 1]]), np.array([0, 0])  # Unbounded polyhedron (line x_1 = x_2) in 2D space

    V_unbounded_line, rays_unbounded_line = enum_verts(np.column_stack((A_unbounded_line, b_unbounded_line)), np.empty((0, 3)))
    print("---- Line x_1 = x_2 ---- (H-repr)")
    print(f"Vertices:\n\033[35m{V_unbounded_line + 0}\t\t\t(k = {V_unbounded_line.shape[0]}) | Should be [0, 0], k = 1: rays should always have a 'vertex' (even if non-pointed)\033[0m")
    print(f"Rays:\n{rays_unbounded_line + 0}\t\t(k_rays = {rays_unbounded_line.shape[0]})\n")

    Ab_unbounded_line_reconstructed, Ab_eq_unbounded_line_reconstructed = enum_facets(V_unbounded_line, rays_unbounded_line)
    print(f"Reconstructed facets:\n{Ab_unbounded_line_reconstructed + 0}\t\t(m = {Ab_unbounded_line_reconstructed.shape[0]})\n")
    print(f"Reconstructed facets (eq):\n{Ab_eq_unbounded_line_reconstructed + 0}\t\t(m_eq = {Ab_eq_unbounded_line_reconstructed.shape[0]})\n")

    # ---- Line segment x_1 = x_2, x_1 >= 1 ----
    A_unbounded_line_segment, b_unbounded_line_segment = np.array([[-1, 0]]), np.array([-1])  # Unbounded polyhedron (line segment x_1 = x_2) with vertex [1, 1] in 2D space

    V_unbounded_line_segment, rays_unbounded_line_segment = enum_verts(np.column_stack((A_unbounded_line_segment, b_unbounded_line_segment)), np.column_stack((A_unbounded_line, b_unbounded_line)))
    print("---- Line segment x_1 = x_2, x_1 >= 1 ---- (H-repr)")
    print(f"Vertices:\n{V_unbounded_line_segment + 0}\t\t(k = {V_unbounded_line_segment.shape[0]})")
    print(f"Rays:\n{rays_unbounded_line_segment + 0}\t\t(k_rays = {rays_unbounded_line_segment.shape[0]})\n")

    Ab_unbounded_line_segment_reconstructed, Ab_eq_unbounded_line_segment_reconstructed = enum_facets(V_unbounded_line_segment, rays_unbounded_line_segment)
    print(f"Reconstructed facets:\n{Ab_unbounded_line_segment_reconstructed + 0}\t\t(m = {Ab_unbounded_line_segment_reconstructed.shape[0]})\n")
    print(f"Reconstructed facets (eq):\n{Ab_eq_unbounded_line_segment_reconstructed + 0}\t\t(m_eq = {Ab_eq_unbounded_line_segment_reconstructed.shape[0]})\n")

    # ---- Entire plane x_1, x_2 (empty constraints) ----
    A_entire_plane, b_entire_plane = np.empty((0, 2)), np.empty((0,))  # No constraints (entire plane) in 2D space

    V_entire_plane, rays_entire_plane = enum_verts(np.column_stack((A_entire_plane, b_entire_plane)), np.empty((0, 3)))
    print("---- Entire plane x_1, x_2 ---- (H-repr)")
    print(f"Vertices:\n{V_entire_plane + 0}\t\t(k = {V_entire_plane.shape[0]})")
    print(f"Rays:\n\033[33m{rays_entire_plane + 0}\t\t(k_rays = {rays_entire_plane.shape[0]}) | We could also have a representation with 3 rays, but this is still minimal (and maybe cleaner)\n\033[0m")

    Ab_entire_plane_reconstructed, Ab_eq_entire_plane_reconstructed = enum_facets(V_entire_plane, rays_entire_plane)
    print(f"Reconstructed facets:\n{Ab_entire_plane_reconstructed + 0}\t\t\t(m = {Ab_entire_plane_reconstructed.shape[0]})\n")
    print(f"Reconstructed facets (eq):\n{Ab_eq_entire_plane_reconstructed + 0}\t\t\t(m_eq = {Ab_eq_entire_plane_reconstructed.shape[0]})\n")

    # ---- Entire plane x_1, x_2 (trivial constraints) ----
    A_entire_plane_trivial, b_entire_plane_trivial = np.array([[0, 0], [0, 0]]), np.array([1, 1])  # Trivial constraints (entire plane) in 2D space

    V_entire_plane_trivial, rays_entire_plane_trivial = enum_verts(np.column_stack((A_entire_plane_trivial, b_entire_plane_trivial)), np.empty((0, 3)))
    print("---- Entire plane x_1, x_2 (trivial constraints) ---- (H-repr)")
    print(f"Vertices:\n{V_entire_plane_trivial + 0}\t\t(k = {V_entire_plane_trivial.shape[0]})")
    print(f"Rays:\n\033[33m{rays_entire_plane_trivial + 0}\t\t(k_rays = {rays_entire_plane_trivial.shape[0]}) | We could also have a representation with 3 rays, but this is still minimal (and maybe cleaner)\n\033[0m")

    A_entire_plane_trivial_reconstructed, b_entire_plane_trivial_reconstructed = enum_facets(V_entire_plane_trivial, rays_entire_plane_trivial)
    print(f"Reconstructed facets:\n{A_entire_plane_trivial_reconstructed + 0}\t\t\t(m = {A_entire_plane_trivial_reconstructed.shape[0]})\n")
    print(f"Reconstructed facets (eq):\n{b_entire_plane_trivial_reconstructed + 0}\t\t\t(m_eq = {b_entire_plane_trivial_reconstructed.shape[0]})\n")

    # ---- Entire plane x_1, x_2 (three rays) ----
    verts_entire_plane_three_rays = np.zeros((1, 2))  # Origin as vertex to ensure non-empty vertex set
    rays_entire_plane_three_rays = np.array([[1, 0], [0, 1], [-1, -1]])  # Three rays representing the entire plane in 2D space

    # NOTE: Passing `verts=np.empty((0, 2))` raises "IndexError: tuple index out of range"
    Ab_entire_plane_three_rays, Ab_eq_entire_plane_three_rays = enum_facets(verts_entire_plane_three_rays, rays_entire_plane_three_rays)
    print("---- Entire plane x_1, x_2 (three rays) ---- (V-repr)")
    print(f"Facets:\n{Ab_entire_plane_three_rays + 0}\t\t\t(m = {Ab_entire_plane_three_rays.shape[0]})")
    print(f"Facets (eq):\n{Ab_eq_entire_plane_three_rays + 0}\t\t\t(m_eq = {Ab_eq_entire_plane_three_rays.shape[0]})\n")

    V_entire_plane_three_rays_reconstructed, rays_entire_plane_three_rays_reconstructed = enum_verts(Ab_entire_plane_three_rays, Ab_eq_entire_plane_three_rays)
    print(f"Reconstructed vertices:\n{V_entire_plane_three_rays_reconstructed + 0}\t\t(k = {V_entire_plane_three_rays_reconstructed.shape[0]})")
    print(f"Reconstructed rays:\n{rays_entire_plane_three_rays_reconstructed + 0}\t\t(k_rays = {rays_entire_plane_three_rays_reconstructed.shape[0]})\n")

    # ---- Line x_1 - 1 = x_2 ----
    A_unbounded_line_offset, b_unbounded_line_offset = np.array([[1, -1], [-1, 1]]), np.array([1, -1])  # Unbounded polyhedron (line x_1 - 1 = x_2) in 2D space

    V_unbounded_line_offset, rays_unbounded_line_offset = enum_verts(np.column_stack((A_unbounded_line_offset, b_unbounded_line_offset)), np.empty((0, 3)))
    print("---- Line x_1 - 1 = x_2 ---- (H-repr)")
    print(f"Vertices:\n{V_unbounded_line_offset + 0}\t\t(k = {V_unbounded_line_offset.shape[0]})")
    print(f"Rays:\n{rays_unbounded_line_offset + 0}\t\t(k_rays = {rays_unbounded_line_offset.shape[0]})\n")

    Ab_unbounded_line_offset_reconstructed, Ab_eq_unbounded_line_offset_reconstructed = enum_facets(V_unbounded_line_offset, rays_unbounded_line_offset)
    print(f"Reconstructed facets:\n{Ab_unbounded_line_offset_reconstructed + 0}\t\t(m = {Ab_unbounded_line_offset_reconstructed.shape[0]})\n")
    print(f"Reconstructed facets (eq):\n{Ab_eq_unbounded_line_offset_reconstructed + 0}\t\t(m_eq = {Ab_eq_unbounded_line_offset_reconstructed.shape[0]})\n")

    # ---- Flat unbounded region 3D ---- (H-repr)
    A_unbounded_flat_3d, b_unbounded_flat_3d = np.array([[-1, 0, 0], [0, -1, 0]]), np.array([0, 0])
    A_eq_unbounded_flat_3d, b_eq_unbounded_flat_3d = np.array([[0, 0, 1]]), np.array([0])

    V_unbounded_flat_3d, rays_unbounded_flat_3d = enum_verts(np.column_stack((A_unbounded_flat_3d, b_unbounded_flat_3d)), np.column_stack((A_eq_unbounded_flat_3d, b_eq_unbounded_flat_3d)))
    print("---- Flat unbounded region 3D ---- (H-repr)")
    print(f"Vertices:\n{V_unbounded_flat_3d + 0}\t\t(k = {V_unbounded_flat_3d.shape[0]})")
    print(f"Rays:\n{rays_unbounded_flat_3d + 0}\t\t(k_rays = {rays_unbounded_flat_3d.shape[0]})\n")

    Ab_unbounded_flat_3d_reconstructed, Ab_eq_unbounded_flat_3d_reconstructed = enum_facets(V_unbounded_flat_3d, rays_unbounded_flat_3d)
    print(f"Reconstructed facets:\n{Ab_unbounded_flat_3d_reconstructed + 0}\t(m = {Ab_unbounded_flat_3d_reconstructed.shape[0]})\n")
    print(f"Reconstructed facets (eq):\n{Ab_eq_unbounded_flat_3d_reconstructed + 0}\t(m_eq = {Ab_eq_unbounded_flat_3d_reconstructed.shape[0]})\n")

    # ---- Flat bounded triangle 3D ---- (V-repr)
    verts_flat_triangle_3d = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    Ab_flat_triangle_3d, Ab_eq_flat_triangle_3d = enum_facets(verts_flat_triangle_3d)
    print("---- Flat bounded triangle 3D ---- (V-repr)")
    print(f"Facets:\n{Ab_flat_triangle_3d + 0}\t(m = {Ab_flat_triangle_3d.shape[0]})")
    print(f"Facets (eq):\n{Ab_eq_flat_triangle_3d + 0}\t(m_eq = {Ab_eq_flat_triangle_3d.shape[0]})\n")

    verts_flat_triangle_3d_reconstructed, rays_flat_triangle_3d_reconstructed = enum_verts(Ab_flat_triangle_3d, Ab_eq_flat_triangle_3d)
    print(f"Reconstructed vertices:\n{verts_flat_triangle_3d_reconstructed + 0}\t\t(k = {verts_flat_triangle_3d_reconstructed.shape[0]})")
    print(f"Reconstructed rays:\n{rays_flat_triangle_3d_reconstructed + 0}\t\t\t(k_rays = {rays_flat_triangle_3d_reconstructed.shape[0]})\n")

if __name__ == "__main__":
    main()