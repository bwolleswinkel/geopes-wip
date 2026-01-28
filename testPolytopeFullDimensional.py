"""Script to test whether a polytope is full-dimensional, or whether it is degenerate (vol = 0)"""

from typing import Never

import numpy as np
from numpy.typing import NDArray
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from testRunPyCDDLIBDegenerate import enum_facets, enum_verts
from testMatplotlibColorCycle import set_display_options


class Polytope:
    """Simple Polytope class for testing purposes"""

    def __init__(self, *args: list[Never] | NDArray | tuple[NDArray, NDArray], **kwargs) -> None:
        if len(args) == 0:
            self._verts, self._Ab = kwargs['verts'], np.column_stack((kwargs['A'], kwargs['b']))
            self.is_vrepr, self.is_hrepr = True, True
            self.n = self._verts.shape[0]
            self._chebcr, self._is_full_dim, self._vol = None, None, None
        if len(args) == 1:
            self._verts, self._Ab = args[0], None
            self.is_vrepr, self.is_hrepr = True, False
            self.n = self._verts.shape[0]
            self._chebcr, self._is_full_dim, self._vol = None, None, None
        elif len(args) == 2:
            self._verts, self._Ab = None, np.column_stack(args)
            self.is_vrepr, self.is_hrepr = False, True
            self.n = self._Ab.shape[1] - 1
            self._chebcr, self._is_full_dim, self._vol = None, None, None
        else:
            raise ValueError("Polytope must be initialized with either vertices or (A, b) representation.")
        
    @property
    def verts(self) -> NDArray:
        if not self.is_vrepr:
            self._verts, _ = enum_verts(np.column_stack((self.A, self.b)))
            self.is_vrepr = True
        return self._verts
    
    @property
    def A(self) -> NDArray:
        if not self.is_hrepr:
            self._Ab = enum_facets(self.verts)
            self.is_hrepr = True
        return self._Ab[:, :-1]
    
    @property
    def b(self) -> NDArray:
        if not self.is_hrepr:
            self._Ab = enum_facets(self.verts)
            self.is_hrepr = True
        return self._Ab[:, -1]
    
    @property
    def m(self) -> int:
        return self.A.shape[0]
    
    @property
    def k(self) -> int:
        return self.verts.shape[1]
    
    @property
    def chebc(self) -> NDArray:
        """Compute the Chebyshev center of the polytope"""
        if self._chebcr is None:
            self._chebcr: NDArray = self._comp_chebcr()
        return self._chebcr[:-1]
    
    @property
    def chebr(self) -> float:
        """Compute the Chebyshev radius of the polytope"""
        if self._chebcr is None:
            self._chebcr: NDArray = self._comp_chebcr()
        return self._chebcr[-1]
        
    @property
    def is_full_dim(self) -> bool:
        """Check if the polytope is full-dimensional, or type-0 degenerate (i.e., 'flat')"""
        if self._is_full_dim is None:
            if self.is_vrepr:
                # FROM: GitHub Copilot GPT-4.1 | 2026/01/28 [untested/unverified]
                self._is_full_dim = np.linalg.matrix_rank(self.verts - np.mean(self.verts, axis=1, keepdims=True)) == self.n
            elif self.is_hrepr:
                self._is_full_dim = not np.isclose(self.chebr, 0)
            else:
                raise ValueError("Polytope must have either H-representation or V-representation to determine full-dimensionality")
            if not self._is_full_dim:
                if self._chebcr is not None:
                    self._chebcr[-1] = 0
                self._vol = 0
        return self._is_full_dim

    def _comp_chebcr(self) -> NDArray:
        """Compute the Chebyshev center and radius of the polytope"""
        A_ext = np.hstack((self.A, np.linalg.norm(self.A, axis=1, keepdims=True)))
        c = np.zeros(self.n + 1)
        c[-1] = -1.0
        res = sp.optimize.linprog(c, A_ub=A_ext, b_ub=self.b, bounds=([(None, None)] * self.n + [(0, None)]), method='highs')
        if not res.success:
            raise ValueError("Linear program to compute Chebyshev center/radius failed")
        return res.x


def main() -> None:
    verts_cube = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).T
    
    poly_cube_verts = Polytope(verts_cube)
    print(f"Cube Chebyshev radius (should be > 0): {poly_cube_verts.chebr} (center = {poly_cube_verts.chebc})")

    A_cube, b_cube = np.array([[ 1,  0,  0],
                               [ 0,  1,  0],
                               [ 0,  0,  1],
                               [-1,  0,  0],
                               [ 0, -1,  0],
                               [ 0,  0, -1]]), np.array([1, 1, 1, 0, 0, 0])
    
    poly_cube_facets = Polytope(A_cube, b_cube)
    print(f"Cube (from h-repr) Chebyshev radius (should be > 0): {poly_cube_facets.chebr} (center = {poly_cube_facets.chebc})")

    A_plane, b_plane = np.array([[0, 0, -1],
                                 [0, 0, 1],
                                 [1, 0, 0],
                                 [-1, 0, 1],
                                 [0, 1, 0],
                                 [0, -1, 1]]), np.array([0, 0, 1, 0, 1, 0])

    poly_plane_facets = Polytope(A_plane, b_plane)
    print(f"Plane Chebyshev radius (should be 0): {poly_plane_facets.chebr} (center = {poly_plane_facets.chebc})")

    verts_plane = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]).T
    poly_plane_verts = Polytope(verts_plane)

    verts_line = np.array([[0, 0, 0], [1, 0, 0]]).T
    poly_line_verts = Polytope(verts_line)

    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    verts_dodecahedron = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                                   [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
                                   [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
                                   [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]]).T
    poly_dodecahedron_verts = Polytope(verts_dodecahedron)

    print("Is cube full-dimensional? ", poly_cube_verts.is_full_dim)
    print("Is cube (from facets) full-dimensional? ", poly_cube_facets.is_full_dim)
    _ = poly_cube_facets.verts
    print("Is cube (converted to v-repr) full-dimensional? ", poly_cube_facets.is_full_dim)
    print("Is plane full-dimensional? ", poly_plane_facets.is_full_dim)
    print("Is plane (from verts) full-dimensional? ", poly_plane_verts.is_full_dim)
    print("Is line full-dimensional? ", poly_line_verts.is_full_dim)
    # FIXME: The dodecahedron is definitely full-dimensional, but the rank calculation seems to fail here for some reason?
    print("Is dodecahedron full-dimensional? ", poly_dodecahedron_verts.is_full_dim)



if __name__ == "__main__":
    main()