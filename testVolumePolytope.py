"""Script to test calculating volume of a polytope"""

from typing import Any, Callable, Literal, Never

import numpy as np
from numpy.typing import NDArray
import scipy as sp

from testRunPyCDDLIBDegenerate import enum_facets, enum_verts


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
    def vol(self) -> float:
        if self._vol is None:
            self._vol = self.volume()
        return self._vol
    
    def volume(self) -> float:
        """Calculate the volume of the polytope using scipy's ConvexHull"""
        hull = sp.spatial.ConvexHull(self.verts)
        return hull.volume


def main():
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    verts_dodecahedron = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                                   [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
                                   [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
                                   [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]])
    verts_dodecahedron += (2 * np.ones(verts_dodecahedron.shape))

    poly_dodecahedron = Polytope(verts_dodecahedron)
    print(f"Volume of the dodecahedron: {poly_dodecahedron.vol:.2f}")

    verts_cube = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    poly_cube = Polytope(verts_cube)
    print(f"Volume of the cube: {poly_cube.vol:.2f}")

    verts_house = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 2]])

    poly_house = Polytope(verts_house)
    print(f"Volume of the house-shaped polytope: {poly_house.vol:.2f}")

    verts_triangle_3d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    poly_triangle_3d = Polytope(verts_triangle_3d)
    print(f"Volume of the 3D triangle (tetrahedron): {poly_triangle_3d.vol:.4f}")

    verts_hypercube_4d = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
                             [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
    verts_hypercube_4d *= 2  # Scale the hypercube to have volume 16 instead of 1
    
    poly_hypercube_4d = Polytope(verts_hypercube_4d)
    print(f"Volume of the 4D hypercube: {poly_hypercube_4d.vol:.2f}")


if __name__ == "__main__":
    main()