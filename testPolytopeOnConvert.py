"""Script to test raising an warning or error when converting a Polytope from one representation to another"""

from typing import Never, Literal

import numpy as np
from numpy.typing import NDArray
import scipy as sp

from testPolytopeOnConvertConfig import CFG, set_on_poly_convert, on_poly_convert
from testRunPyCDDLIBDegenerate import enum_verts, enum_facets


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
        if not self.is_vrepr and CFG.on_poly_convert(poly=self):
            self._verts, _ = enum_verts(np.column_stack((self.A, self.b)))
            self.is_vrepr = True
        return self._verts
    
    @verts.setter
    def verts(self, value: NDArray) -> None:
        if CFG.on_poly_assign == 'reduce':
            value = conv(value)
        self._verts = value
        self.is_vrepr = True
        self._Ab = None
        self.is_hrepr = False
    
    @property
    def A(self) -> NDArray:
        if not self.is_hrepr and CFG.on_poly_convert(poly=self):
            self._Ab = enum_facets(self.verts)
            self.is_hrepr = True
        return self._Ab[:, :-1]
    
    @property
    def b(self) -> NDArray:
        if not self.is_hrepr and CFG.on_poly_convert(poly=self):
            self._Ab = enum_facets(self.verts)
            self.is_hrepr = True
        return self._Ab[:, -1]
    
    @property
    def m(self) -> int:
        return self.A.shape[0]
    
    @property
    def k(self) -> int:
        return self.verts.shape[1]
    
    __array_ufunc__ = None

    def __matmul__(self, other: NDArray) -> NDArray:
        _ = self.verts
        ...

    def __rmatmul__(self, other: NDArray) -> NDArray:
        return self.__matmul__(other)
    
    def __repr__(self) -> str:
        repr_str = f"Polytope(n={self.n}, is_vrepr={self.is_vrepr}, is_hrepr={self.is_hrepr}, _verts_shape={self._verts.shape if self._verts is not None else None}, _Ab_shape={self._Ab.shape if self._Ab is not None else None})"
        return repr_str


def conv(verts: NDArray) -> NDArray:
    """Compute the convex hull of a set of points given by `verts`"""
    # FROM: GitHub Copilot GPT-4.1 | 2026/01/28 [untested/unverified]
    centered = verts - np.mean(verts, axis=1, keepdims=True)
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    rank = np.sum(S > 1E-12)
    if rank == 0:
        return verts[:, [0]]
    coords = U[:, :rank].T @ centered 
    if rank == 1:
        idx_min, idx_max = np.argmin(coords[0]), np.argmax(coords[0])
        return verts[:, np.unique([idx_min, idx_max])]
    hull = sp.spatial.ConvexHull(coords.T)
    return verts[:, hull.vertices]
    

def main():

    def inner_scope(poly: Polytope):
        _ = np.eye(3) @ poly

    def second_inner_scope(poly: Polytope):
        with on_poly_convert('warning'):
            poly.verts = np.ones((3, 5))

    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    verts_dodecahedron = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                                   [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
                                   [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
                                   [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]]).T
    verts_dodecahedron += (2 * np.ones(verts_dodecahedron.shape))

    poly_dodecahedron_verts = Polytope(verts_dodecahedron)
    print(f"Dodecahedron initialized with vertices: {poly_dodecahedron_verts.verts.shape[1]} vertices.")
    try:
        print("Attempting to access H-representation (should raise error)...")
        A = poly_dodecahedron_verts.A
    except RuntimeError as e:
        print(f"Caught expected error: {e}")

    A_cube, b_cube = np.array([[1, 0, 0], 
                               [-1, 0, 0], 
                               [0, 1, 0], 
                               [0, -1, 0], 
                               [0, 0, 1], 
                               [0, 0, -1]]), np.array([1, 1, 1, 1, 1, 1])
    
    set_on_poly_convert('error')
    
    poly_cube_Ab = Polytope(A_cube, b_cube)
    print(f"Cube initialized with H-representation: {poly_cube_Ab.A.shape[0]} facets.")
    try:
        print("Attempting to access V-representation (should raise warning), from inner scope...")
        inner_scope(poly_cube_Ab)
    except RuntimeError as _:
        print(f"Caught expected error: {_}")

    with on_poly_convert('warning'):
        inner_scope(poly_cube_Ab)
        print(f"No errors raised within 'warning' context manager.")

    try:
        second_inner_scope(poly_cube_Ab)
    except RuntimeError as _:
        print(f"Caught expected error: {_}")


if __name__ == "__main__":
    main()