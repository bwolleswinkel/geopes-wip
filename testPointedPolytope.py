"""Script to test detecting whether a polytope is pointed"""

from typing import Never

import numpy as np
from numpy.typing import NDArray
import scipy as sp

from testRunPyCDDLIBDegenerate import enum_facets, enum_verts


class Polytope:
    """Simple Polytope class for testing purposes"""

    def __init__(self, *args: tuple[()] | NDArray | tuple[NDArray, NDArray], **kwargs) -> None:
        if len(args) == 0:
            if 'n' in kwargs:
                self.n = kwargs['n']
                self._verts, self._Ab = None, None
                self._rays = None
                self.is_vrepr, self.is_hrepr = False, False
            elif 'verts' in kwargs and 'A' in kwargs and 'b' in kwargs:
                self._verts, self._Ab = kwargs['verts'], np.column_stack((kwargs['A'], kwargs['b']))
                self._rays = None
                self.is_vrepr, self.is_hrepr = True, True
                self.n = self._verts.shape[1]
            else:
                raise ValueError("Polytope must be initialized with either vertices, (A, b) representation, or dimension n.")
        elif len(args) == 1:
            self._verts, self._Ab = args[0], None
            self._rays = None
            self.is_vrepr, self.is_hrepr = True, False
            self.n = self._verts.shape[1]
        elif len(args) == 2:
            self._verts, self._Ab = None, np.column_stack(args)
            self._rays = None
            self.is_vrepr, self.is_hrepr = False, True
            self.n = self._Ab.shape[1] - 1
        else:
            raise ValueError("Polytope must be initialized with either vertices or (A, b) representation.")
        
    @property
    def verts(self) -> NDArray:
        if not self.is_vrepr:
            self._verts, _ = enum_verts(np.column_stack((self.A, self.b)), np.empty((0, self.n)))
            self.is_vrepr = True
        return self._verts
    
    @property
    def rays(self) -> NDArray:
        if not self.is_vrepr or self._rays is None:
            self._verts, self._rays = enum_verts(np.column_stack((self.A, self.b)), np.empty((0, self.n)))
            self.is_vrepr = True
        return self._rays
    
    @property
    def A(self) -> NDArray:
        if not self.is_hrepr:
            self._Ab, self.Ab_eq = enum_facets(self.verts)
            self.is_hrepr = True
        return self._Ab[:, :-1]
    
    @property
    def b(self) -> NDArray:
        if not self.is_hrepr:
            self._Ab, self.Ab_eq = enum_facets(self.verts)
            self.is_hrepr = True
        return self._Ab[:, -1]
    
    @property
    def m(self) -> int:
        return self.A.shape[0]
    
    @property
    def k(self) -> int:
        return self.verts.shape[1]
    
    @property
    def m_eq(self) -> int:
        return self.Ab_eq.shape[0]

    @property
    def k_rays(self) -> int:
        return self.rays.shape[0]
    
    @property
    def vol(self) -> float:
        if self._vol is None:
            self._vol = self.volume()
        return self._vol
    
    @property
    def is_pointed(self) -> bool:
        if self.is_hrepr:
            return np.linalg.matrix_rank(self.A) == self.n
        elif self.is_vrepr:
            if self.rays.shape[0] == 0:
                return True
            else:
                c, A_eq, b_eq = np.ones(self.k_rays), self.rays.T, np.zeros(self.n)
                res = sp.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                if res.status == 0:
                    return False
                elif res.status == 2:
                    return True
                else:
                    raise ValueError(f"Unexpected optimization result: {res.message}")
        else:
            raise ValueError("Polytope must be initialized with either vertices or (A, b) representation.")


def main():
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    verts_dodecahedron = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                                   [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
                                   [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
                                   [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]])
    verts_dodecahedron += (2 * np.ones(verts_dodecahedron.shape))

    poly_dodecahedron = Polytope(verts_dodecahedron)
    print(f"Dodecahedron is pointed (V-repr): {poly_dodecahedron.is_pointed}")
    _ = poly_dodecahedron.A, poly_dodecahedron.b  # Force H-repr computation
    print(f"Dodecahedron is pointed (H-repr): {poly_dodecahedron.is_pointed}")

    A_quadrant_unbounded, b_quadrant_unbounded = np.array([[-1, 0], [0, -1]]), np.array([-1, -1])  # Unbounded quadrant in 2D space (x_1 >= 1, x_2 >= 1)
    poly_quadrant_unbounded = Polytope(A_quadrant_unbounded, b_quadrant_unbounded)
    print(f"Unbounded quadrant is pointed (H-repr): {poly_quadrant_unbounded.is_pointed}")
    _ = poly_quadrant_unbounded.verts  # Force V-repr computation
    print(f"Unbounded quadrant is pointed (V-repr): {poly_quadrant_unbounded.is_pointed}")

    A_eq_line, b_eq_line = np.array([[1, 0]]), np.array([1])  # Line x_1 = 1 in 2D space (unbounded in both directions)

    poly_line = Polytope(n=2)
    poly_line._Ab = np.column_stack((A_eq_line, b_eq_line))
    poly_line.is_hrepr = True
    print(f"Line is pointed (H-repr): {poly_line.is_pointed}")
    _ = poly_line.verts  # Force V-repr computation
    print(f"Line is pointed (V-repr): {poly_line.is_pointed}")


if __name__ == "__main__":
    main()