"""Script to test membership of a point in a polytope defined by its vertices"""

import numpy as np
from numpy.typing import NDArray
import scipy as sp



class Polytope:
    """Simple Polytope class for testing purposes"""

    def __init__(self, verts: NDArray) -> None:
        self._verts = verts
        self.n: int = verts.shape[0]
        self.k: int = verts.shape[1]
    
    @property
    def verts(self) -> NDArray:
        return self._verts
    
    def __contains__(self, point: NDArray) -> bool:
        """Check if a point is inside the polytope using linear programming"""
        # FROM: "Frequently Asked Questions in Polyhedral Computation," Komei Fukuda (2024) | §2.19, Eq. (5)
        c = np.concatenate(([-1], point))
        A_ub = np.column_stack((-np.ones(self.k + 1), np.column_stack((self.verts, point)).T))
        b_ub = np.concatenate((np.zeros(self.k), [1]))
        res = sp.optimize.linprog(-c, A_ub=A_ub, b_ub=b_ub)
        if res.fun <= 0 and not np.isclose(res.fun, 0):
            return False
        else: 
            return True
        

def main() -> None:
    verts = np.array([[0, 0], [1, 0], [0, 1]]).T
    poly = Polytope(verts)
    
    point_inside = np.array([0.1, 0.1])
    point_outside = np.array([1.0, 1.0])
    point_on_edge = np.array([0.5, 0.0])
    point_close_outside = np.array([1.0, 1E-6])

    print(f"Internal point {point_inside} is inside the polytope: {point_inside in poly}")
    print(f"External point {point_outside} is inside the polytope: {point_outside in poly}")
    print(f"Point on edge {point_on_edge} is inside the polytope: {point_on_edge in poly}")
    print(f"Point close outside {point_close_outside} is inside the polytope: {point_close_outside in poly}")

if __name__ == "__main__":
    main()


