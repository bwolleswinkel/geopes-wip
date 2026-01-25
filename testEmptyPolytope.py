"""Simple script to test whether a polytope is empty or not"""

import numpy as np
from numpy.typing import NDArray
import scipy as sp


class Polytope:
    """Simple Polytope class for testing purposes"""

    def __init__(self, A: NDArray, b: NDArray) -> None:
        self._Ab: NDArray | None = np.column_stack((A, b))
        self.m: int = b.size
        self.n: int = A.shape[1]
    
    @property
    def A(self) -> NDArray:
        if self._Ab is None:
            raise ValueError("Polytope is not in H-representation.")
        return self._Ab[:, :-1]
    
    @property
    def b(self) -> NDArray:
        if self._Ab is None:
            raise ValueError("Polytope is not in H-representation.")
        return self._Ab[:, -1]
    
    @property
    def is_empty(self) -> bool:
        """Check whether the polytope is empty using linear programming"""
        # TODO: I should probably wrap this in a method `x_star, f_star, status = solve_lp(c, A, b)` somewhere
        res = sp.optimize.linprog(c=np.zeros(self.n), A_ub=self.A, b_ub=self.b, bounds=(None, None))
        if res.success:
            return False
        elif res.status == 2:
            return True
        else:
            raise RuntimeError(f"Linear program to check polytope emptiness failed unexpectedly (status code {res.status})")
    

if __name__ == "__main__":
    A, b = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]), np.array([1, 1, -2, -2])  # This defines an empty polytope

    poly_empty = Polytope(A, b)
    print(f"Empty polytope .is_empty: {poly_empty.is_empty}")  # Expected: True

    A, b = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]), np.array([1, 1, 0, 0])  # This defines a non-empty polytope
    poly_non_empty = Polytope(A, b)
    print(f"Non-empty polytope .is_empty: {poly_non_empty.is_empty}")  # Expected: False