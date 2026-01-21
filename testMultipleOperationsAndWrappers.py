"""Script to test how to properly dispatch multiple operations and wrappers"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from testDocstringWrappers import tightwrap_wraps as wraps


class Polytope:
    def __init__(self, verts: NDArray) -> None:
        self._verts = verts

    @property
    def verts(self) -> NDArray:
        return self._verts
    
    def __add__(self, other: Polytope) -> Polytope:
        return self.mink_sum(other)
    
    def mink_sum(self, other: Polytope) -> Polytope:
        return Polytope(self.verts + other.verts)
    

class Subspace:
    def __init__(self, basis: NDArray) -> None:
        self._basis = basis

    @property
    def basis(self) -> NDArray:
        return self._basis
    
    # NOTE: The order of the methods matters here; __add__ must come after mink_sum to properly wrap it
    def mink_sum(self, other: Subspace, in_place: bool = True) -> Subspace:
        """Compute the Minkowski sum of two subspaces"""
        return Subspace(self.basis + other.basis)
    
    @wraps(mink_sum)  # FIXME: This is actually referencing the external function, not the method...
    def __add__(self, other: Subspace) -> Subspace:
        """Should dispatch to mink_sum"""
        return self.mink_sum(other, in_place=False)
    
    @wraps(__add__)
    def __radd__(self, other: Subspace) -> Subspace:
        """Should dispatch to mink_sum"""
        return other + self
    
    @wraps(__add__)
    def __iadd__(self, other: Subspace) -> Subspace:
        """Should dispatch to mink_sum"""
        return self + other
    

def mink_sum_dp(obj_1: Polytope | Subspace, obj_2: Polytope | Subspace) -> Polytope | Subspace:
    """This method should have it's own description, with dispatching"""
    if isinstance(obj_1, Polytope) and isinstance(obj_2, Polytope):
        return obj_1.mink_sum(obj_2)
    elif isinstance(obj_1, Subspace) and isinstance(obj_2, Subspace):
        return obj_1.mink_sum(obj_2)
    else:
        raise TypeError("Minkowski sum is only defined between objects of the same type.")
    

def main() -> None:
    verts = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]).T
    
    poly_1 = Polytope(verts)
    poly_2 = Polytope(verts * 2)

    poly_res_1 = mink_sum_dp(poly_1, poly_2)
    poly_res_2 = poly_1 + poly_2
    poly_res_3 = poly_1.mink_sum(poly_2)

    basis = np.array([[1, 0], [0, 1]]).T

    sub_1 = Subspace(basis)
    sub_2 = Subspace(basis * 2)

    sub_res_1 = mink_sum_dp(sub_1, sub_2)
    sub_res_2 = sub_1 + sub_2  # FIXME: Why does the signature not get picked up here? Incorrectly displays "Should dispatch to mink_sum"
    sub_res_3 = sub_1.mink_sum(sub_2)

    print("Vertices of Polytope:")
    print(poly_1.verts)

if __name__ == "__main__":
    main()