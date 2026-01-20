"""This is a script to test implementation of initialization with and without automatic 'minimal' computation"""

from __future__ import annotations
import time
import numpy as np
from numpy.typing import NDArray
import scipy as sp

global REDUCE
REDUCE: bool = True


class Polytope:
    def __init__(self, verts: NDArray | None = None, n: int | None = None) -> None:
        if verts is not None and n is None:
            self._verts = verts if not REDUCE else conv(verts)
        else:
            if verts is None and n is not None:
                self._verts = np.empty((n, 0))
            else:
                raise ValueError("Either 'verts' or 'n' must be provided")
        self.n = self.verts.shape[1] if verts is not None else n
        self.A: NDArray | None = None
        self.b: NDArray | None = None

    @property
    def A(self) -> NDArray:
        if self._A is None:
            time.sleep(3)  # Simulate expensive computation
            self._A = ...
        return self._A
    
    @A.setter
    def A(self, value: NDArray | None) -> None:
        self._A = value
    
    @property
    def b(self) -> NDArray:
        if self._b is None:
            time.sleep(3)  # Simulate expensive computation
            self._b = ...
        return self._b
    
    @b.setter
    def b(self, value: NDArray | None) -> None:
        self._b = value

    @property
    def verts(self) -> NDArray:
        return self._verts
    
    def reduce(self, in_place: bool = False) -> None | Polytope:
        if in_place:
            self._verts = conv(self.verts)
        else:
            return Polytope(conv(self.verts))
        

def conv(verts: NDArray) -> NDArray:
    hull = sp.spatial.ConvexHull(verts)
    return verts[hull.vertices]


from contextlib import contextmanager


@contextmanager
def disable_reduce():
    global REDUCE
    old_value = REDUCE
    REDUCE = False
    try:
        yield
    finally:
        REDUCE = old_value


if __name__ == "__main__":
    verts = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
    
    # Default behavior with reduction
    poly_red = Polytope(verts)

    # Workaround behavior 
    poly_workaround = Polytope(n=2)
    poly_workaround._verts = verts  # Manually set vertices without reduction

    # With a context manager to disable reduction
    with disable_reduce():
        poly_no_red = Polytope(verts)

    # Default behavior with reduction
    poly_red_end = Polytope(verts)

    # Print results
    print("With reduction (default):")
    print("Vertices:\n", poly_red.verts)

    print("\nWithout reduction (workaround):")
    print("Vertices:\n", poly_workaround.verts)

    print("\nWithout reduction (context manager):")
    print("Vertices:\n", poly_no_red.verts)

    print("\nWith reduction (back to default):")
    print("Vertices:\n", poly_red_end.verts)

    # NOTE: By Sam's request, we can also 'compute' both representations 'at initialization' time
    poly_both = Polytope(verts)
    print("Triggering both representations computation...")
    poly_both.A, poly_both.b  # Trigger computation of H-representation

    print("\nWith both representations computed (fast printing):")
    print("Vertices:\n", poly_both.verts)
    print(f"A: {poly_both.A}, b: {poly_both.b}")