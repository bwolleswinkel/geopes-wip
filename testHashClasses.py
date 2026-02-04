"""Test if we can hash the Polytope and Subspace class correctly, using a 'canonical' representation"""

from typing import Literal
from dataclasses import dataclass
from functools import lru_cache
import time

import numpy as np
from numpy.typing import NDArray
import scipy as sp

from testRunPyCDDLIBDegenerate import enum_facets, enum_verts

# ====== PLACEHOLDER ======

@dataclass
class GlobalConfig:
    rtol: float = 1E-5
    atol: float = 1E-8

global CFG
CFG = GlobalConfig()

# ====== PLACEHOLDER ======


class Polytope:
    """Simple Polytope class for testing purposes"""

    def __init__(self, *args: NDArray | tuple[NDArray, NDArray]) -> None:
        if len(args) == 1:
            self._verts = args[0]
            self.is_vrepr, self.is_hrepr = True, False
            self._Ab = None
        elif len(args) == 2:
            self._Ab = np.column_stack(args)
            self.is_vrepr, self.is_hrepr = False, True
            self._verts = None
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
    
    def __eq__(self, other: Polytope) -> bool:
        if not isinstance(other, Polytope):
            return NotImplemented("Other object is not a Polytope")
        return np.isclose(np.sort(self.verts, axis=0), np.sort(other.verts, axis=0), rtol=CFG.rtol, atol=CFG.atol).all()
    
    def __hash__(self) -> int:
        # QUESTION: What about the two canonical representations (V-repr and H-repr)? Is there a way to store EITHER one, with two hashes, and check if ONE matches? Because for large polytopes, converting between the two is expensive.
        # Use a placeholder for rays
        rays: Literal[None] = None
        # Use a canonical representation for hashing: sorted vertices (what about rays?)
        return hash((self._sort_and_round(self.verts).tobytes(), rays))
    
    def _sort_and_round(self, matrix: NDArray) -> NDArray:
        """Helper function to sort and round a matrix for comparison"""
        matrix_rounded = np.round(matrix / CFG.atol) * CFG.atol
        return np.sort(matrix_rounded, axis=1)
    

class Ellipsoid:
    """Simple Ellipsoid class for testing purposes"""

    def __init__(self, c: NDArray, Q: NDArray) -> None:
        self.c = c
        self.Q = Q
        self.n = c.size
        self.is_degen = not is_pos_def(Q)

    def __eq__(self, other: Ellipsoid) -> bool:
        if not isinstance(other, Ellipsoid):
            return NotImplemented("Other object is not an Ellipsoid")
        return np.array_equal(self._sort_and_round(self.c), self._sort_and_round(other.c)) and np.array_equal(self._sort_and_round(self.Q), self._sort_and_round(other.Q))
    
    def __hash__(self) -> int:
        if self.is_degen:
            # FIXME: Instead of raising a NotImplementedError, we could also generate a random hash for degenerate ellipsoids, but that would mean that two equal degenerate ellipsoids would not have the same hash; this might be acceptable if we only use the hash for caching purposes.
            '''
            return hash((np.random.randint(0, 2 ** 32),))
            '''
            raise NotImplementedError("Currently, hashing degenerate ellipsoids is not supported")
        return hash((self._sort_and_round(self.c).tobytes(), self._sort_and_round(self.Q).tobytes()))
    
    def _sort_and_round(self, matrix: NDArray) -> NDArray:
        """Helper function to sort and round a matrix for comparison"""
        matrix_rounded = np.round(matrix / CFG.atol) * CFG.atol
        return np.sort(matrix_rounded, axis=1 if matrix.ndim > 1 else 0)


class Subspace:
    """Simple Subspace class for testing purposes"""

    def __init__(self, basis: NDArray) -> None:
        self.basis = basis

    def __contains__(self, vector: NDArray) -> bool:
        """Check if the vector is in the subspace"""
        _, residuals, *_ = np.linalg.lstsq(self.basis, vector, rcond=None)
        return np.isclose(residuals, 0).all()

    @property
    def perp(self) -> NDArray:
        """Compute the orthogonal complement of the subspace"""
        return Subspace(sp.linalg.null_space(self.basis.T))
    
    @property
    def proj_mat(self) -> NDArray:
        """Compute the projection matrix onto the subspace"""
        B = self.basis
        return B @ np.linalg.inv(B.T @ B) @ B.T
    
    def __eq__(self, other: Subspace) -> bool:
        if not isinstance(other, Subspace):
            return NotImplemented("Other object is not a Subspace")
        return np.array_equal(self._sort_and_round(self.proj_mat), self._sort_and_round(other.proj_mat))
    
    def __hash__(self) -> int:
        return hash(self._sort_and_round(self.proj_mat).tobytes())
    
    def _sort_and_round(self, matrix: NDArray) -> NDArray:
        """Helper function to sort and round a matrix for comparison"""
        matrix_rounded = np.round(matrix / CFG.atol) * CFG.atol
        return np.sort(matrix_rounded, axis=1)
    

@lru_cache(maxsize=None)
def expensive_computation(obj: Polytope) -> bool:
    time.sleep(0.5)  # Simulate an expensive computation
    return hash(obj) % 2 == 0  # Dummy condition


def is_pos_def(A: NDArray, allow_semidef: bool = False) -> bool:

    def is_sym(A: ArrayLike) -> bool:
        return np.allclose(A, A.T, CFG.atol, CFG.rtol)

    if allow_semidef:
        if not is_sym(A):
            return False
        else:
            eigvals = np.linalg.eigvalsh(A)
            return np.all(eigvals >= 0)
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def main():

    global CFG
    CFG.atol = 1E-10

    # Construct several polytopes
    poly_1 = Polytope(np.array([[0, 0], [1, 0], [0, 1]]))
    poly_1_reordered = Polytope(np.array([[0, 0], [0, 1], [1, 0]]))  # Same as poly_1 but different vertex order
    poly_1_perturbed = Polytope(np.array([[0, 0], [1 + 1E-9, 0], [0, 1]]))  # Slightly perturbed version of poly_1
    poly_2 = Polytope(np.array([[0, 0], [2, 0], [0, 2]]))  # Different polytope

    poly_set = {poly_1, poly_1_reordered, poly_1_perturbed, poly_2}
    print(f"Number of unique polytopes in set: {len(poly_set)}")  # Should be 2
    print(f"List of hashes of the polytopes: {[hash(p) for p in [poly_1, poly_1_reordered, poly_1_perturbed, poly_2]]}")

    # Construct several subspaces
    subs_1 = Subspace(np.array([[1, 0, 0], [0, 1, 0]]).T)
    subs_1_scaled = Subspace(np.array([[2, 0, 0], [0, 2, 0]]).T)  # Same subspace but scaled basis
    subs_1_different = Subspace(np.array([[2, 1, 0], [1, -3, 0]]).T)  # Same subspace but different basis
    subs_1_reordered = Subspace(np.array([[0, 1, 0], [1, 0, 0]]).T)  # Same subspace but different order for same basis vectors
    subs_1_perturbed = Subspace(np.array([[1, 0, 0], [0, 1 + 1E-9, 0]]).T)  # Slightly perturbed basis vectors
    subs_2 = Subspace(np.array([[1, 0, 0], [0, 0, 1]]).T)  # Different subspace

    subs_set = {subs_1, subs_1_scaled, subs_1_different, subs_1_reordered, subs_1_perturbed, subs_2}
    print(f"Number of unique subspaces in set: {len(subs_set)}")  # Should be 2
    print(f"List of hashes of the subspaces: {[hash(s) for s in [subs_1, subs_1_scaled, subs_1_different, subs_1_reordered, subs_1_perturbed, subs_2]]}")

    CFG.atol = 1E-6

    # Construct several ellipsoids
    ellps_1 = Ellipsoid(np.array([0, 0]), np.array([[1, 0], [0, 1]]))
    ellps_1_perturbed = Ellipsoid(np.array([0, 0]), np.array([[1 + 1E-9, 0], [0, 1]]))  # Slightly perturbed Q matrix
    ellps_2 = Ellipsoid(np.array([1, 1]), np.array([[1, 0], [0, 1]]))  # Different ellipsoid

    ellps_set = {ellps_1, ellps_1_perturbed, ellps_2}
    print(f"Number of unique ellipsoids in set: {len(ellps_set)}")  # Should be 2

    try:
        degenerate_ellps = Ellipsoid(np.array([0, 0]), np.array([[1, 0], [0, 0]]))  # Degenerate ellipsoid
        expensive_computation(degenerate_ellps)
    except NotImplementedError as e:
        print(f"Expected error for degenerate ellipsoid hashing: \033[35m NotImplementedError: {e}\033[0m")

    # FIXME: Why does the hash change between runs? Is it because of the lru_cache? Ohh no it's because of the randomization in Python `hash` function for security; it adds a random salt.
    for idx in range(niter := 10):
        time_start = time.time()
        res = expensive_computation(poly_1)
        time_end = time.time()
        if idx in [0, 1, niter - 1]:
            print(f"Expensive computation for poly_1 (run {idx + 1}): {res} (took {time_end - time_start:.6f} seconds)")
        elif idx == 2:
            print("...")


if __name__ == "__main__":
    main()