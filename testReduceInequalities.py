"""Script to test reducing inequalities to compute a minimal representation of a polytope"""

from typing import Literal
from dataclasses import dataclass, field
import warnings

import numpy as np
from numpy.typing import NDArray
import scipy as sp


class FeasibilityWarning(Warning):
    """Warning raised when an optimization problem does not have a feasible solution"""
    pass


class Polytope:
    """Simple Polytope class for testing purposes"""

    def __init__(self, A: NDArray, b: NDArray) -> None:
        self._Ab = np.column_stack((A, b))
        self.m = b.size
        self.is_hrepr = True
    
    @property
    def A(self) -> NDArray:
        if not self.is_hrepr:
            raise ValueError("Polytope is not in H-representation.")
        return self._Ab[:, :-1]
    
    @property
    def b(self) -> NDArray:
        if not self.is_hrepr:
            raise ValueError("Polytope is not in H-representation.")
        return self._Ab[:, -1]
    
    def reduce(self):
        """Remove redundant inequalities from the polytope representation to obtain a minimal representation"""
        redundant = np.zeros(self.m, dtype=bool)
        for idx in range(self.m):
            res = sp.optimize.linprog(c=-self.A[idx, :], A_ub=self.A[~redundant, :], b_ub=self.b[~redundant], bounds=(None, None))
            # FIXME: How to treat the case when there is no feasible solution? Should not happen in theory, but numerically it might. Currently, we just ignore that case.
            if not res.success:
                # NOTE: This should theoretically never happen, but numerically it might
                warnings.warn(f"Linear program to check redundancy of inequality {idx} was not successful. Ignoring this inequality for redundancy checking.", FeasibilityWarning, stacklevel=2)
                continue
            # FIXME: I need to check whether this is actually the right condition/implementation
            elif -res.fun < self.b[idx] and not np.isclose(-res.fun, self.b[idx]):
                redundant[idx] = True
        self._Ab = self._Ab[~redundant, :]
        # QUESTION: Do we want to make this a property instead? Such that `m` is computed on the fly?
        self.m = self._Ab.shape[0]


class Subspace:
    """Simple Subspace class for testing purposes"""

    def __init__(self, basis: NDArray) -> None:
        self.basis = basis

    def reduce(self):
        self.basis = span(self.basis)


def span(A: NDArray) -> NDArray:
    """Compute a basis spanned by the matrix `a` by removing linearly dependent columns"""
    if A.shape[1] == 0 or np.linalg.matrix_rank(A) == A.shape[1]:
        return A
    basis = [A[:, 0]]
    for i in range(1, A.shape[1]):
        if np.linalg.matrix_rank(np.column_stack((*basis, A[:, i]))) > len(basis):
            basis.append(A[:, i])
    return np.column_stack(basis)
    

# FIXME: Do we even want to have this as an external function
def reduce(obj: Polytope | Subspace) -> Polytope | Subspace:
    """..."""
    return obj.reduce()


# ====== TEST FUNCTION ====== 


# QUESTION: Is this a good way to go about testing classes? Because we don't want to import the actual class, as we want to test the external function
@dataclass
class PolytopeAB:
    A: NDArray
    b: NDArray

    def __post_init__(self) -> None:
        self._Ab = np.column_stack((self.A, self.b))
        self.m = self.b.size
        self.is_hrepr = True

    def reduce(*args, **kwargs) -> None:
        Polytope.reduce(*args, **kwargs)


def test_reduce() -> None:
    """This is an example test function to test the `reduce` function"""
    A = np.array([[-1,  0], 
                  [ 0, -1], 
                  [ 1,  1],
                  [-1,  0],
                  [ 0, -1]])
    b = np.array([0, 0, 1, 1, 1])

    poly_1 = PolytopeAB(A, b)
    poly_1.reduce()
    assert poly_1._Ab.shape[0] == 3, "Polytope reduction failed for redundant inequalities"
    assert np.allclose(poly_1._Ab, np.array([[-1,  0, 0],
                                             [ 0, -1, 0],
                                             [ 1,  1, 1]])), "Polytope reduction produced incorrect result"
    try:  # NOTE: This is an incorrect test, but serves as an example
        assert isinstance(poly_1._Ab, str), "Here, as expected, an AssertionError should be raised (because the condition is false)"
    except AssertionError as e:
        print("Caught expected AssertionError:", e)


# ====== TEST FUNCTION ====== 


def main():
    A, b = np.array([[0, -1], [-1, 0], [1, 1]]), np.array([0, 0, 1])
    A_red, b_red = np.array([[0, -1], [-1, 0]]), np.array([1, 1])

    A, b = np.vstack((A, A_red)), np.hstack((b, b_red))
    P_hrepr = Polytope(A, b)
    P_hrepr.reduce()
    print("Triangle with redundant inequalities:")
    print("Reduced A:\n", P_hrepr.A)
    print("Reduced b:\n", P_hrepr.b)

    A, b = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]]), np.array([0, 0, 1, 1])
    A_red, b_red = np.array([[1, 0], [0, 1]]), np.array([2, 2])

    A, b = np.vstack((A, A_red)), np.hstack((b, b_red))
    P_hrepr = Polytope(A, b)
    reduce(P_hrepr)
    print("Cube with redundant inequalities:")
    print("Reduced A:\n", P_hrepr.A)
    print("Reduced b:\n", P_hrepr.b)

    basis = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]]).T

    S = Subspace(basis)
    reduce(S)
    print("Subspace with redundant basis vectors:")
    print("Reduced basis:\n", S.basis)

    # Check the test function
    test_reduce()


if __name__ == "__main__":
    main()