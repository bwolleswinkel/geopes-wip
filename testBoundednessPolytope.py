"""Script to test the boundedness of polytopes using various methods"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal
import warnings

import numpy as np
from numpy.typing import NDArray
import scipy as sp


# ====== PLACEHOLDER ======

@dataclass
class CFG:
    REDUCE: bool = field(default=True, init=False)
    ALGO_BOUNDED: Literal['cardinal_directions', 'single_lp'] = field(default='cardinal_directions', init=False)

global cfg
cfg = CFG()

# ====== PLACEHOLDER ======


class FeasibilityWarning(Warning):
    """Warning raised when an optimization problem does not have a feasible solution"""
    pass


class Polytope:
    """Simple Polytope class for testing purposes"""

    def __init__(self, A: NDArray, b: NDArray) -> None:
        self._Ab: NDArray | None = np.column_stack((A, b))
        self.m: int = b.size
        self._is_bounded: bool | None = None
    
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
    def is_bounded(self, method: Literal['cardinal_directions', 'single_lp'] | None = None) -> bool:
        """Check whether the polytope is bounded using linear programming"""
        if method is None:
            method = cfg.ALGO_BOUNDED
        if False:  # NOTE: This replace `if self._is_bounded is not None`, because we want to always recompute for testing
            return self._is_bounded  
        else:
            match method:
                case 'cardinal_directions':
                    # FROM: GitHub Copilot GPT-4.1, adapted | 2026/01/23 [untested/unverified]
                    n = self.A.shape[1]; c = np.zeros(n)
                    for i in range(n):
                        c[i] = 1
                        res_max = sp.optimize.linprog(-c, A_ub=self.A, b_ub=self.b)
                        res_min = sp.optimize.linprog(c, A_ub=self.A, b_ub=self.b)
                        c[i] = 0
                        if res_max.status == 3 or res_min.status == 3:
                            self._is_bounded = False
                            return self._is_bounded
                        elif not res_max.success or not res_min.success:
                            warnings.warn("Linear program did not solve successfully. Cannot determine boundedness.", RuntimeWarning)
                            self._is_bounded = None
                            return self._is_bounded
                    self._is_bounded = True
                case 'single_lp':
                    # FROM: GitHub Copilot GPT-4.1 | 2026/01/23 [untested/unverified]
                    # Mathematically correct single-LP feasibility check for boundedness:
                    # Find d != 0 such that A d <= 0, ||d||_1 >= 1, by splitting d = d+ - d-, d+, d- >= 0
                    m, n = self.A.shape
                    c = np.zeros(2 * n)  # Feasibility only
                    # A (d+ - d-) <= 0  -->  [A, -A] [d+; d-] <= 0
                    A_ub = np.hstack([self.A, -self.A])
                    b_ub = np.zeros(m)
                    # ||d||_1 = sum(d+ + d-) >= 1  -->  -sum(d+ + d-) <= -1
                    A_norm = -np.ones((1, 2 * n))
                    b_norm = -np.ones(1)
                    # Stack all constraints
                    A_total = np.vstack([A_ub, A_norm])
                    b_total = np.hstack([b_ub, b_norm])
                    bounds = [(0, None)] * (2 * n)
                    eps = 1e-6
                    # Update norm constraint for numerical robustness
                    A_norm = -np.ones((1, 2 * n))
                    b_norm = -np.ones(1) * (1 + eps)
                    A_total = np.vstack([A_ub, A_norm])
                    b_total = np.hstack([b_ub, b_norm])
                    res = sp.optimize.linprog(c, A_ub=A_total, b_ub=b_total, bounds=bounds, method="highs")
                    if res.success and res.status == 0:
                        d_plus = res.x[:n]
                        d_minus = res.x[n:]
                        d = d_plus - d_minus
                        # If the solution is numerically zero, treat as bounded
                        if np.linalg.norm(d, 1) > 1 - 1e-4:
                            self._is_bounded = False
                        else:
                            self._is_bounded = True
                    elif res.status == 2:
                        self._is_bounded = True
                    else:
                        warnings.warn(f"Linear program did not solve successfully (status={res.status}). Cannot determine boundedness.", RuntimeWarning)
                        self._is_bounded = None
                case _:
                    raise ValueError(f"Unknown method '{method}' for boundedness check")
        return self._is_bounded


class Subspace:
    """Simple Subspace class for testing purposes"""

    def __init__(self, basis: NDArray) -> None:
        self.basis: NDArray = basis
        self.dim: int | None = self.basis.shape[1] if cfg.REDUCE else None

    @property
    def basis(self) -> NDArray:
        return self._basis
    
    @basis.setter
    def basis(self, value: NDArray) -> None:
        self._basis = span(value) if cfg.REDUCE else value


def span(A: NDArray) -> NDArray:
    """Compute a basis spanned by the matrix `a` by removing linearly dependent columns"""
    if A.shape[1] == 0 or np.linalg.matrix_rank(A) == A.shape[1]:
        return A
    basis = [A[:, 0]]
    for i in range(1, A.shape[1]):
        if np.linalg.matrix_rank(np.column_stack((*basis, A[:, i]))) > len(basis):
            basis.append(A[:, i])
    return np.column_stack(basis)


def main():
    A, b = np.array([[0, -1], [-1, 0], [1, 1]]), np.array([0, 0, 1])

    P_bounded_2d = Polytope(A, b)
    print(f"2D Bounded Polytope is bounded: {P_bounded_2d.is_bounded})")
    
    cfg.ALGO_BOUNDED = 'single_lp'
    print(f"2D Bounded Polytope is bounded (single_lp): {P_bounded_2d.is_bounded})")
    cfg.ALGO_BOUNDED = 'cardinal_directions'

    A, b = np.array([[-1, -1], [-1, 1]]), np.array([0, 0])

    P_unbounded_2d = Polytope(A, b)
    print("2D Unbounded Polytope is bounded:", P_unbounded_2d.is_bounded)
    cfg.ALGO_BOUNDED = 'single_lp'
    print("2D Unbounded Polytope is bounded (single_lp):", P_unbounded_2d.is_bounded)
    cfg.ALGO_BOUNDED = 'cardinal_directions'

    A, b = np.array([[0, -1, 0], [-1, 0, 0], [1, 1, 1], [0, 0, -1]]), np.array([0, 0, 1, 0])

    P_bounded_3d = Polytope(A, b)
    print("3D Bounded Polytope is bounded:", P_bounded_3d.is_bounded)
    cfg.ALGO_BOUNDED = 'single_lp'
    print("3D Bounded Polytope is bounded (single_lp):", P_bounded_3d.is_bounded)
    cfg.ALGO_BOUNDED = 'cardinal_directions'

    A, b = np.array([[-1, -1, 0], [-1, 1, 0], [0, 0, -1]]), np.array([0, 0, 0])

    P_unbounded_3d = Polytope(A, b)
    print("3D Unbounded Polytope is bounded:", P_unbounded_3d.is_bounded)
    cfg.ALGO_BOUNDED = 'single_lp'
    print("3D Unbounded Polytope is bounded (single_lp):", P_unbounded_3d.is_bounded)
    cfg.ALGO_BOUNDED = 'cardinal_directions'

    A, b = np.array([[-1, -1, 0], [-1, 1, 0]]), np.array([0, 0])

    P_unbounded_3d_2 = Polytope(A, b)
    print("3D Unbounded Polytope is bounded:", P_unbounded_3d_2.is_bounded)
    cfg.ALGO_BOUNDED = 'single_lp'
    print("3D Unbounded Polytope is bounded (single_lp):", P_unbounded_3d_2.is_bounded)
    cfg.ALGO_BOUNDED = 'cardinal_directions'

    A, b = np.vstack((A, np.array([[0, 0, 1], [0, 0, -1]]))), np.append(b, np.array([0, 0]))

    P_unbounded_3d_flat = Polytope(A, b)
    print("3D Unbounded flat Polytope is bounded:", P_unbounded_3d_flat.is_bounded)
    cfg.ALGO_BOUNDED = 'single_lp'
    print("3D Unbounded flat Polytope is bounded (single_lp):", P_unbounded_3d_flat.is_bounded)
    cfg.ALGO_BOUNDED = 'cardinal_directions'

    A, b = np.vstack((A, np.array([1, 0, 0]))), np.append(b, 1)

    P_bounded_3d_flat = Polytope(A, b)
    print("3D Bounded flat Polytope is bounded:", P_bounded_3d_flat.is_bounded)
    cfg.ALGO_BOUNDED = 'single_lp'
    print("3D Bounded flat Polytope is bounded (single_lp):", P_bounded_3d_flat.is_bounded)
    cfg.ALGO_BOUNDED = 'cardinal_directions'


if __name__ == "__main__":
    main()