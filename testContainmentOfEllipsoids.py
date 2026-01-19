"""Script to test distance computation between ellipsoids"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import cvxpy as cvx


# ------ FUNCTIONS ------


@dataclass
class Ellipsoid:
    c: NDArray  # Center
    Q: NDArray  # Shape matrix (positive definite)

    def __post_init__(self):
        self.n = self.c.size


def containment(ellps_1: Ellipsoid, ellps_2: Ellipsoid, EPS: float = 1E-6) -> bool:
    """Compute whether ellps_1 is contained in ellps_2 using the S-procedure"""
    if ellps_1.n != ellps_2.n:
        raise ValueError("Ellipsoids must be in the same dimension to compute distance")
    
    c_1, c_2 = ellps_1.c, ellps_2.c
    Q_1, Q_2 = ellps_1.Q, ellps_2.Q
    
    # NOTE: Should we use `lambda_` or `lbd` as variable name? `lambda` is a reserved keyword in Python
    lambda_ = cvx.Variable(nonneg=True)

    cons_pos = [lambda_ >= EPS]
    
    # FROM: GitHub Copilot Claude Sonnet 4 | 2026/01/19
    n = ellps_1.n
    
    # Create full matrices manually
    M1 = np.zeros((n + 1, n + 1))
    M1[:n, :n] = Q_1
    M1[:n, n] = -Q_1 @ c_1
    M1[n, :n] = -c_1.T @ Q_1
    M1[n, n] = c_1.T @ Q_1 @ c_1 - 1
    
    M2 = np.zeros((n + 1, n + 1))
    M2[:n, :n] = Q_2
    M2[:n, n] = -Q_2 @ c_2
    M2[n, :n] = -c_2.T @ Q_2
    M2[n, n] = c_2.T @ Q_2 @ c_2 - 1
    
    # LMI constraint: M2 - lambda*M1 >= epsilon*I
    cons_LMI = [M2 - lambda_ * M1 >> EPS * np.eye(n + 1)]
    
    objective = cvx.Minimize(0)
    problem = cvx.Problem(objective, cons_pos + cons_LMI)
    
    problem.solve()
    if problem.status == cvx.OPTIMAL:
        return True
    else:
        return False


if __name__ == "__main__":
    c_1, Q_1 = np.array([0, 0]), np.eye(2)
    c_2, Q_2 = np.array([1, 1]), np.eye(2)

    ellps_1 = Ellipsoid(c_1, Q_1)
    ellps_2 = Ellipsoid(c_2, Q_2)

    containment_result = containment(ellps_1, ellps_2)
    print(f"Containment result: {containment_result}")

