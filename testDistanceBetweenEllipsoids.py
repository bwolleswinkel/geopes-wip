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


def dist(ellps_1: Ellipsoid, ellps_2: Ellipsoid) -> float:
    """Compute the distance between two ellipsoids using QCQP"""
    if ellps_1.n != ellps_2.n:
        raise ValueError("Ellipsoids must be in the same dimension to compute distance")
    
    x, y = cvx.Variable(ellps_1.n), cvx.Variable(ellps_2.n)

    c_1, c_2 = ellps_1.c, ellps_2.c
    Q_1, Q_2 = ellps_1.Q, ellps_2.Q
    
    cons_1 = [cvx.quad_form(x - c_1, Q_1) <= 1]
    cons_2 = [cvx.quad_form(y - c_2, Q_2) <= 1]
    
    objective = cvx.Minimize(cvx.sum_squares(x - y))
    problem = cvx.Problem(objective, cons_1 + cons_2)
    
    problem.solve()
    if problem.status == cvx.OPTIMAL:
        return np.sqrt(problem.value)  # NOTE: problem.value is the squared distance due to sum_squares


if __name__ == "__main__":
    c_1, Q_1 = np.array([0, 0]), np.eye(2)
    c_2, Q_2 = np.array([2, 2]), np.eye(2)

    ellps_1 = Ellipsoid(c_1, Q_1)
    ellps_2 = Ellipsoid(c_2, Q_2)

    dist_1_2 = dist(ellps_1, ellps_2)
    print(f"Distance between ellipsoids 1 and 2: {dist_1_2:.6f}")

