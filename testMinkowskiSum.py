"""Script to test Minkowski sum implementation"""

from __future__ import annotations
import time

import numpy as np
from numpy.typing import NDArray
import scipy as sp
import matplotlib.pyplot as plt


class Polytope:
    def __init__(self, verts: NDArray | None = None, n: int | None = None) -> None:
        if verts is not None and n is None:
            self._verts = conv(verts)
        else:
            if verts is None and n is not None:
                self._verts = np.empty((n, 0))
            else:
                raise ValueError("Either 'verts' or 'n' must be provided")
        self.n = self.verts.shape[1] if verts is not None else n

    def __add__(self, other: Polytope) -> Polytope:
        return Polytope(mink_sum(self, other))

    @property
    def verts(self) -> NDArray:
        return self._verts
        

def conv(verts: NDArray) -> NDArray:
    hull = sp.spatial.ConvexHull(verts.T)
    return verts[:, hull.vertices.T]


def mink_sum(poly_1: Polytope, poly_2: Polytope) -> NDArray:
    # FROM: GitHub Copilot Claude Sonnet 4 | 2026/01/20
    total_verts_sum = poly_1.verts.T[:, None, :] + poly_2.verts.T[None, :, :]
    return total_verts_sum.reshape(-1, total_verts_sum.shape[-1]).T


def mink_sum_naive(poly_1: Polytope, poly_2: Polytope) -> NDArray:
    total_verts = np.array([v_1 + v_2 for v_1 in poly_1.verts.T for v_2 in poly_2.verts.T]).T
    return total_verts


if __name__ == "__main__":
    verts_1 = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]).T
    verts_2 = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]).T
    
    poly_1 = Polytope(verts_1)
    poly_2 = Polytope(verts_2)

    poly_sum = poly_1 + poly_2

    print("Vertices of Polytope 1:")
    print(poly_1.verts)
    print("Vertices of Polytope 2:")
    print(poly_2.verts)
    print("Vertices of Minkowski Sum:")
    print(poly_sum.verts)

    # NOTE: This scales very poorly with dimension n > 5
    poly_big_1 = Polytope(np.random.rand(5, 1000))
    poly_big_2 = Polytope(np.random.rand(5, 1000))

    start_time = time.time()
    poly_sum_big = poly_big_1 + poly_big_2
    end_time = time.time()
    print(f"Time taken for Minkowski sum of big polytopes: {end_time - start_time} seconds")

    start_time_naive = time.time()
    verts_naive = conv(mink_sum_naive(poly_big_1, poly_big_2))
    end_time_naive = time.time()
    print(f"Time taken for naive Minkowski sum of big polytopes: {end_time_naive - start_time_naive} seconds")  # NOTE: This is only fractionally slower, as conv(...) takes by far the most time

    # ------ PLOTTING ------

    _, ax = plt.subplots()
    ax.fill(poly_1.verts[0, :], poly_1.verts[1, :], alpha=0.5, label="Polytope 1")
    ax.fill(poly_2.verts[0, :], poly_2.verts[1, :], alpha=0.5, label="Polytope 2")
    ax.fill(poly_sum.verts[0, :], poly_sum.verts[1, :], alpha=0.5, label="Minkowski Sum")
    ax.legend()
    ax.set_title(f"Minkowski Sum of Two Polytopes")
    ax.set_xlabel(r"$x_{1}$")
    ax.set_ylabel(r"$x_{2}$")
    ax.grid()
    ax.axis('equal')

    # Show the plot
    plt.show()