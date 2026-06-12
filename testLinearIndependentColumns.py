"""Compute linear independent columns using QR-decomposition"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp

if TYPE_CHECKING:
    from typing import Final

    from numpy.typing import NDArray


@dataclass
class Config:
    atol: float = 1E-8


CFG: Final[Config] = Config()


def span(A: NDArray) -> NDArray:
    _, R, P = sp.linalg.qr(A, pivoting=True)
    rank = np.sum(np.abs(np.diag(R)) > CFG.atol)
    return A[:, sorted(P[:rank])]


if __name__ == '__main__':
    # Create the matrix
    A = np.array([[1, 0, 0, 0],
                  [0, 0, 0, 1],
                  [1, 1, 1, 0],
                  [0, 0, 0, 1]])
    
    # Extract the basis
    basis = span(A)

    # Print the results
    print(f"Basis:\n{basis} (shape={basis.shape})")