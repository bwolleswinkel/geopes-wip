"""Contains several generators of known polytopes in n-dimensional space for testing purposes"""

from itertools import product
from typing import Literal

import numpy as np


def hypercube(n: int, repr: Literal["hrepr", "vrepr"], centered: bool = False) -> np.ndarray:
    vals = [-1, 1] if centered else [0, 1]
    return np.array(list(product(vals, repeat=n)))
