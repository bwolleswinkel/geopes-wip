"""Contains several archetype examples of polytopes for testing purposes"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolytopeData:
    name: str
    n: int
    A: np.ndarray | None = None
    b: np.ndarray | None = None
    verts: np.ndarray | None = None
    m: int | None = None
    k: int | None = None
    A_eq: np.ndarray | None = None
    b_eq: np.ndarray | None = None
    rays: np.ndarray | None = None
    # QUESTION: Should we actually set these flags in the data? Or should we infer them when needed?
    is_degen: bool | None = None
    is_bounded: bool | None = None
    is_full_dim: bool | None = None
    is_singleton: bool | None = None
    is_empty: bool | None = None
    is_minimal: bool | None = None

    def __post_init__(self):
        if self.A.shape[0] != self.m:
            raise ValueError(
                f"Expected A to have {self.m} rows, but got {self.A.shape[0]}"
            )
        if self.A.shape[1] != self.n:
            raise ValueError(
                f"Expected A to have {self.n} columns, but got {self.A.shape[1]}"
            )


# ============================================
# 1D ARCHETYPES (nondegenerate, non-redundant)
# ============================================

UNIT_LINE_SEGMENT = PolytopeData(
    name="unit_line_segment",
    n=1,
    A=np.array([[1], [-1]]),
    b=np.array([1, 0]),
    verts=np.array([[0], [1]]),
    m=2,
    k=2,
    is_degen=False,
    is_bounded=True,
    is_full_dim=True,
    is_singleton=False,
    is_empty=False,
    is_minimal=True,
)

OFFSET_LINE_SEGMENT = PolytopeData(
    name="offset_line_segment",
    n=1,
    A=np.array([[1], [-1]]),
    b=np.array([2, 1]),
    verts=np.array([[1], [2]]),
    m=2,
    k=2,
    is_degen=False,
    is_bounded=True,
    is_full_dim=True,
    is_singleton=False,
    is_empty=False,
    is_minimal=True,
)

SCALED_LINE_SEGMENT = PolytopeData(
    name="scaled_line_segment",
    n=1,
    A=np.array([[1], [-1]]),
    b=np.array([2, 0]),
    verts=np.array([[0], [2]]),
    m=2,
    k=2,
    is_degen=False,
    is_bounded=True,
    is_full_dim=True,
    is_singleton=False,
    is_empty=False,
    is_minimal=True,
)

# =============
# 2D ARCHETYPES
# =============

UNIT_SQUARE = PolytopeData(
    name="unit_square",
    n=2,
    A=np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
    b=np.array([1, 1, 0, 0]),
    verts=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
    m=4,
    k=4,
    is_degen=False,
    is_bounded=True,
    is_full_dim=True,
    is_singleton=False,
    is_empty=False,
    is_minimal=True,
)
