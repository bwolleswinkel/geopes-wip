"""Script to test the abstract base class ConvexRegion and its subclasses Polytope and Subspace"""

from abc import ABC, abstractmethod
import numpy as np


class ConvexRegion(ABC):
    """Abstract base class for convex regions in Euclidean space"""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the Hausdorff dimension of the convex region"""
        pass

    @property
    @abstractmethod
    def n(self) -> int:
        """Return the ambient dimension of the convex region"""
        pass

    @property
    @abstractmethod
    def vol(self) -> float:
        """Calculate the volume of the convex region"""
        pass

    @property
    @abstractmethod
    def __contains__(self, item):
        pass


class Polytope(ConvexRegion):
    """Class representing a convex polytope defined by linear inequalities"""

    def __init__(self):
        ...


def main():
    poly = Polytope()


if __name__ == "__main__":
    main()