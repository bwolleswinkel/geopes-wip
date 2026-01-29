"""Script to test calculating the (principal) angles between subspaces"""

import numpy as np
from numpy.typing import NDArray, ArrayLike
import scipy as sp


class Subspace:
    """Simple class representing a subspace via its basis vectors"""

    def __init__(self, basis: NDArray, to_orth: bool = False) -> None:
        self.basis = atleast_2d_col(basis) if not to_orth else sp.linalg.orth(atleast_2d_col(basis))

    @property
    def basis(self) -> NDArray:
        return self._basis
    
    @basis.setter
    def basis(self, value: NDArray) -> None:
        self._basis = value

    def angle(self, other: Subspace) -> list[float]:
        """Calculate the principal angles between this subspace and another"""
        return sp.linalg.subspace_angles(self.basis, other.basis)
    
    @classmethod
    def from_normal(cls, normal: NDArray) -> Subspace:
        """Create a Subspace from its normal vectors"""
        normal = atleast_2d_col(normal)
        if normal.shape[0] > normal.shape[1]:
            basis = sp.linalg.null_space(normal.T)
        else:
            basis = np.empty((normal.shape[0], 0))
        return cls(basis, to_orth=True)
    

def atleast_2d_col(arr: ArrayLike) -> NDArray:
    """Convert scalars/1D arrays to column vectors, and leave ND arrays (where N >= 2) unchanged"""
    arr = np.asarray(arr)
    if arr.ndim < 1:
        return arr.reshape(1, 1)
    elif arr.ndim == 1:
        return arr.reshape(-1, 1)
    else:
        return arr


def main() -> None:
    basis_1 = np.array([1, 0, 0])
    basis_2 = np.array([[1, 1], [0, 0], [0, 1]])
    basis_3 = np.array([3, -1, 2])

    subs_1 = Subspace(basis_1)
    subs_2 = Subspace(basis_2, to_orth=True)
    subs_3 = Subspace(basis_3, to_orth=True)
    subs_4 = Subspace.from_normal(np.array([1, 1, 0]))
    subs_5 = Subspace.from_normal(np.eye(3))
    
    basis_x = np.random.randint(0, 10, size=(10, 5))
    basis_y = np.random.randint(0, 10, size=(10, 7))
    subs_x = Subspace(basis_x, to_orth=True)
    subs_y = Subspace(basis_y, to_orth=True)

    np.set_printoptions(precision=2, suppress=True)

    print("Subspace 1 basis:\n", subs_1.basis)
    print("Subspace 2 basis:\n", subs_2.basis)
    print("Subspace 3 basis:\n", subs_3.basis)
    print("Subspace 4 basis (from normal):\n", subs_4.basis)
    print("Subspace 5 basis (from normal):\n", subs_5.basis)

    print(f"Angles between Subspace 1 and 2: {subs_1.angle(subs_2)} (len={len(subs_1.angle(subs_2))} == {min(subs_1.basis.shape[1], subs_2.basis.shape[1])})")
    print(f"Angles between Subspace 1 and 3: {subs_1.angle(subs_3)} (len={len(subs_1.angle(subs_3))} == {min(subs_1.basis.shape[1], subs_3.basis.shape[1])})")
    print(f"Angles between Subspace 2 and 3: {subs_2.angle(subs_3)} (len={len(subs_2.angle(subs_3))} == {min(subs_2.basis.shape[1], subs_3.basis.shape[1])})")
    print(f"Angles between Subspace 1 and 4: {subs_1.angle(subs_4)} (len={len(subs_1.angle(subs_4))} == {min(subs_1.basis.shape[1], subs_4.basis.shape[1])})")
    print(f"Angles between Subspace 2 and 4: {subs_2.angle(subs_4)} (len={len(subs_2.angle(subs_4))} == {min(subs_2.basis.shape[1], subs_4.basis.shape[1])})")
    print(f"Angles between Subspace 3 and 4: {subs_3.angle(subs_4)} (len={len(subs_3.angle(subs_4))} == {min(subs_3.basis.shape[1], subs_4.basis.shape[1])})")
    print(f"Angles between Subspace 1 and 5: {subs_1.angle(subs_5)} (len={len(subs_1.angle(subs_5))} == {min(subs_1.basis.shape[1], subs_5.basis.shape[1])})")

    print(f"Angles between Subspace X and Y: {subs_x.angle(subs_y)} (len={len(subs_x.angle(subs_y))} == {min(subs_x.basis.shape[1], subs_y.basis.shape[1])})")

    print(f"Angles between Subspace 1 and 1: {subs_1.angle(subs_1)} (len={len(subs_1.angle(subs_1))} == {min(subs_1.basis.shape[1], subs_1.basis.shape[1])})")


if __name__ == "__main__":
    main()
