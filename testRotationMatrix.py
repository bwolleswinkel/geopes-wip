"""Script to generate rotation matrices in arbitrary number of dimensions"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


# FROM: GitHub Copilot GPT-4.1 | 2026/01/22
# TODO: All this code is completely untested/unverified!
def rot_mat(angles: list[float] | float, n: int | None = None, convention: Literal['givens', 'euler', 'tait_bryan', 'yaw_pitch_roll'] = 'givens') -> NDArray:
    """Compute the rotation matrix in ND given a list of angles in degrees. The dimension N is inferred from the number of angles provided.

    Parameters
    ----------
    angles : list[float] or float
        List of angles (or single angle if rotation is 2D). Should be of length N * (N - 1) / 2 ∈ ℕ for ND.
    convention : 'givens' | 'euler' | 'tait_bryan' | 'yaw_pitch_roll', default 'givens'
        Rotation convention for the order of applying rotations.
        - 'givens': Sequential Givens rotations in all unique planes (ND)
        - 'euler': 3D only, Z-X-Z extrinsic (or ZYZ intrinsic) convention
        - 'tait_bryan': 3D only, Z-Y-X extrinsic (or XYZ intrinsic) convention
        - 'yaw_pitch_roll': 3D only, Z-Y-X extrinsic (same as 'tait_bryan')

    Returns
    -------
    R : NDArray
        Rotation matrix

    Raises
    ------
    ValueError  # TODO: Change to DimensionError
        If the number of angles or convention is not compatible with the dimension.

    Notes
    -----
    In SO(n), there are n * (n - 1) / 2 unique angles. Tait–Bryan angles are also called Cardan angles, nautical angles, or heading, elevation, and bank.

    """

    def givens_rotation(n: int, angles: list[float]) -> NDArray:
        """Compute the rotation matrix using sequential Givens rotations"""
        R, idx = np.eye(n), 0
        for i in range(n):
            for j in range(i + 1, n):
                theta, G = angles[idx], np.eye(n)
                G[i,i], G[i,j] = np.cos(theta), -np.sin(theta)
                G[j,i], G[j,j] = np.sin(theta), np.cos(theta)
                R = R @ G
                idx += 1
        return R
    
    def euler_rotation(alpha: float, beta: float, gamma: float) -> NDArray:
        """Compute the rotation matrix using Z-X-Z extrinsic (ZYZ intrinsic) Euler angles"""
        Rz1 = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                        [np.sin(alpha),  np.cos(alpha), 0],
                        [0, 0, 1]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(beta), -np.sin(beta)],
                       [0, np.sin(beta),  np.cos(beta)]])
        Rz2 = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                        [np.sin(gamma),  np.cos(gamma), 0],
                        [0, 0, 1]])
        return Rz1 @ Rx @ Rz2
    
    def tait_bryan_rotation(yaw: float, pitch: float, roll: float) -> NDArray:
        """Compute the rotation matrix using Z-Y-X extrinsic (XYZ intrinsic) Tait-Bryan angles"""
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw),  np.cos(yaw), 0],
                       [0, 0, 1]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll),  np.cos(roll)]])
        return Rz @ Ry @ Rx
    
    angles = np.radians([angles] if isinstance(angles, float | int) else angles)
    # NOTE: In SO(n), there are n * (n - 1) / 2 unique angles. Here, we infer n from the number of angles provided.
    if n is None:
        n = int((1 + np.sqrt(1 + 8 * len(angles))) / 2)
        if n * (n - 1) // 2 != len(angles):
            raise ValueError(f"Invalid number of angles provided ({len(angles) = }). Note that for SO(n), the number of angles must equal n * (n - 1) / 2.")
    else:
        if n * (n - 1) // 2 != len(angles):
            raise ValueError(f"Number of angles {len(angles)} does not match the specified dimension n = {n} (requires n * (n - 1) / 2 = {n * (n - 1) // 2} angle{'s' if n * (n - 1) // 2 != 1 else ''})")
    if convention != 'givens' and n != 3:
        raise ValueError(f"Convention '{convention}' only supported for 3D rotations")
    match convention:
        case 'givens':
            return givens_rotation(n, angles)
        case 'euler': 
            return euler_rotation(*angles)
        case 'tait_bryan' | 'yaw_pitch_roll':
            return tait_bryan_rotation(*angles)
        case _:
            raise ValueError(f"Unknown convention '{convention}'.")
        

def rot_mat_2d(angle: float) -> NDArray:
    """Compute the 2D rotation matrix for a given angle in degrees"""
    return rot_mat(angle)


def rot_mat_3d(angles: list[float], convention: Literal['givens', 'euler', 'tait_bryan', 'yaw_pitch_roll'] = 'givens') -> NDArray:
    """Compute the 3D rotation matrix for given angles in degrees"""
    if len(angles) != 3:
        raise ValueError("3D rotation requires exactly 3 angles")
    return rot_mat(angles, convention=convention)


def main():

    R_2d = rot_mat(80)
    print("Rotation Matrix in 2D:")
    print(np.round(R_2d, 2) + np.zeros_like(R_2d))

    R_2d_alt = rot_mat_2d(80)
    print("\nRotation Matrix in 2D (alt):")
    print(np.round(R_2d_alt, 2) + np.zeros_like(R_2d_alt))

    R_3d = rot_mat([0, 45, 45])
    print("\nRotation Matrix in 3D:")
    print(np.round(R_3d, 2) + np.zeros_like(R_3d))

    R_3d_alt = rot_mat_3d([0, 45, 45])
    print("\nRotation Matrix in 3D (alt):")
    print(np.round(R_3d_alt, 2) + np.zeros_like(R_3d_alt))

    R_3d_givens = rot_mat([0, 0, 60], convention='givens')
    print("\nRotation Matrix in 3D (Givens):")
    print(np.round(R_3d_givens, 2) + np.zeros_like(R_3d_givens))

    R_3d_euler = rot_mat([0, 0, 60], convention='euler')
    print("\nRotation Matrix in 3D (Euler Z-X-Z):")
    print(np.round(R_3d_euler, 2) + np.zeros_like(R_3d_euler))

    R_3d_tait_bryan = rot_mat([0, 0, 60], convention='tait_bryan')
    print("\nRotation Matrix in 3D (Tait-Bryan Z-Y-X):")
    print(np.round(R_3d_tait_bryan, 2) + np.zeros_like(R_3d_tait_bryan))

    R_3d_yaw_pitch_roll = rot_mat([0, 0, 60], convention='yaw_pitch_roll')
    print("\nRotation Matrix in 3D (Yaw-Pitch-Roll Z-Y-X):")
    print(np.round(R_3d_yaw_pitch_roll, 2) + np.zeros_like(R_3d_yaw_pitch_roll))

    R_4d = rot_mat([30, 45, 60, 15, 25, 35])
    print("\nRotation Matrix in 4D:")
    print(np.round(R_4d, 2) + np.zeros_like(R_4d))

    try:
        _ = rot_mat([30, 45, 60, 15])  # Should raise ValueError
    except ValueError as e:
        print(f"Expected error for invalid input: {e}")

    try:
        _ = rot_mat([30, 45], n=2)  # Should raise ValueError
    except ValueError as e:
        print(f"Expected error for invalid input: {e}")


if __name__ == "__main__":
    main()