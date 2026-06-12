"""Test integration of `rspatial` with a Rust backend"""

import time

import numpy as np

import rspatial as rs


if __name__ == '__main__':
    # Compute the linearly independent rows
    A=np.array([[-1,  0],
                [ 1,  0],
                [ 0, -1],
                [-1,  1],
                [ 1,  1]])
    b=np.array([0,
                1,
                0,
                1,
                2])

    Ab = np.column_stack((A, b))

    Ab_eq = np.empty((4, 0))

    time_start = time.process_time()
    verts, rays = rs.enum_gens(Ab, Ab_eq)
    time_stop = time.process_time()

    print(f"Verts:\n{verts} (shape={verts.shape})")

    print(f"Rays:\n{rays} (shape={rays.shape})")