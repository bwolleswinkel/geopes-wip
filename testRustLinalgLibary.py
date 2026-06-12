"""Test integration with a Rust backend"""

import time

import numpy as np
import scipy as sp

import rlinalg as rl


if __name__ == '__main__':
    # Compute the linearly independent rows
    A = np.array([
        [0, 1, 0, 0],
        [0, 1, 4, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 0],])
    # A = np.array([[[1], [2]], [[3], [4]]])
    A = np.random.randint(1, 3, size=(100, 100))
    time_start = time.process_time()
    A_red = rl.span(A)
    time_stop = time.process_time()

    time_start_qr = time.process_time()
    _ = np.linalg.qr(A)
    time_stop_qr = time.process_time()

    print(A_red)
    print(A_red.shape)
    print(f"Rust: {time_stop - time_start}")

    print(f"Qr: {time_stop_qr - time_start_qr}")

    # A is your matrix
    time_start_sp_qr = time.process_time()
    _, R, P = sp.linalg.qr(A, pivoting=True)

    # P is an array of indices that orders columns by independence.
    # The first 'rank' columns are the linearly independent ones.
    # To get the first independent columns in their ORIGINAL relative order:
    rank = np.sum(np.abs(np.diag(R)) > 1e-10) # Adjust tolerance as needed
    independent_indices = np.sort(P[:rank])
    A_qr = A[:, independent_indices]
    time_stop_sp_qr = time.process_time()

    print(A_qr)
    print(f"Qr (sp): {time_stop_sp_qr - time_start_sp_qr}")