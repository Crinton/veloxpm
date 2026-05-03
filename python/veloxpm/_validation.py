from __future__ import annotations

import numpy as np


def ensure_real_square_matrix(matrix: np.ndarray, n: int, owner: str) -> np.ndarray:
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if matrix.shape != (n, n):
        raise ValueError(f"{owner} expects an input matrix with shape ({n}, {n}).")
    if np.iscomplexobj(matrix):
        raise TypeError(f"{owner} expects a real matrix H.")
    return matrix
