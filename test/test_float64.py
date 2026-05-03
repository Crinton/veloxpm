from __future__ import annotations

import time

import numpy as np
from scipy.linalg import expm

from veloxpm import ExpMatFloat64


SIZES = [128, 256, 512, 1024, 2048, 4096]
SEED = 20260427


def generate_matrix(size: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + size)
    matrix = rng.standard_normal((size, size)).astype(np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    norm1 = np.linalg.norm(matrix, ord=1)
    if norm1 > 0:
        matrix *= 0.25 / norm1
    return np.ascontiguousarray(matrix)


def main() -> None:
    print("float64: exp(H)")
    print("size | veloxpm_s | scipy_s | speedup | rel_fro_error | max_abs_error")
    print("-" * 78)

    for size in SIZES:
        matrix = generate_matrix(size)

        calc = ExpMatFloat64(size)
        try:
            t0 = time.perf_counter()
            result = calc.run(matrix)
            veloxpm_time = time.perf_counter() - t0
        finally:
            calc.free()

        t0 = time.perf_counter()
        reference = expm(matrix)
        scipy_time = time.perf_counter() - t0

        diff = result - reference
        rel_fro = np.linalg.norm(diff) / max(np.linalg.norm(reference), 1e-30)
        max_abs = np.max(np.abs(diff))
        speedup = scipy_time / veloxpm_time

        print(
            f"{size:4d} | "
            f"{veloxpm_time:8.3f} | "
            f"{scipy_time:7.3f} | "
            f"{speedup:6.1f}x | "
            f"{rel_fro:13.3e} | "
            f"{max_abs:13.3e}"
        )


if __name__ == "__main__":
    main()
