from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

class ExpMatFloat32:
    n: int

    def __init__(self, n: int) -> None:
        """Initialize for n x n matrices, allocating GPU memory."""
        ...

    def run(self, arr_a: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute exp(H) for real float32 matrix H. Returns flat (n*n,) array."""
        ...

    def free(self) -> None:
        """Free GPU memory."""
        ...


class ExpMatFloat64:
    n: int

    def __init__(self, n: int) -> None:
        """Initialize for n x n matrices, allocating GPU memory."""
        ...

    def run(self, arr_a: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute exp(H) for real float64 matrix H. Returns flat (n*n,) array."""
        ...

    def free(self) -> None:
        """Free GPU memory."""
        ...


class ExpMatComplex64:
    n: int

    def __init__(self, n: int) -> None:
        """Initialize for n x n matrices, allocating GPU memory."""
        ...

    def run(self, arr_a: NDArray[np.float32]) -> NDArray[np.complex64]:
        """Compute exp(iH) for real float32 matrix H. Returns flat (n*n,) complex64 array."""
        ...

    def free(self) -> None:
        """Free GPU memory."""
        ...


class ExpMatComplex128:
    n: int

    def __init__(self, n: int) -> None:
        """Initialize for n x n matrices, allocating GPU memory."""
        ...

    def run(self, arr_a: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Compute exp(iH) for real float64 matrix H. Returns flat (n*n,) complex128 array."""
        ...

    def free(self) -> None:
        """Free GPU memory."""
        ...
