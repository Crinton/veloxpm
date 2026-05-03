from __future__ import annotations

from typing import Any

import numpy as np

from . import _veloxpm_core as _core
from ._validation import ensure_real_square_matrix



class _BaseCalculator:
    _backend_cls: type[Any]

    def __init__(self, n: int) -> None:
        self.n = int(n)
        self._backend = self._backend_cls(self.n)

    def __enter__(self) -> "_BaseCalculator":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        self.free()
        return False

    def run(self, matrix) -> np.ndarray:
        array = ensure_real_square_matrix(matrix, self.n, self.__class__.__name__)
        return self._backend.run(array).reshape(self.n, self.n)

    def free(self) -> None:
        self._backend.free()


class ExpMatFloat32(_BaseCalculator):
    _backend_cls = _core.ExpMatFloat32


class ExpMatFloat64(_BaseCalculator):
    _backend_cls = _core.ExpMatFloat64


class ExpMatComplex64(_BaseCalculator):
    _backend_cls = _core.ExpMatComplex64


class ExpMatComplex128(_BaseCalculator):
    _backend_cls = _core.ExpMatComplex128
