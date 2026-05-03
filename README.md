# veloxpm

`veloxpm` is a CUDA-accelerated matrix exponential library with Python bindings.

It provides two execution paths:

- `ExpMatFloat32` and `ExpMatFloat64` compute `exp(H)` for real input matrices.
- `ExpMatComplex64` and `ExpMatComplex128` accept a real matrix `H` and compute `exp(iH)` using the optimized imaginary-form implementation.

## Features

- CUDA backend built on cuBLAS and cuSolver
- Support for `float32`, `float64`, `complex64`, and `complex128`
- Wheel packaging for Python installation
- Pytest coverage for small reference cases and larger random diagonal cases

## Requirements

- Python 3.10+
- CUDA toolkit available to CMake/NVCC
- NumPy

## Installation

### Option 1: Build a wheel, then install it

From the project root:

```bash
python setup.py bdist_wheel
pip install dist/veloxpm-2025.6012a0-*.whl
```

The generated wheel is written to `dist/`.

### Option 2: Install directly with setuptools

```bash
python setup.py install
```

## Quick Start

### Context manager style

```python
import numpy as np
from veloxpm import ExpMatFloat32, ExpMatComplex64

H_real = np.array([[0.0, 0.25], [-0.5, 0.0]], dtype=np.float32)
with ExpMatFloat32(2) as calc:
    exp_h = calc.run(H_real)

H_imag = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
with ExpMatComplex64(2) as calc:
    exp_ih = calc.run(H_imag)
```

### Reuse one calculator for better efficiency

If you need to evaluate many matrices with the same shape, create the calculator once and reuse it:

```python
import numpy as np
from veloxpm import ExpMatFloat32, ExpMatComplex64

N = 1024
expMator_real = ExpMatFloat32(N)
expMator_imag = ExpMatComplex64(N)

A_real = np.random.randn(N, N).astype(np.float32)
H_imag = np.random.randn(N, N).astype(np.float32)

exp_a = expMator_real.run(A_real)   # computes exp(A_real)
exp_ih = expMator_imag.run(H_imag)  # computes exp(iH_imag)

expMator_real.free()
expMator_imag.free()
```

Equivalent shorthand:

```python
expMator = ExpMatFloat32(N)   # or ExpMatFloat64 / ExpMatComplex64 / ExpMatComplex128
result = expMator.run(A)      # A must be an N x N matrix
expMator.free()
```

### SciPy equivalents

For the real-valued calculators:

```python
from scipy.linalg import expm

exp_h = expm(H_real)
```

For the optimized complex calculators, note the semantic difference carefully:

```python
from scipy.linalg import expm

# veloxpm ExpMatComplex64/128 expects a real matrix H
# and computes exp(iH) internally
exp_ih = expm(1j * H_imag)
```

## Testing

Run the packaged tests with:

```bash
cd ./test
python test_complex64.py
```

The main functional test file is:

- `test/test_float32.py`
- `test/test_float64.py`
- `test/test_complex64.py`
- `test/test_complex128py`

## Notes

- `ExpMatComplex64` and `ExpMatComplex128` expect a real matrix `H`, not a complex matrix.
- The complex calculators compute `exp(iH)` internally.

## Research Context

`veloxpm` is the core matrix-exponential computation component used in the `FastCTQW` simulator introduced in the following publication:

He, X., Ma, S., Qiang, X. (2026). *FastCTQW: A GPU-Accelerated Simulator for Ultra-large Scale Continuous-Time Quantum Walks.* In: Li, X., Wu, J., Zhang, J. (eds) *Quantum Computation.* CQCC 2025. Communications in Computer and Information Science, vol 2733. Springer, Singapore. https://doi.org/10.1007/978-981-95-4791-3_4

If you use this library in academic work, please cite the above reference.
