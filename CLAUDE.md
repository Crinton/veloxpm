# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
# Build wheel (output to dist/)
python setup.py bdist_wheel

# Install the built wheel
pip install dist/veloxpm-*.whl

# Alternative: direct install
python setup.py install

# CMake build alone (for compile-checking changes to src/)
mkdir -p build && cd build && cmake .. && make -j$(nproc)
```

## Test

Tests are **direct-run numerical scripts**, not pytest tests (despite `pytest.ini` existing).

```bash
python test/test_float32.py
python test/test_float64.py
python test/test_complex64.py
python test/test_complex128.py
```

Each script generates random matrices at sizes `[128, 256, 512, 1024, 2048, 4096]`, runs `veloxpm`, compares against `scipy.linalg.expm`, and reports runtime + numerical error. `test_complex64.py` and `test_complex128.py` compare against `expm(1j * H)`.

## Architecture

```
python/veloxpm/            Python package (public API)
  __init__.py              Exports 4 classes, version, author
  api.py                   _BaseCalculator wrappers + dtype-specific subclasses
  _validation.py           Input validation (real, square, shape check)

src/main.cu                pybind11 bindings → _veloxpm_core extension module
src/MatrixExpCalculator.h  Template class for exp(H): real float/double path
src/MatrixExpCalculator_imag.h  Template class for exp(iH): optimized complex path
src/matrix.h               CUDA kernels (gemm wrappers, fuse kernels, norm, solve,
                           combinePQ for real→complex assembly)
src/cuapi.h                cuBLAS/cuSolver trait specializations for all 4 dtypes
src/cusolver_utils.h       NVIDIA-derived type traits + error-checking macros
src/matrix.cu              Gemm/solve/combinePQ kernel implementations
src/cuapi.cu               cuBLAS/cuSolver API specializations implementations
CMakeLists.txt             Builds _veloxpm_core, links cuBLAS + cuSolver
setup.py                   setuptools + CMakeBuild extension builder
```

**Two calculation paths:**

| Python class | Backend template | Input | Computes |
|---|---|---|---|
| `ExpMatFloat32` | `MatrixExpCalculator<float>` | real float32 H | `exp(H)` |
| `ExpMatFloat64` | `MatrixExpCalculator<double>` | real float64 H | `exp(H)` |
| `ExpMatComplex64` | `MatrixExpCalculator_imag<float>` | real float32 H | `exp(iH)` |
| `ExpMatComplex128` | `MatrixExpCalculator_imag<double>` | real float64 H | `exp(iH)` |

**Algorithm:** Scaled Pade approximation. Norm-based scaling factor `s = floor(log2(nrmA / th13))`, then the appropriate Pade order (3/5/7/9/13) is selected via `digitize_cpp`. After solving `V \ U`, the result is squared back `s` times.

The imag path differs: it computes `H^2 = -H@H`, builds real U and V polynomials, then assembles complex P/Q matrices via `combinePQ` (U → imaginary part of P, V → real part of Q).

## Key constraints

- The complex calculators (`ExpMatComplex64`, `ExpMatComplex128`) accept only **real** input matrices — they compute `exp(iH)` internally. Do not change this semantic.
- The private extension module name is `_veloxpm_core`. Keep it unless explicitly asked to rename.
- The public package name is `veloxpm`.
- `python/veloxpm/api.py` reshapes the 1D output from the C++ backend to (n, n) — this reshape is required.
