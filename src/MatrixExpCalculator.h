#ifndef MATRIXEXPCALCULATOR_H
#define MATRIXEXPCALCULATOR_H
#include <cmath>       // 包含 expf, cosf, sinf 等数学函数
#include <vector>
#include "matrix.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename VT>
constexpr VT get_th13_val() {
    if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, cuComplex>) {
        return 5.371920351148152f;
    } else if constexpr (std::is_same_v<VT, double> || std::is_same_v<VT, cuDoubleComplex>) {
        return 3.925724783138660;
    }
}

// 2. 编译时获取 theta_vec 的值（返回 std::array）
template<typename VT>
constexpr auto get_theta_vec_array() {
    if constexpr (std::is_same_v<VT, double> || std::is_same_v<VT, cuDoubleComplex>) {
        return std::array<VT, 4>{1.495585217958292e-002f, 2.539398330063230e-001f, 9.504178996162932e-001f, 2.097847961257068e+000f};
    } else if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, cuComplex>) {
        return std::array<VT, 2>{4.258730016922831e-001, 1.880152677804762e+000};
    }
}




template <typename VT>
class MatrixExpCalculator {

// using RT = std::conditional_t<
//     std::is_same_v<VT, cuComplex>, float, // 如果 T 是 cuComplex，结果是 float
//     std::conditional_t<
//         std::is_same_v<VT, cuDoubleComplex>, double, // 否则，如果 T 是 cuDoubleComplex，结果是 double
//         VT // 否则（对于 float, double 或其他），结果是 T 本身
//     >
// >;
public:
    size_t n;
    cublasHandle_t cublasH;      // cuBLAS 句柄
    cusolverDnHandle_t cusolverH; // cuSolver 句柄
    cudaStream_t stream;         // CUDA 流
    VT* d_A;
    VT* d_A2;
    VT* d_A4;
    VT* d_A6;
    VT* d_u1;
    VT* d_u2;
    VT* d_v1;
    VT* d_v2;
    VT *d_nrmA;
    MatrixExpCalculator(size_t n); 

    ~MatrixExpCalculator();

    void _pade3();

    void _pade5();

    void _pade7();

    void _pade9();

    void _pade13();


    py::array_t<VT> run(py::array_t<VT>& arr_a);
    
    int32_t getN();

    void free();

};

// MatrixExpCalculator 构造函数的实现
template <typename VT>
MatrixExpCalculator<VT>::MatrixExpCalculator(size_t n) : n(n) {
    /*
    初始化Context资源
    */
    cudaStreamCreate(&stream);
    cublasCreate_v2(&cublasH);
    cusolverDnCreate(&cusolverH);
    cublasSetStream_v2(cublasH, stream);
    cusolverDnSetStream(cusolverH, stream);

    /*
    分配内存
    */

    // 数值算法的必要显存
    cudaMallocAsync(&d_A, n * n * sizeof(VT), stream);
    cudaMallocAsync(&d_A2, n * n * sizeof(VT), stream);
    cudaMallocAsync(&d_A4, n * n * sizeof(VT), stream);
    cudaMallocAsync(&d_A6, n * n * sizeof(VT), stream);
    cudaMallocAsync(&d_u1, n *n *sizeof(VT), stream);
    cudaMallocAsync(&d_u2, n * n *sizeof(VT), stream);
    cudaMallocAsync(&d_v1, n * n *sizeof(VT), stream);
    cudaMallocAsync(&d_v2, n * n *sizeof(VT), stream);

    // 小显存并初始化

    cudaMallocAsync(&d_nrmA, 1 * sizeof(VT), stream);
    cudaMemsetAsync(d_nrmA, 0, 1 * sizeof(VT), stream);
}

template <typename VT>
MatrixExpCalculator<VT>::~MatrixExpCalculator() {
    free();
}

template <typename VT>
int32_t MatrixExpCalculator<VT>::getN(){
    return this->n;
}

template <typename VT>
void MatrixExpCalculator<VT>::free() {
    /*
    销毁该对象
    */

    if (d_A) {
        CUDA_CHECK(cudaFree(d_A));
        d_A = nullptr;
    };
    if (d_A2) {
        CUDA_CHECK(cudaFree(d_A2));
        d_A2 = nullptr;
    };
    if (d_A4) {
        CUDA_CHECK(cudaFree(d_A4));
        d_A4 = nullptr;
    };
    if (d_A6) {
        CUDA_CHECK(cudaFree(d_A6));
        d_A6 = nullptr;
    };
    if (d_u1) {
        CUDA_CHECK(cudaFree(d_u1));
        d_u1 = nullptr;
    };
    if (d_u2) {
        CUDA_CHECK(cudaFree(d_u2));
        d_u2 = nullptr;
    };
    if (d_v1) {
        CUDA_CHECK(cudaFree(d_v1));
        d_v1 = nullptr;
    };
    if (d_v2) {
        CUDA_CHECK(cudaFree(d_v2));
        d_v2 = nullptr;
    };
    if (d_nrmA) {
        CUDA_CHECK(cudaFree(d_nrmA));
        d_nrmA = nullptr;
    };

    // Destroy cuBLAS handle
    if (cublasH) {
        cublasDestroy_v2(cublasH);
        cublasH = nullptr;
    }
    // Destroy cuSolver handle
    if (cusolverH) {
        cusolverDnDestroy(cusolverH);
        cusolverH = nullptr;
    }
    // Destroy CUDA stream
    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

template <typename VT>
void MatrixExpCalculator<VT>::_pade3() {
    /*
    直接对类的类型进行修改，不需要显式的输入和返回
    A2 = A@A
    U = A@(b3*A2 + b1), U 用d_u1表示
    V = b2*A2 + b0,  V 用d_v1表示
    */

    VT b[] = {120., 60., 12., 1.};

    gemm(cublasH, n, n, n, d_A, d_A, d_A2, 1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, 1.0f, 0.0f);

    _fuse3(d_A2, d_u2, d_v1, n, stream); //(b3*A2 + b1) 直接写到d_u2上, d_v1 = V = b2*A2 + b0

    gemm(cublasH, n, n, n, d_A, d_u2, d_u1, 1.0f, 0.0f); //d_u1 = U = d_A@d_u2

    //此时, d_u1为U, d_v1为V
    VT val_minus_1 = -1.0;
    VT val_plus_1 = 1.0;
    
    cudaMemcpyAsync(d_u2, d_u1, n * n * sizeof(VT), cudaMemcpyDeviceToDevice, stream); //d_u2 := d_u1
    cublasAPI<VT>::Axpy(cublasH, n*n, &val_plus_1, d_v1, 1 ,d_u1, 1); // d_u1 = U+V
    cublasAPI<VT>::Axpy(cublasH, n*n, &val_minus_1, d_u2, 1 ,d_v1, 1); // d_v1 = -U+V

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }
}



template <typename VT>
void MatrixExpCalculator<VT>::_pade5() {
    /*
    直接对类的类型进行修改，不需要显式的输入和返回
    A2 = A@A
    A4 = A2@A2
    U = A@(b5*A4 + b3*A2 + b1), U 用d_u1表示
    V = b4*A4 + b2*A2 + b0,  V 用d_v1表示
    */

    VT b[] = {30240., 15120., 3360., 420., 30., 1.};

    gemm(cublasH, n, n, n, d_A, d_A, d_A2, 1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, 1.0f, 0.0f);

    _fuse5(d_A2, d_A4, d_u2, d_v1, n, stream); //(b5*A4 + b3*A2 + b1) 直接写到d_u2上, d_v1 = V = b4*A4 + b2*A2 + b0

    gemm(cublasH, n, n, n, d_A, d_u2, d_u1, 1.0f, 0.0f); //d_u1 = U = d_A@d_u2

    //此时, d_u1为U, d_v1为V
    VT val_minus_1 = -1.0;
    VT val_plus_1 = 1.0;
    
    cudaMemcpyAsync(d_u2, d_u1, n * n * sizeof(VT), cudaMemcpyDeviceToDevice, stream); //d_u2 := d_u1
    cublasAPI<VT>::Axpy(cublasH, n*n, &val_plus_1, d_v1, 1 ,d_u1, 1); // d_u1 = U+V
    cublasAPI<VT>::Axpy(cublasH, n*n, &val_minus_1, d_u2, 1 ,d_v1, 1); // d_v1 = -U+V

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }
}

template <typename VT>
void MatrixExpCalculator<VT>::_pade7() {
    /*
    直接对类的类型进行修改，不需要显式的输入和返回
    A2 = A@A
    A4 = A2@A2
    A6 = A2@A4
    U = A@(b7*A6 + b5*A4 + b3*A2 + b1), U 用d_u1表示
    V = b6*A6 + b4*A4 + b2*A2 + b0,  V 用d_v1表示
    */

    VT b[] = {17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.};

    gemm(cublasH, n, n, n, d_A, d_A, d_A2, 1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, 1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A2, d_A4, d_A6, 1.0f, 0.0f);


    _fuse7(d_A2, d_A4, d_A6, d_u2, d_v1, n, stream); //(b7*A6 + b5*A4 + b3*A2 + b1) 直接写到d_u2上, d_v1 = V = b6*A6 + b4*A4 + b2*A2 + b0

    gemm(cublasH, n, n, n, d_A, d_u2, d_u1, 1.0f, 0.0f); //d_u1 = U = d_A@d_u2

    //此时, d_u1为U, d_v1为V
    VT val_minus_1 = -1.0;
    VT val_plus_1 = 1.0;
    
    cudaMemcpyAsync(d_u2, d_u1, n * n * sizeof(VT), cudaMemcpyDeviceToDevice, stream); //d_u2 := d_u1
    cublasAPI<VT>::Axpy(cublasH, n*n, &val_plus_1, d_v1, 1 ,d_u1, 1); // d_u1 = U+V
    cublasAPI<VT>::Axpy(cublasH, n*n, &val_minus_1, d_u2, 1 ,d_v1, 1); // d_v1 = -U+V

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }
}

template <typename VT>
void MatrixExpCalculator<VT>::_pade9() {
    /*
    直接对类的类型进行修改，不需要显式的输入和返回
    A2 = A@A
    A4 = A2@A2
    A6 = A2@A4
    A8 = A4@A4, 需要一个额外的A8矩阵，这里用d_u2替代
    U = A@(b9*A8 + b7*A6 + b5*A4 + b3 * A2 + b1), U 用d_u1表示
    V = b8*A8 + b6*A6 + b4*A4 + b2*A2 + b0,  V 用d_v1表示
    */

    VT b[] = {17643225600., 8821612800., 2075673600., 302702400., 30270240.,
       2162160., 110880., 3960., 90., 1.};
    gemm(cublasH, n, n, n, d_A, d_A, d_A2, 1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, 1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A2, d_A4, d_A6, 1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A4, d_A4, d_u2, 1.0f, 0.0f); //d_u2 等于 d_A8


    _fuse9(d_A2, d_A4, d_A6, d_u2, d_v1, n, stream); //(b9*A8 + b7*A6 + b5*A4 + b3 * A2 + b1) 直接写到d_u2上，也就是写回到d_A8上, d_v1 = V = b8*A8 + b6*A6 + b4*A4 + b2*A2 + b0

    gemm(cublasH, n, n, n, d_A, d_u2, d_u1, 1.0f, 0.0f); //d_u1 = U = d_A@d_u2

    //此时, d_u1为U, d_v1为V
    VT val_minus_1 = -1.0;
    VT val_plus_1 = 1.0;
    
    cudaMemcpyAsync(d_u2, d_u1, n * n * sizeof(VT), cudaMemcpyDeviceToDevice, stream); //d_u2 := d_u1
    cublasAPI<VT>::Axpy(cublasH, n*n, &val_plus_1, d_v1, 1 ,d_u1, 1); // d_u1 = U+V
    cublasAPI<VT>::Axpy(cublasH, n*n, &val_minus_1, d_u2, 1 ,d_v1, 1); // d_v1 = -U+V

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }
}

template <typename VT>
void MatrixExpCalculator<VT>::_pade13() {
    /*
    直接对类的类型进行修改，不需要显式的输入和返回
    */
    gemm(cublasH, n, n, n, d_A, d_A, d_A2, 1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, 1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A2, d_A4, d_A6, 1.0f, 0.0f);

    _fuse13(d_A2, d_A4, d_A6, d_u1, d_u2, d_v1, d_v2, n, stream);

    gemm(cublasH, n, n, n, d_A6, d_u1, d_u2, 1.0f, 1.0f); //d_u2 = A6 @ u1 + u2

    gemm(cublasH, n, n, n, d_A, d_u2, d_u1, 1.0f, 0.0f); //d_u1 = u = A @ d_u2
    
    gemm(cublasH, n, n, n, d_A6, d_v1, d_v2, 1.0f, 1.0f); //d_v2 = v = A6 @ d_v1 + d_v2

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }
    VT val_minus_1 = -1.0;
    VT val_plus_1 = 1.0;
    cudaMemcpyAsync(d_v1, d_u1, n * n * sizeof(VT), cudaMemcpyDeviceToDevice, stream); // d_v1 = d_u1
    cublasAPI<VT>::Axpy(cublasH, n * n, &val_minus_1, d_v2, 1, d_v1, 1); // d_v1 = -u + v = -d_v2 + d_v1
    cublasAPI<VT>::Scal(cublasH, n * n, &val_minus_1, d_v1, 1);
    cublasAPI<VT>::Axpy(cublasH, n * n, &val_plus_1, d_v2, 1, d_u1, 1); // d_u1 = u + v = d_u1 + d_v2
}


// MatrixExpCalculator::run 方法的实现
template <typename VT>
py::array_t<VT> MatrixExpCalculator<VT>::run(py::array_t<VT>& arr_a) {
    /*
    每次run从numpy接受一个数组指针，
    下面修改为 pade3 pade5 pade7 pade9 pade11 pade13
    */
    constexpr VT th13 = get_th13_val<VT>();
    constexpr auto theta_vec = get_theta_vec_array<VT>(); // auto 推断出 std::array 类型和大小
    constexpr int size = theta_vec.size(); // std::array 的 size() 也是 constexpr

    // cudaEvent_t start, end;
    // cudaEventCreate(&start);
    // cudaEventCreate(&end);
    // cudaEventRecord(start);
    py::buffer_info bufA = arr_a.request();
    auto shape = bufA.shape;
    if (shape[0] != n || shape[1] != n)
    {
        throw std::runtime_error("matrix size is error");
    }
    VT *A = (VT *)bufA.ptr; // 获取 NumPy 数组的原始指针

    VT nrmA;

    cudaMemcpyAsync(d_A, A, n * n *sizeof(VT), cudaMemcpyHostToDevice,stream);
    
    cudaMemsetAsync(d_nrmA, 0, 1 * sizeof(VT), stream); 

    RowMaxAbsSum(d_A, n, d_nrmA, stream);


    cudaMemcpyAsync(&nrmA, d_nrmA, 1 * sizeof(VT), cudaMemcpyDeviceToHost);
    int s;
    VT s_pow;
    cudaStreamSynchronize(stream);


    // if (nrmA > th13) {
    //     s = static_cast<int>(std::ceil(std::log2(nrmA / th13)));
    //     s_pow = 1 / (static_cast<VT>(std::pow(2,s)));
    // } else {
    //     s = 1;
    //     s_pow = 0.5f;
    // }

    // 计算s缩放系数
    s = std::max(static_cast<int>(std::floor(std::log2(nrmA / th13))), 0);
    s_pow = 1 / (static_cast<VT>(std::pow(2,s)));
    cublasAPI<VT>::Scal(cublasH, n*n, &s_pow, d_A, 1); // A = A/2**s

    // 计算m系数, float只会取[0,1,2], double取[0,1,2,3,4]
    const int idx = digitize_cpp(nrmA, theta_vec); // idx=0, m=3; idx=1, m=5. idx2,m=7; idx=3,m=9; idx=4,m=13;
    switch (idx)
    {
    case 0:
        /* code */
        _pade3();
        break;
    case 1:
        _pade5();
        break;
    case 2:
        _pade7();
        break;
    case 3:
        _pade9();
        break;
    case 4:
        _pade13();
        break;
    }

    // gemm(cublasH, n, n, n, d_A, d_A, d_A2, 1.0f, 0.0f);

    // gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, 1.0f, 0.0f);

    // gemm(cublasH, n, n, n, d_A2, d_A4, d_A6, 1.0f, 0.0f);

    // fuse(d_A2, d_A4, d_A6, d_u1, d_u2, d_v1, d_v2, n, stream);

    // gemm(cublasH, n, n, n, d_A6, d_u1, d_u2, 1.0f, 1.0f); //d_u2 = A6 @ u1 + u2

    // gemm(cublasH, n, n, n, d_A, d_u2, d_u1, 1.0f, 0.0f); //d_u1 = u = A @ d_u2
    
    // gemm(cublasH, n, n, n, d_A6, d_v1, d_v2, 1.0f, 1.0f); //d_v2 = v = A6 @ d_v1 + d_v2

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
    // }
    // VT val_minus_1 = -1.0;
    // VT val_plus_1 = 1.0;
    // cudaMemcpyAsync(d_v1, d_u1, n * n * sizeof(VT), cudaMemcpyDeviceToDevice, stream); // d_v1 = d_u1
    // cublasAPI<VT>::Axpy(cublasH, n * n, &val_minus_1, d_v2, 1, d_v1, 1); // d_v1 = -u + v = -d_v2 + d_v2
    // cublasAPI<VT>::Scal(cublasH, n * n, &val_minus_1, d_v1, 1);
    // cublasAPI<VT>::Axpy(cublasH, n * n, &val_plus_1, d_v2, 1, d_u1, 1); // d_u1 = u + v = d_u1 + d_v2

    solve(cusolverH, d_v1, d_u1, n);
    for (int i = 0; i < s; ++i) {
        gemm(cublasH, n, n, n, d_u1, d_u1, d_v1, 1.0, 0.0);
        cudaMemcpyAsync(d_u1, d_v1, n * n * sizeof(VT), cudaMemcpyDeviceToDevice,stream);
    }
    // compute [13/13] Pade approximant
    VT *eA_ptr = (VT *)malloc(n * n * sizeof(VT));
    
    cudaMemcpy(eA_ptr, d_u1, n * n * sizeof(VT), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }
    std::vector<size_t> eA_shape{n*n};
    py::array_t<VT> eA(eA_shape, eA_ptr);

    // float ms;
    // cudaEventRecord(end);
    // cudaEventSynchronize(end);
    // cudaEventElapsedTime(&ms, start, end);
    // printf("GEMM Kernel Time: %.4f ms\n", ms);

    return eA;

}

float get_current_gpu_memory_gb() {
    size_t freeMemBytes, totalMemBytes;
    cudaError_t cudaStatus = cudaMemGetInfo(&freeMemBytes, &totalMemBytes);

    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Error: cudaMemGetInfo failed! " + std::string(cudaGetErrorString(cudaStatus)));
    }

    // 将字节转换为 GB
    const float GB = 1024.0f * 1024.0f * 1024.0f;
    float totalMemGB = static_cast<float>(totalMemBytes) / GB;
    float freeMemGB = static_cast<float>(freeMemBytes) / GB;

    return freeMemGB;
}

template <>
class MatrixExpCalculator<cuComplex> {
public:
    size_t n;
    cublasHandle_t cublasH;      // cuBLAS 句柄
    cusolverDnHandle_t cusolverH; // cuSolver 句柄
    cudaStream_t stream;         // CUDA 流
    cuComplex *d_A;
    cuComplex *d_A2;
    cuComplex *d_A4;
    cuComplex *d_A6;
    cuComplex *d_u1;
    cuComplex *d_u2;
    cuComplex *d_v1;
    cuComplex *d_v2;
    float *d_nrmA;
    MatrixExpCalculator(size_t n) : n(n) {
        /*
        初始化Context资源
        */
        cudaStreamCreate(&stream);
        cublasCreate_v2(&cublasH);
        cusolverDnCreate(&cusolverH);
        cublasSetStream_v2(cublasH, stream);
        cusolverDnSetStream(cusolverH, stream);
        /*
        分配内存
        */
        // 数值算法的必要显存
        CUDA_CHECK(cudaMallocAsync(&d_A, n * n * sizeof(cuComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_A2, n * n * sizeof(cuComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_A4, n * n * sizeof(cuComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_A6, n * n * sizeof(cuComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_u1, n *n *sizeof(cuComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_u2, n * n *sizeof(cuComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_v1, n * n *sizeof(cuComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_v2, n * n *sizeof(cuComplex), stream));

        // 小显存并初始化
        CUDA_CHECK(cudaMallocAsync(&d_nrmA, 1 * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_nrmA, 0, 1 * sizeof(float), stream));
    }

    ~MatrixExpCalculator() {
        free();
    }

    void _pade3() {
    /*
    直接对类的类型进行修改，不需要显式的输入和返回
    A2 = A@A
    U = A@(b3*A2 + b1), U 用d_u1表示
    V = b2*A2 + b0,  V 用d_v1表示
    */

    cuComplex alpha_one = make_cuComplex(1.0,0.0);
    cuComplex alpha_zero = make_cuComplex(0.0,0.0);

    gemm(cublasH, n, n, n, d_A, d_A, d_A2, alpha_one, alpha_zero);

    gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, alpha_one, alpha_zero);

    _fuse3(d_A2, d_u2, d_v1, n, stream); //(b3*A2 + b1) 直接写到d_u2上, d_v1 = V = b2*A2 + b0

    gemm(cublasH, n, n, n, d_A, d_u2, d_u1, alpha_one, alpha_zero); //d_u1 = U = d_A@d_u2

    //此时, d_u1为U, d_v1为V
    cuComplex val_minus_1 = make_cuComplex(-1.0, 0.0);
    cuComplex val_plus_1 = make_cuComplex(1.0, 0.0);
    
    cudaMemcpyAsync(d_u2, d_u1, n * n * sizeof(cuComplex), cudaMemcpyDeviceToDevice, stream); //d_u2 := d_u1
    cublasAPI<cuComplex>::Axpy(cublasH, n*n, &val_plus_1, d_v1, 1 ,d_u1, 1); // d_u1 = U+V
    cublasAPI<cuComplex>::Axpy(cublasH, n*n, &val_minus_1, d_u2, 1 ,d_v1, 1); // d_v1 = -U+V

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }
}

    void _pade5() {
        /*
        直接对类的类型进行修改，不需要显式的输入和返回
        A2 = A@A
        A4 = A2@A2
        U = A@(b5*A4 + b3*A2 + b1), U 用d_u1表示
        V = b4*A4 + b2*A2 + b0,  V 用d_v1表示
        */
        cuComplex alpha_one = make_cuComplex(1.0,0.0);
        cuComplex alpha_zero = make_cuComplex(0.0,0.0);

        gemm(cublasH, n, n, n, d_A, d_A, d_A2, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, alpha_one, alpha_zero);

        _fuse5(d_A2, d_A4, d_u2, d_v1, n, stream); //(b5*A4 + b3*A2 + b1) 直接写到d_u2上, d_v1 = V = b4*A4 + b2*A2 + b0

        gemm(cublasH, n, n, n, d_A, d_u2, d_u1, alpha_one, alpha_zero); //d_u1 = U = d_A@d_u2

        //此时, d_u1为U, d_v1为V
        cuComplex val_minus_1 = make_cuComplex(-1.0, 0.0);
        cuComplex val_plus_1 = make_cuComplex(1.0, 0.0);
        
        cudaMemcpyAsync(d_u2, d_u1, n * n * sizeof(cuComplex), cudaMemcpyDeviceToDevice, stream); //d_u2 := d_u1
        cublasAPI<cuComplex>::Axpy(cublasH, n*n, &val_plus_1, d_v1, 1 ,d_u1, 1); // d_u1 = U+V
        cublasAPI<cuComplex>::Axpy(cublasH, n*n, &val_minus_1, d_u2, 1 ,d_v1, 1); // d_v1 = -U+V

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
        }
    }

    void _pade7() {
        /*
        直接对类的类型进行修改，不需要显式的输入和返回
        A2 = A@A
        A4 = A2@A2
        A6 = A2@A4
        U = A@(b7*A6 + b5*A4 + b3*A2 + b1), U 用d_u1表示
        V = b6*A6 + b4*A4 + b2*A2 + b0,  V 用d_v1表示
        */


        cuComplex alpha_one = make_cuComplex(1.0,0.0);
        cuComplex alpha_zero = make_cuComplex(0.0,0.0);

        gemm(cublasH, n, n, n, d_A, d_A, d_A2, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A4, d_A6, alpha_one, alpha_zero);


        _fuse7(d_A2, d_A4, d_A6, d_u2, d_v1, n, stream); //(b7*A6 + b5*A4 + b3*A2 + b1) 直接写到d_u2上, d_v1 = V = b6*A6 + b4*A4 + b2*A2 + b0

        gemm(cublasH, n, n, n, d_A, d_u2, d_u1, alpha_one, alpha_zero); //d_u1 = U = d_A@d_u2

        //此时, d_u1为U, d_v1为V
        cuComplex val_minus_1 = make_cuComplex(-1.0, 0.0);
        cuComplex val_plus_1 = make_cuComplex(1.0, 0.0);
        
        cudaMemcpyAsync(d_u2, d_u1, n * n * sizeof(cuComplex), cudaMemcpyDeviceToDevice, stream); //d_u2 := d_u1
        cublasAPI<cuComplex>::Axpy(cublasH, n*n, &val_plus_1, d_v1, 1 ,d_u1, 1); // d_u1 = U+V
        cublasAPI<cuComplex>::Axpy(cublasH, n*n, &val_minus_1, d_u2, 1 ,d_v1, 1); // d_v1 = -U+V

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
        }
    }

    void _pade9() {
        /*
        直接对类的类型进行修改，不需要显式的输入和返回
        A2 = A@A
        A4 = A2@A2
        A6 = A2@A4
        A8 = A4@A4, 需要一个额外的A8矩阵，这里用d_u2替代
        U = A@(b9*A8 + b7*A6 + b5*A4 + b3 * A2 + b1), U 用d_u1表示
        V = b8*A8 + b6*A6 + b4*A4 + b2*A2 + b0,  V 用d_v1表示
        */
        cuComplex alpha_one = make_cuComplex(1.0,0.0);
        cuComplex alpha_zero = make_cuComplex(0.0,0.0);

        gemm(cublasH, n, n, n, d_A, d_A, d_A2, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A4, d_A6, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A4, d_A4, d_u2, alpha_one, alpha_zero); //d_u2 等于 d_A8


        _fuse9(d_A2, d_A4, d_A6, d_u2, d_v1, n, stream); //(b9*A8 + b7*A6 + b5*A4 + b3 * A2 + b1) 直接写到d_u2上，也就是写回到d_A8上, d_v1 = V = b8*A8 + b6*A6 + b4*A4 + b2*A2 + b0

        gemm(cublasH, n, n, n, d_A, d_u2, d_u1, alpha_one, alpha_zero); //d_u1 = U = d_A@d_u2

        //此时, d_u1为U, d_v1为V
        cuComplex val_minus_1 = make_cuComplex(-1.0, 0.0);
        cuComplex val_plus_1 = make_cuComplex(1.0, 0.0);
        
        cudaMemcpyAsync(d_u2, d_u1, n * n * sizeof(cuComplex), cudaMemcpyDeviceToDevice, stream); //d_u2 := d_u1
        cublasAPI<cuComplex>::Axpy(cublasH, n*n, &val_plus_1, d_v1, 1 ,d_u1, 1); // d_u1 = U+V
        cublasAPI<cuComplex>::Axpy(cublasH, n*n, &val_minus_1, d_u2, 1 ,d_v1, 1); // d_v1 = -U+V

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
        }
    }

    void _pade13() {
        /*
        直接对类的类型进行修改，不需要显式的输入和返回
        */
        cuComplex alpha_one = make_cuComplex(1.0,0.0);
        cuComplex alpha_zero = make_cuComplex(0.0,0.0);
        gemm(cublasH, n, n, n, d_A, d_A, d_A2, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A4, d_A6, alpha_one, alpha_zero);

        _fuse13(d_A2, d_A4, d_A6, d_u1, d_u2, d_v1, d_v2, n, stream);

        gemm(cublasH, n, n, n, d_A6, d_u1, d_u2, alpha_one, alpha_one); //d_u2 = A6 @ u1 + u2

        gemm(cublasH, n, n, n, d_A, d_u2, d_u1, alpha_one, alpha_zero); //d_u1 = u = A @ d_u2
        
        gemm(cublasH, n, n, n, d_A6, d_v1, d_v2, alpha_one, alpha_one); //d_v2 = v = A6 @ d_v1 + d_v2

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
        }
        cuComplex val_minus_1 = make_cuComplex(-1.0, 0.0);
        cuComplex val_plus_1 = make_cuComplex(1.0, 0.0);
        cudaMemcpyAsync(d_v1, d_u1, n * n * sizeof(cuComplex), cudaMemcpyDeviceToDevice, stream); // d_v1 = d_u1
        cublasAPI<cuComplex>::Axpy(cublasH, n * n, &val_minus_1, d_v2, 1, d_v1, 1); // d_v1 = -u + v = -d_v2 + d_v2
        cublasAPI<cuComplex>::Scal(cublasH, n * n, &val_minus_1, d_v1, 1);
        cublasAPI<cuComplex>::Axpy(cublasH, n * n, &val_plus_1, d_v2, 1, d_u1, 1); // d_u1 = u + v = d_u1 + d_v2
    }




    py::array_t<std::complex<float>> run(py::array_t<std::complex<float>>& arr_a) {
        

        constexpr float th13 = get_th13_val<float>();
        constexpr auto theta_vec = get_theta_vec_array<float>(); // auto 推断出 std::array 类型和大小

        py::buffer_info bufA = arr_a.request();
        auto shape = bufA.shape;
        if (shape[0] != n | shape[1] != n)
        {
            throw std::runtime_error("matrix size is error");
        }
        if (shape[0] != shape[1])
        {
            throw std::runtime_error("Error: The matrix is not square (rows != cols).");
        }
        std::complex<float> *A = (std::complex<float> *)bufA.ptr; // 获取 NumPy 数组的原始指针

        float nrmA;        

        cudaMemcpyAsync(d_A, A, n * n *sizeof(cuComplex), cudaMemcpyHostToDevice,stream);
        

        cudaMemsetAsync(d_nrmA, 0, 1 * sizeof(float), stream); 
        RowMaxAbsSum(d_A, n, d_nrmA, stream);
        cudaMemcpyAsync(&nrmA, d_nrmA, 1 * sizeof(float), cudaMemcpyDeviceToHost);

        int s;
        cuComplex s_pow;
        // 计算s缩放系数
        s = std::max(static_cast<int>(std::floor(std::log2(nrmA / th13))), 0);
        s_pow = make_cuComplex(1 / (static_cast<float>(std::pow(2,s))), 0.0f);
        cublasAPI<cuComplex>::Scal(cublasH, n*n, &s_pow, d_A, 1);
        
        // cudaStreamSynchronize(stream);

        cuComplex alpha_one = make_cuComplex(1.0,0.0);
        cuComplex alpha_zero = make_cuComplex(0.0,0.0);

        // 计算m系数, float/cuComplex只会取[0,1,2], double/cuDoubleComplex取[0,1,2,3,4]
        const int idx = digitize_cpp(nrmA, theta_vec); // idx=0, m=3; idx=1, m=5. idx2,m=7; idx=3,m=9; idx=4,m=13;
        switch (idx)
        {
        case 0:
            /* code */
            _pade3();
            break;
        case 1:
            _pade5();
            break;
        case 2:
            _pade7();
            break;
        case 3:
            _pade9();
            break;
        case 4:
            _pade13();
            break;
        }

        cudaStreamSynchronize(stream);

        solve(cusolverH, d_v1, d_u1, n);
        for (int i = 0; i < s; ++i) {
            CUBLAS_CHECK(gemm(cublasH, n, n, n, d_u1, d_u1, d_v1,alpha_one, alpha_zero));
            CUDA_CHECK(cudaMemcpyAsync(d_u1, d_v1, n * n * sizeof(cuComplex), cudaMemcpyDeviceToDevice,stream));
        }
        // compute [13/13] Pade approximant

        std::complex<float> *eA_ptr = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
        
        cudaMemcpyAsync(eA_ptr, d_u1, n * n * sizeof(cuComplex), cudaMemcpyDeviceToHost, stream); // cuComplex -> std::complex<float>
        cudaStreamSynchronize(stream);

        std::vector<size_t> eA_shape{n*n};
        py::array_t<std::complex<float>> eA(eA_shape, eA_ptr);

        // float ms;
        // cudaEventRecord(end);
        // cudaEventSynchronize(end);
        // cudaEventElapsedTime(&ms, start, end);
        // printf("GEMM Kernel Time: %.4f ms\n", ms);

        return eA;
    }
    
    int32_t getN() {
        return this->n;
    }

    void free() {
        /*
        销毁该对象
        */
        if (stream) {
            // 同步 CUDA 流，确保所有待处理的操作（包括之前的异步释放请求）
            // 都已完成。这使得该对象流上的内存释放变为同步操作。
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        if (d_A) {
            CUDA_CHECK(cudaFree(d_A));
            d_A = nullptr;
        };
        if (d_A2) {
            CUDA_CHECK(cudaFree(d_A2));
            d_A2 = nullptr;
        };
        if (d_A4) {
            CUDA_CHECK(cudaFree(d_A4));
            d_A4 = nullptr;
        };
        if (d_A6) {
            CUDA_CHECK(cudaFree(d_A6));
            d_A6 = nullptr;
        };
        if (d_u1) {
            CUDA_CHECK(cudaFree(d_u1));
            d_u1 = nullptr;
        };
        if (d_u2) {
            CUDA_CHECK(cudaFree(d_u2));
            d_u2 = nullptr;
        };
        if (d_v1) {
            CUDA_CHECK(cudaFree(d_v1));
            d_v1 = nullptr;
        };
        if (d_v2) {
            CUDA_CHECK(cudaFree(d_v2));
            d_v2 = nullptr;
        };
        if (d_nrmA) {
            CUDA_CHECK(cudaFree(d_nrmA));
            d_nrmA = nullptr;
        };

        // Destroy cuBLAS handle
        if (cublasH) {
            cublasDestroy_v2(cublasH);
            cublasH = nullptr;
        }
        // Destroy cuSolver handle
        if (cusolverH) {
            cusolverDnDestroy(cusolverH);
            cusolverH = nullptr;
        }
        // Destroy CUDA stream
        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }

};

template <>
class MatrixExpCalculator<cuDoubleComplex> {
public:
    size_t n;
    cublasHandle_t cublasH;      // cuBLAS 句柄
    cusolverDnHandle_t cusolverH; // cuSolver 句柄
    cudaStream_t stream;         // CUDA 流
    cuDoubleComplex *d_A;
    cuDoubleComplex *d_A2;
    cuDoubleComplex *d_A4;
    cuDoubleComplex *d_A6;
    cuDoubleComplex *d_u1;
    cuDoubleComplex *d_u2;
    cuDoubleComplex *d_v1;
    cuDoubleComplex *d_v2;
    double *d_nrmA;
    MatrixExpCalculator(size_t n) : n(n) {
        /*
        初始化Context资源
        */
        cudaStreamCreate(&stream);
        cublasCreate_v2(&cublasH);
        cusolverDnCreate(&cusolverH);
        cublasSetStream_v2(cublasH, stream);
        cusolverDnSetStream(cusolverH, stream);
        /*
        分配内存
        */

        // 数值算法的必要显存
        CUDA_CHECK(cudaMallocAsync(&d_A, n * n * sizeof(cuDoubleComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_A2, n * n * sizeof(cuDoubleComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_A4, n * n * sizeof(cuDoubleComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_A6, n * n * sizeof(cuDoubleComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_u1, n *n *sizeof(cuDoubleComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_u2, n * n *sizeof(cuDoubleComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_v1, n * n *sizeof(cuDoubleComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_v2, n * n *sizeof(cuDoubleComplex), stream));

        // 小显存并初始化

        CUDA_CHECK(cudaMallocAsync(&d_nrmA, 1 * sizeof(double), stream));
        CUDA_CHECK(cudaMemsetAsync(d_nrmA, 0, 1 * sizeof(double), stream));

    }

    ~MatrixExpCalculator() {
        free();
    }


    void _pade3() {
        /*
        直接对类的类型进行修改，不需要显式的输入和返回
        A2 = A@A
        U = A@(b3*A2 + b1), U 用d_u1表示
        V = b2*A2 + b0,  V 用d_v1表示
        */

        cuDoubleComplex alpha_one = make_cuDoubleComplex(1.0,0.0);
        cuDoubleComplex alpha_zero = make_cuDoubleComplex(0.0,0.0);

        gemm(cublasH, n, n, n, d_A, d_A, d_A2, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, alpha_one, alpha_zero);

        _fuse3(d_A2, d_u2, d_v1, n, stream); //(b3*A2 + b1) 直接写到d_u2上, d_v1 = V = b2*A2 + b0

        gemm(cublasH, n, n, n, d_A, d_u2, d_u1, alpha_one, alpha_zero); //d_u1 = U = d_A@d_u2

        //此时, d_u1为U, d_v1为V
        cuDoubleComplex val_minus_1 = make_cuDoubleComplex(-1.0, 0.0);
        cuDoubleComplex val_plus_1 = make_cuDoubleComplex(1.0, 0.0);
        
        cudaMemcpyAsync(d_u2, d_u1, n * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, stream); //d_u2 := d_u1
        cublasAPI<cuDoubleComplex>::Axpy(cublasH, n*n, &val_plus_1, d_v1, 1 ,d_u1, 1); // d_u1 = U+V
        cublasAPI<cuDoubleComplex>::Axpy(cublasH, n*n, &val_minus_1, d_u2, 1 ,d_v1, 1); // d_v1 = -U+V

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
        }
    }



    void _pade5() {
        /*
        直接对类的类型进行修改，不需要显式的输入和返回
        A2 = A@A
        A4 = A2@A2
        U = A@(b5*A4 + b3*A2 + b1), U 用d_u1表示
        V = b4*A4 + b2*A2 + b0,  V 用d_v1表示
        */
        cuDoubleComplex alpha_one = make_cuDoubleComplex(1.0,0.0);
        cuDoubleComplex alpha_zero = make_cuDoubleComplex(0.0,0.0);

        gemm(cublasH, n, n, n, d_A, d_A, d_A2, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, alpha_one, alpha_zero);

        _fuse5(d_A2, d_A4, d_u2, d_v1, n, stream); //(b5*A4 + b3*A2 + b1) 直接写到d_u2上, d_v1 = V = b4*A4 + b2*A2 + b0

        gemm(cublasH, n, n, n, d_A, d_u2, d_u1, alpha_one, alpha_zero); //d_u1 = U = d_A@d_u2

        //此时, d_u1为U, d_v1为V
        cuDoubleComplex val_minus_1 = make_cuDoubleComplex(-1.0, 0.0);
        cuDoubleComplex val_plus_1 = make_cuDoubleComplex(1.0, 0.0);
        
        cudaMemcpyAsync(d_u2, d_u1, n * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, stream); //d_u2 := d_u1
        cublasAPI<cuDoubleComplex>::Axpy(cublasH, n*n, &val_plus_1, d_v1, 1 ,d_u1, 1); // d_u1 = U+V
        cublasAPI<cuDoubleComplex>::Axpy(cublasH, n*n, &val_minus_1, d_u2, 1 ,d_v1, 1); // d_v1 = -U+V

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
        }
    }

    void _pade7() {
        /*
        直接对类的类型进行修改，不需要显式的输入和返回
        A2 = A@A
        A4 = A2@A2
        A6 = A2@A4
        U = A@(b7*A6 + b5*A4 + b3*A2 + b1), U 用d_u1表示
        V = b6*A6 + b4*A4 + b2*A2 + b0,  V 用d_v1表示
        */


        cuDoubleComplex alpha_one = make_cuDoubleComplex(1.0,0.0);
        cuDoubleComplex alpha_zero = make_cuDoubleComplex(0.0,0.0);

        gemm(cublasH, n, n, n, d_A, d_A, d_A2, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A4, d_A6, alpha_one, alpha_zero);


        _fuse7(d_A2, d_A4, d_A6, d_u2, d_v1, n, stream); //(b7*A6 + b5*A4 + b3*A2 + b1) 直接写到d_u2上, d_v1 = V = b6*A6 + b4*A4 + b2*A2 + b0

        gemm(cublasH, n, n, n, d_A, d_u2, d_u1, alpha_one, alpha_zero); //d_u1 = U = d_A@d_u2

        //此时, d_u1为U, d_v1为V
        cuDoubleComplex val_minus_1 = make_cuDoubleComplex(-1.0, 0.0);
        cuDoubleComplex val_plus_1 = make_cuDoubleComplex(1.0, 0.0);
        
        cudaMemcpyAsync(d_u2, d_u1, n * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, stream); //d_u2 := d_u1
        cublasAPI<cuDoubleComplex>::Axpy(cublasH, n*n, &val_plus_1, d_v1, 1 ,d_u1, 1); // d_u1 = U+V
        cublasAPI<cuDoubleComplex>::Axpy(cublasH, n*n, &val_minus_1, d_u2, 1 ,d_v1, 1); // d_v1 = -U+V

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
        }
    }

    void _pade9() {
        /*
        直接对类的类型进行修改，不需要显式的输入和返回
        A2 = A@A
        A4 = A2@A2
        A6 = A2@A4
        A8 = A4@A4, 需要一个额外的A8矩阵，这里用d_u2替代
        U = A@(b9*A8 + b7*A6 + b5*A4 + b3 * A2 + b1), U 用d_u1表示
        V = b8*A8 + b6*A6 + b4*A4 + b2*A2 + b0,  V 用d_v1表示
        */
        cuDoubleComplex alpha_one = make_cuDoubleComplex(1.0,0.0);
        cuDoubleComplex alpha_zero = make_cuDoubleComplex(0.0,0.0);

        gemm(cublasH, n, n, n, d_A, d_A, d_A2, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A4, d_A6, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A4, d_A4, d_u2, alpha_one, alpha_zero); //d_u2 等于 d_A8


        _fuse9(d_A2, d_A4, d_A6, d_u2, d_v1, n, stream); //(b9*A8 + b7*A6 + b5*A4 + b3 * A2 + b1) 直接写到d_u2上，也就是写回到d_A8上, d_v1 = V = b8*A8 + b6*A6 + b4*A4 + b2*A2 + b0

        gemm(cublasH, n, n, n, d_A, d_u2, d_u1, alpha_one, alpha_zero); //d_u1 = U = d_A@d_u2

        //此时, d_u1为U, d_v1为V
        cuDoubleComplex val_minus_1 = make_cuDoubleComplex(-1.0, 0.0);
        cuDoubleComplex val_plus_1 = make_cuDoubleComplex(1.0, 0.0);
        
        cudaMemcpyAsync(d_u2, d_u1, n * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, stream); //d_u2 := d_u1
        cublasAPI<cuDoubleComplex>::Axpy(cublasH, n*n, &val_plus_1, d_v1, 1 ,d_u1, 1); // d_u1 = U+V
        cublasAPI<cuDoubleComplex>::Axpy(cublasH, n*n, &val_minus_1, d_u2, 1 ,d_v1, 1); // d_v1 = -U+V

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
        }
    }

    void _pade13() {
        /*
        直接对类的类型进行修改，不需要显式的输入和返回
        */
        cuDoubleComplex alpha_one = make_cuDoubleComplex(1.0,0.0);
        cuDoubleComplex alpha_zero = make_cuDoubleComplex(0.0,0.0);
        gemm(cublasH, n, n, n, d_A, d_A, d_A2, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A4, d_A6, alpha_one, alpha_zero);

        _fuse13(d_A2, d_A4, d_A6, d_u1, d_u2, d_v1, d_v2, n, stream);

        gemm(cublasH, n, n, n, d_A6, d_u1, d_u2, alpha_one, alpha_one); //d_u2 = A6 @ u1 + u2

        gemm(cublasH, n, n, n, d_A, d_u2, d_u1, alpha_one, alpha_zero); //d_u1 = u = A @ d_u2
        
        gemm(cublasH, n, n, n, d_A6, d_v1, d_v2, alpha_one, alpha_one); //d_v2 = v = A6 @ d_v1 + d_v2

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
        }
        cuDoubleComplex val_minus_1 = make_cuDoubleComplex(-1.0, 0.0);
        cuDoubleComplex val_plus_1 = make_cuDoubleComplex(1.0, 0.0);
        cudaMemcpyAsync(d_v1, d_u1, n * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, stream); // d_v1 = d_u1
        cublasAPI<cuDoubleComplex>::Axpy(cublasH, n * n, &val_minus_1, d_v2, 1, d_v1, 1); // d_v1 = -u + v = -d_v2 + d_v2
        cublasAPI<cuDoubleComplex>::Scal(cublasH, n * n, &val_minus_1, d_v1, 1);
        cublasAPI<cuDoubleComplex>::Axpy(cublasH, n * n, &val_plus_1, d_v2, 1, d_u1, 1); // d_u1 = u + v = d_u1 + d_v2
    }

    py::array_t<std::complex<double>> run(py::array_t<std::complex<double>>& arr_a) {
        
        constexpr float th13 = get_th13_val<double>();
        constexpr auto theta_vec = get_theta_vec_array<double>(); // auto 推断出 

        py::buffer_info bufA = arr_a.request();
        auto shape = bufA.shape;
        if (shape[0] != n | shape[1] != n)
        {
            throw std::runtime_error("matrix size is error");
        }
        if (shape[0] != shape[1])
        {
            throw std::runtime_error("Error: The matrix is not square (rows != cols).");
        }
        std::complex<double> *A = (std::complex<double> *)bufA.ptr; // 获取 NumPy 数组的原始指针

        double nrmA;        
        
        cudaMemcpyAsync(d_A, A, n * n *sizeof(cuDoubleComplex), cudaMemcpyHostToDevice,stream);

        cudaMemsetAsync(d_nrmA, 0, 1 * sizeof(double), stream); 

        RowMaxAbsSum(d_A, n, d_nrmA, stream);

        cudaMemcpyAsync(&nrmA, d_nrmA, 1 * sizeof(double), cudaMemcpyDeviceToHost);
        int s;
        cuDoubleComplex s_pow;
        cudaStreamSynchronize(stream);
        // if (nrmA > th13) {
        //     s = static_cast<int>(std::ceil(std::log2(nrmA / th13))) + 1;
        //     s_pow = make_cuDoubleComplex(1/std::pow(2,s),0.0);
        // } else {
        //     s = 1;
        //     s_pow.x = 0.5;
        //     s_pow.y = 0.0;
        // }
        s = std::max(static_cast<int>(std::floor(std::log2(nrmA / th13))), 0);
        s_pow = make_cuDoubleComplex(1 / (static_cast<double>(std::pow(2,s))), 0.0);
        cublasAPI<cuDoubleComplex>::Scal(cublasH, n*n, &s_pow, d_A, 1);
        // cudaStreamSynchronize(stream);

        cuDoubleComplex alpha_one = make_cuDoubleComplex(1.0,0.0);
        cuDoubleComplex alpha_zero = make_cuDoubleComplex(0.0,0.0);

        // 计算m系数, float/cuComplex只会取[0,1,2], double/cuDoubleComplex取[0,1,2,3,4]
        const int idx = digitize_cpp(nrmA, theta_vec); // idx=0, m=3; idx=1, m=5. idx2,m=7; idx=3,m=9; idx=4,m=13;
        switch (idx)
        {
        case 0:
            /* code */
            _pade3();
            break;
        case 1:
            _pade5();
            break;
        case 2:
            _pade7();
            break;
        case 3:
            _pade9();
            break;
        case 4:
            _pade13();
            break;
        }
        
        cudaStreamSynchronize(stream);

        solve(cusolverH, d_v1, d_u1, n);
        for (int i = 0; i < s; ++i) {
            gemm(cublasH, n, n, n, d_u1, d_u1, d_v1,alpha_one, alpha_zero);
            cudaMemcpyAsync(d_u1, d_v1, n * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice,stream);
        }
        // compute [13/13] Pade approximant
        std::complex<double> *eA_ptr = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
        
        cudaMemcpy(eA_ptr, d_u1, n * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost); // cuComplex -> std::complex<double>
        cudaStreamSynchronize(stream);

        std::vector<size_t> eA_shape{n*n};
        py::array_t<std::complex<double>> eA(eA_shape, eA_ptr);

        // float ms;
        // cudaEventRecord(end);
        // cudaEventSynchronize(end);
        // cudaEventElapsedTime(&ms, start, end);
        // printf("GEMM Kernel Time: %.4f ms\n", ms);

        return eA;
    }
    
    int32_t getN() {
        return this->n;
    }

    void free() {
        /*
        销毁该对象
        */
        if (d_A) {
            CUDA_CHECK(cudaFree(d_A));
            d_A = nullptr;
        };
        if (d_A2) {
            CUDA_CHECK(cudaFree(d_A2));
            d_A2 = nullptr;
        };
        if (d_A4) {
            CUDA_CHECK(cudaFree(d_A4));
            d_A4 = nullptr;
        };
        if (d_A6) {
            CUDA_CHECK(cudaFree(d_A6));
            d_A6 = nullptr;
        };
        if (d_u1) {
            CUDA_CHECK(cudaFree(d_u1));
            d_u1 = nullptr;
        };
        if (d_u2) {
            CUDA_CHECK(cudaFree(d_u2));
            d_u2 = nullptr;
        };
        if (d_v1) {
            CUDA_CHECK(cudaFree(d_v1));
            d_v1 = nullptr;
        };
        if (d_v2) {
            CUDA_CHECK(cudaFree(d_v2));
            d_v2 = nullptr;
        };
        if (d_nrmA) {
            CUDA_CHECK(cudaFree(d_nrmA));
            d_nrmA = nullptr;
        };

        // Destroy cuBLAS handle
        if (cublasH) {
            cublasDestroy_v2(cublasH);
            cublasH = nullptr;
        }
        // Destroy cuSolver handle
        if (cusolverH) {
            cusolverDnDestroy(cusolverH);
            cusolverH = nullptr;
        }
        // Destroy CUDA stream
        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }

};


#endif