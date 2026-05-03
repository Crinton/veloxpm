#ifndef MATRIXEXPCALCULATOR_imag_H
#define MATRIXEXPCALCULATOR_imag_H
#include <cmath>       // 包含 expf, cosf, sinf 等数学函数
#include <vector>
#include "matrix.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>



namespace py = pybind11;

template<typename VT>
constexpr VT get_th13_val_imag() {
    if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, cuComplex>) {
        return 5.371920351148152f;
    } else if constexpr (std::is_same_v<VT, double> || std::is_same_v<VT, cuDoubleComplex>) {
        return 3.925724783138660;
    }
}

// 2. 编译时获取 theta_vec 的值（返回 std::array）
template<typename VT>
constexpr auto get_theta_vec_array_imag() {
    if constexpr (std::is_same_v<VT, double> || std::is_same_v<VT, cuDoubleComplex>) {
        return std::array<VT, 4>{1.495585217958292e-002f, 2.539398330063230e-001f, 9.504178996162932e-001, 2.097847961257068e+000};
    } else if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, cuComplex>) {
        return std::array<VT, 2>{4.258730016922831e-001f, 1.880152677804762e+000f};
    }
}



template <typename VT>
class MatrixExpCalculator_imag {


public:
    using cuComplexType = typename complexType<VT>::type;
    using cComplexType = typename complexType<VT>::ctype; // 假设 ctype 也是一个依赖类型
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
    VT *d_mu;
    VT *d_nrmA;

    cuComplexType *d_P;
    cuComplexType *d_Q;
    
    MatrixExpCalculator_imag(size_t n); 

    ~MatrixExpCalculator_imag();

    void _pade3();

    void _pade5();

    void _pade7();

    void _pade9();

    void _pade13();


    py::array_t<cComplexType> run(py::array_t<VT>& arr_a);
    
    int32_t getN();

    void free();

};

// MatrixExpCalculator 构造函数的实现
template <typename VT>
MatrixExpCalculator_imag<VT>::MatrixExpCalculator_imag(size_t n) : n(n) {
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
    
    // 两个复数矩阵显存
    cudaMallocAsync(&d_P, n * n * sizeof(cuComplexType), stream);
    cudaMallocAsync(&d_Q, n * n * sizeof(cuComplexType), stream);

    // 小显存并初始化
    cudaMallocAsync(&d_mu, 1 * sizeof(VT), stream);
    cudaMemsetAsync(d_mu, 0, 1 * sizeof(VT), stream);

    cudaMallocAsync(&d_nrmA, 1 * sizeof(VT), stream);
    cudaMemsetAsync(d_nrmA, 0, 1 * sizeof(VT), stream);
}

template <typename VT>
MatrixExpCalculator_imag<VT>::~MatrixExpCalculator_imag() {
    free();
}

template <typename VT>
int32_t MatrixExpCalculator_imag<VT>::getN(){
    return this->n;
}

template <typename VT>
void MatrixExpCalculator_imag<VT>::free() {
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
    if (d_P) {
        CUDA_CHECK(cudaFree(d_P));
        d_P = nullptr;
    };
    if (d_Q) {
        CUDA_CHECK(cudaFree(d_Q));
        d_Q = nullptr;
    };
    if (d_mu) {
        CUDA_CHECK(cudaFree(d_mu));
        d_mu = nullptr;
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
void MatrixExpCalculator_imag<VT>::_pade3() {
    /*
    直接对类的类型进行修改，不需要显式的输入和返回
    A2 = A@A
    U = A@(b3*A2 + b1), U 用d_u1表示
    V = b2*A2 + b0,  V 用d_v1表示
    */


    gemm(cublasH, n, n, n, d_A, d_A, d_A2, -1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, 1.0f, 0.0f);

    _fuse3(d_A2, d_u2, d_v1, n, stream); //(b3*A2 + b1) 直接写到d_u2上, d_v1 = V = b2*A2 + b0

    gemm(cublasH, n, n, n, d_A, d_u2, d_u1, 1.0f, 0.0f); //d_u1 = U = d_A@d_u2

    //此时, d_u1为U, d_v1为V
    combinePQ(d_P, d_Q, d_u1, d_v1, n, stream);

    // VT val_minus_1 = -1.0;
    // VT val_plus_1 = 1.0;
    
    // cudaMemcpyAsync(d_u2, d_u1, n * n * sizeof(VT), cudaMemcpyDeviceToDevice, stream); //d_u2 := d_u1
    // cublasAPI<VT>::Axpy(cublasH, n*n, &val_plus_1, d_v1, 1 ,d_u1, 1); // d_u1 = U+V
    // cublasAPI<VT>::Axpy(cublasH, n*n, &val_minus_1, d_u2, 1 ,d_v1, 1); // d_v1 = -U+V

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }
}


template <typename VT>
void MatrixExpCalculator_imag<VT>::_pade5() {
    /*
    直接对类的类型进行修改，不需要显式的输入和返回
    A2 = A@A
    A4 = A2@A2
    U = A@(b5*A4 + b3*A2 + b1), U 用d_u1表示
    V = b4*A4 + b2*A2 + b0,  V 用d_v1表示
    */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 1: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }

    gemm(cublasH, n, n, n, d_A, d_A, d_A2, -1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, 1.0f, 0.0f);

    _fuse5(d_A2, d_A4, d_u2, d_v1, n, stream); //(b5*A4 + b3*A2 + b1) 直接写到d_u2上, d_v1 = V = b4*A4 + b2*A2 + b0

    gemm(cublasH, n, n, n, d_A, d_u2, d_u1, 1.0f, 0.0f); //d_u1 = U = d_A@d_u2

    //此时, d_u1为U, d_v1为V
    combinePQ(d_P, d_Q, d_u1, d_v1, n, stream);

    // VT val_minus_1 = -1.0;
    // VT val_plus_1 = 1.0;
    
    // cudaMemcpyAsync(d_u2, d_u1, n * n * sizeof(VT), cudaMemcpyDeviceToDevice, stream); //d_u2 := d_u1
    // cublasAPI<VT>::Axpy(cublasH, n*n, &val_plus_1, d_v1, 1 ,d_u1, 1); // d_u1 = U+V
    // cublasAPI<VT>::Axpy(cublasH, n*n, &val_minus_1, d_u2, 1 ,d_v1, 1); // d_v1 = -U+V

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 2: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }
}

template <typename VT>
void MatrixExpCalculator_imag<VT>::_pade7() {
    /*
    直接对类的类型进行修改，不需要显式的输入和返回
    A2 = A@A
    A4 = A2@A2
    A6 = A2@A4
    U = A@(b7*A6 + b5*A4 + b3*A2 + b1), U 用d_u1表示
    V = b6*A6 + b4*A4 + b2*A2 + b0,  V 用d_v1表示
    */

    gemm(cublasH, n, n, n, d_A, d_A, d_A2, -1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, 1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A2, d_A4, d_A6, 1.0f, 0.0f);


    _fuse7(d_A2, d_A4, d_A6, d_u2, d_v1, n, stream); //(b7*A6 + b5*A4 + b3*A2 + b1) 直接写到d_u2上, d_v1 = V = b6*A6 + b4*A4 + b2*A2 + b0

    gemm(cublasH, n, n, n, d_A, d_u2, d_u1, 1.0f, 0.0f); //d_u1 = U = d_A@d_u2

    // std::vector<VT> x(n*n);
    // std::cout << "d_u1: " << "\n";
    // cudaMemcpy(x.data(), d_u1, n * n * sizeof(VT), cudaMemcpyDeviceToHost);
    // print_mat(x.data(), 5, 5, n);

    //此时, d_u1为U，还差一个i, d_v1为V
    combinePQ(d_P, d_Q, d_u1, d_v1, n, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }
}

template <typename VT>
void MatrixExpCalculator_imag<VT>::_pade9() {
    /*
    直接对类的类型进行修改，不需要显式的输入和返回
    A2 = A@A
    A4 = A2@A2
    A6 = A2@A4
    A8 = A4@A4, 需要一个额外的A8矩阵，这里用d_u2替代
    U = A@(b9*A8 + b7*A6 + b5*A4 + b3 * A2 + b1), U 用d_u1表示
    V = b8*A8 + b6*A6 + b4*A4 + b2*A2 + b0,  V 用d_v1表示
    */

    gemm(cublasH, n, n, n, d_A, d_A, d_A2, -1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, 1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A2, d_A4, d_A6, 1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A4, d_A4, d_u2, 1.0f, 0.0f); //d_u2 等于 d_A8


    _fuse9(d_A2, d_A4, d_A6, d_u2, d_v1, n, stream); //(b9*A8 + b7*A6 + b5*A4 + b3 * A2 + b1) 直接写到d_u2上，也就是写回到d_A8上, d_v1 = V = b8*A8 + b6*A6 + b4*A4 + b2*A2 + b0

    gemm(cublasH, n, n, n, d_A, d_u2, d_u1, 1.0f, 0.0f); //d_u1 = `U = d_A@d_u2，距离U还欠一个i

    //此时, d_u1为U, d_v1为V
    combinePQ(d_P, d_Q, d_u1, d_v1, n, stream);

    // VT val_minus_1 = -1.0;
    // VT val_plus_1 = 1.0;
    
    // cudaMemcpyAsync(d_u2, d_u1, n * n * sizeof(VT), cudaMemcpyDeviceToDevice, stream); //d_u2 := d_u1
    // cublasAPI<VT>::Axpy(cublasH, n*n, &val_plus_1, d_v1, 1 ,d_u1, 1); // d_u1 = U+V
    // cublasAPI<VT>::Axpy(cublasH, n*n, &val_minus_1, d_u2, 1 ,d_v1, 1); // d_v1 = -U+V

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }
    //这一步需要d_u1 = P, d_v1 = Q

}

template <typename VT>
void MatrixExpCalculator_imag<VT>::_pade13() {
    /*
    直接对类的类型进行修改，不需要显式的输入和返回
    */
    gemm(cublasH, n, n, n, d_A, d_A, d_A2, -1.0f, 0.0f); // d_A2 = -H^2

    gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, 1.0f, 0.0f); //d_A4 = H^4

    gemm(cublasH, n, n, n, d_A2, d_A4, d_A6, 1.0f, 0.0f); //d_A6 = -H^6

    _fuse13(d_A2, d_A4, d_A6, d_u1, d_u2, d_v1, d_v2, n, stream); 

    gemm(cublasH, n, n, n, d_A6, d_u1, d_u2, 1.0f, 1.0f); //d_u2 = -H^6 @ u1 + u2

    gemm(cublasH, n, n, n, d_A, d_u2, d_u1, 1.0f, 0.0f); //d_u1 = U` = H @ d_u2, 距离U还差一个i
    
    gemm(cublasH, n, n, n, d_A6, d_v1, d_v2, 1.0f, 1.0f); //d_v2 = V = -H^6 @ d_v1 + d_v2

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }
    // 现在d_u1 = U`, d_v2 = V

    combinePQ(d_P, d_Q, d_u1, d_v2, n, stream);
    
    // 现在d_P与d_Q已经得到

    // // 这一步，需要d_u1 = U, d_v2 = V
    // // 要不直接在这一步用一个融合的核函数计算最终的P和Q算了，
    // VT val_minus_1 = -1.0;
    // VT val_plus_1 = 1.0;
    // cudaMemcpyAsync(d_v1, d_u1, n * n * sizeof(VT), cudaMemcpyDeviceToDevice, stream); // d_v1 = d_u1
    // cublasAPI<VT>::Axpy(cublasH, n * n, &val_minus_1, d_v2, 1, d_v1, 1); // d_v1 = -u + v = -d_v2 + d_v1
    // cublasAPI<VT>::Scal(cublasH, n * n, &val_minus_1, d_v1, 1);
    // cublasAPI<VT>::Axpy(cublasH, n * n, &val_plus_1, d_v2, 1, d_u1, 1); // d_u1 = u + v = d_u1 + d_v2

    //这一步需要d_u1 = P, d_v1 = Q
}


// MatrixExpCalculator::run 方法的实现
template <typename VT> //VT = {float, double} 对应 {cuComplex, cuDoubleComplex}
py::array_t<typename MatrixExpCalculator_imag<VT>::cComplexType> MatrixExpCalculator_imag<VT>::run(py::array_t<VT>& arr_a) {
    /*
    每次run从numpy接受一个数组指针，
    下面修改为 pade3 pade5 pade7 pade9 pade11 pade13
    */
    using cuComplexType = typename MatrixExpCalculator_imag<VT>::cuComplexType;
    using cComplexType = typename MatrixExpCalculator_imag<VT>::cComplexType; 

    constexpr VT th13 = get_th13_val_imag<VT>();
    constexpr auto theta_vec = get_theta_vec_array_imag<VT>(); // auto 推断出 std::array 类型和大小
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

    VT mu;
    VT nrmA;
    /*
    d_A = H,实数矩阵，需要在solve之前还原成复数矩阵
    */
    cudaMemcpyAsync(d_A, A, n * n *sizeof(VT), cudaMemcpyHostToDevice,stream);

    // cudaMemsetAsync(d_mu, 0, 1 * sizeof(VT), stream);
    cudaMemsetAsync(d_nrmA, 0, 1 * sizeof(VT), stream); 

    // minus_eye_matrix_trace(d_A, n, d_mu, stream); 
    RowMaxAbsSum(d_A, n, d_nrmA, stream);


    cudaMemcpyAsync(&mu, d_mu, 1 * sizeof(VT), cudaMemcpyDeviceToHost);
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
    // std::cout << "s: " << s << "nrmA: " << nrmA <<", idx: " << idx << "\n";
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

    // solve(cusolverH, d_v1, d_u1, n);
    // for (int i = 0; i < s; ++i) {
    //     gemm(cublasH, n, n, n, d_u1, d_u1, d_v1, 1.0, 0.0);
    //     cudaMemcpyAsync(d_u1, d_v1, n * n * sizeof(VT), cudaMemcpyDeviceToDevice,stream);
    // }
    cuComplexType alpha_one = complexType<VT>::makeComplex(1.0,0.0);
    cuComplexType alpha_zero = complexType<VT>::makeComplex(0.0,0.0);
    solve(cusolverH, d_Q, d_P, n);

    
    for (int i = 0; i < s; ++i) {
        gemm(cublasH, n, n, n, d_P, d_P, d_Q, alpha_one, alpha_zero);
        cudaMemcpyAsync(d_P, d_Q, n * n * sizeof(cuComplexType), cudaMemcpyDeviceToDevice,stream);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 3: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }
    // auto emu = complexType<VT>::exp(complexType<VT>::makeComplex(0.0, mu)); // 计算 e^mu
    cudaStreamSynchronize(stream);
    // cublasAPI<VT>::Scal(cublasH, n*n, &emu, d_u1, 1);
    // cublasAPI<cuComplexType>::Scal(cublasH, n*n, &emu, d_P, 1);
    // compute [13/13] Pade approximant
    cComplexType *eA_ptr = (cComplexType *)malloc(n * n * sizeof(cComplexType));
    
    cudaMemcpy(eA_ptr, d_P, n * n * sizeof(cuComplexType), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 4: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }
    std::vector<size_t> eA_shape{n*n};
    py::array_t<cComplexType> eA(eA_shape, eA_ptr);

    // float ms;
    // cudaEventRecord(end);
    // cudaEventSynchronize(end);
    // cudaEventElapsedTime(&ms, start, end);
    // printf("GEMM Kernel Time: %.4f ms\n", ms);

    return eA;

}

float get_current_gpu_memory_gb_imag() {
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


#endif
