#ifndef MATRIXXXX_H
#define MATRIXXXX_H
#include <fstream>
#include <complex>
#include <type_traits>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cub/cub.cuh>
#include <cuComplex.h>

#include "cusolver_utils.h"
#include "cuapi.h"


// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                        \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

namespace cg = cooperative_groups;


constexpr int BLOCK_THREADS = 256;
constexpr int ITEMS_PER_THREAD = 4;
constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;


cuComplex cuCexpf(cuComplex z);
cuDoubleComplex cuCexp(cuDoubleComplex z);

template<typename VT>
struct complexType{

};

template<>
struct complexType<float>{
    using type = cuComplex;
    using ctype = std::complex<float>;
    static type makeComplex(float real, float imag) {
        return make_cuComplex(real, imag);
    }

    static type exp(type z_in) {
        return cuCexpf(z_in);
    }
};

template<>
struct complexType<double>{
    using type = cuDoubleComplex;
    using ctype = std::complex<double>;
    static  type makeComplex(double real, float imag) {
        return make_cuDoubleComplex(real, imag);
    }
    static type exp(type z_in) {
        return cuCexp(z_in);
    }
};


__constant__ float b13[] = {64764752532480000., 32382376266240000., 7771770303897600.,
       1187353796428800., 129060195264000., 10559470521600., 670442572800.,
       33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.};

__constant__ float b9[] = {17643225600., 8821612800., 2075673600., 302702400., 30270240.,
       2162160., 110880., 3960., 90., 1.};

__constant__ float b7[] = {17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.};


__constant__ float b5[] = {30240., 15120., 3360., 420., 30., 1.};

__constant__ float b3[] = {120., 60., 12., 1.};
cuComplex cuCexpf(cuComplex z_in);
cuDoubleComplex cuCexp(cuDoubleComplex z_in);



template <typename VT>
__global__ void minus_eye_matrix_trace_kernel(VT *d_A, int32_t n, VT *d_trace) {
    typedef typename cub::BlockReduce<VT, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    cg::grid_group grid = cg::this_grid();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    VT value = (idx < n) ? d_A[idx * n + idx] : 0.0f;
    

    // Step 2: Block-level reduction
    VT block_sum = BlockReduce(temp_storage).Sum(value);
    __syncthreads();

    // Step 3: First thread in each block writes to global memory
    if (threadIdx.x == 0) {
        block_sum /= n; // 确保除法使用浮点数
        atomicAdd(d_trace, block_sum);
    }

    // Step 4: Grid-level synchronization
    grid.sync();

    // Step 5: Update diagonal
    if (idx < n) {
        d_A[idx * n + idx] -= *d_trace; // 统一符号，改为减法
    }
}

template <typename VT>
struct ComplexSumOp {
    __device__ VT operator()(const VT &a, const VT &b) const {
        if constexpr (std::is_same_v<VT, cuComplex>) {
            return cuCaddf(a, b);
        } else if constexpr (std::is_same_v<VT, cuDoubleComplex>) {
            return cuCadd(a, b);
        } else {
            return a + b;
        }
    }
};


template <>
inline __global__ void minus_eye_matrix_trace_kernel<cuComplex>(cuComplex *d_A, int32_t n, cuComplex *d_trace) {
    typedef typename cub::BlockReduce<cuComplex, BLOCK_THREADS> BlockReduce;
    __shared__ BlockReduce::TempStorage temp_storage;
    cg::grid_group grid = cg::this_grid();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    cuComplex value = (idx < n) ? d_A[idx * n + idx] : make_cuComplex(0.0, 0.0);
    

    // Step 2: Block-level reduction
    cuComplex block_sum = BlockReduce(temp_storage).Reduce(value, ComplexSumOp<cuComplex>());
    __syncthreads();

    // Step 3: First thread in each block writes to global memory
    if (threadIdx.x == 0) {
        // block_sum /= n; // 确保除法使用浮点数
        block_sum.x /= n;
        block_sum.y /= n;
        atomicAdd(&((*d_trace).x), block_sum.x);
        atomicAdd(&((*d_trace).y), block_sum.y);
        // atomicAdd(d_trace, block_sum);
    }

    // Step 4: Grid-level synchronization
    grid.sync();

    // Step 5: Update diagonal
    if (idx < n) {
        // d_A[idx * n + idx] -= *d_trace; // 统一符号，改为减法
        d_A[idx * n + idx] = cuCsubf(d_A[idx * n + idx], *d_trace);
    }
}

template <>
inline __global__ void minus_eye_matrix_trace_kernel<cuDoubleComplex>(cuDoubleComplex *d_A, int32_t n, cuDoubleComplex *d_trace) {
    typedef typename cub::BlockReduce<cuDoubleComplex, BLOCK_THREADS> BlockReduce;
    __shared__ BlockReduce::TempStorage temp_storage;
    cg::grid_group grid = cg::this_grid();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    cuDoubleComplex value = (idx < n) ? d_A[idx * n + idx] : make_cuDoubleComplex(0.0, 0.0);
    

    // Step 2: Block-level reduction
    cuDoubleComplex block_sum = BlockReduce(temp_storage).Reduce(value, ComplexSumOp<cuDoubleComplex>());
    __syncthreads();

    // Step 3: First thread in each block writes to global memory
    if (threadIdx.x == 0) {
        // block_sum /= n; // 确保除法使用浮点数
        block_sum.x /= n;
        block_sum.y /= n;
        atomicAdd(&((*d_trace).x), block_sum.x);
        atomicAdd(&((*d_trace).y), block_sum.y);
        // atomicAdd(d_trace, block_sum);
    }

    // Step 4: Grid-level synchronization
    grid.sync();

    // Step 5: Update diagonal
    if (idx < n) {
        // d_A[idx * n + idx] -= *d_trace; // 统一符号，改为减法
        d_A[idx * n + idx] = cuCsub(d_A[idx * n + idx], *d_trace);
    }
}


template <typename VT>
void minus_eye_matrix_trace(VT *d_A, int32_t n, VT *d_trace, cudaStream_t stream) {
    dim3 blockSize(BLOCK_THREADS);
    dim3 gridSize((n + BLOCK_THREADS - 1) / BLOCK_THREADS);

    // 检查协作启动支持
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (!deviceProp.cooperativeLaunch) {
        throw std::runtime_error("Device does not support cooperative launch");
    }

    // 初始化 d_trace
    VT trace;
    if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, double> ) {
        trace = 0.0;
    } else if constexpr (std::is_same_v<VT, cuComplex>) {
        trace = make_cuComplex(0.0, 0.0);
    } else if constexpr (std::is_same_v<VT, cuDoubleComplex>) {
        trace = make_cuDoubleComplex(0.0, 0.0);
    }
    cudaMemcpy(d_trace, &trace, 1 *sizeof(VT), cudaMemcpyHostToDevice);

    // 设置核函数参数
    void *kernelArgs[] = { &d_A, &n, &d_trace };

    // 启动协作核函数
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)minus_eye_matrix_trace_kernel<VT>,
        gridSize, blockSize, kernelArgs, 0, stream
    );
    cudaStreamSynchronize(stream);

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA Launch Error: " + std::string(cudaGetErrorString(err)));
    }

}


template <typename VT, typename OT> //VT ={float, double, cuComplex, cuDoubleComplex}, OT = {float, double, float, double}
__global__ void RowMaxAbsSumKernel(const VT* __restrict__ A, int n, OT* max_result) {
    int row = blockIdx.x;
    if (row >= n) return;

    using BlockLoad = cub::BlockLoad<VT, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<VT, BLOCK_THREADS>;

    __shared__ typename BlockLoad::TempStorage load_temp;
    __shared__ typename BlockReduce::TempStorage reduce_temp;

    VT thread_data[ITEMS_PER_THREAD];
    VT thread_sum = 0.0f;

    for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        int remaining = n - tile_start;
        int valid_items = min(remaining, TILE_SIZE);
        const VT* row_ptr = A + row * n + tile_start;

        BlockLoad(load_temp).Load(row_ptr, thread_data, valid_items);
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int col = tile_start + threadIdx.x * ITEMS_PER_THREAD + i;
            if (col < n) {
                thread_sum += fabsf(thread_data[i]);
            }
        }
    }

    VT row_sum = BlockReduce(reduce_temp).Sum(thread_sum);

    if (threadIdx.x == 0) {
        cuda::atomic_ref<VT, cuda::thread_scope_system> a(*max_result);
        a.fetch_max(row_sum);
    }
}

template <>
inline __global__ void RowMaxAbsSumKernel<cuComplex>(const cuComplex* __restrict__ A, int n, float* max_result) {
    int row = blockIdx.x;
    if (row >= n) return;

    using BlockLoad = cub::BlockLoad<cuComplex, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;

    __shared__ typename BlockLoad::TempStorage load_temp;
    __shared__ typename BlockReduce::TempStorage reduce_temp;

    cuComplex thread_data[ITEMS_PER_THREAD];
    float thread_sum = 0.0f; //复数的绝对值都是实数

    for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        int remaining = n - tile_start;
        int valid_items = min(remaining, TILE_SIZE);
        const cuComplex* row_ptr = A + row * n + tile_start;

        BlockLoad(load_temp).Load(row_ptr, thread_data, valid_items);
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int col = tile_start + threadIdx.x * ITEMS_PER_THREAD + i;
            if (col < n) {
                // thread_sum += fabsf(thread_data[i]);
                thread_sum += cuCabsf(thread_data[i]);
                // thread_sum.x += thread_data_real;
            }
        }
    }

    float row_sum = BlockReduce(reduce_temp).Sum(thread_sum);

    if (threadIdx.x == 0) {
        cuda::atomic_ref<float, cuda::thread_scope_system> a(*max_result);
        a.fetch_max(row_sum);
    }
}

template <>
inline __global__ void RowMaxAbsSumKernel<cuDoubleComplex>(const cuDoubleComplex* __restrict__ A, int n, double* max_result) {
    int row = blockIdx.x;
    if (row >= n) return;

    using BlockLoad = cub::BlockLoad<cuDoubleComplex, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<double, BLOCK_THREADS>;

    __shared__ typename BlockLoad::TempStorage load_temp;
    __shared__ typename BlockReduce::TempStorage reduce_temp;

    cuDoubleComplex thread_data[ITEMS_PER_THREAD];
    double thread_sum = 0.0; //复数的绝对值都是实数

    for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        int remaining = n - tile_start;
        int valid_items = min(remaining, TILE_SIZE);
        const cuDoubleComplex* row_ptr = A + row * n + tile_start;

        BlockLoad(load_temp).Load(row_ptr, thread_data, valid_items);
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int col = tile_start + threadIdx.x * ITEMS_PER_THREAD + i;
            if (col < n) {
                // thread_sum += fabsf(thread_data[i]);
                thread_sum += cuCabs(thread_data[i]);
                // thread_sum.x += thread_data_real;
            }
        }
    }

    double row_sum = BlockReduce(reduce_temp).Sum(thread_sum);

    if (threadIdx.x == 0) {
        cuda::atomic_ref<double, cuda::thread_scope_system> a(*max_result);
        a.fetch_max(row_sum);
    }
}


template <typename VT, typename OT> //VT ={float, double, cuComplex, cuDoubleComplex}, OT = {float, double, float, double}
void RowMaxAbsSum(const VT* d_A, int n, OT* d_max_result, cudaStream_t stream) {
    dim3 blockSize(BLOCK_THREADS);     // 128
    dim3 gridSize(n);                  // 每行一个 block
    RowMaxAbsSumKernel<<<gridSize, blockSize, 0, stream>>>(d_A, n, d_max_result);
}

template <typename VT> //VT = {flaot, double}
__global__ void fuse3_kernel( const VT *d_A2, VT *d_u2, VT *d_v1, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        VT a2 = d_A2[idx];
        VT mask = (idx % (m+1) == 0) ? 1.0f : 0.0f;
        d_u2[idx] = b3[3] * a2 + b3[1] * mask;
        d_v1[idx] = b3[2] * a2 + b3[0] * mask;
    }
}



template <>
inline __global__ void fuse3_kernel<cuComplex>( const cuComplex *d_A2, cuComplex *d_u2, cuComplex *d_v1, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        cuComplex a2 = d_A2[idx];
        cuComplex mask = (idx % (m+1) == 0) ? make_cuComplex(1.0,0.0) : make_cuComplex(0.0,0.0);

        d_u2[idx].x = b3[3] * a2.x + b3[1] * mask.x;
        d_u2[idx].y = b3[3] * a2.y + b3[1] * mask.y;


        d_v1[idx].x = b3[2] * a2.x + b3[0] * mask.x;
        d_v1[idx].y = b3[2] * a2.y + b3[0] * mask.y;
    }
}

template <>
inline __global__ void fuse3_kernel<cuDoubleComplex>( const cuDoubleComplex *d_A2,cuDoubleComplex *d_u2, cuDoubleComplex *d_v1, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        cuDoubleComplex a2 = d_A2[idx];
        cuDoubleComplex mask = (idx % (m+1) == 0) ? make_cuDoubleComplex(1.0,0.0) : make_cuDoubleComplex(0.0,0.0);

        d_u2[idx].x = b3[3] * a2.x + b3[1] * mask.x;
        d_u2[idx].y = b3[3] * a2.y + b3[1] * mask.y;

        d_v1[idx].x = b3[2] * a2.x + b3[0] * mask.x;
        d_v1[idx].y = b3[2] * a2.y + b3[0] * mask.y;
    }
}

template <typename VT>
void _fuse3( const VT *d_A2, VT *d_u2, VT *d_v1, int m, cudaStream_t stream) {
    dim3 BlockSize(BLOCK_THREADS);
    dim3 GridSize((m * m+ BlockSize.x -1) / BlockSize.x);
    fuse3_kernel<<<GridSize, BlockSize, 0, stream>>>(d_A2, d_u2, d_v1, m);
}

template <typename VT> //VT = {flaot, double}
__global__ void fuse5_kernel( const VT *d_A2, const VT *d_A4, VT *d_u2, VT *d_v1, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        VT a4 = d_A4[idx];
        VT a2 = d_A2[idx];
        VT mask = (idx % (m+1) == 0) ? 1.0f : 0.0f;
        d_u2[idx] = b5[5] * a4 + b5[3] * a2 + b5[1] * mask;
        d_v1[idx] = b5[4] * a4 + b5[2] * a2 + b5[0] * mask;
    }
}



template <>
inline __global__ void fuse5_kernel<cuComplex>( const cuComplex *d_A2, const cuComplex *d_A4, cuComplex *d_u2, cuComplex *d_v1, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        cuComplex a4 = d_A4[idx];
        cuComplex a2 = d_A2[idx];
        cuComplex mask = (idx % (m+1) == 0) ? make_cuComplex(1.0,0.0) : make_cuComplex(0.0,0.0);

        d_u2[idx].x = b5[5] * a4.x + b5[3] * a2.x + b5[1] * mask.x;
        d_u2[idx].y = b5[5] * a4.y + b5[3] * a2.y + b5[1] * mask.y;


        d_v1[idx].x = b5[4] * a4.x + b5[2] * a2.x + b5[0] * mask.x;
        d_v1[idx].y = b5[4] * a4.y + b5[2] * a2.y + b5[0] * mask.y;
    }
}

template <>
inline __global__ void fuse5_kernel<cuDoubleComplex>( const cuDoubleComplex *d_A2, const cuDoubleComplex *d_A4, cuDoubleComplex *d_u2, cuDoubleComplex *d_v1, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        cuDoubleComplex a4 = d_A4[idx];
        cuDoubleComplex a2 = d_A2[idx];
        cuDoubleComplex mask = (idx % (m+1) == 0) ? make_cuDoubleComplex(1.0,0.0) : make_cuDoubleComplex(0.0,0.0);

        d_u2[idx].x = b5[5] * a4.x + b5[3] * a2.x + b5[1] * mask.x;
        d_u2[idx].y = b5[5] * a4.y + b5[3] * a2.y + b5[1] * mask.y;


        d_v1[idx].x = b5[4] * a4.x + b5[2] * a2.x + b5[0] * mask.x;
        d_v1[idx].y = b5[4] * a4.y + b5[2] * a2.y + b5[0] * mask.y;
    }
}

template <typename VT>
void _fuse5( const VT *d_A2, const VT *d_A4, VT *d_u2, VT *d_v1, int m, cudaStream_t stream) {
    dim3 BlockSize(BLOCK_THREADS);
    dim3 GridSize((m * m+ BlockSize.x -1) / BlockSize.x);
    fuse5_kernel<<<GridSize, BlockSize, 0, stream>>>(d_A2, d_A4, d_u2, d_v1, m);
}

template <typename VT> //VT = {flaot, double}
__global__ void fuse7_kernel( const VT *d_A2, const VT *d_A4, const VT *d_A6, VT *d_u2, VT *d_v1, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        VT a6 = d_A6[idx];
        VT a4 = d_A4[idx];
        VT a2 = d_A2[idx];
        VT mask = (idx % (m+1) == 0) ? 1.0f : 0.0f;
        d_u2[idx] = b7[7] * a6 + b7[5] * a4 + b7[3] * a2 + b7[1] * mask;
        d_v1[idx] = b7[6] * a6 + b7[4] * a4 + b7[2] * a2 + b7[0] * mask;
    }
}



template <>
inline __global__ void fuse7_kernel<cuComplex>( const cuComplex *d_A2, const cuComplex *d_A4, const cuComplex *d_A6, cuComplex *d_u2, cuComplex *d_v1, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        cuComplex a6 = d_A6[idx];
        cuComplex a4 = d_A4[idx];
        cuComplex a2 = d_A2[idx];
        cuComplex mask = (idx % (m+1) == 0) ? make_cuComplex(1.0,0.0) : make_cuComplex(0.0,0.0);

        d_u2[idx].x = b7[7] * a6.x + b7[5] * a4.x + b7[3] * a2.x + b7[1] * mask.x;
        d_u2[idx].y = b7[7] * a6.y + b7[5] * a4.y + b7[3] * a2.y + b7[1] * mask.y;


        d_v1[idx].x = b7[6] * a6.x + b7[4] * a4.x + b7[2] * a2.x + b7[0] * mask.x;
        d_v1[idx].y = b7[6] * a6.y + b7[4] * a4.y + b7[2] * a2.y + b7[0] * mask.y;
    }
}

template <>
inline __global__ void fuse7_kernel<cuDoubleComplex>( const cuDoubleComplex *d_A2, const cuDoubleComplex *d_A4, const cuDoubleComplex *d_A6, cuDoubleComplex *d_u2, cuDoubleComplex *d_v1, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        cuDoubleComplex a6 = d_A6[idx];
        cuDoubleComplex a4 = d_A4[idx];
        cuDoubleComplex a2 = d_A2[idx];
        cuDoubleComplex mask = (idx % (m+1) == 0) ? make_cuDoubleComplex(1.0,0.0) : make_cuDoubleComplex(0.0,0.0);

        d_u2[idx].x = b7[7] * a6.x + b7[5] * a4.x + b7[3] * a2.x + b7[1] * mask.x;
        d_u2[idx].y = b7[7] * a6.y + b7[5] * a4.y + b7[3] * a2.y + b7[1] * mask.y;

        d_v1[idx].x = b7[6] * a6.x + b7[4] * a4.x + b7[2] * a2.x + b7[0] * mask.x;
        d_v1[idx].y = b7[6] * a6.y + b7[4] * a4.y + b7[2] * a2.y + b7[0] * mask.y;
    }
}

template <typename VT>
void _fuse7(const VT *d_A2, const VT *d_A4, const VT *d_A6, VT *d_u2,  VT *d_v1, int m, cudaStream_t stream) {
    dim3 BlockSize(BLOCK_THREADS);
    dim3 GridSize((m * m+ BlockSize.x -1) / BlockSize.x);
    fuse7_kernel<<<GridSize, BlockSize, 0, stream>>>(d_A2, d_A4, d_A6, 
                                  d_u2, d_v1, m);
}

template <typename VT> //VT = {flaot, double}
__global__ void fuse9_kernel(const VT *d_A2, const VT *d_A4, const VT *d_A6, VT *d_u2, VT *d_v1, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        VT a8 = d_u2[idx];
        VT a6 = d_A6[idx];
        VT a4 = d_A4[idx];
        VT a2 = d_A2[idx];
        VT mask = (idx % (m+1) == 0) ? 1.0f : 0.0f;
        d_u2[idx] = b9[9] * a8 + b9[7] * a6 + b9[5] * a4 + b9[3] * a2 + b9[1] * mask;
        d_v1[idx] = b9[8] * a8 + b9[6] * a6 + b9[4] * a4 + b9[2] * a2 + b9[0] * mask;
    }
}



template <>
inline __global__ void fuse9_kernel<cuComplex>( const cuComplex *d_A2, const cuComplex *d_A4, const cuComplex *d_A6, cuComplex *d_u2, cuComplex *d_v1, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        cuComplex a8 = d_u2[idx];
        cuComplex a6 = d_A6[idx];
        cuComplex a4 = d_A4[idx];
        cuComplex a2 = d_A2[idx];
        cuComplex mask = (idx % (m+1) == 0) ? make_cuComplex(1.0,0.0) : make_cuComplex(0.0,0.0);


        d_u2[idx].x = b9[9] * a8.x + b9[7] * a6.x + b9[5] * a4.x + b9[3] * a2.x + b9[1] * mask.x;
        d_u2[idx].y = b9[9] * a8.y + b9[7] * a6.y + b9[5] * a4.y + b9[3] * a2.y + b9[1] * mask.y;

        d_v1[idx].x = b9[8] * a8.x + b9[6] * a6.x + b9[4] * a4.x + b9[2] * a2.x + b9[0] * mask.x;
        d_v1[idx].y = b9[8] * a8.y + b9[6] * a6.y + b9[4] * a4.y + b9[2] * a2.y + b9[0] * mask.y;
    }
}

template <>
inline __global__ void fuse9_kernel<cuDoubleComplex>( const cuDoubleComplex *d_A2, const cuDoubleComplex *d_A4, const cuDoubleComplex *d_A6, cuDoubleComplex *d_u2, cuDoubleComplex *d_v1,int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        cuDoubleComplex a8 = d_u2[idx];
        cuDoubleComplex a6 = d_A6[idx];
        cuDoubleComplex a4 = d_A4[idx];
        cuDoubleComplex a2 = d_A2[idx];
        cuDoubleComplex mask = (idx % (m+1) == 0) ? make_cuDoubleComplex(1.0,0.0) : make_cuDoubleComplex(0.0,0.0);

        d_u2[idx].x = b9[9] * a8.x + b9[7] * a6.x + b9[5] * a4.x + b9[3] * a2.x + b9[1] * mask.x;
        d_u2[idx].y = b9[9] * a8.y + b9[7] * a6.y + b9[5] * a4.y + b9[3] * a2.y + b9[1] * mask.y;

        d_v1[idx].x = b9[8] * a8.x + b9[6] * a6.x + b9[4] * a4.x + b9[2] * a2.x + b9[0] * mask.x;
        d_v1[idx].y = b9[8] * a8.y + b9[6] * a6.y + b9[4] * a4.y + b9[2] * a2.y + b9[0] * mask.y;

    }
}

template <typename VT>
void _fuse9(const VT *d_A2, const VT *d_A4, const VT *d_A6, VT *d_u2,  VT *d_v1, int m, cudaStream_t stream) {
    dim3 BlockSize(BLOCK_THREADS);
    dim3 GridSize((m * m+ BlockSize.x -1) / BlockSize.x);
    fuse9_kernel<<<GridSize, BlockSize, 0, stream>>>(d_A2, d_A4, d_A6, 
                                  d_u2, d_v1, m);
}

template <typename VT> //VT = {flaot, double}
__global__ void fuse13_kernel( const VT *d_A2, const VT *d_A4, const VT *d_A6,
          VT *d_u1, VT *d_u2, VT *d_v1, VT *d_v2, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        VT a6 = d_A6[idx];
        VT a4 = d_A4[idx];
        VT a2 = d_A2[idx];
        VT mask = (idx % (m+1) == 0) ? 1.0f : 0.0f;
        d_u1[idx] = b13[13] * a6 + b13[11] * a4 + b13[9] * a2;
        d_u2[idx] = b13[7] * a6 + b13[5] * a4 + b13[3] * a2 + b13[1] * mask;
        d_v1[idx] = b13[12] * a6 + b13[10] * a4 + b13[8] * a2;
        d_v2[idx] = b13[6] * a6 + b13[4] * a4 + b13[2] * a2 + b13[0] * mask;
    }
}



template <>
inline __global__ void fuse13_kernel<cuComplex>( const cuComplex *d_A2, const cuComplex *d_A4, const cuComplex *d_A6,
          cuComplex *d_u1, cuComplex *d_u2, cuComplex *d_v1, cuComplex *d_v2, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        cuComplex a6 = d_A6[idx];
        cuComplex a4 = d_A4[idx];
        cuComplex a2 = d_A2[idx];
        cuComplex mask = (idx % (m+1) == 0) ? make_cuComplex(1.0,0.0) : make_cuComplex(0.0,0.0);
        d_u1[idx].x = b13[13] * a6.x + b13[11] * a4.x + b13[9] * a2.x;
        d_u1[idx].y = b13[13] * a6.y + b13[11] * a4.y + b13[9] * a2.y;

        d_u2[idx].x = b13[7] * a6.x + b13[5] * a4.x + b13[3] * a2.x + b13[1] * mask.x;
        d_u2[idx].y = b13[7] * a6.y + b13[5] * a4.y + b13[3] * a2.y + b13[1] * mask.y;

        d_v1[idx].x = b13[12] * a6.x + b13[10] * a4.x + b13[8] * a2.x;
        d_v1[idx].y = b13[12] * a6.y + b13[10] * a4.y + b13[8] * a2.y;

        d_v2[idx].x = b13[6] * a6.x + b13[4] * a4.x + b13[2] * a2.x + b13[0] * mask.x;
        d_v2[idx].y = b13[6] * a6.y + b13[4] * a4.y + b13[2] * a2.y + b13[0] * mask.y;
    }
}

template <>
inline __global__ void fuse13_kernel<cuDoubleComplex>( const cuDoubleComplex *d_A2, const cuDoubleComplex *d_A4, const cuDoubleComplex *d_A6,
          cuDoubleComplex *d_u1, cuDoubleComplex *d_u2, cuDoubleComplex *d_v1, cuDoubleComplex *d_v2, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        cuDoubleComplex a6 = d_A6[idx];
        cuDoubleComplex a4 = d_A4[idx];
        cuDoubleComplex a2 = d_A2[idx];
        cuDoubleComplex mask = (idx % (m+1) == 0) ? make_cuDoubleComplex(1.0,0.0) : make_cuDoubleComplex(0.0,0.0);
        d_u1[idx].x = b13[13] * a6.x + b13[11] * a4.x + b13[9] * a2.x;
        d_u1[idx].y = b13[13] * a6.y + b13[11] * a4.y + b13[9] * a2.y;

        d_u2[idx].x = b13[7] * a6.x + b13[5] * a4.x + b13[3] * a2.x + b13[1] * mask.x;
        d_u2[idx].y = b13[7] * a6.y + b13[5] * a4.y + b13[3] * a2.y + b13[1] * mask.y;

        d_v1[idx].x = b13[12] * a6.x + b13[10] * a4.x + b13[8] * a2.x;
        d_v1[idx].y = b13[12] * a6.y + b13[10] * a4.y + b13[8] * a2.y;

        d_v2[idx].x = b13[6] * a6.x + b13[4] * a4.x + b13[2] * a2.x + b13[0] * mask.x;
        d_v2[idx].y = b13[6] * a6.y + b13[4] * a4.y + b13[2] * a2.y + b13[0] * mask.y;
    }
}

template <typename VT>
void _fuse13( const VT *d_A2, const VT *d_A4, const VT *d_A6, VT *d_u1, VT *d_u2, VT *d_v1, VT *d_v2, int m, cudaStream_t stream) {
    dim3 BlockSize(BLOCK_THREADS);
    dim3 GridSize((m * m+ BlockSize.x -1) / BlockSize.x);
    fuse13_kernel<<<GridSize, BlockSize, 0, stream>>>(d_A2, d_A4, d_A6, 
                                  d_u1, d_u2, d_v1, d_v2, m);
}


/* ----------------------------------实数化-------------------------------------------*/


template <typename VT> //VT = {flaot, double}
__global__ void fuse13_real_kernel( const VT *d_A2, const VT *d_A4, const VT *d_A6,
          VT *d_u1, VT *d_u2, VT *d_v1, VT *d_v2, int m) {
    /*
    这里好像不用改变下面线性组合的符号，d_A2, d_A4, d_A6的符号以及在矩阵里面了，这里直接做加法就行
    */
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        VT a6 = d_A6[idx];
        VT a4 = d_A4[idx];
        VT a2 = d_A2[idx];
        VT mask = (idx % (m+1) == 0) ? 1.0f : 0.0f;
        d_u1[idx] = b13[13] * a6 + b13[11] * a4 + b13[9] * a2;
        d_u2[idx] = b13[7] * a6 + b13[5] * a4 + b13[3] * a2 + b13[1] * mask;
        d_v1[idx] = b13[12] * a6 + b13[10] * a4 + b13[8] * a2;
        d_v2[idx] = b13[6] * a6 + b13[4] * a4 + b13[2] * a2 + b13[0] * mask;
    }
}

template <typename VT>
void _fuse13_real( const VT *d_A2, const VT *d_A4, const VT *d_A6, VT *d_u1, VT *d_u2, VT *d_v1, VT *d_v2, int m, cudaStream_t stream) {
    dim3 BlockSize(BLOCK_THREADS);
    dim3 GridSize((m * m+ BlockSize.x -1) / BlockSize.x);
    fuse13_real_kernel<<<GridSize, BlockSize, 0, stream>>>(d_A2, d_A4, d_A6, 
                                  d_u1, d_u2, d_v1, d_v2, m);
}

cublasStatus_t gemm(cublasHandle_t handle, int32_t M, int32_t K, int32_t N, float *d_A,
        float *d_B, float *d_C, float alpha, float beta);

cublasStatus_t gemm(cublasHandle_t handle, int32_t M, int32_t K, int32_t N, double *d_A,
        double *d_B, double *d_C, double alpha, double beta);

cublasStatus_t gemm(cublasHandle_t handle, int32_t M, int32_t K, int32_t N, cuComplex *d_A,
        cuComplex *d_B, cuComplex *d_C, cuComplex alpha, cuComplex beta);

cublasStatus_t gemm(cublasHandle_t handle, int32_t M, int32_t K, int32_t N, cuDoubleComplex *d_A,
        cuDoubleComplex *d_B, cuDoubleComplex *d_C, cuDoubleComplex alpha, cuDoubleComplex beta);


void solve(cusolverDnHandle_t handle, float *d_A, float *d_B, int m);

void solve(cusolverDnHandle_t handle, double *d_A, double *d_B, int m);

void solve(cusolverDnHandle_t handle, cuComplex *d_A, cuComplex *d_B, int m);

void solve(cusolverDnHandle_t handle, cuDoubleComplex *d_A, cuDoubleComplex *d_B, int m);


template <typename VT>
void readBinaryFloatArray(const std::string& filename, 
                          std::vector<VT>& data, 
                          int numElements) {
    // 打开文件并验证
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 验证文件大小是否有效
    size_t fileElements = fileSize / sizeof(VT);
    if (fileSize % sizeof(VT) != 0) {
        throw std::runtime_error("文件大小不匹配: 非浮点数组格式");
    }
    
    // 检查元素数量是否匹配
    if (numElements > 0 && static_cast<size_t>(numElements) != fileElements) {
        throw std::runtime_error("元素数量不匹配: 预期 " + std::to_string(numElements) +
                                " 但实际 " + std::to_string(fileElements));
    }
    
    // 调整向量大小并读取数据
    data.resize(fileElements);
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    
    // 验证读取完整性
    if (!file || file.gcount() != static_cast<std::streamsize>(fileSize)) {
        throw std::runtime_error("读取不完整: 只读取了 " + 
                                std::to_string(file.gcount()) + "/" + 
                                std::to_string(fileSize) + " 字节");
    }
}

template <typename VT>
void writeBinaryFloatArray(const std::string& filename,
                           const std::vector<VT>& data) {
    // 以二进制写入方式打开文件
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("无法打开文件进行写入: " + filename);
    }

    // 写入数据
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(VT));

    // 验证写入完整性
    if (!file) {
        throw std::runtime_error("写入失败: " + filename);
    }
}

template <typename VT>
void print_mat(VT *mat, int m, int n, int lda) {
    std::ios_base::fmtflags orig_flags = std::cout.flags();
    int orig_precision = std::cout.precision();
    std::cout << std::fixed << std::setprecision(12);
    for (int i = 0; i < m; ++i) {
        for (int j = 0 ; j < n; ++j) {
            if constexpr(std::is_same_v<VT, float> || std::is_same_v<VT, double>) {
                std::cout << mat[i*lda + j] << ", ";
            } else if constexpr(std::is_same_v<VT, cuComplex>) {
                std::cout << mat[i*lda + j].x << "," << mat[i*lda + j].y << "i" << ",";
            } else if constexpr(std::is_same_v<VT, cuComplex>) {
                std::cout << mat[i*lda + j].x << "," << mat[i*lda + j].y << "i" << ",";
            } 
        }
        std::cout << "\n";
    }
// 恢复 cout 的原始状态
    std::cout.flags(orig_flags);
    std::cout.precision(orig_precision);
}

template<typename VT, std::size_t N> // 模板参数增加 std::size_t N 来匹配 std::array 的大小
int digitize_cpp(VT value, const std::array<VT, N>& bins) { // 接收 const std::array 引用
    // std::upper_bound 现在使用 std::array 的迭代器
    auto it = std::upper_bound(bins.begin(), bins.end(), value);
    
    // 计算索引
    return std::distance(bins.begin(), it);
}




/*
combinePQ实现了从d_u2 d_v1 直接得到P Q矩阵
U是纯虚复数矩阵，V是纯实复数矩阵
d_u2是U的虚数部矩阵, d_v1是V的实数部矩阵
*/
__global__ void combinePQ_kernel(cuComplex *d_P, cuComplex *d_Q, float *d_U, float *d_V, int size);

__global__ void combinePQ_kernel(cuDoubleComplex *d_P, cuDoubleComplex *d_Q, double *d_U, double *d_V, int size);

void combinePQ(cuComplex *d_P, cuComplex *d_Q, float *d_U, float *d_V, int size, cudaStream_t stream);

void combinePQ(cuDoubleComplex *d_P, cuDoubleComplex *d_Q, double *d_U, double *d_V, int size, cudaStream_t stream);

#endif