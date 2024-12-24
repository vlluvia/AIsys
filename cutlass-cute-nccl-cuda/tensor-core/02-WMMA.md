
# WMMA


## WMMA (Warp-level Matrix Multiply Accumulate) API

```
template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;

void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);
void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
void fill_fragment(fragment<...> &a, const T& v);
void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);
```

1. fragment：Tensor Core数据存储类，支持matrix_a、matrix_b和accumulator
2. load_matrix_sync：Tensor Core数据加载API，支持将矩阵数据从global memory或shared memory加载到fragment
3. store_matrix_sync：Tensor Core结果存储API，支持将计算结果从fragment存储到global memory或shared memory
4. fill_fragment：fragment填充API，支持常数值填充
5. mma_sync：Tensor Core矩阵乘计算API，支持D = AB + C或者C = AB + C


###  示例

#### CUDA Core
https://github.com/Bruce-Lee-LY/cuda_hgemm

```
__global__ void simtNaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                                size_t N, size_t K) {
    size_t row = threadIdx.y + blockDim.y * blockIdx.y;
    size_t col = threadIdx.x + blockDim.x * blockIdx.x;

    if (row >= M && col >= N) {
        return;
    }

    float tmp = 0.0;
#pragma unroll
    for (size_t i = 0; i < K; ++i) {
        tmp += __half2float(A[row * K + i]) * __half2float(B[i + col * K]);
    }

    C[row * N + col] = __float2half(tmp);
}

void simtNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    dim3 block(16, 16);
    dim3 grid(div_ceil(N, block.x), div_ceil(M, block.y));

    simtNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}
```

#### Tensor Core

https://github.com/NVIDIA/cuda-samples

```
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARP_SIZE 32

using namespace nvcuda;

__global__ void wmmaNaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                                size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, WMMA_K);

    const size_t warp_row = blockIdx.y * WMMA_M;
    const size_t warp_col = blockIdx.x * WMMA_N;

    if (warp_row >= M && warp_col >= N) {
        return;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;

    wmma::fill_fragment(C_frag, 0.0f);

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> B_frag;

        wmma::load_matrix_sync(A_frag, A + warp_row * K + i * WMMA_K, K);
        wmma::load_matrix_sync(B_frag, B + i * WMMA_K + warp_col * K, K);

        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, C_frag, N, wmma::mem_row_major);
}

void wmmaNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M));

    wmmaNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}
```

计算层级：CUDA Core是线程级别，Tensor Core是warp级别
计算维度：CUDA Core是一维逐点计算，Tensor Core是二维逐tile计算
计算依赖：WMMA调用Tensor Core需要借助数据存储类fragment，CUDA Core不需要借助其他


## HGEMM优化



