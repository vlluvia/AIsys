


![Alt text](image-hmma-1.png)

实现HGEMM：C = AB，其中矩阵A（M * K，row major）、B（K * N，col major）和C（M * N，row major）的精度均为FP16

MMA PTX的编程思路类似于WMMA API，都是按照每个warp处理一个矩阵C的tile的思路来构建naive kernel。首先确定当前warp处理矩阵C的tile坐标，声明计算tilie所需的shared memory和寄存器，再以MMA_K为步长遍历K并从global memory经shared memory由LDMATRIX PTX加载所需A、B矩阵tile到寄存器参与计算，最后将计算结果从寄存器经shared memory写回矩阵C。所有block计算完成之后即可得到矩阵C。


```
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

#define WARP_SIZE 32

__global__ void mmaNaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C, size_t M,
                               size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    if (warp_row >= M || warp_col >= N) {
        return;
    }

    __shared__ half A_shmem[MMA_M][MMA_K];
    __shared__ half B_shmem[MMA_N][MMA_K];
    __shared__ half C_shmem[MMA_M][MMA_N];

    const size_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t RC[2] = {0, 0};

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        *((int4 *)(&A_shmem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);

        if (lane_id < MMA_N * 2) {
            *((int4 *)(&B_shmem[lane_id / 2][0]) + lane_id % 2) =
                *((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
        }

        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(&A_shmem[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_shmem_lane_addr);

        uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(&B_shmem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        LDMATRIX_X2(RB[0], RB[1], B_shmem_lane_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

        __syncthreads();
    }

    *((uint32_t *)(&C_shmem[lane_id / 4][0]) + lane_id % 4) = RC[0];
    *((uint32_t *)(&C_shmem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

    __syncthreads();

    if (lane_id < MMA_M) {
        *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&C_shmem[lane_id][0]));
    }
}

void mmaNaive(half *A, half *B, half *C, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));

    mmaNaiveKernel<<<grid, block>>>(A, B, C, M, N, K);
}
```









