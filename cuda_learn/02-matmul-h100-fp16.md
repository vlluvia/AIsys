

## version 1

* blockid, block y/x

* warpid，warp y/x

threadIdx.x / 32  
warpIdx % (BN / WN)  
warpIdx / (BN / WN)  

* Warp ITER(y, x), WSUBM, WSUBN

WNITER  
WMITER = (WM * WN) / (32 * TM * TN * WNITER)  
WSUBM = WM / WMITER  
WSUBN = WN / WNITER  

* thread in warp: id, y/x

threadIdx.x % 32  
threadIdxInWarp % (WSUBN / TN)  
threadIdxInWarp / (WSUBN / TN)  

* smem, reg

As[BM * BK], Bs[BK * BN]  
regM[WMITER * TM], regN[WNITER * TN]  

* A行起始位置，B列起始位置



* 数据加载到共享存储的索引

innerRowA/B = threadIdx.x / (BK/N / 4)  
innerColA/B = threadIdx.x % (BK/N / 4)  
rowStrideA/B = NUM_THREADS / (BK/N / 4)  


* bkidx , 0 -> K , +BK

1. load A - > Asmem  
2. load B - > Bsmem  
   
* dotidx, 0 -> BKm, +1

1. load Asmem -> Areg  
2. load Bsmem -> Breg

* matmul

wSubRowIdx, 0 - > WMITER, + 1  
wSubColIdx, 0 - > WNITER, + 1  

计算TM/TN中每个线程的结果  

threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                            (wSubColIdx * TN) + resIdxN] +=
                  __bfloat162float(regM[wSubRowIdx * TM + resIdxM]) *
                  __bfloat162float(regN[wSubColIdx * TN + resIdxN]);


* 存回HBM


## version 2 - TMA / Tensor Core

### 生成共享内存描述符

1. 编码共享内存地址到描述符的低 16 位

desc |= matrix_descriptor_encode(addr);

2. 编码矩阵的行数（16）到描述符的 16-31 位

desc |= matrix_descriptor_encode((uint64_t)16) << 16;

3. 编码矩阵的列数（1024）到描述符的 32-47 位

desc |= matrix_descriptor_encode((uint64_t)1024) << 32;

4. 设置 128B swizzle 标志位（第 62 位）

desc |= 1llu << 62; // 128B swizzle

### wgmma 

#### wgmma 大局
* asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

wgmma.fence.sync.aligned ： 用于确保在执行 WGMMA 操作之前，所有先前的内存操作（如加载数据到共享内存）已经完成  

memory：确保内存的一致性

* asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");

wgmma.commit_group.sync.aligned： 提交一组 WGMMA 操作，并等待这些操作完成

* asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");

wgmma.wait_group.sync.aligned：等待特定 WGMMA 操作组完成时调用



* Example
```
// 定义 WGMMA 操作组数量
constexpr int N = 0;

// 确保数据加载完成
warpgroup_arrive();

// 执行 WGMMA 操作
wgmma64<1, 1, 1, 0, 0>(d, &sA[0], &sB[0]);

// 提交 WGMMA 操作并等待完成
warpgroup_commit_batch();

// 等待第 N 组 WGMMA 操作完成
warpgroup_wait<N>();
```

#### wgmma64
1. 参数

ScaleD, ScaleA, ScaleB： 缩放因子

TransA, TransB： 0 表示不转置，1 表示转置


2. 生成共享内存描述符

告诉 WGMMA 指令如何访问共享内存中的数据

uint64_t desc_a = make_smem_desc(&sA[0]);

uint64_t desc_b = make_smem_desc(&sB[0]);

3. 内联汇编调用 WGMMA 指令

* 执行异步的矩阵乘加操作：

m64n64k16：表示矩阵乘法的维度（64x64x16
）

f32.bf16.bf16： 输入矩阵 A 和 B 的数据类型是 bf16，输出矩阵 d 的数据类型是 f32。

wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16

* 输出操作数

d 的 32 个元素

* 输入操作数

共享内存描述符 desc_a/desc_b

模板参数 ScaleD, ScaleA, ScaleB, TransA, TransB

```
asm volatile(
    "{\n"
    "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
    "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
    " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
    " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
    " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
    " %32,"
    " %33,"
    " %34, %35, %36, %37, %38;\n"
    "}\n"
    : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
      "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
      "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
      "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
      "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
      "+f"(d[3][6]), "+f"(d[3][7])
    : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
      "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
```





### TMA

* 全局/共享内存形状（shape）：{水平方向大小，垂直方向大小，其余维度为1 表示这是一个 2D 张量}

gmem_prob_shape[5] = {(uint64_t)BlockMinorSize*blocks_width, (uint64_t)BlockMajorSize*blocks_height, 1, 1, 1};

* 全局/共享内存步幅（stride）Swizzle：{水平方向的步幅，垂直方向的步幅，其余步幅为0表示这是一个连续的内存布局}

相邻元素之间的字节数

uint64_t gmem_prob_stride[5] = {sizeof(bf16), sizeof(bf16) * BlockMinorSize*blocks_width, 0, 0, 0};

* cuTensorMapEncodeTiled

tma_map: 输出的张量映射。

CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: 数据类型为 bfloat16。

2: 张量的维度（2D）。

gmem_address: 全局内存地址。

gmem_prob_shape: 全局内存形状。

gmem_prob_stride + 1: 全局内存步幅（跳过第一个步幅）。

smem_box_shape: 共享内存形状。

smem_box_stride: 共享内存步幅。

CU_TENSOR_MAP_INTERLEAVE_NONE: 不使用交错存储。

CU_TENSOR_MAP_SWIZZLE_128B: 使用 128B swizzle 内存布局。

CU_TENSOR_MAP_L2_PROMOTION_NONE: 不使用 L2 缓存提升。

CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE: 不填充越界数据。

```
CUresult result = cuTensorMapEncodeTiled(
    tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address, gmem_prob_shape,
    gmem_prob_stride + 1, smem_box_shape, smem_box_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
```


* 分配设备内存 并 创建 TMA

1. 创建并分配一个 CUtensorMap （gpu上）对象的内存
```
CUtensorMap *tma_map_d;
cudaMalloc(&tma_map_d, sizeof(CUtensorMap));
```


2. 创建CUtensorMap （cpu上）对象，调用 cuTensorMapEncodeTiled
```
CUtensorMap tma_map_host;
create_tensor_map<BlockMajorSize, BlockMinorSize>(&tma_map_host, src, blocks_height, blocks_width);
```

3. 将主机端张量映射复制到设备
```
cudaMemcpy(tma_map_d, &tma_map_host, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
```

### 主要逻辑


* 定义

using barrier = cuda::barrier<cuda::thread_scope_block>;

namespace cde = cuda::device::experimental;


alignas(128)：确保 sA 和 sB 的起始地址是 128 的倍数

d[WGMMA_N/16][8]：存储 wgmma 操作的结果， WGMMA_N为64
\_\_shared__ barrier barA/barB;


cde::fence_proxy_async_shared_cta(): 内存屏障, 确保在调用该函数之前的所有内存操作（如共享内存的初始化）在后续操作之前完成。

barrier: 通常用于管理异步操作（如 cp_async）的同步，而不涉及线程块内的其他操作

\_\_syncthreads(): 线程同步, 确保线程块中的所有线程在执行到该点时都达到同步状态(bar 初始化完成)，所有操作，包括共享内存的读写、寄存器操作等

```
if (threadIdx.x == 0) {
    init(&barA, blockDim.x);
    init(&barB, blockDim.x);
    cde::fence_proxy_async_shared_cta();
}
__syncthreads();
```


* barrier 大纲操作
```
barrier::arrival_token tokenA, tokenB;
for (xxxx) {
    if(threadIdx.x == 0){
        // Load data
        cde::cp_async_bulk_tensor_2d_global_to_shared(&sA[0], tensorMapA, block_k_iter*BK, num_block_m*BM, barA);
        // set tokenA/tokenB cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA));
    } else {
        tokenA/tokenB = barA/barB.arrive();
    }
    barA/barB.wait(std::move(tokenA/tokenB));
    __syncthreads();

    xxxxxxxxxx 计算操作
}
```

* load data A/B
cde::cp_async_bulk_tensor_2d_global_to_shared：

用于将全局内存中的数据异步拷贝到共享内存中。

意味着拷贝操作会在后台执行，而线程可以继续执行其他任务，直到需要同步时再等待拷贝完成。

&sA[0]：共享内存的目标地址。

tensorMapA：全局内存的 Tensor Map，描述了全局内存的布局和访问模式。

block_k_iter * BK：全局内存中的行偏移量。

num_block_m * BM：全局内存中的列偏移量。

barA：barrier 对象，用于管理异步拷贝操作的同步。


```
tokenA = cuda::device::barrier_arrive_tx(barA, 1, sizeof(sA))
```
异步操作中到达 barrier
barA：barrier 对象，用于管理同步
1：表示当前线程到达 barrier 的次数（通常为 1）。
sizeof(sA)：表示异步操作的数据大小（以字节为单位）。


* 计算操作
```
warpgroup_arrive();  // 同步 Warp Group
wgmma64<1, 1, 1, 0, 0>(d, &sA[0], &sB[0]);  // 执行 WGMMA 操作
wgmma64<1, 1, 1, 0, 0>(d, &sA[WGMMA_K], &sB[WGMMA_K]);
wgmma64<1, 1, 1, 0, 0>(d, &sA[2*WGMMA_K], &sB[2*WGMMA_K]);
wgmma64<1, 1, 1, 0, 0>(d, &sA[3*WGMMA_K], &sB[3*WGMMA_K]);
warpgroup_commit_batch();  // 提交 WGMMA 操作
warpgroup_wait<0>();  // 等待 WGMMA 操作完成
```
WGMMA_K: 16

BK: 64



* 存储数据
BM 64

BN 64

BK 64

```
 {
        int tid = threadIdx.x;
        int lane = tid % 32;            当前线程在其 warp 中的索引
        int warp = tid / 32;            当前线程所属的 warp 的索引
        uint32_t row = warp*16 + lane / 4;      当前线程在结果矩阵 C 中的行索引, warp * 16 表示每个 warp 处理 16 行, lane / 4 表示每个线程在其 warp 中处理 4 行的一部分
        bf16 *block_C = C + num_block_n*BN*M + num_block_m*BM; 指向结果矩阵 C 的起始位置

        for (int m_it = 0; m_it < BM/WGMMA_M; ++m_it) {
            for (int n_it = 0; n_it < BN/WGMMA_N; ++n_it) {
                for (int w = 0; w < WGMMA_N/16; ++w) {
                    int col = 16*w + 2*(tid % 4);    每个 bf16 元素占用 2 个字节。
                    #define IDX(i, j) ((j + n_it*WGMMA_N)*M + ((i) + m_it*WGMMA_M))

                    block_C[IDX(row, col)] = d[w][0];
                    block_C[IDX(row, col+1)] = d[w][1];
                    block_C[IDX(row+8, col)] = d[w][2];
                    block_C[IDX(row+8, col+1)] = d[w][3];
    
                    block_C[IDX(row, col+8)] = d[w][4];
                    block_C[IDX(row, col+9)] = d[w][5];
                    block_C[IDX(row+8, col+8)] = d[w][6];
                    block_C[IDX(row+8, col+9)] = d[w][7];

                    #undef IDX
                }
            }
        }
    }
```


## Version 3 - Warp group , wgmma

wgmma (16x16、32x32、64x64、128x128、192x192、256x256)


### wgmma
* wgmma256

wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16

* wgmma192
wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16

* wgmma128
wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 

* wgmma64
wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16

* wgmma32
wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16

* wgmma16
wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16


### Warp Group

* 定义 
  
1. 每个 Warp Group 负责计算的矩阵行数。

constexpr int B_WG_M = BM / (NUM_THREADS / 128);

BM：矩阵分块的行数。

NUM_THREADS：线程块中的线程总数。

128：每个 Warp Group 的线程数（通常为 128）。

2. 存储wgmma中间结果

float d[B_WG_M/WGMMA_M][WGMMA_N/16][8];

a. B_WG_M/WGMMA_M

B_WG_M：每个 Warp Group 负责计算的行数。

WGMMA_M：WGMMA 操作的行粒度（通常是 64）。

B_WG_M/WGMMA_M：每个 Warp Group 负责计算的行块数。

b. WGMMA_N/16

WGMMA_N：WGMMA 操作的列粒度（例如 64、128 等）。

16：每个 WGMMA 操作生成的列块数。

WGMMA_N/16：每个 Warp Group 负责计算的列块数。

c. 第三维：8

WGMMA 操作通常生成 8 个标量结果（例如，f32 类型的 8 个浮点数）。


3. wg_idx 

int wg_idx = threadIdx.x / 128;


4. tokenA/tokenB

tokenA/tokenB = cuda::device::barrier_arrive_tx(barA/barB, 1, BK*BM*sizeof(bf16));

三个参数是 事务大小（transaction size），它表示当前线程或 Warp 在 Barrier 中需要同步的数据量。


* 实际计算
```
#pragma unroll
for (int m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {
    bf16 *wgmma_sA = sA + BK*(m_it + wg_idx*B_WG_M/WGMMA_M)*WGMMA_M;
    #pragma unroll
    for (int k_it = 0; k_it < BK/WGMMA_K; ++k_it) {
        wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it*WGMMA_K], &sB[k_it*WGMMA_K]);
    }
}
```

1. 计算 smemA/B 的当前 Warp Group 负责的起始地址
2. 调用 WGMMA 操作，执行矩阵乘法累加计算


* 实际存储
uint32_t tid = threadIdx.x % 128;

uint32_t m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it

yo = m_it*WGMMA_M + wg_idx*B_WG_M;

#define IDX(i, j) ((j)*M + ((i) + yo))

block_C[IDX(row, col)] = d[m_it][w][0];




### other tricks

* smem 
template <int BM, int BN, int BK>
struct SMem {
    alignas(128) bf16 A[BM*BK];
    alignas(128) bf16 B[BK*BN];
};

extern __shared__ SMem<BM, BN, BK> s;




## Version 4 - 生产者和消费者 num_consumers = 1

### 定义 

* QSIZE
constexpr int num_consumers = (NUM_THREADS / 128) - 1;

constexpr int B_WG_M = BM / num_consumers;


* smem
extern \_\_shared__ \_\_align__(128) uint8_t smem[];

SMem<BM, BN, BK, QSIZE> &s = *reinterpret_cast<SMem<BM, BN, BK, QSIZE>*>(smem);


* barrier

\_\_shared__ barrier full[QSIZE], empty[QSIZE];

```
if (threadIdx.x == 0) {
    for (int i = 0; i < QSIZE; ++i) {
        init(&full[i], num_consumers * 128 + 1);
        init(&empty[i], num_consumers * 128 + 1);
    }
    cde::fence_proxy_async_shared_cta();
}
__syncthreads();
```


### 生产者和消费者

* 大纲

```
// Producer
if (wg_idx == 0) {
    constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
    if (tid == 0) {
        int qidx = 0;
        for (xxx; xxx; xxx, ++qidx) {
            if (qidx == QSIZE) qidx = 0;
            empty[qidx].wait(empty[qidx].arrive());
            xxx 数据准备
            barrier::arrival_token _ = cuda::device::barrier_arrive_tx(full[qidx], 1, (BK*BN+BK*BM)*sizeof(bf16));
        }
    }
} else {
    for (int i = 0; i < QSIZE; ++i) {
        barrier::arrival_token _ = empty[i].arrive();
    }

    xxx
    for (xxx; xxx; xxx, ++qidx) {
        if (qidx == QSIZE) qidx = 0;
        full[qidx].wait(full[qidx].arrive());

        xxx warp group compute

        barrier::arrival_token _ = empty[qidx].arrive();
    }
}


```


### other tricks




## Version 5 -  寄存器优化 / 生产者和消费者 num_consumers = 2

### Reg 寄存器优化

* 增加当前线程的寄存器使用上限为 RegCount
setmaxnreg.inc.sync.aligned.u32

* 减少当前线程的寄存器使用上限为 RegCount
setmaxnreg.dec.sync.aligned.u32

```
template <uint32_t RegCount>
__device__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}
template <uint32_t RegCount>
__device__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}
```


### 使用 Reg

```
// Producer
if(wg_idx == 0){
    constexpr int num_regs = (num_consumers <= 2 ? 24 : 32);
    warpgroup_reg_dealloc<num_regs>();
    
    xxx

} else {
    constexpr int num_regs = (num_consumers == 1 ? 256 : (num_consumers == 2 ? 240 : 160));
    warpgroup_reg_alloc<num_regs>();
    --wg_idx;        // wg_idx 的初始值为 1 或 2, 由于消费者工作组的索引需要从 0 开始（方便后续计算），因此通过 --wg_idx; 将 wg_idx 的值调整为 0 或 1。
}

```


### other tricks

* BM - > B_WG_M
```
for (int m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {
    bf16 *wgmma_sA = sA + qidx*BK*BM + BK*(m_it + wg_idx*B_WG_M/WGMMA_M)*WGMMA_M;
    #pragma unroll
    for (int k_it = 0; k_it < BK/WGMMA_K; ++k_it) {
        wgmma<WGMMA_N, 1, 1, 1, 0, 0>(d[m_it], &wgmma_sA[k_it*WGMMA_K], &sB[qidx*BK*BN + k_it*WGMMA_K]);
    }
}
```



## Version 6 - SM Schedule


### 定义

NUM_SM：SM（流式多处理器）的数量

BM（Block Rows）：表示每个线程块（Block）负责计算的矩阵块的行数

BN（Block Columns）： 表示每个线程块负责计算的矩阵块的列数

TM（Thread Rows）：表示每个线程（Thread）负责计算的子块的行数

TN（Thread Columns）：表示每个线程负责计算的子块的列数

st：当前 SM 的起始任务索引。

en：当前 SM 的结束任务索引。

* 均匀分配调度
```
template<int VERSION, int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule;

template<int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule<0, NUM_SM, BM, BN, TM, TN> {
    int st, en;

    __device__ __forceinline__ Schedule(int M, int N, int block) {
        int total_blocks = M*N/(BM*BN);                 计算总任务数 total_blocks
        int blocks_per_sm = total_blocks / NUM_SM;      计算每个 SM 的平均任务数 blocks_per_sm
        int extra_blocks = total_blocks % NUM_SM;       计算剩余任务数 extra_blocks
        if (block < extra_blocks) {                     根据当前 SM 的索引 block 分配任务范围
            st = block*(blocks_per_sm + 1);
            en = st + blocks_per_sm + 1;
        } else {
            st = extra_blocks + block*blocks_per_sm;
            en = st + blocks_per_sm;
        }
    }

    __device__ __forceinline__ int next() {             获取当前 SM 的下一个任务索引。
        if (en == st) return -1;                        检查任务是否分配完毕，如果当前 SM 的任务已分配完毕，返回 -1
        return st++;                                    返回当前任务索引并递增 st
    }
};
```

1. 计算总任务数 total_blocks
2. 计算每个 SM 的平均任务数 blocks_per_sm
3. 计算剩余任务数 extra_blocks
4. 根据当前 SM 的索引 block 分配任务范围


example

```
total_blocks = 10（总任务数）
NUM_SM = 3（SM 数量）
blocks_per_sm = 10 / 3 = 3（每个 SM 的平均任务数）
extra_blocks = 10 % 3 = 1（余数任务数）

SM 0（block = 0）：/
    block < extra_blocks 为真（0 < 1）。
    分配到 blocks_per_sm + 1 = 4 个任务。
    任务范围：[0, 4)。

SM 1（block = 1）：
    block < extra_blocks 为假（1 >= 1）。
    分配到 blocks_per_sm = 3 个任务。
    任务范围：[4, 7)。

SM 2（block = 2）：
    block < extra_blocks 为假（2 >= 1）。
    分配到 blocks_per_sm = 3 个任务。
    任务范围：[7, 10)
```


* 分块调度

```
template<int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule<1, NUM_SM, BM, BN, TM, TN> {
    int block;
    int it;
    int total_blocks_m;
    int total_blocks_n;

    __device__ __forceinline__ Schedule(int M, int N, int _block) {
        block = _block;
        it = 0;
        total_blocks_m = M/BM;
        total_blocks_n = N/BN;
        assert(total_blocks_m%TM == 0 && total_blocks_n%TN == 0);
    }

    __device__ __forceinline__ int next() {
        int num = it*NUM_SM + block;                    // 计算当前任务的全局编号 num   
        if (num >= total_blocks_m*total_blocks_n) return -1;        // 检查任务是否分配完毕
        // 计算当前任务在分块网格中的位置
        int cur_tile = num / (TM*TN);                   // 当前任务所属的“大块”索引
        int cur_tile_pos = num % (TM*TN);               // 当前任务在“大块”中的位置
        int m = TM*(cur_tile / (total_blocks_n/TN));    // 计算当前任务在矩阵中的行号和列号
        int n = TN*(cur_tile % (total_blocks_n/TN));    
        m += cur_tile_pos / TN;
        n += cur_tile_pos % TN;
        ++it;                                           // 递增任务迭代次数 it
        return m*total_blocks_n + n;                    // 返回当前任务的一维索引
    }
};
```



### 使用 Schedule

```
xxxx

Schedule<1, NUM_SM, BM, BN, 16, 8> schedule(M, N, blockIdx.x);
// Producer
if (wg_idx == 0) {
    xxx
    if (tid == 0) {
        int qidx = 0;
        for (int num_block = schedule.next(); num_block >= 0; num_block = schedule.next()) {
            int num_block_n = num_block % (N / BN);
            int num_block_m = num_block / (N / BN);
            for (xxx) {
                xxx
            }
        }
    }

} else {
    xxx
    for (int num_block = schedule.next(); num_block >= 0; num_block = schedule.next()) {
        int num_block_n = num_block % (N / BN);
        int num_block_m = num_block / (N / BN);
        for (xxx) {
        
        }
    }
}

```



### other tricks

* 直接使用 \_\_grid_constant__ CUtensorMap，将 TensorMap 存储在常量内存中，避免了动态内存分配和显式的内存拷贝。

1. 删除 allocate_and_create_tensor_map 里的 CUtensorMap *tma_map_d;
```
__host__ static inline CUtensorMap allocate_and_create_tensor_map(bf16* src, int blocks_height, int blocks_width) {
    CUtensorMap tma_map_host;
    create_tensor_map<st_rows, st_cols>(&tma_map_host, src, blocks_height, blocks_width);
    return tma_map_host;
}

```


* wgmma
wgmma256<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);

```
template<int WGMMA_N, int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ __forceinline__ void wgmma(float d[WGMMA_N/16][8], bf16* sA, bf16* sB) {
    static_assert(WGMMA_N == 32 || WGMMA_N == 64 || WGMMA_N == 128 || WGMMA_N == 192 || WGMMA_N == 208 || WGMMA_N == 256);
    if  constexpr (WGMMA_N == 256)
        wgmma256<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
    if  constexpr (WGMMA_N == 192)
        wgmma192<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
    if  constexpr (WGMMA_N == 128)
        wgmma128<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
    if constexpr (WGMMA_N == 64)
        wgmma64<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
    if constexpr (WGMMA_N == 32)
        wgmma32<ScaleD, ScaleA, ScaleB, TransA, TransB>(d, sA, sB);
}
```





## Version 7 - 5d / mbarrier

2D Tensor Map 适合传统的二维矩阵乘法，实现简单，适合常规场景。

5D Tensor Map 适合复杂的内存访问模式，尤其是对宽矩阵的分块加载，支持更高效的性能优化。

### 5d

* create_tensor_map

1. shapeA/B
uint64_t gmem_prob_shape[5] = {64, (uint64_t)global_height, (uint64_t)global_width/64, 1, 1};

将全局内存从二维（[行, 列]）扩展为五维（[D0, D1, D2, D3, D4]），以支持更复杂的分块加载。

第一维 64 表示每个分块的大小（64 个元素），这是为了与 CUDA 的硬件特性（如 Tensor Core 的 128B 对齐）对齐。

第二维 global_height 表示矩阵的行数。

第三维 global_width/64 表示将矩阵的列数分成多个 64 元素的块。

最后两维 1, 1 是占位符，表示没有额外的维度。


2. strideA/B

uint64_t gmem_prob_stride[5] = {sizeof(bf16) * global_width, 64*sizeof(bf16), 0, 0, 0};

第一维的步长是 sizeof(bf16) * global_width，表示在全局内存中，每个 64 元素块之间的跨度（即一行的大小）。

第二维的步长是 64*sizeof(bf16)，表示在全局内存中，每个 64 元素块内部的跨度。

最后三维的步长为 0，表示这些维度是连续的。




### mbarrier

Barrier（屏障）： CUDA 中的一种线程同步机制，用于确保线程块内的所有线程都到达某个同步点后才能继续执行。通常通过 __syncthreads() 或 cuda::barrier 实现。

MBarrier（内存屏障）： CUDA 9.0 引入的一种更高级的同步机制，专门用于 异步内存事务 的同步。不仅可以同步线程，还可以同步内存事务（如 cp.async 异步加载）。


* 初始化
mbarrier.init.shared::cta.b64

bar_ptr ：表示 MBarrier 的地址

thread_count + transaction_count：需要等待的线程数和内存事务数。

```
__device__ static __forceinline__ void init_barrier(uint64_t* bar, int thread_count, int transaction_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_ptr), "r"(thread_count+transaction_count)
    );
}
```


* expect_bytes
用于通知 MBarrier（内存屏障） 期望的内存事务字节数

```
__device__ static __forceinline__ void expect_bytes(uint64_t* bar, uint32_t bytes) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
    asm volatile ("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"        
        :: "r"(bar_ptr), "r"(bytes));
}
```

mbarrier.arrive.expect_tx.shared::cta.b64：通知 MBarrier 期望的内存事务字节数



* load_async

用于从全局内存中异步加载数据到共享内存，并使用 MBarrier（内存屏障） 来同步内存事务

cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes

cp.async.bulk.tensor.5d：异步加载 5D 张量数据的指令。

shared::cluster：目标地址位于共享内存中，作用域为线程块簇（cluster）。

global.tile：源地址位于全局内存中，使用 Tensor Map 描述数据布局。

mbarrier::complete_tx::bytes：使用 MBarrier 同步异步加载操作，并指定加载的字节数。


[%0] : 目标共享内存地址，由 dst_ptr 提供
[%1, {%3, %4, %5, 0, 0}]: 源全局内存地址，由 tma_ptr 提供
%3：固定为 0（未使用）。

%4：全局内存中的行索引，由 global_row_idx 提供。

%5：全局内存中的列索引除以 64，由 global_col_idx/64 提供。

最后两个维度固定为 0（未使用）

[%2] : MBarrier 的地址，由 mbar_ptr 提供

memory : 告诉编译器内联汇编代码可能会修改内存

```
__device__ static inline void load_async(bf16 *dst, void const* const src_tma_map, uint64_t* bar, int global_col_idx, int global_row_idx) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

    asm volatile (
        "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5, 0, 0}], [%2];"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
        "n"(0), "r"(global_row_idx), "r"(global_col_idx/64)
        : "memory"
    );
}
```



* wait
用于等待 MBarrier（内存屏障） 达到预期状态

```
__device__ static __forceinline__ void wait(uint64_t* bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}
```

* arrive
当前线程已经到达同步点，并释放 MBarrier
```
__device__ static __forceinline__ void arrive(uint64_t* bar, uint32_t count=1) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar)); 
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}
```


### 使用 mbarrier

full[qidx].wait(full[qidx].arrive()); - > wait(&full[qidx], p);

barrier::arrival_token _ = empty[qidx].arrive(); -> arrive(&empty[qidx], 1);



```
if (threadIdx.x == 0) {
    for (int i = 0; i < QSIZE; ++i) {
        init_barrier(&full[i], 0, 1);
        init_barrier(&empty[i], 0, num_consumers);
    }
}
__syncthreads();
```

```
while (schedule.next(num_block_m, num_block_n)) {
    for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
        if (qidx == QSIZE) { qidx = 0; p ^= 1; }
        wait(&empty[qidx], p);
        expect_bytes(&full[qidx], (BK*BN+BK*BM)*sizeof(bf16));
        load_async(&sA[qidx*BK*BM], &tensorMapA, &full[qidx], block_k_iter*BK, num_block_m*BM);
        load_async(&sB[qidx*BK*BN], &tensorMapB, &full[qidx], block_k_iter*BK, num_block_n*BN);
    }   
}
```
```
for (int i = 0; i < QSIZE; ++i) {
    barrier::arrival_token _ = empty[i].arrive();
    if (tid == 0) arrive(&empty[i], 1);
}
```



### other tricks


* 优化 schedule
```
template<int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule<1, NUM_SM, BM, BN, TM, TN> {
    int block;
    int it;
    int total_blocks_m, total_blocks_n;

    __device__ __forceinline__ Schedule(int M, int N, int _block) {
        block = _block;
        it = 0;
        total_blocks_m = CEIL_DIV(M, BM);
        total_blocks_n = CEIL_DIV(N, BN);
        assert(CEIL_DIV(M, BM)%TM == 0 && total_blocks_n%TN == 0);
    }

    __device__ __forceinline__ bool next(int &block_m, int& block_n) {
        int num = it*NUM_SM + block;
        if (num >= total_blocks_m*total_blocks_n) {return false;}
        
        int cur_tile = num / (TM*TN);
        int cur_tile_pos = num % (TM*TN);
        block_m = TM*(cur_tile / (total_blocks_n/TN));
        block_n = TN*(cur_tile % (total_blocks_n/TN));
        block_m += cur_tile_pos / TN;
        block_n += cur_tile_pos % TN;
        ++it;
        return true;
    }
};
```




## Version 8 - 集群， 3d 并支持多播（Multicast）功能

集群是 CUDA 9.0 引入的一个概念，允许在一个 GPU 上创建多个线程块（Thread Blocks）的集合，这些线程块可以协同工作，共享资源并执行更复杂的任务

### cluster

* wait_cluster

用于在集群（Cluster）内实现屏障同步。它的作用是让线程等待，直到集群中的所有线程都到达了屏障点，并且屏障的状态满足特定的条件（由 kPhaseBit 控制）

```
__device__ static __forceinline__ void wait_cluster(uint64_t* bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}
```

* load_async_multicast

从全局内存中加载数据到共享内存，并且可以将数据广播到集群（Cluster）中的多个线程块

dst：目标共享内存地址，数据将被加载到这里。

src_tma_map：源数据的张量内存访问（TMA）映射对象，描述了全局内存中的数据布局。

bar：指向共享内存中屏障对象的指针，用于同步数据加载。

global_col_idx 和 global_row_idx：全局内存中的列索引和行索引，用于定位要加载的数据。

cluster_mask：多播掩码，指定集群中哪些线程块需要接收数据。

```
__device__ static inline void load_async_multicast(bf16 *dst, void const* const src_tma_map, uint64_t* bar, int global_col_idx, int global_row_idx, uint16_t cluster_mask) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src_tma_map);
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile (
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%3, %4, %5}], [%2], %6;"
        :
        : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
        "n"(0), "r"(global_row_idx), "r"(global_col_idx/64), "h"(cluster_mask)
        : "memory"
    );
}
```


* arrive_cluster

让当前线程块（CTA）通知屏障对象，表示它已经到达了同步点，并更新屏障的状态。

```
__device__ void arrive_cluster(uint64_t* bar, uint32_t cta_id, uint32_t count=1) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n\t"
        ".reg .b32 remAddr32;\n\t"
        "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
        "mbarrier.arrive.shared::cluster.b64  _, [remAddr32], %2;\n\t"
        "}"
        :
        : "r"(smem_addr), "r"(cta_id), "r"(count));
}
```


### 使用


* 定义

\_\_cluster_dims__(CLUSTER_M * CLUSTER_N, 1, 1)

constexpr int CLUSTERS = CLUSTER_M * CLUSTER_N;

* 获取 Cluster ID
uint32_t rank;

asm volatile("mov.u32 %0, %clusterid.x;\n" : "=r"(rank) :);


*  Cluster 级别实现线程块的同步
init_barrier(&full[i], 0, 1);

init_barrier(&empty[i], 0, num_consumers*CLUSTERS);

* Schedule
Schedule<1, NUM_SM/CLUSTERS, BM*CLUSTER_M, BN*CLUSTER_N, 16/CLUSTER_M, 8/CLUSTER_N> schedule(M, N, rank);

* 获取当前线程块在 Cluster 中的逻辑坐标（rank_m 和 rank_n），并将其分解为二维坐标（行和列）

asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(rank) :);

uint32_t rank_m = rank / CLUSTER_N;

uint32_t rank_n = rank % CLUSTER_N;


* 
```
// Producer
if (wg_idx == 0){
    xxx
    if(tid == 0){
        xxx
        uint32_t col_mask = 0;               // 在多 Cluster 环境中标识特定列的所有线程块
        for (int i = 0; i < CLUSTER_M; ++i) {
            col_mask |= (1 << (i * CLUSTER_N));
        }
        xxx
        while (schedule.next(num_block_m, num_block_n)) {
            num_block_n = num_block_n * CLUSTER_N + rank_n;
            num_block_m = num_block_m * CLUSTER_M + rank_m;

            for (int block_k_iter = 0; block_k_iter < num_blocks_k; ++block_k_iter, ++qidx) {
                xxx

                if constexpr (CLUSTER_N > 1) {
                    uint32_t mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N);
                    if (rank_n == 0) {
                        load_async_multicast(&sA[qidx*BK*BM], &tensorMapA, &full[qidx], block_k_iter*BK, num_block_m*BM, mask);         异步多播加载函数
                    }
                } else {
                    load_async(&sA[qidx*BK*BM], &tensorMapA, &full[qidx], block_k_iter*BK, num_block_m*BM);
                }

                xxx
            }
        }
    }
} else {
    for (int qidx = 0; qidx < QSIZE; ++qidx) {
        if (tid < CLUSTERS) arrive_cluster(&empty[qidx], tid);
    }

    xxx

    while (schedule.next(num_block_m, num_block_n)) {
        num_block_n = num_block_n * CLUSTER_N + rank_n;
        num_block_m = num_block_m * CLUSTER_M + rank_m;

        xxx
        for (xxx) {
            if (tid < CLUSTERS) arrive_cluster(&empty[qidx], tid);
        }
    }
}

```




## Version 9 - __stwt

```
// int col = 8*w + (tid % 4);
// #define IDX(i, j) ((((i) + yo)*N/2 + (j)))
// block_C[IDX(row, col)] = __halves2bfloat162(d[m_it][w][0], d[m_it][w][1]);
// block_C[IDX(row + 8, col)] = __halves2bfloat162(d[m_it][w][2], d[m_it][w][3]);
// block_C[IDX(row, col + 4)] = __halves2bfloat162(d[m_it][w][4], d[m_it][w][5]);
// block_C[IDX(row + 8, col + 4)] = __halves2bfloat162(d[m_it][w][6], d[m_it][w][7]);
// #undef IDX
int col = w + 2*(tid % 4);
#define IDX(i, j) ((j)*M + ((i) + yo))
#define ST(i, j, v) __stwt(&block_C[IDX(i, j)], v);

ST(row+8, col, d[m_it][w/16][2]);
ST(row, col, d[m_it][w/16][0]);

ST(row+8, col+1, d[m_it][w/16][3]);
ST(row, col+1, d[m_it][w/16][1]);

ST(row+8, col+8, d[m_it][w/16][6]);
ST(row, col+8, d[m_it][w/16][4]);

ST(row+8, col+9, d[m_it][w/16][7]);
ST(row, col+9, d[m_it][w/16][5]);

#undef IDX
#undef ST
```




## Version 10 - 异步写入全局内存


* 定义
CUtensorMap d_tma_map_A;

CUtensorMap d_tma_map_B;

CUtensorMap d_tma_map_C;

```
template <int BM, int BN, int BK, int QSIZE>
struct SMem {
    alignas(128) bf16 A[BM*BK*QSIZE];
    alignas(128) bf16 B[BK*BN*QSIZE];
    alignas(128) bf16 C[BN*BM];
    alignas(8) uint64_t full[QSIZE], empty[QSIZE];
};
```

* cp.async.bulk.tensor

```
__device__ static inline void store_async(void const* dst_tma_map, bf16 *src, int global_col_idx, int global_row_idx) {
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(dst_tma_map);
    uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(src));

    asm volatile (
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%2, %3, %4}], [%1];"
        :
        : "l"(tma_ptr), "r"(src_ptr),
        "n"(0), "r"(global_row_idx), "r"(global_col_idx / 64)
        : "memory"
    );
}
```


* 加载到全局内存
```
asm volatile("cp.async.bulk.wait_group 0;");  // 等待异步操作完成

int lane = tid % 32, warp = tid / 32;
int row = warp*16 + lane / 4;
bf16* block_sC = sC + wg_idx*B_WG_M*BN;

#pragma unroll
for (int m_it = 0; m_it < B_WG_M/WGMMA_M; ++m_it) {


}
// 用于同步线程块中的线程，并异步地将共享内存中的数据写入全局内存
asm volatile("bar.sync 10, 256;\n");
if (threadIdx.x == 128) {
    store_async(&tensorMapC, (bf16*)&sC[0], num_block_m*BM, num_block_n*BN);
    // 提交异步操作组
    asm volatile("cp.async.bulk.commit_group;"); 
}

```


## Version 10 -  Hilbert调度

```

// Rotate/flip quadrant appropriately
void rot(int n, int& x, int& y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) {
            x = n-1 - x;
            y = n-1 - y;
        }
        // Swap x and y
        int t = x;
        x = y;
        y = t;
    }
}

// Convert distance along curve to (x,y) point
void d2xy(int n, int d, int& x, int& y) {
    int rx, ry, s, t = d;
    x = y = 0;
    for (s = 1; s < n; s *= 2) {
        rx = 1 & (t/2);
        ry = 1 & (t ^ rx);
        rot(s, x, y, rx, ry);
        x += s * rx;
        y += s * ry;
        t /= 4;
    }
}

void createHilbert(int M, int N, int CORES, int *space) {
    int dim = (1 << (32 - __builtin_clz(max(M, N) - 1)));
    int core = 0;
    std::vector<std::string> v(dim, std::string(dim, '.'));
    memset(space, -1, sizeof(int)*CORES*SPACE_LEN);
    int FCORES = 64;
    int total = 0;
    std::vector<std::vector<int>> pos(CORES, std::vector<int>());
    for (int i = 0; i < dim*dim; ++i) {
        int x, y;
        d2xy(dim, i, x, y);
        if (x < M && y < N) {
            assert(loc < SPACE_LEN);
            assert(v[x][y] == '.');
            v[x][y] = '*';
            ++total;
            pos[core].push_back((x << 16) | y);
            ++core;
            if (core == FCORES) {core = 0;}
        }
    }
    core = FCORES;
    for (int i = 0; i < FCORES; ++i) {
        if (pos.back().size() >= pos[0].size()-1) break;
        pos[core].push_back(pos[i].back());
        pos[i].pop_back();
        ++core;
        if (core == CORES) {core = FCORES;}
    }
    for (int i = 0; i < CORES; ++i) {
        for (int j = 0; j < pos[i].size(); ++j) {
            space[i*SPACE_LEN + j] = pos[i][j];
        }
    }
    assert(total == M*N);
}
```

```
template<int NUM_SM, int BM, int BN, int TM, int TN>
struct Schedule<2, NUM_SM, BM, BN, TM, TN> {
    int it;
    int *space;

    __device__ __forceinline__ Schedule(int M, int N, int block, int *_space) {
        it = 0;
        space = _space;
    }

    __device__ __forceinline__ bool next(int &block_m, int& block_n) {
        if (it >= SPACE_LEN) {
            return false;
        }
        int now = space[it];
        if (now == -1) {
            return false;
        }
        block_m = now >> 16;
        block_n = (now & ((1<<16)-1));
        ++it;
        return true;
    }
};
```

```
int *space;
space = (int*)malloc(sizeof(int)*NUM_SM*SPACE_LEN);
createHilbert(CEIL_DIV(M, BM*CLUSTER_M), CEIL_DIV(N, BN*CLUSTER_N), NUM_SM/CLUSTER_M/CLUSTER_N, space);
cudaCheck(cudaMalloc((void **)&_dspace, sizeof(int)*NUM_SM*SPACE_LEN));
cudaCheck(cudaMemcpy(_dspace, space, sizeof(int)*NUM_SM*SPACE_LEN, cudaMemcpyHostToDevice));
```

