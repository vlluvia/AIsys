

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
3. load Asmem -> Areg
4. load Bsmem -> Breg

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
m64n64k16：表示矩阵乘法的维度（64x64x16）
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



