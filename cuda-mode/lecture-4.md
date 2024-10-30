

# PMPP 第4-5章 计算和内存基础


## GPU 架构

![alt text](../img/cuda-mode/lc-4-image-1.png)

* RTX 3090有82个流式多处理器（SM, Streaming Multiprocessor），每个SM包含多个RT Core（光线追踪核心）和Tensor Core（张量核心）。所有SM共用L2缓存。
* GA102 GPU实际上有168个FP64单元（每个SM两个），但Slides中未显示。FP64的TFLOP（每秒浮点运算次数）速率是FP32的1/64。包含少量FP64硬件单元是为了确保任何包含FP64代码的程序都能正确运行，包括FP64 Tensor Core代码。
> GA：代表 "Graphics Ampere"，指的是 NVIDIA 的 Ampere 架构。102：是这个特定 GPU 型号的数字标识符。通常，较高的数字表示更高端或更大规模的 GPU 设计。GA102 被用于多款显卡，包括 GeForce RTX 3090, RTX 3080 和一些 Quadro 系列专业卡。

![alt text](../img/cuda-mode/lc-4-image-2.png)


### NVIDIA GA10x GPU架构中的流式多处理器(Streaming Multiprocessor, SM)的结构和特性

* SM结构
  * 4个处理单元，每个包含FP32（单精度浮点）和INT32（整数）运算单元
  * 每个处理单元有一个第三代Tensor Core
  * 寄存器文件（16,384 x 32位）
  * L0 I-Cache和Warp调度器
  * 128KB的L1数据缓存/共享内存
  * 第二代RT Core（光线追踪核心）
* 线程块分配
  * 一个线程块被分配给一个SM
  * 每个SM最多可分配1536个线程
  * 无法控制网格中的哪个块分配到哪里（Hopper+架构可以有线程块组）
* Warp执行
  * 4个warp或"部分warp"可以在一个周期内计算
  * 这些warp共享一条指令（Volta+架构每个线程都有程序计数器）
* 计算单元
  * 32个FP32单元（这32个FP32单元对应一个warp的32个线程，在任何给定的时钟周期，32个FP32单元可以同时处理一个warp中的32个线程）
  * 其中16个同时支持INT32运算
* 寄存器
  * 16k个32位寄存器在同一块上调度的任务之间共享
* 缓存和共享内存
  * L1缓存和共享内存共享128KB硬件
  * 共享内存可以配置为0/8/16/32/64/100KB
  * L1缓存使用剩余空间（至少28KB）

### CUDA编程中的线程(Threads)、线程束(Warps)和线程块(Blocks)的概念和关系

![alt text](../img/cuda-mode/lc-4-image-3.png)

* CUDA内核启动：指定块布局（每个块中的线程数）；指定网格布局（要启动的块数）
* 一个线程块内的线程：同一块内的线程在同一个流式多处理器(SM)上并行执行（可以访问SM的共享内存）
* 除了及其新的GPU，块之间完全独立；CUDA可以自由地将块分配给SM；块的执行顺序是随机的
* 一个线程块在SM上运行时被划分为32线程的线程束；每个线程束在SM的固定处理单元上运行；同时分配给处理单元的所有线程束轮流执行，但寄存器状态保持不变（这里应该指的是线程束切换的时候可以保留寄存器状态，例如当一个线程束暂停执行让位于另一个线程束时，它的寄存器状态会被保存。当这个线程束再次获得执行时间时，它可以从之前的状态继续执行，而不需要重新初始化。）；
* 在AMD硬件和术语中，线程束称为Wavefronts，默认大小为64？
* 右侧图表展示了线程块如何分配到不同的SM上。

### CUDA中线程的线性化和分组为线程束（warps）的过程
![alt text](../img/cuda-mode/lc-4-image-4.png)

使用T(x,y,z)表示线程索引，其中x、y、z表示三个维度的索引。将多维的线程索引转换为一维的线性索引的公式为：threadId = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)。线性化后的线程被分组为32个线程一组的线程束，图底部显示了线程如何被分组成连续的线程束。

![alt text](../img/cuda-mode/lc-4-image-6.png)
![alt text](../img/cuda-mode/lc-4-image-5.png)

这个kernel的目的是为3D空间中的每个点计算其在warp内的32个"邻居"的索引。它利用了CUDA的warp级别shuffle操作来高效地在线程间交换数据。输出是一个5D张量，维度为(8, 8, 8, 32, 3)，其中：

* 前三个维度(8, 8, 8)对应3D空间中的点
* 32表示每个点计算32个邻居
* 3表示每个邻居的x、y、z坐标

![alt text](../img/cuda-mode/lc-4-image-7.png)



