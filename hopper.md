# hopper

1. 数据局部性
2. 异步执行

* Hopper 架构针对这两个关键因素，在软件层提供了两方面的编程能力
1. 新线程、显存层次：通过新增 Thread Block Cluster 这一线程层次，提供跨 Thread Block 的 Shared Memory 访问。开发者可以基于 Thread Block Cluster ，利用 Distributed Shared Memory 实现高效的多 Thread Block 的协同运行；
2. 访存计算异步执行：Hopper 在硬件层提供了 TMA 单元，在软件层可以通过 cuda::memcpy_async 使用 TMA 单元实现异步的 Global Memory 和 Shared Memory 之间的拷贝。


## Thread Block Clusters 的前世今生

H800为例，其 Global Memory 的访问延迟约为 478 个时钟周期，Shared Memory 的访问延迟约为 30 个时钟周期，Register 约为 1 个时钟周期

* GPU 编程中，Kernel 的设计是以 Thread Block 这个粒度展开的。但这样会导致两个问题：

1. 单个 Thread Block 处理的数据规模有限
Shared Memory 的容量有限
H800为例，其Global Memory 大小为 80GB，而每个 Thread Block 最大可用 Shared Memory 仅 227 KB
这使得单个 Thread Block 为了使用 Shared Memory加速性能时，只能处理一个数据规模较小的子任务。
任务规模一旦变大，Shared Memory 不够用，Thread Block 就只能用高访问延迟的 Global Memory 完成任务，导致 Kernel 性能降低。

2. SM 利用率较低
单个 Thread Block 可配置的最大线程数为 1024，每个 Thread Block 会分配到一个 SM 上运行
假如每个 Thread Block 处理较大规模的数据、计算，Kernel 一次仅发射很少的 Thread Block，可能导致某些 SM 处于空闲状态，计算资源没有被充分挖掘，这样同样会限制 Kernel 的整体性能。例如在 LLM 长文本推理 进行 Decoding Attention 时， K、V 长度较长，此时由于显存上限问题， batch size 会小，这导致单个 Thread Block 访问的数据量、计算量较大，同时发射的 Thread Block 的数量较少，导致某些 SM 处于空闲状态，限制 Kernel 性能。


解决这个问题的最直接的方式是：提供更大粒度的线程组。


###  Thread Block Clusters

为了解决 Thread Block 粒度过小导致的 Kernel 运行效率不足的问题，Hooper 在 Thread Block 之上再引入一层结构——Thread Block Clusters。

![Alt text](image-hopper-1.png)

