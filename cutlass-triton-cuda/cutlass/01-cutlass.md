
# 基础知识

在深度学习框架中，一个非常需要解决的问题是，密集计算的性能问题，如conv2d，dense，matmul，transformers(attention)

1. 基于DSL的编译器路径，如triton，tvm script

基于DSL的编译器路径在前端写算子的时候相对比较轻松，新增算子也比较容易，作为使用者来讲，不用太多去关注内部的一些实现以及领域知识（基于DSL本身的优化能力可以满足需求为前提）
如果优化能力不足需要自己做二次开发的话，认知成本和修改量是比较大的，比如新增IR的优化pass，增加新的指令codegen等等

2. 模块化的C++模板库，如cutlass

在写算子以及新增算子的时候就不那么容易了，也许需要实现新的组件来满足自己特定的需求（如fusion kernel），但由于整个库的软件架构比较简单，所见即所得，新增一些自己的优化思路等改动相对DSL来说比较容易，也更容易debug

## MAIN
针对于密集计算的优化，都可以归类于如何优化一个矩阵乘法，Conv2d(implicit gemm)，attention(batch matmul)


## CUTLASS的实现好在哪里，为什么性能可以超越cudnn？

### 比较重要的指令介绍
ldmatrix指令：搬运4个8x8的矩阵，从shared memory 到 local(register)
mma指令，计算一个小块的矩阵乘，比如mma.m16n8k16，会完成一个rowmajor (16,16) x colmajor (8, 16)的矩阵乘法

https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

### 所有优化手段都是围绕着这两个核心指令展开的

1. bank conflict free的shared memory layout：cutlass首先提出的一种同时消除shared memory 存(global memory->shared memory)&&读(shared memory->register)的手段

2. block swizzle：这个优化对于中大型矩阵乘法比较明显，更改了发射block的顺序，以增加locality，从而提高l2cache的命中率，实现上非常简单，核心代码就是一个取余操作，但有用

3. 多级流水线(software pipeline)：2条可以不要async.copy这个指令(sm80才有的)，大于2条流水就需要了，原理上没什么，和CPU的多级流水一个道理，主要是指令的应用。


4. predicate iterator：这个是一个软件层组件写法的优化，叫predicate的原因是，这个iterator会返回一个布尔值，在gpu的指令里是一个special register，用来表示这块内存是不是需要load，这个在软件层会涉及一些优化手段，比较有趣的是会在host侧precompute了哪些下标需要load，用位运算来mask，计算开销(位运算在gpu里开销较小)和存储开销(一个byte可以存8个mask值)都很小。为什么需要让存储开销很小？因为在gpu架构里，register是很贵的，一个thread只能使用255个register，如果超出了就会存在local memory里，register读取很快，一个cycle就可以完成，而local memory就会慢非常多，register用超了会非常非常影响性能！

5. shared memory重排搬出：mma指令计算完成之后，结果是存在register里的，且register中存储的数据是不连续的(32bits连续)，原因是由于mma指令造成的，我们知道vectorize load/store会提高访存带宽，所以我们可以在shared memory里重新排序，一并搬出。但并不是什么情况下重排都是正优化，因为重排还是会增加一次shared memory store/load，比如在小channel的conv2d中，直接从register搬出到global memory性能会更好

6. cooperative fetching和vectorize load：这两个是GPU的一些基本优化方法，即尽量用更大的data type来搬运，以及尽量让一个warp里的不同线程是连续地访存同一块内存地址，原理可以参考

7.  tiling description: 提供了实例化方法，来调整block计算量和warp计算量，也就是说用模版参数来优化spacial，主要贡献在于给用户提供了一种自定义循环切分的方法，来定义循环切分的搜索空间，针对不同的workload搜索一个性能最优的选择，对于刚接触cutlass的同学而言，增加tiling description是一个比较容易的方法来提高kernel性能（因为原始的tiling确实太少了），在这个问题上多说一些，loop tiling是非常经典的编译器优化问题，目前除了polyhedral的方法以外都是tuning base的，只不过生成实例化的方法不同(预定义的options，机器学习base的搜索如基因算法(tvm-ansor)), 对于tensorcore的矩阵乘法优化问题，我们是有强先验的，即tensorcore的计算访存指令都是固定的(mma, ldmatrix)，相当于looptiling的子问题是一个确定的解，所以搜索空间并不会特别大(但也很大了。。)


