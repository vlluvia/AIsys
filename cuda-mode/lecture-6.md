
# 如何优化PyTorch中的优化器

## 讲述了运行时间（runtime）和内存使用（memory usage）之间的权衡关系
![Alt text](../img/cuda-mode/lc-6-image-1.png)


## Fusion, the high level idea


展示了一个naive的优化器实现，核心要点是假设有M个参数，对于每个参数有N个操作，那么遍历所有参数并处理完共需要M * N个操作。
![Alt text](../img/cuda-mode/lc-6-image-2.png)

介绍了一种称为"horizontally fused optimizer"（水平融合优化器）的优化方法，可以把naive的优化器实现中的for循环fuse掉。

![Alt text](../img/cuda-mode/lc-6-image-3.png)

实际上我们可以把整个优化器的操作fuse成一个cuda kernel
![Alt text](../img/cuda-mode/lc-6-image-4.png)

![Alt text](../img/cuda-mode/lc-6-image-5.png)
核心信息是：在CUDA编程中，通过减少kernel启动的次数可以提高程序的执行效率。这是因为每次启动CUDAkernel都会有一定的开销，如果能够将多个操作合并到更少的kernel中，就可以减少这些开销，从而提高整体性能。水平融合和垂直融合是实现这一目标的两种主要策略：水平融合合并了相似的并行操作；垂直融合则进一步合并了不同的计算步骤。


## Fusion, the nitty gritty

![Alt text](../img/cuda-mode/lc-6-image-6.png)
类比了线粒体是细胞的能量工厂，而multi_tensor_apply是高速优化器的"动力卡车"。展示了一辆装载多辆小汽车的大卡车，暗示multi_tensor_apply可以同时处理多个张量。说明multi_tensor_apply允许我们对张量列表进行操作，而不是单个张量。
![Alt text](../img/cuda-mode/lc-6-image-7.png)
对比了普通的torch.add操作（左侧小车+小卡车）和_foreach_add操作（右侧大卡车装载多辆小车）。

_foreach_add操作
如何在CUDA中实现一个用于多个张量的add操作（_foreach_add）时输入应该怎么如何传递。
![Alt text](../img/cuda-mode/lc-6-image-8.png)
* vector
尝试使用std::vector<float*>来实现_foreach_add_kernel，这种方法不行，因为CUDA不识别std::vector。
![Alt text](../img/cuda-mode/lc-6-image-9.png)

* C-style
尝试使用C风格的数组（float**）来实现_foreach_add_kernel，结论：这种方法也不行，会导致非法内存访问（IMA），因为外层指针*是CPU地址。
![Alt text](../img/cuda-mode/lc-6-image-10.png)

* pass by chonky boi（通过大块数据传递）
实现多张量操作（specifically _foreach_add）的第三种尝试方法，称为"pass by chonky boi"（通过大块数据传递）
![Alt text](../img/cuda-mode/lc-6-image-11.png)

![Alt text](../img/cuda-mode/lc-6-image-12.png)

![Alt text](../img/cuda-mode/lc-6-image-13.png)
尝试上面的大块数据传递方式之后作者碰到了CUDA中的非法内存访问。问题似乎与张量列表的大小（N）有关。在N=423和N=424之间存在一个临界点，可能与CUDA的内存管理或某些硬件限制有关。

![Alt text](../img/cuda-mode/lc-6-image-14.png)
这里继续说明了当尝试传递大量数据（在这里是张量地址）作为kernel参数时，可能会超出CUDAkernel参数空间的4KB限制，导致程序失败。这就解释了为什么只有当NUM_TENSORS小于某个特定值（这里提到424）时，代码才能正常工作。

* 方案一
提出了"Attempt 4"（第四次尝试）的解决方案，建议通过多次启动kernel来解决问题，即"make more trips"（多次运输）。
![Alt text](../img/cuda-mode/lc-6-image-15.png)

* 方案二
展示了当前的方法是进行水平融合（Horizontal Fusion），将多个操作合并到一个kernel中，但实际上常常会产生多个水平融合的kernel和垂直融合的kernel。
![Alt text](../img/cuda-mode/lc-6-image-16.png)

* 方案三
总结了最终的解决方案，提出了结构体（struct）和memcpy的混合使用策略。左侧：如果数据量较小，符合kernel参数空间限制，就直接使用结构体传递。右侧：如果数据量超过限制，则使用memcpy将数据复制到GPU内存，然后传递指针。
![Alt text](../img/cuda-mode/lc-6-image-17.png)


### torch.compile()
* torch.compile()的主要优势是垂直融合（vertical fusion）。图示展示了如何将多个水平融合（horizontal fusion）的操作进一步垂直融合成一个更大的操作。
![Alt text](../img/cuda-mode/lc-6-image-18.png)
* 展示了如何在优化器中使用torch.compile()
![Alt text](../img/cuda-mode/lc-6-image-19.png)
* 展示了torch.compile()生成的Triton kernel的一部分代码。这是一个大型的、高度优化的kernel，包含了许多临时变量（tmp0, tmp1等）和复杂的数学运算。这说明torch.compile()确实可以生成非常复杂和高效的fuse kernel。
![Alt text](../img/cuda-mode/lc-6-image-20.png)



### 总结

![Alt text](../img/cuda-mode/lc-6-image-21.png)