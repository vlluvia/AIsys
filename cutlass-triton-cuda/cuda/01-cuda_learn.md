

## cuda常见库


* #include <cuda_runtime.h>
这是CUDA运行时API的头文件，包含了所有与CUDA设备交互的函数和类型定义。例如，内存分配、数据传输、内核启动等操作都需要使用这个头文件中的函数。
* #include <helper_functions.h>
这是一个辅助函数库的头文件，通常包含了一些通用的实用函数和计时函数。它可能包括了cuda.h和cuda_runtime_api.h，以及其他一些有用的工具函数。
* #include <helper_cuda.h>
这是CUDA辅助函数库的头文件，包含了一些用于检查CUDA错误和处理CUDA设备的辅助函数。这些函数可以帮助你更方便地进行CUDA编程，减少错误处理的工作量。

* cudaGetDeviceCount(&num_gpus);

* #include <cassert>
#include <cassert>
assert(grid < N);

* AtomicIntrinsics
```
// Atomic addition
atomicAdd(&g_odata[0], 10);

// Atomic subtraction (final should be 0)
atomicSub(&g_odata[1], 10);

// Atomic exchange
atomicExch(&g_odata[2], tid);

// Atomic maximum
atomicMax(&g_odata[3], tid);

// Atomic minimum
atomicMin(&g_odata[4], tid);

// Atomic increment (modulo 17+1)
atomicInc((unsigned int *)&g_odata[5], 17);

// Atomic decrement
atomicDec((unsigned int *)&g_odata[6], 137);

// Atomic compare-and-swap
atomicCAS(&g_odata[7], tid - 1, tid);

// Bitwise atomic instructions

// Atomic AND
atomicAnd(&g_odata[8], 2 * tid + 7);

// Atomic OR
atomicOr(&g_odata[9], 1 << tid);

// Atomic XOR
atomicXor(&g_odata[10], tid);
```

* L2的访问策略窗口
Samples\0_Introduction\simpleAttributes\simpleAttributes.cu
允许开发者显式地控制哪些内存区域应该被持久化到L2缓存中，从而优化内存访问性能。
```
cudaAccessPolicyWindow initAccessPolicyWindow(void) {
  cudaAccessPolicyWindow accessPolicyWindow = {0};
  accessPolicyWindow.base_ptr = (void *)0;
  accessPolicyWindow.num_bytes = 0;
  accessPolicyWindow.hitRatio = 0.f;
  accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  return accessPolicyWindow;
}

cudaStreamAttrValue streamAttrValue;

streamAttrValue.accessPolicyWindow = initAccessPolicyWindow();
accessPolicyWindow = initAccessPolicyWindow();
```

* 协作组和屏障
Samples\0_Introduction\simpleAWBarrier.cu
使用CUDA的协作组和屏障来实现高效的向量归一化。通过使用协作组和屏障，代码能够在块内和块间同步线程，从而实现高效的并行计算


* 流和回调





## CUDA Thrust库
== c++ STL




