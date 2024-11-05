
#   Ring Attention


![alt text](../img/cuda-mode/lc-13-image-1.png)

![alt text](../img/cuda-mode/lc-13-image-2.png)


## 多模态任意到任意（Any-to-Any）自回归预测模型

![alt text](../img/cuda-mode/lc-13-image-3.png)

LWM (Large World Model)
LLaVA

![alt text](../img/cuda-mode/lc-13-image-4.png)


### 处理长上下文（Long Context）时的注意力机制方法

1. 近似法（如Sparse注意力、LoRA）
2. RAG / 向量数据库（使用ANN搜索、LSH）
3. 暴力计算（如tiling、blockwise方法


### Vanilla Attention（原始注意力机制）

![alt text](../img/cuda-mode/lc-13-image-5.png)


![alt text](../img/cuda-mode/lc-13-image-6.png)


### 模型大小和上下文长度对每个token的FLOPS（浮点运算次数）缩放的影响

![alt text](../img/cuda-mode/lc-13-image-7.png)


![alt text](../img/cuda-mode/lc-13-image-8.png)

![alt text](../img/cuda-mode/lc-13-image-9.png)



### 计算Softmax的挑战


FlashAttention和RingAttention算法中应用Softmax，必须“分块”或“在线”地计算Softmax，即只处理部分和，这样可以更高效地计算出结果。

![alt text](../img/cuda-mode/lc-13-image-10.png)



* 通过Python中的PyTorch库定义和验证一个简单的Softmax函数
![alt text](../img/cuda-mode/lc-13-image-11.png) 

Naive & Numerical unstable”（朴素且数值不稳定）
当前定义的朴素Softmax函数在某些输入情况下会出现问题

![alt text](../img/cuda-mode/lc-13-image-12.png)

* 通过“sum exp”（指数和）撤销Softmax的归一化，从而将分块计算的结果合并

![alt text](../img/cuda-mode/lc-13-image-13.png)
![alt text](../img/cuda-mode/lc-13-image-14.png)

* 如何使用数值稳定的方式将分块的Softmax结果进行合并
![alt text](../img/cuda-mode/lc-13-image-15.png)
![alt text](../img/cuda-mode/lc-13-image-16.png)
![alt text](../img/cuda-mode/lc-13-image-17.png)
* 并逐步过渡到Log-Sum-Exp的更新


* trick
1. RingAttention 可以使用内部 Flash Attention 的一些函数，这些函数可以返回 log-sum-exp，从而帮助进行逐块或者增量地计算注意力Value的投影
2. Flash Attention V2的逐chunk更新softmax结果和输出，实际上也适用于这里的Ring Attention的更新
3. zhuzilin/ring-flash-attention中对Ring Attention的开源实现，我没可以看到除了通信之外Ring Attention调用的是TriDao的Flash Attention来做每个块（设备）上的Attention计算和lse的更新


### 序列并行


![alt text](../img/cuda-mode/lc-13-image-18.png)

每个设备分别计算一部分注意力值，并通过 Send & Recv KV 操作在设备间进行通信，从而实现跨设备的高效并行计算

#### Ring attention

![alt text](../img/cuda-mode/lc-13-image-19.png)
Ring Attention的伪代码  
![alt text](../img/cuda-mode/lc-13-image-20.png)


### 自回归模型（Autoregressive Models）中的因果掩码（Causal Masking）的概念和作用

![alt text](../img/cuda-mode/lc-13-image-21.png)

![alt text](../img/cuda-mode/lc-13-image-22.png)



#### 自回归模型中使用Ring Attention时遇到的主要问题及其影响

![alt text](../img/cuda-mode/lc-13-image-23.png)

![alt text](../img/cuda-mode/lc-13-image-24.png)

![alt text](../img/cuda-mode/lc-13-image-25.png)

![alt text](../img/cuda-mode/lc-13-image-26.png)


> 上面讲的都是Ring Attention的负载不均衡问题，接下来介绍个解决方案。


#### Stripe Permutation（条带置换）


![alt text](../img/cuda-mode/lc-13-image-27.png)
![alt text](../img/cuda-mode/lc-13-image-28.png)

通过 Stripe Permutation（条带置换） 的策略，将K，V和Q在序列维度上按条带重新排列（比如将KV0分成了0,4,8,12，而不是连续的0,1,2,3），通过重新排列KV和Q块，Striped Attention能够更好地分配计算资源，从而减轻设备之间的不平衡性，提高整体计算效率。从第二张Slides可以看到，经过条带置换后的计算过程几乎能够完美地均衡分配计算负载，从而使得设备之间的计算更加平衡，避免了Ring Attention中存在的设备空闲问题。在每个回合中，只有当“host_id < round”时，需要丢弃第一个查询和最后一个键的计算，这样做能够避免不必要的计算，进一步提升效率。




## FlashAttention 和 Flash-Decoding 两种不同的方法在长文本推理任务中的表现差异


![alt text](../img/cuda-mode/lc-13-image-29.png)

![alt text](../img/cuda-mode/lc-13-image-30.png)


![alt text](../img/cuda-mode/lc-13-image-31.png)


# Today

![alt text](../img/cuda-mode/lc-13-image-32.png)

