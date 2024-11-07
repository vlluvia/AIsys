
# flash attention v2

https://github.com/Dao-AILab/flash-attention

flash attention v1计算速度提升2至4倍，但是它的整体吞吐量仍然较低
在A100上，FlashAttention v1的forward吞吐量只有GPU极限的30%至50%；而backward甚至只有25%至35%。


## 优化

### 算法（Algorithm）
FlashAttention v2在纯算法层面对前向过程和反向过程都做了细微的改进。我们先来看前向过程。

* 前向过程
![Alt text](img/attention/flash-attention-v2/image-1.png)

分析
![Alt text](img/attention/flash-attention-v2/image-2.png)

结论
![Alt text](img/attention/flash-attention-v2/image-3.png)
直到处理完最后一个分块后，直接用此时的全局EXP求和项来做分母即可。

例子
![Alt text](img/attention/flash-attention-v2/image-4.png)

* Causal Masking的简单优化
![Alt text](img/attention/flash-attention-v2/image-5.png)

* 反向过程
![Alt text](img/attention/flash-attention-v2/image-6.png)


### 并行（Parallelism）

#### FlashAttention v1的并行策略
![Alt text](img/attention/flash-attention-v2/image-7.png)

#### FlashAttention v2的并行策略

![Alt text](img/attention/flash-attention-v2/image-8.png)
![Alt text](img/attention/flash-attention-v2/image-9.png)


### 计算分片（Work Partitioning）

![Alt text](img/attention/flash-attention-v2/image-10.png)

#### FlashAttention v1
![Alt text](img/attention/flash-attention-v2/image-11.png)

#### FlashAttention v2
![Alt text](img/attention/flash-attention-v2/image-12.png)




# 引用
https://arxiv.org/pdf/2307.08691
https://zhuanlan.zhihu.com/p/642962397

 