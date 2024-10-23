
# Megatron-LM
https://github.com/NVIDIA/Megatron-LM
PTD 并行

![alt text](img/llm-meg/image-5.png)
## DP
![alt text](img/llm-meg/image.png)

## TP
> TP张量并行被用于 intra-code transformer层
> 消耗大量带宽，使用nvlink
![alt text](img/llm-meg/image-2.png)

* 并行原理
> Matmul 与 transformer 应用 

Matual
![alt text](img/llm-meg/image-6.png)
![alt text](img/llm-meg/image-7.png)
![alt text](img/llm-meg/image-8.png)
多次连续矩阵乘
![alt text](img/llm-meg/image-9.png)

transformer : MLP
![alt text](img/llm-meg/image-10.png)
![alt text](img/llm-meg/image-11.png)
![alt text](img/llm-meg/image-12.png)
* TP具体实现
1. ColumnParallelLinear
![alt text](img/llm-meg/image-14.png)
![alt text](img/llm-meg/image-15.png)
2. RowParallelLinear
![alt text](img/llm-meg/image-16.png)
![alt text](img/llm-meg/image-17.png)
* Transformer
1. Embedding 并行
2. layerNorm 并行
3. Attention 并行
![alt text](img/llm-meg/image-13.png)
4. Add & Norm结构
5. mlp并行




## PP
> PP张量并行被用于 intra-code transformer层
> 带宽占用少，集群多级多卡

* 基本原理
1. BUbble 空泡率影响性能
2. PP 切分到不同卡

* Gpipe
* PipeDream 1F1B
* Interleavel 1F1B

![alt text](img/llm-meg/image-3.png)

![alt text](img/llm-meg/image-4.png)

