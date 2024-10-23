
# Zero

## DP
![alt text](img/zero/image.png)

## DDP
> 通信隐藏在计算过程中
![alt text](img/zero/image-2.png)


## Zero 1
![alt text](img/zero/image-3.png)

![alt text](img/zero/image-4.png)

![alt text](img/zero/image-5.png)


## Zero2
![alt text](img/zero/image-6.png)

## Zero3

![alt text](img/zero/image-7.png)


## Zero ++
> 减少Zero 3跨服务器通信量
1. 每个服务器有完整的模型参数
2. 基于块的量化，模型参数从FP16转换成INT8
3. 分层ALLToALL解决量化问题

![alt text](img/zero/image-8.png)

![alt text](img/zero/image-9.png)

![alt text](img/zero/image-10.png)

