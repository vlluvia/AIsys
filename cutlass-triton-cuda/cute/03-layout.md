# layout

Layout 是一个数据排列的描述体系，其可以实现将逻辑坐标映射到索引坐标（offset表示）。Layout包含Shape和Stride两部分。其中Shape描述排列的分块层次和结构。Stride描述块内或块间的数据排列连续性。Shape和Stride都是层级的嵌套表示。也就是说Shape可以包含Int数据和Shape。Shape和Stride需要有相同的层次关系。

* 一维向量的表示

![alt text](../../img/cutlass-cute-nccl-cuda/cute/image-laout-1.png)


![alt text](../../img/cutlass-cute-nccl-cuda/cute/image-laout-2.png)

![alt text](../../img/cutlass-cute-nccl-cuda/cute/image-laout-3.png)

![alt text](../../img/cutlass-cute-nccl-cuda/cute/image-laout-4.png)


* 二维矩阵的表示

![alt text](../../img/cutlass-cute-nccl-cuda/cute/image-laout-5.png)

![alt text](../../img/cutlass-cute-nccl-cuda/cute/image-laout-6.png)


* 有层次的Layout（Heriarchy Layout）
![alt text](../../img/cutlass-cute-nccl-cuda/cute/image-laout-7.png)


* 常量Shape（编译时Shape）
> Int<K>{}

```
auto shape = make_shape(Int<2>{}, Int<3>{});
auto shape1 = make_shape(shape, Int<3>{});
```

* 变量Shape（运行时Shape）
```
auto shape = make_shape(2, 3);
auto shape = make_shape(m, n);
```


## Layout 的代数和几何解释

### 基本属性
* shape为((2, 4), (3, 5)), stride为((3, 6), (1, 24))的Layout示意图
![alt text](../../img/cutlass-cute-nccl-cuda/cute/image-laout-8.png)

![alt text](../../img/cutlass-cute-nccl-cuda/cute/image-laout-9.png)


### 坐标（coordinate）

```
auto row_coord = make_coord(1, 3); 
auto col_coord = make_coord(2, 4); 
auto coord = make_coord(row_coord, col_coord);
```

coord: ((1, 3), (2, 4))所表示的位置和访问层次
![alt text](../../img/cutlass-cute-nccl-cuda/cute/image-laout-10.png)


### 切片（slice）
auto layout_out = slice(coord, layout_in);
* Layout的切片运算示意图
![alt text](../../img/cutlass-cute-nccl-cuda/cute/image-laout-11.png)


### 补集（complement）

![alt text](../../img/cutlass-cute-nccl-cuda/cute/image-laout-12.png)


### 乘法（product）

* 约定了各个乘法的顺序和层级如下
![alt text](../../img/cutlass-cute-nccl-cuda/cute/image-laout-13.png)


Layout乘法的几何解释

![alt text](../../img/cutlass-cute-nccl-cuda/cute/image-laout-14.png)

### 除法（divide）

![alt text](../../img/cutlass-cute-nccl-cuda/cute/image-laout-15.png)


### 复合函数和逆（composition & inverse）


![alt text](../../img/cutlass-cute-nccl-cuda/cute/image-laout-16.png)



