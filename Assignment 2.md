# Assignment 2

以实现此作业（不抄袭ai直接答案，不直接询问作业问题，遇到问题细化到知识点与类比补充知识）为目的

​	**在深度学习领域为什么使用Batch Normalization ，应用场景，解决了什么问题，直观比喻例子解释，最简单的代码输出比喻**

1. 为什么使用Batch Normalization？

###  直观比喻例子

想象你在一家工厂的流水线上生产蛋糕：

- **问题**：每个工人（神经网络层）需要面粉、鸡蛋等原料（输入数据），但每天原料的质量和数量都在变化（输入分布变化）。有的工人拿到特别多的面粉，有的拿到很少，导致蛋糕质量不稳定（训练不稳定）。

- **Batch Normalization的作用**：BN就像一个“标准化车间”，在每道工序前，把原料按比例调整到标准量（均值0，方差1），然后根据需要加点“调味料”（可学习的缩放和偏移参数），确保每个工人拿到的原料都差不多。这样，工人能更高效地工作（加速训练），蛋糕质量也更稳定（模型性能提升）。

- **正则化效果**：由于每天原料略有不同（mini-batch的统计噪声），工人会稍微调整做法，相当于增加了一些随机性（类似Dropout的正则化效果）。

  

```
text原始原料（输入数据）：
tensor([[ 0.1234, -1.2345,  0.5678,  2.3456],
        [-0.9876,  0.4567, -0.2345, -1.1234],
        ...])  # 随机值，分布不稳定
标准化后的原料（BN输出）：
tensor([[ 0.1000, -1.0000,  0.5000,  1.8000],
        [-0.8000,  0.4000, -0.2000, -0.9000],
        ...])  # 均值约0，方差约1，分布更稳定
```

### 2. 应用场景

- **卷积神经网络（CNN）**：BN广泛用于图像分类、目标检测、图像分割等任务。例如，ResNet、Inception等经典CNN架构中都使用了BN。
- **循环神经网络（RNN）**：在某些NLP任务中，BN可以用于稳定RNN的输入或隐藏状态。
- **生成对抗网络（GAN）**：BN帮助稳定GAN的训练，减少模式崩塌。
- **任何深层网络**：只要网络层数较多，BN通常能带来性能提升，尤其在计算机视觉和自然语言处理任务中。

**数据特征之间不相关，且均值为零、[方差]([方差_百度百科](https://baike.baidu.com/item/方差/3108412))为单位。在训练神经网络时，我们可以在数据输入网络前进行预处理，以明确消除特征间的关联性。**

```
假设你有一组成绩：80、82、78、81、79 它们都在80左右，波动不大 → 方差小。

但如果是：60、90、75、85、70 虽然平均值可能差不多，但波动很大 → 方差大。
```



![]()

##  Batch Normalization 中的求导示例

Batch Normalization 的前向传播：

python

```
# 前向传播
μ = (1/m) * Σ xᵢ                    # 均值
σ² = (1/m) * Σ (xᵢ - μ)²           # 方差
x̂ᵢ = (xᵢ - μ) / √(σ² + ε)          # 归一化
yᵢ = γ * x̂ᵢ + β                    # 缩放和偏移
```



### 反向传播求导过程

**已知**：上游梯度 ∂L/∂yᵢ

**目标**：计算 ∂L/∂xᵢ, ∂L/∂γ, ∂L/∂β

1. 计算 ∂L/∂β

text

```
∂L/∂β = Σᵢ (∂L/∂yᵢ) * (∂yᵢ/∂β)
      = Σᵢ ∂L/∂yᵢ * 1
      = Σᵢ ∂L/∂yᵢ
```



2. 计算 ∂L/∂γ

text

```
∂L/∂γ = Σᵢ (∂L/∂yᵢ) * (∂yᵢ/∂γ)
      = Σᵢ ∂L/∂yᵢ * x̂ᵢ
```



3. 计算 ∂L/∂x̂ᵢ

text

```
∂L/∂x̂ᵢ = (∂L/∂yᵢ) * (∂yᵢ/∂x̂ᵢ)
       = ∂L/∂yᵢ * γ
```



4. 计算 ∂L/∂σ²

text

```
∂L/∂σ² = Σᵢ (∂L/∂x̂ᵢ) * (∂x̂ᵢ/∂σ²)
       = Σᵢ ∂L/∂x̂ᵢ * (xᵢ - μ) * (-1/2) * (σ² + ε)^(-3/2)
       = -1/2 * (σ² + ε)^(-3/2) * Σᵢ ∂L/∂x̂ᵢ * (xᵢ - μ)
```



5. 计算 ∂L/∂μ

text

```
∂L/∂μ = Σᵢ (∂L/∂x̂ᵢ) * (∂x̂ᵢ/∂μ) + (∂L/∂σ²) * (∂σ²/∂μ)

∂x̂ᵢ/∂μ = -1/√(σ² + ε)
∂σ²/∂μ = -2/m * Σᵢ (xᵢ - μ)

∂L/∂μ = Σᵢ ∂L/∂x̂ᵢ * (-1/√(σ² + ε)) 
       + ∂L/∂σ² * (-2/m * Σᵢ (xᵢ - μ))
```



6. 计算 ∂L/∂xᵢ

text

```
∂L/∂xᵢ = (∂L/∂x̂ᵢ) * (∂x̂ᵢ/∂xᵢ) + (∂L/∂σ²) * (∂σ²/∂xᵢ) + (∂L/∂μ) * (∂μ/∂xᵢ)

∂x̂ᵢ/∂xᵢ = 1/√(σ² + ε)
∂σ²/∂xᵢ = 2/m * (xᵢ - μ)
∂μ/∂xᵢ = 1/m

∂L/∂xᵢ = ∂L/∂x̂ᵢ * (1/√(σ² + ε))
        + ∂L/∂σ² * (2/m * (xᵢ - μ))
        + ∂L/∂μ * (1/m)
```



4. 实际代码实现

python

```
import numpy as np

def batchnorm_backward(dout, cache):
    """
    dout: 上游梯度 ∂L/∂y, shape (N, D)
    cache: 前向传播时保存的中间变量
    """
    x, x_norm, mean, var, gamma, beta, eps = cache
    N, D = dout.shape
    
    # 1. 计算 ∂L/∂β 和 ∂L/∂γ
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    
    # 2. 计算 ∂L/∂x̂
    dx_norm = dout * gamma
    
    # 3. 计算 ∂L/∂σ²
    dvar = np.sum(dx_norm * (x - mean) * -0.5 * (var + eps)**(-1.5), axis=0)
    
    # 4. 计算 ∂L/∂μ
    dmean1 = np.sum(dx_norm * -1 / np.sqrt(var + eps), axis=0)
    dmean2 = dvar * np.sum(-2 * (x - mean), axis=0) / N
    dmean = dmean1 + dmean2
    
    # 5. 计算 ∂L/∂x
    dx1 = dx_norm / np.sqrt(var + eps)
    dx2 = dvar * 2 * (x - mean) / N
    dx3 = dmean / N
    dx = dx1 + dx2 + dx3
    
    return dx, dgamma, dbeta
```

![](./Assignment%202.assets/image-20251012150330777.png)

![](./Assignment%202.assets/image-20251012150344755.png)

**问![image-20251014163117939](./Assignment%202.assets/image-20251014163117939.png)

第一项 n * dout *gamma  第二项 

![image-20251014163908101](./Assignment%202.assets/image-20251014163908101.png)

第二项   np.sum(dout *gamma ,axis= o)

第三项    x_hat * np.sum(dx_hat * x_hat, axis=0)

![image-20251014164331428](./Assignment%202.assets/image-20251014164331428.png)

![image-20251014165039400](./Assignment%202.assets/image-20251014165039400.png)

running_mean = momentum * running_mean + (1 - momentum) * sample_mean

​    running_var = momentum * running_var + (1 - momentum) * sample_var

什么意思

***变量传输有问题     不要钻牛角尖数学推导过程   理解抽象过程以及伪代码     把公式翻译正确代码。***

```python

    sample_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    sample_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))
     格式化
    

running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
    公式
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
       存储
```



## Fully Connected Networks with Batch Normalization

{affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

第一步：计算    

第二步：让它们大小差不多（均值0，方差1）[batch/layer norm]*可选*

**第三步：放大缩小魔法（ReLU）**

- 有一个“能量开关”，只让正数的积木通过，负数的积木变成0。

**第四步（可选）：随机休息（dropout）***  *可选*

- 有时候，魔法师会让一些积木“休息”（随机置0）

**重复：L−1L-1L−1 次**

- 除了最后一关，其他关卡都会重复这个“计算-整理-放大-休息”的流程

**最后一步：猜答案（affine - softmax）**

- 最后一关再算一次（affine），然后用“选答案魔法”（softmax）把结果变成概率，

作业代码变量与架构  

每一层的权重矩阵 `W` 的形状都是 `(输入维度, 输出维度)`

第 1 步：理清网络结构和维度

第 2 步：构建一个完整的维度列表 📏



为了在代码中轻松地处理这个维度传递，一个非常聪明的技巧是先把所有层的维度放在一个列表里。

这个列表应该包含：**输入维度 +所有隐藏层维度 + 输出维度**。

根据 `input_dim`, `hidden_dims`, `num_classes` 这三个变量，如何创建一个像 `[D, H1, H2, C]` 这样的完整维度列表

我们的目标是把输入维度、所有隐藏层的维度和最终的输出（分类数）维度整合到一个列表里，方便后续的循环操作

将输入维度、隐藏层维度列表、输出维度拼接成一个完整的列表

```
dims = [input_dim] + hidden_dims + [num_classes]
```



**题词更新：（不要直接输出代码答案，而是描述代码输入，输出，以及逻辑与框架，分步让我学会，提示需要用到的函数，使用简单解释功能）**

```python
dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
          input_dim_of_layer = dims[i]
          output_dim_of_layer = dims[i+1]
          key_W = f'W{i+1}'
          self.params[key_W] = np.random.randn(input_dim_of_layer, output_dim_of_layer) * weight_scale

          key_b = f'b{i+1}'
          self.params[key_b] = np.zeros(output_dim_of_layer)
          if self.normalization == 'batchnorm' and i < self.num_layers - 1:
            gamma = f'gamma{i+1}'
            self.params[gamma] = np.ones(output_dim_of_layer,)

            beta = f'beta{i+1}'
            self.params[beta] = np.zeros(output_dim_of_layer,)
```

loss

