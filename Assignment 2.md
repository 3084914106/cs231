# Assignment 2

## BatchNormalization

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

代码作业

`batchnorm_forward` **内实现批量归一化的前向传播**

想象你在学校组织一个合唱团，学生们的声音大小（音量）都不一样：

- 有的学生唱得特别响（数据值很大）。
- 有的学生唱得特别轻（数据值很小）。
- 还有的学生声音忽大忽小（数据分布不稳定）

**具体步骤**（以合唱团为例）：

1. **测量平均音量**：听听所有学生的声音，算出平均音量（就像算数据的平均值）。
2. **测量声音差异**：看看每个人的声音和平均音量差多少（就像算数据的方差）。

调整音量：把每个人的声音调到差不多大小（让数据的均值变成0，方差变成1）

比如：音量 [10, 20, 30] 离平均值20的差异是 [-10, 0, 10]，方差是 ((-10)² + 0² + 10²)/3 = 66.67。

**标准化**：把每个人的音量调整到“平均0，差异1”的标准。

- 公式：新音量 = (原音量 - 平均值) / √(方差 + 小数字)
- 小数字（比如0.00001）是为了避免除以0。

```
import numpy as np

# 模拟10个学生的音量（数据）
volumes = np.array([10, 20, 30, 15, 25, 5, 35, 12, 18, 22])

# Batch Normalization的简单实现
def batch_norm(volumes):
    # 1. 算平均音量
    avg_volume = np.mean(volumes)  # 平均值
    print(f"平均音量: {avg_volume}")

    # 2. 算音量差异（方差）
    variance = np.mean((volumes - avg_volume) ** 2)  # 方差
    print(f"音量差异: {variance}")

    # 3. 标准化：让音量均值为0，方差为1
    normalized_volumes = (volumes - avg_volume) / np.sqrt(variance + 0.00001)  # 加小数字避免除以0
    print(f"标准化后的音量: {normalized_volumes}")

    # 4. 个性化调整（用gamma=1, beta=0模拟简单情况）
    gamma = 1  # 缩放
    beta = 0   # 偏移
    final_volumes = gamma * normalized_volumes + beta
    print(f"最终输出: {final_volumes}")

    return final_volumes

# 运行
print("原始音量:", volumes)
batch_norm(volumes)
```

原始音量: [10 20 30 15 25  5 35 12 18 22]
平均音量: 19.2
音量差异: 79.76
标准化后的音量: [-1.03  0.09  1.22 -0.47  0.65 -1.59  1.78 -0.81 -0.14  0.31]
最终输出: [-1.03  0.09  1.22 -0.47  0.65 -1.59  1.78 -0.81 -0.14  0.31]

**训练时**：BN用当前一小批数据（mini-batch）的均值和方差来整理数据，同时慢慢更新一个“长期平均值”作为备用。

**测试时**：不再用每批数据的均值和方差，而是用训练时记下的“长期平均值”来整理数据。

**指数衰减的移动平均**：这是一种“记住过去但更关注现在”的方法，用来更新长期平均值。

#### 测试时：用长期记录来调整

- 演出当天（测试时），你没有时间去算当天学生的音量（因为没有mini-batch）。所以，你用之前训练时记下的“长期平均音量”和“长期音量差异”来调整每个人的音量，确保演出效果和训练时差不多。

```python
import numpy as np

# 模拟3天的音量数据（3个mini-batch，每个batch有3个学生）
data_day1 = np.array([10, 20, 30])  # 第一天音量
data_day2 = np.array([15, 25, 35])  # 第二天音量
data_day3 = np.array([12, 22, 32])  # 第三天音量

# BN的简单实现
def batch_norm_train(data, running_mean, running_var, momentum=0.9):
    # 1. 算当天均值和方差
    mean = np.mean(data)  # 平均音量
    var = np.var(data)    # 音量差异
    print(f"当天均值: {mean}, 方差: {var}")

    # 2. 标准化数据（均值0，方差1）
    normalized_data = (data - mean) / np.sqrt(var + 0.00001)  # 加小数字避免除以0
    print(f"标准化后的数据: {normalized_data}")

    # 3. 更新长期记录（指数衰减）
    running_mean = momentum * running_mean + (1 - momentum) * mean
    running_var = momentum * running_var + (1 - momentum) * var
    print(f"更新后的长期均值: {running_mean}, 长期方差: {running_var}")

    return normalized_data, running_mean, running_var

# 初始化长期记录
running_mean = 0
running_var = 1

# 模拟3天训练
print("第一天:")
normalized_data, running_mean, running_var = batch_norm_train(data_day1, running_mean, running_var)
print("\n第二天:")
normalized_data, running_mean, running_var = batch_norm_train(data_day2, running_mean, running_var)
print("\n第三天:")
normalized_data, running_mean, running_var = batch_norm_train(data_day3, running_mean, running_var)

# 测试时：用长期记录
print("\n测试时：用长期均值和方差来标准化")
test_data = np.array([14, 24, 34])
normalized_test = (test_data - running_mean) / np.sqrt(running_var + 0.00001)
print(f"测试数据: {test_data}")
print(f"标准化后的测试数据: {normalized_test}")
```

```
第一天:
当天均值: 20.0, 方差: 66.66666666666667
标准化后的数据: [-1.22474487  0.          1.22474487]
更新后的长期均值: 2.0, 长期方差: 6.666666666666667

第二天:
当天均值: 25.0, 方差: 66.66666666666667
标准化后的数据: [-1.22474487  0.          1.22474487]
更新后的长期均值: 4.3, 长期方差: 13.333333333333334

第三天:
当天均值: 22.0, 方差: 66.66666666666667
标准化后的数据: [-1.22474487  0.          1.22474487]
更新后的长期均值: 5.87, 长期方差: 19.666666666666668

测试时：用长期均值和方差来标准化
测试数据: [14 24 34]
标准化后的测试数据: [ 1.84967629  4.08500399  6.3203317 ]
```

错误点   

测试阶段的 batch normalization 不再使用当前 batch 的统计量，而是使用训练阶段累积的 **running_mean** 和 **running_var**。这些是为了让模型在推理时保持稳定。



训练阶段写好了方差，平均值。测试阶段直接使用



> 训练阶段的变量怎么传输过去的呢，又不是全局变量

通过 `bn_param` 这个字典在函数之间传递的

```
running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))
```

# Backward Pass

反向传播（神经网络学习的“纠错”过程）中的一个关键点：有些中间结果（数据）会被多个后续步骤用到，当我们计算“纠错信号”（梯度）时，需要把这些后续步骤的反馈加起来。

**“Intermediates”**：神经网络计算过程中的中间数据，就像游戏中的“道具”或“分数”。

**“Multiple outgoing branches”**：这些中间数据可能会被多个后续关卡（层）使用，就像一个道具被好几个队友拿去用。

**“Sum gradients across these branches”**：当队友们反馈这个道具的“问题”时，你需要把所有队友的反馈加起来，才能知道这个道具到底要怎么改进。

黑话    奖池还在积累

###  简单点说

- 神经网络里有些数据（中间结果）会被多个后续层用，就像一个道具被好几个关卡用。
- 在反向传播时，这些后续层会各自送回“纠错信号”（梯度）。
- 你得把所有这些信号加起来，告诉前面的层“这个数据要怎么改”，不然只改一部分，效果就不对。

比喻学习

演出结果不好（有误差），观众给反馈（dout），你需要反过来算：

- 每个人的原始音量（输入 xxx）要怎么改（dx）？
- 放大器 γ要怎么调（dgamma）？
- 高低调整器 β 要怎么调（dbeta）

![image-20251011201220959](F:\items\cs231\assets\image-20251011201220959.png)

前向 均值 方差  归一   **缩放和平移** 
$$
𝑑
𝑜
𝑢
𝑡
=
∂
𝐿/
∂
𝑦
$$
依据dout修改  gamma beta x

![image-20251011205233902](F:\items\cs231\assets\image-20251011205233902.png)

损失函数 L 依赖于输出 yi，所以 ∂L/∂β需要通过 yi回推

![image-20251011210843421](F:\items\cs231\assets\image-20251011210843421.png)

L为差值      

β 像一个“全局开关”，它对每个人的 yi 都有同样的影响（加一个固定值）。

观众的反馈 douti是每个人的声音需要调整的量。

因为 β同时影响所有人，把所有 douti加起来，就能知道 β 整体应该调多大（正数表示调高，负数表示调低）

求偏导（里面的）   再用链式法则串起来

![image-20251011213420256](F:\items\cs231\assets\image-20251011213420256.png)

![image-20251011222234913](F:\items\cs231\assets\image-20251011222234913.png)

**对谁求偏导，就只动谁，其他量当常数**。

**如果有多个路径依赖，梯度要相加**。

**保持形状一致**：矩阵/向量情况要注意广播和 sum。

**计算图思维**：每个节点写局部导数，最后链式相乘、相加。

![image-20251011224512938](F:\items\cs231\assets\image-20251011224512938.png)

![image-20251011224535482](F:\items\cs231\assets\image-20251011224535482.png)

隐函数求导口诀   画关系图，找出每一个因子影响到的底层

均值、方差以及偏移（通常指的是Batch Normalization中的参数）以及使用链式法则求偏导的过程

```
# 前向传播
μ = (1/m) * Σ xᵢ                    # 均值
σ² = (1/m) * Σ (xᵢ - μ)²           # 方差
x̂ᵢ = (xᵢ - μ) / √(σ² + ε)          # 归一化
yᵢ = γ * x̂ᵢ + β                    # 缩放和偏移
```

   + ```
        ∂L/∂β = Σᵢ (∂L/∂yᵢ) * (∂yᵢ/∂β)
              = Σᵢ ∂L/∂yᵢ * 1
              = Σᵢ ∂L/∂yᵢ
        
        ∂L/∂γ = Σᵢ (∂L/∂yᵢ) * (∂yᵢ/∂γ)
              = Σᵢ ∂L/∂yᵢ * x̂ᵢ
        
        ∂L/∂x̂ᵢ = (∂L/∂yᵢ) * (∂yᵢ/∂x̂ᵢ)
               = ∂L/∂yᵢ * γ
        
        ∂L/∂σ² = Σᵢ (∂L/∂x̂ᵢ) * (∂x̂ᵢ/∂σ²)
               = Σᵢ ∂L/∂x̂ᵢ * (xᵢ - μ) * (-1/2) * (σ² + ε)^(-3/2)
               = -1/2 * (σ² + ε)^(-3/2) * Σᵢ ∂L/∂x̂ᵢ * (xᵢ - μ)
        
        ∂L/∂μ = Σᵢ (∂L/∂x̂ᵢ) * (∂x̂ᵢ/∂μ) + (∂L/∂σ²) * (∂σ²/∂μ)
        
        ∂x̂ᵢ/∂μ = -1/√(σ² + ε)
        ∂σ²/∂μ = -2/m * Σᵢ (xᵢ - μ)
        
        ∂L/∂μ = Σᵢ ∂L/∂x̂ᵢ * (-1/√(σ² + ε)) 
        
           + ∂L/∂σ² * (-2/m * Σᵢ (xᵢ - μ))
        
        ∂L/∂xᵢ = (∂L/∂x̂ᵢ) * (∂x̂ᵢ/∂xᵢ) + (∂L/∂σ²) * (∂σ²/∂xᵢ) + (∂L/∂μ) * (∂μ/∂xᵢ)
        
        ∂x̂ᵢ/∂xᵢ = 1/√(σ² + ε)
        ∂σ²/∂xᵢ = 2/m * (xᵢ - μ)
        ∂μ/∂xᵢ = 1/m
        
        ∂L/∂xᵢ = ∂L/∂x̂ᵢ * (1/√(σ² + ε))
        
           + ∂L/∂σ² * (2/m * (xᵢ - μ))
             /∂μ * (1/m)
        ```

        

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

