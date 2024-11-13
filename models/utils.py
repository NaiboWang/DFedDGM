"""
---
title: Utility functions for DDPM experiment
summary: >
  Utility functions for DDPM experiment
---

# Utility functions for [DDPM](index.html) experiemnt

这个函数 gather 是一个采用 PyTorch 框架的 Python 函数，它执行了两个主要操作：1) 使用 PyTorch 的 gather 方法收集张量里的特定值，2) 然后改变收集到的值的形状（或称维度）。我们可以逐步地解释每一部分：

参数: - consts: 一个 torch.Tensor 对象，这个张量包含一些常数值，可能是一维或多维的。 - t: 一个 torch.Tensor 对象，这个张量包含索引值，用来从 consts 张量中选择元素。t 的类型和形状应当与 consts 张量兼容，且用于索引的最后一个维度。

函数体:

c = consts.gather(-1, t) 这行代码中的 gather 是 PyTorch 中的一个方法，用来从输入张量 consts 中，按照指定轴（维度）和索引张量 t 来收集数据。对于输入 gather 的参数 -1，意味着操作是在 consts 的最后一个维度上进行索引。换言之，这将从 consts 每个最后一维的 slice 中，根据 t 的对应值来选择元素。

return c.reshape(-1, 1, 1, 1) 这行代码用 reshape 方法改变了张量 c 的形状。这个方法将 c 重塑为一个四维张量。其中 -1 表示不改变该维度的元素总数（也就是自动计算这个维度的大小），后面三个 1 表示其他三个维度的大小。这种形状改变对于匹配到某些特别的数据结构，比如需要四维张量的卷积层，特别有用。

简而言之，这个函数的作用是：首先选择 consts 张量中，由张量 t 指定的元素，并将结果张量的形状重塑为四维，其中第一个维度是元素总数，其他三个维度都是1。这个函数可能用于为神经网络层准备特定的参数或权重。

"""
import torch.utils.data


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

"""
# 创建一个 3x3 的二维张量
consts = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

# 创建一个索引张量，对于 consts 的每一行，我们将选取一个元素
# 索引张量的维度应与我们想要选择的维度大小相匹配
indices = torch.tensor([[0, 1, 2],
                        [1, 0, 1],
                        [2, 2, 0]])

# 使用 gather 方法沿着列的方向（dim=1）来选择元素
# 对于每一行，indices 指定了应该选取的列
gathered = consts.gather(1, indices)

print(gathered)
 
现在，我们逐行解释这个示例：

consts 是一个 3x3 的 tensor。
indices 也是一个 3x3 的 tensor，其值表示要从 consts 的每一行中提取的列索引。
执行 consts.gather(1, indices) 时，我们告诉 PyTorch ，我们想在 consts 的第二个维度（列，dim=1）上进行操作，并按 indices 的值提取，所以我们得到以下结果：

# 假设输出如下：
# tensor([[1, 2, 3],
#         [5, 4, 5],
#         [9, 9, 7]])
 
第一行的输出 [1, 2, 3] 是因为 indices 第一行 [0, 1, 2] 表示从 consts 的第一行中选取第 0 个、第 1 个和第 2 个元素。
第二行和第三行的选择逻辑相同。
注意到，使用 gather 时，indices 的形状必须和 consts 在没有进行操作的维度上相同，或者这些维度上的值是 1。这是因为 gather 按照广播规则来处理不同形状的 tensor。

"""
