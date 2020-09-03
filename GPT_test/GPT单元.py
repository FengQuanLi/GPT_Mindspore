import mindspore.ops.operations as P
import mindspore.nn as nn
import mindspore
from mindspore import Tensor
import numpy as np

class 前向传播网络(nn.Cell):
    """
    与transformer不同的是把激活Relu换替换为Gelu。
    """
    def __init__(self, 输入_接口, 输出_接口=2048, 丢弃率=0.1):
        super(前向传播网络, self).__init__()
        self.linear_1 = 全连接层(输入_接口, 输出_接口)
        self.gelu = P.Gelu()
        self.linear_2 = 全连接层(输出_接口, 输入_接口)
        self.Dropout = nn.Dropout(1-丢弃率)

    def construct(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.Dropout(x)
        x = self.linear_2(x)
        return x


class 多头_注意力(nn.Cell):
    """
    核心中的核心值得进一步研究。

    """
    def __init__(self, 头数, 尺寸, 丢弃率=0.1):
        super(多头_注意力, self).__init__()
        self.d_model = 尺寸
        self.d_k = 尺寸 // 头数
        self.d_k_Tensor = Tensor(尺寸 // 头数, mindspore.float32)
        self.h = 头数
        self.q_linear = 全连接层(尺寸, 尺寸)
        self.v_linear = 全连接层(尺寸, 尺寸)
        self.k_linear = 全连接层(尺寸, 尺寸)
        self.dropout = nn.Dropout(1-丢弃率)
        self.out = 全连接层(尺寸, 尺寸)
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.shape = P.Shape()

        self.batch_matmul = P.BatchMatMul()
        self.add = P.TensorAdd()
        self.transpose = P.Transpose()
        self.sqrt = P.Sqrt()
        self.softmax = P.Softmax(-1)

    def construct(self, 输入, 遮罩=None):
        bs = self.shape(输入)[0]
        查询向量 = self.reshape(self.q_linear(self.reshape(输入, (-1, self.d_model))), (bs, -1, self.h, self.d_k))
        键向量 = self.reshape(self.k_linear(self.reshape(输入, (-1, self.d_model))), (bs, -1, self.h, self.d_k))
        值向量 = self.reshape(self.v_linear(self.reshape(输入, (-1, self.d_model))), (bs, -1, self.h, self.d_k))

        查询向量 = self.transpose(查询向量, (0, 2, 1, 3))
        键向量 = self.transpose(键向量, (0, 2, 1, 3))
        值向量 = self.transpose(值向量, (0, 2, 1, 3))
        #-----------------------------------------------------------------
        键向量_转置 = self.transpose(键向量, (0, 1, 3, 2))
        关连度 = self.batch_matmul(查询向量, 键向量_转置)
        关连度 = 关连度 / self.sqrt(self.d_k_Tensor)

        if 遮罩 is not None:
            关连度 = self.add(关连度, 遮罩)  # 遮罩目前来看很重要，否则训练不出理想结果
        关连度 = self.softmax(关连度)
        关连度 = self.dropout(关连度)  # 粗暴丢弃似乎并非一个好方法
        输出 = self.batch_matmul(关连度, 值向量)

        #-----------------------------------------------------------------
        输出 = self.transpose(输出, (0, 2, 1, 3))
        输出 = self.reshape(输出, (-1, self.d_model))
        输出 = self.out(输出)
        输出 = self.reshape(输出, (bs, -1, self.d_model))
        return 输出

class 解码层(nn.Cell):
    """
    GPT核心是transformer的解码部分作细微改动而成。

    """
    def __init__(self, 尺寸, 头数, 丢弃率=0.1):
        super(解码层, self).__init__()
        self.norm_1 = nn.LayerNorm((尺寸,), epsilon=1e-6)
        self.d_model = 尺寸
        self.norm_3 = nn.LayerNorm((尺寸,), epsilon=1e-6)

        self.dropout_1 = nn.Dropout(1-丢弃率)
        self.dropout_3 = nn.Dropout(1-丢弃率)

        self.attn_1 = 多头_注意力(头数, 尺寸, 丢弃率=丢弃率)
        self.ff = 前向传播网络(尺寸, 丢弃率=丢弃率)
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, 输入, 目标_遮罩):
        x = self.norm_1(输入)
        x = self.attn_1(x, 目标_遮罩)
        x = self.dropout_1(x)
        x = 输入 + x
        x1 = self.norm_3(x)
        bs = self.shape(x1)[0]
        x1 = self.reshape(x1, (-1, self.d_model))
        x1 = self.ff(x1)
        x1 = self.reshape(x1, (bs, -1, self.d_model))
        输出 = x + self.dropout_3(x1)
        return 输出

class 全连接层(nn.Cell):
    """
    替代mindspore的nn.Dense。
    可用numpy的随机数种子生成固定的参数以便对比研究。
    """
    def __init__(self, 输入_接口, 输出_接口):
        super(全连接层, self).__init__()
        np.random.seed(1)  # 可以生成固定参数以便对比研究，不需要固定参数注释掉此行就可。
        self.weight = mindspore.Parameter(Tensor(np.random.uniform(-1/np.sqrt(输入_接口), 1/np.sqrt(输入_接口),(输入_接口, 输出_接口)), mindspore.float32), "w")
        self.bias = mindspore.Parameter(Tensor(np.random.uniform(-1/np.sqrt(输入_接口), 1/np.sqrt(输入_接口),输出_接口), mindspore.float32), "b")
        self.MatMul=P.MatMul()

    def construct(self, x):
        输出=self.MatMul(x, self.weight)
        输出=输出+self.bias
        return 输出