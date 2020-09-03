import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
import mindspore.nn as nn
import mindspore
import numpy as np
class Embedding2(nn.Cell):
    """
    从mindspore的源码nn.Embedding抄过来的，做了细微的改动。
    可用numpy的随机数种子生成固定的参数以便对比研究。
    """
    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal', dtype=mstype.float32):
        super(Embedding2, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.use_one_hot = use_one_hot
        np.random.seed(1)  # 可以生成固定参数以便对比研究，不需要固定参数注释掉此行就可。
        self.embedding_table = Parameter(Tensor(np.random.uniform(0, 1, (vocab_size, embedding_size)),mindspore.float32),
                                         name='embedding_table')

        self.dtype = dtype
        self.expand = P.ExpandDims()
        self.reshape_flat = P.Reshape()
        self.shp_flat = (-1,)
        self.gather = P.GatherV2()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, self.dtype)
        self.off_value = Tensor(0.0, self.dtype)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.get_shp = P.Shape()

    def construct(self, ids):
        extended_ids = self.expand(ids, -1)
        out_shape = self.get_shp(ids) + (self.embedding_size,)
        flat_ids = self.reshape_flat(extended_ids, self.shp_flat)

        if self.use_one_hot:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)

        output = self.reshape(output_for_reshape, out_shape)
        return output

    def extend_repr(self):
        s = 'vocab_size={}, embedding_size={},' \
            'use_one_hot={}, ' \
            'embedding_table={}, dtype={}'.format(
                self.vocab_size,
                self.embedding_size,
                self.use_one_hot,
                self.embedding_table,
                self.dtype)
        return s

class 词向量印刻(nn.Cell):
    """
    Embedding的另一个马甲,训练的时候刻上去，用的时候印出来。
    """
    def __init__(self, 词库总数, 向量维度):
        super(词向量印刻, self).__init__()
        self.d_model = 向量维度
        self.embed = Embedding2(词库总数, 向量维度)

    def construct(self, 词编号集):
        return self.embed(词编号集)

