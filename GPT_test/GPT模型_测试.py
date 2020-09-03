from mindspore import Tensor, nn
import mindspore
import numpy as np
import mindspore.ops.operations as P
from 词向量印刻 import 词向量印刻
from GPT单元 import 解码层, 全连接层


def 创建_上三角_遮罩(尺寸):
    上三角_遮罩= np.triu(np.ones((1, 尺寸, 尺寸)), k=1)
    上三角_遮罩= (上三角_遮罩 != 0)
    return 上三角_遮罩


def 创建_遮罩(目标_数组,头数):

    if 目标_数组 is not None:
        目标_遮罩 = (目标_数组 != -1)
        目标_遮罩 = 目标_遮罩[:, None, :]
        尺寸 = 目标_数组.shape[1]
        上三角_遮罩 = 创建_上三角_遮罩(尺寸)
        目标_遮罩 = 目标_遮罩 & 上三角_遮罩
        目标_遮罩 = 目标_遮罩.astype('float32')
        目标_遮罩=目标_遮罩*(-1e9)
        目标_遮罩=目标_遮罩[:, None, :, :]
        目标_遮罩=[目标_遮罩 for i in range(头数)]
        目标_遮罩=np.concatenate(目标_遮罩, axis=1)
    else:
        目标_遮罩 = None
    return  目标_遮罩


class 多层解码(nn.Cell):
    def __init__(self, 词库总数, 向量维度, 层数, 头数, 丢弃率,辞数,最大长度=1024):
        super(多层解码, self).__init__()
        self.N = 层数
        self.embed = 词向量印刻(词库总数, 向量维度)
        self.embedP = 词向量印刻(最大长度, 向量维度)
        self.decoders = nn.CellList([解码层(向量维度, 头数, 丢弃率) for i in range(层数)])
        self.norm = nn.LayerNorm((向量维度,), epsilon=1e-6)
        a = [i for i in range(辞数)]
        b = np.array(a).reshape(1, 辞数)
        self.po = Tensor(b, mindspore.int32)
        self.dropout = nn.Dropout(1 - 丢弃率)

    def construct(self, 输入,  输入_遮罩):
        x = self.embed(输入)
        长=P.Shape()(x)[1]
        a = [i for i in range(长)]
        b = np.array(a).reshape(1, 长)
        po = Tensor(b, mindspore.int32)
        x = x + self.embedP(po)
        x = self.dropout(x)
        for i in range(self.N):
            x = self.decoders[i](x,  输入_遮罩)
        x = self.norm(x)
        return x

class 输出函数_GPT(nn.Cell):
    def __init__(self, 词库总数, 向量维度, 层数, 头数, 丢弃率,辞数=666):
        super(输出函数_GPT, self).__init__()
        self.Transformer = Transformer(词库总数, 向量维度, 层数, 头数, 丢弃率,辞数)
        输入 = np.ones((1, 辞数))
        MAKS1 = 创建_遮罩(输入, 12)
        self.MAKS1 = Tensor(MAKS1)

    def construct(self, 输入):
        输出 = self.Transformer(输入, self.MAKS1)
        return 输出

    def 前向(self, 输入, 输入_遮罩=None):
        输出 = self.Transformer(输入, 输入_遮罩)
        return 输出


class Transformer(nn.Cell):
    def __init__(self,  词库总数, 向量维度, 层数, 头数, 丢弃率, 辞数):
        super(Transformer, self).__init__()

        self.d_model = 向量维度
        self.trg_vocab = 词库总数
        self.decoder = 多层解码(词库总数, 向量维度, 层数, 头数, 丢弃率, 辞数)
        self.out = 全连接层(向量维度, 词库总数)
        self.shape = P.Shape()
        self.reshape = P.Reshape()

    def construct(self, 输入, 输入_遮罩):
        bs = self.shape(输入)[0]
        d_output = self.decoder(输入,  输入_遮罩)
        d_output = self.reshape(d_output, (-1, self.d_model))
        output = self.out(d_output)
        输出 = self.reshape(output, (bs, -1, self.trg_vocab))
        return 输出

