from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
import mindspore.dataset as mds
from mindspore import context, nn
from GPT模型 import 输出函数_GPT
import os.path
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from 生成midspore训练数据 import json到minecord
def create_dataset(base_path, batch_size, num_epochs):
    path = base_path
    dataset = mds.MindDataset(path, columns_list=["input", "output"], num_parallel_workers=4)
    dataset = dataset.shuffle(buffer_size=dataset.get_dataset_size())
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.repeat(count=num_epochs)
    return dataset
#注意！！这里的训练的文本长度为666 这在数据预处理时已经定了，可利用预处理的代码重定文本长度。预处理的代码和数据放在一起。
#以下是模型参数
文本长度 = 666   #  固定长度就不用每次训练都生成mask
丢弃率 = 0.0
词库总数 = 50001  # 组成是UTF8单字符和部分英文字母组合请务必保留此数，如需更改则需要重改json查询表并重新生成训练数据
向量维度 = 768
层数 = 12
头数 = 12


#________________________以下是Adam优化算法的参数
学习率 = 6.25e-5
beta1 = 0.9
beta2=0.98
eps=1e-09


# 每一步耗时约1.5秒
epoch = 3
batch_size = 2
原始文件路径 = '../data/ROCStories_训练和测试/ROCStories/ROCStories_dev.json'
过渡文件路径 = '../data/ROCStories_训练和测试/ROCStories/ROCStories_dev.npz'
最终文件名 = 'ROCStories_dev'
预训练数据路径='../data/checkpoint/ROCStories/checkpoint_ROCStories-1_5227.ckpt'  #请确保有预训练的文件
if os.path.isfile("../data/mindrecord/"+最终文件名+".minecord.db") and os.path.isfile("..data/mindrecord/"+最终文件名+".minecord"):
    print("发现mindrecord类型训练数据，直接加载训练。")
    dataset = create_dataset("../data/mindrecord/"+最终文件名+".minecord", batch_size, 1)
else:
    print("未发现mindrecord类型训练数据，正在生成，请稍等......")

    json到minecord(原始文件路径, 过渡文件路径, 最终文件名)
    dataset = create_dataset("../data/mindrecord/"+最终文件名+".minecord", batch_size, 1)

device_target = "GPU"
context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
network = 输出函数_GPT(词库总数, 向量维度, 层数, 头数, 丢弃率, 文本长度)
if os.path.isfile(预训练数据路径):

    param_dict = load_checkpoint(预训练数据路径)
    load_param_into_net(network, param_dict)
net_loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
net_opt = nn.Adam(network.trainable_params(), learning_rate=学习率, beta1=beta1, beta2=beta2, eps=eps)
time_cb = TimeMonitor(data_size=dataset.get_dataset_size())
config_ck = CheckpointConfig(save_checkpoint_steps=200, keep_checkpoint_max=1)
ckpoint_cb = ModelCheckpoint(prefix="checkpoint_" + 最终文件名, directory="../data/checkpoint/"+最终文件名, config=config_ck)
model = Model(network, net_loss, net_opt, {'acc': Accuracy()})
print("开始训练单步时长1.5秒")
model.train(epoch=epoch, train_dataset=dataset, callbacks=[time_cb, ckpoint_cb, LossMonitor()],
            dataset_sink_mode=False)