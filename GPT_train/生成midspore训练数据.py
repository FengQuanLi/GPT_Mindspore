from 取训练数据 import 读取训练数据_A, 读出引索, 生成训练用numpy数组_A,生成训练用numpy数组_B
import os.path
import numpy as np
from mindspore.mindrecord import FileWriter

def 数据预处理_json到minecord():

    路径 = "../data/训练材料_英语3.json"
    输入表单 = 读取训练数据_A(路径)
    词_数表路径 = "../data/词_数50000.json"
    数_词表路径 = "../data/数_词50000.json"

    if os.path.isfile(词_数表路径) and os.path.isfile(数_词表路径):
        词_数表, 数_词表 = 读出引索(词_数表路径, 数_词表路径)


        numpy数组路径 = "../data/训练材料_英语3.npz"
        if os.path.isfile(numpy数组路径):
            npz文件 = np.load(numpy数组路径, allow_pickle=True)
            输出np, 输入np = npz文件["输出np"], npz文件["输入np"]
        else:

            生成训练用numpy数组_B(输入表单, 词_数表, numpy数组路径)  #汉语用A英语用B
            npz文件 = np.load(numpy数组路径, allow_pickle=True)
            输出np, 输入np = npz文件["输出np"], npz文件["输入np"]

        if os.path.isfile(numpy数组路径):
            npz文件 = np.load(numpy数组路径)
            输出np, 输入np = npz文件["输出np"], npz文件["输入np"]
        else:
            print("训练用numpy数组不存在")
        数据_表 = []
        print("正在打包numpy数组为mindspore所需json格式......")
        for i in range(输入np.shape[0]):

            输入_分 = 输入np[i:i+1, :]
            输入_分 = 输入_分.reshape(-1)
            输出_分 = 输出np[i:i+1, :]
            输出_分 = 输出_分.reshape(-1)
            数据_json = {"id": i, "input": 输入_分.astype(np.int32), "output": 输出_分.astype(np.int32)}
            数据_表.append(数据_json)

        纲要_json = {"id": {"type": "int32"},
                      "input": {"type": "int32", "shape": [-1]},
                      "output": {"type": "int32", "shape": [-1]}}
        if os.path.isfile("../data/mindrecord/训练材料_英语3.minecord.db"):
            os.remove("../data/mindrecord/训练材料_英语3.minecord.db")
        if os.path.isfile("../data/mindrecord/训练材料_英语3.minecord"):
            os.remove("../data/mindrecord/训练材料_英语3.minecord")
        print("写入mindspore格式需要预留约10G内存")
        print("正在写入mindspore格式......")
        writer = FileWriter("../data/mindrecord/训练材料_英语3.minecord", shard_num=1)
        writer.add_schema(纲要_json, "nlp_1")
        writer.add_index(["id"])
        writer.write_raw_data(数据_表)
        writer.commit()
        print("写入mindspore格式完成。")
    else:
        print('词_数表路径或数_词表路径不存在')
def json到minecord(路径,numpy路径,最终文件名):

    #路径 = "../data/预训练数据.json"
    输入表单 = 读取训练数据_A(路径)
    词_数表路径 = "../data/词_数50000.json"
    数_词表路径 = "../data/数_词50000.json"

    if os.path.isfile(词_数表路径) and os.path.isfile(数_词表路径):
        词_数表, 数_词表 = 读出引索(词_数表路径, 数_词表路径)



        numpy数组路径 = numpy路径
        if os.path.isfile(numpy数组路径):
            npz文件 = np.load(numpy数组路径, allow_pickle=True)
            输出np, 输入np = npz文件["输出np"], npz文件["输入np"]
        else:

            生成训练用numpy数组_B(输入表单, 词_数表, numpy数组路径)
            npz文件 = np.load(numpy数组路径, allow_pickle=True)
            输出np, 输入np = npz文件["输出np"], npz文件["输入np"]

        if os.path.isfile(numpy数组路径):
            npz文件 = np.load(numpy数组路径)
            输出np, 输入np = npz文件["输出np"], npz文件["输入np"]
        else:
            print("训练用numpy数组不存在")
        数据_表 = []
        print("正在打包numpy数组为mindspore所需json格式......")
        for i in range(输入np.shape[0]):

            输入_分 = 输入np[i:i+1, :]
            输入_分 = 输入_分.reshape(-1)
            输出_分 = 输出np[i:i+1, :]
            输出_分 = 输出_分.reshape(-1)
            数据_json = {"id": i, "input": 输入_分.astype(np.int32), "output": 输出_分.astype(np.int32)}
            数据_表.append(数据_json)

        纲要_json = {"id": {"type": "int32"},
                      "input": {"type": "int32", "shape": [-1]},
                      "output": {"type": "int32", "shape": [-1]}}
        if os.path.isfile("../data/mindrecord/"+最终文件名+".minecord.db"):
            os.remove("../data/mindrecord/"+最终文件名+".minecord.db")
        if os.path.isfile("../data/mindrecord/"+最终文件名+".minecord"):
            os.remove("../data/mindrecord/"+最终文件名+".minecord")
        print("正在写入mindspore格式......")
        writer = FileWriter("../data/mindrecord/"+最终文件名+".minecord", shard_num=1)
        writer.add_schema(纲要_json, "nlp_1")
        writer.add_index(["id"])
        writer.write_raw_data(数据_表)
        writer.commit()
        print("写入mindspore格式完成。")

    else:

        print('词_数表路径或数_词表路径不存在')