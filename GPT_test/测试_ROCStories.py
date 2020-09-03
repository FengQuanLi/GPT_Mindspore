import json
from 取训练数据 import 生成测试用numpy数组_A, 读出引索, 生成测试用numpy数组_B
import os.path
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context, Tensor
import mindspore
import numpy as np
import mindspore.ops.operations as P
from GPT模型_测试 import 创建_遮罩, 输出函数_GPT
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
def 单步测试(单元_捆, 词_数表, 数_词表, network):
    枝数=len(单元_捆)
    for i in range(枝数):
        if i==0:
            测试_捆 = 单元_捆[i]["待测数组"]
        else:
            测试_捆 = np.vstack((测试_捆, 单元_捆[i]["待测数组"] ))



    MAKS1 = 创建_遮罩(测试_捆, 12)
    MAKS1 = Tensor(MAKS1)
    累加 = Tensor(测试_捆, mindspore.int32)
    结果_A = network.前向(累加, MAKS1)
    softmax = P.Softmax(-1)
    成功数=0
    和报=''
    for i in range(枝数):

        结果 = 结果_A[i:i+1, -1-单元_捆[i]["标差"], :]
        结果 = softmax(结果)
        结果 = 结果.asnumpy()

        结果 = np.argmax(结果, -1)
        返回 ,简报= 生成测试简报(数_词表, 结果, 单元_捆[i])
        和报=和报+简报+"\n"
        if 返回==True:
            成功数=成功数+1

        # torch.cat([a, b], dim=0)
    #累加 = 累加.cpu().numpy()

    return 成功数,枝数 ,和报
def 生成测试简报(数_词表,数据, 单元):
    临 = 数据
    欲打印=[数_词表[str(临[i])] for i in range(临.size)]
    打印=""
    for i in range(len(欲打印)):
        打印=打印+欲打印[i]


    标签=list(单元['标准结果'])[0]
    if 标签==打印:
        简报="成功  ——预测:" + 打印 + "-标准回答:" + 单元['标准结果'] + "   题目：" + 单元["待测目标"]
        return True,简报
    else:
        简报="失败  ——预测:"+打印+"-标准回答:"+单元['标准结果']+"   题目："+单元["待测目标"]
        return False,简报

丢弃率 = 0.0
词库总数 = 50001
向量维度 = 768
层数 = 12
头数 = 12
文件名 = 'ROCStories_dev'
#模型数据文件夹="../data/checkpoint/"+文件名+"/"
测试报告路径="../data/ROCStories_训练和测试/测试报告_ROCStories.txt"
测试题目路径="../data/ROCStories_训练和测试/ROCStories/ROCStories_spring2016_test.json"
词_数表路径 = "../data/词_数50000.json"
数_词表路径 = "../data/数_词50000.json"
模型数据路径=''
# for maindir, subdir, file_name_list in os.walk(模型数据文件夹):
#
#     for filename in file_name_list:
#         print(filename[-4:])
#         if filename[-4:]=='ckpt':
#             模型数据路径=模型数据文件夹+filename

模型数据路径='../data/checkpoint/ROCStories_dev/checkpoint_ROCStories_test.ckpt'
if os.path.isfile(模型数据路径) and os.path.isfile(测试题目路径):

    network = 输出函数_GPT(词库总数, 向量维度, 层数, 头数, 丢弃率)
    param_dict = load_checkpoint(模型数据路径)
    load_param_into_net(network, param_dict)


    if os.path.isfile(词_数表路径) and os.path.isfile(数_词表路径):
        词_数表, 数_词表 = 读出引索(词_数表路径, 数_词表路径)
    else:
        #写出词标号引索(总表单, 词_数表路径, 数_词表路径)  # 生成标号——字符相互查找的字典
        #词_数表, 数_词表 = 读出引索(词_数表路径, 数_词表路径)
        print('词_数表路径或数_词表路径不存在')
    路2=测试题目路径
    总计数=0
    成功数=0
    with open(路2, encoding="utf8") as f甲:
        测试数组_临=[]
        总辞数=0
        最大辞数=0
        枝数=0
        while True:
            行=f甲.readline()
            if not 行:
                break
            js=json.loads(行)

            测试文本 = js["input"]


            测试_表单 = list(测试文本)
            待测数组 = 生成测试用numpy数组_B(测试_表单, 词_数表).reshape(1, -1)
            #js["labe"]
            待测单元={}
            待测单元["待测目标"]=js["input"]
            待测单元["标准结果"] =js["labe"]
            待测单元["待测数组"] = 待测数组
            待测单元["目标长度"]=待测数组.shape[1]
            测试数组_临.append(待测单元)

        测试数组_临 = sorted(测试数组_临, key=lambda x: x["目标长度"], reverse=True)

        游标=0
        测试总_捆=[]
        置到尾部=False
        while True:
            标长=测试数组_临[游标]["目标长度"]
            单元数 = 666*3//标长
            单元_捆=[]
            for i in range(单元数):
                if 游标+i==len(测试数组_临)-1:
                    置到尾部=True
                if 标长==测试数组_临[游标+i]["目标长度"]:
                    测试数组_临[游标 + i]["标差"]=0
                    单元_捆.append(测试数组_临[游标+i])
                else:
                    标差=标长-测试数组_临[游标 + i]["目标长度"]
                    测试数组_临[游标 + i]["标差"] = 标差
                    补丁 = np.ones((1,标差))*50001
                    测试数组_临[游标 + i]["待测数组"]=np.hstack((测试数组_临[游标 + i]["待测数组"], 补丁))
                    抽样=测试数组_临[游标 + i]["待测数组"]
                    单元_捆.append(测试数组_临[游标 + i])
                if 置到尾部==True:
                    break
            游标 = 游标 + 单元数
            测试总_捆.append(单元_捆)
            if 置到尾部 == True:
                break
    成功数_总,枝数_总 =0,0
    with open(测试报告路径, 'w', encoding='utf8') as f_乙:

        for i in range(len(测试总_捆)):

            成功数,枝数 ,和报=单步测试(测试总_捆[i], 词_数表, 数_词表, network)
            成功数_总=成功数_总+成功数
            枝数_总=枝数_总+枝数
            f_乙.write(和报)
            print("完成度{:.2f}%".format((i+1)*100/len(测试总_捆)))
        尾报 = "成功率：{:.2f}".format(成功数_总 / 枝数_总 * 100) +"%测试总数：" + str(枝数_总) + "\n"

        f_乙.write(尾报)
else:
    print('没有发现模型参数！或测试题目')

print("成功率：{:.2f}".format(成功数_总 / 枝数_总 * 100) , 枝数_总,"测试报告在："+测试报告路径)
