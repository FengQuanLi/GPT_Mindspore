import json
import numpy as np

def 读出引索(词_数表路径, 数_词表路径):
    with open(词_数表路径, encoding='utf-8') as f:
        词_数表= json.load(f)

    with open(数_词表路径, encoding='utf-8') as f:
        数_词表 = json.load(f)
    return 词_数表, 数_词表

def 生成训练用numpy数组_A(输入表单,  词_数表, numpy数组路径):
    """
    将预处理过的文本转化为numpy数组并保存并用于训练。
    """
    表_1 = []
    表_2 = []
    i=0
    临=''
    for 表单 in 输入表单:
        表_3=[]
        for 字符 in 表单:
            if (u'\u0041' <= 字符 <= u'\u005a') or (u'\u0061' <= 字符 <= u'\u007a'):
                if 临 == '':

                    临 = 字符
                else:
                    临 = 临 + 字符
            else:

                if 临 == '':

                    if 字符 in 词_数表:
                        表_3.append(词_数表[字符])
                    else:
                        表_3.append(14991)
                else:
                    if 临 in 词_数表:
                        表_3.append(词_数表[临])
                    else:
                        表_3.append(14991)
                    临=''
                    if 字符 in 词_数表:
                        表_3.append(词_数表[字符])
                    else:
                        表_3.append(14991)
        if 临!='':
            if 临 in 词_数表:
                表_3.append(词_数表[临])
            else:
                表_3.append(14991)
            临 = ''


        if len(表_3)!=667:
            #表_1.append(np.array(表_3[0:-1]))
            #表_2.append(np.array(表_3[1:]))
            print(表_3)
        else:

            表_1.append(np.array(表_3[0:-1]))
            表_2.append(np.array(表_3[1:]))
        if i % 1000 == 0:
            print("数据转化为numpy数组完成度百分比{}".format(i/len(输入表单)*100))
        i = i + 1
    print("数据转化为numpy数组完成。")


    输入np = np.array(表_1)
    输出np = np.array(表_2)
    np.savez(numpy数组路径, 输出np=输出np, 输入np=输入np)

def 读取训练数据_A(路径):
    输入表单 = []
    with open(路径, encoding='utf-8') as f:
        while True:
            行 = f.readline()
            if not 行:
                break
            json_行 = json.loads(行)

            内容 = json_行['input']
            输入表单.append(内容)

    return 输入表单
def 生成训练用numpy数组_B(输入表单,  词_数表, numpy数组路径):
    """
    将预处理过的文本转化为numpy数组并保存并用于训练。
    """
    表_1 = []
    表_2 = []
    i=0
    临=''
    for 表单 in 输入表单:
        表_3=[]
        for 字符 in 表单:
            if (u'\u0041' <= 字符 <= u'\u005a') or (u'\u0061' <= 字符 <= u'\u007a'):
                if 临 == '':

                    临 = 字符
                else:
                    临 = 临 + 字符
            else:

                if 临 == '':

                    if 字符.lower() in 词_数表:
                        if 字符 != ' ':
                            表_3.append(词_数表[字符.lower()])
                    else:
                        表_3.append(49999)
                else:
                    if 临.lower() in 词_数表:
                        if 临 != ' ':
                            表_3.append(词_数表[临.lower()])
                    else:
                        表_3.append(49999)
                    临=''
                    if 字符.lower() in 词_数表:
                        if 字符 != ' ':
                            表_3.append(词_数表[字符.lower()])
                    else:
                        表_3.append(49999)
        if 临!='':
            if 临.lower() in 词_数表:
                if 临 != ' ':
                    表_3.append(词_数表[临.lower()])
            else:
                表_3.append(49999)
            临 = ''


        if len(表_3)!=667:
            #表_1.append(np.array(表_3[0:-1]))
            #表_2.append(np.array(表_3[1:]))
            print(表_3)
        else:

            表_1.append(np.array(表_3[0:-1]))
            表_2.append(np.array(表_3[1:]))
        if i % 1000 == 0:
            print("数据转化为numpy数组完成度百分比{}".format(i/len(输入表单)*100))
        i = i + 1
    print("数据转化为numpy数组完成。")


    输入np = np.array(表_1)
    输出np = np.array(表_2)
    np.savez(numpy数组路径, 输出np=输出np, 输入np=输入np)