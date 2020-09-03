import os
import json

def 存(段,文件号,j):
    临2 = {}
    if ('¶' in 段) or ('å' in 段):

        print(段, "--------------------")
        段 = ''

    else:
        临2["id"] = j
        临2["input"] = 段

        json.dump(临2, 文件号, ensure_ascii=False)
        文件号.write('\n')


def txt文本_到训练数据(dirname, 文件号,文本长度=667):
    result = []  # 所有的文件

    j = 0
    i = 0
    临 = ''
    计数 = 0
    小计=0
    段 = ''

    for maindir, subdir, file_name_list in os.walk(dirname):

        for filename in file_name_list:

            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            print(apath)
            f = open(apath, "r", encoding='utf8')
            str1 = f.read()
            for s in str1:

                if (u'\u0041' <= s <= u'\u005a') or (u'\u0061' <= s <= u'\u007a') :
                    if 临 == '':

                        临 = s
                    else:
                        临 = 临 + s


                else:
                    if 临 == '':
                        段 = 段 + s
                        if s != ' ':
                            小计 = 小计 + 1
                        if 小计 == 文本长度:
                            存(段,  文件号, j)
                            小计=0
                            j=j+1
                            段 = ''


                        计数 = 计数 + 1
                    else:

                        段 = 段 + 临
                        if 临!= ' ':
                            小计 = 小计 + 1
                        if 小计 == 文本长度:
                            存(段,  文件号, j)
                            小计=0
                            j=j+1
                            段=''

                        计数 = 计数 + 1
                        段 = 段 + s
                        if s!=' ':
                            小计 = 小计 + 1
                        if 小计 == 文本长度:
                            存(段,  文件号, j)
                            小计=0
                            j=j+1
                            段 = ''

                        临 = ''
                        计数 = 计数 + 1

                if 计数 % 100000 == 0 :
                    print(计数)
            段 = ''
            临 = ''

        return "临2"



with open("./新生成的训练数据/XXX训练材料.json", "w", encoding="utf8") as f:
    json数据 = txt文本_到训练数据("./再训练txt", f)
    print(json数据)
