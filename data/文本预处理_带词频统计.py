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
    标号_到_字符 = {}
    字符_到_标号 = {}
    字符_到_词频 = {}
    标号_字符 = []
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
                        小计=小计+1
                        if 小计 == 文本长度:
                            存(段,  文件号, j)
                            小计=0
                            j=j+1
                            段 = ''
                        if s not in 标号_字符:
                            标号_字符.append(s)
                            字符_到_标号[s] = i
                            标号_到_字符[i] = s
                            字符_到_词频[s] = 1
                            i = i + 1
                        else:
                            字符_到_词频[s]=字符_到_词频[s]+1

                        计数 = 计数 + 1
                    else:

                        段 = 段 + 临
                        小计 = 小计 + 1
                        if 小计 == 文本长度:
                            存(段,  文件号, j)
                            小计=0
                            j=j+1
                            段=''
                        if 临 not in 标号_字符:
                            标号_字符.append(临)
                            字符_到_标号[临] = i
                            标号_到_字符[i] = 临
                            字符_到_词频[临] =1
                            i = i + 1
                        else:
                            数=字符_到_词频[临] + 1
                            # if 数>100:
                            #     print(临,"  这个词的词频为{}".format(数))
                            字符_到_词频[临]=数
                        计数 = 计数 + 1
                        段 = 段 + s
                        小计 = 小计 + 1
                        if 小计 == 文本长度:
                            存(段,  文件号, j)
                            小计=0
                            j=j+1
                            段 = ''
                        if s not in 标号_字符:
                            标号_字符.append(s)
                            字符_到_标号[s] = i
                            标号_到_字符[i] = s
                            字符_到_词频[s] = 1
                            i = i + 1
                        else:
                            字符_到_词频[s]=字符_到_词频[s]+1
                        临 = ''
                        计数 = 计数 + 1

                if 计数 % 100000 == 0 and i != 0:
                    print(i, 标号_到_字符[i - 1], filename,计数)
            段 = ''
            临 = ''
        print(标号_到_字符[1], 标号_到_字符[111], len(标号_到_字符))
        with open("./新生成的训练数据/词_数表路径.json", 'w', encoding='utf-8') as f:
            json.dump(字符_到_标号, f, ensure_ascii=False)
        with open("./新生成的训练数据/数_词表路径.json", 'w', encoding='utf-8') as f:
            json.dump(标号_到_字符, f, ensure_ascii=False)
        with open("./新生成的训练数据/字符_到_词频.json", 'w', encoding='utf-8') as f:
            json.dump(字符_到_词频, f, ensure_ascii=False)
        return "临2"

#####################################
#基本处理原则是：
#【1】按一定字符长度把要处理的文本切割成块
#【2】把所有连续英语字母当作一个字符
#【3】同时采集词频和生成查询表
#####################################



with open("./新生成的训练数据/XXX训练材料.json", "w", encoding="utf8") as f:
    json数据 = txt文本_到训练数据("./TXT", f)
    print(json数据)
