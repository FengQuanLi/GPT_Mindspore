import csv
import os
import json
import numpy as np
def _rocstories(path,txtpath):
    with open(path, encoding='utf_8') as f:
        with open(txtpath, 'w', encoding='utf-8') as f2:
            f = csv.reader(f)
            s1=''
            lf=list(f)
            for i, line in enumerate(lf):
                if i > 0:
                    s = line[1:5]
                    for i in s:
                        s1=s1+i

                    c1 = line[5]
                    c2 = line[6]
                    y=int(line[-1])

                    if y==1:
                        组1 = '#' +s1 + '|' + c1 + '|' + c2 + '&' + str(1)


                    else:

                        组1 = '#' +s1 + '|' + c1 + '|' + c2 + '&' + str(2)

                    s1=''
                    f2.write(组1+ '\n')

            for i, line in enumerate(lf):
                if i > 0:
                    s = line[1:5]
                    for i in s:
                        s1=s1+i

                    c1 = line[5]
                    c2 = line[6]
                    y=int(line[-1])

                    if y==1:

                        组2 = '#' +s1 + '|' + c2 + '|' + c1 + '&' + str(2)

                    else:


                        组2 = '#' +s1 + '|' + c2 + '|' + c1 + '&' + str(1)
                    s1=''
                    f2.write(组2+ '\n')

            return y

def _rocstories3(path,txtpath):
    with open(path, encoding='utf_8') as f:
        with open(txtpath, 'w', encoding='utf-8') as f2:
            f = csv.reader(f)
            s1=''
            x1=3
            x2=3
            lf=list(f)

            for i, line in enumerate(lf):
                if i > 0:
                    s = line[2:6]
                    for ss in s:
                        s1=s1+ss

                    c1 = line[6]
                    x1=np.random.randint(1,len(lf))
                    x2 = np.random.randint(2, 7)
                    c2 = lf[x1][x2]
                    x2 = np.random.randint(1, 3)
                    if x2==1:

                        组1 = '#' +s1 + '|' + c1 + '|' + c2 + '&' + str(1)
                    else:
                        组1 = '#' +s1 + '|' + c2 + '|' + c1 + '&' + str(2)
                    s1=''
                    f2.write(组1+ '\n')

            return x1


def _rocstories_test(path,txtpath):
    with open(path, encoding='utf_8') as f:
        with open(txtpath, 'w', encoding='utf-8') as f2:
            f = csv.reader(f)
            s1=''
            字典={}

            for i, line in enumerate(list(f)):
                if i > 0:
                    s = line[1:5]
                    for i in s:
                        s1=s1+i

                    c1 = line[5]
                    c2 = line[6]
                    y=int(line[-1])


                    组1 = s1 + '|' + c1 + '|' + c2 + '&'
                    字典['input']=组1
                    字典['labe']=str(y)
                    s1=''

                    json.dump(字典, f2, ensure_ascii=False)
                    f2.write('\n')

            return y

路径="./csvdata/cloze_test_test__spring2016 - cloze_test_ALL_test.csv"

路径json="./ROCStories_训练和测试/ROCStories/ROCStories_spring2016_test.json"
#_rocstories(路径,路径json)
_rocstories_test(路径,路径json)
