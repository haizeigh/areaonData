import csv
import os
import numpy as np
import re
# import info_gain
import xlrd
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 打开指定文件
path = os.getcwd()
path = path[0: path.rfind('/')]
filePath = path+'/source/B_train_tiny.csv'
train_file = csv.reader(open(filePath,'r'))


# print(A_train_file[0])
# print(A_train_file.line_num)

#字符转化为浮点数
def str_row_to_float(row):
    for i in range(len(row)):
        if(len(row[i].strip()) == 0):
            # todo 修改
            # row[i] = None
            row[i] = 0
        else:
            row[i] = float(row[i].strip())

# 将文件内容转换为np数组fileData
count = 0
# 这里存的是列名
columnNameList = []
labelList=[]
for i in train_file:
    if(count == 0):
        count += 1
        columnNameList = i
        fileData = np.empty(shape=(0,489))
        continue
    # fileData[count -1] = i
    str_row_to_float(i)
    labelList.append(i[0])
    print(len(i))
    i = np.reshape(i[2:], (1,489) )
    fileData = np.append(fileData,i, 0)
    count += 1
# fileData_T =fileData.T
# a = fileData[:,0].tolist()
# print(count)

# todo 数据空缺值处理


# todo 逻辑回顾
labelnp = np.array(labelList)
# fileData = fileData[:, 1:]
# 2.拆分测试集、训练集。
X_train, X_test, Y_train, Y_test = train_test_split(fileData, labelnp, test_size=0.3, random_state=0)
# 设置随机数种子，以便比较结果。

# 3.标准化特征值
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#
# # 4. 训练逻辑回归模型
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train_std, Y_train)
#
# # 5. 预测
prepro = logreg.predict_proba(X_test_std)
acc = logreg.score(X_test_std, Y_test)
print(acc)

# todo 神经网络
