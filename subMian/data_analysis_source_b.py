import csv
import os
import numpy as np
import pandas as pd
import re
# import info_gain
import xlrd
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from xgboost.sklearn import XGBClassifier as XGBC
from sklearn.metrics import roc_auc_score as AUC
import xgboost as xgb

# 打开指定文件
path = os.getcwd()
path = path[0: path.rfind('/')]
filePath = path+'/source/B_train_tiny.csv'
train_file = csv.reader(open(filePath,'r'))

data_a = pd.read_csv(path+'/source/A_train.csv')
data_b = pd.read_csv(path+'/source/B_train.csv')
# test = pd.read_csv(path+'/source/B_test.csv')
# 拆分成测试数据
test = data_b.sample(1000, random_state = 999)
data_b = data_b[~data_b['no'].isin(test['no'])]
print("加载数据完成")



# 观察userinfo product webinfo 三种属性的缺失情况
# a_columns = data_a.columns
#
# user_infos = [i for i in a_columns if 'UserInfo' in i]
# product_infos = [i for i in a_columns if 'ProductInfo' in i]
# web_infos = [i for i in a_columns if 'WebInfo' in i]
#
# test = []
# test.append(data_a)
# test.append(data_b)
# for data_y in test:
#     user_infos_data = data_y[user_infos]
#     user_infos_data_null_num = np.sum(user_infos_data.isnull(), axis=0)
#
#     product_infos_data = data_y[product_infos]
#     product_infos_data_null_num = np.sum(product_infos_data.isnull(), axis=0)
#
#     web_infos_data = data_y[web_infos]
#     web_infos_data_null_num = np.sum(web_infos_data.isnull(), axis=0)
#
#     plt.subplot(131).plot(np.sort(user_infos_data_null_num))
#     plt.subplot(132).plot(np.sort(product_infos_data_null_num))
#     plt.subplot(133).plot(np.sort(web_infos_data_null_num))
#     plt.show()


# meaningful_col = []
# describe = data_b.describe()
# colunms = describe.columns
# for colunm in colunms:
#     colunm_0 = describe.ix[0, colunm]
#     if colunm_0 > data_b.shape[0]:
#         meaningful_col.append(colunm_0)



fea = np.sum(data_a.isnull(),axis=0)
feb = np.sum(data_b.isnull(),axis=0)
fet = np.sum(test.isnull(),axis=0)

plt.subplot(211).plot(fea.values)
plt.subplot(212).plot(feb.values)
plt.show()

plt.subplot(211).plot(np.sort(fea))
plt.subplot(212).plot(np.sort(feb))
plt.show()



# 筛选元素的方法， fea[330] = 14936 ；筛选出空的数据量等于 14936 的列
# a = fea[fea==fea[330]]
# b = feb[feb==feb[330]]
# t = fet[fet==fet[330]]
real_data_a = data_a
# 利用下面的代码 做个转换
data_a = data_b

# 尝试排除a数据的多余特征
fea_all_num = data_a.shape[0]
# 按照数据 空白数量筛选
fea_effective = fea[fea < fea_all_num * 0.7]
plt.plot(np.sort(fea_effective))
plt.show()
# 按照重要程度筛选
# 排除no列
fea_effective_column_name = fea_effective.index
data_a_effective = data_a[fea_effective_column_name]
data_a_effective.drop('no', axis= 1, inplace=True)
# 空白填充
data_a_effective = data_a_effective.fillna(99)

# 分离参数
a_flag = pd.DataFrame(data_a_effective['flag'])
data_a_effective.drop('flag', axis= 1, inplace=True)
X_train, X_test, Y_train, Y_test = train_test_split(data_a_effective, a_flag, test_size=0.3, random_state=0)
# 获取有效列名
effective_columns  = data_a_effective.columns.values

# xg_train = xgb.DMatrix(X_train, label = Y_train)
# xg_test = xgb.DMatrix(X_test, label = Y_test)
#
# param = {'booster':'gbtree',
#          'max_depth':10,
#          'eta':0.1,
#          'silent':1,
#          'objective':'binary:logistic',
#          'eval_metric':'auc',
#          'subsample': 1,
#          "colsample_bytree": 0.7,
#          "min_child_weight":2,
#               'gamma':3.1,
#               'lambda':1,
#         "thread":-1,}
# num_boost_round = 1500
# # watchlist = [(xg_train, 'train'), (xg_test, 'eval')]
# num_round=15
# bst = xgb.train(param, xg_train, num_round)
# preds = bst.predict(xg_test)
# auc = AUC(Y_test , preds)
# print(auc)

# 根据重要性筛选
# score = bst.get_score(importance_type='gain')
# print(score)
xgbc = XGBC(max_depth=200 , seed=999, n_estimators=100, scale_pos_weight = 13 )
xgbc.fit(X_train, Y_train)
Y_test_proba = xgbc.predict_proba(X_test)
print("训练a的数据，得分auc：")
print(AUC(Y_test, Y_test_proba[:,1]))

# 默认是信息增益 importance_type="gain"
feature_importances_ = xgbc.feature_importances_
feature_importances_series = pd.Series(feature_importances_)
feature_importances_series.index = effective_columns
effective_columns_before = effective_columns
#去除分类效果低的属性 反而效果更差
feature_importances_series = feature_importances_series[feature_importances_series.values > 0]

print("根据属性重要性，提取属性大小")
print(feature_importances_series.shape)
effective_columns = feature_importances_series.index.values
# 展示列的重要性排序
sort_values = feature_importances_series.sort_values(ascending=False)
plt.plot(sort_values.values)
plt.show()

# 重新设置数据集的属性
# print(data_b.shape)
# print(len(effective_columns_before))
effective_columns_test = test[effective_columns_before]
effective_columns_test = effective_columns_test.fillna(99)
flag_test = pd.DataFrame(test['flag'])
# 预检验b的auc
proba_test = xgbc.predict_proba(effective_columns_test)
print("预检验test的数据，得分auc：")
print(AUC(flag_test, proba_test[:,1]))

append_data = real_data_a.append(data_b)
append_data = append_data.fillna(99)
effective_columns_append_data = append_data[effective_columns]
flag_append_data = pd.DataFrame(append_data['flag'])

effective_columns_test = test[effective_columns]
effective_columns_test = effective_columns_test.fillna(99)
flag_test = pd.DataFrame(test['flag'])


xgbc = XGBC(max_depth=200 , seed=999, n_estimators=200, scale_pos_weight = 8 )
xgbc.fit(effective_columns_append_data, flag_append_data)
proba_test = xgbc.predict_proba(effective_columns_test)

print("测试的数据，得分auc：")
# 0.5444150148097516
print(AUC(flag_test, proba_test[:,1]))





# 找到 a b c 三者的共同 索引
# a_and_b = a.index.difference(a.index.difference(b.index))
# a_and_b_and_t = a_and_b.difference(a_and_b.difference(t.index))
# c = pd.DataFrame(a_and_b_and_t)
# # todo 这里为什添加
# c = c.append(['ProductInfo_89'])
# # c[0] 取出的是 序列， 这里过滤 排除 no 的列名
# c = c[c[0] != 'no']
# c.to_csv(r'../source/index.csv',index=True)
#
# # todo 优化
# xgbc = XGBC(max_depth=8, seed=999, n_estimators=100)
# train = data_b.sample(3000, random_state=999)

# data_b['no'] 得到序列； isin 判断是否存在的结果，得到 no列的 布尔序列； ~ 表示取反；
# 最后形式如 df[df.A>0] 得到pandas 对象的筛选结果，依据的是 no列的 布尔序列, 得到 pandas 对象
# v = data_b[~data_b['no'].isin(train['no'])]
#
# # 去除no flag 的列表 得到 index 对象
# train_data_name = train.columns.difference(['no', 'flag'])
# # 得到pandas对象
# train_data = train[train_data_name]
# # 得到序列对象
# train_flag_data = train['flag']
#
# v_data_name = v.columns.difference(['no', 'flag'])
# v_data = v[v_data_name]
# v_flag_data = v['flag']

# 测试集 和 验证集合 是不是反了
# xgbc.fit(train_data, train_flag_data)
# p = xgbc.predict_proba(v_data)[:,1]
# print(AUC(v_flag_data, p))

# xgbc.pre
# 另外的尝试  train_data  train_flag_data
# xgbc.fit(v_data, v_flag_data)
# p = xgbc.predict_proba(train_data)[:,1]
# print(AUC(train_flag_data, p))

# 特征重要性
# importances_ = xgbc.feature_importances_
# a = pd.Series(importances_)
# a.index = v_data_name
# select_a = a[a > 0]

# xgbc.sc
#
# fillna = data_a['ProductInfo_89'].fillna(-1)
# fillna = fillna + 1.1
# log = np.log(fillna)

# todo 平滑处理？
# data_a['ProductInfo_89'] = (np.log(data_a['ProductInfo_89'].fillna(-1) + 1.1)).astype(np.int)
# data_b['ProductInfo_89'] = (np.log(data_b['ProductInfo_89'].fillna(-1) + 1.1)).astype(np.int)
# test['ProductInfo_89'] = (np.log(test['ProductInfo_89'].fillna(-1) + 1.1)).astype(np.int)
#
# cross = select_a[select_a > 0]
# cross.shape
# cu=[];cp=[];cw=[]
# for i in cross.index:
#     # i 是字符串，列名
#     if(i[0]=='P'):
#         cp.append(i)
#     if(i[0]=='U'):
#         cu.append(i)
#     if(i[0]=='W'):
#         cw.append(i)
#
# # 建立 复核 特征
# for i in cw:
#     for j in cp:
#         data_a[i + j] = data_a[i] * data_a[j]
#         data_b[i + j] = data_b[i] * data_b[j]
#         test[i + j] = test[i] * test[j]
#
# data_a.to_csv(r'../source/A_train_new.csv',index=False)
# data_b.to_csv(r'../source/B_train_new.csv',index=False)
# test.to_csv(r'../source/B_test_new.csv',index=False)
#
# # (40000, 573) (4000, 573) (13463, 572)
# print(data_a.shape,data_b.shape,test.shape)
#
#


# print(A_train_file[0])
# print(A_train_file.line_num)

#字符转化为浮点数
# def str_row_to_float(row):
    # for i in range(len(row)):
    #     if(len(row[i].strip()) == 0):
    #         # todo 修改
    #         # row[i] = None
    #         row[i] = 0
    #     else:
    #         row[i] = float(row[i].strip())

# 将文件内容转换为np数组fileData
# count = 0
# # 这里存的是列名
# columnNameList = []
# labelList=[]
# for i in train_file:
#     if(count == 0):
#         count += 1
#         columnNameList = i
#         fileData = np.empty(shape=(0,489))
#         continue
#     # fileData[count -1] = i
#     str_row_to_float(i)
#     labelList.append(i[0])
#     print(len(i))
#     i = np.reshape(i[2:], (1,489) )
#     fileData = np.append(fileData,i, 0)
#     count += 1
# fileData_T =fileData.T
# a = fileData[:,0].tolist()
# print(count)

# todo 数据空缺值处理


# todo 逻辑回顾
# labelnp = np.array(labelList)
# # fileData = fileData[:, 1:]
# # 2.拆分测试集、训练集。
# X_train, X_test, Y_train, Y_test = train_test_split(fileData, labelnp, test_size=0.3, random_state=0)
# # 设置随机数种子，以便比较结果。
#
# # 3.标准化特征值
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)
# #
# # # 4. 训练逻辑回归模型
# logreg = linear_model.LogisticRegression(C=1e5)
# logreg.fit(X_train_std, Y_train)
# #
# # # 5. 预测
# prepro = logreg.predict_proba(X_test_std)
# acc = logreg.score(X_test_std, Y_test)
# print(acc)

# todo 神经网络
