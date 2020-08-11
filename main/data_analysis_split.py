import csv
import os
import numpy as np
import pandas as pd
import re
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
all_data_b = data_b
# test = pd.read_csv(path+'/source/B_test.csv')
# 拆分成测试数据
test = data_b.sample(1000, random_state = 999)
data_b = data_b[~data_b['no'].isin(test['no'])]
print("加载数据完成")



# 观察userinfo product webinfo 三种属性的缺失情况
a_columns = data_a.columns

user_infos = [i for i in a_columns if 'UserInfo' in i]
product_infos = [i for i in a_columns if 'ProductInfo' in i]
web_infos = [i for i in a_columns if 'WebInfo' in i]

flag_data_a = pd.DataFrame(data_a['flag'])
user_infos_data_a = data_a[user_infos]
product_infos_data_a = data_a[product_infos]
web_infos_data_a = data_a[web_infos]

flag_data_b = pd.DataFrame(data_b['flag'])
user_infos_data_b = data_b[user_infos]
product_infos_data_b = data_b[product_infos]
web_infos_data_b = data_b[web_infos]


flag_test = pd.DataFrame(test['flag'])
user_infos_test = test[user_infos]
product_infos_test = test[product_infos]
web_infos_test = test[web_infos]

# 观察 a、b 数据缺失情况
plt.subplot(131).plot(np.sort(user_infos_data_a.isnull().sum()))
plt.subplot(132).plot(np.sort(product_infos_data_a.isnull().sum()))
plt.subplot(133).plot(np.sort(web_infos_data_a.isnull().sum()))
plt.show()

plt.subplot(131).plot(np.sort(user_infos_data_b.isnull().sum()))
plt.subplot(132).plot(np.sort(product_infos_data_b.isnull().sum()))
plt.subplot(133).plot(np.sort(web_infos_data_b.isnull().sum()))
plt.show()

# 过滤数据a的空值
user_infos_data_a_columns = user_infos_data_a.columns
user_infos_data_a_columns_big = []
for i in user_infos_data_a_columns:
    if ( user_infos_data_a[i].isnull().sum() <= 14936 ):
        user_infos_data_a_columns_big.append(i)
user_infos_data_a_less_null = user_infos_data_a[user_infos_data_a_columns_big]

product_infos_data_a_columns = product_infos_data_a.columns
product_infos_data_a_columns_big = []
for i in product_infos_data_a_columns:
    if ( product_infos_data_a[i].isnull().sum() <= 18936 ):
        product_infos_data_a_columns_big.append(i)
product_infos_data_a_less_null = product_infos_data_a[product_infos_data_a_columns_big]


web_infos_data_a_columns = web_infos_data_a.columns
web_infos_data_a_columns_big = []
for i in web_infos_data_a_columns:
    if ( web_infos_data_a[i].isnull().sum() <= 14936 ):
        web_infos_data_a_columns_big.append(i)
web_infos_data_a_less_null = web_infos_data_a[web_infos_data_a_columns_big]

plt.subplot(131).plot(np.sort(user_infos_data_a_less_null.isnull().sum()))
plt.subplot(132).plot(np.sort(product_infos_data_a_less_null.isnull().sum()))
plt.subplot(133).plot(np.sort(web_infos_data_a_less_null.isnull().sum()))
plt.show()


# 建立模型
xgbc = XGBC(max_depth=100 , seed=999, n_estimators=100, scale_pos_weight = 12 )

user_infos_data_b_test = user_infos_data_b
user_infos_data_b_test = user_infos_data_b_test.fillna(1)
user_infos_data_b_train, user_infos_data_b_test, flag_data_b_train, flag_data_b_test = train_test_split(user_infos_data_b_test, flag_data_b, test_size=0.2, random_state=0)
# 默认是信息增益 importance_type="gain"
xgbc.fit(user_infos_data_b_train, flag_data_b_train)
user_infos_data_b_test_proba = xgbc.predict_proba(user_infos_data_b_test)
print("训练user_infos的数据，得分auc：")
# 填充99 0.5650669642857142
# 填充0 0.5518080357142857
# 填充1 0.5973660714285715
print(AUC(flag_data_b_test, user_infos_data_b_test_proba[:,1]))
feature_importances_ = xgbc.feature_importances_
feature_importances_series = pd.Series(feature_importances_)
feature_importances_series.index = user_infos
#去除分类效果低的属性 反而效果更差
feature_importances_series = feature_importances_series[feature_importances_series.values > 0]

print("根据属性重要性，提取属性大小")
print(feature_importances_series.shape)
effective_user_infos = feature_importances_series.index.values

product_infos_data_test = product_infos_data_b
product_infos_data_test = product_infos_data_test.fillna(0)
product_infos_data_b_train, product_infos_data_b_test, flag_data_b_train, flag_data_b_test = train_test_split(product_infos_data_test, flag_data_b, test_size=0.2, random_state=0)
# 默认是信息增益 importance_type="gain"
xgbc.fit(product_infos_data_b_train, flag_data_b_train)
product_infos_data_b_test_proba = xgbc.predict_proba(product_infos_data_b_test)
print("训练product_infos的数据，得分auc：")
# 填充0 0.6202455357142858
# 填充1 0.6353125
print(AUC(flag_data_b_test, product_infos_data_b_test_proba[:,1]))
feature_importances_ = xgbc.feature_importances_
feature_importances_series = pd.Series(feature_importances_)
feature_importances_series.index = product_infos
#去除分类效果低的属性 反而效果更差
feature_importances_series = feature_importances_series[feature_importances_series.values > 0.04 ]

# plt.plot(np.sort(feature_importances_series))
# plt.show()


print("根据属性重要性，提取属性大小")
print(feature_importances_series.shape)
effective_product_infos = feature_importances_series.index.values

web_infos_data_b_test = web_infos_data_b
web_infos_data_b_test = web_infos_data_b_test.fillna(0)
web_infos_data_b_train, web_infos_data_b_test, flag_data_b_train, flag_data_b_test = train_test_split(web_infos_data_b_test, flag_data_b, test_size=0.2, random_state=0)
# 默认是信息增益 importance_type="gain"
xgbc.fit(web_infos_data_b_train, flag_data_b_train)
web_infos_data_b_test_proba = xgbc.predict_proba(web_infos_data_b_test)
print("训练web_infos的数据，得分auc：")
# 填充0 0.5318080357142857
# 填充1 0.6060491071428571
print(AUC(flag_data_b_test, web_infos_data_b_test_proba[:,1]))
feature_importances_ = xgbc.feature_importances_
feature_importances_series = pd.Series(feature_importances_)
feature_importances_series.index = web_infos
#去除分类效果低的属性 反而效果更差
feature_importances_series = feature_importances_series[feature_importances_series.values > 0]

print("根据属性重要性，提取属性大小")
print(feature_importances_series.shape)
effective_web_infos = feature_importances_series.index.values

# 建立模型
# xgbc = XGBC(max_depth=260 , seed=999, n_estimators=100, scale_pos_weight = 1 )
# # 融合数据  product_infos
# flag_data = flag_data_a.append(flag_data_b)
# product_infos_data = product_infos_data_a.append(product_infos_data_b).fillna(0)
# # effective_product_infos_columns = list(set(product_infos_data_a_columns_big).union(set(effective_product_infos)))
# # effective_product_infos_columns = list(set(product_infos_data_a_columns_big).intersection(set(effective_product_infos)))
# effective_product_infos_columns = product_infos_data_a_columns_big
# # effective_product_infos_columns = effective_product_infos
# product_infos_data = product_infos_data[effective_product_infos_columns]
# # flag_test
# product_infos_test = product_infos_test.fillna(0)
# product_infos_test = product_infos_test[effective_product_infos_columns]
#
# xgbc.fit(product_infos_data, flag_data)
# product_infos_test_proba = xgbc.predict_proba(product_infos_test)
# print("训练融合的product_infos数据，得分auc：")
# # 使用并集数据 0.5357997265892003
# # 0.5501609136477558
# print(AUC(flag_test, product_infos_test_proba[:,1]))


xgbc = XGBC(max_depth=260 , seed=999, n_estimators=100, scale_pos_weight = 0.5 )
# 融合数据  user_infos
flag_data = flag_data_a.append(flag_data_b)
effective_user_infos_columns = list(set(user_infos_data_a_columns_big).intersection(set(effective_user_infos)))
user_infos_data = user_infos_data_a.append(user_infos_data_b).fillna(1)
user_infos_data = user_infos_data[effective_user_infos_columns]
# flag_test
user_infos_test = user_infos_test.fillna(1)
user_infos_test = user_infos_test[effective_user_infos_columns]

xgbc.fit(user_infos_data, flag_data)
user_infos_test_proba = xgbc.predict_proba(user_infos_test)
# 0.6362568352699932
print("训练融合的user_infos数据，得分auc：")
print(AUC(flag_test, user_infos_test_proba[:,1]))



