import pandas as pd
import numpy as np

def ent(data):
    '''
    calculate entropy
    :param data:
    :return:
    '''
    prob = pd.value_counts(data)/len(data)
    return sum(np.log2(prob)*prob*(-1))
def get_info_gain(data, feat, label):
    '''
    :param data: DataFrame
    :param feat: feature
    :param label: target
    :return:
    '''
    e1 = data.groupby(feat).apply(lambda x:ent(x[label]))
    p1 = pd.value_counts(data[feat])/len(data[feat])
    e2 = sum(e1*p1)
    return ent(data[label]) - e2
    pass

if __name__ == '__main__':
    data = pd.DataFrame({'年龄':['青年','青年','青年','青年','青年','中年','中年','中年','中年','中年','老年','老年','老年','老年','老年'],
                         '有工作':['否','否','是','是','否','否','否','是','否','否','否','否','是','是','否'],
                         '有自己的房子':['否','否','否','是','否','否','否','是','是','是','是','是','否','否','否'],
                         '贷款情况':['一般','好','好','一般','一般','一般','好','好','非常好','非常好','非常好','好','好','非常好','一般'],
                         '类别':['否','否','是','是','否','否','否','是','是','是','是','是','是','是','否']})
    print(ent(data['类别']))

    # csv文件
    # aa = data['类别']
    # cc = data['类别'][14]
    data.to_csv("./Testin.csv", encoding="utf-8-sig", mode="a", header=True, index=False)
    for i in range(15):
        print(i)
    # for feat in ['年龄', '有工作', '有自己的房子', '贷款情况']:
    #     feat_ = data[feat]
    #     feat__ = feat_[1]
    # 0.9709505944546686
    label = '类别'
    for feat in ['年龄','有工作','有自己的房子','贷款情况']:
        print(get_info_gain(data, feat, label))
    # 0.08300749985576883
    # 0.32365019815155627
    # 0.4199730940219749
    # 0.36298956253708536

    def savefile(self, my_list):
        """
        把文件存成csv格式的文件，header 写出列名，index写入行名称
        :param my_list: 要存储的一条列表数据
        :return:
        """
        df = pd.DataFrame(data=[my_list])
        df.to_csv("./Test.csv", encoding="utf-8-sig", mode="a", header=False, index=False)
