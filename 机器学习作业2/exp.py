#梯度下降
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
import torch
from sklearn import preprocessing

data_df = pd.read_csv('./cwurData.csv')  # 读入 csv 文件为 pandas 的 DataFrame
#print(data_df.head(3).T)  # 观察前几列并转置方便观察
data_df = data_df.dropna()  # 舍去包含 NaN 的 row
feature_cols = ['quality_of_faculty', 'publications', 'citations', 'alumni_employment',
                'influence', 'quality_of_education', 'broad_impact', 'patents']
X = data_df[feature_cols]
Y = data_df['score']
all_y = Y.values
all_x = X.values
min_max_scaler = preprocessing.MinMaxScaler()
all_x = min_max_scaler.fit_transform(all_x)#数据归一化
x_train, x_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.2)#将数据分为训练集与测试集
feature = torch.tensor(x_train,dtype=torch.float)
lable = torch.tensor(y_train,dtype=torch.float)#numpy转tensor
bais = torch.ones(feature.shape[0])
bais = bais.view([1600,1])
feature = torch.cat((feature,bais),1)#为特征增添偏移量
x_t = torch.tensor(x_test,dtype=torch.float)
baiss = torch.ones(x_t.shape[0])
baiss = baiss.view([len(x_t),1])
x_t = torch.cat((x_t,baiss),1)
w_pre = torch.rand(len(feature_cols)+1,1)
num_epoch = 500
lr = 0.3
for i in range(1,num_epoch+1):
        y_pre = torch.mm(feature,w_pre)
        loss = y_pre - lable.view([1600,1])
        loss = loss/len(feature)
        loss = torch.mm(torch.t(feature),loss)
        w_pre = w_pre - lr*loss#核心部分，进行梯度下降
y_pre = torch.mm(x_t,w_pre)
loss = torch.tensor(y_test).view([400,1]) - y_pre
loss = torch.mm(torch.t(loss),loss)
rms = (loss.item()/len(y_pre))**0.5
print(rms)#计算RMSE
var = (torch.tensor(y_test)-torch.mean(torch.tensor(y_test))).view([-1,1])
var = torch.mm(torch.t(var),var)
R = 1-loss.item()/var.item()
print(R)#计算R^2