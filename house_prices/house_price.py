import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from sklearn import linear_model

data_train = DataFrame(pd.read_csv("data/train.csv"))
columns=data_train.columns
decribe=data_train.describe()

'''
#各属性与价格的关系
var="GrLivArea" #房产面积
# var="TotalBsmtSF"#地下室总面积
data=pd.concat([data_train["SalePrice"],data_train[var]],axis=1)
print(data)
data.plot.scatter(x=var,y="SalePrice",ylim=0.800)
'''


#var ="OverallQual"#整体材料和完成质量
var="YearBuilt"#建造时间
data=pd.concat([data_train["SalePrice"],data_train[var]],axis=1)
f,ax=plt.subplots(figsize=(16,8))
fig=sns.boxplot(x=var,y="SalePrice",data=data)
fig.axis(ymin=0,ymax=800000)
plt.xticks(rotation=90)


'''
总结：
观察上述4个特征对于价格的影响，居住面积和地下室面积与价格成正线性相关。
材料质量和建造时间与价格也有一定关系，其中材料质量关系更大。
'''

'''
#correlation matrix 各属性之间的关系，协方差
corrmat=data_train.corr()
f,ax=plt.subplots(figsize=(12,9))
# sns.heatmap(data=corrmat,vmax=.8,square=True)
# plt.show()


#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index#前k个与SalePrice相关系数最大的属性(材料质量、居住面积、车库大小...)
cm = np.corrcoef(data_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
'''

'''
#绘制各属性之间的关系
sns.set()
cols=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(data_train[cols],height=2.6)
plt.show()
'''

# 查看丢失数值的属性，超过15%则将其丢弃
total = data_train.isnull().sum().sort_values(ascending=False)
percent = (data_train.isnull().sum() / data_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

# 删除缺少的数据
data_train = data_train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
data_train = data_train.drop(data_train.loc[data_train["Electrical"].isnull()].index)  # 将电力数据为空的记录删除
# print(data_train.isnull().sum().max())检查还有无缺损值，将为空的最大总和打印，应为0

# 对销售数据进行标准化
saleprice_scaled = StandardScaler().fit_transform(
    data_train["SalePrice"][:, np.newaxis])  # np.newaxis作用是在当前位置增加一维。原先shape(1459,)-->现在shape(1459,1)
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
# print('outer range (low) of the distribution:')
# print(low_range)
# print('\nouter range (high) of the distribution:')
# print(high_range)

# 对标准化的数据进行绘图
var = "GrLivArea"
data = pd.concat([data_train["SalePrice"], data_train[var]], axis=1)
# data.plot.scatter(x=var,y="SalePrice",ylim=(0.800))

# 删除偏离特别离谱的点
temp = data_train.sort_values(by="GrLivArea", ascending=False)[:2]  # 删除GrLivArea值最高的两个点
delete_list = [1299, 524]
data_train = data_train.drop(delete_list)

var = "TotalBsmtSF"
data = pd.concat([data_train["SalePrice"], data_train[var]], axis=1)
# data.plot.scatter(x=var,y="SalePrice",ylim=(0.800))

'''
#寻找正态分布的量，但是SalePrice不是正态分布的
sns.distplot(data_train["SalePrice"],fit=norm)
fig=plt.figure()
res=stats.probplot(data_train["SalePrice"],plot=plt)
'''

# 应用log transformation使数据呈现正态分布
# in case of positive skewness, log transformations usually works well.
data_train["SalePrice"] = np.log(data_train["SalePrice"])
# sns.distplot(data_train['SalePrice'], fit=norm);
# fig = plt.figure()
# res = stats.probplot(data_train['SalePrice'], plot=plt)

'''
#检查GrLivArea的分布情况,发现同样的情况
sns.distplot(data_train["GrLivArea"],fit=norm)
fig=plt.figure()
res=stats.probplot(data_train['GrLivArea'],plot=plt)
'''

#使用log transformation
data_train["GrLivArea"] = np.log(data_train["GrLivArea"])
# sns.distplot(data_train["GrLivArea"],fit=norm)
# fig=plt.figure()
# res=stats.probplot(data_train['GrLivArea'],plot=plt)

'''
#检查TotalBsmtSF,由于大量房子没有地下室，值为0，所以不能使用log transform
sns.distplot(data_train["TotalBsmtSF"],fit=norm)
fig=plt.figure()
res=stats.probplot(data_train['TotalBsmtSF'],plot=plt)
'''

#增加一列，作为有无地下室的标志
data_train['HasBsmt'] = pd.Series(len(data_train['TotalBsmtSF']), index=data_train.index)
data_train['HasBsmt'] = 0
data_train.loc[data_train['TotalBsmtSF']>0,'HasBsmt'] = 1

#对有地下室的记录进行transform data
data_train.loc[data_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(data_train.loc[data_train['HasBsmt']==1,'TotalBsmtSF'])
# sns.distplot(data_train[data_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
# fig = plt.figure()
# res = stats.probplot(data_train[data_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

#建立线性回归模型
#正则化取出所需的feature
train_df=data_train.filter(regex='SalePrice|OverallQual|GrLivArea|TotalBsmtSF|GarageCars|FullBath|YearBuilt')
train_up=train_df.values

y=train_up[:,-1].ravel()
X=train_up[:,:-1]

clf=linear_model.LinearRegression(normalize=True)
clf.fit(X,y)

#对测试集数据也做同样的处理
data_test=DataFrame(pd.read_csv("data/test.csv"))
#对地下室面积和居住面积进行log transform，使其呈现正态分布
data_test["GrLivArea"] = np.log(data_test["GrLivArea"])
data_test['HasBsmt'] = pd.Series(len(data_test['TotalBsmtSF']), index=data_test.index)
data_test['HasBsmt'] = 0
data_test.loc[data_test['TotalBsmtSF']>0,'HasBsmt'] = 1
data_test.loc[data_test['HasBsmt']==1,'TotalBsmtSF'] = np.log(data_test.loc[data_test['HasBsmt']==1,'TotalBsmtSF'])
#对空值进行赋值
data_test.loc[data_test['TotalBsmtSF'].isnull().index,'TotalBsmtSF']=0
data_test.loc[data_test['BsmtFullBath'].isnull(),'BsmtFullBath']=0
data_test.loc[data_test['GarageCars'].isnull(),'GarageCars']=0

#对测试集数据进行预测
test_df=data_test.filter(regex='OverallQual|GrLivArea|TotalBsmtSF|GarageCars|FullBath|YearBuilt')
predictions=clf.predict(test_df)
origin_price=np.exp(predictions)
results=DataFrame({"Id":data_test["Id"].values,"SalePrice":origin_price.astype(np.float32)})
#results.to_csv("result/linear_regression_prediction.csv",index=False)

