## 数据探索-EDA
#### EDA目标
- EDA的价值主要在于熟悉数据集，了解数据集，对数据集进行验证来确定所获得数据集可以用于接下来的机器学习或者深度学习使用。
- 当了解了数据集之后我们下一步就是要去了解变量间的相互关系以及变量与预测值之间的存在关系。
- 引导数据科学从业者进行数据处理以及特征工程的步骤,使数据集的结构和特征集让接下来的预测问题更加可靠。
- 完成对于数据的探索性分析，并对于数据进行一些图表或者文字总结并打卡。
### 1.数据预览
```
# 导入相应的包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import missingno as msno
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
pd.set_option('display.max_columns', None) # 数据预览时显示所有列
```
***
```
# 导入数据集
train_dat = pd.read_csv("./data/train.csv")
test_dat = pd.read_csv("./data/testA.csv")
```
***
```
# train_data概览
train_dat.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 800000 entries, 0 to 799999
Data columns (total 47 columns):
 #   Column              Non-Null Count   Dtype  
---  ------              --------------   -----  
 0   id                  800000 non-null  int64  
 1   loanAmnt            800000 non-null  float64
 2   term                800000 non-null  int64  
 3   interestRate        800000 non-null  float64
 4   installment         800000 non-null  float64
 5   grade               800000 non-null  object 
 6   subGrade            800000 non-null  object 
 7   employmentTitle     799999 non-null  float64
 8   employmentLength    753201 non-null  object 
 9   homeOwnership       800000 non-null  int64  
 10  annualIncome        800000 non-null  float64
 11  verificationStatus  800000 non-null  int64  
 12  issueDate           800000 non-null  object 
 13  isDefault           800000 non-null  int64  
 14  purpose             800000 non-null  int64  
 15  postCode            799999 non-null  float64
 16  regionCode          800000 non-null  int64  
 17  dti                 799761 non-null  float64
 18  delinquency_2years  800000 non-null  float64
 19  ficoRangeLow        800000 non-null  float64
 20  ficoRangeHigh       800000 non-null  float64
 21  openAcc             800000 non-null  float64
 22  pubRec              800000 non-null  float64
 23  pubRecBankruptcies  799595 non-null  float64
 24  revolBal            800000 non-null  float64
 25  revolUtil           799469 non-null  float64
 26  totalAcc            800000 non-null  float64
 27  initialListStatus   800000 non-null  int64  
 28  applicationType     800000 non-null  int64  
 29  earliesCreditLine   800000 non-null  object 
 30  title               799999 non-null  float64
 31  policyCode          800000 non-null  float64
 32  n0                  759730 non-null  float64
 33  n1                  759730 non-null  float64
 34  n2                  759730 non-null  float64
 35  n3                  759730 non-null  float64
 36  n4                  766761 non-null  float64
 37  n5                  759730 non-null  float64
 38  n6                  759730 non-null  float64
 39  n7                  759730 non-null  float64
 40  n8                  759729 non-null  float64
 41  n9                  759730 non-null  float64
 42  n10                 766761 non-null  float64
 43  n11                 730248 non-null  float64
 44  n12                 759730 non-null  float64
 45  n13                 759730 non-null  float64
 46  n14                 759730 non-null  float64
dtypes: float64(33), int64(9), object(5)
memory usage: 286.9+ MB
```
```
# test_data概览
test_dat.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200000 entries, 0 to 199999
Data columns (total 46 columns):
 #   Column              Non-Null Count   Dtype  
---  ------              --------------   -----  
 0   id                  200000 non-null  int64  
 1   loanAmnt            200000 non-null  float64
 2   term                200000 non-null  int64  
 3   interestRate        200000 non-null  float64
 4   installment         200000 non-null  float64
 5   grade               200000 non-null  object 
 6   subGrade            200000 non-null  object 
 7   employmentTitle     200000 non-null  float64
 8   employmentLength    188258 non-null  object 
 9   homeOwnership       200000 non-null  int64  
 10  annualIncome        200000 non-null  float64
 11  verificationStatus  200000 non-null  int64  
 12  issueDate           200000 non-null  object 
 13  purpose             200000 non-null  int64  
 14  postCode            200000 non-null  float64
 15  regionCode          200000 non-null  int64  
 16  dti                 199939 non-null  float64
 17  delinquency_2years  200000 non-null  float64
 18  ficoRangeLow        200000 non-null  float64
 19  ficoRangeHigh       200000 non-null  float64
 20  openAcc             200000 non-null  float64
 21  pubRec              200000 non-null  float64
 22  pubRecBankruptcies  199884 non-null  float64
 23  revolBal            200000 non-null  float64
 24  revolUtil           199873 non-null  float64
 25  totalAcc            200000 non-null  float64
 26  initialListStatus   200000 non-null  int64  
 27  applicationType     200000 non-null  int64  
 28  earliesCreditLine   200000 non-null  object 
 29  title               200000 non-null  float64
 30  policyCode          200000 non-null  float64
 31  n0                  189889 non-null  float64
 32  n1                  189889 non-null  float64
 33  n2                  189889 non-null  float64
 34  n3                  189889 non-null  float64
 35  n4                  191606 non-null  float64
 36  n5                  189889 non-null  float64
 37  n6                  189889 non-null  float64
 38  n7                  189889 non-null  float64
 39  n8                  189889 non-null  float64
 40  n9                  189889 non-null  float64
 41  n10                 191606 non-null  float64
 42  n11                 182425 non-null  float64
 43  n12                 189889 non-null  float64
 44  n13                 189889 non-null  float64
 45  n14                 189889 non-null  float64
dtypes: float64(33), int64(8), object(5)
memory usage: 70.2+ MB
```
**对数据进行压缩加快分析速度**
```
# 定义内存压缩函数
def reduce_memory_use(data, verbose=True):
    ini_memroy = data.memory_usage().sum()/1024 ** 2
    print("内存优化前: {:.2f} MB".format(ini_memroy))
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    for col in data.columns:
        col_type = data[col].dtypes
        if col_type in numerics:
            val_min = data[col].min()
            val_max = data[col].max()
            if str(col_type)[:3] == "int":
                if val_min > np.iinfo(np.int8).min and val_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif val_min > np.iinfo(np.int16).min and val_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif val_min > np.iinfo(np.int32).min and val_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif val_min > np.iinfo(np.int64).min and val_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int16)
            else:
                if val_min > np.finfo(np.float16).min and val_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif val_min > np.finfo(np.float32).min and val_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

    trans_memory = data.memory_usage().sum()/1024 ** 2
    if verbose:
        print("内存优化后: {:.2f} MB".format(trans_memory))
        print("内存压缩比例: {:.2f}%".format((100*(ini_memroy - trans_memory)/ini_memroy)))
    return data
```
```
train_dat = reduce_memory_use(train_dat)
test_dat = reduce_memory_use(test_dat)
```
```
内存优化前: 286.87 MB
内存优化后: 94.60 MB
内存压缩比例: 67.02%
内存优化前: 70.19 MB
内存优化后: 23.46 MB
内存压缩比例: 66.58%
```
***
```
# 查看连续型特征的描述统计: train_dat
train_dat.describe(exclude="object")
```
```
# 查看连续型特征的描述统计: test_dat
test_dat.describe(exclude="object")
```
### 2.缺失值统计
**定义函数统计数据的缺失情况**
```
def stat_missing_col(data):
    i = 0
    for col in data.columns:
        if data[col].isnull().sum() == 0:
            continue
        else:
            i += 1
            null_radio = data[col].isnull().sum()/data.shape[0]
            print("特征: {}, 缺失值比例: {}".format(col, null_radio))
    print("有缺失值特征共计: {}".format(i))
    return
print("train_dat缺失值情况=>")
stat_missing_col(train_dat)
print("="*40)
print("test_dat缺失值情况=>")
```
```
train_dat缺失值情况=>
特征: employmentTitle, 缺失值比例: 1.25e-06
特征: employmentLength, 缺失值比例: 0.05849875
特征: postCode, 缺失值比例: 1.25e-06
特征: dti, 缺失值比例: 0.00029875
特征: pubRecBankruptcies, 缺失值比例: 0.00050625
特征: revolUtil, 缺失值比例: 0.00066375
特征: title, 缺失值比例: 1.25e-06
特征: n0, 缺失值比例: 0.0503375
特征: n1, 缺失值比例: 0.0503375
特征: n2, 缺失值比例: 0.0503375
特征: n3, 缺失值比例: 0.0503375
特征: n4, 缺失值比例: 0.04154875
特征: n5, 缺失值比例: 0.0503375
特征: n6, 缺失值比例: 0.0503375
特征: n7, 缺失值比例: 0.0503375
特征: n8, 缺失值比例: 0.05033875
特征: n9, 缺失值比例: 0.0503375
特征: n10, 缺失值比例: 0.04154875
特征: n11, 缺失值比例: 0.08719
特征: n12, 缺失值比例: 0.0503375
特征: n13, 缺失值比例: 0.0503375
特征: n14, 缺失值比例: 0.0503375
有缺失值特征共计: 22
========================================
test_dat缺失值情况=>
特征: employmentLength, 缺失值比例: 0.05871
特征: dti, 缺失值比例: 0.000305
特征: pubRecBankruptcies, 缺失值比例: 0.00058
特征: revolUtil, 缺失值比例: 0.000635
特征: n0, 缺失值比例: 0.050555
特征: n1, 缺失值比例: 0.050555
特征: n2, 缺失值比例: 0.050555
特征: n3, 缺失值比例: 0.050555
特征: n4, 缺失值比例: 0.04197
特征: n5, 缺失值比例: 0.050555
特征: n6, 缺失值比例: 0.050555
特征: n7, 缺失值比例: 0.050555
特征: n8, 缺失值比例: 0.050555
特征: n9, 缺失值比例: 0.050555
特征: n10, 缺失值比例: 0.04197
特征: n11, 缺失值比例: 0.087875
特征: n12, 缺失值比例: 0.050555
特征: n13, 缺失值比例: 0.050555
特征: n14, 缺失值比例: 0.050555
有缺失值特征共计: 19
```
***
### 3. 特征分析
#### 3.1 特征筛选
```
# 筛选查看离散型特征: 包括数值型离散特征和字符型数值特征
def stat_disp_feat(data):
    disp_feat_list = []
    for feat in data.columns:
        if data[feat].dtypes == "object":
            disp_feat_list.append(feat)
        else:
            if data[feat].nunique() <= 10:
                disp_feat_list.append(feat)
    return disp_feat_list
disp_feat_list = stat_disp_feat(test_dat)


# 获取连续型特征
conus_feat_list = []
for feat in test_dat.columns:
    if feat in disp_feat_list:
        continue
    elif feat == "id":
        continue
    else:
        conus_feat_list.append(feat)
```
#### 3.2 连续特征分布分析
**查看train_dat和test_dat连续特征的分布情况，删除分布不一致的特征**
```
dist_cols = 4
dist_rows = np.ceil(len(conus_feat_list) / dist_cols)
plt.figure(figsize=(dist_cols * 4, dist_rows * 4))
i = 0
for feat in conus_feat_list:
    i += 1
    plt.subplot(dist_rows, dist_cols, i)
    sns.kdeplot(train_dat[feat], color="Red", shade=True)
    sns.kdeplot(test_dat[feat], color="Blue", shade=True)
    plt.xlabel(feat + "密度图")
    plt.ylabel("")
    plt.legend(["train_dat", "test_dat"])
plt.show()
```
![]()
#### 3.3 离散特征分布分析
**统计每个特征的值的个数，找出只有一个值的特征，直接删除**
```
one_val_feat_train = [col for col in train_dat.columns if train_dat[col].nunique() <= 1]
one_val_feat_test = [col for col in test_dat.columns if test_dat[col].nunique() <= 1]
print(one_val_feat_train, one_val_feat_test)
```
```
['policyCode'] ['policyCode']
```
**画图查看train_dat,test_dat的离散特征分布, 针对分布特别有偏的特征需要删除**
```
dist_cols = 4
dist_rows = np.ceil(len(disp_feat_list)*2/ dist_cols)
plt.figure(figsize=(dist_cols * 4, dist_rows * 4))
i = 0
for feat in disp_feat_list:

    i += 1
    plt.subplot(dist_rows, dist_cols, i)
    sns.countplot(x=feat, data=train_dat)
    plt.title("Train_Data of " + feat)
    plt.xlabel("")
    plt.ylabel("")
    
    i += 1
    plt.subplot(dist_rows, dist_cols, i)
    sns.countplot(x=feat, data=test_dat)
    plt.title("Test_Data of " + feat)
    plt.xlabel("")
    plt.ylabel("")

plt.show()
```
![]()
#### 3.4 连续特征异常值分析
```
# 通过箱线图查看异常值情况
dist_cols = 6
dist_rows = np.ceil(len(conus_feat_list)*2/ dist_cols)
plt.figure(figsize=(dist_cols * 4, dist_rows * 4))
i = 0
for feat in conus_feat_list:

    i += 1
    plt.subplot(dist_rows, dist_cols, i)
    sns.boxplot(train_dat[feat], color="green", orient="v", width=0.5)
    plt.title("Train_Data of " + feat)
    plt.xlabel("")
    plt.ylabel("")
    
    i += 1
    plt.subplot(dist_rows, dist_cols, i)
    sns.boxplot(test_dat[feat], color="blue", orient="v", width=0.5)
    plt.title("Test_Data of " + feat)
    plt.xlabel("")
    plt.ylabel("")

plt.show()
```
![]()
#### 3.5 连续特征相关性查看
```
# 利用热力图进一步查看相关系数
plt.figure(figsize=(20, 12))
sns.heatmap(train_dat[conus_feat_list].corr(), annot=True, linewidth=0.5, fmt=".2f", cmap="YlGnBu")
plt.show()
```
```
# 查看连续变量间的相关系数
sns.pairplot(train_dat[conus_feat_list])
```


