## 数据探索
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
