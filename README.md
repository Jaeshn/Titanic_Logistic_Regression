# Titanic_Logistic_Regression
**通过Logistic Regression预测Titanic乘客能否在事故中生还**

1. [导入工具库和数据](#t1.)
2. [查看缺失数据](#t2.)
   - 2.1. [年龄](#t2.1.)
   - 2.2. [仓位](#t2.2.)
   - 2.3. [登船地点](#t2.3.)
   - 2.4. [对数据进行调整](#t2.4.)
     - 2.4.1 [额外的变量](#t2.4.1.)
3. [数据分析](#t3.)
4. [Logistic Regression](#t4.)

<a id="t1."></a>
# 1. 导入工具库和数据
```python
import numpy as np 
import pandas as pd 

from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #设置seaborn画图的背景为白色
sns.set(style="whitegrid", color_codes=True)

# 将数据读入 DataFrame
df = pd.read_csv("./titanic_data.csv")

# 预览数据
df.head()
```



