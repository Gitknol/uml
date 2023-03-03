import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# read data

df=pd.read_csv('D:/uml/python/数据/DTW_prec.csv',header= 'infer')
#空数据删除
df.dropna(inplace=True)
#数据转化
df.index = pd.to_datetime(df['DATE'])
df=df['PRCP']
#日输出图像
ax=df.plot(kind= 'line',figsize=(15,3))
ax.set_title('Daily Precipitation (variance = %.4f)' % (df.var()))
plt.show()
#月输出图像
mouth = df.groupby(pd.Grouper(freq='M')).sum()
ax_1=mouth.plot(kind= 'line',figsize=(15,3))
ax_1.set_title('Monthly Precipitation (variance = %.4f)' % (mouth.var()))
plt.show()

#年输出图像
Year = df.groupby(pd.Grouper(freq='Y')).sum()
ax_2=Year.plot(kind= 'line',figsize=(15,3))
ax_2.set_title('Yearly Precipitation (variance = %.4f)' % (Year.var()))
plt.show()

