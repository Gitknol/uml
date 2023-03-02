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
ax=df.plot(kind= 'line',figsize=(15,3))
ax.set_title('Daily Precipitation (variance = %.4f)' % (df.var()))
plt.show()

