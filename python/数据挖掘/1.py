import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# read data

df=pd.read_csv('D:/uml/python/数据/Data.csv')
X=df.iloc[:,:-1].values
y = df.iloc[:, 3].values
imputer = SimpleImputer(missing_values= np.nan, strategy="mean")
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print(X)

