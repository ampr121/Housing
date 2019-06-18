import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

df = pd.read_csv('train.csv')

df = df[['SalePrice','GrLivArea','TotalBsmtSF']]

plt.scatter(df['TotalBsmtSF'],df['SalePrice'])

for colname, col in df.iteritems():
    df = df[(np.abs(stats.zscore(df['SalePrice']) > 3) & np.abs(stats.zscore(df[colname]) > 3)) | (
                np.abs(stats.zscore(df[colname]) <4))]

plt.scatter(df['TotalBsmtSF'],df['SalePrice'])
plt.show()
