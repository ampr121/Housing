import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
plt.style.use('fivethirtyeight')
from scipy import stats

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

for c in train.columns:
    if sum(train[c].isnull())> len(train.index)*0.4:
        del train[c]

train = train.dropna()

missing = train.isna().sum()
#missing = missing[missing>0]
missing_perc = missing/train.shape[0]*100
na = pd.DataFrame([missing, missing_perc], index = ['missing_num', 'missing_perc']).T
na = na.sort_values(by = 'missing_perc', ascending = False)
na = na.reset_index()
na = na.rename(columns={'index':'name'})
#print(na)

train = train.dropna()

for colname, col in train.select_dtypes(exclude='object').iteritems():
    train = train[(np.abs(stats.zscore(train['SalePrice']) > 3) & np.abs(stats.zscore(train[colname]) > 3)) | (
                np.abs(stats.zscore(train[colname]) <4))]
train.to_csv('CleanTrain.csv')
