import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
plt.style.use('fivethirtyeight')

train = pd.read_csv('CleanTrain.csv')

train = train[['Id', 'LotArea', 'BldgType', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'GrLivArea', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'GarageCars', 'YrSold', 'SalePrice']]

train['Bath']=train['FullBath']+train['HalfBath']
train = train.drop(['FullBath','HalfBath'], axis=1)
train = pd.get_dummies(train)

X_train = train.drop(["SalePrice",'Id'], axis=1)
Y_train = train["SalePrice"]

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

#Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')

corr_matrix = train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
train = train.drop(train[to_drop], axis=1)

cor = train.corr()
cor_target = abs(cor["SalePrice"])
relevant_features = cor_target[cor_target>0.55]
train = train[['Id', 'OverallQual','GrLivArea','TotRmsAbvGrd','GarageCars','Bath','SalePrice']]
print(train.head())
print(len(train.columns))
train.to_csv('Feature_Selected_Data.csv')
