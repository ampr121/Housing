# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
# fix random seed for reproducibility
np.random.seed(7)
# load Feature Selected Data
dataset = pd.read_csv("Feature_Selected_Data.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset.iloc[:,1:6]
Y = dataset.iloc[:,6]
# create model
model = Sequential()
model.add(Dense(12, input_dim=5, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=50, batch_size=20)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
