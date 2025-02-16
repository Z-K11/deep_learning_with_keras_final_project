import numpy as np 
import pandas as pd
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
data=pd.read_csv('csv_data/concrete_data.csv')
print(data.head())
print(data.shape)
print(data.columns)
data_column = data.columns
predictors = data[data_column[data_column!='Strength']]
target=data['Strength']
from sklearn.model_selection import train_test_split
xtrain,ytrain,xtest,ytest=train_test_split(predictors,target,test_size=30,random_state=4)
print(predictors.shape)
