import numpy as np 
import pandas as pd
from keras.models import Sequential
from tensorflow.keras.layers import Input
from keras.layers import Dense
data=pd.read_csv('csv_data/concrete_data.csv')
print(data.head())
print(data.shape)
print(data.columns)
data_column = data.columns
predictors = data[data_column[data_column!='Strength']]
target=data['Strength']
from sklearn.model_selection import train_test_split
print(predictors.shape)
predictors_normalize=((predictors-predictors.mean())/predictors.std())
input_params = predictors.shape[1]
print(predictors_normalize.head())

def regression_model():
    model = Sequential()
    model.add(Input(shape=(input_params,)))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mse')
    return model
mse=[]
for i in range(50):
    print('Iteration {}'.format(i+1))
    xtrain,xtest,ytrain,ytest=train_test_split(predictors_normalize,target,test_size=0.3,random_state=i+1)
    model=regression_model()
    model.fit(xtrain,ytrain,epochs=50,verbose=0)
    score=model.evaluate(xtest,ytest,verbose=1)
    mse.append(score)
print('Mean of the mean squared error {:.4f}'.format(np.mean(mse)))
print('Standard Deviation of the mean squared error {:.4f}'.format(np.std(mse)))
