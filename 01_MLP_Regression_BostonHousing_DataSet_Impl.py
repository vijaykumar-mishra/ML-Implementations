# MLP for regression on the boston housing dataset.
# Download the data from https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html 
# There are 506 patterns and 14 features with MEDV as the target variable.
# (a) Train an MLP net to get the best squared error value on the regression datasetin using 1, 2, 3, and 4 hidden layers.
# (b) Vary the parameters considered in Subtask 1(d) and compute the squared error.

import numpy as np
import math
import random
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

#prepare the data set
df = pd.read_csv('BostonHousingData.csv')

X = df.iloc[:, [5,10,12]].values
y = df.iloc[:, 13].values

index = np.where(np.round(stats.zscore(y),2) < 2.7)
y = y[index]
X = X[index]


activation_function_arr = ['relu', 'tanh', 'logistic']
length = len(activation_function_arr)
for c in range(length):
    print("Activation Function =",activation_function_arr[c])
    print("======================================================")
    for i in range(4):
        #creating the MLP regressor
        if(i==0):
            print(" 1 Layer having 64 nodes per layer")
            nr = MLPRegressor(activation=activation_function_arr[c],hidden_layer_sizes=(64),max_iter=100)
        if(i==1):
            print(" 2 Layer having [64,32] nodes per layer")
            nr = MLPRegressor(activation=activation_function_arr[c],hidden_layer_sizes=(64,32),max_iter=100)
        if(i==2):
            print(" 3 Layer having [64,32,16] nodes per layer")
            nr = MLPRegressor(activation=activation_function_arr[c],hidden_layer_sizes=(64,32,16),max_iter=100)
        if(i==3):
            print(" 4 Layer having [64,32,16,8] nodes per layer")
            nr = MLPRegressor(activation=activation_function_arr[c],hidden_layer_sizes=(64,32,16,8),max_iter=100)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)   
        
        sc = StandardScaler()        
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        nr.fit(X_train, y_train)
        
        y_pred = nr.predict(X_test)
        #print(y_pred)
        #print(y_test)
        
        rmse =  np.round(np.sqrt(mean_squared_error(y_test,y_pred)),2)
        
        print(" ===> RMSE  for MLP regressor train size 0.8 = %f" %rmse)

