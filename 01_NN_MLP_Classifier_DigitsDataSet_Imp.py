# MLP Classifier with Backpropagation [Neural Networks]
# Use the digits dataset available under SKLearn. 
# Train an MLP each with 1, 2, 3 and 4 hidden layers using Backpropagation to classify patterns of all the 10 classes.
# Take 80%. for training and the rest for testing. Compute the classificationcation accuracy on the test patterns.
# Repeat with different train_test_size.
# Pick the best MLP.

import numpy as np
import math
import random
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

#prepare the data set
digits = datasets.load_digits()

X_df = digits.data
y = digits.target 

X = (np.absolute(np.array(X_df)//8)).astype(np.int32)

#There are some 2's in the array, so we iterate over the array and convert the 2's to 1
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if(X[i][j] == 2):
            X[i][j] = 1

test_size_arr = [0.1, 0.2, 0.3, 0.4]
length = len(test_size_arr)

for i in range(4):
    #creating the MLP classifier
    if(i==0):
        clf = MLPClassifier(hidden_layer_sizes=(100),max_iter=100,alpha=1e-4,
                                solver="sgd", verbose=0,random_state=1,learning_rate_init=0.1,)
    if(i==1):
        clf = MLPClassifier(hidden_layer_sizes=(100,100),max_iter=100,alpha=1e-4,
                                solver="sgd", verbose=0,random_state=1,learning_rate_init=0.1,)
    if(i==2):
        clf = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=100,alpha=1e-4,
                                solver="sgd", verbose=0,random_state=1,learning_rate_init=0.1,)
    if(i==3):
        clf = MLPClassifier(hidden_layer_sizes=(100,100,100,100),max_iter=100,alpha=1e-4,
                                solver="sgd", verbose=0,random_state=1,learning_rate_init=0.1,)
    for j in range(length):     
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_arr[j],stratify=y)    
        clf.fit(X_train, y_train)
        print("Accuracy for MLP Classifier with", i+1,"Layers and test_size=", test_size_arr[j],"======> %f" % clf.score(X, y))

    fig, axes = plt.subplots(1, 1)
    axes.plot(clf.loss_curve_, 'o-')
    num_layers = i+1
    xlabel = "number of iteration with Layers:" + str(num_layers)
    axes.set_xlabel(xlabel)
    axes.set_ylabel("loss")
    plt.show()

