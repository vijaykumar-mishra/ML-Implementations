# MLP Classifier with Varying ACttivation functions
# Use the digits dataset available under SKLearn. 
# Train an MLP each with 1, 2, 3 and 4 hidden layers using Backpropagation to classify patterns of all the 10 classes.
# Take 80%. for training and the rest for testing. Compute the classificationcation accuracy on the test patterns.
# Repeat with different train_test_size.
# Build a 10-class classifier based on varying the following:
# (a) Number of nodes in the hidden layers.
# (b) Tanh, Relu, and Logistic activation functions
# (c) Hyperparameters: Momentum term, Early stopping, and Learning Rate
# Share results and pick best activation functions

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

activation_function_arr = ['tanh', 'relu', 'logistic']
length = len(activation_function_arr)
for c in range(length):
    print("Activation Function =",activation_function_arr[c])
    print("======================================================")
    for i in range(7):
        #creating the MLP classifier
        if(i==0):
            print(" using default momentum = 0.9 having 1 Layer having 100 nodes per layer, Costant Learning Rate")
            clf = MLPClassifier(activation=activation_function_arr[c],hidden_layer_sizes=(100),max_iter=100,alpha=1e-4,
                                    solver="sgd", verbose=0,random_state=1,learning_rate_init=0.1,)
        if(i==1):
            print(" using default momentum = 0.9 having 2 Layer having [100,80] nodes per layer, Costant Learning Rate")
            clf = MLPClassifier(activation=activation_function_arr[c],hidden_layer_sizes=(100,80),max_iter=100,alpha=1e-4,
                                    solver="sgd", verbose=0,random_state=1,learning_rate_init=0.1,)
        if(i==2):
            print(" using default momentum = 0.9 having 3 Layer having [100,80,50] nodes per layer, Costant Learning Rate")
            clf = MLPClassifier(activation=activation_function_arr[c],hidden_layer_sizes=(100,80,50),max_iter=100,alpha=1e-4,
                                    solver="sgd", verbose=0,random_state=1,learning_rate_init=0.1,)
        if(i==3):
            print(" using default momentum = 0.9 having 4 Layer having [100,80,50,20] nodes per layer, Costant Learning Rate")
            clf = MLPClassifier(activation=activation_function_arr[c],hidden_layer_sizes=(100,80,50,20),max_iter=100,alpha=1e-4,
                                    solver="sgd", verbose=0,random_state=1,learning_rate_init=0.1,)
        if(i==4):        
            print(" using default momentum(0.9), 4 Hidden Layers having [100,100,100,100] nodes per layer, Adaptive Learning rate")
            clf = MLPClassifier(activation=activation_function_arr[c],hidden_layer_sizes=(100,100,100,100),momentum=0.8, max_iter=100,alpha=1e-4,
                                learning_rate="adaptive", solver="sgd", verbose=0,random_state=1,learning_rate_init=0.1,)
        if(i==5):        
            print(" using momentum 0.8 with 4 Hidden Layers, having 100,100,100,100 nodes per layer, Costant Learning Rate")
            clf = MLPClassifier(activation=activation_function_arr[c],hidden_layer_sizes=(100,100,100,100),momentum=0.8, max_iter=100,alpha=1e-4,
                                    solver="sgd", verbose=0,random_state=1,learning_rate_init=0.1,)

        if(i==6):        
            print(" using momentum 0.7 with 4 Hidden Layers, having 100,100,100,100 nodes per layer, Costant Learning Rate")
            clf = MLPClassifier(activation=activation_function_arr[c],hidden_layer_sizes=(100,100,100,100),momentum=0.7,max_iter=100,alpha=1e-4,
                                    solver="sgd", verbose=0,random_state=1,learning_rate_init=0.1,)



        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)    
        clf.fit(X_train, y_train)
        print(" ===> Accuracy for MLP Classifier train size 0.8 = %f" % clf.score(X, y))

        fig, axes = plt.subplots(1, 1)
        axes.plot(clf.loss_curve_, 'o-')
        if(i <= 2):
            num_layers = i+1
        else:
            num_layers = 4
        xlabel = "number of iteration with Layers:" + str(num_layers) + " and activation function: " + activation_function_arr[c]
        axes.set_xlabel(xlabel)
        axes.set_ylabel("loss")
        plt.show()

