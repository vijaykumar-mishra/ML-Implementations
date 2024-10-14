# Linear SVM Classifier
# Use the digits dataset available under SKLearn. 
# Consider the data corresponding to classes 0 and 1 only. 
# Each pattern is a 8 X 8 sized character where each value is an integer in the range 0 to 16. 
# Convert it into a binary form by replacing a value below 8 by 0 and other values (>= 8) by 1.
# split the dataset into train and test parts. 
# Do this splitting randomly 10 times and report the average accuracy using Linear SVM classifiers.

import numpy as np
import math
import random
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


#prepare the data set
digits = datasets.load_digits()

X_digits = digits.data
y_digits = digits.target 

X0_new = []
y0_new = []
X1_new = []
y1_new = []

for i in range(X_digits.shape[0]):
    if(y_digits[i] == 0) :
        X0_new.append(X_digits[i])
        y0_new.append(y_digits[i])
    else :
        if(y_digits[i] == 1) :
            X1_new.append(X_digits[i])
            y1_new.append(y_digits[i])

# In order to convert the array values to 0/1, just floor division with 8 shall suffice           
X0_new_arr = (np.absolute(np.array(X0_new)//8)).astype(int)
X1_new_arr = (np.absolute(np.array(X1_new)//8)).astype(int)
y0_new_arr = np.array(y0_new)
y1_new_arr = np.array(y1_new)


#There are some 2's in the array, so we iterate over the array and convert the 2's to 1
for i in range(X0_new_arr.shape[0]):
    for j in range(X0_new_arr.shape[1]):
        if(X0_new_arr[i][j] == 2):
            X0_new_arr[i][j] = 1

for m in range(X1_new_arr.shape[0]):
    for n in range(X1_new_arr.shape[1]):
        if(X1_new_arr[m][n] == 2):
            X1_new_arr[m][n] = 1   

#creating data frames for X anad y to be used in our perceptron model
X = np.concatenate((X0_new_arr, X1_new_arr), axis=0)
y = np.concatenate((y0_new_arr, y1_new_arr), axis=0)

test_size_arr = [0.1, 0.2, 0.3, 0.4, 0.5]
length = len(test_size_arr)

for p in range(length):
    total_accuracy = 0;
    for q in range(10):
        #creating training and testing dataframes from the created data for class 0 and 1 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)

        #Create a svm Classifier and do the training classification and accuracy calculations
        clf = svm.SVC(kernel='linear') # Linear Kernel
        clf.fit(X_train, y_train)
        pred_i_test = clf.predict(X_test)
        accuracy = accuracy_score(y_test,pred_i_test)
        total_accuracy = total_accuracy + accuracy
        
    print("Average Accuracy for SVM Classifier for test_size=",test_size_arr[p], " : ", total_accuracy/10)

