# KNNC - Semisupervised applied over Digits Dataset
# Use the digits dataset available under SKLearn. 
# Consider the data corresponding to classes 0 and 1 only. 
# Each pattern is a 8 X 8 sized character where each value is an integer in the range 0 to 16. 
# Convert it into a binary form by replacing a value below 8 by 0 and other values (>= 8) by 1.
# Use 20 patterns from each class with labels and the remaining without the lables for this subtask. 
# Use the KNNC and label the patterns without lables.
# Obtain the % classification accuracy. Perform this taks with K values in the set { 1,3,5,10,20 }

import numpy as np
import math
import random
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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



n_list = [1,3,5,10,20]
length = len(n_list)
for p in range(length):
    total_accuracy = 0;
    for q in range(10):
        X0_train, X0_test, y0_train, y0_test = train_test_split(X0_new_arr, y0_new_arr, train_size=20,stratify=y0_new_arr)
        X1_train, X1_test, y1_train, y1_test = train_test_split(X1_new_arr, y1_new_arr, train_size=20,stratify=y1_new_arr)

        X_train = np.concatenate((X0_train, X1_train), axis=0)  
        X_test = np.concatenate((X0_test, X1_test), axis=0)  
        y_train = np.concatenate((y0_train, y1_train), axis=0)  
        y_test = np.concatenate((y0_test, y1_test), axis=0)  

        #create the KNNC and do the prediction 
        knn = KNeighborsClassifier(n_neighbors=n_list[p])
        knn.fit(X_train, y_train)
        pred_i_test = knn.predict(X_test)
        accuracy = accuracy_score(y_test,pred_i_test)
        total_accuracy = total_accuracy + accuracy
        
    print("Average Accuracy for n_neigbours=",n_list[p], " : ", total_accuracy/10)

