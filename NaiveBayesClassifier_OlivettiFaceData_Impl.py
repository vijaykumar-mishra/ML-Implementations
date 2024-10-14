# Gaussian Naive Bayes classifier (NBC)
# Download the Olivetti faces dataset. 
# Visit https://scikit-learn.org/0.19/datasets/olivetti_faces.html
# There are 40 classes (corresponding to 40 people), each class having 10 faces of the individual; so there are a total of 400 images. 
# Here each face is viewed as an imgae of size 64 × 64 (= 4096) pixels; each pixel having values 0 to 255 which are ultimately converted into floating numbers in the range [0,1]. 
# Split the dataset into train and test parts such that there are 320 images, 8 images per person (8 X 40 ) for training and 80 images, 2 images per person, (2 X 40) for testing.
# Repeat the experiment using 10 diferent random splits having 320 training faces and 80 test faces as specified above and report the average accuracy
# Use the Gaussian Naive Bayes classifier (NBC) to classify the test data and report the results

import numpy as np
import math
import random
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = fetch_olivetti_faces()

target = dataset.target  #numpy array of shape (400, )
faces = dataset.images   #numpy array of shape (400, 64, 64)
X =  dataset.data #numpy array of share (400,4096)

X_train_arr = []
X_test_arr = []
y_train_arr = []
y_test_arr = []

total_accuracy = 0

for n in range(10):
    for i in range(40):
        X_inp = []
        target_inp = []
        for j in range(10):
            ind = i*10+j
            X_inp.append(X[ind,:])
            target_inp.append(target[ind])

        X_train, X_test, y_train, y_test = train_test_split(X_inp, target_inp, test_size=0.2,stratify=target_inp)
        X_train_arr.extend(X_train)
        X_test_arr.extend(X_test)
        y_train_arr.extend(y_train)
        y_test_arr.extend(y_test)


    #print(y_test_arr)

    model = GaussianNB()
    model.fit(X_train_arr,y_train_arr)

    pred_i_test = model.predict(X_test_arr)
    accuracy = accuracy_score(y_test_arr,pred_i_test)
    total_accuracy = total_accuracy+accuracy
    
    print("accuracy_score for iteration", n+1,  accuracy)
    
print("Avearge accuracy  = ", total_accuracy/10)

