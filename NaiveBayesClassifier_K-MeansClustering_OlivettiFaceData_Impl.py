# Advanced Gaussian Naive Bayes classifier (NBC) -  with K-means++ clustering algorithm
# Download the Olivetti faces dataset. 
# Visit https://scikit-learn.org/0.19/datasets/olivetti_faces.html
# There are 40 classes (corresponding to 40 people), each class having 10 faces of the individual; so there are a total of 400 images. 
# Here each face is viewed as an imgae of size 64 × 64 (= 4096) pixels; each pixel having values 0 to 255 which are ultimately converted into floating numbers in the range [0,1]. 
# Split the dataset into train and test parts such that there are 320 images, 8 images per person (8 X 40 ) for training and 80 images, 2 images per person, (2 X 40) for testing.
# Repeat the experiment using 10 diferent random splits having 320 training faces and 80 test faces as specified above and report the average accuracy.
# Cluster the 4096 features into K = 1200; 1600; 2000; 3000 clusters using the K-means++ clustering algorithm. 
# So,320 X 4096 training data matrix is reduced to 320 X K matrix and 80 X 4096 test data matrix is reduced to 80 X K, for a given K. Classify the test data, of size
# 80 X K using the Gausssin NBC and report the test accuracy
# Use the Gaussian Naive Bayes classifier (NBC) to classify the test data and report the results

import numpy as np
import math
import random
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = fetch_olivetti_faces()

target = dataset.target  #numpy array of shape (400, )
faces = dataset.images   #numpy array of shape (400, 64, 64)
X =  dataset.data #numpy array of share (400,4096)



k_list = [1200,1600,2000,3000]
k_len = len(k_list)

for k in range(k_len):
    total_accuracy = 0
    for n in range(10):
        X_train_arr = []
        X_test_arr = []
        y_train_arr = []
        y_test_arr = []
        X_train_centroid_arr = []
        X_test_centroid_arr = []
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



        cluster =  KMeans(n_clusters= k_list[k], init = 'k-means++', random_state=0)
        cluster.fit(X_train_arr)
        #print(cluster.labels_)
        labels = cluster.labels_
        X_train_centroid_arr =cluster.cluster_centers_           
        print(X_train_centroid_arr.shape)   
            
        cluster_test =  KMeans(n_clusters= k_list[k], init = 'k-means++', random_state=0)
        cluster_test.fit(X_train_arr)
        #print(cluster.labels_)
        labels_test = cluster_test.labels_
        X_test_centroid_arr =cluster_test.cluster_centers_    
        print(X_test_centroid_arr.shape)       

    
        model = GaussianNB()
        model.fit(X_train_centroid_arr,y_train_arr)

        pred_i_test = model.predict(X_test_centroid_arr)
        accuracy = accuracy_score(y_test_arr,pred_i_test)

        total_accuracy = total_accuracy+accuracy

        print("accuracy_score for iteration", n+1, "and K value", k_list[k], "= ", accuracy)

    print("Average accuracy for  K value ", k_list[k], "= ", total_accuracy/10)

