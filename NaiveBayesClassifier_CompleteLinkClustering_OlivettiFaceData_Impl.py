import numpy as np
import math
import random
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
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

        #Get Xtrain_centroid_arr of shape 320*k
        for row in range(320):
            X_train_samples = X_train_arr[row].reshape(4096,-1)

            cluster = AgglomerativeClustering(n_clusters=k_list[k], affinity='euclidean', linkage='complete')
            cluster.fit_predict(X_train_samples)
            #print(cluster.labels_)
            labels = cluster.labels_
            #print(len(labels))

            sum_arr = [0]*k_list[k]
            cnt_arr = [0]*k_list[k]
            X_train_centroid = [0]*k_list[k]



            for t in range (4096) :
                ind = labels[t]
                sum_arr[ind] += X_train_samples[t]
                cnt_arr[ind] += 1
                X_train_centroid[ind] = sum_arr[ind]/cnt_arr[ind]
           
            X_train_centroid_arr.extend(X_train_centroid)
            
        X_tr = np.array(X_train_centroid_arr).reshape(320,k_list[k])
        #print(len(X_tr[0]))
        #print(X_tr.shape)  
            
        #Get Xtest_centroid_arr of shape 80*k
        for row in range(80):
            X_test_samples = X_test_arr[row].reshape(4096,-1)

            cluster_test = AgglomerativeClustering(n_clusters=k_list[k], affinity='euclidean', linkage='complete')
            cluster_test.fit_predict(X_test_samples)
            #print(cluster.labels_)
            labels_test = cluster_test.labels_
            #print(len(labels))

            sum_arr_test = [0]*k_list[k]
            cnt_arr_test = [0]*k_list[k]
            X_test_centroid = [0]*k_list[k]

            for u in range (4096) :
                ind = labels_test[u]
                sum_arr_test[ind] += X_test_samples[u]
                cnt_arr_test[ind] += 1
                X_test_centroid[ind] = sum_arr_test[ind]/cnt_arr_test[ind]

            X_test_centroid_arr.append(X_test_centroid)
            
        X_tst = np.array(X_test_centroid_arr).reshape(80,k_list[k])
        #print(X_tst.shape)   
    
        model = GaussianNB()
        model.fit(X_tr,y_train_arr)

        pred_i_test = model.predict(X_tst)
        accuracy = accuracy_score(y_test_arr,pred_i_test)

        total_accuracy = total_accuracy+accuracy

        print("accuracy_score for iteration", n+1, "and K value", k_list[k], "= ", accuracy)

    print("Average accuracy for  K value ", k_list[k], "= ", total_accuracy/10)
