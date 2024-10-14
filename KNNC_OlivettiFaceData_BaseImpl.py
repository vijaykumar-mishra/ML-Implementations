# This is a classification task using KNNC.
# (a) Download the Olivetti faces dataset. There are 40 classes (corresponding to 40
# people), each class having 10 faces of the individual; so there are a total of 400
# images. Here each face is viewed as an image of size 64 × 64 (= 4096) pixels; each pixel having values 0 to 255 which are ultimately converted into floating numbers in the range [0,1]. 
# Visit https://scikit-learn.org/0.19/datasets/olivetti_faces.html for more details.
# (b) Use KNNC with values of K = 1; 3; 5; 10; 20; 100. For each value of K, use KNNC based on Minkowski distance with r = 1; 2; 1. 
# Also consider fractional norms with r = 0:8; 0:5; 0:3. Compute the percentage accuracy using Leaveone-out-strategy and report results..

import numpy as np
import math
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


dataset = fetch_olivetti_faces()

target = dataset.target  #numpy array of shape (400, )
faces = dataset.images   #numpy array of shape (400, 64, 64)
X = dataset.data #numpy array of share (400,4096)

print("faces: ", faces.shape)
print("target: ", target.shape)
print("data shape:",X.shape)


X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3,stratify=target)
print("x_train: ",X_train.shape)
print("x_test: ",X_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)



r_inf = math.inf
r_list = [1,2, r_inf]
r_length = len(r_list)

for j in range(r_length):
    n_list = [1,3,5,10,20,100]
    length = len(n_list)
    accuracy_score_arr = []
    loo_accuracy_score = []
    for i in range(length):
        knn1 = KNeighborsClassifier(n_neighbors = n_list[i], p = r_list[j], metric='minkowski')
        knn1.fit(X_train, y_train)
        pred_i_test = knn1.predict(X_test)
        pred_i_train = knn1.predict(X_train)
        accuracy = accuracy_score(y_test,pred_i_test)
        print('For r =',r_list[j],'K =',n_list[i])
        print('accuracy_score:', accuracy)
        accuracy_score_arr.append(accuracy)  
        #Ealuating with leave one out method
        loo_cv = LeaveOneOut()
        cv_scores=cross_val_score(knn1,
                             X,
                             target,
                             scoring='accuracy',
                             cv=loo_cv)
        loo_accuracy = mean(cv_scores)
        loo_accuracy_score.append(loo_accuracy)
        print('Leave One Out Accuracy: %.3f' % loo_accuracy) 


    #print(accuracy_score_arr)
    fig = plt.figure(figsize=(12, 6))
    plt.plot(n_list, loo_accuracy_score, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Graph of LOO Accuracy Score vs K Value')
    plt.xlabel('K Value')
    plt.ylabel('LOO Accuracy Score')
    if(r_list[j] == 1):  
        fig.savefig("D:\ML_Implementations\OlivettiBase_Impl\K_value_vs_loo_accuracy_r1_Comparison.png")
    else:
        if(r_list[j] == 2):  
            fig.savefig("D:\ML_Implementations\OlivettiBase_Impl\K_value_vs_loo_accuracy_r2Comparison.png")
        else:
            fig.savefig("D:\ML_Implementations\OlivettiBase_Impl\K_value_vs_loo_accuracy_rInfinity_Comparison.png")
            
