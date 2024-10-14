# This is a classification task practice with KNNC.
# (a) Download the Olivetti faces dataset. There are 40 classes (corresponding to 40
# people), each class having 10 faces of the individual; so there are a total of 400
# images. Here each face is viewed as an imgae of size 64 × 64 (= 4096) pixels; each pixel having values 0 to 255 which are ultimateley converted into floating numbers in the range [0,1]. Visit https://scikit-learn.org/0.19/datasets/olivetti_faces.html for more details.
# 
# Using Bootstrapping generate 400 more samples in total by adding 1o more samples for each face
# 
# In this task you are supposed to reduce the dimensionality using l Principal Components
# projections with the values of l = 200; 400; 600; 800 .Use KNNC with values of
# K = 1; 3; 5; 10; 20; 100 on the 800×4096 data matrix obtained in problem 2 and step (d).
# For each value of K, use KNNC based on Minkowski distance with r = 1; 2; 1. Also
# consider fractional norms with r = 0:8; 0:5; 0:3. 
# Compute the percentage accuracy using Leave-one-out-strategy and report results

import numpy as np
import math
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


dataset = fetch_olivetti_faces()
target = dataset.target  #numpy array of shape (400, )
faces = dataset.images   #numpy array of shape (400, 64, 64)
X = dataset.data #numpy array of shape (400,4096)
X_dash = []

#Find three nearest neighbours
for j in range(400):
    sq_sum_arr = [0]*400
    sq_sum_arr_copy = []
    for i in range(400):
        sq_sum_arr[i] = np.sum(np.square(X[j] - X[i]))
        
    #work with indexes to get the slot of 10 within which we need to find 3 closest neiggbours 
    multiplier = j//10
    start = multiplier*10 
    end = start + 10

    for k in range(start, end):
        sq_sum_arr_copy.append(sq_sum_arr[k])

    #find 4 smallest values in this copy array (1 will be 0 as dist of point from itself is 0, other 3 will be closest points)
    idx = np.argpartition(sq_sum_arr_copy, 4)
    X_dash.append((X[multiplier*10 + idx[0]] + X[multiplier*10 + idx[1]] + X[multiplier*10 + idx[2]] + X[multiplier*10 + idx[3]])/4)


X_dash_arr = np.array(X_dash)
#print("X_dash shape:",X_dash_arr.shape)
 
#Create final Data Set XF of size 800 by appending X and X_dash
X_final = np.concatenate([X, X_dash])
target_array = np.array(target)
target_final = np.concatenate([target,target])
print("X shape:",X_final.shape)
print("target shape:",target_final.shape)

X_train, X_test, y_train, y_test = train_test_split(X_final, target_final, test_size=0.3,stratify=target_final)
print("X_train: ",X_train.shape)
print("X_test: ",X_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)


#Transform data with PCA
n_components_list = [200,400,600,800]
n_length = len(n_components_list)
for iter in range(n_length):
    pca=PCA(n_components=n_components_list[iter], whiten=True)

    pca.fit(X_final)
    X_final_pca=pca.transform(X_final)
    print(' ')
    print("X_pca: ",X_final_pca.shape)

    X_train_pca=pca.transform(X_train)
    X_test_pca=pca.transform(X_test)

    print("X_train_pca: ",X_train_pca.shape)
    print("X_test_pca: ",X_test_pca.shape)



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
            knn1.fit(X_train_pca, y_train)
            pred_i_test = knn1.predict(X_test_pca)
            pred_i_train = knn1.predict(X_train_pca)
            accuracy = accuracy_score(y_test,pred_i_test)
            print('For PCA n_component',n_components_list[iter],'For r =',r_list[j],'K =',n_list[i])
            print('accuracy_score:', accuracy)
            accuracy_score_arr.append(accuracy)  
            #Ealuating with leave one out method
            loo_cv = LeaveOneOut()
            cv_scores=cross_val_score(knn1,
                                 X_final_pca,
                                 target_final,
                                 scoring='accuracy',
                                 cv=loo_cv)
            loo_accuracy = mean(cv_scores)
            loo_accuracy_score.append(loo_accuracy)
            print('Leave One Out Accuracy: %.3f' % loo_accuracy)

        fig = plt.figure(figsize=(12, 6))
        plt.plot(n_list, loo_accuracy_score, color='red', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=10)
        plt.title('Graph of LOO Accuracy Score vs K Value on Bootstrapped Data')
        plt.xlabel('K Value')
        plt.ylabel('LOO Accuracy Score')
        if(r_list[j] == 1): 
            if(n_components_list[iter] == 200): 
                fig.savefig("D:\M_Implementations\KNNC_DimRed_PCA\PCA200_K_value_vs_loo_accuracy_r1_Comparison.png")
            if(n_components_list[iter] == 400): 
                fig.savefig("D:\M_Implementations\KNNC_DimRed_PCA\PCA400_K_value_vs_loo_accuracy_r1_Comparison.png")
            if(n_components_list[iter] == 600): 
                fig.savefig("D:\M_Implementations\KNNC_DimRed_PCA\PCA600_K_value_vs_loo_accuracy_r1_Comparison.png")
            if(n_components_list[iter] == 800): 
                fig.savefig("D:\M_Implementations\KNNC_DimRed_PCA\PCA800_K_value_vs_loo_accuracy_r1_Comparison.png")

        else:
            if(r_list[j] == 2):  
                if(n_components_list[iter] == 200): 
                    fig.savefig("D:\M_Implementations\KNNC_DimRed_PCA\PCA200_K_value_vs_loo_accuracy_r2_Comparison.png")
                if(n_components_list[iter] == 400): 
                    fig.savefig("D:\M_Implementations\KNNC_DimRed_PCA\PCA400_K_value_vs_loo_accuracy_r2_Comparison.png")
                if(n_components_list[iter] == 600): 
                    fig.savefig("D:\M_Implementations\KNNC_DimRed_PCA\PCA600_K_value_vs_loo_accuracy_r2_Comparison.png")
                if(n_components_list[iter] == 800): 
                    fig.savefig("D:\M_Implementations\KNNC_DimRed_PCA\PCA800_K_value_vs_loo_accuracy_r2_Comparison.png")
            else:
                if(n_components_list[iter] == 200): 
                    fig.savefig("D:\M_Implementations\KNNC_DimRed_PCA\PCA200_K_value_vs_loo_accuracy_rInf_Comparison.png")
                if(n_components_list[iter] == 400): 
                    fig.savefig("D:\M_Implementations\KNNC_DimRed_PCA\PCA400_K_value_vs_loo_accuracy_rInf_Comparison.png")
                if(n_components_list[iter] == 600): 
                    fig.savefig("D:\M_Implementations\KNNC_DimRed_PCA\PCA600_K_value_vs_loo_accuracy_rInf_Comparison.png")
                if(n_components_list[iter] == 800): 
                    fig.savefig("D:\M_Implementations\KNNC_DimRed_PCA\PCA800_K_value_vs_loo_accuracy_rInf_Comparison.png")