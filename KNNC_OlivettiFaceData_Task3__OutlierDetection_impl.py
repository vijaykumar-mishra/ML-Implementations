# In this task we would like to detect the outliers in the olivetti faces data as follows:
# (a) Consider Xi 2 X for i = 1; 2; · · · ; 400.
# (b) Let Xi1; Xi2; · · · ; Xi9 be the 9 nearest neighbors of Xi from the remaining 399 patterns of X .
# (c) Classify Xi as outlier if all of its 9 neaarest neighbors are from classes other than that of Xi.
# (d) Classify Xi as a possible outlier if at least 5 and at most 8 out of its 9 neighbors are from classes other than that of Xi.
# (e) Classify Xi as core otherwise; that is when 5 or more out of its 9 neighbors are from the same class as that of Xi.
# (f) So, each Xi is either an outlier, or a possible outlier, or a core pattern.
# (g) For each variant of KNNC specified in Task 1 (b), find out the number of core, outlier, possible outlier patterns out of 400 patterns in X. 

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
X = dataset.data #numpy array of shape (400,4096)

print("Olivetti faces input data arrays================")
print("faces: ", faces.shape)
print("target: ", target.shape)
print("data shape:",X.shape)



n_list = [10,20,100]
length = len(n_list)

for iter in range(length):    
    core_count = 0
    outlier_count = 0
    possible_outlier_count = 0
    #Find nearest neighbours
    for j in range(400):
        sq_sum_arr = [0]*400
        sq_sum_arr_copy = []
        for i in range(400):
            sq_sum_arr[i] = np.sum(np.square(X[j] - X[i]))

        sq_sum_arr_copy = sq_sum_arr.copy()

        #Finding k nearest neighbours
        #find 10 smallest values in this copy array (1 will be 0 as dist of point from itself is 0, other 9 will be closest points)
        idx = np.argpartition(sq_sum_arr_copy, n_list[iter])

        #check whether point is core/outlier/possible outlier
        refernce_class = target[j]
        #count number of reference class entries in the point as well as 9 nearest neighbours
        refernce_class_count = 0
        for k in range(n_list[iter]):
            if(refernce_class == target[idx[k]]):
                refernce_class_count = refernce_class_count + 1

        #update the count for 
        if(refernce_class_count == 1):
            outlier_count = outlier_count + 1        
        else:
            if(refernce_class_count >= (n_list[iter]/2)):
                core_count = core_count + 1  
            else:
                possible_outlier_count = possible_outlier_count + 1
    print(" ")
    print("For KNN with n_neighbors == ",n_list[iter])
    print("Numer of Core Points:",core_count)
    print("Numer of Outlier Points:",outlier_count)
    print("Numer of Possible Outlier Points:",possible_outlier_count)
