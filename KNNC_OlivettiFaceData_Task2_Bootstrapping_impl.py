import numpy as np
import math
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


dataset = fetch_olivetti_faces()
target = dataset.target  #numpy array of shape (400, )
faces = dataset.images   #numpy array of shape (400, 64, 64)
X = dataset.data #numpy array of shape (400,4096)
X_dash = []

print("faces: ", faces.shape)
print("target: ", target.shape)
print("data shape:",X.shape)

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
print("X_final shape:",XF.shape)
print("target_final shape:",target_final.shape)

X_train, X_test, y_train, y_test = train_test_split(X_final, target_final, test_size=0.3,stratify=target_final)
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
                             X_final,
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
        fig.savefig("D:\IISC_Course\Assignment_2\Bootstrapped_K_value_vs_loo_accuracy_r1_Comparison.png")
    else:
        if(r_list[j] == 2):  
            fig.savefig("D:\IISC_Course\Assignment_2\Bootstrapped_K_value_vs_loo_accuracy_r2Comparison.png")
        else:
            fig.savefig("D:\IISC_Course\Assignment_2\Bootstrapped_K_value_vs_loo_accuracy_rInfinity_Comparison.png")