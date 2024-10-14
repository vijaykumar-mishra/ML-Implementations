import numpy as np
import math
import random
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = fetch_olivetti_faces()

target = dataset.target  #numpy array of shape (400, )
faces = dataset.images   #numpy array of shape (400, 64, 64)
X = dataset.data #numpy array of share (400,4096)

test_size_arr = [0.2,0.25, 0.3,0.35]
num_trees_arr = [100, 200, 300, 400]


test_size_len = len(test_size_arr)
num_trees_len = len(num_trees_arr)

for j in range(num_trees_len):
    total_accuracy = 0
    for i in range(10):
        testsize = random.choice(test_size_arr)
        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=testsize,stratify=target)
        clf = RandomForestClassifier(n_estimators = num_trees_arr[j])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        total_accuracy = total_accuracy + accuracy

        importances = clf.feature_importances_
        indices = np.arange(4096) 
        print("Num Trees in Forest:", num_trees_arr[j])
        print("Test Size Ratio:", testsize)
        print("Accuracy :",accuracy)
        #print("Feature Importance  :",importances)
        
        if(num_trees_arr[j] == 100 and testsize == 0.20):
            fig = plt.figure(figsize=(12, 6))
            plt.title('Feature Importances')
            plt.barh(range(len(importances)), importances[indices], color='b', align='center')
            plt.xlabel('Relative Importance')
            fig.savefig("D:\IISC_Course\Assignment_3\Q2\Feature_Importance_for_Testsize20percent_numTrees100.png")
            
        if(num_trees_arr[j] == 200 and testsize == 0.3 ):
            fig = plt.figure(figsize=(12, 6))
            plt.title('Feature Importances')
            plt.barh(range(len(importances)), importances[indices], color='b', align='center')
            plt.xlabel('Relative Importance')
            fig.savefig("D:\IISC_Course\Assignment_3\Q2\Feature_Importance_for_Testsize30percent_numTrees200.png")
            
        if(num_trees_arr[j] == 300 and testsize == 0.20):
            fig = plt.figure(figsize=(12, 6))
            plt.title('Feature Importances')
            plt.barh(range(len(importances)), importances[indices], color='b', align='center')
            plt.xlabel('Relative Importance')
            fig.savefig("D:\IISC_Course\Assignment_3\Q2\Feature_Importance_for_Testsize20percent_numTrees300.png")
            
        if(num_trees_arr[j] == 400 and testsize == 0.3 ):
            fig = plt.figure(figsize=(12, 6))
            plt.title('Feature Importances')
            plt.barh(range(len(importances)), importances[indices], color='b', align='center')
            plt.xlabel('Relative Importance')
            fig.savefig("D:\IISC_Course\Assignment_3\Q2\Feature_Importance_for_Testsize30percent_numTrees400.png")
            
    print("avg accuracy for tree size",  num_trees_arr[j], " = ", total_accuracy/10)
