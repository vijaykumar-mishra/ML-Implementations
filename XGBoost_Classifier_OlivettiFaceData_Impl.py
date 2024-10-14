# XGBoost Classifier
# Download the Olivetti faces dataset. 
# Visit https://scikit-learn.org/0.19/datasets/olivetti_faces.html
# There are 40 classes (corresponding to 40 people), each class having 10 faces of the individual; so there are a total of 400 images. 
# Here each face is viewed as an imgae of size 64 × 64 (= 4096) pixels; each pixel having values 0 to 255 which are ultimately converted into floating numbers in the range [0,1]. 
# 
# Split the dataset into train
# and test parts. Do this splitting randomly 10 times and report the average accuracy.
# You may vary the test and train dataset sizes.
# 
# Use XGBoost classifier to classify the test dataset. Tune any asso-
# ciated parameters. Get the accuracy on the test dataset. Obtain the importance of features.

import numpy as np
import math
import random
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import XGBRFRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = fetch_olivetti_faces()

target = dataset.target  #numpy array of shape (400, )
faces = dataset.images   #numpy array of shape (400, 64, 64)
X = dataset.data #numpy array of share (400,4096)

test_size_arr = [0.2, 0.25, 0.3, 0.35]
num_trees_arr = [100, 200, 300, 400]

test_size_len = len(test_size_arr)
num_trees_len = len(num_trees_arr)

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2,stratify=target)

#XGBoost hyper-parameter tuning

param_tuning = {
    'learning_rate': [0.1],
    'max_depth': [10, 20, 50],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.5, 0.7],
    'colsample_bytree': [0.5, 0.7],
    'n_estimators' : [100, 200, 400],
}

xgb_model = XGBClassifier()

grid_search = GridSearchCV(estimator = xgb_model,
                       param_grid = param_tuning,                        
                       cv = 4,
                       n_jobs = -1,
                       verbose = 1,
                       scoring = "accuracy")

grid_search.fit(X_train,y_train)
xgb_best = grid_search.best_estimator_
print(xgb_best)
total_accuracy = 0
for i in range(10):
    testsize = random.choice(test_size_arr)
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=testsize,stratify=target)        
    xgb_best.fit(X_train, y_train)
    y_pred = xgb_best.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    total_accuracy = total_accuracy + accuracy

    importances = xgb_best.feature_importances_
    print("Test Size Ratio:", testsize)
    print("Accuracy :",accuracy)
    #print("Feature Importance  :",importances)

    if(testsize == 0.2):
        fig = plt.figure(figsize=(12, 6))
        plt.title('XG Boost Feature Importances')
        plt.barh(range(len(importances)), importances[indices], color='b', align='center')
        plt.xlabel('Relative Importance')
        #fig.savefig("D:\ML_Implementations\XGBoost\XGBoost_Feature_Importance_for_Testsize20percent.png")

    if(testsize == 0.25):
        fig = plt.figure(figsize=(12, 6))
        plt.title('XG Boost Feature Importances')
        plt.barh(range(len(importances)), importances[indices], color='b', align='center')
        plt.xlabel('Relative Importance')
        #fig.savefig("D:\ML_Implementations\XGBoost\XGBoost_Feature_Importance_for_Testsize25percen.png")

    if(testsize == 0.3):
        fig = plt.figure(figsize=(12, 6))
        plt.title('XG Boost Feature Importances')
        plt.barh(range(len(importances)), importances[indices], color='b', align='center')
        plt.xlabel('Relative Importance')
        #fig.savefig("D:\ML_Implementations\XGBoost\XGBoost_Feature_Importance_for_Testsize30percent.png")

    if(testsize == 0.35):
        fig = plt.figure(figsize=(12, 6))
        plt.title('XG Boost Feature Importances')
        plt.barh(range(len(importances)), importances[indices], color='b', align='center')
        plt.xlabel('Relative Importance')
        #fig.savefig("D:\ML_Implementations\XGBoost\XGBoost_Feature_Importance_for_Testsize35percent.png")

print("avg accuracy = ", total_accuracy/10)

