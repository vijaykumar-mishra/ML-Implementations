# Decision Tree
# Download the Olivetti faces dataset. 
# Visit https://scikit-learn.org/0.19/datasets/olivetti_faces.html
# There are 40 classes (corresponding to 40 people), each class having 10 faces of the individual; so there are a total of 400 images. 
# Here each face is viewed as an imgae of size 64 × 64 (= 4096) pixels; each pixel having values 0 to 255 which are ultimately converted into floating numbers in the range [0,1]. 
# 
# Split the dataset into train
# and test parts. Do this splitting randomly 10 times and report the average accuracy.
# You may vary the test and train dataset sizes.
# 
# Build a decision tree using the training data. Tune the parameters
# corresponding to pruning the decision tree. Use the best decision tree to classify
# the test dataset and obtain the accuracy. Use misclassification impurity also.

import numpy as np
import math
import random
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = fetch_olivetti_faces()

target = dataset.target  #numpy array of shape (400, )
faces = dataset.images   #numpy array of shape (400, 64, 64)
X = dataset.data #numpy array of share (400,4096)

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2,stratify=target)
dt = DecisionTreeClassifier(random_state=42)
params = {
    'max_depth': [5, 10, 20, 50, 100],
    'min_samples_leaf': [1, 2, 5, 10, 20],
    'criterion': ["gini", "entropy"]
}
grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")

grid_search.fit(X_train, y_train)
dt_best = grid_search.best_estimator_
print(dt_best)

test_size_arr = [0.2, 0.25, 0.3, 0.35]
total_accuracy = 0
for i in range(10):
    testsize = random.choice(test_size_arr)
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=testsize,stratify=target)
    dt_best.fit(X_train, y_train)
    y_pred = dt_best.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    total_accuracy = total_accuracy + accuracy

    print("Test Size Ratio:", testsize)
    print("Accuracy :",accuracy)

print("avg accuracy  = ", total_accuracy/10)