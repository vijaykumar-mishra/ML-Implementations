# KNNC Classification
# Randomly generate 120 values of x in the range [0,1]. Let them be x1, x2, · · · , x120
#   i. Label the first 80 points {x1, · · · , x80} as follows. If xi ≤ 0.5 then xi ∈ Class1, else (if xi > 0.5) xi ∈ Class2 for i = 1, 2, · · · , 80.
#   ii. Classify the remaining points, that is x81, · · · , x120 using kNNC. Do this for k = 1, k = 3, k = 4, k = 5, k = 40, k = 80.
#   iii. Compute classification accuracy.

import random
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

#Generate a data set of 120 random numbers
random_dataset = np.random.uniform(low=0, high=1, size=120)

#Convert the data set to array
dataset_arr = np.array(random_dataset)

#create a data frame and classify the data in class type 1 or 2
df = pd.DataFrame(dataset_arr, columns = ['Data'])
df['Class'] = np.where(df['Data']<=0.5, 'Class1', 'Class2')

#Verification of data generated
#print(df)
#df.to_csv("D:\IISC_Course\Assignment_1\classification.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=80, test_size=40)

#Get KNN Classifier and do the predictions for nearest neighbours =5
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Accuracy Score 
accuracy = accuracy_score(y_test,y_pred)
print('accuracy_score : ', accuracy)


#Accuracy comparison for various values of n (1,3,4,5,40,80)
n_list = [1,3,4,5,40,80]
length = len(n_list)
error = []
accuracy_score_arr = []

for i in range(length):
    knn = KNeighborsClassifier(n_neighbors=n_list[i])
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    accuracy_score_arr.append(accuracy_score(y_test, pred_i))
    
fig = plt.figure(figsize=(12, 6))
plt.plot(n_list, accuracy_score_arr, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Graph of Accuracy Score vs K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy Score')

fig.savefig("D:\KNNC_Classification\Impl_1\K_value_vs_accuracy_Comparison.png")

print(accuracy_score_arr)


