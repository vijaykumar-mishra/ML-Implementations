import random
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


#Generate a data set of 120 random numbers
random_dataset = np.random.uniform(low=0, high=1, size=120)

#Convert the data set to array
dataset_arr = np.array(random_dataset)

#create a data frame and populate the Y data with y = x*x + X + 1
df = pd.DataFrame(dataset_arr, columns = ['X'])
df['Y'] = df['X']*df['X'] + df['X'] + 1

#Verification of data generated
#print(df)
#df.to_csv("D:/ML_Implementations/Impl_1/regression.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=80, test_size=40)

#Get KNN Regressor and do the regressions for nearest neighbours values [1,3,4,5]
error_arr = []
n_list = [1,3,4,5]
length = len(n_list)
for i in range(length):
    knn_regressor = KNeighborsRegressor(n_neighbors=n_list[i])
    knn_regressor.fit(X_train, y_train)
    y_pred = knn_regressor.predict(X_test)
    error = 0
	# calculate total squared error
    for i in range(0,39):  
        error = error + abs(y_test[i] - y_pred[i])**2
    error_arr.append(error)
    
fig = plt.figure(figsize=(12, 6))
plt.plot(n_list, error_arr, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Graph of Total Squared Error vs K Value')
plt.xlabel('K Value')
plt.ylabel('Total Squared Error')

fig.savefig("D:\\ML_Implementations\\Impl_1\\K_value_vs_TotalSquaredError_Comparison.png")
    
print(error_arr)
