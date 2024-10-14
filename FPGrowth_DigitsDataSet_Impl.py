import numpy as np
import math
import random
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth


#prepare the data set
digits = datasets.load_digits()

X_digits = digits.data
y_digits = digits.target 

X0_new = []
y0_new = []
X1_new = []
y1_new = []

for i in range(X_digits.shape[0]):
    if(y_digits[i] == 0) :
        X0_new.append(X_digits[i])
        y0_new.append(y_digits[i])
    else :
        if(y_digits[i] == 1) :
            X1_new.append(X_digits[i])
            y1_new.append(y_digits[i])

# In order to convert the array values to 0/1, just floor division with 8 shall suffice           
X0_new_arr = (np.absolute(np.array(X0_new)//8)).astype(int)
X1_new_arr = (np.absolute(np.array(X1_new)//8)).astype(int)
y0_new_arr = np.array(y0_new)
y1_new_arr = np.array(y1_new)


#There are some 2's in the array, so we iterate over the array and convert the 2's to 1
for i in range(X0_new_arr.shape[0]):
    for j in range(X0_new_arr.shape[1]):
        if(X0_new_arr[i][j] == 2):
            X0_new_arr[i][j] = 1

for m in range(X1_new_arr.shape[0]):
    for n in range(X1_new_arr.shape[1]):
        if(X1_new_arr[m][n] == 2):
            X1_new_arr[m][n] = 1   

i=0
j=0
m=0
n=0

#print(X0_new_arr.shape)
#print(X1_new_arr.shape)

tr_data_array0 = []
tr_data_array1 = []

#Crearting transaction dataframe for digit 0

for i in range(X0_new_arr.shape[0]):
    temp = []
    for j in range(X0_new_arr.shape[1]):        
        if(X0_new_arr[i][j] == 1):
            temp.append(j+1)
    tr_data_array0.append(temp)
i=0
j=0
for i in range(X1_new_arr.shape[0]):
    temp = []
    for j in range(X1_new_arr.shape[1]):        
        if(X1_new_arr[i][j] == 1):
            temp.append(j+1)
    tr_data_array1.append(temp)

#print(tr_data_array0)
te0 = TransactionEncoder()
te_ary0 = te0.fit(tr_data_array0).transform(tr_data_array0)
df0 = pd.DataFrame(te_ary0, columns=te0.columns_)

te1 = TransactionEncoder()
te_ary1 = te1.fit(tr_data_array1).transform(tr_data_array1)
df1 = pd.DataFrame(te_ary1, columns=te1.columns_)

minsup_arr = [0.1,0.3,0.5,0.7]
length = len(minsup_arr)

for z in range(length) :
    freq_items_0 = fpgrowth(df0, min_support=minsup_arr[z], use_colnames=True)
    freq_items_1 = fpgrowth(df1, min_support=minsup_arr[z], use_colnames=True)

    if(minsup_arr[z] == 0.1):
        freq_items_0.to_csv("D:\ML_Implementations\FP_Growth\Class_0_freqitems_minsup_0_1.csv")
        freq_items_1.to_csv("D:\ML_Implementations\FP_Growth\Class_1_freqitems_minsup_0_1.csv")

    if(minsup_arr[z] == 0.3):
        freq_items_0.to_csv("D:\ML_Implementations\FP_Growth\Class_0_freqitems_minsup_0_3.csv")
        freq_items_1.to_csv("D:\ML_Implementations\FP_Growth\Class_1_freqitems_minsup_0_3.csv")

    if(minsup_arr[z] == 0.5):
        freq_items_0.to_csv("D:\ML_Implementations\FP_Growth\Class_0_freqitems_minsup_0_5.csv")
        freq_items_1.to_csv("D:\ML_Implementations\FP_Growth\Class_1_freqitems_minsup_0_5.csv")   

    if(minsup_arr[z] == 0.7):
        freq_items_0.to_csv("D:\ML_Implementations\FP_Growth\Class_0_freqitems_minsup_0_7.csv")
        freq_items_1.to_csv("D:\ML_Implementations\FP_Growth\Class_1_freqitems_minsup_0_7.csv")

    print("Freq Items with Min Sup =", minsup_arr[z])
    print(freq_items_0.head(10))
    print(freq_items_1.head(10))
    print(" ")
