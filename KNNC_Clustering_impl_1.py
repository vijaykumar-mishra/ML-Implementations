import random
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

#Generate a data set of 120 random numbers
random_dataset = np.random.uniform(low=0, high=1, size=120)

#Convert the data set to array
dataset_arr = np.array(random_dataset)

dataset_class_arr = []
leader_arr = []
class1_count = [0]*120
class2_count = [0]*120
threshold = 0.25
leader_index = 0
min_dist = 1
min_dist_leader_index = 0
cluster_count = 0

for i in range(0, len(dataset_arr)):
    if(dataset_arr[i] <= 0.5) :
        dataset_class_arr.insert(i,"Class1")
    else:
        dataset_class_arr.insert(i,"Class2")

#create a data frame and classify the data in class type 1 or 2
df = pd.DataFrame(dataset_arr, columns = ['Data'])
df['Class'] = np.where(df['Data']<=0.5, 'Class1', 'Class2')

#print(dataset_arr)
#print(dataset_class_arr)


for i in range(0, 120):
    if(i==0):
        leader_arr.insert(leader_index, dataset_arr[i])
        if(dataset_class_arr[i] is "Class1"):
            class1_count[leader_index] = class1_count[leader_index] + 1
        else :
            class2_count[leader_index] = class2_count[leader_index] + 1
        leader_index = leader_index + 1
        cluster_count = cluster_count + 1
        
    else:
        min_dist = 1
        for j in range(0,len(leader_arr)):
            dist = abs(dataset_arr[i] - leader_arr[j])
            if(dist < min_dist) :
                min_dist = dist
                min_dist_leader_index = j

        if(min_dist<=threshold):
            if(dataset_class_arr[i] is "Class1"):
                class1_count[min_dist_leader_index] = class1_count[min_dist_leader_index] + 1
            else :
                class2_count[min_dist_leader_index] = class2_count[min_dist_leader_index] + 1

        else :
            if(dataset_class_arr[i] is "Class1"):
                class1_count[leader_index] = class1_count[leader_index] + 1
            else :
                class2_count[leader_index] = class2_count[leader_index] + 1
            leader_index = leader_index + 1
            cluster_count = cluster_count + 1
            leader_arr.insert(leader_index, dataset_arr[i])
purity = 0            
for i in range(0, cluster_count) :
    purity += max(class1_count[i], class2_count[i])

print("Threshold =", threshold)
print("Number of Clusters =", cluster_count)
print("Leaders of Clusters =",leader_arr)
print("Purity =",purity)
print("  ")
