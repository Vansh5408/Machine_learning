#   DAY1


import numpy as np
arr = np.array([10,20,30,40])
print("Array",arr)
print("Maximum",np.max(arr))
print("Summation",np.sum(arr))
a=np.array([10,20,30,40])
b=np.array([50,60,70,80])
sum = a+b
print("Summation",sum)
arr=np.array([10,20,30,40,50])
print(arr[1])
print(arr[:2])
print(arr[1:3])
# output 20 30
print(arr[-1])
# output 50
data=[10,20,30,40]
print("Multiplication",data*2)
# output will be double digit
data=np.array([10,20,30,40])
print("Multiplication",data*2)
# Multiplication [20 40 60 80]
arr_1d=np.array([1,2,3,4,5,6])
print("Matrix 1d",arr_1d)
# Matrix 1d [1 2 3 4 5 6]
arr_2d=np.array([1,2,3,4,5,6]).reshape(2,3)
print("Matrix 2d",arr_2d)
# Matrix 2d [[1 2 3]
# [4 5 6]]
print("Matrix 2d",arr_2d.T)
arr= np.arange(1,11)
print(arr)
zero=np.zeros((3,3))
print(zero)
one=np.ones((3,3))
print(one)
ran=np.random.rand(5)
print(ran)
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[10,20,30,40,50]
plt.scatter(x,y,color="red")
plt.title("study hours vs score")
plt.xlabel("study hours")
plt.ylabel("score")
plt.grid(True)
plt.show()
import pandas as pd
data = {'name':['ashish','raj','ram'],
        'Age':[27,28,29],
        'Salary':[50000,60000,70000]}
df=pd.DataFrame(data)
print(df)
print("Summary",df.describe) 



