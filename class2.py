import pandas as pd
data={'exp':[5,6,7,8,9,10,11,12,13,14],'salary':[27000,77000,29000,30000,41000,37000,33000,34000,39000,96000]}
df=pd.DataFrame(data)
print(df)
df.to_csv('user.csv')
df=pd.read_csv('user.csv')
print(df)
from sklearn.model_selection import train_test_split
x=df[['exp']]
y=df[['salary']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print("x_train",x_train)
print("x_test",x_test)
print("y_train",y_train)
print("y_test",y_test)
print(df.describe())
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("prediction",y_pred)
from sklearn.metrics import mean_squared_error,r2_score
print("mse",mean_squared_error(y_test,y_pred))
print("r2_score",r2_score(y_test,y_pred))
import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred,color="blue",s=100)
plt.plot(y_test,y_pred,color="blue")
plt.title("Actual vs Predicton")
plt.xlabel("Actual")
plt.ylabel("Prediction")
plt.grid("true")
plt.show()

import pandas as pd
week={'hours':[1,2,3,4,5], 'units':[10,20,30,40,50]}
df=pd.DataFrame(week)
print(df)
from sklearn.model_selection import train_test_split
x=df[['hours']]
y=df[['units']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print("x_train",x_train)
print("x_test",x_test)
print("y_train",y_train)
print("y_test",y_test)
print(df.describe())
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("prediction",y_pred)
from sklearn.metrics import mean_squared_error,r2_score
print("mse",mean_squared_error(y_test,y_pred))
print("r2_score",r2_score(y_test,y_pred))
import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred,color="blue",s=100)
plt.plot(y_test,y_pred,color="blue")
plt.title("Actual vs Predicton")
plt.xlabel("Actual")
plt.ylabel("Prediction")
plt.grid("true")
plt.show()
