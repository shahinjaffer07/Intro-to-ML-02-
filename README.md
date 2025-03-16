# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students
2. Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored
3. Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis
4. Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b
5. for each data point calculate the difference between the actual and predicted marks
6. Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error
7. Once the model parameters are optimized, use the final equation to predict marks for any new input data

## Program:
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: SHAHIN J

RegisterNumber: 212223040190
```
import numpy as np
import pandas as pd
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#read csv file
df=pd.read_csv('/content/student_scores (1).csv')

#displaying the content in datafile
print(df.head())
print(df.tail())

# Segregating data to variables
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)

# shape of x and y
print(x.shape)
print(y.shape)

#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

#displaying predicted values
y_pred=reg.predict(x_test)
x_pred=reg.predict(x_train)
print(y_pred)
print(x_pred)

#training and testing shape
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#find mae,mse,rmse
mse = mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae = mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse = np.sqrt(mse)
print('RMSE = ',rmse)

#graph plot for training data
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,y_pred,color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#array values
y_pred1=reg.predict(np.array([[13]]))
y_pred1

```

## Output:
### Head Values
![Screenshot 2025-03-15 194620](https://github.com/user-attachments/assets/262e5dc9-5436-4ceb-95fb-f9a8a200ae1d)

### Tail Values
![Screenshot 2025-03-15 194710](https://github.com/user-attachments/assets/e8606296-da4d-4f16-a061-4fdd63f892f0)

### X Values
![Screenshot 2025-03-15 194809](https://github.com/user-attachments/assets/47bb02a3-32a7-45cf-bce4-86799c5d38a7)

### Y Values
![Screenshot 2025-03-15 194848](https://github.com/user-attachments/assets/1923ae11-862a-4d04-8dd0-6a14cd2127ce)

### Shape Values
![Screenshot 2025-03-15 194916](https://github.com/user-attachments/assets/675ee7f7-f1a9-466e-9326-08e65632f23e)

### Predicted Values
![Screenshot 2025-03-15 195042](https://github.com/user-attachments/assets/8f4638ff-6110-459a-9b3e-8f877ed53bd4)

### Training and Testing shape
![Screenshot 2025-03-15 195109](https://github.com/user-attachments/assets/57476e30-ec36-4015-b32a-1bca0639cd4b)

### MSE, MAE and RMSE
![Screenshot 2025-03-15 195155](https://github.com/user-attachments/assets/5c1b6be0-1fe2-48e9-ad50-433ead8748fb)

### Graph Plot
![Screenshot 2025-03-15 195315](https://github.com/user-attachments/assets/b11d36d8-27a5-4f4e-8040-b67a59543a4c)

### Array Values
![Screenshot 2025-03-15 195345](https://github.com/user-attachments/assets/c030524c-e9d3-439f-a5ad-8b1a66bb724b)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
