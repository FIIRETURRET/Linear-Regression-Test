# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:40:57 2019

@author: joutras
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import time

# Access the data and save it into memory
dataset = pd.read_csv('C:\\Users\\joutras\\Documents\\Python Scripts\\Iris\\Weather.csv')
print(dataset.shape)
print(dataset.describe())

# Plot the max and min tempuratures for eaach day
dataset.plot(x='MinTemp', y='MaxTemp', style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

# Plot the average max temperature
plt.figure(figsize=(10,5))
plt.tight_layout()
seabornInstance.distplot(dataset['MaxTemp'])
plt.show()

'''
Attributes are the independent variables while labels are dependent variables whose values are to be predicted. 
In our dataset, we only have two columns. We want to predict the MaxTemp depending upon the MinTemp recorded. 
Therefore our attribute set will consist of the “MinTemp” column which is stored in the X variable, and the label 
will be the “MaxTemp” column which is stored in y variable.
'''
X = dataset['MinTemp'].values.reshape(-1,1)
Y = dataset['MaxTemp'].values.reshape(-1,1)
# Create a 80% 20% split of the data to be used for training and testing.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Create a regression model
regressor = LinearRegression()
# Train the model
regressor.fit(X_train, Y_train)

'''
the linear regression model basically finds the best value for the intercept and slope, 
which results in a line that best fits the data. 
To see the value of the intercept and slop calculated by the linear regression algorithm for our dataset, 
execute the following code.
'''
# To retrieve the intercept
print("Intercept: ", regressor.intercept_)
# For retrieving the slope
print("Slope: ", regressor.coef_)
"""
The result should be approximately 10.66185201 and 0.92033997 respectively.
This means that for every one unit of change in Min temperature, the change in the Max temperature is about 0.92%.
"""

# Test the trained model
predicted = regressor.predict(X_test)

# Compare the actual ouput values with the predicted values
df = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': predicted.flatten()})
print(df)

# Visualize the comparison as a bar graph
df1 = df.head(25)
df1.plot(kind='bar', figsize=(10,5))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# Plot the straight line with the test data
plt.scatter(X_test, Y_test, color='gray')
plt.plot(X_test, predicted, color='red', linewidth=2)
plt.show()

# The mean absolute error is the difference between the actual value and the predicted value
print('Mean Absolute Error: ', metrics.mean_absolute_error(Y_test, predicted))
# The mean squared error tells you how close the regression line is to a set of points.
# the distance of the points are squared to remove negative values and add more weight to larger differences. 
print('Mean Squared Error: ', metrics.mean_squared_error(Y_test, predicted))
# Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). 
# Residuals are a measure of how far from the regression line data points are; 
# RMSE is a measure of how spread out these residuals are. In other words, 
# it tells you how concentrated the data is around the line of best fit.
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(Y_test, predicted)))



