# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:40:31 2019

@author: kuuku
"""

#import all your libries, numpy, matplotlib etc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
style.use('ggplot')

#Importing the dataset

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y= dataset.iloc[:, 1].values

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, random_state = 0)

#feature scaling
"""from sklearn.preprocessing

#Fitting Simple linear Regression to  the Trainign set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#visualising the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train,regressor.predict(X_train), color = 'blue' )
plt.title('Salary vs Experience (Training set)')
plt.xlabel( 'Years of Experience')
plt.ylabel('Salary')
plot.show()

#visualising the test set results
plt.sc