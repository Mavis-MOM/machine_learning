# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:59:03 2019

@author: kuuku
"""

#Importing your data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use('ggplot')

# Importing the dataset and separating your features from the targets. 
dataset = pd.read_csv('Position_Salaries.csv') #this brings your dataset for you to use. the dataset here is the Position_Salary
X = dataset.iloc[:, 1:2].values #this gives you all of the first column and all the rows in them
y = dataset.iloc[:, 2].values

#importing your decision tree, fitting decision tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

#predicting a new result
y_pred = regressor.predict([[6.5]])

#visualising the decision tree regression results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff(Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


