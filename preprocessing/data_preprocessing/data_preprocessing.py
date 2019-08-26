# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:49:30 2019

@author: kuuku
"""
#Import various libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Data.csv') # to import your excel file
X = dataset.iloc[:, :-1].values # X is the feature
y = dataset.iloc[:, 3].values #y is the target

#Splitting the data into testing and training. you mostly test at 20% which is 0.2, then the random is 0
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0)


#Dealing with missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Encoding categorical data- data with names we have to import two thing
# form sckit learning. the categorical dta is converted to a dummy variable so that the ML will not assume
# one variable is important or better than another

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features =[0])
X = onehotencoder.fit_transform(X).toarray()

# Encoding the dependent variable. Here we do not need to convert to dummy because it is in order
labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

