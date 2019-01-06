#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 00:43:55 2018

@author: adityas
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
#In python, indexes start with 0
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Splitting into training and test data
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Fitting Simple Linear Regressor to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Xtrain,Ytrain)

#Predicting the test set results
y_pred = regressor.predict(Xtest)

#Visualizing the results
plt.scatter(Xtrain, Ytrain, color='red')
plt.plot(Xtrain, regressor.predict(Xtrain), color= 'blue')
plt.title('Salary Vs. Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()