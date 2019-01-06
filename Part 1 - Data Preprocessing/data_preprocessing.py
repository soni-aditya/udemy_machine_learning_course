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
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#Taking care of missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding categorical data into numbers/values
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('oh_enc',OneHotEncoder(sparse=False),[0])],remainder='passthrough')
X = ct.fit_transform(X).astype(float)

labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)

#Splitting into training and test data
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
Xtrain = sc_X.fit_transform(Xtrain)
Xtest = sc_X.transform(Xtest)


