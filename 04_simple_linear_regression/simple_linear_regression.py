#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:20:15 2017

@author: mvezer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# splitting the dataset to train and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 10, random_state = 0)

# fitting simple lkinear regression model to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
p = regressor.predict(X_test)
