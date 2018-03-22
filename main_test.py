# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 05:39:59 2018

@author: ankush
"""
import random

from supervised.linear_regression import *
from supervised import data

x,y = data.load('supervised/data/ex1data2.txt')

X = data.normalize(x)

fit(X,y)

y_predicted = predict(X[random.randint(0,X.shape[0])])

print(y_predicted)