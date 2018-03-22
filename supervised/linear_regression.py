# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 05:44:47 2018

@author: ankush
"""

import random
import numpy as np
import matplotlib.pyplot as plt

import data

#calculate cost
def cost(X,y,theta):
    
    m = y.shape[0]
    h = np.matmul(X,theta)
    err = h-y
    J = 0.5*(np.matmul(err.T,err))/m
    
    return J

#batch gradient descent 
def gradient_descent(X,y,theta,alpha=0.03,iterations=500):

    m = y.shape[0]
    cost_data = np.zeros(shape=(iterations,1))
    for i in range(iterations):
        h = np.matmul(X,theta)
        descent = (alpha/m)*(np.matmul((h-y).T,X).T)
        theta -= descent
        cost_data[i] = cost(X,y,theta)
    
    plt.plot([i for i in range(iterations)],cost_data)
    np.save('supervised/parameters/theta',theta)

#train model
def fit(X,y):
    
    m = y.shape[0]
    
    X = np.hstack((np.ones(shape=(m,1)),X))
    n = X.shape[1]
    initial_theta = np.random.rand(n,1)
    gradient_descent(X,y,initial_theta)
    

def test(X,y):
    i = random.randint(0,X.shape[0])
    x_test = X[i].reshape((1,X.shape[1]))
    u,std = np.load('supervised/features/norm.npy')
    x_norm = (x_test-u)/std;
    x = np.hstack((np.ones((x_norm.shape[0],1)),x_norm))
    theta = np.load('supervised/parameters/theta.npy')
    y_predict = np.matmul(x,theta)
    print('predicted value of y :')
    print(y_predict[0])
    print('actual value of y :')
    print(y[i])
    

#predict from trained model
def predict(x):
    x = x.reshape((1,x.shape[0]))
    u,std = np.load('supervised/features/norm.npy')
    x = (x-u)/std;
    x = np.hstack((np.ones(shape=(x.shape[0],1)),x))
    theta = np.load('supervised/parameters/theta.npy')
    return np.matmul(x,theta)
