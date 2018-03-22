# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 06:03:44 2018

@author: ankush
"""
import numpy as np

def load(fname,delimiter=','):
    
    data = np.loadtxt(fname,delimiter=delimiter, skiprows=1)
    return data[:,:-1],data[:,-1].reshape(data.shape[0],1)


def normalize(features):
    
    u = np.mean(features,0)
    std = np.std(features,0)
    
    norm = np.array([u,std])
    
    np.save('supervised/features/norm',norm)
    
    return (features - u)/std