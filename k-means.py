# -*- coding: utf-8 -*-
"""
Created on Fri Apr 8 16:21:41 2016

@author: jgapper
"""

# import packages
import numpy as np
import pandas as pd
import random

# import data
dat = np.genfromtxt('C:\Users\jgapper\Desktop\CSDS\CS530\Assignment4\iris.csv', delimiter=',')
dat = dat[1:,0:4]

#pd.read_csv('C:\Users\jgapper\Desktop\CSDS\CS530\Assignment4\iris.csv', header=True)

class cluster_f(object):
    def _init_(classes_ = 3):
        self.classes_ = classes_
        self.data = data

    def calc_var(self, data, x):
        # calculate variance
        self.sqdiff = (x[:, np.newaxis, :]-data)**2
        return np.sum(np.min(np.sum(self.sqdiff, axis=2), axis=0))
        
    def k_means(self, data, k=3, n=50):
        # initiate random centers and square differences with the number of iterations
        self.class_center = data[np.random.choice(range(data.shape[0]),k,replace=False),:]
        self.sq_differences = []
        
        for iteration in range(n):
            # iterate over the square differences, calc closest center for each obs
            self.sqdist = np.sum((self.class_center[:, np.newaxis, :]-data)**2, axis=2)
            self.closest_ctr = np.argmin(self.sqdist, axis=0)
            
            self.sq_differences.append(self.calc_var(data, self.class_center))

            # move class center
            for i in range(k):
                self.class_center[i, :] = data[self.closest_ctr==i,:].mean(axis=0)
                
        self.sq_differences.append(self.calc_var(data, self.class_center))
        return self.class_center

cf = cluster_f()
cf.k_means(dat)