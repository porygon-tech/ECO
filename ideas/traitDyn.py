#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:52:34 2022

@author: roman
"""

import numpy as np 
import matplotlib.pyplot as plt
#from pylab import meshgrid, arange
#%%

alpha = 0.2 # prey growth rate
beta  = 0.2 # predation rate
gamma = 0.1 # predator death rate
delta = 0.2 # predator growth rate

x, y = np.meshgrid(np.arange(0, 3, 0.1), np.arange(0, 3, 0.1))
dx =   alpha * x - beta  * x * y
dy = - gamma * y + delta * x * y

fig = plt.figure(figsize = (12,9))
plt.streamplot(x, y, dx, dy, density=2.5, linewidth=0.8, arrowsize=0.8)
plt.show()

#%%

def fdx(x,y):
    global alpha, beta
    return alpha * x - beta  * x * y

def fdy(x,y):
    global gamma, delta
    return - gamma * y + delta * x * y

time_delta=0.01
duration=5
ntimesteps = int(duration/time_delta)
series = np.zeros((ntimesteps, 2))

series[0,0]=x
series[0,1]=y
for i in range(1,ntimesteps):
    series[i,0]=
    series[i,1]=
    


#%%

fig = plt.figure(figsize = (12,9))
plt.streamplot(x, y, dx, dy, density=2.5, linewidth=0.8, arrowsize=0.8)
plt.show()




-3 * (x - a) * (x - (a + b) / 2) * (x - b)