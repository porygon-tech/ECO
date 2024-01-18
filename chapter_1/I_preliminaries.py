#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:34:06 2023

@author: ubuntu
"""

#%%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#%% developer tools

#plt.switch_backend('Qt5Agg')
import contextlib
import matplotlib

def switch_backend(gui):
    with contextlib.suppress(ValueError):
        matplotlib.use(gui, force=True)
    globals()['plt'] = matplotlib.pyplot


switch_backend('qt5agg') # or 'qtagg'
switch_backend('module://matplotlib_inline.backend_inline')
#%%

# Define the system of differential equations
def model(y, t, sigma, beta, rho):
    # y is a vector of the state variables
    # t is the time variable
    dydt = np.zeros(3)
    dydt[0] = sigma*(y[1] - y[0])
    dydt[1] = y[0]*(rho - y[2]) - y[1]
    dydt[2] = y[0]*y[1] - beta*y[2]
    return dydt

# Define the initial conditions
y0 = [0.1, 0.1, 0.1]

# Define the time points at which to evaluate the solution
t = np.linspace(0, 100, 10001)

# Solve the system of differential equations
sigma, beta, rho = 10, 8/3, 28

y = odeint(model, y0, t, args=(sigma, beta, rho))
#%%
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
ax.plot(*y.T,lw=0.2)
plt.show()
#%%
from scipy.optimize import root_scalar
#%%
y0=[1,1,1]
sol = root_scalar(model, y0, args=(sigma, beta, rho), xtol=1e-08, maxiter=10000, method='del2')

#%%









#%%

def model(y, t, phi, theta, s, A):
    dydt = np.zeros(len(y))
    dydt[0] = phi[0]*((theta[0] - y[0])*s + (y[1] - y[0])*(1-s)*A[0,1])
    dydt[1] = phi[1]*((theta[1] - y[1])*s + (y[0] - y[1])*(1-s)*A[1,0])
    return dydt
    
#%%
# Define the initial conditions
y0 = [1,3]

# Define the time points at which to evaluate the solution
t = np.linspace(0, 100, 10001)

# Set parameters
theta = [2,2]
s=0.73
phi=[1,5]
A = np.ones((2,2))
A[1,0] = -4
A[0,1] = 4

# Solve the system of differential equations
y = odeint(model, y0, t, args=(phi, theta, s, A))

#%
# fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
# ax.plot(y,lw=1)
# plt.show()
# #%%
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
ax.plot(*y.T,lw=1)
plt.show()
#%%
'''
The problem is for trait repulsion
'''








#%%

def model(y, t, phi, theta, s, A):
    dydt = np.zeros(len(y))
    dydt[0] = phi[0]*((theta[0] - y[0])*s + (y[1] - y[0])*(1-s)*A[0,1])
    dydt[1] = phi[1]*((theta[1] - y[1])*s + (y[0] - y[1])**A[1,0])*(1-s)
    return dydt
    
#%%
# Define the initial conditions
y0 = [1,3]

# Define the time points at which to evaluate the solution
t = np.linspace(0, 100, 10001)

# Set parameters
theta = [2,2]
s=0.5
phi=[1,5]
A = np.ones((2,2))
A[1,0] = 0.1
A[0,1] = 1

# Solve the system of differential equations
y = odeint(model, y0, t, args=(phi, theta, s, A))


fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
ax.plot(y,lw=1)
plt.show()
#%%

# fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
# ax.plot(*y.T,lw=1)
# plt.show()












