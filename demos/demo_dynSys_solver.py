#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:58:05 2023

@author: roman
"""
#%%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#%%

def vecF (model, space=((-3,3),(-3,3)), definition=(20,20)):    
    # Define the x and y ranges
    x = np.linspace(space[0][0], space[0][1], definition[0])
    y = np.linspace(space[1][0], space[1][1], definition[1])
    X, Y = np.meshgrid(x, y)
    # Compute the vector field
    u, v = np.zeros(X.shape), np.zeros(Y.shape)
    NI, NJ = X.shape
    for i in range(NI):
        for j in range(NJ):
            x = X[i, j]
            y = Y[i, j]
            yprime = model([x, y], 0)
            u[i,j] = yprime[0]
            v[i,j] = yprime[1]
    return u,v,X,Y

def phaseportrait(model, space=((-3,3),(-3,3)), definition=(10,10), density=1.5):
    u,v,X,Y = vecF(model,space,definition)
    # Plot the phase portrait
    fig, ax = plt.subplots(figsize=(7,6))
    stream = ax.streamplot(X, Y, u, v, color=np.sqrt(u**2+v**2), density=density, linewidth=0.8, arrowsize=0.8, cmap=plt.cm.autumn)
    ax.set_xlabel('y1')
    ax.set_ylabel('y2')
    ax.set_title('Phase portrait')
    # Add a colorbar
    cbar = fig.colorbar(stream.lines)
    cbar.set_label('Velocity')
    plt.show()

#%%
# Define the system of differential equations
def model(y, t):
    # y is a vector of the state variables
    # t is the time variable
    dydt = np.zeros(2)
    dydt[0] = -0.1*y[0] + 0.2*y[1]
    dydt[1] = 0.1*y[0] - 0.2*y[1]
    return dydt
#%%
# Define the initial conditions
y0 = [1, 0]

# Define the time points at which to evaluate the solution
t = np.linspace(0, 100, 1001)
#%%
# Solve the system of differential equations
y = odeint(model, y0, t)

#%%
# Plot the solution
plt.plot(t, y[:, 0], label='y1')
plt.plot(t, y[:, 1], label='y2')
plt.xlabel('Time')
plt.ylabel('State variables')
plt.legend()
plt.show()
#%%




#%%
# Define the system of differential equations
def model(y, t):
    # y is a vector of the state variables
    # t is the time variable
    dydt = np.zeros(2)
    dydt[0] = -y[0] + y[0]**3 - 3*y[0]*y[1]**2
    dydt[1] = -y[1] + 3*y[0]**2*y[1] - y[1]**3
    return dydt

# Define the initial conditions
y0 = [0.1, 0.1]

# Define the time points at which to evaluate the solution
t = np.linspace(0, 100, 1001)

# Solve the system of differential equations
y = odeint(model, y0, t)

phaseportrait(model)




#%%





#%%

u,v,_,_=vecF(model, definition=(100,100))
plt.imshow(u,cmap='seismic')
plt.colorbar()

thr=0.2
plt.imshow(np.logical_and(np.abs(u)<thr, np.abs(v)<thr))




#%%

from scipy.fftpack import fft, fftfreq

# Define the Lorenz system
def lorenz(state, t, sigma, beta, rho):
    x, y, z = state
    dxdt = sigma*(y - x)
    dydt = x*(rho - z) - y
    dzdt = x*y - beta*z
    return [dxdt, dydt, dzdt]

# Set the parameters
sigma, beta, rho = 10, 8/3, 28

# Set the initial conditions
state0 = [1.0, 1.0, 1.0]

# Set the time steps
t = np.linspace(0, 100, 10000)

# Integrate the Lorenz system
state = odeint(lorenz, state0, t, args=(sigma, beta, rho))

# Compute the power spectral density
psd = np.abs(fft(state[:, 0]))**2
freqs = fftfreq(state[:, 0].size, t[1]-t[0])

# Plot the time series and its PSD
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))
ax[0].plot(t, state[:, 0], 'b', label='x')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('State variable')
ax[1].plot(freqs, psd, 'r')
ax[1].set_xlim(-10,10)
ax[1].set_xlabel('Frequency')
ax[1].set_ylabel('Power spectral density')
plt.show()