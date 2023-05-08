#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 22:30:24 2023

@author: roman
"""


from os import chdir, listdir
from pathlib import Path
import pickle5
import bz2
chdir('/home/roman/LAB/ECO') #this line is for Spyder IDE only
root = Path(".")
obj_path = root / 'data/obj'
img_path = root / 'gallery/timeseries'
#%% imports 
import sys
sys.path.insert(0, "./lib")
import evo
import matriX as mx
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


#%% 
nloci = 100
nstates = nloci+1
ps = (500,500+nloci)
mu=0.001
filename='oc_tensor_' + str(nloci) + '.obj'
with bz2.BZ2File(obj_path / filename, 'rb') as f:
	h = pickle5.load(f)

mut= evo.generate_mut_matrix(nstates,mu=mu)
#%% 
pop = evo.population(200,nloci, skew= 0.25,phenoSpace=ps)

def f(x):
    return (x-500)/26
pop.set_fitnessLandscape(f)
pop.showfitness()

states = np.linspace(ps[0],ps[1], nstates)
l = np.zeros((nstates,1))
for i in range(nstates):
    l[i] = f(states[i])
    #


#%% 
ntimesteps=50
K=2000 
c=pop.reproduce(ntimesteps,verbose=True,fixedSize=False,K=K,mu=mu)
# c=pop.reproduce(ntimesteps,verbose=True,fixedSize=True,K=K,mu=mu)

mx.showdata(c.history, colorbar=True, color='binary')
c.show()

c.popsize_history

mx.showlist(c.popsize_history)

#%% 

N=np.zeros(ntimesteps+1)
v = np.zeros((ntimesteps+1, nstates,1))
v[0] = pop.history[0][:,np.newaxis]
N[0]=200
for t in range(1,ntimesteps+1):
#for t in range(1,10):
    w = v[t-1]*l
    
    r = w.sum()
    N[t] = (1-1/(N[t-1] * r/K+1))*K
    # N[t] = N[t-1] * r * (1 - N[t-1]/K)
    
    # r = np.log(w.sum())
    # N[t] = N[t-1] * np.e**r * (1 - N[t-1]/K)
    
    #N[t] = (N[t-1]*r - K) / (1 - np.exp(1*(N[t-1]*r - K))) + N[t-1]*r
    #maynard smith 1968, May 1972?
    v[t] = ((w.T @ h @ w) / w.sum()**2)[:,0]
    v[t] = mut @ v[t]
    print(t)

mx.showdata(v, colorbar=True, color='binary')
mx.showlist(N[:20])
mx.showlist(N)


c.fitnessValues.sum()/c.m
r
w.sum()
c.fitnessValues
v[t-1]
c.fitnessValues

#%%
ntimesteps=100
r=1.5
K=1000 
N=np.zeros(ntimesteps+1)
N[0]=200
for t in range(1,ntimesteps+1):
    N[t] = (1-1/(N[t-1] * r/K+1))*K
    #N[t] = K * np.log(r*N[t-1] + 1) / np.log(r*K + 1) 
    #N[t] = r * K / (r + K / N[t-1] - 1) #logistic?

mx.showlist(N)