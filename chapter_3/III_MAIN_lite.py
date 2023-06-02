#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:41:57 2023

@author: roman
"""
from os import chdir, listdir, environ
from pathlib import Path
import pickle5
import bz2
chdir(environ['HOME'] + '/LAB/ECO') #this line is for Spyder IDE only
root = Path(".")
obj_path = root / 'data/obj'
img_path = root / 'gallery/timeseries'
dataPath = root / 'data/dataBase'


#%% imports 
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import networkx as nx
import pandas as pd
#%% OWN LIBS
sys.path.insert(0, "./lib")
import evo
import matriX as mx
#%%

#%% DATA LOAD
#%% network load
df = pd.read_csv(dataPath / 'M_PL_058.csv', index_col=0)
b=df.to_numpy()
if not np.equal(b.shape[0],b.shape[1]):
    b=mx.to_square(b)
elif not mx.is_symmetric(b):
    b=mx.to_square(b)

mx.showdata(b)
A = (b>0)+0
N=A.shape[0]
#----------------------
#%% tensor object load
nloci=100
filename='oc_tensor_' + str(nloci) + '.obj'
with bz2.BZ2File(obj_path / filename, 'rb') as f:
	h = pickle5.load(f)
#%% mutual effects matrix generation
N=23
A = mx.symmetric_connected_adjacency(N,0.2); mx.showdata(A)
#A_e = A/(N**2)*np.sqrt(A.sum())
A_e = A* (np.random.rand(N,N)*0.1-0.01);mx.showdata(A_e,symmetry=True,colorbar=True)

#%%
ps=(23,37)
dev=np.random.rand(N) # p for the binomial distributions
theta=dev*np.diff(ps)+ps[0] # we set it to start at their environmental optima
#%%------
v0=evo.initialize_bin_explicit(N,nloci,dev); #mx.showlist(v0.T)
v0=evo.initialize_bin_explicit(N,nloci,np.random.rand(N)); #mx.showlist(v0.T)

v,D = evo.simulate_explicit(
    v0=v0,
    ntimesteps=50,
    h=h,
    mutual_effs=A_e,
    theta=theta,
    ps=ps,
    alpha=0.01,
    xi_S=0.1,
    complete_output=True
)

mx.showlist(evo.dist_averages(v,ps))
mx.showlist(D); print('{0} species went extinct out of {1}.'.format(((D[-1]<2).sum()),N))



#C0AF852B6365458F1396DE679CC974FF
#%% DELETE AFTERW
I = np.newaxis
nstates=nloci+1
states = np.linspace(ps[0],ps[1], nstates)
statesdiff=np.outer(np.ones(nstates),states)-np.outer(states,np.ones(nstates))
assortMat = evo.interactors.pM(statesdiff,alpha=0.2); mx.showdata(assortMat)
#assortTen = np.repeat(assortMat[...,I],N,axis=-1)
assortTen = np.repeat(assortMat[I,...],N,axis=0)


vs = np.c_[v[-1,15]]
test = (vs.T @ assortMat).T * vs
comp = (assortMat @ vs) * vs
mx.showdata(np.repeat(comp,23,1))

vs = v[-1,...,I]
vs = v[-1,...].T[...,I]

vs = v[-1,...]
test = np.tensordot(vs, assortTen,axes=1)
test = vs @ assortTen 
test = assortTen.T @ vs

test.shape
mx.showdata(test)
mx.showdata(test[15,:,:])
mx.showdata(test[:,15,:])
mx.showdata(test[:,:,15])
mx.showdata(test[-1])

test2=vs*test[0]
test2[15]
comp
mx.showdata(np.repeat(test2[15,I],23,0))




vs = v[-1,...]
assortTen = np.repeat(assortMat[I,...],N,axis=0)
test = vs @ assortTen 
test2=vs*test[0]