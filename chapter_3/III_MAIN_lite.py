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
#import time
import sys
import numpy as np 
#import matplotlib.pyplot as plt
#from copy import deepcopy
#import networkx as nx
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
#%% HYPERPARAMETERS
N=23 # number of species. Has an important relation with the mutualRange and c parameters
nloci=100 # number of loci
ps=(23,37) # phenotypic space. Has an important relation with the alpha parameter
ntimesteps = 200 # number of generations simulated
K=1000 # carrying capacity  

# tensor object load
filename='oc_tensor_' + str(nloci) + '.obj'
with bz2.BZ2File(obj_path / filename, 'rb') as f:
	h = pickle5.load(f)

#%% MODEL PARAMETERS
c=0.4 # expected connectance of the allowed links matrix
mutualRange = (-0.01, 0.02) # range of values for the uniform distribution of ecological effects
a=-0.01 # assortative mating coefficients, real value (single value or array of values)
alpha= 0.1 # strength of the trait matching mechanism. Positive real value. Lower values = greater promiscuity
xi_S=0.1 # level of environmental selection (from 0 to 1).
D0=50 # initial population sizes (single value or array of values)


# a=np.linspace(0, 1, N) # assortative mating coefficients, real value (array)

#%% mutual effects matrix generation

A = mx.symmetric_connected_adjacency(N,c); mx.showdata(A)
A_e = A* (np.random.rand(N,N)*np.diff(mutualRange)+mutualRange[0]);mx.showdata(A_e,symmetry=True,colorbar=True)

# mx.showdata((A_e>0) & (A_e.T>0)) # mutualisms
# mx.showdata((A_e<0) & (A_e.T<0)) # antagonisms (competitors)
# mx.showdata((A_e>0) & (A_e.T<0)) # antagonisms (predation)
print("connectance of " + str(A.sum()/N**2) + ", expected " + str(c))
print("generated with: \n{0} mutualisms, \n{1} antagonisms (competitors), and \n{2} antagonisms (predation)".format(int(((A_e>0) & (A_e.T>0)).sum()/2),
                                                int(((A_e<0) & (A_e.T<0)).sum()/2),
                                                ((A_e>0) & (A_e.T<0)).sum()))

#%% environmental optima generation (theta)
dev=np.random.rand(N) 
theta=dev*np.diff(ps)+ps[0] 

#%% initialization of phenotype makeups
v0=evo.initialize_bin_explicit(N,nloci,dev); # set to start at their environmental optima
v0=evo.initialize_bin_explicit(N,nloci,np.random.rand(N)); # set to start at random location
#%% 
xi_d=1-xi_S
m=np.clip(np.random.normal(xi_d,0.01,(N,1)),0,1) # vector of levels of selection imposed by other species (from 0 to 1)

#%% 
import evo
v,D,l = evo.simulate_explicit(
    find_fixedpoints=False,
    v0=v0,
    ntimesteps=130,
    h=h,
    mutual_effs=A_e,
    theta=theta,
    ps=ps,
    alpha=alpha,
    #xi_S=xi_S,
    m=m,
    D0=D0,
    a=0.1,
    K=K,
    complete_output=True
)

#%% 
mx.showlist(evo.dist_averages(v,ps))
mx.showlist(D); #print('{0} species went extinct out of {1}.'.format(((D[-1]<2).sum()),N))

fits = (v*l).sum(2)
mx.showlist(fits)



#C0AF852B6365458F1396DE679CC974FF
#%%
a=0.001
nstates=nloci+1
states = np.linspace(ps[0],ps[1], nstates)
statesdiff=np.outer(np.ones(nstates),states)-np.outer(states,np.ones(nstates))
if a<0:  
    assortMat = 1 - evo.interactors.pM(statesdiff,alpha=1/a**2)
else:
    assortMat =     evo.interactors.pM(statesdiff,alpha=a**2)
        
mx.showdata(assortMat,colorbar=True)
