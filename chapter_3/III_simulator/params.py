#!/usr/bin/env python3
import numpy as np
import os
import sys
from pathlib import Path
os.chdir(os.environ['HOME'] + '/LAB/ECO')
root = Path(".")
sys.path.insert(0, "./lib")
import evo

nsimulations = 64

#%% HYPERPARAMETERS
N=23 # number of species. Has an important relation with the mutualRange and c parameters
nloci=50 # number of loci
ps=(20,30) # phenotypic space. Has an important relation with the alpha parameter
ntimesteps = 80 # number of generations simulated
K=100 # carrying capacity

#%% MODEL PARAMETERS (simulation)
c=np.log(N)/N# expected connectance of the allowed links matrix. Use np.log(N)/N for the connected regime of erdos-renyi graphs (critical point for single component)
a=0. # assortative mating coefficients, real value (single value or array of N values)
d=np.linspace(-0,0,nsimulations) # frequency dependence coefficient
alpha= 0.1 #0.1 # strength of the trait matching mechanism. Positive real value. Lower values = greater promiscuity
xi_S=.5 # level of environmental selection (from 0 to 1).

A = np.ones((N,N)); np.fill_diagonal(A, 0) # fully connected 
g1,g2 = g = np.array([-.0 ,0.1]) # payoffs for symmetric games
A_e = A*g2 # purely mutualist, no intraspecific competition
np.fill_diagonal(A_e, g1) 


''' SILENCED AS IT WILL CHANGE BETWEEN SIMULATIONS
#%% environmental optima generation (theta)
dev=np.random.rand(N)       # values of the environmental optima, normalized to [0,1)
theta=dev*np.diff(ps)+ps[0] # now over the phenotypic space

#%% initialization of phenotype makeups
v0=evo.initialize_bin_explicit(N,nloci,dev) # set to start at their environmental optima
#v0=evo.initialize_bin_explicit(N,nloci,np.random.rand(N)) # set to start at random location
'''

#%% generation of vector of levels of selection imposed by other species
xi_d=1-xi_S
m=np.clip(np.random.normal(xi_d,0.00001,(N,1)),0,1) # vector of levels of selection imposed by other species (from 0 to 1)

# initial population sizes (array of N values or single value (same for all species))
D0=50

#//// CHANGING VALUES //////////////////////////////
dev=[]
theta=[]
v0=[]
for i in range(nsimulations):
    dev.append(np.random.rand(N))
    theta.append(dev[i]*np.diff(ps)+ps[0]) # now over the phenotypic space
    v0.append(evo.initialize_bin_explicit(N,nloci,dev[i])) # set to start at their environmental optima
    

