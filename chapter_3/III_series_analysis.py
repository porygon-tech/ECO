#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 08:21:34 2023

@author: ubuntu
"""

from os import chdir, listdir, environ, system
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
# import matplotlib.pyplot as plt
# from copy import deepcopy
# import networkx as nx
# import pandas as pd
#%% OWN LIBS
sys.path.insert(0, "./lib")
import evo
import matriX as mx

#%% run model
system('python chapter_3/III_MAIN_multiprocessor.py')

#%% load dataset
#pattern = r'^file\d+\.txt$'  # Regular expression pattern


listdir(obj_path)
filename = 'SIMULATIONS_test.obj'
with bz2.BZ2File(obj_path / filename, 'rb') as f:
	simulations = pickle5.load(f)
    
#%%
for sim in simulations:
    mx.showlist(sim['dist_avgs'])

#%%
for sim in simulations:
    mx.showlist(sim['fits'][:-1])

#%%
for sim in simulations:
    mx.showlist(sim['D'][:-1])
    
#%%
[sim['_a'] for sim in simulations]
[sim['_d'] for sim in simulations]


#%% check which parameters are set to be fixed
keylist = simulations[0].keys()
fixed = {}
for key in keylist:
    lst = [sim[key] for sim in simulations]
    fixed[key] = np.all([(par == lst[0]) for par in lst])

print('non-fixed values:')
print(*["\t"+key for key, value in fixed.items() if not value], sep="\n")


#%%
A_e = sim['_mutual_effs']
A=A_e != 0
print("connectance of " + str(A.sum()/N**2))
print("generated with: \n{0} mutualisms, \n{1} antagonisms (competitors), and \n{2} antagonisms (predation)".format(int(((A_e>0) & (A_e.T>0)).sum()/2),
                                                                                                                    int(((A_e<0) & (A_e.T<0)).sum()/2),
                                                                                                                    ((A_e>0) & (A_e.T<0)).sum()))
#%%

mx.showdata(np.diff(simulations[1]['v'][:,5],1,axis=1))
mx.showdata()


np.isclose(np.diff(simulations[1]['v'][:,5],1,axis=1),0,atol=1e-2)

#%%
states = np.linspace(ps[0],ps[1], nstates)
statesdiff=np.outer(np.ones(nstates),states)-np.outer(states,np.ones(nstates))
for alpha in np.linspace(-0.001, 0.001,10):
    assortMat = pM(statesdiff,alpha=abs(alpha))
    if alpha<0:
        assortMat = 1 - assortMat
    mx.showdata(assortMat,colorbar=True)




