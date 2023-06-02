#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:41:57 2023

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
# network
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
# tensor object
nloci=100
filename='oc_tensor_' + str(nloci) + '.obj'
with bz2.BZ2File(obj_path / filename, 'rb') as f:
	h = pickle5.load(f)
#%%


#%%
ps=(23,37)
dev=np.random.rand(N) # p for the binomial distributions
theta=dev*np.diff(ps)+ps[0] # we set it to start at their environmental optima
#------
v0=evo.initialize_bin_explicit(N,nloci,dev); mx.showlist(v0.T)

test = evo.simulate_explicit(
    v0=v0,
    ntimesteps=50,
    h=h,
    theta=theta,
    alpha=0.02,
    xi_S=0.
)

mx.showlist(evo.dist_averages(test,ps))


#C0AF852B6365458F1396DE679CC974FF