#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:52:39 2024

@author: ubuntu
"""

#%% imports 
import sys
sys.path.insert(0, "./lib")
import evo
import matriX as mx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from  matriX import showdata as sd
mx.graphictools.inline_backend(True)

#%%

#%%
Np=10
Na=8
N = Np+Na
#%%
intraP=-1 # intraguild effect of plants
intraA=0 # intraguild effect of animals
inter=1 # 

c=0.2
# ERDOS RENYI
Ab = nx.adjacency_matrix(nx.fast_gnp_random_graph(N,c)).todense()

sd(Ab)

A = Ab.copy()
A[:Np,:Np] *= intraP
A[-Na:,-Na:] *= intraA
A[-Na:,:Np] *= inter
A[:Np,-Na:] *= inter

#np.fill_diagonal(A,-1) # IS THIS DONE?
sd(A)


#%%

T=np.linalg.inv(np.identity(N) - A)
#T=np.linalg.pinv(np.identity(N) - A)
sd(T,colorbar=True)
#%%
T= np.identity(N)
for i in range(1,100):
    T+=np.linalg.matrix_power(A, i)
sd(T,colorbar=True)
#%%

T=np.linalg.inv(np.identity(N) - A)

