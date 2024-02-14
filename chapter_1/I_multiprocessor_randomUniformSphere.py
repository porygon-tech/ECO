#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:56:51 2024

@author: ubuntu
"""

import time
from os import chdir, listdir, environ, system, popen
from pathlib import Path
import pickle5
import bz2
chdir(environ['HOME'] + '/LAB/ECO') #this line is for Spyder IDE only
root = Path(".")
obj_path = root / 'data/obj'
img_path = Path(environ['HOME']) / 'LAB/figures'
dataPath = root / 'data/dataBase'
#%% imports 
import sys
sys.path.insert(0, "./lib")
import evo
import matriX as mx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from  matriX import showdata as sd
'''
mx.graphictools.inline_backend(False)
mx.graphictools.inline_backend(True)
'''
#%%
''' THIS FILE IS NOT NECESSARY

import numpy as np
#%% improve this to adapt to number of cores
import multiprocessing

def ruHyperSphere(N,nsim=20000):
    r=np.random.normal(size=(nsim,N))
    r/=np.c_[(r**2).sum(1)**(1/2)]
    return r

# v=rUHyperSphere(200,200000)
# print(np.all(np.isclose((v**2).sum(1)**(1/2),1)))

def split(string, n):
    return [string[i:i+n] for i in range(0, len(string), n)]

def task(cola):
    global N
    global nsim
    v=ruHyperSphere(N,nsim)
    return v

if __name__ == "__main__":
    N=30
    nsim=200000
    nprocessors = multiprocessing.cpu_count()
    #simulations = np.empty(nsim, dtype=object)
    #colas = split(np.arange(nsim), int(np.ceil(nsim/nprocessors)))
    colas=[int(np.ceil(nsim/nprocessors))]*nprocessors
    print('RUNNING SIMULATION BATCH. SPAWNING ' + str(len(colas)) + ' PROCESSES ('+ str(colas[0])+' tasks each)')
    #nsimulations // nprocessors +1 

    pool = multiprocessing.Pool(processes=nprocessors)
    results = pool.map(task, colas) # this is where magic happens :)
    results = np.array(results).flatten().tolist()
    

    #v=randomUniformHyperSphere(5,2000)
    filename='VEC_' + str(N)+ '_'+ str(time.time()) + '.obj'
    with bz2.BZ2File(obj_path / "randomvectors" / filename, 'wb') as f:
        pickle5.dump(results, f)