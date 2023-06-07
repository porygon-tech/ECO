#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:39:24 2023

@author: ubuntu
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
# import matplotlib.pyplot as plt
# from copy import deepcopy
# import networkx as nx
# import pandas as pd
#%% OWN LIBS
sys.path.insert(0, "./lib")
import evo
import matriX as mx
#%% HYPERPARAMETERS
N=23 # number of species. Has an important relation with the mutualRange and c parameters
nloci=100 # number of loci
ps=(23,37) # phenotypic space. Has an important relation with the alpha parameter
ntimesteps = 80 # number of generations simulated
K=1000 # carrying capacity  

# tensor object load
filename='oc_tensor_' + str(nloci) + '.obj'
with bz2.BZ2File(obj_path / filename, 'rb') as f:
	h = pickle5.load(f)


#%%



#%% MODEL PARAMETERS
c=0.5 # expected connectance of the allowed links matrix
mutualRange = (-0.01, 0.01) # range of values for the uniform distribution of ecological effects
a=0. # assortative mating coefficients, real value (single value or array of values)
d=0. # frequency dependence coefficient
alpha= 0.01 # strength of the trait matching mechanism. Positive real value. Lower values = greater promiscuity
xi_S=0.1 # level of environmental selection (from 0 to 1).
D0=50 # initial population sizes (single value or array of values)


# a=np.linspace(0, 1, N) # assortative mating coefficients, real value (array)

#%% mutual effects matrix generation

A = mx.symmetric_connected_adjacency(N,c); #mx.showdata(A)
A_e = A* (np.random.rand(N,N)*np.diff(mutualRange)+mutualRange[0]);#mx.showdata(A_e,symmetry=True,colorbar=True)

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
'''
# at this point we can run simulations:
sim_test = evo.simulator(
    v0=v0,
    ntimesteps=ntimesteps,
    h=h,
    mutual_effs=A_e,
    theta=theta,
    ps=ps,
    alpha=alpha,
    xi_S=xi_S,
    D0=D0,
    a=a,
    K=K)


sim_test.run()

mx.showlist(sim_test.dist_avgs)
mx.showlist(sim_test.D)
mx.showlist(sim_test.fits)
'''

#%% SIMULATION TOOLS
import multiprocessing
import psutil
import os
def task_simulation_set(cola):
    np.random.seed()
    core_id = psutil.Process(os.getpid()).cpu_num()
    print("Core {0} running task {1}\npid: {2}\n".format(core_id,multiprocessing.current_process().name, os.getpid()))
    #print('starting', flush=True)

    #global simulations
    global nsimulations
    for i in cola:
        #///////////////////////////////////////////////////////////////////////////////
        global ntimesteps
        global h
        global ps
        global K
        
        # global D0
        # global theta
        # global xi_S
        # global alpha
        # global v0
        # global mutual_effs
        # global a
        

        # MODEL PARAMETERS
        c=0.4 # expected connectance of the allowed links matrix
        mutualRange = (-0.01, 0.02) # range of values for the uniform distribution of ecological effects
        # a=np.linspace(-0.001, 0.001,nsimulations)[i] #np.random.rand(N)*2-1#np.random.rand()*2-1 # assortative mating coefficients, real value (single value or array of values)
        d=np.linspace(-5,5,nsimulations)[i] 
        # alpha= np.random.rand()/10 # strength of the trait matching mechanism. Positive real value. Lower values = greater promiscuity
        # xi_S=np.random.rand()# level of environmental selection (from 0 to 1).
        D0=100 # initial population sizes (single value or array of values)
        
        # mutual effects matrix generation
        A = mx.symmetric_connected_adjacency(N,c)
        A_e = A* (np.random.rand(N,N)*np.diff(mutualRange)+mutualRange[0])
        
        # environmental optima generation (theta)
        # dev=np.random.rand(N) 
        # theta=dev*np.diff(ps)+ps[0] 

        # initialization of phenotype makeups
        # v0=evo.initialize_bin_explicit(N,nloci,dev); # set to start at their environmental optima
        # v0=evo.initialize_bin_explicit(N,nloci,np.random.rand(N)); # set to start at random location
        #///////////////////////////////////////////////////////////////////////////////
        
        simulation = evo.simulator(
            ntimesteps=ntimesteps,
            h=h,
            ps=ps,
            K=K,
            D0=D0,
            theta=theta,
            xi_S=xi_S,
            alpha=alpha,
            v0=v0,
            mutual_effs=A_e,
            a=a,
            d=d)
       
        simulation.run()
        simulation_dict=simulation.__dict__
        del simulation_dict['_h']
        simulations[i] = simulation_dict
        return(simulation_dict)



def split(string, n):
    return [string[i:i+n] for i in range(0, len(string), n)]

#%%  SIMULATE

if __name__ == "__main__":
    ntimesteps=200
    nsimulations = 5
    nprocessors = multiprocessing.cpu_count()
    simulations = np.empty(nsimulations, dtype=object)
    colas = split(np.arange(nsimulations), int(np.ceil(nsimulations/nprocessors)))
    #nsimulations // nprocessors +1 
    '''
    processes = []
    for i, cola in enumerate(colas):
        # Creating processes
        processes.append(multiprocessing.Process(target=task_simulation_set, args=(cola,), name='calc_'+str(i)))
    print('SPAWNED ' + str(len(colas)) + ' PROCESSES')
    for process in processes:
        process.start()

    for process in processes:
        process.join()
    '''

    pool = multiprocessing.Pool(processes=nprocessors)
    results = pool.map(task_simulation_set, colas)


    timecode = str(time.time())
    #filename='SIMULATIONS' + str(timecode) + '.obj'
    filename='SIMULATIONS_test.obj'
    print('\nSAVING SIMULATION SET AS ' + str(obj_path / filename))
    with bz2.BZ2File(obj_path / filename, 'wb') as f:
        pickle5.dump(results, f)



