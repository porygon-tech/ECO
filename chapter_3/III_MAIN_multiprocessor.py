#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:39:24 2023

@author: ubuntu
Miki: I do not expect you to understand anything below at a first glance.
Its okay.
"""
from os import chdir, listdir, environ,popen
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
import networkx as nx
# import pandas as pd
#%% OWN LIBS
sys.path.insert(0, "./lib")
import evo
import matriX as mx
#%% HYPERPARAMETERS
N=23 # number of species. Has an important relation with the mutualRange and c parameters
nloci=50 # number of loci
ps=(20,30) # phenotypic space. Has an important relation with the alpha parameter
ntimesteps = 80 # number of generations simulated
K=1000 # carrying capacity  

# tensor object load
filename='oc_tensor_' + str(nloci) + '.obj'
with bz2.BZ2File(obj_path / filename, 'rb') as f:
	h = pickle5.load(f)


#%%
easyname = popen('python ~/LAB/gadgets_cloud/randWord.py').read().strip()

#%% MODEL PARAMETERS
c=np.log(N)/N# expected connectance of the allowed links matrix. Use np.log(N)/N for the connected regime of erdos-renyi graphs (critical point for single component)
# mutualRange = (-0.01, 0.02) # range of values for the uniform distribution of ecological effects
a=0. # assortative mating coefficients, real value (single value or array of values)
d=0. # frequency dependence coefficient
alpha= 0.1#0.1 # strength of the trait matching mechanism. Positive real value. Lower values = greater promiscuity
xi_S=0.25 # level of environmental selection (from 0 to 1).
# D0=50 # initial population sizes (single value or array of values)


# a=np.linspace(0, 1, N) # assortative mating coefficients, real value (array)

#%% mutual effects matrix generation

A = mx.symmetric_connected_adjacency(N,c); #mx.showdata(A)
# A_e = A* (np.random.rand(N,N)*np.diff(mutualRange)+mutualRange[0]);#mx.showdata(A_e,symmetry=True,colorbar=True)

# c=0.5
# A = nx.adjacency_matrix(nx.fast_gnp_random_graph(N,c)).todense()
g1,g2 = g = np.array([-0.1,0.1]) # payoffs for symmetric games
# A_e = np.random.choice(g,(N,N))*A

#g = 1/(100*N*c); g1,g2 = g = (g,g)
# g = 1/(10*N*c); g1,g2 = g = (g,g)

'''
N_producers=10
N_consumers=10
A_e = mx.ecomodels.structured_triple(N,N_producers,N_consumers,g=g,
                                     consumer_comp=True,
                                     consumer_nest=False)
A=(A_e!=0)+0
'''

# mx.showdata((A_e>0) & (A_e.T>0)) # mutualisms
# mx.showdata((A_e<0) & (A_e.T<0)) # antagonisms (competitors)
# mx.showdata((A_e>0) & (A_e.T<0)) # antagonisms (predation)
'''
print("connectance of " + str(A.sum()/N**2) + ", expected " + str(c))
print("generated with: \n{0} mutualisms, \n{1} antagonisms (competitors), and \n{2} antagonisms (predation)".format(int(((A_e>0) & (A_e.T>0)).sum()/2),
                                                int(((A_e<0) & (A_e.T<0)).sum()/2),
                                                ((A_e>0) & (A_e.T<0)).sum()))
'''
#%% environmental optima generation (theta)
dev=np.random.rand(N)       # ran
theta=dev*np.diff(ps)+ps[0] 

#%% initialization of phenotype makeups
v0=evo.initialize_bin_explicit(N,nloci,dev); # set to start at their environmental optima
v0=evo.initialize_bin_explicit(N,nloci,np.random.rand(N)); # set to start at random location

#%% generation of vector of levels of selection imposed by other species
xi_d=1-xi_S
m=np.clip(np.random.normal(xi_d,0.001,(N,1)),0,1) 

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
    res=[]
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
        # c=np.random.rand()*0.7+0.3 # expected connectance of the allowed links matrix
        # mutualRange = np.sort(np.random.rand(2)*0.02-0.01) #(-0.02, 0.02) # range of values for the uniform distribution of ecological effects
        # a=np.clip(np.random.normal(np.random.rand()*3,0.01,N),0,6)#np.linspace(0., 0.005,nsimulations)[i]#(np.random.rand()*2-1)/10#  #np.random.rand(N)*2-1#np.random.rand()*2-1 # assortative mating coefficients, real value (single value or array of values)
        # d=np.random.normal(np.random.rand()*4-2,0.0001,N)#np.random.normal(np.linspace(-2,2,nsimulations)[i],0.01,N) #np.linspace(-5,5,nsimulations)[i]  #frequency dependence coefficient
        #d=np.random.normal(np.linspace(-2,2,nsimulations)[i],0.0001,N)
        
        # alpha= np.random.rand()*0.1 # strength of the trait matching mechanism. Positive real value. Lower values = greater interaction promiscuity
        
        # xi_S=np.random.rand()# level of environmental selection (from 0 to 1).
        # xi_d=1-xi_S
        m=np.clip(np.random.normal(xi_d,0.001,(N,1)),0,1) # vector of levels of selection imposed by other species (from 0 to 1)
        
        D0=300 # initial population sizes (single value or array of values)
        
        # mutual effects matrix generation
        # A = mx.symmetric_connected_adjacency(N,c)
        # A_e = A* (np.random.rand(N,N)*np.diff(mutualRange)+mutualRange[0])
        
        # c=0.9
        graph = nx.fast_gnp_random_graph(N,c)
        while not nx.is_connected(graph):
            graph = nx.fast_gnp_random_graph(N,c)
        A = nx.adjacency_matrix(graph).todense()
        # g1,g2 = g = np.array([-0.02,0.02]) # payoffs for symmetric games
        #A_e = np.random.choice(g,(N,N))*A #mixed interactions
        A_e = A/np.c_[A.sum(1)]*0.05
        np.fill_diagonal(A_e, -0.05) #??? necessary?
        
        # A_e = np.array([[ 0.  , -0.02, -0.02,  0.01, -0.  ,  0.  ,  0.01],
        #        [-0.02, -0.  , -0.02,  0.  , -0.  ,  0.01,  0.01],
        #        [-0.02, -0.02, -0.  ,  0.01, -0.02, -0.  , -0.  ],
        #        [-0.02,  0.  ,  0.01, -0.  , -0.02, -0.  ,  0.  ],
        #        [-0.  ,  0.  , -0.02,  0.01, -0.  ,  0.  , -0.02],
        #        [-0.  ,  0.01, -0.  , -0.  , -0.  ,  0.  ,  0.01],
        #        [-0.02,  0.01, -0.  ,  0.  ,  0.01, -0.02, -0.  ]])
        # A = A_e!=0
        # environmental optima generation (theta)
        dev=np.random.rand(N) 
        theta=dev*np.diff(ps)+ps[0] 
        # theta=np.array([20.05078094, 21.10033134, 23.03367894, 20.38220173, 26.73308045,
        #        24.02761609, 20.23429722])

        # initialization of phenotype makeups
        v0=evo.initialize_bin_explicit(N,nloci,dev); # set to start at their environmental optima
        # v0=evo.initialize_bin_explicit(N,nloci,np.random.rand(N)); # set to start at random location
        #///////////////////////////////////////////////////////////////////////////////
        
        #print('Launching simulation ID={0} in core {1} (pid: {2})\n'.format(i,core_id, os.getpid()))
        os.system("echo \"\033[0;101m\033[1;97mLaunching simulation\033[0m ID="+str(i)+" in core "+str(core_id)+" (pid: "+str(os.getpid())+")\"") #should be echo -e in plain bash
        os.system("echo \"\"")
        simulation = evo.simulator(
            simID=i,
            find_fixedpoints=True,
            ntimesteps=ntimesteps,
            h=h,
            ps=ps,
            K=K,
            D0=D0,
            theta=theta,
            m=m,
            alpha=alpha,
            v0=v0,
            mutual_effs=A_e,
            a=a,
            d=d)
        
        simulation.run(tolD=5,tolZ=1e-7) # tolD=2,tolZ=1e-8
        simulation_dict=simulation.__dict__
        
        del simulation_dict['_h']
        #simulations[i] = simulation_dict
        res.append(simulation_dict)
    return(res)



def split(string, n):
    return [string[i:i+n] for i in range(0, len(string), n)]

#%%  SIMULATE

if __name__ == "__main__":
    
    ntimesteps=100
    nsimulations = 32
    
    FullSummary=False
    nprocessors = multiprocessing.cpu_count()
    simulations = np.empty(nsimulations, dtype=object)
    colas = split(np.arange(nsimulations), int(np.ceil(nsimulations/nprocessors)))
    
    print('RUNNING SIMULATION BATCH \"' + easyname + '\". SPAWNING ' + str(len(colas)) + ' PROCESSES...')
    print(colas,sep="\n")
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
    results = pool.map(task_simulation_set, colas) # this is where magic happens :)
    results = np.array(results).flatten().tolist()

    #timecode = str(time.time())
    timecode = str(time.ctime().replace(' ','_').replace(':',''))
    #timecode = 'test'
    filename='SIMULATIONS_' + easyname + '_' + str(timecode) + '.obj'
    print('\nSAVING SIMULATION SET AS ' + str(obj_path / filename))
    with bz2.BZ2File(obj_path / filename, 'wb') as f:
        pickle5.dump(results, f)


    #---REPORT---
    file = open(str(obj_path / ('report_'+filename+'.txt')),'w')
    keylist = list(results[0].keys())[:-5]
    fixed = {}
    for key in keylist:
        lst = [sim[key] for sim in results]
        fixed[key] = np.all([(par == lst[0]) for par in lst])
    
    file.write('REPORT for simulation ' +str(timecode) + ' ' + '='*23+ '\n')
    file.write('non-fixed values:\n')
    file.write(''.join(["\t"+key+"\n" for key, value in fixed.items() if not value]))
    if FullSummary:
        for sim in results:
            N   = sim['v'].shape[1]
            A_e = sim['_mutual_effs']
            A=A_e != 0
            
            file.write("\n" + '-'*32 + "\n")
            file.write(
            "\n connectance of " + str(A.sum()/N**2) + 
            "\n the population was " + ("" if np.array_equal(sim['dist_avgs'][0], sim['_v0']) else "NOT (!!) ") + "initialized with trait averages at their environmental optima." +
            #("\n the population was initialized with trait averages at their environmental optima." if np.array_equal(sim['dist_avgs'][0], sim['_v0']) else "") +
            "\n generations: " + str(sim['_ntimesteps']) +
            #"\n environmental optima" + str(sim['_theta']) +
            "\n phenotypic space: " + str(sim['_ps']) +
            "\n strength of the trait matching mechanism: " + str(sim['_alpha']) +
            "\n average level of environmental selection: " + str(1-np.mean(sim['_m'])) +
            "\n initial population sizes: " + str(sim['_D0']) +
            "\n assortative mating coefficients: " + str(sim['_a']) +
            "\n frequency dependence coefficients: " + str(sim['_d']) +
            "\n carrying capacity: " + str(sim['_K']) +
            "\n mutualisms: " + str(sim['n_mutualisms']) +
            "\n competitions: " + str(sim['n_competitions']) +
            "\n predations: " + str(sim['n_predations']) + 
            "\n"
            )
        
    file.close()
