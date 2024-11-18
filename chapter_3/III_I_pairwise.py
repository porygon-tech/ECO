#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1/3/2024 (original: Mon Jun  5 13:39:24 2023)

@author: stressed miki
Miki: I do not expect you to understand anything below at a first glance.
Its okay.
"""
import os
from pathlib import Path
import pickle5
import bz2
os.chdir(os.environ['HOME'] + '/LAB/ECO')
root = Path(".")
obj_path = root / 'data/obj'
img_path = root / 'gallery/timeseries'
dataPath = root / 'data/dataBase'

#%% imports 
import time
import sys
import numpy as np
import networkx as nx

#%% OWN LIBS
sys.path.insert(0, "./lib")
import evo
import matriX as mx

#*************** PARAM LOAD **********************************
# Insert the directory into the beginning of the module search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
loadedvarfile = sys.argv[1]
#eval("import " + loadedvarfile + " as params")
import params
#*************************************************************

# tensor object load
filename='oc_tensor_' + str(params.nloci) + '.obj'
with bz2.BZ2File(obj_path / filename, 'rb') as f:
    h = pickle5.load(f)

# silly name for this bundle of simulations
easyname = os.popen('python ~/LAB/gadgets_cloud/randWord.py').read().strip()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%% SIMULATION TOOLS
import multiprocessing
import psutil

def task_simulation_set(cola):
    np.random.seed()
    core_id = psutil.Process(os.getpid()).cpu_num()
    print("Core {0} running task {1}\npid: {2}\n".format(core_id,multiprocessing.current_process().name, os.getpid()))
    res=[]
    for i in cola:
        #///////////////////////////////////////////////////////////////////////////////
        #global ntimesteps
        global h
        #global ps
        #global K
        
        #------------------------------------------------------------------
        #================ thread-specific params (may change with simID). These can be passed as lists of size=nsimulations:
        
        #chkl = lambda a: (len(a) == params.nsimulations && len(a[0] == params.N) if hasattr(a, '__len__') else False) 
        # gives true if it is a list of length <nsimulations> and its first (or any) element is of length N ;false otherwise
        chkl = lambda a: (len(a) == params.nsimulations if hasattr(a, '__len__') else False) 
        # gives true if it is a list of length <nsimulations>;false otherwise

        i_D0       = params.D0       [i] if chkl(params.D0       ) else params.D0     
        i_surround = params.surround [i] if chkl(params.surround ) else params.surround
        i_theta    = params.theta    [i] if chkl(params.theta    ) else params.theta  
        i_m        = params.m        [i] if chkl(params.m        ) else params.m      
        i_alpha    = params.alpha    [i] if chkl(params.alpha    ) else params.alpha  
        i_v0       = params.v0       [i] if chkl(params.v0       ) else params.v0     
        i_A_e      = params.A_e      [i] if chkl(params.A_e      ) else params.A_e    
        i_a        = params.a        [i] if chkl(params.a        ) else params.a      
        i_d        = params.d        [i] if chkl(params.d        ) else params.d      

       #///////////////////////////////////////////////////////////////////////////////
        
        #print('Launching simulation ID={0} in core {1} (pid: {2})\n'.format(i,core_id, os.getpid()))
        os.system("echo \"\033[0;101m\033[1;97mLaunching simulation\033[0m ID="+str(i)+" in core "+str(core_id)+" (pid: "+str(os.getpid())+")\"") #should be echo -e in plain bash
        os.system("echo \"\"")
        simulation = evo.simulator(
            simID=i,
            #---------- =- below: flobal-specific params
            find_fixedpoints=False,
            ntimesteps  = params.ntimesteps,
            h           = h,
            ps          = params.ps,
            K           = params.K,
            #---------- =- below: thread-specific params (may change with simID)
            D0          = i_D0,
            surround    = i_surround,
            theta       = i_theta,
            m           = i_m,
            alpha       = i_alpha,
            v0          = i_v0,
            mutual_effs = i_A_e,
            a           = i_a,
            d           = i_d)
        
        simulation.run(tolD=5,tolZ=1e-7) # tolD=2,tolZ=1e-8
        simulation_dict=simulation.__dict__
        
        del simulation_dict['_h']
        res.append(simulation_dict)
    return(res)



def split(string, n):
    return [string[i:i+n] for i in range(0, len(string), n)]

#%%  SIMULATE

if __name__ == "__main__":
    

    FullSummary=False
    nprocessors = multiprocessing.cpu_count()
    simulations = np.empty(params.nsimulations, dtype=object)
    colas = split(np.arange(params.nsimulations), int(np.ceil(params.nsimulations/nprocessors)))
    
    print('RUNNING SIMULATION BATCH \"' + easyname + '\". SPAWNING ' + str(len(colas)) + ' PROCESSES...')
    print(colas,sep="\n")

    pool = multiprocessing.Pool(processes=nprocessors)
    results = pool.map(task_simulation_set, colas) # this is where magic happens :)
    results = np.array(results).flatten().tolist()
    timecode = str(time.ctime().replace(' ','_').replace(':',''))
    filename='SIMULATIONS_' + easyname + '_' + str(timecode) + loadedvarfile + '.obj'
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
            "\n connectance of " + str(A.sum()/params.N**2) + 
            "\n the population was " + ("" if np.array_equal(sim['dist_avgs'][0], sim['_v0']) else "NOT (!!) ") + "initialized with trait averages at their os.environmental optima." +
            #("\n the population was initialized with trait averages at their os.environmental optima." if np.array_equal(sim['dist_avgs'][0], sim['_v0']) else "") +
            "\n generations: " + str(sim['_ntimesteps']) +
            #"\n os.environmental optima" + str(sim['_theta']) +
            "\n phenotypic space: " + str(sim['_ps']) +
            "\n strength of the trait matching mechanism: " + str(sim['_alpha']) +
            "\n average level of os.environmental selection: " + str(1-np.mean(sim['_m'])) +
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
