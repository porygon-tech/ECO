#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:29:57 2022

@author: roman
"""

#%% IMPORTS
from scipy.special import comb  
import numpy as np
#import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
import sys



#%%
def bindist(n,k,p=0.5):
    return comb(n,k)*p**k*(1-p)**(n-k)

def oc(v,n,i,j):
    #prob of getting phenotype v from parent phenotypes i,j with n loci
    sumvar=0
    v=int(v)
    n=int(n)
    i=int(i)
    j=int(j)
    for x in range(i+1):
        sumvar+=comb(i,x) * comb(n - i, j - x) / comb(n,j)*bindist(i + j - 2*x, v - x)
    return sumvar

#%% OC TENSOR GENERATION
from os import chdir, listdir, environ
from pathlib import Path
import pickle5
import bz2
chdir(environ['HOME'] + '/LAB/ECO') #this line is for Spyder IDE only
root = Path(".")
obj_path = root / 'data/obj'


nloci= int(sys.argv[1]) if len(sys.argv)>1 else 100
n=nloci
nstates=nloci+1
x = np.arange(nstates)
y = np.arange(nstates)
gx,gy = np.meshgrid(x,y)
x, y = gx.flatten(), gy.flatten()

n_list=np.repeat(nloci,nstates**2)
oc_tensor = np.zeros((nstates,nstates,nstates))


#%%
import os
def split(string, n):
    return [string[i:i+n] for i in range(0, len(string), n)]

import threading

def task_ocgen(cola):
    print("Task 1 assigned to thread: {}".format(threading.current_thread().name))
    print("ID of process running task 1: {}".format(os.getpid()))
    global oc_tensor
    global nstates
    global n_list
    for v in cola:
        print(threading.current_thread().name + ' running v='+str(v))
        v_list=np.repeat(v,nstates**2)
        z = list(map(oc, v_list,n_list,x,y))
        mat=np.array(z).reshape((nstates,nstates)).astype('float32')
        oc_tensor[v,...] = mat[np.newaxis,...]
    
#%%
if __name__ == "__main__":
      
    colas = split(np.arange(nstates),int(nstates/32))
    threads=[]
    for i,cola in enumerate(colas):
        # creating threads
        threads.append(threading.Thread(target=task_ocgen, args=[cola], name='calc_'+str(i)))
    
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
        
    #%%
    
    filename='oc_tensor_' + str(n) + '.obj'
    print('\nSAVING TENSOR AS ' + print(obj_path / filename))
    with bz2.BZ2File(obj_path / filename, 'wb') as f:
        pickle5.dump(oc_tensor, f)
    
    
