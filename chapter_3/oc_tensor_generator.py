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
from os import chdir
from pathlib import Path
import pickle5
import bz2
chdir('/home/roman/LAB/ECO') #this line is for Spyder IDE only
root = Path(".")
obj_path = root / 'data/obj'

nloci=int(sys.argv[1])
n=nloci
nstates=nloci+1
x = np.arange(nstates)
y = np.arange(nstates)
gx,gy = np.meshgrid(x,y)
x, y = gx.flatten(), gy.flatten()

n_list=np.repeat(nloci,nstates**2)
oc_tensor = np.zeros((nstates,nstates,nstates))

#
for v in range(nstates):
    print('v='+str(v))
    v_list=np.repeat(v,nstates**2)
    z = list(map(oc, v_list,n_list,x,y))
    mat=np.array(z).reshape((nstates,nstates)).astype('float32')
    oc_tensor[v,...] = mat[np.newaxis,...]



filename='oc_tensor_' + str(n) + '.obj'
with bz2.BZ2File(obj_path / filename, 'wb') as f:
    pickle5.dump(oc_tensor, f)