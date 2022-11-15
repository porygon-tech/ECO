#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:32:48 2022

@author: roman

This script uses evo.py to get an idea of whether population size affects the trait evolution.
"""
from os import chdir
chdir('/home/roman/LAB/ECO') #this line is for Spyder IDE only
import sys
sys.path.insert(0, "./lib")
#%%
import evo
import numpy as np
#%%
nloci = 100
nindivs = 200
ps = (500,500+nloci)
pop = evo.population(2000,nloci, skew= 0.5,phenoSpace=ps);pop.show()

#%%
print(pop.history)

#%%
n=nloci
m=nindivs
a = np.zeros((1,n+1))

b = np.ones((4,n+1))


unique, counts = np.unique(pop.phenotypes, return_counts=True)
pos = (unique - ps[0])*nloci/np.diff(ps)[0] 
b=np.zeros((1,n+1))
b[0,(pos).astype(int)] = counts/m
np.concatenate((a,b),axis=0)

#%%
skew=0.2
pop.mtx = np.random.choice((0,1),(nindivs,nloci), p=(1-skew, skew))

pop.history
#%%

c=pop.reproduce(30)

c.show()

pop.hist()
c.hist()

evo.showdata(c.history)
