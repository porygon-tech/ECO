#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:56:01 2024

@author: ubuntu
"""
from os import chdir, listdir, environ, system
from pathlib import Path
import pickle5
import bz2
chdir(environ['HOME'] + '/LAB/ECO') #this line is for Spyder IDE only
root = Path(".")
obj_path = root / 'data/obj'
img_path = Path(environ['HOME']) / 'LAB/figures'
dataPath = root / 'data/dataBase'
#%% imports 
import time
import sys
import numpy as np
import matplotlib.pyplot as pl
import networkx as nx
#%% OWN LIBS
sys.path.insert(0, "./lib")
import evo
import matriX as mx
from matriX import showdata as sd
#%%

nstates = 51
N=23
mx.graphictools.inline_backend(True)



nloci = nstates -1 
alpha=0.5

mutual_effs = np.random.choice((0,1),(N,N))*0.1
mutual_effs[:,3]*=-2
mutual_effs[:,7]*=-3
mutual_effs[10,:]=0; mutual_effs[:,10]=0; mutual_effs[10,15]=0.1
np.fill_diagonal(mutual_effs, 0) 

A = ((mutual_effs!=0)+(mutual_effs.T!=0))+0

DE = np.c_[np.random.randint(0,100,N)]
dev = np.random.rand(N)

sd(mutual_effs, symmetry=True)
sd(A)



v = np.array([[evo.bindist(nloci,i,dev[species_id]) for i in range(nstates)] for species_id in range(N)])
sd (v)

p = np.array([evo.interactors.convpM(v[species_id], nstates, alpha) for species_id in range(N)])
sd(p)

p.sum(1)#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:56:01 2024

@author: ubuntu
"""


nloci = nstates -1 
alpha=0.5

mutual_effs = np.random.choice((0,1),(N,N))*0.1
mutual_effs[:,3]*=-2
mutual_effs[:,7]*=-3
mutual_effs[10,:]=0; mutual_effs[:,10]=0; mutual_effs[10,15]=0.1
np.fill_diagonal(mutual_effs, 0) 

A = ((mutual_effs!=0)+(mutual_effs.T!=0))+0

DE = np.c_[np.random.randint(0,100,N)]
dev = np.random.rand(N)

sd(mutual_effs, symmetry=True)
sd(A)



v = np.array([[evo.bindist(nloci,i,dev[species_id]) for i in range(nstates)] for species_id in range(N)])
sd (v)

p = np.array([evo.interactors.convpM(v[species_id], nstates, alpha) for species_id in range(N)])
sd(p)

p.sum(1)

DE[3] = 120; DE[7] = 0; DE[10] = 50; DE[15] = 51;
mx.showlist(p[3])

#%%

tst = np.ones((N,N))
tst[20,:] = 0.1
sd(tst)


sd(np.diag(DE.flatten()) @ np.ones((N,N)))
sd(np.diag(DE.flatten()) @ tst)
sd(tst @ np.diag(DE.flatten()))

#%%

tst = DE * p
sd(p)
sd(tst)

tst = ((mutual_effs @ p) * DE) # old method, wrong

tst = mutual_effs @ (DE * p) # new method, correct
tst = mutual_effs @ (np.diag(DE.flatten()) @ p) # alternative form, correct
sd(tst, colorbar=True, symmetry=True)
DE[7] = 100
tst = mutual_effs @ (DE * p)
sd(tst, colorbar=True, symmetry=True)

tst = (A * DE).T
sd(tst)
interacting_indivs = (A * DE).T.sum(1)
interacting_indivs[10] == DE[15]

interacting_indivs_sat = evo.holling_II(interacting_indivs, a=1, h=100) # one individual from species X cannot interact with more than 100 individuals, no matter the species

dec_rates=interacting_indivs_sat/interacting_indivs
rew = ((A * DE)*dec_rates).T
sd(rew,colorbar=True)
np.isclose(rew.sum(1), interacting_indivs_sat)


tst = mutual_effs @ (DE * p)
sd(tst, colorbar=True, symmetry=True)
tst = (mutual_effs*rew) @ p
sd(tst, colorbar=True, symmetry=True)


# %%========================================
inter = A @ DE #(A * DE).T.sum(1) # Sum of individuals in neighbor nodes to each species.
inter_sat = evo.holling_II(inter, a=1, h=100) # one individual from species X cannot interact with more than 100 individuals, no matter if they belong to same or different species
# this assumption is that the individuals that saturate the interaction environment of species X only come from nodes directly linked with X.'
# Other species don't interfere, which can be understood that they do not share the same spaces unless they are connected in the binary network.
dec_rates=inter_sat/inter
rew = ((A * DE)*dec_rates.T).T # rew = sA * np.outer(dec_rates, DE)
sd(rew,colorbar=True)
np.isclose(rew.sum(1), interacting_indivs_sat)


l_old = mutual_effs @ (DE * p)
l = (mutual_effs*rew) @ p
sd(l_old, colorbar=True, symmetry=True)
sd(l,     colorbar=True, symmetry=True)

l_pos = evo.transformations.negativeSaturator(l,v=100) # l positive
sd(l_pos, color='Reds', colorbar=True)


#%% ALTERNATIVE MATRIX MULTIPLICATION FORM
tst = (np.diag(dec_rates.flatten()) @ mutual_effs) @ (np.diag(DE.flatten()) @ p)
sd(tst, colorbar=True, symmetry=True)
np.all(np.isclose(l, tst))
# %%
m=0.9

theta = np.random.rand(N)*np.diff(ps)+ps[0]

states = np.linspace(ps[0],ps[1], nstates)
#statesdiff=np.outer(np.ones(nstates),states)-np.outer(states,np.ones(nstates))

thetadiff=np.outer(np.ones(N),states)-np.outer(theta,np.ones(nstates))
lll = l_pos ** m * (evo.interactors.pM(thetadiff,alpha=alpha)) ** (1-m)
sd(lll, color='Reds', colorbar=True)

# %%
ll2 = l_pos * m * (evo.interactors.pM(thetadiff,alpha=alpha)) ** (1-m)
sd(ll2, color='Reds', colorbar=True)










DE[3] = 120; DE[7] = 0; DE[10] = 50; DE[15] = 51;
mx.showlist(p[3])

#%%

tst = DE * p
sd(p)
sd(tst)

tst = mutual_effs @ (DE * p)
sd(tst, colorbar=True, symmetry=True)
DE[7] = 100
tst = mutual_effs @ (DE * p)
sd(tst, colorbar=True, symmetry=True)

tst = (A * DE).T
sd(tst)
interacting_indivs = (A * DE).T.sum(1)
interacting_indivs[10] == DE[15]

interacting_indivs_sat = evo.holling_II(interacting_indivs, a=1, h=100) # one individual from species X cannot interact with more than 100 individuals, no matter the species

dec_rates=interacting_indivs_sat/interacting_indivs
rew = ((A * DE)*dec_rates).T
sd(rew,colorbar=True)
np.isclose(rew.sum(1), interacting_indivs_sat)


tst = mutual_effs @ (DE * p)
sd(tst, colorbar=True, symmetry=True)
tst = (mutual_effs*rew) @ p
sd(tst, colorbar=True, symmetry=True)


# %%========================================
inter = (A * DE).T.sum(1) # Sum of individuals in neighbor nodes to each species.
inter_sat = evo.holling_II(inter, a=1, h=100) # one individual from species X cannot interact with more than 100 individuals, no matter if they belong to same or different species
# this assumption is that the individuals that saturate the interaction environment of species X only come from nodes directly linked with X.'
# Other species don't interfere, which can be understood that they do not share the same spaces unless they are connected in the binary network.
dec_rates=inter_sat/inter
rew = ((A * DE)*dec_rates).T

l_old = mutual_effs @ (DE * p)
l = (mutual_effs*rew) @ p
sd(l_old, colorbar=True, symmetry=True)
sd(l,     colorbar=True, symmetry=True)

l_pos = evo.transformations.negativeSaturator(l,v=100) # l positive
sd(l_pos, color='Reds', colorbar=True)

# %%
m=0.9

theta = np.random.rand(N)*np.diff(ps)+ps[0]

states = np.linspace(ps[0],ps[1], nstates)
#statesdiff=np.outer(np.ones(nstates),states)-np.outer(states,np.ones(nstates))

thetadiff=np.outer(np.ones(N),states)-np.outer(theta,np.ones(nstates))
lll = l_pos ** m * (evo.interactors.pM(thetadiff,alpha=alpha)) ** (1-m)
sd(lll, color='Reds', colorbar=True)

# %%
ll2 = l_pos * m * (evo.interactors.pM(thetadiff,alpha=alpha)) ** (1-m)
ll2 = l_pos *     (evo.interactors.pM(thetadiff,alpha=alpha)) ** (1-m)
sd(ll2, color='Reds', colorbar=True)



# %%========================================
#   =========== IMPLEMENTATION =============
#   ========================================

inter = A @ DE # Sum of individuals in neighbor nodes to each species.
inter_sat = evo.holling_II(inter, a=1, h=100) # one individual from species X cannot interact with more than 100 individuals, no matter if they belong to same or different species
# this assumption is that the individuals that saturate the interaction environment of species X only come from nodes directly linked with X.'
# Other species don't interfere, which can be understood that they do not share the same spaces unless they are connected in the binary network.
dec_rates=inter_sat/inter

#method 1 - diagonal matrices
l = (np.diag(dec_rates.flatten()) @ mutual_effs) @ (np.diag(DE.flatten()) @ p)
#method 2 - omega (outer product), almost twice as fast (x1.7)
l = (np.outer(dec_rates, DE) * mutual_effs) @ p
sd(l, colorbar=True, symmetry=True)





# %% speed test
a = time.time()
for i in range(2000):
    l = (np.diag(dec_rates.flatten()) @ mutual_effs) @ (np.diag(DE.flatten()) @ p)
    #l = (np.outer(dec_rates, DE) * mutual_effs) @ p
b = time.time()
b - a


