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
import matplotlib.pyplot as plt
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

lpos = evo.transformations.negativeSaturator(l,v=20) # l positive

# np.where(np.isclose(l,0,atol=1e-15))
# np.where(lpos==np.inf)

lpos[np.where(np.isclose(l,0,atol=1e-15))]=evo.transformations.negativeSaturator(1e-15,v=20) # limit at 0 for negativeSaturator with v=10
sd(l, colorbar=True, symmetry=True)
sd(lpos,color='Reds', colorbar=True)
np.any(lpos==np.inf)

mx.showlist(lpos[10,:])


#lde = lpos * (1-DE/K)



w = v*lpos


filename='oc_tensor_' + str(nloci) + '.obj'
with bz2.BZ2File(obj_path / filename, 'rb') as f:
    h = pickle5.load(f)


r = np.c_[w.sum(1)]

r * DE / (1+DE*r/K)

ii = w[3]@h
tst = w@h

tst.shape
sd(tst[:,3,:])
sd(ii)


i=20
iii = w[i]@h@w[i]
tst = w@h@w.T
sd(tst[:,:,3])
sd(tst[:,3,3,I])
sd(iii[:,I])

sd(tst[:,i,:] - tst[:,:,i], colorbar=True)

np.all(np.isclose(w[i]@h@w[i].T, tst[:,i,i]))

sd(tst[20,:,:])

whw = (w@h@w.T).diagonal(0,1,2).T
sd(whw)

i=np.random.randint(N)
print(np.all(np.isclose(w[i]@h@w[i].T, tst[:,i,i])))
print(np.all(np.isclose(w[i]@h@w[i].T, whw[i])))
print(np.all(tst[:,i,i]==whw[i]))

v = (w@h@w.T).diagonal(0,1,2).T/r**2
sd(v)

w2=w/r
v2 = (w2@h@w2.T).diagonal(0,1,2).T
sd(v2)

np.all(np.isclose(v, v2))

# %% speed test
a = time.time()
for i in range(2000):
    l = (np.diag(dec_rates.flatten()) @ mutual_effs) @ (np.diag(DE.flatten()) @ p)
    #l = (np.outer(dec_rates, DE) * mutual_effs) @ p
b = time.time()
b - a
# %% 



fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(np.arange(nstates),l[1])
ax.plot(np.arange(nstates), evo.transformations.negativeSaturator(l[1],1))
ax.plot(np.arange(nstates), evo.transformations.negativeSaturator(l[1],20))
plt.show()


mx.showlist(negativeSaturator(np.linspace(-5,5,10),100))




# %% 
i=18

W = evo.transformations.negativeSaturator(l[i],20)
mx.showlist(W)


W_bar = (W*v[i]).sum()

w = W/W.sum()
np.isclose(1.0, w.sum())

w_bar = W_bar/W.sum()
np.isclose(w_bar, (w * v[i]).sum())

def fdep(v, phi=0):
    # if phi == 0:
    #     norm = 1
    # else:
    #     norm = phi/(np.exp(p)-1)
    # return np.exp(phi*v)*norm # this expression has integral = 1 in the interval [0,1]
    # return np.exp(phi*(v-v.mean())) #this is the "orthodox" form, but actually any value for v.mean gives the same result in the end.
    # for the sake of speed, we do
    return np.exp(phi*(v-20))
    

#%%
dw = w * fdep(v[i],-10)
dW = W_bar * dw / (dw * v[i]).sum()
#dW = W_bar * W * fdep (v[i],-10)/ (W * v[i] * fdep (v[i],-10)).sum() # alt. form
np.isclose(W_bar, (dW*v[i]).sum()) # absolute mean fitness (growth rate) must stay constant


fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(np.arange(nstates),v[i])
ax.plot(np.arange(nstates), W)
ax.plot(np.arange(nstates), dW)
plt.show()
#%%

#-------------- FDEP --------------
def fdep(v, phi=0):
    #np.exp(phi*v)*phi/(np.exp(p)-1) # this expression has integral = 1 in the interval [0,1]. Has a singularity at phi=0
    # return np.exp(phi*(v-v.mean())) #this is the "orthodox" form, but actually any value for v.mean gives the same result in the end. Even the expression above is pointless.
    # Therefore, for the sake of speed, we simply do
    return np.exp(np.c_[phi] * v)
    
W = evo.transformations.negativeSaturator(l,20)
W_bar = (W*v).sum(1)
Wf = fdep(v,-10) * W #precomputed reweighting of fdep
dW = np.outer((W_bar / (Wf * v).sum(1)), np.ones(nstates)) * Wf

#----------------------------------
#alt.form, faster?
W = evo.transformations.negativeSaturator(l,20)
f = fdep(v,-10)
Wv = W*v
np.outer(np.c_[Wv.sum(1)].T / (Wv*f).sum(1) , np.ones(nstates)) * W * f
np.c_[Wv.sum(1)]






d = np.random.rand(N)
np.exp(np.c_[d]*(v)) 


sd(np.c_[(W_bar / (Wf * v).sum(1))] * np.ones_like(v))

np.c_[d] * np.ones_like(v)

# %% speed test
a = time.time()
for i in range(50000):
    #np.outer((W_bar / (precomp_fdep_W * v).sum(1)), np.ones(nstates))* precomp_fdep_W
    # np.c_[(W_bar / (precomp_fdep_W * v).sum(1))] * precomp_fdep_W
    
    W = evo.transformations.negativeSaturator(l,20)
    f = fdep(v,-10)
    Wv = W*v
    np.outer(Wv.sum(1) / (Wv*f).sum(1) , np.ones(nstates)) * W * f
    
    # W = evo.transformations.negativeSaturator(l,20)
    # W_bar = (W*v).sum(1)
    # precomp_fdep_W = fdep(v,-10) * W #precomputed reweighting of fdep
    # dW = np.outer((W_bar / (precomp_fdep_W * v).sum(1)), np.ones(nstates))* precomp_fdep_W
    
    
    
b = time.time()
b - a



