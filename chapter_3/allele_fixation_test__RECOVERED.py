#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:52:29 2023

@author: roman
"""

#%% set path
#!git clone https://github.com/porygon-tech/ECO
#import sys
#sys.path.insert(0,'/content/ECO/lib')

from os import chdir, listdir, environ, system
from pathlib import Path
import pickle5
import bz2
chdir(environ['HOME'] + '/LAB/ECO') #this line is for Spyder IDE only
root = Path(".")
obj_path = root / 'data/obj'
img_path = Path(environ['HOME']) / 'LAB/figures'
obj_path = root / 'data/obj'
#%% imports 
import sys
sys.path.insert(0, "./lib")
import evo
import matriX as mx
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
#%%
import matplotlib.colors
def rescale(arr: np.ndarray, vmin=0,vmax=1):
    return  (arr - vmin) / (vmax - vmin)

def blendmat(mat1,mat2,mat3=None,saturation = 1.1,additive=False):
    if not mat3:
        mat3=mat2.copy()
    temp_max=np.max((mat1,mat2,mat3))
    temp_min=np.min((mat1,mat2,mat3))

    R_r = rescale(mat1, temp_min,temp_max) #clip?
    G_r = rescale(mat2, temp_min,temp_max)
    B_r = rescale(mat3, temp_min,temp_max)
    if additive:
        cmapgrn = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "green"]) #seagreen also
        cmapred = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "red"])
        cmapblu = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "blue"])
        
        blended = 1 - (1 - cmapred(R_r)) * (1 - cmapgrn(G_r)) * (1 - cmapblu(B_r))
        blended = mx.cNorm(blended,saturation)
    else:
        cmapgrn = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "magenta"]) #seagreen also
        cmapred = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "cyan"])
        cmapblu = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "yellow"])
        
        blended = (cmapred(R_r)+cmapgrn(G_r)+cmapblu(B_r))/3
        blended = mx.cNorm(blended,1/saturation)
    
    fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
    pos = ax.imshow(blended,interpolation='None')
    #fig.suptitle(r'$\alpha=$'+str(alpha)+r'$, a_{12}=$'+str(a12)+r'$, a_{13}=$'+str(a13)+', b='+str(b),y=0.75)
    #ax.set_ylim(0,n)  # decreasing time
    ax.set_ylabel('Trait value')
    ax.set_xlabel('Time (generations)')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

#%% START
'''
Here we compare the trait trajectories for three different cases.
1. Normal population
2. The same, but when a locus becomes fixed, the chromosome of the most variable individual gets rearranged to keep variability in this locus.
3. The analytical prediction assuming infinite populations

'''
#%% 
nloci = 100
ps = (500,500+nloci)

filename='oc_tensor_' + str(nloci) + '.obj'
with bz2.BZ2File(obj_path / filename, 'rb') as f:
	h = pickle5.load(f)
#%%
pop = evo.population(500,nloci, skew= 0.25,phenoSpace=ps)

def f(x):
  return x-500
pop.set_fitnessLandscape(f)

#%%
pop.hist(20)
pop.showfitness()

#%%
ntimesteps = 200
nstates=nloci+1

c=pop.reproduce(ntimesteps,verbose=True)
mx.showdata(c.history, colorbar=True, color='binary')
c.show()

cf=pop.reproduce(ntimesteps,verbose=True,nofix=True)
mx.showdata(cf.history, colorbar=True, color='binary')
cf.show()

#%% analytical prediction
states = np.linspace(ps[0],ps[1], nstates)
v = np.zeros((ntimesteps+1, nstates,1))
v[0] = pop.history[0][:,np.newaxis]

l = np.zeros((nstates,1))
for i in range(nstates):
    l[i] = f(states[i])

for t in range(1,ntimesteps+1):
    w = v[t-1]*l
    v[t] = ((w.T @ h @ w) / w.sum()**2)[:,0]

mx.showdata(v, colorbar=True, color='binary')
mx.showlist(l)

mat1=c.history.T
mat2=cf.history.T
mat3=v.squeeze().T

mx.blendmat(mat1,mat2,mat3,saturation = 3)



#%%
#-------------------------------

def f(x):
  return 1200-2*x

for i in range(nstates):
    l[i] = f(states[i])


a=deepcopy(c)
b=deepcopy(cf)
a.set_fitnessLandscape(f)
b.set_fitnessLandscape(f)

a.showfitness()
a.show()
#evo.nofixation(b.mtx)
b.show()


ntimesteps_2 = 200
a=a.reproduce(ntimesteps_2,verbose=True)
b=b.reproduce(ntimesteps_2,verbose=True,nofix=True)

evo.showdata(a.history, colorbar=True)
evo.showdata(b.history, colorbar=True)

#analytical pred.
v2 = np.zeros((ntimesteps_2+1, nstates,1))
v2[0] = v[-1]

for t in range(1,ntimesteps_2+1):
    w = v2[t-1]*l
    v2[t] = ((w.T @ h @ w) / w.sum()**2)[:,0]
    
v3=np.append(v,v2[1:],0)
mx.showdata(v3)

mat1=a.history.T
mat2=b.history.T
mat3=v3.squeeze().T

mx.blendmat(mat1,mat2,mat3,saturation = 5.5,additive=True)
mx.blendmat(mat1,mat2,mat3,saturation = 5.5,additive=False)

#%%
''' THIS ONE DOES NOT MAKE SENSE UNLESS THERE IS LINKAGE DISEQUILIBRIUM
fig = plt.figure(); ax = fig.add_subplot(111)
ax.scatter(np.arange(ps[0],ps[1]), (c.mtx.mean(0)*2-1)**2,s=5)
ax.set_xlabel('position', labelpad=10)
ax.set_ylabel('locus fixation coefficient', labelpad=10)
plt.show()
'''


#%%


#%%
n, bins, patches = plt.hist((c2.mtx.mean(0)*2-1)**2, 20, density=True)
plt.xlim(0,1)
plt.grid(True)
plt.xlabel('locus fixation coefficient', labelpad=10)
plt.ylabel('density', labelpad=10)
plt.show()
c2.show()


#%% mutation time!

mx.showlist(v3[-1])


from scipy.special import comb  
def bindist(n,k,p=0.5):
    return comb(n,k)*p**k*(1-p)**(n-k)


n=nloci
mu = 0.1
def mutate_dep(p,mu):
    n=len(p)-1
    nstates=n+1
    mutao = np.zeros((nstates))
    
    for b in range(nstates): 
        for i in range(nstates):
            s=0
            for k in range(i+1):
                s+= bindist(i,i-k,mu) * bindist(n-i,b-k,mu)
            mutao[b]+=p[i] * s
    return mutao


# faster function?:
def mutate(p,mu):
    #p is a vector containing the probabilities of each state before mutation
    n=len(p)-1
    nstates=n+1
    return list(map(lambda b: sum(list(map(lambda i: p[i] * sum(list(map(lambda k: bindist(i,i-k,mu) * bindist(n-i,b-k,mu), list(range(i+1))))), list(range(nstates))))), list(range(nstates))))

import time
def profile(f, *args):
    start = int(round(time.time() * 1000))
    f(*args)
    print('spent {0} milliseconds'.format(int(round(time.time() * 1000))-start))


profile(mutate_dep,p,mu)
profile(mutate,p,mu)

'''
import threading
import os
def mutateOPT(p,mu):
    #p is a vector containing the probabilities of each state before mutation
    n=len(p)-1
    nstates=n+1
    p_out=np.zeros((nstates))
    def task1():
        print("Task 1 assigned to thread: {}".format(threading.current_thread().name))
        print("ID of process running task 1: {}".format(os.getpid()))
        p_out[:int(nstates/2)]  = list(map(lambda b: sum(list(map(lambda i: p[i] * sum(list(map(lambda k: bindist(i,i-k,mu) * bindist(n-i,b-k,mu), list(range(i+1))))), list(range(nstates))))), list(range(int(nstates/2)))))
    def task2():
        print("Task 2 assigned to thread: {}".format(threading.current_thread().name))
        print("ID of process running task 1: {}".format(os.getpid()))
        p_out[int(nstates/2):] = list(map(lambda b: sum(list(map(lambda i: p[i] * sum(list(map(lambda k: bindist(i,i-k,mu) * bindist(n-i,b-k,mu), list(range(i+1))))), list(range(nstates))))), list(range(int(nstates/2),nstates))))
        
    # creating threads
    t1 = threading.Thread(target=task1, name='t1')
    t2 = threading.Thread(target=task2, name='t2')  
  
    # starting threads
    t1.start()
    t2.start()
  
    # wait until all threads finish
    t1.join()
    t2.join()
    return p_out
'''




#%%
p=np.zeros((nstates))
p[10:30]=1/(30-10)
mx.showlist(p)
mx.showlist(mutate(p,0.02))
mx.showlist(mutate(p,0.999))



ps = (500,500+nloci)
pop = evo.population(1500,37, skew= 0.1,phenoSpace=ps)
pop.show()
pop.mutate(rate=0.1)
pop.hist()


#%%
pop = evo.population(500,nloci, skew= 0.25,phenoSpace=ps)

def f(x):
  return x-500
pop.set_fitnessLandscape(f)
ntimesteps = 200
nstates=nloci+1
#%%
mu=0.01
c =pop.reproduce(ntimesteps,verbose=True,mu=mu)
c2=pop.reproduce(ntimesteps,verbose=True)
#%%
c.show() # we will find a mutation-selection balance worth studying :)
c2.show()

mat1= c.history.T
mat2=c2.history.T
mat3=mat2
blendmat(mat1,mat2,mat3,saturation = 3)

#-----------------------------------------
#we generate the mutation matrix
mut=np.array([ list(map(lambda i: sum(list(map(lambda k: bindist(i,i-k,mu) * bindist(n-i,b-k,mu), list(range(i+1))))), list(range(nstates)))) for b in list(range(nstates))])
mx.showdata(mut)

states = np.linspace(ps[0],ps[1], nstates)
v = np.zeros((ntimesteps+1, nstates,1))
v[0] = pop.history[0][:,np.newaxis]

l = np.zeros((nstates,1))
for i in range(nstates):
    l[i] = f(states[i])

for t in range(1,ntimesteps+1):
    w = v[t-1]*l
    v[t] = ((w.T @ h @ w) / w.sum()**2)[:,0]
    v[t] = mut @ v[t]
    print(t)

mx.showdata(v, colorbar=True, color='binary')


mat3=v.squeeze().T
blendmat(mat1,mat2,mat3,saturation = 3)

#%%
'''
mu=0.1
p=np.zeros((nstates))
p[10:30]=1/(30-10)
mx.showlist(p)
mx.showlist(mutate(p,mu))

mx.showlist(mut@p)

(mut@p).shape
'''
ntimesteps=150
K=2000 
N=np.zeros(ntimesteps+1)
v = np.zeros((ntimesteps+1, nstates,1))
v[0] = pop.history[0][:,np.newaxis]


N[0]=200
for t in range(1,ntimesteps+1):
#for t in range(1,10):
    w = v[t-1]*l/30
    r = w.sum()
    N[t] = (1-1/(N[t-1] * r/K+1))*K
    #N[t] = (N[t-1]*r - K) / (1 - np.exp(1*(N[t-1]*r - K))) + N[t-1]*r
    #maynard smith 1968, May 1972
    v[t] = ((w.T @ h @ w) / w.sum()**2)[:,0]
    v[t] = mut @ v[t]
    print(t)

mx.showdata(v, colorbar=True, color='binary')
mx.showlist(N[:20])
mx.showlist(N)



def predict(N0,l,mu=0.,ntimesteps=100):
    if type(mut) == float:
        mut= evo.generate_mut_matrix(nstates,mu=mu)

    
l = np.zeros((nstates,1))
for i in range(nstates):
    l[i] = 
np.array([f(s) for s in states])
        
 
























#%%
import evo
from copy import deepcopy
nloci = 37
ps = (500,500+nloci)
pop = evo.population(1000,nloci, skew= 0.25,phenoSpace=ps)

def f(x):
    mxo=4 # can have 5 offsprings under optimal conditions
    alpha=0.008; m=525
    #return (x-499)*mxo/100
    return np.exp(-alpha*(x-m)**2)*mxo

pop.set_fitnessLandscape(f)
pop.showfitness()

#%%

c=pop.reproduce(100,verbose=True)

c=deepcopy(pop)
K=2000

ngenerations = 50
timeseries_popsize = np.zeros(ngenerations)
for g in range(ngenerations):
    print(g, c.m)
    timeseries_popsize[g] = c.m
    ft = lambda pv: f(pv)*(1 - c.m/K)
    c.set_fitnessLandscape(ft)
    c=c.reproduce(1,fixedSize=False)


#%%
c.show()
c.hist()
evo.showdata(c.history, colorbar=True)
c.showfitness()
#%%
fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(np.arange(ngenerations), timeseries_popsize)
ax.set_xlabel('time (generations)', labelpad=10)
ax.set_ylabel('population size', labelpad=10)
plt.show()
#%%
c=c.reproduce(100,verbose=True)
#%%
n, bins, patches = plt.hist((c.mtx.mean(0)*2-1)**2, int(c.n/4), density=True)
plt.xlim(0,1)
plt.grid(True)
plt.xlabel('locus fixation coefficient', labelpad=10)
plt.ylabel('density', labelpad=10)
plt.show()

#%%
n, bins, patches = plt.hist(1/2-np.abs(c.mtx.sum(0)/c.m -1/2), int(c.n/4), density=True)
plt.xlim(0,0.5)
plt.grid(True)
plt.xlabel('minimum allele frequency (MAF)', labelpad=10)
plt.ylabel('density', labelpad=10)
plt.show()

#%%
c=deepcopy(pop)
ngenerations = 300
timeseries_fixations = np.zeros((ngenerations,nloci))
for g in range(ngenerations):
    print(g)
    c=c.reproduce(1)
    timeseries_fixations[g]=c.mtx.mean(0)

#%%

evo.showdata(c.mtx[100:200,:])
evo.showdata((c.mtx.mean(0)*2-1)*np.ones((10,c.n)),colorbar=True,symmetry=True)

maf=1/2-np.abs(timeseries_fixations -1/2)

evo.showdata(1-maf,colorbar=True)
evo.showdata((1-maf)==1,colorbar=True)

evo.showdata(timeseries_fixations*2-1,symmetry=True)






