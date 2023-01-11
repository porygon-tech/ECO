#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:32:15 2022

@author: roman
"""
#%% IMPORTS
from scipy.special import comb  
import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit

def bindist(n,k,p=0.5):
    return comb(n,k)*p**k*(1-p)**(n-k)

def showfunc(f,xlim=(-5,5),definition=100, **kwargs):
            x= np.linspace(xlim[0],xlim[1],definition)
            fig = plt.figure(); ax = fig.add_subplot(111)
            ax.plot(x,f(x,**kwargs))
            plt.show()
            
def showlist(l, distbins=False):
            fig = plt.figure(); ax = fig.add_subplot(111)
            ax.plot(np.arange(len(l)),list(l))
            plt.show()

def showdata(mat, color='magma', symmetry=False):
    mat = np.copy(mat)
    if symmetry:
        top = np.max([np.abs(np.nanmax(mat)),np.abs(np.nanmin(mat))])
        plt.imshow(mat.astype('float32'), interpolation='none', cmap='seismic',vmax=top,vmin=-top)
    else:
        plt.imshow(mat.astype('float32'), interpolation='none', cmap=color)
    plt.colorbar()
    plt.show()
    
def augment(x,a=10,b=5):
    return 1-a**(-b*x)

#%%  
def pM (i,j, alpha=50):
    return np.exp(-alpha*(i-j)**2)

def pB (i,j, alpha=50):
    return 1/(1+np.exp(-alpha*(i-j)))

#%%  
from os import chdir
from pathlib import Path
import pickle5
import bz2
chdir('/home/roman/LAB/ECO') #this line is for Spyder IDE only
root = Path(".")
obj_path = root / 'data/obj'

#%%  

n=50
nstates=n+1
filename='oc_tensor_' + str(n) + '.obj'
with bz2.BZ2File(obj_path / filename, 'rb') as f:
	oc_tensor = pickle5.load(f)
    
#%%

cx,cy,cz = list(map(np.ndarray.flatten, np.meshgrid(np.arange(nstates),np.arange(nstates),np.arange(nstates))))

fig = plt.figure(figsize=(12,9)); ax = fig.add_subplot(projection='3d')
#scat = ax.scatter3D(cz,cx,cy, c=(1-10**(-10*oc_tensor)).flatten(),alpha=0.9, s=40,cmap='gnuplot')
scat = ax.scatter3D(cz,cx,cy, c=oc_tensor.flatten(),alpha=0.9, s=10,cmap='gnuplot')
fig.colorbar(scat)
plt.show()
#%% POPULATION INITIALISATION
#------------------------
skw=0.7
ntimesteps = 200

def f(i):
    return 1+n-i
    #return 2**(-i/2)
    #return n**2-i**2
    #alpha=0.001; m=1/4*n
    #return np.exp(-alpha*(i-m)**2)

v = np.zeros((ntimesteps, nstates,1))
l = np.zeros((nstates,1))
for i in range(nstates):
    v[0,i] = bindist(n,i,skw)
    
for i in range(nstates):
    l[i] = f(i)

#------------------------
showlist(v[0])
showlist(l)
'''
w = v[0]*l
(w).sum()
n*(1-skw)+1
showlist(w)
'''
#%% STANDARD RUN

for t in range(1,ntimesteps):
    w = v[t-1]*l
    v[t] = ((w.T @ oc_tensor @ w) / w.sum()**2)[:,0]

showdata(v)

#%% AFTER A CHANGE IN ENVIRONMENTAL SELECTION
v2 = np.zeros((ntimesteps, nstates,1))
l2 = np.flip(l)

showlist(v[-1])
v2[0] = v[-1]
for t in range(1,ntimesteps):
    w2 = v2[t-1]*l2
    v2[t] = ((w2.T @ oc_tensor @ w2) / w2.sum()**2)[:,0]

showdata(        np.append(v,v2,axis=0))
showdata(augment(np.append(v,v2,axis=0)))









#%% 

alpha=0.5
s1=v[np.random.randint(ntimesteps)]
s2=v[np.random.randint(ntimesteps)]

skw_1, skw_2 = np.random.rand(2)

for i in range(nstates):
    s1[i] = bindist(n,i,skw_1)
    s2[i] = bindist(n,i,skw_2)


fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(np.arange(nstates),s1)
ax.plot(np.arange(nstates),s2)
plt.show()

m_comb_probs=s1@s2.T
showdata(m_comb_probs)
#%% 

v_s1 = np.zeros((ntimesteps, nstates,1))
v_s2 = np.zeros((ntimesteps, nstates,1))
l_s1 = np.zeros((ntimesteps, nstates,1))
l_s2 = np.zeros((ntimesteps, nstates,1))

l_s1[0] = s2
l_s2[0] = s1

v_s1[0] = s1
v_s2[0] = s2

for t in range(1,ntimesteps):
    
    w_s1 = v_s1[t-1]*l_s1[t-1]
    w_s2 = v_s2[t-1]*l_s2[t-1]
    v_s1[t] = ((w_s1.T @ oc_tensor @ w_s1) / w_s1.sum()**2)[:,0]
    v_s2[t] = ((w_s2.T @ oc_tensor @ w_s2) / w_s2.sum()**2)[:,0]
    #l_s1[t] = np.max(v_s2[t])-v_s2[t]
    #l_s2[t] = np.max(v_s1[t])-v_s1[t]
    l_s1[t] = 1/(1+100*v_s2[t])#*1/(1+1000*v_s1[t])
    #l_s2[t] = 1/(1+10000*v_s1[t])#*1/(1+10000*v_s2[t])
    l_s2[t] = v_s1[t]*1/(1+100*v_s2[t])

#%%
showdata(v_s1)
showdata(v_s2)

#%%
showlist(v_s1[5])

#%%

gi,gj = np.meshgrid(np.arange(nstates),np.arange(nstates))
i,j = gi.flatten(), gj.flatten()
z = list(map(pB, i,j, (np.repeat(0.1,nstates**2))))
p=np.array(z).reshape((nstates,nstates)).astype('float32')
showdata(p)

t=10
m_comb_probs=v_s1[t]@v_s2[t].T
showdata(m_comb_probs)

interactors= m_comb_probs*p
interactors/=interactors.sum()
showdata(interactors)





showdata(v_s1[t]@np.ones((1,nstates)))

showdata((np.ones((nstates,1))@v_s2[t].T)*p)
showdata(interactors / (v_s1[t]@np.ones((1,nstates))))


showlist(.sum(0))






showlist(l)
showlist(v_s2[t])
showlist(1/(1+100*l))

showlist()

v_s2[t]

#antagonism + self-frequency selection can lead to multimodal distributions


showdata(v_s1[t]@v_s2[t].T)





#%%
l=np.zeros(nstates)
for i in range(nstates):
    l+=v_s2[t,i]*pM(np.arange(nstates),i,alpha=0.1)
    
showlist(l)
