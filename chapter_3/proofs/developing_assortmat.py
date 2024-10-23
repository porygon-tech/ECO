#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:56:58 2024

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
import matplotlib.pyplot as plt
from copy import deepcopy
import networkx as nx
import pandas as pd
#%% OWN LIBS
sys.path.insert(0, "./lib")
import evo
import matriX as mx
mx.graphictools.inline_backend(True)
sd = mx.showdata
I=np.newaxis
#%%
'''
BRIEF SUMMARY:
I need a mathematical expression for assortative mating. I will check if it coincides with the individual based.
'''
#%%
nloci = 100
skw = 0.25
ntimesteps=223
d=-3.
a=0.1

ps = (500,500+nloci)
pop = evo.population(500,nloci, skew= skw,phenoSpace=ps);pop.show()
#%%
nstates = nloci+1
v0 = np.c_[np.array([evo.bindist(nloci,i,skw) for i in range(nstates)])]
mx.showlist(v0)
l = evo.interactors.pM(np.arange(nstates)-20,0.007) + evo.interactors.pM(np.arange(nstates)-80,0.007)
l = np.c_[np.array(4*l)]
mx.showlist(l)
#%%
filename='oc_tensor_' + str(nloci) + '.obj'
with bz2.BZ2File(obj_path / filename, 'rb') as f:
	h = pickle5.load(f)
    
#%%
a=-0.01
states = np.linspace(ps[0],ps[1], nstates)
statesdiff=np.outer(np.ones(nstates),states)-np.outer(states,np.ones(nstates))
assortMat = evo.interactors.pM(statesdiff,alpha=abs(a))

if a<0:
    assortMat = 1 - assortMat
aT = np.repeat(assortMat[I,...],nstates,axis=0)

sd(assortMat, color="viridis",colorbar=True)

#%%
mx.showdata(aT1[20,:,:])

v = np.zeros((ntimesteps+1, nstates,1))
v[0] = v0
t=1
mx.showdata(v)
f = np.exp(d * v[t-1])
# Wv = v[t-1]*l
# Wbar = Wv.sum(); np.nan_to_num(Wbar);
# dl = Wbar / (Wv*f).sum() * l * f
# Wv = dl      
# r=Wbar
# dlr = dl / r ; np.nan_to_num(dlr)
# vv =  dlr * v[t-1] # reproductive potential

Wvf = f*v[t-1]*l
rho = Wvf/Wvf.sum()


#------
    ha = h*aT # not the same as h@aT1
    #v1 = (rho.T@h@rho)[:,0]
    v1 = (rho.T@ha@rho)[:,0]; v1/=v1.sum()
    mx.showlist(v1)
    
    
    mx.showlist(rho)
    mx.showlist(Wvf)
    mx.showlist((rho.T@h@rho)[:,0])
    mx.showlist((Wvf.T@h@Wvf)[:,0]/Wvf.sum()**2)
    
    WvfA = assortMat@Wvf
    rhoA = WvfA/WvfA.sum()
    v1A = assortMat@(rho.T@h@rho)[:,0]
    mx.showlist(v1A)
    mx.showlist(rhoA)
    
#------

sex=True
if sex:
    ha = h*aT # not the same as h@aT1
    #v1 = (rho.T@h@rho)[:,0]
    v1 = (rho.T@ha@rho)[:,0]; v1/=v1.sum()
    v[t,:]=v1 
else:
    v[t,:] = rho # NO SEX :(
    

mx.showdata(h[:,:,50])
mx.showdata(h[:,50,:])
mx.showdata(h[50,:,:])



mx.showdata(ha[:,:,50])
mx.showdata(ha[:,50,:])
mx.showdata(ha[50,:,:])


#%%
d=-29.
a=-0.001
skw=.51
v0 = np.c_[np.array([evo.bindist(nloci,i,skw) for i in range(nstates)])]
# v0 =1/nstates
v[0] = v0
assortMat = evo.interactors.pM(statesdiff,alpha=abs(a))
if a<0:
    assortMat = 1 - assortMat
aT = np.repeat(assortMat[I,...],nstates,axis=0)
mx.showdata(assortMat)
for t in range(1,ntimesteps+1):
#for t in range(1,10):
    f = np.exp(d * v[t-1])
    Wvf = f*v[t-1]*l
    rho = Wvf/Wvf.sum()
    sex=True
    if sex:
        ha = h*aT # not the same as h@aT1
        #v1 = (rho.T@h@rho)[:,0]
        v1 = (rho.T@ha@rho)[:,0]; v1/=v1.sum() # same as (Wvf.T@h@Wvf)[:,0]/Wvf.sum()**2
        v[t,:]=v1 
    else:
        v[t,:] = rho # NO SEX :(
    
mx.showdata(v)
#%% minimal example
nstates =101
v0 = [1/nstates]*nstates
l = evo.interactors.pM(np.arange(nstates)-20,0.007) + evo.interactors.pM(np.arange(nstates)-80,0.007)
v = evo.predict(v0,l,ntimesteps=100,h=h,
                d=-30.01,
                a=0.01,
                sex=True)
mx.showdata(v)

#%%many species
N=13
ps=[500,700]
alpha=0.0035
d=-29.
a=0.001

v = np.zeros((ntimesteps+1, N, nstates))
#l = np.zeros((ntimesteps+1, N, nstates))

states = np.linspace(ps[0],ps[1], nstates)
statesdiff=np.outer(np.ones(nstates),states)-np.outer(states,np.ones(nstates))
assortMat = evo.interactors.pM(statesdiff,alpha=abs(a))

v[0] = [1/nstates]*nstates
l =np.array(list(map(lambda x: evo.interactors.pM(states-(x*(ps[1]-ps[0])+ps[0]),alpha)+evo.interactors.pM(states-states.mean(),alpha), np.random.rand(N))))
l = np.repeat(l[I,...],ntimesteps,axis=0)
#sd(l[0])


assortMat = evo.interactors.pM(statesdiff,alpha=abs(a))
if a<0:
    assortMat = 1 - assortMat
aT = np.repeat(assortMat[I,...],nstates,axis=0)
ha = h*aT # not the same as h@aT1
#sd(assortMat)

for t in range(1,ntimesteps+1):
#for t in range(1,10):
    f = np.exp(np.c_[d] * v[t-1])
    Wvf = f*v[t-1]*l[t-1]
    rho = Wvf/np.c_[Wvf.sum(1)]
    sex=True
    if sex:
        #v1 = (rho.T@ha@rho)[:,0]
        v1 = (rho@ha@rho.T).diagonal(0,1,2).T; v1=v1/np.c_[v1.sum(1)]# same as (Wvf.T@h@Wvf)[:,0]/Wvf.sum()**2
        v[t,:]=v1 
    else:
        v[t,:] = rho # NO SEX :(
    
mx.showdata(v[-1,:])
#%% compare with individual version
i=2
v_i = evo.predict(v0=v[0,i],
                l=l[0,i],
                ntimesteps=100,
                h=h,
                d=d,
                a=a,
                ps=ps,
                sex=True)
mx.showdata(v_i)
mx.showdata(v[:,i])


#%% many species with different assort coeffs
N=13
ps=[500,700]
alpha=0.0035
d=-25.

v = np.zeros((ntimesteps+1, N, nstates))

states = np.linspace(ps[0],ps[1], nstates)
statesdiff=np.outer(np.ones(nstates),states)-np.outer(states,np.ones(nstates))

v[0] = [1/nstates]*nstates
l =np.array(list(map(lambda x: evo.interactors.pM(states-(x*(ps[1]-ps[0])+ps[0]),alpha)+evo.interactors.pM(states-states.mean(),alpha), np.random.rand(N))))
l = np.repeat(l[I,...],ntimesteps,axis=0)

#------ alternative: same fitness landscape for everyone
n_defpeaks=3
pks = np.linspace(0,nstates,n_defpeaks+2)[1:-1]
l = np.array([evo.interactors.pM(np.arange(nstates)-pk,alpha) for pk in pks]).sum(0)*10
l = np.repeat(l[I,...],N,axis=0)
l = np.repeat(l[I,...],ntimesteps,axis=0)
#------

a = np.linspace(-.01,.01,N)
assortTen = np.zeros((N,nstates,nstates))

for i in range(N):
    assortMat = evo.interactors.pM(statesdiff,alpha=abs(a[i]))
    if a[i]<0:  
        assortMat = 1 - assortMat
    assortTen[i] = assortMat

hal = np.array([h*np.repeat(aM[I],nstates,axis=0) for aM in assortTen])

for t in range(1,ntimesteps+1):
#for t in range(1,10):
    f = np.exp(np.c_[d] * v[t-1])
    Wvf = f*v[t-1]*l[t-1]
    rho = Wvf/np.c_[Wvf.sum(1)]
    sex=True
    if sex:
        #v1 = (rho.T@ha@rho)[:,0]
        # v1 = (rho@ha@rho.T).diagonal(0,1,2).T;                  v1=v1/np.c_[v1.sum(1)]# same as (Wvf.T@h@Wvf)[:,0]/Wvf.sum()**2
        v1 = (rho@hal@rho.T).diagonal(0,2,3).diagonal(0,0,2).T; v1=v1/np.c_[v1.sum(1)]
        v[t,:]=v1 
    else:
        v[t,:] = rho # NO SEX :(
    
mx.showdata(v[-1,:])
# pz=rho@hal@rho.T
# v1 = (rho@hal@rho.T).diagonal(0,2,3).diagonal(0,0,2).T; v1=v1/np.c_[v1.sum(1)]
# sd(v1)
# sd(rho)

#%% make sure tensor calculations are properly done in both the individual and the multispecies cases
a=0.01
assortMat = evo.interactors.pM(statesdiff,alpha=abs(a))
if a<0:
    assortMat = 1 - assortMat
aT = np.repeat(assortMat[I,...],nstates,axis=0)
ha = h*aT # not the same as h@aT1

tst = (rho@ha@rho.T).diagonal(0,1,2).T
sd(tst)

i=np.random.randint(N)
rho_1 = np.c_[rho[i]]
print(np.all(np.isclose(np.c_[tst[i]],(rho_1.T@ha@rho_1)[:,0])))

# multiple a -------------------------------------------
assortTen = np.repeat(assortMat[I,...],N,axis=0)
hal = np.array([h*np.repeat(aM[I],nstates,axis=0) for aM in assortTen])
tst2 = (rho@hal@rho.T).diagonal(0,2,3).diagonal(0,0,2).T
print(np.all(np.isclose(np.c_[tst2[i]],(rho_1.T@ha@rho_1)[:,0])))
print(np.all(np.isclose(np.c_[tst2[i]],np.c_[tst[i]])))

#%% comprobamos de nuevo que todo se parece, tambien a lo largo de la simulacion
i=2
v_i = evo.predict(v0=v[0,i],
                l=l[0,i],
                ntimesteps=ntimesteps,
                h=h,
                d=d,
                a=a[i],
                ps=ps,
                sex=True)
sd(v[:,i])
sd(v_i)
np.all(np.isclose(np.squeeze(v_i), v[:,i]))

#%% speed test
import time
ntries = 1000
dummy_a=0.

start = time.time()
for i in range(ntries):
    (rho@hal@rho.T).diagonal(0,2,3).diagonal(0,0,2).T
print(time.time()-start)

start = time.time()
for i in range(ntries):
    for hal_i in hal:
        (rho@hal_i@rho.T).diagonal(0,1,2).T
print(time.time()-start)

start = time.time()
for i in range(ntries):
    if not hasattr(dummy_a, "__len__"): 
        (rho@ha@rho.T).diagonal(0,1,2).T
print(time.time()-start)

#%% Find how many peaks it fills
n_defpeaks = 6
alpha=0.035
res=20
#-----------
nstates =101
v0 = [1/nstates]*nstates

pks = np.linspace(0,nstates,n_defpeaks+2)[1:-1]
# pks = np.linspace(0,nstates,n_defpeaks)

l = np.array([evo.interactors.pM(np.arange(nstates)-pk,alpha) for pk in pks]).sum(0)*10
#l = evo.interactors.pM(np.arange(nstates)-20,0.007) + evo.interactors.pM(np.arange(nstates)-80,0.007)

mx.showlist(l)
#%% 

xr = np.linspace(0, 0.03,res)
yr = np.linspace(-50, 0, res)
gx,gy = np.meshgrid(xr,yr)
x, y = gx.flatten(), gy.flatten()

ntimesteps=120
def f(x,y):
    return evo.predict(v0,l,ntimesteps=ntimesteps,h=h,
                       d=y,
                       a=x,
                       sex=True)

z = list(map(f, x,y))
#%%
from scipy.signal import find_peaks
peaks = list(map(find_peaks,[zi[-1,:,0] for zi in z]))
npeaks = [p[0].size for p in peaks]

plt.imshow(np.array(npeaks).reshape((res,res)).astype('float32'), interpolation='none', cmap='GnBu_r');plt.show()
plt.imshow(np.array(npeaks).reshape((res,res)).astype('float32'), interpolation='bicubic', cmap='GnBu_r');plt.show()
#%%
plt.imshow(np.array(npeaks).reshape((res,res)).astype('float32'), interpolation='none', cmap='GnBu_r',origin='upper')
plt.colorbar(label='number of peaks in the distribution')

tk = np.arange(0,res,2)
plt.xticks(ticks=tk, labels=np.round(xr,3)[tk], rotation=45)
plt.yticks(ticks=tk+1, labels=np.round(yr,1)[tk+1], rotation=45)
plt.xlabel(r"assortative mating $a$")
plt.ylabel(r"frequency-dependent selection $\phi$")
plt.show()

#%%
max(npeaks)
np.where(peaks)

mx.showdata(z[np.ravel_multi_index((12,17), (res,res))])
mx.showdata(z[np.ravel_multi_index((7,6), (res,res))])


#%%
'''
sudo apt install msttcorefonts -qq -y
rm ~/.cache/matplotlib -rf 
'''
#%%
csfont = {'fontname':'Comic Sans MS'}
hfont = {'fontname':'Helvetica'}
coords = (res-19,5)
plt.imshow(z[np.ravel_multi_index(coords, (res,res))], interpolation='none', cmap='plasma',origin='lower')
plt.xlabel("trait value",**csfont)
plt.ylabel("generation", **hfont)
plt.colorbar(label='frequency')
plt.title(r"$a=$" + str(round(xr[coords[1]],3)) + r", $\phi=$" + str(round(yr[coords[0]],2)))
plt.show()


#%%

'''
p0=(rho@ha@rho.T)
p0.shape
pz=rho@hal

pz.shape
sd(pz.diagonal(0,1,3))

sd()
res=pz.diagonal(0,1,3).diagonal(0,0,1).T
sd(res)

sd(pz[10,:,:,0])
sd(pz[:,0,10,:])

p = (rho@ha@rho.T).diagonal(0,1,2).T
sd(p)
sd(rho)

assortTen = np.linspace(0,1,N)[:,I,I] * np.repeat(assortMat[I,...],N,axis=0)

rhoA = rho[:,I,:]@assortTen
rhoA.shape
sd(rhoA[:,:,-1])

haN =np.repeat(ha[I,...],N,axis=0)
haN =np.linspace(0,1,N)[:,I,I,I]*haN
assortTen.shape

(rho@haN).shape
(rho@ha).shape
(rho).shape
sd(rho)
(rho@haN).shape
(rho@haN@rho.T).shape
(rho@ha@rho.T).shape

p=(rho@assortTen)
p=(rho@ha@rho.T)
p.shape
p.diagonal()
sd(p[])
#------------------------------------------
'''