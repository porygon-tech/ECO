#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:41:57 2023

@author: roman
"""


from os import chdir, listdir
from pathlib import Path
import pickle5
import bz2
chdir('/home/roman/LAB/ECO') #this line is for Spyder IDE only
root = Path(".")
obj_path = root / 'data/obj'
img_path = root / 'gallery/timeseries'
#%% imports 
import sys
sys.path.insert(0, "./lib")
import evo
import matriX as mx
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import networkx as nx
#%%
def pM (zdiffs, alpha=50):
    return np.exp(-alpha*(zdiffs)**2)

def convpM(values,nstates,alpha):
  c = np.zeros((nstates))
  for i in range(nstates):
    c = c + pM(np.arange(nstates)-i, alpha)*values[i]
  return c

#%% Initial setup
nloci = 100
ps = (500,500+nloci)
mu=0.00
# ---------------

nstates = nloci+1
states = np.linspace(ps[0],ps[1], nstates)

filename='oc_tensor_' + str(nloci) + '.obj'
with bz2.BZ2File(obj_path / filename, 'rb') as f:
	h = pickle5.load(f)


mut= evo.generate_mut_matrix(nstates,mu=mu)

#%% generate agent population 

N0=300 # initial population size
pop = evo.population(N0,nloci, skew= 0.25,phenoSpace=ps)

#%% define individual fitness generating function
a=0.001
def f(x):
    #return (x-500)/26
    return pM(x-570,alpha=a)*5

l = np.zeros((nstates,1))
for i in range(nstates):
    l[i] = f(states[i])

pop.set_fitnessLandscape(f) ; pop.showfitness()

#%%STANDARD RUN
'''
1. the population reproduces with constant size and forced variability (to palliate loci fixation effects)
2. predictions are made
3. the average 
'''
ntimesteps = 200 

# 1.
c=pop.reproduce(ntimesteps,verbose=True,mu=mu, fixedSize=True,nofix=True)

N=np.zeros(ntimesteps+1)
v = np.zeros((ntimesteps+1, nstates,1))
v[0] = pop.history[0][:,np.newaxis]
#N[0]=N0
for t in range(1,ntimesteps+1):
    w = v[t-1]*l
    r = w.sum()
    #N[t] = (1-1/(N[t-1] * r/K+1))*K
    # N[t] = N[t-1] * r * (1 - N[t-1]/K)
    
    # r = np.log(w.sum())
    # N[t] = N[t-1] * np.e**r * (1 - N[t-1]/K)
    
    #N[t] = (N[t-1]*r - K) / (1 - np.exp(1*(N[t-1]*r - K))) + N[t-1]*r
    #maynard smith 1968, May 1972?
    v[t] = ((w.T @ h @ w) / w.sum()**2)[:,0]
    v[t] = mut @ v[t]
    print(t)

phi=0.05
# phi=a**(1 / np.e)+a # approx. but shitty
z = np.zeros((ntimesteps+1,1))
z[0]=(np.arange(nstates)*v[0].T).sum() #mean
theta=np.where(l==max(l))[0][0] #max of f (should have only one)


for t in range(1,ntimesteps+1):
    z[t]=z[t-1]+phi*(theta-z[t-1])

# mx.showlist(N[:20])
# mx.showlist(N)

#%% 

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
imsh = ax.imshow(v[...,0].T, interpolation='none', cmap='magma',origin='lower')
ax.plot(np.arange(ntimesteps+1),(v[...,0]*np.arange(nstates)).mean(1)*nstates,color='yellow')
ax.plot(np.arange(ntimesteps+1),z,color='cyan')
fig.colorbar(imsh)
plt.show()


#%% Here we see all three ways to represent evolutionary change
mx.showlist(z)
mx.showdata(v, colorbar=True, color='binary')
mx.showdata(c.history, colorbar=True, color='binary')
c.show()

#%%


#%% set simulation parameters
#N=b.shape[0]
N=23 # number of species
c=0.3 # expected connectance
ntimesteps=50

#%%
initial_l= mx.generateWithoutUnconnected(N,N,c) 
initial_l=initial_l-np.diag(np.diag(initial_l))
initial_l=np.tril(initial_l,0)+np.tril(initial_l,0).T
#initial_l=b
mx.showdata(initial_l)

print("connectance of " + str(initial_l.sum()/N**2) + ", expected " + str(c))
nx.draw(nx.from_numpy_array(initial_l,parallel_edges=False))
#%% generate theta
theta=np.random.rand(N)*np.diff(ps)+ps[0] # values favoured by env. selection
#%% simulate
mechanism_type = 1 # the rest of mechanisms can be taken from coevolutionary models/andreazzi...
# 1: 'mutualism + trait matching' or 
# 2: 'mutualism + exploitation barrier' or 
# 3: 'antagonism + trait matching (trait mismatching)' or
# 4: 'antagonism + exploitation barrier' 

xi_S=0.7 # level of environmental selection (from 0 to 1).
xi_d=1-xi_S # level of selection imposed by resource species (from 0 to 1).
alpha= 0.1 # strength of the mechanism. Controls how the difference in species traits affects the probability of pairwise interaction.
epsilon = 0.5 # threshold, assumed to be fixed and identical for all species.
phi=0.1 # slope of the selection gradient.

#-------------------------------
z=np.zeros((ntimesteps,N)) # trait values
S=np.zeros((ntimesteps,N)) # partial selection differentials caused by the environment
p=np.zeros((ntimesteps,N,N)) # pairwise interaction probabilities
l=np.zeros((ntimesteps,N,N)) # 1 if species i and j interact at time t and 0 otherwise 
a=np.zeros((ntimesteps,N,N)) # evolutionary effect of the interaction between species i and j at time t 
u=np.zeros((ntimesteps,N,N))
M=np.zeros((ntimesteps,N,N)) # partial selection differentials caused by other species (trait matching)
B=np.zeros((ntimesteps,N,N)) # partial selection differentials caused by other species (exploitation barrier)


z[0,:] = theta # for simplicity, species start at their environmental optima
S[0,:] = xi_S*(theta-z[0,:]) 
p[0,:,:]=pM(z[0,:]*np.ones((N,1)) - (z[0,:]*np.ones((N,1))).T,alpha)
#p[0,:,:]=pB(z[0,:]*np.ones((N,1)), (z[0,:]*np.ones((N,1))).T)
mx.showdata(p[0,:,:])
l[0,:,:]=initial_l

lxp=l[0,:,:]*p[0,:,:]
a[0,:,:] = lxp/(lxp-np.diag(np.diag(lxp))).sum(1)[:,np.newaxis]*np.ones((1,N))
#print(a[0,:,:].sum(1).round()) # all probabilities sum up to one


zdiffs = (z[0,:]*np.ones((N,1))).T - z[0,:]*np.ones((N,1))
if mechanism_type == 1:
    M[0,:,:]=xi_d*a[0,:,:]*zdiffs.T
    for t in range(1,ntimesteps):
        z[t,:] = z[t-1,:] + phi*(S[t-1,:]+M[t-1,:,:].sum(1))
        zdiffs = (z[t,:]*np.ones((N,1))).T - z[t,:]*np.ones((N,1))
        S[t,:] = xi_S*(theta-z[t,:])
        p[t,:,:]=pM(zdiffs,alpha)
        lxp=l[0,:,:]*p[t,:,:]
        a[t,:,:] = lxp/(lxp-np.diag(np.diag(lxp))).sum(1)[:,np.newaxis]*np.ones((1,N))
        
        M[t,:,:]=xi_d*a[t,:,:]*zdiffs.T
       
    mx.showlist((S+M.sum(2))[:500]) # all differences converge to zero when the system reaches a fixed point

mx.showlist(z)

#%% DIST SIMULATION
# theta=np.random.rand(N)*np.diff(ps)+ps[0] # values favoured by env. selection

mutual_effs = initial_l.copy()
allowed_links = initial_l.copy()

ntimesteps = 90
alpha=0.01
xi_S=0.5 # level of environmental selection (from 0 to 1).
xi_d=1-xi_S # level of selection imposed by resource species (from 0 to 1).
turnover=1
v = np.zeros((ntimesteps+1, N, nstates))
temp_thetanorm = (theta-ps[0])/(ps[1]-ps[0])
for species_id in range(N):
    v[0,species_id] = [bindist(n,i,temp_thetanorm[species_id]) for i in range(nstates)]

thetadiff=np.outer(np.ones(N),states)-np.outer(theta,np.ones(nstates))

mx.showlist(v[0].T)
#mx.showlist(v[0,2])
#temp_thetanorm[2]


def f(x):
    return (x-500)/26


p = np.zeros((ntimesteps+1, N, nstates))
l = np.zeros((ntimesteps+1, N, nstates)) # fitness landscape
for t in range(1,ntimesteps+1):
    for species_id in range(N):
        p[t-1,species_id]=convpM(v[t-1,species_id],nstates,alpha)
    l[t-1] = (mutual_effs @ p[t-1]) * xi_d + xi_S*pM(thetadiff,alpha=alpha)
    # l[t-1] = np.outer(np.ones(N),f(states))
    w = v[t-1]*l[t-1]
    for species_id in range(N):
        newgen= w[species_id] @ h @ w[species_id] / w[species_id].sum()**2
        v[t,species_id] = v[t-1,species_id]*(1-turnover) + newgen*turnover
        # v[t] = mut @ v[t]
        
    print(t)
#%%
mx.showlist(l[-2].T)
avgseries = (v*states*nstates).mean(2)
mx.showlist(avgseries[:20])
mx.showdata(v[:,1].T)
mx.showdata(l[:,1].T)

mx.showlist(v[0].T)
mx.showlist(v[-1].T)
mx.showlist(l[0,0].T)


'''
mx.showlist(l[0].T)
mx.showdata(l[0])

mx.showlist(v[1].T)
mx.showdata(pM(thetadiff,alpha=alpha))
mx.showdata(v[:,0])
'''
#%%
import matplotlib.animation as animation
matplotlib.use('TkAgg') # or 'Qt5Agg'

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
div = make_axes_locatable(ax)
#cax = div.append_axes('right', '5%', '5%')
def frame(t):
    ax.clear()
    ax.plot(v[t].T)
    ax.set_title('t = {} of {}'.format(t, v.shape[0]))
    return ax

ani = animation.FuncAnimation(fig, frame, frames=v.shape[0], blit=False)
# Save the animation as a GIF file
ani.save('animation.gif')

#%%
'''
t=2
intensities=np.array([convpM(v[t,species_id],nstates,alpha) for species_id in range(N)])

mx.showlist(intensities[0])

k1=(allowed_links[...,np.newaxis] @ v[t,:,np.newaxis,:])
k=(allowed_links[...,np.newaxis] @ intensities[:,np.newaxis,:])
mx.showdata(k[8,:,:]) # efectos de 8 sobre los demas
mx.showdata(k[:,8,:]) # efectos de los demas sobre 8
mx.showdata(allowed_links)

e = k * np.swapaxes(k1,0,1)


mx.showdata(k1[2,:,:])
mx.showdata(e[2,:,:])
mx.showdata(e[:,2,:])

e[2,:,:].sum(1)
e[:,2,:].sum(1)

mx.showdata(e.sum(2))
'''
#%%
adj_timeseries = []
for t in range(ntimesteps):
    print(t)
    intensities=np.array([convpM(v[t,species_id],nstates,alpha) for species_id in range(N)])
    k1=(allowed_links[...,np.newaxis] @ v[t,:,np.newaxis,:])
    k=(allowed_links[...,np.newaxis] @ intensities[:,np.newaxis,:])
    e = k * np.swapaxes(k1,0,1)
    adj_timeseries.append(e.sum(2))

vmax= np.max(adj_timeseries)
#%%
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
div = make_axes_locatable(ax)
def frame(t):
    ax.clear()
    ax.imshow(adj_timeseries[t],cmap='afmhot',vmax=vmax,vmin=0) # afmhot
    ax.set_title('t = {}'.format(t))
    return ax

ani = animation.FuncAnimation(fig, frame, frames=ntimesteps, blit=False)
ani.save('adj.gif')
#%%

pos = nx.layout.kamada_kawai_layout(nx.from_numpy_array(initial_l))
G=nx.from_numpy_array(adj_timeseries[t])
nx.draw(G, pos=pos)

linewidths = list(3*np.array(list(nx.get_edge_attributes(G, 'weight').values())))
nx.draw(G, pos=pos, width=linewidths)

mx.showdata(allowed_links)
#%%
pos = nx.layout.kamada_kawai_layout(nx.from_numpy_array(initial_l))
fig = plt.figure(figsize=(16,12)); ax = fig.add_subplot(111)
div = make_axes_locatable(ax)
def frame(t):
    print(t)
    global pos
    ax.clear()
    G=nx.from_numpy_array(adj_timeseries[t])
    pos=nx.layout.fruchterman_reingold_layout(G, weight='weight', pos=pos,threshold=1e-8,iterations=20)
    linewidths = list(3*np.array(list(nx.get_edge_attributes(G, 'weight').values())))
    nx.draw_networkx(G,ax=ax, pos=pos, width=linewidths, edge_color=linewidths, edge_cmap=plt.cm.turbo)
    ax.set_title('t = {}'.format(t))
    return ax

ani = animation.FuncAnimation(fig, frame, frames=ntimesteps, interval=50, blit=False)
ani.save('net.gif')



#%%


#%%
pos = nx.layout.kamada_kawai_layout(nx.from_numpy_array(initial_l))
fig = plt.figure(figsize=(16,16)); 
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(212)
div = make_axes_locatable(ax)
def frame(t):
    print(t)
    global pos
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    ax1.imshow(adj_timeseries[t],cmap='afmhot',vmax=vmax,vmin=0) # afmhot
    
    G=nx.from_numpy_array(adj_timeseries[t])
    linewidths = list(3*np.array(list(nx.get_edge_attributes(G, 'weight').values())))
    # pos=nx.layout.kamada_kawai_layout(G, weight='weight', pos=pos)
    nx.draw_networkx(G,ax=ax2, pos=pos, width=linewidths, edge_color=linewidths, edge_cmap=plt.cm.turbo)
    
    ax3.plot(v[t].T)
    return ax1,ax2,ax3

ani = animation.FuncAnimation(fig, frame, frames=ntimesteps, interval=50, blit=False)
ani.save('complete.gif')







