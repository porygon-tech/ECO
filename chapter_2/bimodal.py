#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 08:07:40 2023

@author: ubuntu
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from os import chdir, listdir, environ,popen
from pathlib import Path
import pickle5
import bz2
chdir(environ['HOME'] + '/LAB/ECO') #this line is for Spyder IDE only
root = Path(".")
obj_path = root / 'data/obj'
img_path = root / 'gallery/timeseries'
dataPath = root / 'data/dataBase'
#%% OWN LIBS
import sys
sys.path.insert(0, "./lib")
import evo
import matriX as mx
#%% developer tools

#plt.switch_backend('Qt5Agg')
import contextlib
import matplotlib

def spyder_backend(io=True):
    if io:
        gui = 'module://matplotlib_inline.backend_inline'
    else: 
        gui = 'qt5agg'
    with contextlib.suppress(ValueError):
        matplotlib.use(gui, force=True)
    globals()['plt'] = matplotlib.pyplot

from matriX import showdata as sd
#%% HYPERPARAMETERS
N=3 # number of species. Has an important relation with the mutualRange and c parameters
nloci=50 # number of loci
ps=(20,30) # phenotypic space. Has an important relation with the alpha parameter


# tensor object load
filename='oc_tensor_' + str(nloci) + '.obj'
with bz2.BZ2File(obj_path / filename, 'rb') as f:
	h = pickle5.load(f)

#%%
#https://github.com/porygon-tech/Graph_To_Adjacency
spyder_backend(True)
txt= '''
0 1 0
1 0 1
0 1 0
'''

#%% MODEL PARAMETERS

A = np.genfromtxt(txt.splitlines()); N = A.shape[0]
#g1,g2 = g = np.array([-0.02,0.02]) # payoffs for symmetric games
#A_e = np.random.choice((g1,g2),(N,N))*A
A_e = A*0.05
A_e[0,1] = A_e[2,1] = 0.0
GA = nx.from_numpy_array(A_e)
fig, ax = plt.subplots()
pos=nx.layout.spring_layout(GA)

linewidths = list(nx.get_edge_attributes(GA, 'weight').values())

nx.draw_networkx(GA,
                 pos=pos,
                 ax=ax,
                 #node_color='orange',
                 #edge_color='lightgray'
                 width=1, 
                 edge_color=linewidths, 
                 edge_cmap=plt.cm.bwr
                 )

ax.axis('off')
#fig.set_facecolor('#002233')
plt.title('net',color='white')
plt.show()

#%% initialization 


K=1000
dev=[0.2, 0.501, 0.8]
theta=dev*np.diff(ps)+ps[0]
v0=evo.initialize_bin_explicit(N,nloci,dev); # set to start at their environmental optima
#v0=evo.initialize_bin_explicit(N,nloci,np.random.rand(N)); # set to start at random location
mx.showlist(v0.T)
xi_d=1-xi_S
m=np.c_[.0001,0.5,.0001].T
# m=np.repeat([[.9]],3, axis=0)
#%% run
simulation = evo.simulator(
    find_fixedpoints=False,
    ntimesteps=700,  # number of generations simulated
    h=h,
    ps=ps,
    K=K, # carrying capacity  
    D0=500,
    theta=theta,
    m=m,
    alpha=0.2, # strength of the trait matching mechanism. Positive real value. Lower values = greater promiscuity
    v0=v0,
    mutual_effs=A_e,
    a=0,
    d=0)

simulation.run(tolD=2,tolZ=1e-8)
simulation_dict=simulation.__dict__
#mx.showdata(simulation_dict['v'][:400,1].T)
mx.showdata(mx.graphictools.resize_image(simulation_dict['v'][:,1].T, (200,400)))
#%%
mx.showdata(mx.graphictools.resize_image(simulation_dict['l'][:,1].T, (200,400)),color='viridis')
#%%
mx.showlist(simulation_dict['D'])
#%%
mx.showlist(simulation_dict['dist_avgs'])
#%% plot
mx.showdata(mx.graphictools.resize_image(simulation_dict['v'][:,0].T, (200,400)),color='viridis')
mx.showdata(mx.graphictools.resize_image(simulation_dict['v'][:,1].T, (200,400)),color='viridis')
mx.showdata(mx.graphictools.resize_image(simulation_dict['v'][:,2].T, (200,400)),color='viridis')

#%% plot
mx.blendmat(
simulation_dict['v'][:,0].T,
simulation_dict['v'][:,1].T,
simulation_dict['v'][:,2].T,

saturation = 6,
additive=True)


#%%
a=0.2
nstates=nloci+1
states = np.linspace(ps[0],ps[1], nstates)
statesdiff=np.outer(np.ones(nstates),states)-np.outer(states,np.ones(nstates))
assortMat = evo.interactors.pM(statesdiff,alpha=abs(a))
if a<0:
    assortMat = 1 - assortMat
    
plt.imshow(assortMat, interpolation='none', cmap='plasma',vmax=1,vmin=0); plt.show()
