#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:12:04 2023

@author: ubuntu
"""


#%% show population sizes
switch_backend('module://matplotlib_inline.backend_inline')

sort_simulations =  sorted(enumerate(simulations), key=lambda x: x[1]['_d'].mean())
filter_1 = [sim['_d'].mean() > 1 for sim in sort_simulations]
filter_2 = np.array(power_mutu) >0.1
# filtered_i = np.where(np.logical_and(filter_1,filter_2))[0]
filtered_i = np.arange(len(sort_simulations))

for i, sim in enumerate(sort_simulations):
    if i in filtered_i:
        R = np.array(list(gamma['nodes']['power_pred']  [i].values()))
        G = np.array(list(gamma['nodes']['power_mutu']  [i].values()))
        B = np.array(list(gamma['nodes']['power_comp']  [i].values()))


        rgblist = np.array([
        np.nan_to_num(mx.cNorm(mx.renormalize(R),3)*255),
        np.nan_to_num(mx.cNorm(mx.renormalize(G),3)*255),
        np.nan_to_num(mx.cNorm(mx.renormalize(B),3)*255)]).astype('int').T

        colors = ['#%02x%02x%02x' % (r,g,b) for r,g,b in rgblist]
        x = np.arange(sim['D'][:-1].shape[0])
        for i__N in range(N):
            plt.plot(x, sim['D'][:-1,i__N], c=colors[i__N])
            
            # plt.plot(x, sim['dist_avgs'][:-1,i__N], c=colors[i__N])
        plt.xlim(0,4000)
        plt.ylim(0,1000)
        plt.title(str(i)+': '+str(sim['_d'].mean()))
        plt.show()



#%% show average traits
switch_backend('module://matplotlib_inline.backend_inline')

filter_1 = [sim['_d'].mean() > 1 for sim in sort_simulations]
filter_2 = np.array(power_mutu) <0.1
# filtered_i = np.where(np.logical_and(filter_1,filter_2))[0]
filtered_i = np.arange(len(sort_simulations))

for i,sim in enumerate(sort_simulations):
    if i in filtered_i:
        R = np.array(list(gamma['nodes']['power_pred']  [i].values()))
        G = np.array(list(gamma['nodes']['power_mutu']  [i].values()))
        B = np.array(list(gamma['nodes']['power_comp']  [i].values()))


        rgblist = np.array([
        np.nan_to_num(mx.cNorm(mx.renormalize(R),3)*255),
        np.nan_to_num(mx.cNorm(mx.renormalize(G),3)*255),
        np.nan_to_num(mx.cNorm(mx.renormalize(B),3)*255)]).astype('int').T

        colors = ['#%02x%02x%02x' % (r,g,b) for r,g,b in rgblist]
        x = np.arange(sim['D'][:-1].shape[0])
        for i__N in range(N):
            plt.plot(x, sim['dist_avgs'][:-1,i__N], c=colors[i__N])
            
        plt.title(str(i)+': '+str(sim['_d'].mean()))
        plt.show()



#%%
from os import chdir, environ
chdir(environ['HOME'] + '/LAB/ECO') #this line is for Spyder IDE only
import sys
sys.path.insert(0, "./lib")
import matriX as mx
import networkx as nx
import numpy as np
from matriX import showdata as sd
#%%
#=============================================
#===== TOPOLOGIES ============================
#=============================================
# https://networkx.org/documentation/stable/reference/generators.html

# 1. ERDOS RENYI
c=0.1
# A = np.random.choice((0,1),(N,N),p=(1-c,c))
A = nx.adjacency_matrix(nx.fast_gnp_random_graph(N,c)).todense()


# 2. RANDOM NESTED (MIKI)
A = mx.nullmodels.nestedRand(N,10)

# 3. BARBELL
ballsize = 6
A = nx.adjacency_matrix(nx.barbell_graph(ballsize,N - ballsize*2)).todense()
A = mx.swaplinks(A, 5, connected=True) # link swapping in barbell graphs allows to create semi-random bimodular graphs

# 4. newman_watts_strogatz (small world)
A = nx.adjacency_matrix(nx.newman_watts_strogatz_graph(N,4,0.1)).todense()

# 5. LATTICES
A = nx.adjacency_matrix(nx.hexagonal_lattice_graph(2,5)).todense() # periodic=True

# 6. RANDOM MODULAR (MIKI)
N=25
A = mx.nullmodels.clusterchain(N,3)
A = mx.swaplinks(A, 5, connected=True)


#=============================================
#===== PAYOFFS ===============================
#=============================================
N=A.shape[0]
g1,g2 = np.array([-2,1])

# 1. UNIFORMLY CHOSEN PAYOFFS
A_e = np.random.choice((g1,g2),(N,N))*A
sd(A_e,symmetry=True)

# 2. DEGREE-BIASED PAYOFFS
sums_tmp = np.outer(A.sum(0),np.ones(N))
p=(sums_tmp+sums_tmp.T)/(2*N-1)
bool_array = np.random.rand(N,N)<p
# <p: more degree more enemies
# >p: more degree more friends
A_e = np.empty_like(bool_array, dtype=object)
A_e[ bool_array] = g1
A_e[~bool_array] = g2
A_e*=A
sd(A_e,symmetry=True)

mx.totext(A)
#%%
#=============================================
N=7
c=0.4
A = nx.adjacency_matrix(nx.fast_gnp_random_graph(N,c)).todense()
g1,g2 = np.array([-2,1])
A_e = np.random.choice((g1,g2),(N,N))*A

sd(A_e,symmetry=True)
mx.totext(A)




