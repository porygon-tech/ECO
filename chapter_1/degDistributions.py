#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:34:24 2023

@author: ubuntu
"""
#%% imports 
import sys
sys.path.insert(0, "./lib")
import evo
import matriX as mx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from  matriX import showdata as sd
mx.graphictools.inline_backend(True)
#%%
N = 50
c=0.2

# 1. ERDOS RENYI
Ab = nx.adjacency_matrix(nx.fast_gnp_random_graph(N,c)).todense()

# 2. BARBELL
# ballsize = int(N/3)
# Ab = nx.adjacency_matrix(nx.barbell_graph(ballsize,N - ballsize*2)).todense()
# Ab = mx.swaplinks(Ab, int(N/2), connected=True) # link swapping in barbell graphs allows to create semi-random bimodular graphs

# 2. NESTED
# Ab = mx.nullmodels.nestedRand(N,2*N)

#np.fill_diagonal(Ab,1)
# A = np.random.exponential(scale=.01, size=(N,N)) * Ab 
# A = np.random.lognormal(mean=0.5,sigma=1, size=(N,N)) * Ab / N
A = np.random.uniform(size=(N,N))*0.05 * Ab 
sd(A,'viridis',colorbar=True)
print(A.sum(0))
#%%
G=nx.from_numpy_array(A)
pos = nx.layout.kamada_kawai_layout(G)
node_weights=np.array(list(dict(G.degree( weight='weight')).values()))*500
linkwidth = np.array(list(nx.get_edge_attributes(G, 'weight').values()))*10
nx.draw(G,pos=pos,
        width=linkwidth,
        node_size=node_weights)
plt.show()

#%%
T=np.linalg.inv(np.identity(N) - A)
sd(T,colorbar=True)
#%%
T= np.identity(N)
for i in range(1,100):
    T+=np.linalg.matrix_power(A, i)
sd(T,colorbar=True)
#%%
mVec = np.diag(T)
J = T.copy()
np.fill_diagonal(J,0)
#%%
data=J.flatten()
data_A=A.flatten()
plt.hist(data,   bins=np.linspace(min(data), max(data),30), edgecolor='black', rwidth=0.8, histtype='step')
plt.hist(data_A, bins=np.linspace(min(data), max(data),30), edgecolor='red',   rwidth=0.8, histtype='step')
plt.xscale('log')
plt.yscale('log')
plt.show()
#%%
plt.plot(np.arange(len(data)),np.sort(data)    [::-1],color='black',linewidth=5)
plt.plot(np.arange(len(data_A)),np.sort(data_A)[::-1],color='red')
plt.plot((len(data)*c,len(data)*c),(0,max(data)),color='blue',linewidth=0.5)
plt.xscale('log')
plt.yscale('log')
plt.show()
#%%
data=J.flatten()
data_A=A.flatten()
plt.hist(data,   bins=np.linspace(min(data), max(data),30), edgecolor='black', rwidth=0.8, histtype='step')
plt.hist(data_A, bins=np.linspace(min(data), max(data),30), edgecolor='red',   rwidth=0.8, histtype='step')
plt.show()
#%%
plt.plot(np.arange(len(data)),np.sort(data)    [::-1],color='black',linewidth=5)
plt.plot(np.arange(len(data_A)),np.sort(data_A)[::-1],color='red')
plt.plot((len(data)*c,len(data)*c),(0,max(data)),color='blue',linewidth=0.5)
plt.show()
#%%
asum = A.sum(0)
jsum = J.sum(0) 
j_zero_sum = (J*(A==0)).sum(0)
j_dir_sum = (J*(A!=0)).sum(0)
#%%
plt.hist(jsum,   bins=np.linspace(0,2,30), edgecolor='black', rwidth=0.8, histtype='step',density=True)
plt.hist(asum, bins=np.linspace(0,2,30), edgecolor='red',   rwidth=0.8, histtype='step',density=True)
plt.hist(j_zero_sum, bins=np.linspace(0,2,30), edgecolor='blue',   rwidth=0.8, histtype='step',density=True) #oscar
plt.hist(j_dir_sum, bins=np.linspace(0,2,30), edgecolor='green',   rwidth=0.8, histtype='step',density=True) #jonas
plt.show()

#%%
plt.plot(np.arange(len(jsum)),np.sort(jsum)    [::-1],color='black',linewidth=5)
plt.plot(np.arange(len(asum)),np.sort(asum)[::-1],color='red')
plt.plot(np.arange(len(j_dir_sum)),np.sort(j_dir_sum)    [::-1],color='green',linewidth=5)
plt.plot(np.arange(len(j_zero_sum)),np.sort(j_zero_sum)    [::-1],color='blue',linewidth=2)
# plt.plot((len(data)*c,len(data)*c),(0,max(data)),color='blue',linewidth=0.5)
plt.show()

#%%
lamb_A = np.linalg.eigvals(Ab)
np.sum(np.real(lamb_A)**1)
pathlen=np.arange(30)
nloops = [int(np.sum(np.real(lamb_A)**pl)) for pl in pathlen]
nloopseig = [np.real(lamb_A)**pl for pl in pathlen]

plt.plot(pathlen,nloops,color='black',linewidth=5)
plt.plot(pathlen,nloopseig,color='red',linewidth=0.5)

# plt.xscale('log')
plt.yscale('log')
plt.show()


#%%
#np.sum(Ab2)/(N*N-N)
Ab1 = mx.nullmodels.nestedRand(N,0)
Ab2 = mx.nullmodels.nestedRand(N,1/10*N*N)
c=np.sum(Ab1)/(N*N-N)
Ab3 = nx.adjacency_matrix(nx.fast_gnp_random_graph(N,c)).todense()

lamb_Ab1 = np.linalg.eigvals(Ab1)
lamb_Ab2 = np.linalg.eigvals(Ab2)
lamb_Ab3 = np.linalg.eigvals(Ab3)

pathlen=np.arange(30)
nloops1 = [int(np.sum(np.real(lamb_Ab1)**pl)) for pl in pathlen]
nloops2 = [int(np.sum(np.real(lamb_Ab2)**pl)) for pl in pathlen]
nloops3 = [int(np.sum(np.real(lamb_Ab3)**pl)) for pl in pathlen]

plt.plot(pathlen,nloops1,color='blue',linewidth=2)
plt.plot(pathlen,nloops2,color='red',linewidth=2)
plt.plot(pathlen,nloops3,color='green',linewidth=2)

plt.plot(pathlen,10**(pathlen),linewidth=2)


# plt.xscale('log')
plt.yscale('log')
plt.show()
#%%
'''
pathlen=np.arange(7)
cm = plt.cm.get_cmap("gnuplot")
nloopseig = [np.real(lamb_Ab1)**pl for pl in pathlen]
a = plt.hist(nloopseig,   bins=np.linspace(min(data), max(data),10), rwidth=0.1, histtype='step')
x=a[1][:-1]
y=a[0]
plt.show()

for pl in pathlen:
    color = cm(pl/pathlen[-1])
    plt.plot(x, y[pl],color=color)
    
plt.show()
'''


#%%
sd(Ab)
sd(np.linalg.matrix_power(Ab, 2),colorbar=True)

lamb_Ab = np.linalg.eigvals(Ab)

k=3
ncl = np.diag(np.linalg.matrix_power(Ab, k))
total = 1/k*np.sum(np.real(lamb_Ab)**k)
total = 1/k*np.trace(np.linalg.matrix_power(Ab, 3))



#%%
import numpy as np

# Assuming Ab is the adjacency matrix of your graph
N = 30
c=0.5

# 1. ERDOS RENYI
Ab = nx.adjacency_matrix(nx.fast_gnp_random_graph(N,c)).todense()
sd(Ab)

# Calculate the eigenvalues of Ab
lamb_Ab = np.linalg.eigvals(Ab)

# Set the power of the closed walks, k
k = 4

# Calculate the diagonal elements of Ab^k, representing closed walks of length k at each node
ncl = np.diag(np.linalg.matrix_power(Ab, k))

# Calculate the total number of closed walks of length k in the graph using the trace
total_trace = 1/k * np.trace(np.linalg.matrix_power(Ab, k))

# Alternatively, you can use the sum of eigenvalues raised to the power of k
total_eigenvalues = 1/k * np.sum(np.real(lamb_Ab) ** k)

# Print the results
print("Diagonal elements (closed walks at each node):", ncl)
print("Total number of closed walks using trace:", total_trace)
print("Total number of closed walks using eigenvalues:", total_eigenvalues)

sum(ncl)/k
#%%
N = 60
c=0.2
k = 5
Ab = nx.adjacency_matrix(nx.fast_gnp_random_graph(N,c)).todense()

np.trace(np.linalg.matrix_power(Ab, k))
a=np.diag(np.linalg.matrix_power(Ab, k))
b=np.real(np.linalg.eigvals(Ab)) ** k
print(a.mean(), b.mean())
#%%
plt.plot(np.arange(N),np.sort(a)[::-1],color='black')
plt.plot(np.arange(N),np.sort(b)[::-1],color='red')
plt.yscale('log')
plt.show()
minv=np.min([a,b])
maxv=np.max([a,b])/4 # !!!!!


plt.hist(a, bins=np.linspace(minv, maxv,100), edgecolor='black', rwidth=0.8, histtype='stepfilled')
plt.hist(b, bins=np.linspace(minv, maxv,100), edgecolor='red',   rwidth=0.8, histtype='stepfilled',alpha=0.5)
# plt.xscale('log')
plt.yscale('log')
plt.show()


#%%
import scipy
A=Ab/N

sd(np.exp(Ab))
sd(Ab)
eA = scipy.linalg.expm(Ab)
A=Ab.copy().astype('float64')
for k in range(1,5):
    tmp=np.linalg.matrix_power(A, k) / scipy.special.factorial(k)
    A+= tmp
    sd(A)


sd(A)
sd(eA)

sd(np.round(scipy.linalg.logm(eA),10))
#%%

I = np.identity(N)

Al=A.copy().astype('complex128')
# for k in range(1,500):
#     tmp=scipy.linalg.logm(I + A/k)
#     Al+= tmp
#     # sd(A)

sd(scipy.linalg.logm(Al))
Al= np.round(Al,2)

T=np.linalg.inv(np.identity(N) - A)

sd(Al)
sd(A)
sd(T)
sd(T-I)
