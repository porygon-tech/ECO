#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:56:58 2024

@author: ubuntu
"""



from scipy.special import comb  
def bindist(n,k,p=0.5):
    return comb(n,k)*p**k*(1-p)**(n-k)



n=120
k=5
cp = 1
p=0
while p < k:
    cp*=(n-p)/(k-p)
    p+=1
cp
comb(n,k)
#%%

n=120
k=73
x=31
comb(n-x,k-x)

cp = 1
p=0
while p < x:
    cp*=(k-p)/(n-p)
    p+=1
comb(n,k)*cp

#%% imports 
import sys
sys.path.insert(0, "./lib")
import evo
import matriX as mx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from  matriX import showdata as sd
mx.graphictools.inline_backend(False)
mx.graphictools.inline_backend(True)
#%%
import numpy as np

def hyperspherical_to_cartesian(r, angles):
    """
    Convert hyperspherical coordinates to Cartesian coordinates in n dimensions.

    Parameters:
    - r: float, radial coordinate
    - angles: list or array of floats, n-1 angular coordinates in radians

    Returns:
    - Cartesian coordinates as a NumPy array
    """
    n = len(angles) + 1  # Number of dimensions
    cartesian_coords = [r * np.prod(np.sin(angles[:i])) * np.cos(angles[i]) for i in range(n-1)]
    cartesian_coords.append(r * np.prod(np.sin(angles[:n-1])))
    return np.array(cartesian_coords)

# Example usage:
hyperspherical_coords = (2.0, [np.pi/4, np.pi/3])  # Example hyperspherical coordinates
cartesian_coords = hyperspherical_to_cartesian(*hyperspherical_coords)
print("Hyperspherical Coordinates:", hyperspherical_coords)
print("Cartesian Coordinates:", cartesian_coords)

#%%
N = 3
c=0.2

A = nx.adjacency_matrix(nx.fast_gnp_random_graph(N,c)).todense()
T=np.linalg.inv(np.identity(N) - A)
A_1=-np.linalg.inv(A)
sd(A)

nsim = 10000
vectors = []
for simID in range(nsim):
    r=np.random.rand(N)-0.5
    r/=sum(r**2)**(1/2) #normalize to be contained in the unit sphere. Does it keep angles untouched?
    # r = hyperspherical_to_cartesian(np.random.randint(1,3),np.random.rand(N-1)*np.pi)
    vectors.append(r)
vectors=np.array(vectors)


#%%
nsim = 10000
vectors = []
for simID in range(nsim):
    v_ang = np.random.rand(N-1)*2*np.pi
    sins = np.sin(v_ang)
    sinM = np.repeat(np.c_[sins],N-1,axis=1)
    sinM[np.where(np.triu(sinM,1))]=1
    sp=sinM.prod(0)
    sp=np.concatenate((sp,[1]))
    sp=np.flip(sp)
    cp=np.concatenate((np.cos(v_ang),[1]))
    r = cp*sp
    r=np.array([np.cos(v_ang[0]),
                np.sin(v_ang[0])*np.cos(v_ang[1]),
                np.sin(v_ang[0])*np.sin(v_ang[1])])
    vectors.append(r)

vectors=np.array(vectors)
#%%
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
scatter1=ax.scatter(vectors[:,2],
                    vectors[:,1],
                    vectors[:,0],
                    s=0.1)
ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y") 
# plt.legend(handles=[scatter1, scatter2])
plt.colorbar(scatter1)
plt.show()


#%%
nsim = 20000
vectors = []
for simID in range(nsim):
    r=np.random.rand(N)*2-1
    vectors.append(r)
vectors=np.array(vectors)
v = vectors[[sum(r**2)**(1/2)<1  for r in vectors]]
v /= np.c_[(v**2).sum(1)**(1/2)]

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
scatter1=ax.scatter(v[:,2],
                    v[:,1],
                    v[:,0],
                    s=0.1)
ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y") 
# plt.legend(handles=[scatter1, scatter2])
plt.colorbar(scatter1)
plt.show()
len(v)
#%% force number of dots
nsim = 20000
vectors = []
simID=0
while simID < nsim:
    r=np.random.rand(N)*2-1
    if sum(r**2)**(1/2)<1: 
        vectors.append(r)
        simID+=1
vectors=np.array(vectors)
vectors /= np.c_[(vectors**2).sum(1)**(1/2)]

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
scatter1=ax.scatter(vectors[:,2],
                    vectors[:,1],
                    vectors[:,0],
                    s=0.1)
ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y") 
# plt.legend(handles=[scatter1, scatter2])
plt.colorbar(scatter1)
plt.show()
len(vectors)


#%% improve this to adapt to number of cores
def randomUniformHyperSphere(N,nsim= 20000):
    '''
    
    Parameters
    ----------
    N : int
        Number of dimensions.
    nsim : int, optional
        Number of points generated. The default is 20000.
    Returns
    -------
    v : array of vectors
        Each element is a vector for a random point, uniformly distributed in the unit N-dimensional hypersphere.

    '''
    vectors = []
    simID=0
    while simID < nsim:
        r=np.random.rand(N)*2-1
        modulus = sum(r**2)**(1/2)
        if modulus<1:  #crops the hypersphere inscribed in the tesseract
            r /= modulus # remove this line to adapt the code for a hyperball instead of a hypersphere
            vectors.append(r)
            simID+=1
            print (f'{simID/nsim*100}%')
    v=np.array(vectors)
    # v /= np.c_[(vectors**2).sum(1)**(1/2)]

    return v

def split(string, n):
    return [string[i:i+n] for i in range(0, len(string), n)]

def task(cola, n=100):
    
    return randomUniformHyperSphere(n,cola)

def antiHoundsOfTindalos(N,nsim= 20000):
    #The full artillery. Much safer than messing around with angles

    
    nprocessors = multiprocessing.cpu_count()
    #simulations = np.empty(nsim, dtype=object)
    #colas = split(np.arange(nsim), int(np.ceil(nsim/nprocessors)))
    colas=[int(np.ceil(nsim/nprocessors))]*nprocessors
    print('RUNNING SIMULATION BATCH. SPAWNING ' + str(len(colas)) + ' PROCESSES ('+ str(colas[0])+' tasks each)')
    #nsimulations // nprocessors +1 

    pool = multiprocessing.Pool(processes=nprocessors)
    results = pool.map(task, colas) # this is where magic happens :)
    results = np.array(results).flatten().tolist()
    
    return results
    

v = antiHoundsOfTindalos(5,nsim= 2000)
v=randomUniformHyperSphere(5,2000)
#%%

v=randomUniformHyperSphere(3,nsim= 2000)
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
scatter1=ax.scatter(v[:,2],
                    v[:,1],
                    v[:,0],
                    s=0.1)
ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y") 
# plt.legend(handles=[scatter1, scatter2])
plt.colorbar(scatter1)
plt.show()
#%%
                             
N = 15
c=0.3
nr=2000
rl=randomUniformHyperSphere(N,nsim= nr)
#%%
A = mx.symmetric_connected_adjacency(N,c,ntries=1000)
A=A/np.c_[A.sum(1)]*0.35
np.fill_diagonal(A, -1)
T=np.linalg.inv(np.identity(N) - A)
A_1=np.linalg.inv(A)
sd(A,colorbar=True)
sd(A_1)

#%%
f0=[]
f1=[]
for r in rl:
    #r*=6
    N_star=-A_1@r
    if np.all(N_star > 0):
        f1.append(r)
    else:
        f0.append(r)

print(len(f1),len(f0))
print(f'feasibility: {len(f1)/nr}')

