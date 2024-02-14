#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:32:45 2024

@author: miki, project w leo
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
mx.graphictools.inline_backend(False)
mx.graphictools.inline_backend(True)
#%%
import numpy as np
I=np.newaxis
def ruHyperSphere(N,nsim=20000):
    r=np.random.normal(size=(nsim,N))
    r/=np.c_[(r**2).sum(1)**(1/2)]
    return r

# v=rUHyperSphere(200,200000)
# print(np.all(np.isclose((v**2).sum(1)**(1/2),1)))

def calculate_Feasibility_Space(A,nsim):
    assert(A.shape[0]==A.shape[1])
    N=A.shape[0]
    A_1=np.linalg.inv(A)
    rl=ruHyperSphere(N,nsim= nsim)
    res=-A_1[I,...]@rl.T[I,...]
    res=res.squeeze()
    omega = ((res>0).sum(0)==N).sum()/nr # hypervolume
    return omega
#%%
                            
N = 3
nr=5000
rl=ruHyperSphere(N,nsim= nr)
#%%
mx.graphictools.inline_backend(True)
c=0.9
Q = mx.symmetric_connected_adjacency(N,c,ntries=1000)
# Q=sp_nets[3]
#%%################
Q= Q/np.c_[Q.sum(1)]*0.1
A=Q.copy()
np.fill_diagonal(A, -1)
T=np.linalg.inv(np.identity(N) - A)
A_1=np.linalg.inv(A)
sd(Q,colorbar=True)
sd(A_1,colorbar=True)
#%%
calculate_Feasibility_Space(A,nr)
#%%
A_1=np.linalg.inv(A)
Nstar=(-A_1[I,...]@rl.T[I,...]).squeeze()
boolFeasib = ((Nstar>0).sum(0)==N)
omega = boolFeasib.sum()/nr # hypervolume
print(omega)
#%%
i=1
rl_test=rl.copy()

rl_test = np.delete(rl_test, i, axis=1)  # Remove column i
rl_mean = rl_test.mean(1)
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(rl_mean,
                    rl[:,i],
                    s=0.1,
                    c=boolFeasib,
                    cmap='bwr')
ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y") 
# plt.legend(handles=[scatter1, scatter2])
plt.colorbar(scatter1)
plt.title(f'feasibility volume = {omega}')
plt.show()
    
#%%
IN = np.identity(N)
B = IN + Q #just direc effects
F=-A_1-B   #just indirect effects

NstarB=(B[I,...]@rl.T[I,...]).squeeze()
boolFeasib_B = ((NstarB>0).sum(0)==N)

NstarF=(F[I,...]@rl.T[I,...]).squeeze()
boolFeasib_F = ((NstarF>0).sum(0)==N)
#%%
offd = A_1.copy()
offd=-offd
diag=np.diag(offd)+0.
np.fill_diagonal(offd, 0)
rowsum=offd.sum(1)
slopes=-rowsum/diag

#%%



i=1
rl_test=rl.copy()

categories = boolFeasib*1+boolFeasib_B*2+boolFeasib_F*4 #more categories combined with the powers of 2
colors = (mx.graphictools.get_colors(2**3)*255).astype("int")
hexcolors = np.array(mx.graphictools.rgb2hex(colors))
c=hexcolors[categories]
print(np.unique(categories))

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
ax.scatter(np.arange(2**3),
           np.repeat(0, 2**3),
           s=1000,
           c=hexcolors)
ax.set_xlabel(r"category", fontsize=16)
ax.set_ylabel(r"y") 
plt.show()

rl_test=rl.copy()

rl_test = np.delete(rl_test, i, axis=1)  # Remove column i
rl_mean = rl_test.mean(1)
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(rl_mean,
                    rl[:,i],
                    s=1,
                    c=c)
ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y") 
# plt.legend(handles=[scatter1, scatter2])

plt.title(f'feasibility volume = {omega}')
plt.show()


omega_B = boolFeasib_B.sum()/nr # hypervolume B
omega_F = boolFeasib_F.sum()/nr # hypervolume B
print(omega,omega_B,omega_F)




#%%
mx.graphictools.inline_backend(False)

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
scatter1=ax.scatter(rl[:,0],
                    rl[:,1],
                    rl[:,2],
                    s=0.8,
                    c=c)
ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y") 
# plt.legend(handles=[scatter1, scatter2])
#plt.colorbar(scatter1)
plt.show()
#%%


mx.graphictools.inline_backend(False)

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
scatter1=ax.scatter(rl[:,0],
                    rl[:,1],
                    rl[:,2],
                    s=2,
                    c=c)
scatter2=ax.scatter(rl[:,0],
                    rl[:,1],
                    (rl[:,0]+rl[:,1])/2,
                    s=0.1)
# scatter3=ax.scatter(rl[:,0],
#                     rl[:,1],
#                     rl[:,2]/((rl[:,0]+rl[:,1])/2),
#                     s=5)
ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y") 
# plt.legend(handles=[scatter1, scatter2])
#plt.colorbar(scatter1)

ax.set_proj_type('persp', focal_length=0.2)  # FOV = 157.4 deg
ax.set_title("'persp'\nfocal_length = 0.2", fontsize=10)
plt.show()
#%%
mx.graphictools.inline_backend(True)
