#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:01:40 2023

@author: roman
"""


import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root
import matplotlib.pyplot as plt

from os import chdir, listdir
chdir('/home/roman/LAB/ECO') #this line is for Spyder IDE only
import sys
sys.path.insert(0, "./lib")
import matriX as mx

#%%
# Define the system of differential equations
def model(y, t, h, 
          n_P,
          n_A,
          alpha_P,beta_P,gamma_P,
          alpha_A,beta_A,gamma_A):
    # y is a vector of the state variables
    # t is the time variable
    
    # Define the model hardcoded structure
    p=y[:n_P ,np.newaxis]
    a=y[-n_A:,np.newaxis]
    dydt = np.zeros(n_P+n_A)
    
    gP_temp = np.sum(gamma_P*a.T)
    gA_temp = np.sum(gamma_A*p.T)
    dydt[:n_P ] = (p*(alpha_P - np.sum(beta_P*p.T) + gP_temp/(1+h*gP_temp))).flatten()
    dydt[-n_A:] = (a*(alpha_A - np.sum(beta_A*a.T) + gA_temp/(1+h*gA_temp))).flatten()
    return dydt

#%%
# Define the model structure

n_P=15
n_A=6
'''
alpha_P = np.random.rand(n_P)[:,np.newaxis]*2
beta_P  = np.random.rand(n_P,n_P)*0.1
gamma_P = np.random.rand(n_P,n_A)*5

alpha_A = np.random.rand(n_A)[:,np.newaxis]*4
beta_A  = np.random.rand(n_A,n_A)*0.1
gamma_A = np.random.rand(n_A,n_P)*5
'''
prob=0.4 #expected connectance
y = np.random.choice((0,1),(n_A,n_P), p=(1-prob, prob))
while np.any(y.sum(0)==0) or np.any(y.sum(1)==0):
    y = np.random.choice((0,1),(n_A,n_P), p=(1-prob, prob));
plt.imshow(y)

#k is the node degree
k_A=y.sum(1)[:,np.newaxis]
k_P=y.sum(0)[:,np.newaxis]

# Define the initial conditions
p0 = np.random.randint(50,200,n_P)[:,np.newaxis].astype('float64')
a0 = np.random.randint(50,200,n_A)[:,np.newaxis].astype('float64')
x0 = np.append(p0,a0)

#%%
# Define the model parameters
h=0.8 # saturating constant of the beneficial effect of mutualisms, aka handling time. If h=0, the model becomes linear
rho     = 0.1 # interspecies competition (intraguild)
delta   = .5 # mutualistic trade off
gamma_0 = .2 # level of mutualistic strength

#alpha is the growth rate vector
np.random.seed(0)
alpha_P = np.random.normal(20,0.3,(n_P, 1))
alpha_A = np.random.normal(8,0.1,(n_A, 1))

#a is the total growth rate vector
a = np.append(alpha_P,alpha_A,0)

gamma_P = (gamma_0*y.T)/k_P**delta
gamma_A = (gamma_0*y  )/k_A**delta

#beta is the competition matrix
beta_P  = rho + np.zeros((n_P, n_P))
beta_A  = rho + np.zeros((n_A, n_A))
for i in range(n_P): beta_P[i,i]=1
for i in range(n_A): beta_A[i,i]=1

#%%
# Solve the system of differential equations
t = np.linspace(0, 150, 200)
x = odeint(model, x0, t, (h, int(n_P), int(n_A), alpha_P,beta_P,gamma_P, alpha_A,beta_A,gamma_A))
plt.plot(x[1:])

fixedpoint = root(model, x0=x[-1], args=(None, h, n_P, n_A, alpha_P,beta_P,gamma_P, alpha_A, beta_A, gamma_A))

#x[-1]
#fixedpoint['x']
np.all(np.isclose(x[-1],fixedpoint['x']))

#%%
ntries=2000
srchr=(-5,5)
fxpts=[]
for i in range(ntries): 
    deleteflag=False
    sol = root(model, x0=np.random.uniform(srchr[0],srchr[1],n_P+n_A), args=(None, h, n_P, n_A, alpha_P,beta_P,gamma_P, alpha_A, beta_A, gamma_A))['x']
    #did not check status, blindly accepting the result. If any bug appears,improve this code to check that
    for i, a in enumerate(fxpts):
        if np.all(np.isclose(a, sol)):
            deleteflag=True
        elif np.any(sol<-1e-7):
            deleteflag=True
    if not deleteflag:
        fxpts.append(sol)
fxpts=np.asarray(fxpts)

#fxpts = np.unique(np.asarray(fxpts),axis=0)
mx.showdata(fxpts,colorbar=True)
#%%
plt.scatter(fxpts[:,0],fxpts[:,-1],s=3)
fxpts[np.where(fxpts[:,0]*fxpts[:,-1]>1)]

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(fxpts.T)

pca = PCA(n_components=4)
pca.fit(scaled_data)

# Transform the data into the lower-dimensional space
pca_data = pca.transform(scaled_data)

plt.scatter(pca_data[:, 0], pca_data[:, 1],s=3)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

plt.scatter(pca_data[:, 0], pca_data[:, 2])

#%%
B = np.append(np.append(beta_P,-gamma_P, axis=1), np.append(-gamma_A, beta_A,axis=1), axis=0); mx.showdata(B,symmetry=True,colorbar=True)
#%%



np.all(np.linalg.eigvals(-B) < 0)
mx.is_Dstable(B, maxiter=3000)
mx.is_Lyapunovstable(B)





#%%
from scipy.misc import derivative as d

def undershoot_corrector(x,hardness=1):
    #this function returns nearly similar values when x is positive, 
    #and very low, positive values when x is negative.
    #hardness is how abrupt is the change when switching to negative numbers.
    return(x/(1-np.exp(-hardness*x)))

def pM (x,y, alpha=50.,peak=1.,base=0.):
    #peak is the value reached at the peak and base is the base fitness
    #return np.exp(-alpha*(x-y)**2)*mult
    #return (np.exp(-alpha*(x-y)**2)+(abs(mult)/mult-1)/2)*mult
    return np.exp(-alpha*(x-y)**2)*(peak - base) + base


def spm(s,alpha=1.,mult=1.,base=0.,hardness=1):
    #fitness for all traits in vector s
    N=len(s)
    if type(mult) == float: mult = np.ones((N,N))*mult
    if type(base) == float: base = np.ones(N)*base
    
    w=np.zeros(N)
    for i in range(N):
        others = np.delete(s, i)
        keys   = np.delete(np.arange(N), i)
        for j, sj in enumerate(others):
            w[i] += pM(s[i],sj, alpha, mult[i,keys[j]])
        w[i]+=base[i]
    #return w
    return undershoot_corrector(w,hardness)

def spmn(x, s, n, alpha=1.,mult=1.,base=0.,hardness=1):
    #fitness landscape for one of the traits n with value x, given other values s. s[n] is ignored
    N=len(s)
    if type(mult) == float: mult = np.ones((N,N))*mult
    if type(base) == float: base = np.ones(N)*base
    
    sc = s.copy()
    sc[n]=x
    return(spm(sc,alpha,mult,base,hardness)[n])

def dspmn(x, s, n, alpha=1,mult=1.,base=0.,hardness=1):
    #partial derivative of the fitness landscape in respect to the trait n with value x
    N=len(s)
    if type(mult) == float: mult = np.ones((N,N))*mult
    
    return(d(lambda x: spmn(x,s,n, alpha,mult,base,hardness),x))

def dspm(s, alpha=1.,mult=1.,base=0.,hardness=1):
    #partial derivative of the fitness of all traits
    N=len(s)
    if type(mult) == float: mult = np.ones((N,N))*mult
    if type(base) == float: base = np.ones(N)*base
    
    dwi_dsi=np.zeros(N)
    for i in range(N):
        dwi_dsi[i] = dspmn(s[i], s, i, alpha,mult,base,hardness)
    return(dwi_dsi)


def cross_spm(s,alpha=1.,mult=1.,base=0.,hardness=1):
    #fitness for all traits in vector s
    N=len(s)
    if type(mult) == float: mult = np.ones((N,N))*mult
    if type(base) == float: base = np.ones(N)*base
    
    w=np.zeros((N,N))
    for i in range(N):
        others = np.delete(s, i)
        keys   = np.delete(np.arange(N), i)
        for j, sj in enumerate(others):
            w[i,keys[j]] += pM(s[i],sj, alpha, mult[i,keys[j]])
        #w[i,:]+=base[i]
    return w



#%%
alpha=0.02
rate=1.
s=np.random.rand(5)*35
k=np.random.rand(5)*2

spm(s,alpha=alpha,mult=rate) # all fitnesses
#%%


fig = plt.figure(); ax = fig.add_subplot(111)
n=100
x=np.linspace(0,50, 500)
y=[spmn(xi,s,3,alpha,rate) for xi in x]
ax.plot(x,y)
ax.set_xlabel('phenotype value', labelpad=10)
ax.set_ylabel('fitness', labelpad=10)
plt.show()

#%%
fig = plt.figure(); ax = fig.add_subplot(111)

i=0
x=np.linspace(0,50, 500)
y=[spmn(xi,s,i,alpha,rate,base) for xi in x]
ax.plot(x,y)

x=np.linspace(0,50, 500)
y=[dspmn(xi,s,i,alpha,rate,base) for xi in x]
ax.plot(x,y,label= r"$\dfrac{\partial W_i}{ \partial s_i}$")

ax.scatter(s[i],spm(s,alpha,rate,base)[i],label= 'you are here',c='red')

ext=7
#ax.plot((s[3],s[3]+ext),(spm(s,alpha,rate)[3],(spm(s,alpha,rate)[3]+ dspm(s,alpha,rate)[3]*ext)))
ax.arrow(s[i],spm(s,alpha,rate,base)[i], ext, dspm(s,alpha,rate,base)[i]*ext,head_width=0.5,head_length=1.5)

ax.set_xlabel('phenotype value', labelpad=10)
ax.set_ylabel('fitness', labelpad=10)
ax.legend()
plt.show()

#%%

def coevo(s, t, k,alpha,rate,base):
    # y is a vector of the state variables
    # t is the time variable
    dsdt = k * dspm(s,alpha,rate,base)
    return dsdt

#%%
alpha=0.02
rate=0.8

t = np.linspace(0, 150, 200)
s0=s.copy()
x = odeint(coevo, s0, t, (k,alpha,rate))

plt.plot(t, x)


#%%

alpha=0.02
N=len(s)
rate=np.ones((N,N))
rate[0,:]=-1.901 #one of them will have trait mismatch (antagonistic interaction)
rate[:,0]=2.2 #furthermore,the other species will have a higher benefit when interacting with this one instead of the mutualism

base=np.ones(N)*0.1
base[0]=3 # this is the base fitness of species 0 when away of its exploiters

t = np.linspace(0, 1000, 200)
s0=s.copy()
x = odeint(coevo, s0, t, (k,alpha,rate,base))
plt.plot(t, x)

#%%
wt=np.array(list(map(spm, x, [alpha]*200,[mult]*200)))
plt.plot(wt)

#%%




#%% PUT EVERYTHING TOGETHER
N=n_P+n_A

rate=-B.copy()*5
s=x0.copy()
k=np.random.rand(N)*10
alpha=0.002
base=1.

t = np.linspace(0, 1000, 200)
s0=s.copy()
x = odeint(coevo, s0, t, (k,alpha,rate,base))

plt.plot(t, x[:,:n_P],c='green')
plt.plot(t, x[:,-n_A:],c='red')

wt=np.array(list(map(spm, x, [alpha]*200,[rate]*200)))
plt.plot(wt)

#%% 
import networkx as nx
def norm01(x):
    return (x-min(x))/max((x-min(x)))

adj = B-np.diag(np.repeat(1, N)) 
G = nx.from_numpy_array(adj)
nx.draw(G)



top  = np.array(G.nodes)[:n_P]
down = np.array(G.nodes)[-n_A:]
pos = nx.bipartite_layout(G,top, align='horizontal')
sorted_top  = np.array(sorted(G.degree(top),  key=lambda x: x[1], reverse=False))[:,0]
sorted_down = np.array(sorted(G.degree(down), key=lambda x: x[1], reverse=False))[:,0]
keys = np.append(sorted_top,sorted_down)
pos = dict(zip(keys, list(pos.values())))
nx.draw(G,pos, width=1, node_size=40)
plt.show()


#%% 
adj = cross_spm(x[-1],alpha,rate)
mx.showdata(adj,colorbar=True,symmetry=True)
mx.showdata(B,colorbar=True,symmetry=True)
G = nx.from_numpy_array(adj)

edges = G.edges()
weights = [G[u][v]['weight'] for u,v in edges];weights=np.array(weights)
#cm=plt.colormaps.get('magma_r')
c=plt.get_cmap('seismic')(norm01(weights))

top  = np.array(G.nodes)[:n_P]
down = np.array(G.nodes)[-n_A:]
pos = nx.bipartite_layout(G,top, align='horizontal')
sorted_top  = np.array(sorted(G.degree(top),  key=lambda x: x[1], reverse=False))[:,0]
sorted_down = np.array(sorted(G.degree(down), key=lambda x: x[1], reverse=False))[:,0]
keys = np.append(sorted_top,sorted_down)
pos = dict(zip(keys, list(pos.values())))

nx.draw(G,pos, width=1, node_size=40,edge_color=c)
plt.show()
#%% 

