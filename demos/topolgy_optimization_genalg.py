#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 18:03:33 2024

@author: ubuntu
"""



from os import chdir, listdir, environ, system, popen
from pathlib import Path
import pickle5
import bz2
chdir(environ['HOME'] + '/LAB/ECO') #this line is for Spyder IDE only
root = Path(".")
obj_path = root / 'data/obj'
img_path = Path(environ['HOME']) / 'LAB/figures'
dataPath = root / 'data/dataBase'


#%% imports 
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# from copy import deepcopy
import networkx as nx
# import pandas as pd
r = np.random.rand

#%% OWN LIBS
sys.path.insert(0, "./lib")
import opt
import matriX as mx

#%%
'''
N=23
c=0.2
A = nx.adjacency_matrix(nx.fast_gnp_random_graph(N,c)).todense()
nx.draw(nx.from_numpy_array(A))
p = opt.evolve(
    indiv = A,
    policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
    mutation= lambda x: mx.double_edge_swap(x,10), 
    ngens=30)

p = opt.evolve(
    indiv = p[0],
    policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
    mutation= lambda x: mx.double_edge_swap(x,2), 
    ngens=20)

p = opt.evolve(
    indiv = p[0],
    policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
    mutation= lambda x: mx.double_edge_swap(x,1), 
    ngens=20)

nx.draw(nx.from_numpy_array(A)); plt.show()
nx.draw(nx.from_numpy_array(p[0])); plt.show()
print(A.sum(0)[::-1] - p[0].sum(0)[::-1])

#%%
N=23
c=0.25
A = nx.adjacency_matrix(nx.fast_gnp_random_graph(N,c)).todense()
p = opt.evolve(
    indiv = A,
    policy  = lambda x: mx.nodf(x),
    mutation= lambda x: mx.swaplinks(x,10,connected=True), 
    ngens=30)
    
nx.draw(nx.from_numpy_array(A))
plt.show()
nx.draw(nx.from_numpy_array(p[0]))



mx.showdata(A)

mx.showdata(p[0])

As = sort_isomorphism(A)
Bs = sort_isomorphism(p[0])
mx.showdata(As)
mx.showdata(Bs)

def sort_isomorphism(B):
    B=B[:,np.argsort(B.sum(axis=0))[::-1]]
    B=B[np.argsort(B.sum(axis=1))[::-1],:]
    return B


#%% force deg dist
n=100
distribution_params = (20, 3)
row_sums = np.random.normal(*distribution_params, n)
row_sums = np.maximum(np.minimum(row_sums, n), 0).astype(int) # how abour np.clip
M = np.zeros((n, n), dtype=int)
for i in range(n): M[i,np.random.choice(range(n),size=row_sums[i],replace=False)]=1
mx.showdata(M)

data = M.sum(0)
plt.hist(data, bins=np.linspace(min(data), max(data),20), edgecolor='red',   rwidth=0.8, histtype='step',density=True)
plt.show()



M = forceDeg(row_sums,n)
mx.showdata(M)

#%% force deg dist 2

# def forceDeg(degrees,m,degrees2=None,n=None): # no sale simetrica

#     degrees = np.sort(degrees)[::-1]
#     if n==None: n=m
#     if degrees2==None: degrees2=degrees

#     r = list(map(tuple, list(map(np.random.choice, [range(n)]*m, degrees2, [False]*m))))
    
#     Br = np.zeros((m,n))

#     for i in range(m):
#         Br[(tuple([i]*len(r[i])),r[i])] = 1
    
#     colsums = Br.sum(axis=0) - degrees
#     initial = (colsums > 0).sum()
#     while (colsums > 0).sum() > 0:
#         Br=Br[:,np.argsort(Br.sum(axis=0))[::-1]] #sort columns 
#         #Br=Br[np.argsort(Br.sum(axis=1))[::-1],:] # no row sorting needed
        
#         colsums = Br.sum(axis=0) - degrees #;colsums
        
#         donor    = np.where(colsums > 0)[0] [-1] #;colsums[donor]
#         acceptor = np.where(colsums < 0)[0] [ 0] #;colsums[acceptor]
#         ''' force deg dist 3
n=100
distribution_params = (10, 100)
degrees = np.random.normal(*distribution_params, n)
degrees = np.maximum(np.minimum(degrees, n), 0).astype(int) # how abour np.clip

degrees = np.sort(degrees)#[::-1]
M = np.zeros((n, n), dtype=int)

for i in range(n):
    triu_row = [x for x in range(n) if x > i]
    nlinksleft = degrees[i] - M.sum(1)[i]
    print (i)
    if nlinksleft >0 and nlinksleft < len(triu_row):
        ix = np.random.choice(triu_row, size=nlinksleft, replace=False)
        M[i,ix] = 1
        M[ix,i] = 1
    else:
        print("skipped {0}".format(i))

    
    #M[:,i]=M[i,:]

    
mx.showdata(M)
degrees-M.sum(1)
data2=M.sum(0)

print("connectance: " + str(M.sum()/(n**2)))
#%% 
# distribution_params = (20, 10)
# degrees = np.random.normal(*distribution_params, n)
# degrees = np.maximum(np.minimum(degrees, n), 0).astype(int) # how abour np.clip
# degrees = np.sort(degrees)[::-1]


#%% modularity evolve
'''
def getMostModularfrom(p):
    # p is a list of adjacency matrices
    return p[np.argmax(list(map( lambda x: mx.mod(nx.from_numpy_array(x)), p)))]

initial=M
# initial=getMostModularfrom(p)
p = opt.evolve(
    indiv = initial,
    policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
    mutation= lambda x: mx.double_edge_swap(x,nswaps=np.random.randint(int(n**2/100)),ntries=int(n**2)), 
    ngens=1,
    popsize=1000,
    podium=20,
    keep_dad=True)

p = opt.evolve(
    indiv = getMostModularfrom(p),
    policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
    mutation= lambda x: mx.double_edge_swap(x,nswaps=1,ntries=int(n**2)), 
    ngens=1,
    popsize=1000,
    podium=20,
    keep_dad=True)

p = opt.evolve(
    indiv = getMostModularfrom(p),
    policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
    mutation= lambda x: mx.double_edge_swap(x,nswaps=5,ntries=int(n**2)), 
    ngens=3,
    podium=10,
    popsize=1000,
    keep_dad=True)

# p = opt.evolve(
#     indiv = getMostModularfrom(p),
#     policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
#     mutation= lambda x: mx.double_edge_swap(x,10), 
#     ngens=15)

p = opt.evolve(
    indiv = getMostModularfrom(p),
    policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
    mutation= lambda x: mx.double_edge_swap(x,2), 
    ngens=10,
    popsize=100,
    keep_dad=True)

p = opt.evolve(
    indiv = getMostModularfrom(p),
    policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
    mutation= lambda x: mx.double_edge_swap(x,1), 
    ngens=10,
    popsize=1000,
    keep_dad=True)
#==============================we could repeat these threee, it gives some margin to improve
p = opt.evolve(
    indiv = getMostModularfrom(p),
    policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
    mutation= lambda x: mx.double_edge_swap(x,10), 
    ngens=15,
    podium=30,
    popsize=200,
    keep_dad=True)

p = opt.evolve(
    indiv = getMostModularfrom(p),
    policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
    mutation= lambda x: mx.double_edge_swap(x,1), 
    ngens=50,
    podium=5,
    popsize=100,
    keep_dad=False) # not keeping dad helps escaping adaptive local optima

p = opt.evolve(
    indiv = getMostModularfrom(p),
    policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
    mutation= lambda x: mx.double_edge_swap(x,1), 
    ngens=50,
    podium=1,
    popsize=100,
    keep_dad=True)

winner = getMostModularfrom(p)
print( mx.mod(nx.from_numpy_array(winner)))
if mx.mod(nx.from_numpy_array(winner)) > mx.mod(nx.from_numpy_array(initial)) :
    print("successfully evolved")
else:
    print("evolution failed")
#%%
initial=getMostModularfrom(p)
p = opt.evolve(
    indiv = initial,
    policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
    mutation= lambda x: mx.double_edge_swap(x,nswaps=np.random.randint(int(n**2/100)),ntries=int(n**2)), 
    ngens=4,
    popsize=1000,
    podium=20,
    keep_dad=True)

p2 = opt.evolve(
    indiv = p[np.random.randint(len(p))],
    policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
    mutation= lambda x: mx.double_edge_swap(x,nswaps=5,ntries=int(n**2)), 
    ngens=4,
    popsize=800,
    podium=20,
    keep_dad=True)

p2 = opt.evolve(
    indiv = getMostModularfrom(p2),
    policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
    mutation= lambda x: mx.double_edge_swap(x,nswaps=1,ntries=int(n**2)), 
    ngens=10,
    popsize=100,
    podium=50,
    keep_dad=True)


p2 = opt.evolve(
    indiv = getMostModularfrom(p2),
    policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
    mutation= lambda x: mx.double_edge_swap(x,1), 
    ngens=50,
    podium=1,
    popsize=100,
    keep_dad=True)

#%%

G= nx.from_numpy_array(initial)
Go=nx.from_numpy_array(winner)

pos = nx.layout.spring_layout(Go)
#pos = nx.layout.kamada_kawai_layout(Go)
nx.draw(G,pos=pos,node_size=10); plt.show()
nx.draw(Go,pos=pos,node_size=10); plt.show()
nx.draw(nx.from_numpy_array(getMostModularfrom(p)),pos=pos,node_size=10); plt.show()

#print(A.sum(0)[::-1] - p[0].sum(0)[::-1])

mx.showdata(initial)
mx.showdata(getMostModularfrom(p))

mx.mod(nx.from_numpy_array(M))

#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++





#%% nestedness evolve
def getMostNODFfrom(p):
    # p is a list of adjacency matrices
    return p[np.argmax(list(map( lambda x: mx.nodf(x), p)))]

initial=Mr
# initial=winner
p = opt.evolve(
    indiv = initial,
    policy  = lambda x: mx.nodf(x),
    mutation= lambda x: mx.double_edge_swap(x,nswaps=np.random.randint(int(n**2/100)),ntries=int(n**2)), 
    ngens=1,
    popsize=1000,
    podium=20,
    keep_dad=True)

p = opt.evolve(
    indiv = getMostNODFfrom(p),
    policy  = lambda x: mx.nodf(x),
    mutation= lambda x: mx.double_edge_swap(x,nswaps=1,ntries=int(n**2)), 
    ngens=1,
    popsize=1000,
    podium=20,
    keep_dad=True)

p = opt.evolve(
    indiv = getMostNODFfrom(p),
    policy  = lambda x: mx.nodf(x),
    mutation= lambda x: mx.double_edge_swap(x,nswaps=5,ntries=int(n**2)), 
    ngens=3,
    podium=10,
    popsize=1000,
    keep_dad=True)

# p = opt.evolve(
#     indiv = getMostModularfrom(p),
#     policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
#     mutation= lambda x: mx.double_edge_swap(x,10), 
#     ngens=15)

p = opt.evolve(
    indiv = getMostNODFfrom(p),
    policy  = lambda x: mx.nodf(x),
    mutation= lambda x: mx.double_edge_swap(x,2), 
    ngens=20,
    popsize=100,
    keep_dad=True)

p = opt.evolve( # this one is very effective
    indiv = getMostNODFfrom(p),
    policy  = lambda x: mx.nodf(x),
    mutation= lambda x: mx.double_edge_swap(x,1), 
    ngens=10,
    popsize=1000,
    keep_dad=True)

# p = opt.evolve(
#     indiv = getMostNODFfrom(p),
#     policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
#     mutation= lambda x: mx.double_edge_swap(x,10), 
#     ngens=15,
#     podium=30,
#     popsize=200,
#     keep_dad=True)

p = opt.evolve(
    indiv = getMostNODFfrom(p),
    policy  = lambda x: mx.nodf(x),
    mutation= lambda x: mx.double_edge_swap(x,1), 
    ngens=50,
    podium=5,
    popsize=100,
    keep_dad=False)

p = opt.evolve(
    indiv = getMostNODFfrom(p),
    policy  = lambda x: mx.nodf(x),
    mutation= lambda x: mx.double_edge_swap(x,1), 
    ngens=50,
    podium=1,
    popsize=100,
    keep_dad=True)


winner = getMostNODFfrom(p)
print( mx.nodf(winner))
if mx.nodf(winner) > mx.nodf(initial) :
    print("successfully evolved")
else:
    print("evolution failed")
'''
#%% force deg dist 3 # the EFFECTIVE ONE

n=50
#if sum of degrees is odd, result will be approximate to deg sequence
#--------------------------------------------------------------------------------------
# degrees = np.random.normal(7,6, n)				;dn = "normal_" + str(n)
# degrees = np.random.lognormal(2.4 ,0.5, n)		;dn = "lognormal_" + str(n)
degrees = np.random.randint(1,int(n/2),n)			;dn = "randint_" + str(n)
# degrees = np.random.negative_binomial(n,0.7, n)	;dn = "negative_binomial_" + str(n)
# degrees =  np.random.pareto(1.2, n) 				;dn = "pareto_" + str(n)
#--------------------------------------------------------------------------------------



dnt = dn + '_'+ str(time.time())

# degrees = np.maximum(np.minimum(degrees, n), 1).astype(int) # how abour np.clip
degrees = np.clip(degrees,1,n).astype(int)
degrees = np.sort(degrees)[::-1]
M = np.zeros((n, n), dtype=int)
for i in range(n):
    #print (i)
    #triu_row = [x for x in range(n) if x > i]
    nlinksleft = degrees[i] - M.sum(1)[i]
    M[i,i+1:i+1+nlinksleft] = M[i+1:i+1+nlinksleft, i] = 1

if M.sum(0)[-1] == 0: M[-2,-1]=M[-1,-2]=1
mx.showdata(M)
print("connectance: " + str(M.sum()/(n**2)))
#Mr=mx.swaplinks(M,int(n**2/4))
Mr=mx.double_edge_swap(M,nswaps=int(n**2/4),ntries=int(n**2))
mx.showdata(Mr)
print(M.sum(0) - Mr.sum(0))
print(M.sum(0) - degrees)
print(dnt)

#%%
if not sum(degrees)%2: 
    G2 = nx.configuration_model(degrees)
    M2 = nx.adjacency_matrix(G2).todense()
    
    nx.draw(G2,node_size=10); plt.show()
    mx.showdata(M2)
    print(M2.sum(0) - degrees)

#%%
lsp = (0,n,30)
data = M.sum(0)[::-1]
data_r = Mr.sum(0)[::-1]
plt.hist(data,    bins=np.linspace(*lsp), edgecolor='red',   rwidth=0.8, histtype='step',density=True)
plt.hist(data_r,  bins=np.linspace(*lsp), edgecolor='green', rwidth=0.8, histtype='step',density=True,alpha=0.5)
# plt.hist(data2,   bins=np.linspace(*lsp), edgecolor='blue',  rwidth=0.8, histtype='step',density=True,alpha=0.5)
plt.hist(degrees, bins=np.linspace(*lsp), edgecolor='black', rwidth=0.8, histtype='step',density=True,alpha=0.5)
plt.show()

    
    
    
#%% general evolve
policyFunc = lambda x: mx.mod(nx.from_numpy_array(x))		;dnts = dnt + "_modular"
# policyFunc = lambda x: -mx.mod(nx.from_numpy_array(x)) 	;dnts = dnt + "_antimodular"
# policyFunc = lambda x: mx.nodf(x)							;dnts = dnt + "_nested"
# policyFunc = lambda x: -mx.nodf(x)						;dnts = dnt + "_antinested"
# policyFunc = lambda x: mx.spectralRnorm(x)				;dnts = dnt + "_spectral"
# policyFunc = lambda x: -mx.spectralRnorm(x) 				;dnts = dnt + "_antispectral"
#(up) change accordingly

dnts += "_"+ str(time.ctime().replace(' ','_').replace(':',''))

def getbestfrom(p):
    global policyFunc
    # p is a list of adjacency matrices
    return p[np.argmax(list(map( policyFunc, p)))]

# initial=M
initial=Mr
# initial=winner
# initial=getMostModularfrom(p)
p = opt.evolve(
    indiv = initial,
    policy  = policyFunc,
    mutation= lambda x: mx.double_edge_swap(x,nswaps=np.random.randint(int(n**2/100)),ntries=int(n**2)), 
    ngens=1,
    popsize=1000,
    podium=20,
    keep_dad=True)

p = opt.evolve(
    indiv = getbestfrom(p),
    policy  = policyFunc,
    mutation= lambda x: mx.double_edge_swap(x,nswaps=1,ntries=int(n**2)), 
    ngens=1,
    popsize=1000,
    podium=20,
    keep_dad=True)

p = opt.evolve(
    indiv = getbestfrom(p),
    policy  = policyFunc,
    mutation= lambda x: mx.double_edge_swap(x,nswaps=5,ntries=int(n**2)), 
    ngens=3,
    podium=10,
    popsize=1000,
    keep_dad=True)

# p = opt.evolve(
#     indiv = getMostModularfrom(p),
#     policy  = lambda x: mx.mod(nx.from_numpy_array(x)),
#     mutation= lambda x: mx.double_edge_swap(x,10), 
#     ngens=15)

p = opt.evolve(
    indiv = getbestfrom(p),
    policy  = policyFunc,
    mutation= lambda x: mx.double_edge_swap(x,2,ntries=int(n**2)), 
    ngens=10,
    popsize=100,
    keep_dad=True)

p = opt.evolve(
    indiv = getbestfrom(p),
    policy  = policyFunc,
    mutation= lambda x: mx.double_edge_swap(x,1,ntries=int(n**2)), 
    ngens=10,
    popsize=1000,
    keep_dad=True)

p = opt.evolve(
    indiv = getbestfrom(p),
    policy  = policyFunc,
    mutation= lambda x: mx.double_edge_swap(x,10,ntries=int(n**2)), 
    ngens=15,
    podium=30,
    popsize=200,
    keep_dad=True)
#==============================we could repeat these two, it gives some margin to improve
p = opt.evolve(
    indiv = getbestfrom(p),
    policy  = policyFunc,
    mutation= lambda x: mx.double_edge_swap(x,2,ntries=int(n**2)), 
    ngens=50,
    podium=5,
    popsize=100,
    keep_dad=False) # not keeping dad helps overcome local optima

p = opt.evolve(
    indiv = getbestfrom(p),
    policy  = policyFunc,
    mutation= lambda x: mx.double_edge_swap(x,1,ntries=int(n**2)), 
    ngens=100, # nodf usually needs around 100
    podium=1,
    popsize=100,
    keep_dad=True)

winner = getbestfrom(p)
print( policyFunc(winner))
if policyFunc(winner) > policyFunc(initial) :
    print("successfully evolved ({0} -> {1})".format(round(policyFunc(initial),4), round(policyFunc(winner),4)))
else:
    print("evolution failed")
#%%

mx.showdata(initial)
mx.showdata(getMostModularfrom(p))

G= nx.from_numpy_array(initial)
Go=nx.from_numpy_array(winner)

# pos = nx.layout.spring_layout(Go)
pos = nx.layout.kamada_kawai_layout(Go)
nx.draw(G,node_size=10); plt.show()
nx.draw(G,pos=pos,node_size=10); plt.show()
nx.draw(Go,pos=pos,node_size=10); plt.show()


#%%SAVE
# file_rawname = input("save network: ")
file_rawname = dnts
filename='NET_' + file_rawname + '.obj'
print('\nSAVING NETWORK AS ' + str(obj_path / "special_networks" / filename))
with bz2.BZ2File(obj_path / "special_networks" / filename, 'wb') as f:
    pickle5.dump(winner, f)

#%%
'''
system('ls -c data/obj/special_networks/NET*')
system('basename -a ls data/obj/special_networks/NET* ')
print('\n'.join(listdir(obj_path)))
'''
#%%
filenames_folder = popen('ls -c data/obj/special_networks/NET*').read().split("\n"); filenames_folder=filenames_folder[:-1]
sp_nets = []
for i, netname in enumerate(filenames_folder):
    with bz2.BZ2File(netname, 'rb') as f:
     	sp_nets.append(pickle5.load(f))
    print("loaded " + netname + " at " + str(i))
         
# #%%
# filename='oc_tensor_' + str(nloci) + '.obj'
# with bz2.BZ2File(obj_path / filename, 'rb') as f:
# 	h = pickle5.load(f)
    