import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import chdir
from pathlib import Path
chdir('/home/roman/LAB/ECO') #this line is for Spyder IDE only
root = Path(".")

import sys
sys.path.insert(0, "./lib")
import matriX as mX

#%%
def showdata(mat, color=plt.cm.gnuplot, symmetry=False):
	mat = np.copy(mat)
	if symmetry:
		top = np.max([np.abs(np.nanmax(mat)),np.abs(np.nanmin(mat))])
		plt.imshow(mat.astype('float32'), interpolation='none', cmap='seismic',vmax=top,vmin=-top)
	else:
		plt.imshow(mat.astype('float32'), interpolation='none', cmap=color)
	plt.colorbar()
	plt.show()

def showlist(l, distbins=False):
            fig = plt.figure(); ax = fig.add_subplot(111)
            ax.plot(np.arange(len(l)),list(l))
            plt.show()

#%%

def showfunc(f,xlim=(-5,5),definition=100, **kwargs):
            x= np.linspace(xlim[0],xlim[1],definition)
            fig = plt.figure(); ax = fig.add_subplot(111)
            ax.plot(x,f(x,**kwargs))
            plt.show()

def pM (zdiffs, alpha=50):
    return np.exp(-alpha*(zdiffs)**2)

def pB (zdiffs, alpha=50):
    return 1/(1+np.exp(-alpha*(zdiffs)))



#%%

showfunc(pM, zj=2., xlim=(0,3))
showfunc(pB, zj=2., alpha=9, xlim=(0,3))
showfunc(pB, zj=2., alpha=2, xlim=(0,3))


#%% NETWORK ANALYSIS IMPORTS
import networkx as nx
from community import best_partition # pip3 install python-louvain
from collections import defaultdict
from networkx.algorithms.community.quality import modularity

from nestedness_calculator import NestednessCalculator #https://github.com/tsakim/nestedness
nodf = lambda x: NestednessCalculator(x).nodf(x)
#%% NETWORK ANALYSIS FUNCTIONS
def groupnodes(G):
    part = best_partition(G)
    inv = defaultdict(list)
    {inv[v].append(k) for k, v in part.items()}
    result = dict(inv)
    return(list(result.values()))

def mod(g):
    if type(g)==nx.classes.graph.Graph:
        comms = groupnodes(g)
        mod = modularity(g, comms)
    elif type(g)==list and np.all(type(G)==nx.classes.graph.Graph for G in g):
        comms = list(map(groupnodes, g))
        mod = list(map(modularity, g, comms))
    return(mod)

def renormalize(vlist):
    x=vlist
    b=np.max(x)
    a=np.min(x)
    x=(x-a)/(b-a)
    return x

def spectralRnorm(a):
    a_norm = renormalize(a)
    #L = sparse.csr_matrix(a_norm)
    #sR = sparse.linalg.eigs(a_norm,k=1,which='LM', return_eigenvectors=False)
    sR = np.max(np.real(np.linalg.eigvals(a_norm)))/np.sqrt((a>0).sum())
    return(sR)
#%% DATA LOAD
dataPath = root / 'data/dataBase'
df = pd.read_csv(dataPath / 'FW_017_03.csv', index_col=0)
#a = df.to_numpy()

np.all(df.columns == df.index)
#l[0,:,:]=(df.to_numpy()>0)+0.

b=(df.to_numpy()>0)+0
showdata(b)
#%% set simulation parameters
#N=b.shape[0]
N=5 # number of species
c=0.7 # expected connectance
ntimesteps=10000

pS=[2,2.5] # phenotypic space
#%%
initial_l=mX.generateWithoutUnconnected(N,N,c) 
initial_l=initial_l-np.diag(np.diag(initial_l))
initial_l=np.tril(initial_l,0)+np.tril(initial_l,0).T
#initial_l=b
showdata(initial_l)
nx.draw(nx.from_numpy_matrix(initial_l,parallel_edges=False))
print("connectance of " + str(initial_l.sum()/N**2) + ", expected " + str(c))

#%% generate theta
theta=np.random.rand(N)*np.diff(pS)+pS[0] # values favoured by env. selection
#%% simulate
mechanism_type = 1
# 1: 'mutualism + trait matching' or 
# 2: 'mutualism + exploitation barrier' or 
# 3: 'antagonism + trait matching (trait mismatching)' or
# 4: 'antagonism + exploitation barrier' 

xi_S=0.2 # level of environmental selection (from 0 to 1).
xi_d=1-xi_S # level of selection imposed by resource species (from 0 to 1).
alpha= 100 # strength of the mechanism. Controls how the difference in species traits affects the probability of pairwise interaction.
epsilon = 0.5 # threshold, assumed to be fixed and identical for all species.
phi=0.25 # slope of the selection gradient.

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
#p[0,:,:]=pM(z[0,:]*np.ones((N,1)), (z[0,:]*np.ones((N,1))).T)
p[0,:,:]=pB(z[0,:]*np.ones((N,1)), (z[0,:]*np.ones((N,1))).T)

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
       
    showlist((S+M.sum(2))[:50]) # all differentials converge to zero when the system reaches a fixed point
        
elif mechanism_type == 2:
    B[0,:,:]=xi_d*a[0,:,:]*(zdiffs.T+epsilon)
    for t in range(1,ntimesteps):
        z[t,:] = z[t-1,:] + phi*(S[t-1,:]+B[t-1,:,:].sum(1))
        zdiffs = (z[t,:]*np.ones((N,1))).T - z[t,:]*np.ones((N,1))
        S[t,:] = xi_S*(theta-z[t,:])
        p[t,:,:]=pB(zdiffs,alpha)
        lxp=l[0,:,:]*p[t,:,:]
        a[t,:,:] = lxp/(lxp-np.diag(np.diag(lxp))).sum(1)[:,np.newaxis]*np.ones((1,N))
        
        B[t,:,:]=xi_d*a[t,:,:]*(zdiffs.T+epsilon)
        
    showlist((S+B.sum(2))[:50]) # all differentials converge to zero when the system reaches a fixed point
    
elif mechanism_type == 3:
    u[0,:,:] = (np.abs(zdiffs) <= epsilon) + 0.
    M[0,:,:]=xi_d*a[0,:,:]*u[0,:,:]*(zdiffs.T + ((zdiffs.T > 0)*2.-1.)*epsilon)
    for t in range(1,ntimesteps):
        z[t,:] = z[t-1,:] + phi*(S[t-1,:]+M[t-1,:,:].sum(1))
        zdiffs = (z[t,:]*np.ones((N,1))).T - z[t,:]*np.ones((N,1))
        S[t,:] = xi_S*(theta-z[t,:])
        p[t,:,:]=pM(zdiffs,alpha)
        lxp=l[0,:,:]*p[t,:,:]
        a[t,:,:] = lxp/(lxp-np.diag(np.diag(lxp))).sum(1)[:,np.newaxis]*np.ones((1,N))
        
        u[t,:,:] = (np.abs(zdiffs)<=epsilon)+ 0.
        M[t,:,:]=xi_d*a[t,:,:]*u[t,:,:]*(zdiffs.T + ((zdiffs.T > 0)*2-1)*epsilon)
       
    showlist((S+M.sum(2))[:50]) # all differentials converge to zero when the system reaches a fixed point

elif mechanism_type == 4:
    u[0,:,:] = (np.abs(zdiffs) <= epsilon) + 0.
    B[0,:,:]=xi_d*a[0,:,:]*u[0,:,:]*(zdiffs.T + epsilon)
    for t in range(1,ntimesteps):
        z[t,:] = z[t-1,:] + phi*(S[t-1,:]+B[t-1,:,:].sum(1))
        zdiffs = (z[t,:]*np.ones((N,1))).T - z[t,:]*np.ones((N,1))
        S[t,:] = xi_S*(theta-z[t,:])
        p[t,:,:]=pB(zdiffs,alpha)
        lxp=l[0,:,:]*p[t,:,:]
        a[t,:,:] = lxp/(lxp-np.diag(np.diag(lxp))).sum(1)[:,np.newaxis]*np.ones((1,N))
        
        u[t,:,:] = (np.abs(zdiffs)<=epsilon)+ 0.
        B[t,:,:]=xi_d*a[t,:,:]*u[t,:,:]*(zdiffs + epsilon)
       
    showlist((S+B.sum(2))[:50]) # all differentials converge to zero when the system reaches a fixed point



showlist(z[:40,:])
#showlist(S[:50])

#%%
'''
i,j = 2,10
zdiffs[i,j] == z[t,i]-z[t,j]


#%%
showdata(p[5,:,:])
showdata(a[3,:,:])
'''


showdata(M[-1])




delta = [phi*np.dot(S[t,np.newaxis].T/N,np.ones((1,N)))+M[t] for t in range(ntimesteps)]


t=1
(S[t]+M[t].sum(1))*phi
delta[t].sum(1)


t=2
np.all(np.real(np.linalg.eigvals(delta[t]))<1)




#%% modularity calculation
G = nx.from_numpy_matrix(a[3,:,:],parallel_edges=False)
nx.draw(G)
nx.average_clustering(G)
mod(G)

#%%
workingwindow = 100

#%%
Gl = [a[t,:,:] for t in range(workingwindow)]
G = list(map(nx.from_numpy_matrix, Gl))
mod_list = mod(G)
showlist(mod_list)

#%%
nsamples=120
mod_list_b_sum = np.zeros((nsamples,workingwindow))

for i in range(nsamples):
    print('sample #' + str(i))
    #thres=np.random.rand(N,N)*.8+.1
    thres=np.random.rand(N,N)
    Glb =  [Gl[t] > thres for t in range(workingwindow)]
    #Glb =  [Gl[t] > 0.1 for t in range(200)]
    Gb = list(map(nx.from_numpy_matrix, Glb))
    mod_list_b = mod(Gb)
    mod_list_b_sum[i,:] = mod_list_b
    
mod_list_b_avg = mod_list_b_sum.sum(axis=0)/nsamples
showlist(mod_list_b_avg)

#%%
'''   FIXED THRESHOLD DOES NOT WORK WELL

Glb_fix =  [Gl[t] > 0.2 for t in range(200)]
Glb_fix =  [np.logical_and(Glb_fix[t],Glb_fix[t].T) for t in range(200)]
Glb_fix = list(map(mX.rmUnco, Glb_fix))
Gb_fix = list(map(nx.from_numpy_matrix, Glb_fix))
mod_list_b_fix = mod(Gb_fix)
showlist(mod_list_b_fix)
showlist(np.stack((mod_list_b_fix, mod_list_b_avg),1))
'''

#%%
fig = plt.figure(); ax = fig.add_subplot(111)
ax.scatter( np.repeat(np.arange(workingwindow), nsamples), mod_list_b_sum.flatten('F'), s=0.5)
ax.plot(np.arange(workingwindow),mod_list_b_avg, color='orange')
plt.show()

#%% nestedness (spectral radius)

sR_list = list(map(spectralRnorm, [a[t,:,:] for t in range(1,workingwindow)]))
#sR_list_bin = list(map(spectralRnorm, [(a[t,:,:] > 0.1)+0 for t in range(1,200)]))
sR_list_bin = list(map(spectralRnorm, [(a[t,:,:] > 0.1)+0 for t in range(1,workingwindow)]))
showlist(sR_list_bin)
showlist(sR_list)

#%% nestedness (NODF)
'''
Haz que discutir como determinar la matriz binaria para el NODF.
El enfoque probabilistico esta bien, pero dependiendo de como sea, el resultado
puede variar mucho.

He probado la dist uniforme entre 0,1. pero da cosas muy distintass a si aplico
un threshold (links por encima de cierto valor siempre estaran, y por debajo de
cierto valor nunca apareceran). Esto puede dar pie a un concepto interesante de 
nestedness a distintos niveles. Quiza los nexos con alta energia aportan mucho 
a la estructura anidada. quiza son los de baja energia. 

Ninguna forma parece asimilarse a la producida por el radio espectral.
'''
nsamples=120
nodf_scores_sum = np.zeros((nsamples,workingwindow))

for i in range(nsamples):
    thres=np.random.rand(N,N) #*.95+.025
    #thres=0.02+(np.random.rand(N,N)-.5)*.01
    Glb = [mX.rmUnco((Gl[t] > thres)+0) for t in range(workingwindow)]
    nodf_scores_sum[i,:]  = list(map(nodf, Glb))
    #test = [np.all(Glb[t].sum(axis=1) != 0) for t in range(workingwindow)]

    
nodf_scores_avg = nodf_scores_sum.sum(axis=0)/nsamples

#nodf_scores = list(map(nodf, [(a[t,:,:] > thres*0.1)+0 for t in range(1,workingwindow)]))
#nodf_scores = list(map(nodf, [(a[t,:,:] > 0.01)+0 for t in range(1,200)]))
showlist(nodf_scores_avg[1:])

#%%

fig = plt.figure(figsize=(12,9)); ax = fig.add_subplot(111)
ax.scatter( np.repeat(np.arange(workingwindow), nsamples), nodf_scores_sum.flatten('C'), s=0.5)
ax.plot(np.arange(workingwindow),nodf_scores_avg, color='orange')
plt.show()

#%%
n_intervals=4
temp_labellist=np.arange(0, workingwindow+1, int(workingwindow/n_intervals))

fig = plt.figure(figsize=(12,9)); ax = fig.add_subplot(111)
ax.plot(np.arange(workingwindow),nodf_scores_avg, color='orange')
ax.boxplot(nodf_scores_sum, sym='.')
plt.ylabel('NODF score')
plt.xlabel('time')
ax.set_xticks(ticks = temp_labellist, labels=temp_labellist)
plt.show()

#%%

g=nx.from_numpy_matrix(a[12,:,:],parallel_edges=False)
pos = nx.spring_layout(g,scale=0.5)
edges = g.edges()
weights = [g[u][v]['weight'] for u,v in edges]
degrees = np.array(g.degree)[:,1]


palette1=plt.cm.summer #binary jet 
palette2=plt.cm.Wistia_r
edgecolors=list(map(plt.cm.colors.rgb2hex,list(map(palette1, renormalize(weights))))) # 1-renormalize(weights)
nodecolors=list(map(plt.cm.colors.rgb2hex,list(map(palette2, renormalize(degrees)))))

#nx.draw(g, width=weights*10, pos=pos, edge_color=edgecolors, node_size=0.7*degrees, node_color=nodecolors)
#nx.draw(g, width=0.2+3*renormalize(weights), pos=pos, edge_color=edgecolors, node_size=2*degrees, node_color=nodecolors)


fig, ax = plt.subplots()
nx.draw(g, width=0.2+3*renormalize(weights), pos=pos, edge_color=edgecolors, node_size=10+60*renormalize(degrees), node_color=nodecolors)
ax.set_facecolor('deepskyblue')
ax.axis('off')
fig.set_facecolor('black')
plt.show()