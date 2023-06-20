#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 08:21:34 2023

@author: ubuntu
"""

from os import chdir, listdir, environ, system
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
# from copy import deepcopy
import networkx as nx
# import pandas as pd
#%% OWN LIBS
sys.path.insert(0, "./lib")
import evo
import matriX as mx

#%% run model
# system('python chapter_3/III_MAIN_multiprocessor.py')
# !python chapter_3/III_MAIN_multiprocessor.py
# python chapter_3/III_MAIN_multiprocessor.py

#%% load dataset
#pattern = r'^file\d+\.txt$'  # Regular expression pattern

system('cat data/obj/report_test.txt')
#system('cat data/obj/report_1686233042.5595455.txt')
#
system('ls -l data/obj')

listdir(obj_path)
filename = 'SIMULATIONS_wet_supermarket_Wed_Jun_14_182043_2023.obj'
#filename = 'SIMULATIONS_scholarly_insurance_Thu_Jun_15_172723_2023.obj'

#filename = 'SIMULATIONS_1686233042.5595455.obj'
with bz2.BZ2File(obj_path / filename, 'rb') as f:
	simulations = pickle5.load(f)


ntimesteps, N, nstates = simulations[0]['v'].shape; ntimesteps-=1
ps = simulations[0]['_ps']
system('head data/obj/report_Thu_Jun_15_172723_2023.txt')
#%% show average traits
for sim in simulations:
    mx.showlist(sim['dist_avgs'])

#%% show fitnesses
for sim in simulations:
    mx.showlist(sim['fits'][:-1])

#%% show population sizes
for sim in simulations:
    mx.showlist(sim['D'][:-1])
  
#%% show trait variances
for sim in simulations:
    mx.showlist(evo.dist_variances(sim['v']),sim['_ps'])

#%% heavy trajectory exploration
'''
N   = sim['v'].shape[1]
for i in range(N):
    for sim in simulations:
        mx.showdata(sim['v'][:,i].T)
'''
system('cat data/obj/report_Thu_Jun_15_172723_2023.txt')
#%%
[sim['_a'] for sim in simulations]
[sim['_d'].mean() for sim in simulations]
[sim['_m'].mean() for sim in simulations]

[(sim['_mutual_effs']!=0).sum()/sim['_mutual_effs'].size for sim in simulations] #connectances

[(sim['_mutual_effs'].min(),sim['_mutual_effs'].max()) for sim in simulations]


for sim in simulations: mx.showdata(sim['v'][:,0])
for sim in simulations: mx.showdata(sim['_mutual_effs'],symmetry=True)

tol = 1e-4
np.all(np.abs(np.diff(sim['v'][-2:],axis=0))<tol)
np.all(np.abs(np.diff(sim['D'][-2:],axis=0))<tol)
(np.abs(np.diff(sim['v'][-2:],axis=0))<tol).sum()/sim['v'][0].size
(np.abs(np.diff(sim['D'][-2:],axis=0))<tol).sum()/N

#%%
v = sim['v'][:,1] ;mx.showdata(v)

levels = np.linspace(0,Z.max(), 8)

# for sim in simulations:
#     v = sim['v'][:,1]
# vs=simulations[-1]['v'] 
# for s in range(vs.shape[1]):
#     v = vs[:,s]
    
    ntimesteps, nstates = v.shape; ntimesteps-=1
    X, Y = np.meshgrid(np.linspace(0,ntimesteps,ntimesteps+1), np.linspace(ps[0],ps[1], nstates))
    Z = np.squeeze(v)
    fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
    CS = ax.contourf(X.T, Y.T, Z, levels=10,cmap='binary')
    #ax.contour(X.T, Y.T, Z, levels=levels,colors='r',linewidths=1)
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel('probability')
    plt.show()

#%% check which parameters are set to be fixed
keylist = list(simulations[0].keys())[:-8]
fixed = {}
for key in keylist:
    lst = [sim[key] for sim in simulations]
    fixed[key] = np.all([(par == lst[0]) for par in lst])

print('non-fixed values:')
print(*["\t"+key for key, value in fixed.items() if not value], sep="\n")

#%% multimodal search
from scipy.signal import find_peaks
for simID, sim in enumerate(simulations):
    for i in range(N):
        #[p[0] for p in list(map(find_peaks,list(sim['v'][:,i])))]
        if np.any([len(p[0])>1 for p in list(map(find_peaks,list(sim['v'][:,i])))]):
            print('found multimodal distribution for species {0} in simulation {1}'.format(i, simID))
    

#sim = simulations[59]; mx.showdata(sim['v'][:200,10])

#%%
final_popsizes   = [sim['D'][-1].sum()for sim in simulations]
final_nonextinct = [(sim['D'][-1] > 2).sum()for sim in simulations]

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
scatter1=ax.scatter([sim['_d'].mean() for sim in simulations],
                    [sim['_m'].mean() for sim in simulations], 
                    final_popsizes/sum(final_popsizes), label='biomass (total n individuals)')
scatter2=ax.scatter([sim['_d'].mean() for sim in simulations],
                    [sim['_m'].mean() for sim in simulations], 
                    final_nonextinct/sum(final_nonextinct), label='n survivors')
ax.set_ylabel(r"$\bar{\xi_d}$", fontsize=16)
ax.set_xlabel(r"$d$", fontsize=16)
plt.legend(handles=[scatter1, scatter2])
plt.show()

#%%


fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
scatter1=ax.scatter([sim['_d'].mean() for sim in simulations],
                    [sim['n_predations'] for sim in simulations], 
                    final_popsizes/sum(final_popsizes), label='final_popsizes', 
                    c=[sim['n_mutualisms'] for sim in simulations], cmap='viridis')

# scatter2=ax.scatter([sim['_d'].mean() for sim in simulations],
#                     [sim['n_predations'] for sim in simulations], 
#                     final_nonextinct/sum(final_nonextinct), label='n survivors')


ax.set_xlabel(r"$d$", fontsize=16)
ax.set_ylabel("n_predations")
plt.legend(handles=[scatter1, scatter2])
plt.colorbar(scatter1)
plt.show()
    

#%% get interesting things
I=np.newaxis
t=-1
adjs=[]
mutu=[]
comp=[]
pred=[]
for sim in simulations:
    
    ntimesteps, N, nstates = sim['v'].shape; ntimesteps-=1
    
    A_e = sim['_mutual_effs']
    A=A_e != 0
    p=np.array([evo.interactors.convpM(sim['v'][t,species_id],nstates,sim['_alpha']) for species_id in range(N)]) # equivalent to p
    k1=(A[...,np.newaxis] @ sim['v'][t,:,np.newaxis,:])
    k=(A[...,np.newaxis] @ p[:,np.newaxis,:])
    e = k * np.swapaxes(k1,0,1)
    # adj_timeseries.append(e.sum(2))
    # BELOW: includes effects of population sizes 
    pop_weights = sim['D'][t-1][:,I]# * sim['_m'] 
    #intensities =  (np.abs(sim['_mutual_effs'])+np.abs(sim['_mutual_effs']).T)/2 * np.sqrt(np.outer(pop_weights,pop_weights))
    intensities = np.sqrt(np.outer(pop_weights,pop_weights))
    
    # mx.showdata(e.sum(2),colorbar=True)
    # mx.showdata(e.sum(2)*intensities,colorbar=True)
    adj = e.sum(2)*intensities
    
    mutu.append(adj *  ((A_e>0) & (A_e.T>0)))
    comp.append(adj *  ((A_e<0) & (A_e.T<0)))
    pred.append(adj * (((A_e>0) & (A_e.T<0)) | ((A_e<0) & (A_e.T>0))))
    
    adjs.append(adj)


#for mat in pred: mx.showdata(mat)

thres_interac=0.01
n_final_mutu = [(mat >thres_interac).sum()/2 for mat in mutu ]
n_final_comp = [(mat >thres_interac).sum()/2 for mat in comp ]
n_final_pred = [(mat >thres_interac).sum()/2 for mat in pred ]

final_mean_mutu = list(map(np.mean, mutu))
final_mean_comp = list(map(np.mean, comp))
final_mean_pred = list(map(np.mean, pred))

n_mutu = [sim['n_mutualisms'] for sim in simulations]
n_comp = [sim['n_competitions'] for sim in simulations]
n_pred = [sim['n_predations'] for sim in simulations]



final_popsizes_mean = [sim['D'][-1].mean()for sim in simulations]
final_popsizes      = [sim['D'][-1].sum()for sim in simulations]
final_nonextinct    = [(sim['D'][-1] > 10).sum()for sim in simulations]


n_connected_comps = [nx.number_connected_components(nx.from_numpy_array(((np.abs(adj)+np.abs(adj.T))/2>0.001))) for adj in adjs]

a = [sim['_a'].mean() for sim in simulations]
d = [sim['_d'].mean() for sim in simulations]

[(mat >thres_interac).sum(0) for mat in adjs ]


mx.showdata(adj)
#%%

Gl = [nx.from_numpy_array(adj) for adj in adjs]
modularities = mx.mod( Gl )

nx.draw(Gl[np.argmax(modularities)],node_size=10)
nx.draw(Gl[np.argmin(modularities)],node_size=10)


Gl = [nx.from_numpy_array(adj * (sim['_mutual_effs'] != 0)) for adj, sim in zip(adjs, simulations)]




#%%



fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
scatter1=ax.scatter([sim['_a'].mean() for sim in simulations],
                    [sim['n_mutualisms'] for sim in simulations], 
                    list(map(np.mean, mutu)), label='mutualism strength')

scatter2=ax.scatter([sim['_a'].mean() for sim in simulations],
                    [sim['n_competitions'] for sim in simulations], 
                    list(map(np.mean, comp)), label='competition strength')

scatter3=ax.scatter([sim['_a'].mean() for sim in simulations],
                    [sim['n_predations'] for sim in simulations], 
                    list(map(np.mean, pred)), label='predation strength')



# scatter2=ax.scatter([sim['_d'].mean() for sim in simulations],
#                     [sim['_m'].mean() for sim in simulations], 
#                     final_nonextinct/sum(final_nonextinct), label='n survivors')

ax.set_xlabel(r"$a$", fontsize=16)
ax.set_ylabel(r"$n$", fontsize=16)
plt.legend(handles=[scatter1, scatter2,scatter3])
plt.show()


#%%
G = nx.from_numpy_array(A_e, create_using=nx.DiGraph)
nx.draw(G)
G.edges(data=True)
cycles = nx.simple_cycles(G)
#weighted_cycles = [cycle for cycle in cycles if sum(G[u][v]['weight'] for u, v in zip(cycle, cycle[1:] + cycle[:1])) != 0]
weighted_in_degree = dict(G.in_degree(weight='weight'))
weighted_out_degree = dict(G.out_degree(weight='weight'))

# weighted_in_degree reflects the overall effect of other species on this one.
# weighted_out_degree reflects the overall effect of this species on the other ones.



cycles = nx.simple_cycles(G)
weighted_cycles = [cycle for cycle in cycles if np.prod(G[u][v]['weight'] for u, v in zip(cycle, cycle[1:] + cycle[:1])) != 0]
r1=[]
r2=[]

if   ftype=='positive':
    for cycle in weighted_cycles:
        cycle_weights = []
        
        for u, v in zip(cycle, cycle[1:] + cycle[:1]):
            weight = G[u][v]['weight']
            cycle_weights.append(weight)

            if np.prod(cycle_weights)>0:
                #print(f"Weights in cycle {cycle}: {cycle_weights}")
                r1.append(cycle); r2.append(cycle_weights)
                
fp = mx.find_feedbacks(G,ftype='positive')
fn = mx.find_feedbacks(G,ftype='negative')

len(fp[0])
len(fn[0])

mx.mod(nx.from_numpy_array(np.abs(A_e)))
mx.mod(nx.from_numpy_array(       A))

nx.eigenvector_centrality(G)

#%%
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')

scatter1=ax.scatter(modularities,
                    n_mutu,
                    n_connected_comps,
                    c=a, cmap='viridis')

'''
scatter1=ax.scatter(final_nonextinct,
                    n_pred,
                    n_connected_comps,
                    c=a, cmap='viridis')


scatter1=ax.scatter(a,
                    ,
                    n_final_mutu/n_mutu,
                    c=d, cmap='viridis')
'''
ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y")
# plt.legend(handles=[scatter1, scatter2])
plt.colorbar(scatter1)
plt.show()

#
#%% select single simulation
survival_percentage = 0.9
simulations_sustainable = [sim for sim in simulations if (sim['D'][-1]>1).sum()/sim['D'].shape[1] > survival_percentage]
sim = np.random.choice(simulations_sustainable); mx.showlist(sim['dist_avgs']); mx.showlist(sim['D'])


#%%# get adjacency timeseries ----------

ntimesteps, N, nstates = sim['v'].shape; ntimesteps-=1

A_e = sim['_mutual_effs']
A=A_e != 0

'''
print("connectance of " + str(A.sum()/N**2))
print("generated with: \n{0} mutualisms, \n{1} antagonisms (competitors), and \n{2} antagonisms (predation)".format(
sim['n_mutualisms'],
sim['n_competitions'],
sim['n_predations']))
'''

adj_timeseries = []
for t in range(ntimesteps):
    print(str(int(round(t/ntimesteps,2)*100)) + '%')
    p=np.array([evo.interactors.convpM(sim['v'][t,species_id],nstates,sim['_alpha']) for species_id in range(N)]) # equivalent to p
    k1=(A[...,np.newaxis] @ sim['v'][t,:,np.newaxis,:])
    k=(A[...,np.newaxis] @ p[:,np.newaxis,:])
    e = k * np.swapaxes(k1,0,1)
    # adj_timeseries.append(e.sum(2))
    # BELOW: includes effects of population sizes 
    pop_weights = sim['D'][t-1][:,I]# * sim['_m'] 
    #intensities =  (np.abs(sim['_mutual_effs'])+np.abs(sim['_mutual_effs']).T)/2 * np.sqrt(np.outer(pop_weights,pop_weights))
    intensities = np.sqrt(np.outer(pop_weights,pop_weights))
    
    # mx.showdata(e.sum(2),colorbar=True)
    # mx.showdata(e.sum(2)*intensities,colorbar=True)
    
    adj_timeseries.append(e.sum(2)*intensities)
   

vmax= np.max(adj_timeseries)


#%%

mx.showdata(adj_timeseries[0],colorbar=True)
mx.showdata(adj_timeseries[-1],colorbar=True)
mx.showdata(adj_timeseries[0]*(np.abs(sim['_mutual_effs'])+np.abs(sim['_mutual_effs']).T)/2,colorbar=True)
mx.showdata(adj_timeseries[0]*sim['_mutual_effs'],symmetry=True,colorbar=True)
EFF=adj_timeseries[0]*sim['_mutual_effs']
eigs = np.linalg.eig(EFF)
np.real(eigs[0])
np.imag(eigs[0])

eigen_timeseries = [np.linalg.eig(adj*sim['_mutual_effs']) for adj in adj_timeseries]

mx.showlist([np.abs(np.real(eig[0])) for eig in eigen_timeseries])
mx.showlist([np.abs(np.imag(eig[0])) for eig in eigen_timeseries])
np.linalg.eig(sim['_mutual_effs'])[0]

#%%
A_e = sim['_mutual_effs']
mx.showdata(A_e,symmetry=True)
mx.showdata(adj_timeseries[-1])


mutualists = ((A_e>0) & (A_e.T>0))
competitors = ((A_e<0) & (A_e.T<0))
predations = ((A_e>0) & (A_e.T<0)) | ((A_e<0) & (A_e.T>0))

# plt.imshow(mutualists);plt.title('mutualists')
# plt.imshow(competitors);plt.title('competitors')
# plt.imshow(predations);plt.title('predations')



adj = adj_timeseries[-1]

adj[np.where(mutualists)].mean()
adj[np.where(competitors)].mean()
adj[np.where(predations)].mean()


powers = np.zeros ((ntimesteps,3))
for t in range(ntimesteps):
    adj = adj_timeseries[t]
    powers[t,0] = adj[np.where(mutualists)].mean()
    powers[t,1] = adj[np.where(competitors)].mean()
    powers[t,2] = adj[np.where(predations)].mean()


fig = plt.figure(figsize=(16,12)); ax = fig.add_subplot(111)
ax.plot(powers[:,0], label='mutualists')
ax.plot(powers[:,1], label='competitors')
ax.plot(powers[:,2], label='predations')
ax.legend(loc='lower right')
plt.show()

#%%
nbins =10
bins= np.linspace(0,vmax,nbins)
powersdist = np.zeros ((ntimesteps,3,nbins-1))
for t in range(ntimesteps):
    adj = adj_timeseries[t]
    hist
    powersdist[t,0], _ = np.histogram(adj[np.where(mutualists)], bins)
    powersdist[t,1], _ = np.histogram(adj[np.where(competitors)], bins)
    powersdist[t,2], _ = np.histogram(adj[np.where(predations)], bins)

mx.showdata(powersdist[:100,0])
mx.showlist(powersdist[:,0].T)

#%%
def samplecolors(n, type='hex',palette=plt.cm.gnuplot):
	if type == 'hex':
		return list(map(plt.cm.colors.rgb2hex, list(map(palette, np.linspace(1,0,n)))))
	elif type == 'rgba':
		return list(map(palette, np.linspace(1,0,n)))



colorgrid = samplecolors(ntimesteps,'rgba',palette=plt.cm.viridis)

fig = plt.figure(figsize=(16,12)); ax = fig.add_subplot(111)
for t in range (ntimesteps):
    ax.plot(powersdist[t,0],color=colorgrid[t])


#%%

import matplotlib.animation as animation

rsc = mx.rescale(sim['fits'])
pos = nx.layout.kamada_kawai_layout(nx.from_numpy_array(A))
fig = plt.figure(figsize=(16,12)); ax = fig.add_subplot(111)
#div = make_axes_locatable(ax)
smooth=0.75
def frame(t):
    print(t)
    global pos
    ax.clear()
    G=nx.from_numpy_array(adj_timeseries[t])
    pos_new=nx.layout.fruchterman_reingold_layout(G, weight='weight', pos=pos,threshold=1e-8,iterations=20) # threshold=1e-10,iterations=10
    pos = dict(zip(G.nodes(), list((np.array(list(pos.values()))*smooth + np.array(list(pos_new.values()))*(1-smooth)))))
    
    linewidths = list(1*np.array(list(nx.get_edge_attributes(G, 'weight').values())))
    node_weights = dict(zip(G.nodes(), rsc[t].tolist()))
    nx.set_node_attributes(G, node_weights, "weight")
    cmap = plt.cm.get_cmap("cool_r")
    
    #forces = np.array([sum([data['weight'] for _, _, data in G.edges(node, data=True)]) for node in G.nodes])

    node_colors = {node: cmap(weight) for node, weight in node_weights.items()}
    nx.draw_networkx(G, ax=ax, pos=pos, 
                     width=linewidths, 
                     edge_color=linewidths, 
                     edge_cmap=plt.cm.turbo, 
                     node_size= sim['D'][t], 
                     node_color=list(node_colors.values())) #node_size= force ?

    # ---------------------------------------------------------------------
    
    ax.set_title('t = {}'.format(t))
    plt.show()
    return ax



timecode = str(time.time())
timecode = '_test'
ani = animation.FuncAnimation(fig, frame, frames=ntimesteps, interval=50, blit=False)
ani.save('../figures/gif/net' + timecode +'.gif')


#%%
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
for t in range(2,ntimesteps):
    strengths = adj_timeseries[t].sum(0)
    #ax.scatter(strengths,sim['fits'][t],   c=plt.cm.get_cmap("cool_r")(t/ntimesteps) )
    ax.plot(np.array(adj_timeseries[t-2:t]).sum(1),sim['fits'][t-2:t],   c=plt.cm.get_cmap("viridis")(t/ntimesteps) )
plt.show()
#mx.showdata(adj_timeseries[t])
#%%


states = np.linspace(0,1, nstates)
statesdiff=np.outer(np.ones(nstates),states)-np.outer(states,np.ones(nstates))
#---------------------------------------------
for i in np.linspace(-2,2,30):

                                                          ## a = -1/a?;assortMat = 1 - assortMat
    mx.showdata(assortMat,colorbar=True)
    
    #%%
    
assortMat = interactors.pM(statesdiff,alpha=abs(i))
if i<0:                                                        ## a = -1/a?;assortMat = 1 - assortMat
    assortMat = 1 - assortMat
mx.showdata(assortMat)




for i in range(100):
    try :
        #tmp = mx.generateWithoutUnconnected(N,N,0.1)
        tmp = mx.symmetric_connected_adjacency(N,0.1)
    except ValueError:
        print(i)
    else:
        break
    
