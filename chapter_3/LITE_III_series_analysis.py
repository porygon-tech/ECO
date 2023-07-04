#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 08:21:34 2023

@author: ubuntu
"""
#python chapter_3/III_MAIN_multiprocessor.py

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
# %matplotlib inline
# from copy import deepcopy
import networkx as nx
# import pandas as pd
r = np.random.rand

#%% OWN LIBS
sys.path.insert(0, "./lib")
import evo
import matriX as mx

#%% run model
# system('python chapter_3/III_MAIN_multiprocessor.py')
# !python chapter_3/III_MAIN_multiprocessor.py
# python chapter_3/III_MAIN_multiprocessor.py


#%% developer tools

#plt.switch_backend('Qt5Agg')
import contextlib
import matplotlib

def switch_backend(gui):
    with contextlib.suppress(ValueError):
        matplotlib.use(gui, force=True)
    globals()['plt'] = matplotlib.pyplot

from matriX import showdata as sd

#%% load dataset
#pattern = r'^file\d+\.txt$'  # Regular expression pattern
from os import popen
most_recent = popen('basename $(ls ' + str(obj_path)+ '/SIMULATIONS* -t1 | head -n 1 )').read().strip()

# system('ls -l data/obj')
# system('basename -a ls data/obj/SIMULATIONS* ')
# print('\n'.join(listdir(obj_path)))

filename = most_recent
# filename = 'SIMULATIONS_Sat_Jun_24_002510_2023_oval_conclusion.obj'
# filename = 'SIMULATIONS_unhappy_ship_Thu_Jun_15_091706_2023.obj'

print("LOADING " + filename)
with bz2.BZ2File(obj_path / filename, 'rb') as f:
	simulations = pickle5.load(f)


_, N, nstates = simulations[0]['v'].shape;
ps = simulations[0]['_ps']


#%% get interesting things

adjs  , mutu  , comp  , pred  , gammas   = evo.getADJs(simulations, t=-1,return_gammas=True)
adjs_i, mutu_i, comp_i, pred_i, gammas_i = evo.getADJs(simulations, t= 0,return_gammas=True)

n_mutu = [sim['n_mutualisms'] for sim in simulations]
n_comp = [sim['n_competitions'] for sim in simulations]
n_pred = [sim['n_predations'] for sim in simulations]
payoffs = [sim['_mutual_effs'] for sim in simulations] # per capita fitness effects

a = [sim['_a'].mean() for sim in simulations]
d = [sim['_d'].mean() for sim in simulations]

thres_gamma=1e-5
c = [(A_e>thres_gamma).sum()/A_e.size for A_e in payoffs]
alpha = [sim['_alpha'] for sim in simulations]

power_mutu = [i__mutu.sum() / i__n_mutu if i__n_mutu != 0 else 0 for i__mutu, i__n_mutu in zip(mutu, n_mutu)]
power_pred = [i__pred.sum() / i__n_pred if i__n_pred != 0 else 0 for i__pred, i__n_pred in zip(pred, n_pred)]
power_comp = [i__comp.sum() / i__n_comp if i__n_comp != 0 else 0 for i__comp, i__n_comp in zip(comp, n_comp)]


#%%
# 
# sd(gammas[0],colorbar =True)
# sd(adjs[0],colorbar =True)

thres_interac=1.5


#spectral radius: final, initial and payoff matrices
# sR   = list(map(mx.spectralRnorm, [adj   for adj   in adjs  ]))
# sR_i = list(map(mx.spectralRnorm, [adj_i for adj_i in adjs_i]))
sR   = list(map(mx.spectralRnorm, [M for M in gammas   ]))
sR_i = list(map(mx.spectralRnorm, [M for M in gammas_i ]))
sRp  = list(map(mx.spectralRnorm, [M for M in payoffs]))
# sRc   = list(map(mx.spectralRnorm, [mx.rmUnco(adj  *(adj  >thres_interac)) for adj   in adjs  ]))
# sRc_i = list(map(mx.spectralRnorm, [mx.rmUnco(adj_i*(adj_i>thres_interac)) for adj_i in adjs_i]))

nodfs = list(map(mx.nodf, [mx.rmUnco(adj>thres_interac)for adj in adjs]))



n_final_mutu = [(mat >thres_interac).sum()/2 for mat in mutu ]
n_final_comp = [(mat >thres_interac).sum()/2 for mat in comp ]
n_final_pred = [(mat >thres_interac).sum()/2 for mat in pred ]

final_mean_mutu = list(map(np.mean, mutu))
final_mean_comp = list(map(np.mean, comp))
final_mean_pred = list(map(np.mean, pred))


final_popsizes_mean = [sim['D'][-1].mean()for sim in simulations]
final_popsizes      = [sim['D'][-1].sum()for sim in simulations]
final_nonextinct    = [(sim['D'][-1] > 10).sum()for sim in simulations]


n_connected_comps = [nx.number_connected_components(nx.from_numpy_array((adj>thres_interac))) for adj in adjs]


[(mat >thres_interac).sum(0) for mat in adjs ]


#adjs_inv = [np.where(adj > 1e-5, 1 / adj, adj) for adj in adjs]
#Gli = [nx.from_numpy_array(adj) for adj in adjs_inv]
Gl = [nx.from_numpy_array(adj) for adj in adjs]
Glc= [nx.from_numpy_array(adj*(adj>thres_interac)) for adj in adjs ]
Gl_payoffs = [nx.from_numpy_array(A_e, create_using=nx.DiGraph) for A_e in payoffs]

modularities = mx.mod( Gl )
modularities = mx.mod( Gl )
modularities = mx.mod( Gl )


data=gammas[0].flatten()
plt.hist(data, bins=np.linspace(min(data), max(data),40), edgecolor='black', alpha=0.5, rwidth=0.8, histtype='step');plt.yscale('log')






#%% show average traits
switch_backend('module://matplotlib_inline.backend_inline')
for sim in simulations:
    mx.showlist(sim['dist_avgs'])

#%% show fitnesses
switch_backend('module://matplotlib_inline.backend_inline')
for sim in simulations:
    mx.showlist(sim['fits'][:-1])

#%% show population sizes
switch_backend('module://matplotlib_inline.backend_inline')
for sim in simulations:
    mx.showlist(sim['D'][:-1])
  
#%% show trait variances
for sim in simulations:
    mx.showlist(evo.dist_variances(sim['v']),sim['_ps'])

#%%

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

# mx.showdata(simulations[57]['v'][:200,11])
# mx.showdata(simulations[107]['v'][1000:1050,6])
# mx.showdata(simulations[107]['v'][290:350,7])
# mx.showdata(simulations[113]['v'][:50,3])
#-------------------------^---------^--^-------

#%%


#%%


#%%



switch_backend('qt5agg') # or 'qtagg'
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
scatter1=ax.scatter(d,
                    sR,
                    modularities,
                    c=nodfs, cmap='viridis')
ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y")
# plt.legend(handles=[scatter1, scatter2])
plt.colorbar(scatter1)
plt.show()



#%% correlation time!

# data=nx.adjacency_matrix(mx.pruning.threshold(G,1.5)).todense().flatten()
# data=np.array(adjs).flatten()
# plt.hist(data, bins=np.linspace(min(data), max(data),30), edgecolor='black', alpha=0.5, rwidth=0.8, histtype='step');plt.yscale('log')
# np.quantile(data, 0.5)

l_deg    = [dict(G.degree()) for G in Gl]
l_deg_pr = [dict(mx.pruning.threshold(G,1.5).degree()) for G in Gl]
l_deg_q  = [dict(G.degree( weight='weight')) for G in Gl]
l_btw_c  = [nx.betweenness_centrality(G)     for G in Gl] 
l_eig_c  = [nx.eigenvector_centrality(G)     for G in Gl]

deg       = [list(i.values()) for i in l_deg       ]
deg_pr    = [list(i.values()) for i in l_deg_pr    ]
deg_q     = [list(i.values()) for i in l_deg_q     ]
btw_c     = [list(i.values()) for i in l_btw_c     ]
eig_c     = [list(i.values()) for i in l_eig_c     ]

l_payoffs_deg_in  = [dict(G.in_degree()) for G in Gl_payoffs]
l_payoffs_deg_out = [dict(G.out_degree()) for G in Gl_payoffs]
l_payoffs_deg_in_q  = [dict(G.in_degree( weight='weight')) for G in Gl_payoffs]
l_payoffs_deg_out_q = [dict(G.out_degree(weight='weight')) for G in Gl_payoffs]
l_payoffs_btw_c = [nx.betweenness_centrality(G)         for G in Gl_payoffs] 
l_payoffs_eig_c = [nx.eigenvector_centrality(G)         for G in Gl_payoffs]

payoffs_deg_out   = [list(i.values()) for i in l_payoffs_deg_out   ]
payoffs_deg_in    = [list(i.values()) for i in l_payoffs_deg_in    ]
payoffs_deg_out_q = [list(i.values()) for i in l_payoffs_deg_out_q ]
payoffs_deg_in_q  = [list(i.values()) for i in l_payoffs_deg_in_q  ]
payoffs_btw_c     = [list(i.values()) for i in l_payoffs_btw_c     ]
payoffs_eig_c     = [list(i.values()) for i in l_payoffs_eig_c     ]

from scipy.stats import pearsonr
import matriX as mx

'''
l_prra = [pearsonr(x, y) for x, y in zip(deg_in, eig_c)]

corrs = np.array([prra[0] for prra in l_prra])
pvals = np.array([prra[1] for prra in l_prra])

np.where(pvals < 0.001)

plt.scatter(x,y,z)
'''

testvars = np.array([
modularities,
sR,
nodfs,
a,
d,
n_mutu,
n_pred,
n_comp])

k = np.array([
deg,
deg_pr,
deg_q,
btw_c,
eig_c])

k_payoff = np.array([
payoffs_deg_out,
payoffs_deg_in,
payoffs_deg_out_q,
payoffs_deg_in_q,
payoffs_btw_c,
payoffs_eig_c])


#cs = np.apply_along_axis(np.tril, axis=0, arr=np.array(cs))


cs  = [np.tril(np.corrcoef(k[:,i,:])) for i in range(len(simulations))]
csp = [np.tril(np.corrcoef(k_payoff[:,i,:])) for i in range(len(simulations))]

# for i in cs: mx.showdata(i, symmetry=True, colorbar=True)
# for i in csp: mx.showdata(i, symmetry=True, colorbar=True)


cs_flat  = np.array(cs ).reshape((len(simulations),np.prod(cs[0].shape)))
csp_flat = np.array(csp).reshape((len(simulations),np.prod(csp[0].shape)))

switch_backend('module://matplotlib_inline.backend_inline')
mx.showdata(np.corrcoef(cs_flat.T), symmetry=True, colorbar=True)
mx.showdata(np.corrcoef(csp_flat.T), symmetry=True, colorbar=True)
mx.showdata(np.corrcoef(testvars), symmetry=True, colorbar=True)

mx.showdata(np.corrcoef(np.append(cs_flat.T, testvars,axis=0))[cs_flat.T.shape[0]:,  :cs_flat.T.shape[0]], symmetry=True, colorbar=True)
mx.showdata(np.corrcoef(np.append(csp_flat.T, testvars,axis=0))[csp_flat.T.shape[0]:,:csp_flat.T.shape[0]], symmetry=True, colorbar=True)



CORRS = np.corrcoef(np.append(cs_flat.T, np.append(csp_flat.T, testvars, axis=0),axis=0))

mx.showdata(CORRS,symmetry=True, colorbar=True)

mx.showdata(CORRS[cs_flat.T.shape[0]:(cs_flat.T.shape[0] + csp_flat.T.shape[0]),
                  :cs_flat.T.shape[0]],
            symmetry=True, colorbar=True)


mx.showdata(CORRS[(cs_flat.T.shape[0] + csp_flat.T.shape[0]):,
                  cs_flat.T.shape[0]:(cs_flat.T.shape[0] + csp_flat.T.shape[0])],
            symmetry=True, colorbar=True)

mx.showdata(CORRS[(cs_flat.T.shape[0] + csp_flat.T.shape[0]):,
                  :cs_flat.T.shape[0]],
            symmetry=True, colorbar=True)

mx.showdata(CORRS[-testvars.shape[0]:,
                  -testvars.shape[0]:],
            symmetry=True, colorbar=True)


mx.showdata(cs_flat, symmetry=True, colorbar=True)
mx.showdata(csp_flat,symmetry=True, colorbar=True)

mx.showdata(cs[4], symmetry=True, colorbar=True)

#%%
# import matplotlib
# gui_env = [i for i in matplotlib.rcsetup.interactive_bk] # interactive backends
# non_gui_backends = matplotlib.rcsetup.non_interactive_bk

switch_backend('qt5agg') # or 'qtagg'
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
scatter1=ax.scatter(n_mutu,
                    np.array(sR)-np.array(sR_i),
                    cs_flat[ :,ravelsquare((3,2), cs_flat)],
                    c=modularities, cmap='turbo')
ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y")
# plt.legend(handles=[scatter1, scatter2])
plt.colorbar(scatter1)
plt.show()


#%%


fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
scatter1=ax.scatter(modularities,
                    nodfs,
                    n_mutu,
                    c=np.array(sR)-np.array(sR_i), cmap='turbo')
ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y")
# plt.legend(handles=[scatter1, scatter2])
ax.view_init(elev=30., azim=0)

plt.colorbar(scatter1)
plt.show()





#%%
switch_backend('module://matplotlib_inline.backend_inline')
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


fps=40
nseconds=8

nframes = fps*nseconds
# nframes = 30
azims = np.linspace(0,360,nframes+1)[:-1]
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')

plt.colorbar(scatter1,ax=ax, label='spectral radius variation')

def frame(t):
    print(t/nframes)
    global azims
    ax.clear()
    scatter1=ax.scatter(modularities,
                        nodfs,
                        n_mutu,
                        c=np.array(sR)-np.array(sR_i), cmap='turbo')
    
    ax.set_xlabel(r"modularity", fontsize=16)
    ax.set_ylabel(r"NODF score")
    ax.set_zlabel(r"number of mutualistic interactions")
    ax.view_init(elev=30., azim=azims[t])
    ax.set_title('azimuth = {}°'.format(int(round(azims[t]))))

    
    # plt.show()
    return ax


timecode = str(time.time())
timecode = '_test'
ani = animation.FuncAnimation(fig, frame, frames=nframes, interval=1000/fps, blit=False)
ani.save('../figures/gif/scatter' + timecode +'.gif')



#%%


fps = 40
nseconds = 8

nframes = fps * nseconds
azims = np.linspace(0, 360, nframes + 1)[:-1]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

def frame(t):
    global azims, ax

    ax.clear()

    color = np.array(sR) - np.array(sR_i)
    scatter1 = ax.scatter(modularities, nodfs, n_mutu, c=color, cmap='turbo')

    ax.set_xlabel(r"modularity", fontsize=16)
    ax.set_ylabel(r"NODF score")
    ax.set_zlabel(r"number of mutualistic interactions")
    ax.view_init(elev=30., azim=azims[t])
    ax.set_title('azim = {}'.format(azims[t]))

    # Add colorbar
    cax = make_axes_locatable(ax).append_axes('right', '5%', '5%')
    cbar = plt.colorbar(scatter1, cax=cax)
    cbar.set_label('Colorbar Label')

    return ax

# Create the animation
animation = animation.FuncAnimation(fig, frame, frames=nframes, interval=1000/fps, blit=False)

plt.show()

#%%

unravelsquare(22,cs_flat)
ravelsquare((4,2),cs_flat)
ravelsquare((4,5),csp_flat)

def ravelsquare(ix, fM):
    shape = (int(np.sqrt(fM.shape[1])),int(np.sqrt(fM.shape[1])))    
    return np.ravel_multi_index(ix, shape)

def unravelsquare(ix, fM):
    shape = (int(np.sqrt(fM.shape[1])),int(np.sqrt(fM.shape[1])))    
    return np.unravel_index(ix, shape)

csp_flat[:,ravelsquare((5,0),csp_flat)]
cs_flat[ :,ravelsquare((4,2), cs_flat)]

'''
testvars = np.array([
modularities,
sR,
nodfs,
a,
d,
n_mutu,
n_pred,
n_comp])

k = np.array([
deg,
deg_pr,
deg_q,
btw_c,
eig_c])

k_payoff = np.array([
payoffs_deg_out,
payoffs_deg_in,
payoffs_deg_out_q,
payoffs_deg_in_q,
payoffs_btw_c,
payoffs_eig_c])

'''









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
    
