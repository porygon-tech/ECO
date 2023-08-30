#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:12:04 2023

@author: ubuntu
"""


#%% show population sizes
switch_backend('module://matplotlib_inline.backend_inline')

sort_tmp =  sorted(enumerate(simulations), key=lambda x: x[1]['_d'].mean())
sort_simulations = [i_[1] for i_ in sort_tmp]
ixs              = [i_[0] for i_ in sort_tmp]
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
        # plt.xlim(0,4000)
        plt.ylim(0,1000)
        plt.title(str(ixs[i])+': '+str(sim['_d'].mean()))
        plt.ylabel('population size')
        plt.xlabel('time (generations)')
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
            
        #plt.title(str(i)+': '+str(sim['_d'].mean()))
        plt.title(str(ixs[i])+': '+str(sim['_d'].mean()))
        plt.ylabel('trait mean value')
        plt.xlabel('time (generations)')
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
c=0.5
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
N=7
A = mx.nullmodels.clusterchain(N,3)
A = mx.swaplinks(A, 5, connected=True)


#=============================================
#===== PAYOFFS ===============================
#=============================================
N=A.shape[0]
g1,g2 = np.array([-1,1])

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
plt.imshow(A_e, norm=matplotlib.colors.TwoSlopeNorm(vmin=-.01, vcenter=0, vmax=.01),cmap='bwr');plt.show()
#%%
#=============================================
N=7
c=0.4
A = nx.adjacency_matrix(nx.fast_gnp_random_graph(N,c)).todense()
g1,g2 = np.array([-2,1])
A_e = np.random.choice((g1,g2),(N,N))*A

sd(A_e,symmetry=True)
mx.totext(A)
#%%

switch_backend('module://matplotlib_inline.backend_inline')
for i, G in enumerate(Gl_payoff):
    # G = G.to_undirected()
    # nx.draw(G);plt.show() 
    sd(payoffs[i])
#!!!

#%%
plt.figure(figsize=(6, 6))
pos = nx.spring_layout(cycle_graph)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='#FF5555', width=0.2)
nx.draw_networkx_edges(cycle_graph, pos, edgelist=cycBasis, edge_color='black', width=2.0)
plt.title('Graph with Cycle Basis')
plt.show()

#%%



window = 100
np.all(np.abs(np.diff(sim['D'][-3:-1],axis=0))<tol)


tolD =1e-1
tolZ =1e-4
sim = simulations[22]

[np.all([sim['D'][-window-1:-1].var(0)     < tolD,
         sim['dist_avgs'][-window:].var(0) < tolZ]) for sim in simulations]



[np.all([np.abs(np.diff(sim['D'][-window-1:-1],    axis=0).mean(0)) < tolD,
         np.abs(np.diff(sim['dist_avgs'][-window:],axis=0).mean(0)) < tolZ]) for sim in simulations]



np.abs(np.diff(sim['dist_avgs'][-window:],axis=0).mean(0))
 # %% TRASH INCOMING =======================================
 
 # ======================================================
 # ======================================================
 # ======================================================
 
 
#%% figure for poster
switch_backend('module://matplotlib_inline.backend_inline')

fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                    (np.repeat(d,N)),
                    [list(gamma['nodes']['bc'][i].values()) for i in range(len(simulations))],
                    # (np.repeat(gamma['mod'],N)),
                    # [list(gamma['nodes']['power_mutu_delta'][i].values()) for i in range(len(simulations))],
                    # c=(np.repeat(gamma['mod'],N)),
                    # c=(np.repeat(d,N)),
                    c=(np.repeat(gamma['mod'],N)),
                    # c=[list(gamma['nodes']['bc'][i].values()) for i in range(len(simulations))],
                   # c=colors,
                   # norm=matplotlib.colors.LogNorm(),
                    cmap="jet",s=15,alpha=0.8)
ax.set_xlabel(r"d", fontsize=16)
ax.set_ylabel(r"equilibrium population size")

plt.show()
 
#%% figure for poster
switch_backend('module://matplotlib_inline.backend_inline')

# R = np.array([list(gamma['nodes']['power_pred_delta']  [i].values()) for i in range(len(simulations))]).flatten()
# G = np.array([list(gamma['nodes']['power_mutu_delta']  [i].values()) for i in range(len(simulations))]).flatten()
# B = np.array([list(gamma['nodes']['power_comp_delta']  [i].values()) for i in range(len(simulations))]).flatten()

# R = np.array([list(gamma['nodes']['power_pred']  [i].values()) for i in range(len(simulations))]).flatten()
# G = np.array([list(gamma['nodes']['power_mutu']  [i].values()) for i in range(len(simulations))]).flatten()
# B = np.array([list(gamma['nodes']['power_comp']  [i].values()) for i in range(len(simulations))]).flatten()

G = np.array([list(mass['nodes']['strength_mutu']  [i].values()) for i in range(len(simulations))]).flatten()
B = np.array([list(mass['nodes']['strength_comp']  [i].values()) for i in range(len(simulations))]).flatten()
R = np.array([list(mass['nodes']['strength_pred']  [i].values()) for i in range(len(simulations))]).flatten()

colors = mx.graphictools.RGB(R,G,B,same=False,sat=2.9)
colors = 255-colors
colorstr = mx.graphictools.rgb2hex(colors)
colorstr = list(map(mx.graphictools.hex_color_invert_hue, colorstr))

fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                    (np.repeat(d,N)),
                    # (np.repeat(gamma['mod'],N)),
                    [list(payoff['nodes']['bc'][i].values()) for i in range(len(simulations))],
                    # [list(payoff['nodes']['qoutdeg'][i].values()) for i in range(len(simulations))],
                    # [list(mass['nodes']['bc'][i].values()) for i in range(len(simulations))],
                    # 
                    c=colorstr,
                   # norm=matplotlib.colors.LogNorm(),
                    # cmap="jet",s=15,
                    s=100,
                    alpha=0.5)
ax.set_xlabel(r"$\phi$", fontsize=16)
ax.set_ylabel(r"$G_{BC}$", fontsize=16)

plt.show()
#%% figure for poster
switch_backend('module://matplotlib_inline.backend_inline')

# R = np.array([list(gamma['nodes']['power_pred_delta']  [i].values()) for i in range(len(simulations))]).flatten()
# G = np.array([list(gamma['nodes']['power_mutu_delta']  [i].values()) for i in range(len(simulations))]).flatten()
# B = np.array([list(gamma['nodes']['power_comp_delta']  [i].values()) for i in range(len(simulations))]).flatten()

# R = np.array([list(gamma['nodes']['power_pred']  [i].values()) for i in range(len(simulations))]).flatten()
# G = np.array([list(gamma['nodes']['power_mutu']  [i].values()) for i in range(len(simulations))]).flatten()
# B = np.array([list(gamma['nodes']['power_comp']  [i].values()) for i in range(len(simulations))]).flatten()

G = np.array([list(mass['nodes']['strength_mutu']  [i].values()) for i in range(len(simulations))]).flatten()
B = np.array([list(mass['nodes']['strength_comp']  [i].values()) for i in range(len(simulations))]).flatten()
R = np.array([list(mass['nodes']['strength_pred']  [i].values()) for i in range(len(simulations))]).flatten()

colors = mx.graphictools.RGB(R,G,B,same=True,sat=2.9)
colors = 255-colors
colorstr = mx.graphictools.rgb2hex(colors)
colorstr = list(map(mx.graphictools.hex_color_invert_hue, colorstr))

fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                    (np.repeat(d,N)),
                    # (np.repeat(gamma['mod'],N)),
                    # [list(payoff['nodes']['bc'][i].values()) for i in range(len(simulations))],
                    [list(payoff['nodes']['qoutdeg'][i].values()) for i in range(len(simulations))],
                    # [list(mass['nodes']['bc'][i].values()) for i in range(len(simulations))],
                    # 
                    c=colorstr,
                   # norm=matplotlib.colors.LogNorm(),
                    # cmap="jet",s=15,
                    s=100,
                    alpha=0.5)
ax.set_xlabel(r"$\phi$", fontsize=16)
ax.set_ylabel(r"$G_{outdeg}$", fontsize=16)

plt.show()

#%% figure for poster
switch_backend('module://matplotlib_inline.backend_inline')
B = (np.repeat(gamma['mod'],N))
# G = (np.repeat(d,N))
B = np.array([list(gamma['nodes']['power_mutu_delta']  [i].values()) for i in range(len(simulations))]).flatten()
# G = np.array([list(payoff['nodes']['lv_mutu']  [i].values()) for i in range(len(simulations))]).flatten()
G = np.array([list(payoff['nodes']['bc']  [i].values()) for i in range(len(simulations))]).flatten()
# G=np.zeros_like(R)
# B=G

colors = mx.graphictools.RGB(R,G,B,same=False,sat=1)
colorsInv = 255-colors
colorstr = mx.graphictools.rgb2hex(colorsInv)
colorstr = list(map(mx.graphictools.hex_color_invert_hue, colorstr))
#
# colorstr = mx.graphictools.rgb2hex(colors)

fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                    (np.repeat(d,N)),
                    
                    # [list(mass['nodes']['bc'][i].values()) for i in range(len(simulations))],
                    [list(payoff['nodes']['lv_mutu']  [i].values()) for i in range(len(simulations))],
                    # [list(gamma['nodes']['qdeg'][i].values()) for i in range(len(simulations))],
                    # (np.repeat(gamma['mod'],N)),
                    
                    c=colorstr,
                   # norm=matplotlib.colors.LogNorm(),
                    # cmap="jet",s=15,
                    alpha=0.8)
ax.set_xlabel(r"d", fontsize=16)
ax.set_ylabel(r"equilibrium population size")

plt.show()
#%% figure for poster
switch_backend('module://matplotlib_inline.backend_inline')

# R = (np.repeat(gamma['sR'],N))
R = (np.repeat(mass['mod'],N))
# R = (np.repeat(d,N))
G = np.array([list(mass['nodes']['strength_mutu']  [i].values()) for i in range(len(simulations))]).flatten()
# R = np.array([list(gamma['nodes']['power_pred']  [i].values()) for i in range(len(simulations))]).flatten()
# R = np.array([list(mass['nodes']['strength_pred']  [i].values()) for i in range(len(simulations))]).flatten()
# R = np.array([list(payoff['nodes']['lv_mutu']  [i].values()) for i in range(len(simulations))]).flatten()
# B = np.array([list(mass['nodes']['ec']  [i].values()) for i in range(len(simulations))]).flatten()
# G=np.zeros_like(R)
B=R

colors = mx.graphictools.RGB(R,G,B,same=False,sat=1.8)
colorsInv = 255-colors
colorstr = mx.graphictools.rgb2hex(colorsInv)
colorstr = list(map(mx.graphictools.hex_color_invert_hue, colorstr))
#
# colorstr = mx.graphictools.rgb2hex(colors)

fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                    # (np.repeat(d,N)),
                    
                    # [list(gamma['nodes']['popsizes'][i].values()) for i in range(len(simulations))],
                    # [list(payoff['nodes']['lv_mutu']  [i].values()) for i in range(len(simulations))],
                    # [list(mass['nodes']['qdeg'][i].values()) for i in range(len(simulations))],
                    [list(payoff['nodes']['qdeg']  [i].values()) for i in range(len(simulations))],
                    (np.repeat(gamma['mod'],N)),
                    
                    c=colorstr,
                   # norm=matplotlib.colors.LogNorm(),
                    # cmap="jet",s=15,
                    alpha=0.8)
ax.set_xlabel(r"$\phi$", fontsize=16)
ax.set_ylabel(r"node strength$_M$", fontsize=16)

plt.show()
# %%

# %%
# %%

#%% figure for poster
switch_backend('module://matplotlib_inline.backend_inline')

fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                    (np.repeat(d,N)),
                    # (np.repeat(gamma['nodf'],N)),
                    # [list(gamma['nodes']['power_mutu'][i].values()) for i in range(len(simulations))],
                    # [list(payoff['nodes']['ec'][i].values()) for i in range(len(simulations))],
                    [list(mass['nodes']['strength_mutu'][i].values()) for i in range(len(simulations))],
                    # 
                    # (np.repeat(gamma['mod'],N)),
                    # [list(gamma['nodes']['fits'][i].values()) for i in range(len(simulations))],
                    # c=(np.repeat(gamma['mod'],N)),
                    # c=(np.repeat(d,N)),
                    c=(np.repeat(gamma['nodf'],N)),
                    # c=[list(gamma['nodes']['bc'][i].values()) for i in range(len(simulations))],
                   # c=colors,
                   # norm=matplotlib.colors.LogNorm(),
                    cmap="jet",s=20,alpha=0.8)
ax.set_xlabel(r"viridis", fontsize=16)
ax.set_ylabel(r"equilibrium population size")

plt.show()

#%%

fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111)
ax.scatter(
        d,
[np.array(list(mass['nodes']['strength_mutu']  [i].values())).sum() for i in range(len(simulations))],
# [np.array(list(gamma['nodes']['power_mutu']  [i].values())).mean() for i in range(len(simulations))],
c=gamma['mod'],cmap="viridis",s=60
        )
plt.show()
#%%
import matplotlib.pyplot as plt
import numpy as np
switch_backend('module://matplotlib_inline.backend_inline')
# Generate example continuous data for x and y
x_data = np.random.randn(100) - np.linspace(10,-10,100)
y_data = np.random.randn(100) - np.linspace(10,-10,100)

plt.scatter(x_data,y_data);plt.show()

# Discretize the X axis into 3 boxes
num_boxes = 5
x_bins = np.linspace(min(x_data), max(x_data), num_boxes + 1)
x_discretized = np.digitize(x_data, x_bins)

# Create lists to store Y values in each box
y_in_boxes = [[] for _ in range(num_boxes)]
for i, x_bin in enumerate(x_discretized):
    # Ensure x_bin is within the valid range [1, num_boxes]
    if 1 <= x_bin <= num_boxes:
        y_in_boxes[x_bin - 1].append(y_data[i])


# Create a box plot for the distribution of Y values in each box
# plt.boxplot(y_in_boxes, labels=[f'Box {i+1}' for i in range(num_boxes)])

plt.boxplot(y_in_boxes, labels=[f'({np.round(x_bins[i],2)}, {np.round(x_bins[i+1],2)})' for i in range(num_boxes)])
plt.title('Distribution of Y Values in Discretized X Boxes')
plt.xlabel('X Discretized Boxes')
plt.ylabel('Y Values')
plt.show()

def quickboxplot(x_data,y_data,num_boxes = 3):
    # Discretize the X axis into 3 boxes  
    x_bins = np.linspace(min(x_data), max(x_data), num_boxes + 1)
    x_discretized = np.digitize(x_data, x_bins)

    # Create lists to store Y values in each box
    y_in_boxes = [[] for _ in range(num_boxes)]
    for i, x_bin in enumerate(x_discretized):
        # Ensure x_bin is within the valid range [1, num_boxes]
        if 1 <= x_bin <= num_boxes:
            y_in_boxes[x_bin - 1].append(y_data[i])


    # Create a box plot for the distribution of Y values in each box
    # plt.boxplot(y_in_boxes, labels=[f'Box {i+1}' for i in range(num_boxes)])

    plt.boxplot(y_in_boxes, labels=[f'({np.round(x_bins[i],2)}, {np.round(x_bins[i+1],2)})' for i in range(num_boxes)])
    plt.title('Distribution of Y Values in Discretized X Boxes')
    plt.xlabel('X Discretized Boxes')
    plt.ylabel('Y Values')
    plt.show()
