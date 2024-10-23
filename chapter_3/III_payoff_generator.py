#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:12:04 2023

@author: ubuntu
Requires LITE_III_series_analysis and TEMP_corrs to have been ran before in this console
"""


sort_tmp =  sorted(enumerate(simulations), key=lambda x: x[1]['_d'].mean())
sort_simulations = [i_[1] for i_ in sort_tmp]
ixs              = [i_[0] for i_ in sort_tmp]
#%% show population sizes
mx.graphictools.inline_backend(True)

filter_1 = [sim['_d'].mean() > 1 for sim in sort_simulations]
#filter_2 = np.array(power_mutu) >0.1
# filtered_i = np.where(np.logical_and(filter_1,filter_2))[0]
filtered_i = np.arange(len(sort_simulations))

for i, sim in enumerate(sort_simulations):
    if i in filtered_i:
        # R = np.array(list(gamma['nodes']['power_pred']  [i].values()))
        # G = np.array(list(gamma['nodes']['power_mutu']  [i].values()))
        # B = np.array(list(gamma['nodes']['power_comp']  [i].values()))


        # rgblist = np.array([
        # np.nan_to_num(mx.cNorm(mx.renormalize(R),3)*255),
        # np.nan_to_num(mx.cNorm(mx.renormalize(G),3)*255),
        # np.nan_to_num(mx.cNorm(mx.renormalize(B),3)*255)]).astype('int').T

        # colors = ['#%02x%02x%02x' % (r,g,b) for r,g,b in rgblist]
        colors = ["#EEDD11"]*N
        x = np.arange(sim['D'][:-1].shape[0])
        for i__N in range(N):
            plt.plot(x, sim['D'][:-1,i__N], c=colors[i__N])
            
            # plt.plot(x, sim['dist_avgs'][:-1,i__N], c=colors[i__N])
        # plt.xlim(0,4000)
        # plt.ylim(0,simulations[0]['_K'])
        plt.title(str(ixs[i])+': '+str(sim['_d'].mean()))
        plt.ylabel('population size')
        plt.xlabel('time (generations)')
        plt.show()



#%% show average traits
mx.graphictools.inline_backend(True)

filter_1 = [sim['_d'].mean() > 1 for sim in sort_simulations]
#filter_2 = np.array(power_mutu) <0.1
# filtered_i = np.where(np.logical_and(filter_1,filter_2))[0]
filtered_i = np.arange(len(sort_simulations))

for i,sim in enumerate(sort_simulations):
    if i in filtered_i:
        # R = np.array(list(gamma['nodes']['power_pred']  [i].values()))
        # G = np.array(list(gamma['nodes']['power_mutu']  [i].values()))
        # B = np.array(list(gamma['nodes']['power_comp']  [i].values()))

        # rgblist = np.array([
        # np.nan_to_num(mx.cNorm(mx.renormalize(R),3)*255),
        # np.nan_to_num(mx.cNorm(mx.renormalize(G),3)*255),
        # np.nan_to_num(mx.cNorm(mx.renormalize(B),3)*255)]).astype('int').T

        # colors = ['#%02x%02x%02x' % (r,g,b) for r,g,b in rgblist]
        colors = ["#DDEE11"]*N
        x = np.arange(sim['D'][:-1].shape[0])
        for i__N in range(N):
            plt.plot(x, sim['dist_avgs'][:-1,i__N], c=colors[i__N])
            
        #plt.title(str(i)+': '+str(sim['_d'].mean()))
        plt.title(str(ixs[i])+': '+str(sim['_d'].mean())+' a: '+str(sim['_a'].mean()))
        plt.ylabel('trait mean value')
        plt.xlabel('time (generations)')
        plt.show()
#%% show fitnesses
filtered_i = np.arange(len(sort_simulations))

for i,sim in enumerate(sort_simulations):
    if i in filtered_i:
        colors = ["#11DDEE"]*N
        x = np.arange(sim['D'][:-1].shape[0])
        for i__N in range(N):
            plt.plot(x, sim['fits'][:-1,i__N], c=colors[i__N])
            
        #plt.title(str(i)+': '+str(sim['_d'].mean()))
        plt.title(str(ixs[i])+': '+str(sim['_d'].mean()))
        plt.ylabel('average per capita fitness')
        plt.xlabel('time (generations)')
        plt.show()
#%%
i=np.random.randint(N)
i2=1
mat=simulations[i]['v'][:6000,i2]
serie=simulations[i]['dist_avgs'][:,i2]
# popserie = simulations[15]['D'][:,13]
#plt.plot(serie);plt.show()
sd(mx.graphictools.resize_image(mat, (356,200)),color='viridis')
sd(mat)

#plt.plot(mat.T); plt.show()

# plt.plot(popserie);plt.show()
thres = 2

#%%
thres = 0.1e-7
window = 500
s=serie[3000:]


check = np.diff(s)**2
checkio = check<thres
for i in range(window):
    checkio = np.logical_and(checkio, np.roll(checkio,1))

checkio_tmp = np.concatenate((checkio,checkio))
np.all(np.concatenate((checkio,checkio_tmp)))

plt.plot(s)
plt.plot((check<thres)*(s.mean()-s.min())+s.min())
plt.plot(checkio      *(s.mean()-s.min())+s.min())
plt.show()

plt.plot(check)
plt.plot((check<thres)*np.median(np.diff(s)))
plt.show()


s1=serie[1000:3000]
s2=serie[5000:7000]*0+23
s1.shape==s2.shape
plt.plot(np.correlate(s1, s2, mode='full'))
plt.plot(np.correlate(s2, s2, mode='full'))
plt.show()


#%%


# Generate a random array "a" with values between 0 and 1
a = np.random.rand(100)  # You can adjust the size as needed

# Define a condition to check the previous 5 values
condition = (a[:-5] < 0.8) & (a[1:-4] < 0.8) & (a[2:-3] < 0.8) & (a[3:-2] < 0.8) & (a[4:-1] < 0.8)
condition = np.all(a[:-5:-1] < 0.8, axis=0)
# Create an array "b" with boolean values based on the condition
b = np.zeros_like(a, dtype=bool)
b[5:] = condition




#%% visualize dists
simID=47
for i in range(N):
    mat=simulations[simID]['v'][-2000:,i]
    
    mat2 = mx.graphictools.resize_image(mat, (356,200))
    plt.imshow(mat2)
    plt.title(str(i))
    plt.show()


mat=simulations[15]['v'][:2500,8]
mat2 = mx.graphictools.resize_image(mat, (356,200))
sd(mat2,color='jet')
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
ballsize = int(N/3)
A = nx.adjacency_matrix(nx.barbell_graph(ballsize,N - ballsize*2)).todense()
A = mx.swaplinks(A, 5, connected=True) # link swapping in barbell graphs allows to create semi-random bimodular graphs

# 4. newman_watts_strogatz (small world)
A = nx.adjacency_matrix(nx.newman_watts_strogatz_graph(N,4,0.1)).todense()

# 5. LATTICES
A = nx.adjacency_matrix(nx.hexagonal_lattice_graph(2,5)).todense(); N =A.shape[0] # periodic=True

# 6. RANDOM MODULAR (MIKI)
N=25
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
sums_tmp = np.outer(A.sum(1),np.ones(N))
p=(sums_tmp+sums_tmp.T)/(2*N-1)
bool_array = np.random.rand(N,N)<p
# <p: more degree more enemies
# >p: more degree more friends
A_e = np.empty_like(bool_array, dtype=object)
A_e[ bool_array] = g1
A_e[~bool_array] = g2
A_e*=A
sd(A_e,symmetry=True)
sd(p)
mx.totext(A)
plt.imshow(A_e, norm=matplotlib.colors.TwoSlopeNorm(vmin=-.01, vcenter=0, vmax=.01),cmap='bwr');plt.show()

#%% predefined structure (with roles)
N=25
N_producers=10
N_consumers=13
#N_apex=N-N_producers-N_consumers
g = np.array([-1,0.5])


A_e = mx.ecomodels.structured_triple(N,N_producers,N_consumers,g=g,
                                     consumer_comp=False,
                                     consumer_nest=True)

sd(A_e,symmetry=True)
mx.totext(A_e!=0)

#%%
G=nx.from_numpy_array(A)
pos = nx.layout.kamada_kawai_layout(G)
cmap = plt.cm.get_cmap("gnuplot")
node_weights=dict(G.degree())
node_colors = {node: cmap(weight) for node, weight in node_weights.items()}
nx.draw(G,pos=pos,node_color=list(node_colors.values()))
plt.show()

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


pr = [(p-p.T)/2 for p in payoffs]
Mpred=[(p>0)+0 for p in pr]
Mprey=[(p<0)+0 for p in pr]
Mmut=[((p+p.T)/2 > 0)+0 for p in payoffs]

R2 = np.array([(M1*1).sum(1) for M1,M2 in zip(Mpred,gammas)])
G2 = np.array([(M1*1).sum(1) for M1,M2 in zip(Mmut,gammas)])
B2 = np.array([(M1*1).sum(1) for M1,M2 in zip(Mprey,gammas)])



for i, Graph in enumerate(Gl_gamma):
    G_p = Gl_payoff[i]
    G_p.remove_edges_from(nx.selfloop_edges(G_p))
    Graph.remove_edges_from(nx.selfloop_edges(Graph))
    R = R2[i]
    G = G2[i]
    B = B2[i]
    #colors = ['#%02x%02x%02x' % (r,g,b) for r,g,b in (mx.renormalize((R,G,B))*255).astype('int').T]
    colors = mx.graphictools.rgb2hex(mx.graphictools.RGB(R,G,B,same=True, sat=2)) # change sat to explore
    linewidths = mx.renormalize(list(np.array(list(nx.get_edge_attributes(Graph, 'weight').values()))))*2
    #pos=nx.layout.spring_layout(G_p)
    pos=nx.layout.spring_layout(Graph)
    degs = np.array(list(payoff['nodes']['deg'][i].values()))

    fig, ax = plt.subplots()
    nx.draw_networkx(G_p,
                     pos=pos,
                     ax=ax,
                     with_labels=False,
                     width=10, 
                     edge_color='blue',
                     node_color='red',
                     node_size=0)
    
    nx.draw_networkx(Graph,
                     pos=pos,
                     ax=ax,
                     with_labels=False,
                     width=linewidths, 
                     edge_color=linewidths, 
                     node_color=colors,
                     edge_cmap=plt.cm.Wistia,
                     node_size=2* np.sqrt(simulations[i]['D'][-1]))
    


    ax.axis('off')
    fig.set_facecolor('#002233')
    plt.title(i,color='white')
    plt.show()

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
                    cmap="viridis",s=15,alpha=0.8)
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

colors = mx.graphictools.RGB(R,G,B,same=True,sat=3.9)
colors = 255-colors
colorstr = mx.graphictools.rgb2hex(colors)
colorstr = list(map(mx.graphictools.hex_color_invert_hue, colorstr))

fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                    (np.repeat(d,N)),
                    # (np.repeat(gamma['mod'],N)),
                    # [list(payoff['nodes']['bc'][i].values()) for i in range(len(simulations))],
                    # [list(payoff['nodes']['qoutdeg'][i].values()) for i in range(len(simulations))],
                    [list(mass['nodes']['bc'][i].values()) for i in range(len(simulations))],
                    # [list(mass_c['nodes']['deg'][i].values()) for i in range(len(simulations))],
                    
                    # 
                    c=colorstr,
                   # norm=matplotlib.colors.LogNorm(),
                    # cmap="jet",s=15,
                    s=100,
                    alpha=0.7)
ax.set_xlabel(r"$\phi$", fontsize=16)
ax.set_ylabel(r"$BC_{M}$", fontsize=16)

plt.show()

#%%
switch_backend('module://matplotlib_inline.backend_inline')
B = (np.repeat(payoff['nodf'],N))
# R = (np.repeat(gamma['nodf'],N))

# R = (np.repeat(d,N))
B = np.array([list(gamma['nodes']['relative_strength_comp']  [i].values()) for i in range(len(simulations))]).flatten()
G = np.array([list(gamma['nodes']['relative_strength_mutu']  [i].values()) for i in range(len(simulations))]).flatten()
R = np.array([list(gamma['nodes']['relative_strength_pred']  [i].values()) for i in range(len(simulations))]).flatten()


# G = (np.repeat(gamma['nodf'],N))
# # B = np.array([list(mass_c['nodes']['bc']  [i].values()) for i in range(len(simulations))]).flatten()
# R = (np.repeat(mass_c['nodf'],N))
# B=(np.repeat(gamma['sR'],N))
# B = np.array([list(mass['nodes']['qdeg']  [i].values()) for i in range(len(simulations))]).flatten()


# G = np.array([list(payoff['nodes']['lv_mutu']  [i].values()) for i in range(len(simulations))]).flatten()
# R = np.array([list(payoff['nodes']['ec']  [i].values()) for i in range(len(simulations))]).flatten()

# R = np.array([list(mass['nodes']['strength_pred']  [i].values()) for i in range(len(simulations))]).flatten()
# G = np.array([list(mass['nodes']['strength_mutu']  [i].values()) for i in range(len(simulations))]).flatten()
# R = np.array([list(gamma['nodes']['power_pred']  [i].values()) for i in range(len(simulations))]).flatten()
# G = np.array([list(gamma['nodes']['power_mutu']  [i].values()) for i in range(len(simulations))]).flatten()
# G = (np.repeat(mass_c['nodf'],N))
# G = np.array([list(payoff['nodes']['lv_mutu']  [i].values()) for i in range(len(simulations))]).flatten()
# G = np.array([list(gamma['nodes']['qdeg']  [i].values()) for i in range(len(simulations))]).flatten()
# B=np.ones_like(R)
# B=R
# B=R.max()-R

# R=G=B
# R=G
# G = (np.repeat(gamma['sR'],N))
#R=G=B
np.random.seed(5)
noise =(np.random.rand(len(simulations)*N)-.5)*0.01
noise2=(np.random.rand(len(simulations)*N)-.5)*0.01

colors = mx.graphictools.RGB(R,G,B,same=False,sat=1.)

colors[:,2]=0

# colors[:,:2]=0

colorsInv = 255-colors
colorstr = mx.graphictools.rgb2hex(colorsInv)
colorstr = list(map(mx.graphictools.hex_color_invert_hue, colorstr))
#
# colorstr = mx.graphictools.rgb2hex(colors)



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                    (np.repeat(d,N)) + noise,
                    # (np.repeat(gamma['mod'],N)) + noise2,
                    # (np.repeat(payoff['sR'],N)) + noise2,
                    # [list(mass_c['nodes']['bc'][i].values()) for i in range(len(simulations))],
                    # [list(payoff['nodes']['lv_pred']  [i].values()) for i in range(len(simulations))],
                    # [list(payoff['nodes']['deg']  [i].values()) for i in range(len(simulations))],
                    # [list(gamma['nodes']['relative_strength_pred']  [i].values()) for i in range(len(simulations))],
                    # [list(mass['nodes']['qdeg']  [i].values()) for i in range(len(simulations))],
                    [list(gamma['nodes']['popsizes']  [i].values()) for i in range(len(simulations))],
                    # 
                    
                    # c=colorstr,
                    
                    cmap="turbo_r", c= traitvars,
                    # cmap="rainbow_r", c= [list(gamma['nodes']['edgedist']  [i].values()) for i in range(len(simulations))],
                    # cmap="coolwarm_r", c= [list(gamma['nodes']['thetavar']  [i].values()) for i in range(len(simulations))], norm=matplotlib.colors.LogNorm(1),
                    
                    # c=np.array([list(gamma['nodes']['power_pred']  [i].values()) for i in range(len(simulations))]).flatten(),
                    # 
                   # norm=matplotlib.colors.LogNorm(),
                    # cmap="Blues", c= (np.repeat(gamma['mod'],N)),
                    s=50,
                    alpha=0.8)
plt.colorbar(scatter1,ax=ax, label=r"no")
ax.set_xlabel(r"$\phi$", fontsize=16)
# plt.yscale('log')
ax.set_ylabel(r"node strength$_M$", fontsize=16)
timecode = str(time.ctime().replace(' ','_').replace(':','_'))
# imgname='qdeg_vs_phi_'+timecode+'.pdf'
# plt.savefig(img_path / imgname)
plt.show()
#%% DISTINGUISH PREDATORS FROM PREYS (part 2)

# pr = [(p-p.T)/2 for p in payoffs] watch out, this one does only work with symmetric payoffs

pr = [p*((p*p.T)<0) for p in payoffs]

Mpred=[(p>0)+0 for p in pr]
Mprey=[(p<0)+0 for p in pr]
Mmut=[((p+p.T)/2 > 0)+0 for p in payoffs]

# relative gamma spent on each interaction type
# R = np.array([(M1*M2).sum(1)/M2.sum(1) for M1,M2 in zip(Mpred,gammas)]).flatten()
# G = np.array([(M1*M2).sum(1)/M2.sum(1) for M1,M2 in zip(Mmut,gammas)]).flatten()
# B = np.array([(M1*M2).sum(1)/M2.sum(1) for M1,M2 in zip(Mprey,gammas)]).flatten()
vero_pred = np.array([np.nan_to_num(((M*g).sum(1)/g.sum(1)) / (M.sum(1)/(p!=0).sum(1)),0)for M,g,p in zip(Mpred,gammas,payoffs)]).flatten()
vero_mutu = np.array([np.nan_to_num(((M*g).sum(1)/g.sum(1)) / (M.sum(1)/(p!=0).sum(1)),0)for M,g,p in zip(Mmut,gammas,payoffs)]).flatten()
vero_prey = np.array([np.nan_to_num(((M*g).sum(1)/g.sum(1)) / (M.sum(1)/(p!=0).sum(1)),0)for M,g,p in zip(Mprey,gammas,payoffs)]).flatten()
R = vero_pred
G = vero_mutu
B = vero_prey

colors = mx.graphictools.RGB(R,G,B,same=False,sat=2.)
# colors[:,1]=0
colorsInv = 255-colors
colorstr = mx.graphictools.rgb2hex(colorsInv)
colorstr = list(map(mx.graphictools.hex_color_invert_hue, colorstr))

plt.rcParams["figure.figsize"] = (8,7)
plt.scatter(
                    (np.repeat(d,N)) + noise,
                    # [list(mass_c['nodes']['qdeg']  [i].values()) for i in range(len(simulations))],
                    [list(gamma['nodes']['fits']  [i].values()) for i in range(len(simulations))],
                    
    c=colorstr,
    s=50
    )
plt.ylabel(r"species average fitness $\bar{W}$")
plt.xlabel(r"frequency-dependent selection $\phi$")

plt.yscale('log')
# Create a legend with custom labels and dots
legend_labels = ['predator', 'mutualist', 'prey']
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color) for label, color in zip(legend_labels, ['red', 'green', 'blue'])]
plt.legend(handles=legend_handles, title='Payoff roles', loc='lower left')  # Adjust loc as needed
timecode = str(time.ctime().replace(' ','_').replace(':','_'))
imgname='preyfitness'+timecode+'.pdf'
# plt.savefig(img_path / imgname)
plt.show()
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


#%%


fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                    [list(gamma['nodes']['edgedist']  [i].values()) for i in range(len(simulations))],
                    traitvars,

                    # norm=matplotlib.colors.LogNorm(),
                    c=colorstr,
                    # cmap="RdYlGn_r", c=(np.repeat(d,N)),
                    s=50,
                    alpha=0.8)
# plt.colorbar(scatter1,ax=ax, label=r"NODF$_{G}$")
ax.set_xlabel(r"$\phi$", fontsize=16)
plt.ylim(0,0.6)
# plt.xscale('log')
ax.set_ylabel(r"$\sigma^2_z$", fontsize=16)
timecode = str(time.ctime().replace(' ','_').replace(':','_'))
# imgname='qdeg_vs_phi_'+timecode+'.pdf'
# plt.savefig(img_path / imgname)
plt.show()
#%%


fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                    (np.repeat(d,N)) + noise,
                    traitvars,

                    # norm=matplotlib.colors.LogNorm(),
                    # c=colorstr,
                    c=np.array([list(gamma['nodes']['ec']  [i].values()) for i in range(len(simulations))]).flatten(),
                    # c=(np.repeat(gamma['sR'],N)),
                    norm=matplotlib.colors.LogNorm(0.001),
                    s=50,
                    alpha=0.8)
plt.colorbar(scatter1,ax=ax, label=r"EC$_{\Gamma}$")
ax.set_xlabel(r"$\phi$", fontsize=16)
# plt.ylim(0,0.6)
plt.yscale('log')
ax.set_ylabel(r"$\sigma^2_z$", fontsize=16)
timecode = str(time.ctime().replace(' ','_').replace(':','_'))
# imgname='qdeg_vs_phi_'+timecode+'.pdf'
# plt.savefig(img_path / imgname)
plt.show()
#%%

fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                    (np.repeat(d,N)) + noise,
                    [list(gamma['nodes']['fits']  [i].values()) for i in range(len(simulations))],
                    cmap="turbo_r", c= traitvars,
                    # cmap="rainbow_r", c= [list(gamma['nodes']['thetavar']  [i].values()) for i in range(len(simulations))],
                    # norm=matplotlib.colors.LogNorm(1),

                    s=50,
                    alpha=0.8)
plt.colorbar(scatter1,ax=ax, label=r"trait variance $\sigma^2_z$")
plt.yscale('log')
plt.ylabel(r"species average fitness $\bar{W}$")
plt.xlabel(r"frequency-dependent selection $\phi$")
timecode = str(time.ctime().replace(' ','_').replace(':','_'))
imgname='traitvariance_'+timecode+'.pdf'
# plt.savefig(img_path / imgname)
plt.show()

#%%
fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                    (np.repeat(d,N)) + noise,
                    [list(gamma['nodes']['fits']  [i].values()) for i in range(len(simulations))],
                    # cmap="turbo_r", c= traitvars,
                    cmap="coolwarm_r", c= [list(gamma['nodes']['thetavar']  [i].values()) for i in range(len(simulations))],
                    norm=matplotlib.colors.LogNorm(1),

                    s=50,
                    alpha=1)
plt.colorbar(scatter1,ax=ax, label=r"distance from environmental optimum $\theta$")
plt.yscale('log')
plt.ylabel(r"species average fitness $\bar{W}$")
plt.xlabel(r"frequency-dependent selection $\phi$")
timecode = str(time.ctime().replace(' ','_').replace(':','_'))
imgname='thetadist_'+timecode+'.pdf'
# plt.savefig(img_path / imgname)
plt.show()

#%%
fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                    (np.repeat(d,N)) + noise,
                    [list(gamma['nodes']['fits']  [i].values()) for i in range(len(simulations))],
                    # cmap="turbo_r", c= traitvars,
                    cmap="rainbow_r", c= [list(gamma['nodes']['edgedist']  [i].values()) for i in range(len(simulations))],
                    # norm=matplotlib.colors.LogNorm(1),

                    s=50,
                    alpha=1)
plt.colorbar(scatter1,ax=ax, label=r"distance from extreme phenotypes")
plt.yscale('log')
plt.ylabel(r"species average fitness $\bar{W}$")
plt.xlabel(r"frequency-dependent selection $\phi$")
timecode = str(time.ctime().replace(' ','_').replace(':','_'))
imgname='extremedist_'+timecode+'.pdf'
# plt.savefig(img_path / imgname)
plt.show()




#%% figure for poster
switch_backend('module://matplotlib_inline.backend_inline')

# R = (np.repeat(gamma['sR'],N))
# R = (np.repeat(mass['mod'],N))
# R = (np.repeat(d,N))
G = np.array([list(mass['nodes']['strength_mutu']  [i].values()) for i in range(len(simulations))]).flatten()
# G = np.array([list(gamma['nodes']['power_pred']  [i].values()) for i in range(len(simulations))]).flatten()
R = np.array([list(mass['nodes']['strength_pred']  [i].values()) for i in range(len(simulations))]).flatten()
# R = np.array([list(payoff['nodes']['lv_mutu']  [i].values()) for i in range(len(simulations))]).flatten()
# R = np.array([list(mass['nodes']['bc']  [i].values()) for i in range(len(simulations))]).flatten()
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
                    (np.repeat(gamma['sR'],N)),
                    # [list(gamma['nodes']['power_pred'][i].values()) for i in range(len(simulations))],
                    # [list(payoff['nodes']['lv_mutu']  [i].values()) for i in range(len(simulations))],
                    [list(mass['nodes']['bc'][i].values()) for i in range(len(simulations))],
                    # 1-np.array([list(payoff['nodes']['lv_mutu']  [i].values()) for i in range(len(simulations))]),
                    # [list(mass['nodes']['qdeg'][i].values()) for i in range(len(simulations))],
                    # [list(payoff['nodes']['qdeg']  [i].values()) for i in range(len(simulations))],
                    # (np.repeat(gamma['mod'],N)),
                    
                    c=colorstr,
                   # norm=matplotlib.colors.LogNorm(),
                    # cmap="jet",s=15,
                    alpha=0.8)
ax.set_xlabel(r"$\phi$", fontsize=16)
ax.set_ylabel(r"node strength$_M$", fontsize=16)

plt.show()

#%%

'''
data/obj/SIMULATIONS_any_ground_Wed_Nov_22_160013_2023.obj
data/obj/SIMULATIONS_stupendous_yard_Wed_Nov_22_155753_2023.obj
data/obj/SIMULATIONS_impressionable_capital_Wed_Nov_22_155451_2023.obj
'''
filename1 = 'SIMULATIONS_any_ground_Wed_Nov_22_160013_2023.obj'
filename2 = 'SIMULATIONS_ragged_extreme_Wed_Nov_22_164359_2023.obj'

print("LOADING " + filename1)
with bz2.BZ2File(obj_path / filename1, 'rb') as f:
	simulations1 = pickle5.load(f)

print("LOADING " + filename2)
with bz2.BZ2File(obj_path / filename2, 'rb') as f:
	simulations2 = pickle5.load(f)
    
adjs1  , mutu  , comp  , pred  , gammas1   = evo.getADJs(simulations1, t=-1,return_gammas=True)
adjs2  , mutu  , comp  , pred  , gammas2   = evo.getADJs(simulations2, t=-1,return_gammas=True)#; del(simulations2)

#%%

Gl_gamma1 = [nx.from_numpy_array(M) for M in gammas1]
Gl_gamma2 = [nx.from_numpy_array(M) for M in gammas2]
Gl_mass1 = [nx.from_numpy_array(M) for M in adjs1]
Gl_mass2 = [nx.from_numpy_array(M) for M in adjs2]

gamma1_mod = np.array(mx.mod( Gl_gamma1 ))
gamma2_mod = np.array(mx.mod( Gl_gamma2 ))
mass1_mod  = np.array(mx.mod( Gl_mass1 ))
mass2_mod  = np.array(mx.mod( Gl_mass2 ))

fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                    d,
                    gamma1_mod
)
scatter2=ax.scatter(
                    d,
                    gamma2_mod
)
ax.set_xlabel(r"$\phi$", fontsize=16)
ax.set_ylabel(r"mod$_M$", fontsize=16)

plt.show()

# %%


fig = plt.figure(figsize=(6,4)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                    d,
                    mass['mod'],
                    c=np.array([list(mass['nodes']['strength_mutu']  [i].values()) for i in range(len(simulations))]).sum(1),
                   # norm=matplotlib.colors.LogNorm(),
                    cmap="viridis",s=50,
                    alpha=0.8)
ax.set_xlabel(r"$\phi$", fontsize=16)
ax.set_ylabel(r"node strength$_M$", fontsize=16)

plt.show()

# %%


fig = plt.figure(figsize=(6,4)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                   
# np.array([list(mass['nodes']['strength_mutu']  [i].values()) for i in range(len(simulations))]).flatten(),

(np.repeat(gamma['mod'],N)),
                    # c=np.array([list(mass['nodes']['strength_mutu']  [i].values()) for i in range(len(simulations))]),
                   # norm=matplotlib.colors.LogNorm(),
                   [list(mass['nodes']['bc'][i].values()) for i in range(len(simulations))],
                   c=np.array([list(gamma['nodes']['power_pred']  [i].values()) for i in range(len(simulations))]).flatten(),
                    cmap="viridis",s=50,
                    alpha=0.8)
ax.set_xlabel(r"$\phi$", fontsize=16)
ax.set_ylabel(r"node strength$_M$", fontsize=16)
# plt.xscale('log')
# plt.yscale('log')
plt.show()
# %%

#%% figure for poster
switch_backend('module://matplotlib_inline.backend_inline')

fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111)
scatter1=ax.scatter(
                    [list(mass['nodes']['strength_pred']  [i].values()) for i in range(len(simulations))],
                    [list(mass['nodes']['bc'][i].values()) for i in range(len(simulations))],
                    c=(np.repeat(d,N)),

                    cmap="viridis",s=50,alpha=0.8)
ax.set_xlabel(r"viridis", fontsize=16)
ax.set_ylabel(r"equilibrium population size")
# plt.xscale('log')
plt.yscale('log')
plt.show()
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
    x_data=np.array(x_data).flatten()
    y_data=np.array(y_data).flatten()
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
