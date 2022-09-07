#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 14:25:48 2022

@author: roman
"""


from os import chdir
from pathlib import Path
chdir('/home/roman/LAB/ECO') #this line is for Spyder IDE only
root = Path(".")


import networkx as nx
import numpy as np
from scipy import sparse
import pandas as pd
import matplotlib.pyplot as plt


#%% data transform
def npfrompd(pdtab):
	return(np.nan_to_num(pdtab.to_numpy(), 0))

def to_square(m):
	rows, cols = m.shape
	uL = np.zeros((rows,rows))
	dR = np.zeros((cols,cols))
	Um = np.concatenate((uL , m ), axis=1)
	Dm = np.concatenate((m.T, dR), axis=1)
	return(np.concatenate((Um,Dm), axis=0))

def pd_to_square(pdtab):
	m = npfrompd(pdtab)
	return(to_square(m))


def table_to_graph(s_table, binary=False):
		p=s_table
		if binary:
			p[p>0]=1
		full = pd_to_square(p)
		sparse_full = sparse.csr_matrix(full)
		G = nx.from_scipy_sparse_matrix(sparse_full, parallel_edges=False)
		return(G)
    
def array_to_graph(nparray):
    G = nx.from_numpy_array(nparray, parallel_edges=False)
    return(G)
    
def remove_unconnected(npArray):
    ac = npArray.copy()    
    ac = np.delete(ac, np.where(np.all(ac==0,0)), axis=1)
    ac = np.delete(ac, np.where(np.all(ac==0,1)), axis=0)
    return(ac)

def renormalize(vlist):
    x=vlist
    b=np.max(x)
    a=np.min(x)
    x=(x-a)/(b-a)
    return x

#%% network properties
def connectance(adj):
	#adj is an adjacency matrix (or incidence matrix, for bipartite cases)
	nlinks = (adj>0).sum()
	return(nlinks/adj.size)
	'''
	if adj.shape[0] == adj.shape[1]:
		nspecies = adj.shape[0]
		return(nlinks/nspecies**2)
	else:
		raise Exception('non-square array')
	'''

def connectance_list(tablelist):
	conn = list()
	for s_table in tablelist:
		#l_conn.append(connectance(pd_to_square(s_table)))
		conn.append(connectance(npfrompd(s_table)))
	return(np.array(conn))

def clustering_list(g):
	#avg_clust = list()
	#for G in g:
	#	avg_clust.append(nx.bipartite.average_clustering(G))
	avg_clust = list(map(nx.bipartite.average_clustering, g))
	return(avg_clust)

#%% visualization
def showdata(mat, color=plt.cm.gnuplot, symmetry=False):
	mat = np.copy(mat)
	if symmetry:
		top = np.max([np.abs(np.nanmax(mat)),np.abs(np.nanmin(mat))])
		plt.imshow(mat.astype('float32'), interpolation='none', cmap='seismic',vmax=top,vmin=-top)
	else:
		plt.imshow(mat.astype('float32'), interpolation='none', cmap=color)
	plt.colorbar()
	plt.show()
    
#%% 
dataPath = root / 'data/dataBase'
df = pd.read_csv(dataPath / 'FW_017_03.csv', index_col=0)


showdata(pd_to_square(df))
showdata(npfrompd(df))


a=npfrompd(df)
ac=remove_unconnected(a)
g=array_to_graph(ac)


#%% 
nx.draw(nx.maximum_spanning_tree(g), width=0.1, node_size=10)

#%% 
pos = nx.spring_layout(g,scale=0.5)
edges = g.edges()
weights = [g[u][v]['weight'] for u,v in edges]
degrees = np.array(g.degree)[:,1]


palette1=plt.cm.jet #binary jet 
palette2=plt.cm.jet
edgecolors=list(map(plt.cm.colors.rgb2hex,list(map(palette1, renormalize(weights))))) # 1-renormalize(weights)
nodecolors=list(map(plt.cm.colors.rgb2hex,list(map(palette2, renormalize(degrees)))))

nx.draw(g, width=weights*10, pos=pos, edge_color=edgecolors, node_size=0.7*degrees, node_color=nodecolors)
nx.draw(g, width=0.1+0.8*renormalize(weights), pos=pos, edge_color=edgecolors, node_size=0.7*degrees, node_color=nodecolors)

st=nx.maximum_spanning_tree(g)
nx.draw(st, pos=pos,node_size=0)
nx.draw(st, width=0.2, pos=pos, node_size=0.7*degrees, node_color=nodecolors)
#nx.draw(st, width=0.2, node_size=0.7*degrees, node_color=nodecolors)

pos = nx.spring_layout(st,scale=0.5)
nx.draw(st, pos=pos,node_size=0)
nx.draw(st, pos=pos,width=0.2, node_size=0.7*degrees, node_color=nodecolors)
nx.draw(g, width=weights*10, pos=pos, edge_color=edgecolors, node_size=0.7*degrees, node_color=nodecolors)
nx.draw(g, width=0.1+0.8*renormalize(weights), pos=pos, edge_color=edgecolors, node_size=0.7*degrees, node_color=nodecolors)


edges = st.edges()
weights = [st[u][v]['weight'] for u,v in edges]
edgecolors=list(map(plt.cm.colors.rgb2hex,list(map(palette1, renormalize(weights))))) # 1-renormalize(weights)
nx.draw(st, pos=pos,node_size=0,width=0.1+2*renormalize(weights), edge_color=edgecolors)

#%%

#%% BIPARTITE

df = pd.read_csv(dataPath / 'M_PL_058.csv', index_col=0)
nA,nB = df.shape

a=npfrompd(df)
#a=np.random.choice((0,1),(nA,nB))
ac=to_square(a)
ac=remove_unconnected(ac)

showdata(a)
showdata(ac)

g=array_to_graph(ac)

edges = g.edges()
weights = [g[u][v]['weight'] for u,v in edges]
degrees = np.array(g.degree)[:,1]

palette1=plt.cm.jet #binary jet 
palette2=plt.cm.jet
edgecolors=list(map(plt.cm.colors.rgb2hex,list(map(palette1, renormalize(weights))))) # 1-renormalize(weights)
nodecolors=list(map(plt.cm.colors.rgb2hex,list(map(palette2, renormalize(degrees)))))

top = np.array(g.nodes)[:nA]
pos = nx.bipartite_layout(g,top, align='vertical')
nx.draw(g,pos, width=0.1, node_size=1)
nx.draw(g,pos=pos, width=0.05+0.8*renormalize(weights), edge_color=edgecolors, node_size=0.7*degrees, node_color=nodecolors)




