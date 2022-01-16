#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 18:27:42 2021

@author: mroman
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

def showdata(mat, color=plt.cm.gnuplot, symmetry=False):
    mat = np.copy(mat)
    if symmetry:
        top = np.max([np.abs(np.nanmax(mat)),np.abs(np.nanmin(mat))])
        plt.imshow(mat.astype('float32'), interpolation='none', cmap='seismic',vmax=top,vmin=-top)
    else:
        plt.imshow(mat.astype('float32'), interpolation='none', cmap=color)
    plt.colorbar()
    plt.show()


def norm(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def fitness(x):
    '''
    fitness landscape function (1 trait)
    '''
    return (norm(x,1.855,0.1) + norm(x,2.145, 0.1))/2

def sexualpreference(x,y,k=1):
    return 1/(1+(x-y)**2/k) #preferring similar phenotypes



#b = np.random.choice((0,1),(m,n),p=(0.7,0.3))

nindivs = 1000
nloci = 10
phenoSpace = [1, 3]
skew=0.5
population = np.random.choice((0,1),(nindivs,nloci), p=(skew, 1-skew))
#==================================================================










#==================================================================

fig = plt.figure(); ax = fig.add_subplot(111)
n=100
x=np.linspace(phenoSpace[0], phenoSpace[1], nloci)
#y=norm(x,0,1)
y=fitness(x)

ax.plot(x,y)
#ax.legend(loc='center right')
ax.set_xlabel('x', labelpad=10)
ax.set_ylabel('y', labelpad=10)
ax.set_xlim(phenoSpace)
#ax.set_ylim([0,1])
plt.show()

#- - - - - -

fig = plt.figure(); ax = fig.add_subplot(111)
n=100
x=np.linspace(phenoSpace[0], phenoSpace[1], nloci)
#y=norm(x,0,1)
y=fitness(x)

ax.plot(x,y)
#ax.legend(loc='center right')
ax.set_xlabel('x', labelpad=10)
ax.set_ylabel('y', labelpad=10)
ax.set_xlim(phenoSpace)
ax.set_ylim([0,1])
plt.show()

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

phenotypes = population.sum(axis=1)/nloci
phenotypes = phenotypes * np.diff(phenoSpace)[0] + phenoSpace[0]

n, bins, patches = plt.hist(phenotypes, 20, density=True, facecolor='g', alpha=0.75)
plt.xlim(phenoSpace)
plt.grid(True)
plt.show()

init_phenotypes = phenotypes

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#==================================================================



#==================================================================

def pois(x,l=1):
    return (l**x * np.math.exp(-l)) / np.math.factorial(x)


def makeChildren(population):
    nindivs, nloci = population.shape
    phenotypes = population.sum(axis=1)/nloci
    phenotypes = phenotypes * np.diff(phenoSpace)[0] + phenoSpace[0]
    relativeFitnessValues = fitness(phenotypes)/fitness(phenotypes).sum()
    children = np.zeros((nindivs,nloci))
    for i in range(nindivs):
        ncrossovers = 1+np.random.poisson() #at least 1 crossover is forced to happen
        #ncrossovers = 1+np.random.poisson(2)
        parents = np.random.choice(nindivs, 2, replace=False ,p=relativeFitnessValues)
        crossover_pos = np.random.choice(range(1,nloci-2), ncrossovers)
        crossover_pos = np.sort(crossover_pos)
        crossover_pos = np.append(crossover_pos, nloci-1)
        start = 0
        parentSwitch = 0
        for end in crossover_pos:
            children[i,start:end] = population[parents[parentSwitch], start:end]
            start = end
            if parentSwitch == 0:
                parentSwitch = 1
            else:
                parentSwitch = 0
    return(children)





def makeChildrenwLove(population,k, mutRate=0):
    nindivs, nloci = population.shape
    phenotypes = population.sum(axis=1)/nloci
    phenotypes = phenotypes * np.diff(phenoSpace)[0] + phenoSpace[0]
    relativeFitnessValues = fitness(phenotypes)/fitness(phenotypes).sum()
    children = np.zeros((nindivs,nloci))
    pA = np.array([phenotypes]*nindivs).flatten()
    pB = np.repeat(phenotypes, nindivs)
    pref=sexualpreference(pA,pB, k) 
    pref=pref.reshape((nindivs,nindivs))
    np.fill_diagonal(pref,0)
    pref = pref/np.sum(pref)
    #pref=np.ones((nindivs,nindivs))
    #
    rc = resourcecompetition(pA,pB,0.005)
    rc=rc.reshape((nindivs,nindivs))
    np.fill_diagonal(rc,0)
    sat = np.sum(rc,axis=0)/np.sum(rc) #saturation
    P = ((pref * relativeFitnessValues*sat**2).T * relativeFitnessValues*sat**2).T
    P = P/np.sum(P)
    #
    #P = ((pref*relativeFitnessValues).T * relativeFitnessValues).T
    #P = P/np.sum(P)
    Pf = P.flatten()[:] 
    parents = np.random.choice(nindivs**2, nindivs, p=Pf)
    parents = np.array(np.unravel_index(parents, (nindivs,nindivs)))
    for i in range(nindivs):
        ncrossovers = 1+np.random.poisson() #at least 1 crossover is forced to happen
        crossover_pos = np.random.choice(range(1,nloci-2), ncrossovers)
        crossover_pos = np.sort(crossover_pos)
        crossover_pos = np.append(crossover_pos, nloci-1)
        start = 0
        parentSwitch = 0
        for end in crossover_pos:
            children[i,start:end] = population[parents[parentSwitch,i], start:end]
            start = end
            if parentSwitch == 0:
                parentSwitch = 1
            else:
                parentSwitch = 0
    return(children)



def mutate(population, m=0.001):
    nindivs, nloci = population.shape
    mutations = np.random.choice((0,1),(nindivs,nloci), p=(1-m, m))
    return (np.logical_xor(mutations, population))

#phenotypes = np.random.choice(phenotypes,nindivs,p=fitness(phenotypes)/fitness(phenotypes).sum())
'''
podiumsize = 5
winner_IDs = np.random.choice(nindivs, podiumsize ,p=fitness(phenotypes)/fitness(phenotypes).sum())
parents = population[winner_IDs, :]
'''




#=================================================================================


nindivs = 100
nloci = 100
phenoSpace = [1, 3]
skew=0.4999
population = np.random.choice((0,1),(nindivs,nloci), p=(1-skew, skew))


def fitness(x):
    '''
    fitness landscape function (1 trait)
    '''
    return (norm(x, 1.855, 0.1) + norm(x, 2.145, 0.1001))/2

'''
def sexualpreference(x,y,k=1):
    return 1+np.sin(x*np.pi/2)*np.sin(y*np.pi/2) #preferring similar phenotypes
'''

ancestors=population

for generation in range(100):
    print(generation)
    children = makeChildren(population)
    population = children




fig = plt.figure(); ax = fig.add_subplot(111)
n=100
x=np.linspace(phenoSpace[0], phenoSpace[1], nloci)
#y=norm(x,0,1)
y=fitness(x)
ax.plot(x,y)

phenotypes = population.sum(axis=1)/nloci
phenotypes = phenotypes * np.diff(phenoSpace)[0] + phenoSpace[0]
ax.hist(phenotypes, 20, density=True, facecolor='r', alpha=0.5)

phenotypes = ancestors.sum(axis=1)/nloci
phenotypes = phenotypes * np.diff(phenoSpace)[0] + phenoSpace[0]
ax.hist(phenotypes, 20, density=True, facecolor='g', alpha=0.5)

#ax.legend(loc='center right')
ax.set_xlabel('x', labelpad=10)
ax.set_ylabel('y', labelpad=10)
ax.set_xlim(phenoSpace)
#ax.set_ylim([0,1])
plt.show()





#////////////////////////////////////////////////////////////////////
pA = np.array([phenotypes]*nindivs).flatten()
pB = np.repeat(phenotypes, nindivs)
pref=sexualpreference(pA,pB, 0.005) 
pref=pref.reshape((100,100))
np.fill_diagonal(pref,0)
pref = pref/np.sum(pref)

P = ((pref*relativeFitnessValues).T * relativeFitnessValues).T
P = P/np.sum(P)

#showdata(pref)
showdata(P)

Pf = P.flatten()[:] 
parents = np.random.choice(nindivs**2, nindivs, p=Pf)
np.unravel_index(parents, (nindivs,nindivs))

mutRate = 0.05
mutations = np.random.choice((0,1),(nindivs,nloci), p=(1-mutRate, mutRate))
showdata(mutations)
population = np.logical_xor(mutations, population)

#////////////////////////////////////////////////////////////////////





#////////////////////////////////////////////////////////////////////
k=0.0005
nindivs, nloci = population.shape
phenotypes = population.sum(axis=1)/nloci
phenotypes = phenotypes * np.diff(phenoSpace)[0] + phenoSpace[0]
relativeFitnessValues = fitness(phenotypes)/fitness(phenotypes).sum()
pA = np.array([phenotypes]*nindivs).flatten()
pB = np.repeat(phenotypes, nindivs)

pref=sexualpreference(pA,pB, k) 
pref=pref.reshape((nindivs,nindivs))
np.fill_diagonal(pref,0)
pref = pref/np.sum(pref)

rc = resourcecompetition(pA,pB,k=0.005)
rc=rc.reshape((nindivs,nindivs))
np.fill_diagonal(rc,0)
sat = np.sum(rc,axis=0)/np.sum(rc) #saturation


P = ((pref*relativeFitnessValues*sat).T * relativeFitnessValues*sat).T
P = P/np.sum(P)

showdata(P)

Pf = P.flatten()[:] 
parents = np.random.choice(nindivs**2, nindivs, p=Pf)
parents = np.array(np.unravel_index(parents, (nindivs,nindivs)))


fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(phenotypes[parents[0]],phenotypes[parents[1]], 'o')
ax.set_xlim(phenoSpace)
ax.set_ylim(phenoSpace)
plt.show()
#////////////////////////////////////////////////////////////////////


ncrossovers = 1+np.random.poisson() #at least 1 crossover is forced to happen
crossover_pos = np.random.choice(range(1,nloci-2), ncrossovers)
crossover_pos = np.sort(crossover_pos)
crossover_pos = np.append(crossover_pos, nloci-1)
start = 0
parentSwitch = 0
children = np.zeros((nindivs,nloci))
i=2
for end in crossover_pos:
    children[i,start:end] = population[parents[parentSwitch,i], start:end]
    start = end
    if parentSwitch == 0:
        parentSwitch = 1
    else:
        parentSwitch = 0




for generation in range(1):
    print(generation)
    children = makeChildrenwLove(population,k)
    children = mutate(children,0.01)
    population = children
























nindivs = 1000
nloci = 100
phenoSpace = [1, 3]
skew=0.4999




'''
def sexualpreference(x,y,k=1):
    return 1+np.sin(x*np.pi/2)*np.sin(y*np.pi/2) #preferring similar phenotypes
'''
def sexualpreference(x,y,k=1):
    return 1/(1+(x-y)**2/k) #preferring similar phenotypes

def resourcecompetition(x,y,k=1):
    return 1-1/(1+(x-y)**2/k) #competing similar phenotypes


def fitness(x):
    '''
    fitness landscape function (1 trait)
    '''
#    return (norm(x, 1, 0.8) + norm(x, 3, 0.8))/4
    return (norm(x, 1.3, 0.1) + norm(x, 2.145, 0.1001))/2


fig = plt.figure(); ax = fig.add_subplot(111)
n=100
x=np.linspace(phenoSpace[0], phenoSpace[1], nloci)
y=fitness(x)
ax.plot(x,y)
ax.set_xlim(phenoSpace)
plt.show()
#---------------------------

skew = 0.35
population = np.random.choice((0,1),(nindivs,nloci), p=(1-skew, skew))
ancestors=population





immigrate = 400
skew_imm = 0.2
#population[0:immigrate] = np.zeros(nloci) 
population[0:immigrate] = np.random.choice((0,1),(immigrate,nloci), p=(1-skew_imm, skew_imm))
#showdata(population)
ancestors=population

for generation in range(10):
    print(generation)
    children = makeChildrenwLove(population,0.0005, mutRate=0.01) 
    children = mutate(children)
    population = children


fig = plt.figure(); ax = fig.add_subplot(111)
n=100
x=np.linspace(phenoSpace[0], phenoSpace[1], nloci)
#y=norm(x,0,1)
y=fitness(x)
ax.plot(x,y)
phenotypes = population.sum(axis=1)/nloci
phenotypes = phenotypes * np.diff(phenoSpace)[0] + phenoSpace[0]
ax.hist(phenotypes, 20, density=True, facecolor='r', alpha=0.5)
phenotypes = ancestors.sum(axis=1)/nloci
phenotypes = phenotypes * np.diff(phenoSpace)[0] + phenoSpace[0]
ax.hist(phenotypes, 20, density=True, facecolor='g', alpha=0.5)
ax.set_xlim(phenoSpace)
#ax.set_ylim([0,1])
plt.show()










