from scipy.special import comb  
import numpy as np

import matplotlib.pyplot as plt
#from itertools import product  

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

def samplecolors(n, type='hex',palette=plt.cm.gnuplot):
    if type == 'hex':
        return list(map(plt.cm.colors.rgb2hex, list(map(palette, np.linspace(1,0,n)))))
    elif type == 'rgba':
        return list(map(palette, np.linspace(1,0,n)))

def showlist(l, distbins=False):
            fig = plt.figure(); ax = fig.add_subplot(111)
            ax.plot(np.arange(len(l)),list(l))
            plt.show()

def generateBinTable(n):
    table = np.zeros((1,n))
    for i in range(0, 2**n):
        code = format(i,'b')
        code = np.array([int(c) for c in code])
        code = np.append(np.zeros(n - len(code)), code)[:,np.newaxis]
        table = np.append(table,code.T, axis=0)
    return table[1:,:]

def bindist(n,k,p=0.5):
    return comb(n,k)*p**k*(1-p)**(n-k)

def p(v,n,parent):
    sumvar=0
    for i in range(0,n+1):
        for k in range(0,n-i+1):
            for x in range(max(0,2*i+k-n),i+1):
                sumvar += comb(i, x)*comb(n - i, i + k - x)/comb(n, i + k)*comb(2*(i - x) + k, v - x) * (1/2)**(2*(i-x)+k) #* parent[i] * parent[i+k]
    return sumvar

def pOverlaps(n,i,k,x):
    return comb(i,x) * comb(n - i, i + k - x) / comb(n,i+k)


#%%
#==================================================



(m,n) = (20,10)

v=5
skw=0.4





parent = []
for j in range(n+1):
    parent.append(bindist(n,j,skw))

child = []
for j in range(n+1):
    child.append(p(j,n,parent))

sum(parent) 
sum(child) 

#%%==================================================0


n=10
i=5
k=1
x=3
states = generateBinTable(n)
#np.sum(np.sum(states,1) == v)
#comb(n,v)

sPi  = states[np.where(np.sum(states,1) == i)]
sPik = states[np.where(np.sum(states,1) == i+k)]

nOverlaps=np.zeros((sPi.shape[0],sPik.shape[0]))
for a in range(sPi.shape[0]):
    for b in range(sPik.shape[0]):
        pi  = sPi[a,:]
        pik = sPik[b,:]
        nOverlaps[a,b] = np.sum(np.logical_and(pi,pik))

#showdata(nOverlaps)
np.sum(nOverlaps==x)
comb(i,x) * comb(n - i, i + k - x) * comb(n,i)

nOverlaps.size 
comb(n,i) * comb(n,i+k)

comb(i,x) * comb(n - i, i + k - x) / comb(n,i+k)

#%%==================================================1

varsites=np.zeros((sPi.shape[0],sPik.shape[0]))
check = varsites.copy()
varsites = varsites.astype('int')
for a in range(sPi.shape[0]):
    for b in range(sPik.shape[0]):
        pi  = sPi[a,:]
        pik = sPik[b,:]
        varsites[a,b] = np.sum(np.logical_xor(pi,pik))
        check[a,b] = (2*(i-nOverlaps[a,b])+k)

#showdata(varsites)

#%%==================================================2

#np.unique(varsites[np.where(nOverlaps == x)])  

overLpositions = np.where(nOverlaps == x)
rc = np.random.randint(overLpositions[0].size)
a = overLpositions[0][rc]
b = overLpositions[1][rc]

pi  = sPi[a,:].astype(int)
pik = sPik[b,:].astype(int)


ntries = 6000
skw=0.5
childPhens = []
for _ in range(ntries):
    recomb = np.random.choice((0,1),n,(1-skw,skw))
    child = np.logical_or(np.logical_and(recomb,pi),np.logical_and(np.logical_not(recomb),pik))+0
    childPhens.append(np.sum(child))

unique, counts = np.unique(childPhens, return_counts=True)
np.asarray((unique, counts/ntries)).T

parent = []
for j in range(varsites[a,b]+1):
    parent.append(bindist(varsites[a,b],j))

parent

counts/ntries-parent
#%%==================================================3





n=10
i=5
k=1
x=3

parent = []
for j in range(n+1):
    parent.append(bindist(n,j,0.99))

#--- x
sumvar=0
for x in range(max(0,2*i+k-n),i+1):
    sumvar += pOverlaps(n,i,k,x)

np.round(sumvar,4)

#--- i,j

checkgrid=np.zeros((n+1,n+1))
sumvar=0
for i in range(0,n+1):
    #for k in range(0,n-i+1):
    for j in range(0,n+1):
        for x in range(0,i+1):
            #sumvar += pOverlaps(n,i,k,x) * parent[i] * parent[i+k]
            #checkgrid[(i,i+k),(i+k,i)] = parent[i] * parent[i+k]
            checkgrid[(i,j),(j,i)] = parent[i] * parent[j]
            sumvar += pOverlaps(n,min(i,j),max(i,j)-min(i,j),x) * parent[i] * parent[j]

np.sum(checkgrid)
np.round(sumvar,4)

#--- v

sumvar=0
for v in range(0,n+1):
    for i in range(0,n+1):
        #for k in range(0,n-i+1):
        for j in range(0,n+1):
            for x in range(0,i+1):
                #sumvar += pOverlaps(n,min(i,j),max(i,j)-min(i,j),x) * parent[i] * parent[j] * comb(2*(i - x) + k, v - x) * (1/2)**(2*(i-x)+k)
                #sumvar += pOverlaps(n,min(i,j),max(i,j)-min(i,j),x) * parent[i] * parent[j] * comb(i - 2*x + j, v - x) * (1/2)**(2*(i-x)+k)
                sumvar += pOverlaps(n,min(i,j),max(i,j)-min(i,j),x) * parent[i] * parent[j] * bindist(i - 2*x + j, v - x)

np.round(sumvar,4)

#--- fitness

def f(i):
    return 1+n-i


fitness=np.zeros(n+1)
for i in range(0,n+1):
    fitness[i] = f(i)

fitness /= fitness.sum()

w=parent*fitness
w /= w.sum()

fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(np.arange(n+1),parent,label='parent frequency')
ax.plot(np.arange(n+1),w,label='fitness reweighted')
ax.legend()
plt.show()

sumvar=0
for v in range(0,n+1):
    for i in range(0,n+1):
        #for k in range(0,n-i+1):
        for j in range(0,n+1):
            for x in range(0,i+1):
                #sumvar += pOverlaps(n,min(i,j),max(i,j)-min(i,j),x) * bindist(i - 2*x + j, v - x) * w[i] * w[j]
                sumvar += pOverlaps(n,i,j-i,x) * bindist(i - 2*x + j, v - x) * w[i] * w[j]

np.round(sumvar,4)


#%%=========================================================================


n=20
skw=0.6
def f(i):
    #return 1+n-i
    #return 2**(-i/2)

#------

fitness=np.zeros(n+1)
for i in range(0,n+1):
    fitness[i] = f(i)

#showlist(fitness)

parent = []
for i in range(n+1):
    parent.append(bindist(n,i,skw))


#showlist(parent)

fitness /= fitness.sum()
w=parent*fitness
w /= w.sum()


child = np.zeros((n+1))
for v in range(0,n+1):
    for i in range(0,n+1):
        #for k in range(0,n-i+1):
        for j in range(0,n+1):
            for x in range(0,i+1):
                child[v] += pOverlaps(n,i,j-i,x) * bindist(i - 2*x + j, v - x) * w[i] * w[j]
                


fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(np.arange(n+1),parent,label='parents frequency')
ax.plot(np.arange(n+1),child, label='children frequency')
ax.legend()
plt.show()

#%%======================================= MULTI GEN

ngenerations = 20
n=50
skw=0.3
def f(i):
    return 1+n-i
    #return 2**(-i/2)

parent = []
for i in range(n+1):
    #parent.append(bindist(n,i,skw))
    parent.append(1/(n+1))

#------

genData=np.zeros((ngenerations, n+1))
genData[0,:] = parent.copy()

fitness=np.zeros(n+1)
for i in range(0,n+1):
    fitness[i] = f(i)

fitness /= fitness.sum()


for g in range(1,ngenerations):
    print('generation {0}'.format(g))
    w=parent*fitness
    w /= w.sum()
    child = np.zeros((n+1))
    for v in range(0,n+1):
        for i in range(0,n+1):
            #for k in range(0,n-i+1):
            for j in range(0,n+1):
                for x in range(0,i+1):
                    child[v] += pOverlaps(n,i,j-i,x) * bindist(i - 2*x + j, v - x) * w[i] * w[j]
    genData[g,:] = child.copy()
    parent = child.copy()



clist = samplecolors(ngenerations)
fig = plt.figure(); ax = fig.add_subplot(111)
for g in range(ngenerations):
    ax.plot(np.arange(n+1),genData[g,:],label='generation {0}'.format(g),color=clist[g])

plt.show()

showdata(genData, color='jet')



height, width = genData.shape
xi = np.linspace(1, width, width)
yi = np.linspace(1, height, height)
axx, axy = np.meshgrid(xi, yi)

fig = plt.figure(figsize =(14, 9)); ax = fig.add_subplot(projection='3d')
surf = ax.plot_trisurf(axx.flatten(),axy.flatten(),genData.flatten(), cmap=plt.cm.jet, linewidth=0)
ax.set_zlim3d(0,0.5)
ax.set_xlim3d(0,20)
ax.set_xlabel('phenotypic value')
ax.set_ylabel('time')
ax.set_zlabel('frequency')
fig.colorbar(surf)
plt.show()

fig = plt.figure(figsize =(14, 9)); ax = plt.axes(projection ='3d') 
surf = ax.plot_surface(axx,axy,genData, cmap=plt.cm.jet, edgecolor ='none')
ax.set_zlim3d(0,0.5)
fig.colorbar(surf, ax = ax, shrink = 0.7, aspect = 7) 
plt.show() 

#%%======================================= STATISTICAL DATA COMPARISON

ngenerations = 40
n=15
m=1000

skw=0.6
def f(i):
    #return 1+n-i
    #return 2**(-i/2)
    return n**2-i**2

def f2(i):
    return 1+i
    #return 2**(-i/2)

parent = []
for i in range(n+1):
    parent.append(bindist(n,i,skw))
    #parent.append(1/(n+1))

#------

mtx = np.random.choice((0,1),(m,n), p=(1-skw, skw))

v_genData=np.zeros((ngenerations, n+1))
v_phen = mtx.sum(axis=1)
unique, counts = np.unique(v_phen, return_counts=True)
np.asarray((unique, counts/m)).T
v_genData[0,(unique).astype(int)] = counts/m

for g in range(1,ngenerations):
    print('generation {0}'.format(g))
    v_phen = mtx.sum(axis=1)
    v_fitn = (f(v_phen)/f(v_phen).sum())[:,np.newaxis]
    #
    #if g>40:
    #    v_fitn = (f2(v_phen)/f2(v_phen).sum())[:,np.newaxis]
    #
    v_pmatch = (v_fitn*v_fitn.T).flatten()[:]
    v_couples = np.random.choice(m**2, m, p=v_pmatch)
    v_couples = np.array(np.unravel_index(v_couples, (m,m)))
    mtx_child = np.zeros((m,n))
    for i in range(m):
        recomb = np.random.choice((0,1),n)
        pA = mtx[v_couples[0,i],:]
        pB = mtx[v_couples[1,i],:]
        mtx_child[i,:] = np.logical_or(np.logical_and(recomb,pA),np.logical_and(np.logical_not(recomb),pB))+0
    #
    v_phen_child = mtx_child.sum(axis=1)
    unique, counts = np.unique(v_phen_child, return_counts=True)
    #np.asarray((unique, counts/m)).T
    v_genData[g,(unique).astype(int)] = counts/m
    mtx = mtx_child.copy()

showdata(v_genData, color='jet')


#------

genData=np.zeros((ngenerations, n+1))
genData[0,:] = parent.copy()

fitness=np.zeros(n+1)
for i in range(0,n+1):
    fitness[i] = f(i)

fitness /= fitness.sum()


for g in range(1,ngenerations):
    print('generation {0}'.format(g))
    w=parent*fitness
    w /= w.sum()
    child = np.zeros((n+1))
    for v in range(0,n+1):
        for i in range(0,n+1):
            for j in range(0,n+1):
                for x in range(0,i+1):
                    child[v] += pOverlaps(n,i,j-i,x) * bindist(i - 2*x + j, v - x) * w[i] * w[j]
    genData[g,:] = child.copy()
    parent = child.copy()

showdata(genData, color='jet')
showdata(v_genData, color='jet')


clist   = samplecolors(ngenerations)
clist_v = samplecolors(ngenerations,palette=plt.cm.viridis)

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
for g in range(ngenerations):
    ax.plot(np.arange(n+1),v_genData[g,:],label='generation {0}, stochastic'.format(g),color=clist_v[g])

plt.show()

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
for g in range(ngenerations):
    ax.plot(np.arange(n+1),  genData[g,:],label='generation {0}'.format(g),color=clist[g])

plt.show()


showlist((genData*np.arange(n+1)).sum(1))
showlist(genData.var(0))


avg = (np.arange(n+1)*np.ones((ngenerations,1))*genData).sum(1)



fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
ax.plot(avg[:-1], np.diff(avg))
ax.set_xlabel('mean trait value') 
ax.set_ylabel('derivative') 
plt.show()

showlist(avg)
showlist(fitness)
