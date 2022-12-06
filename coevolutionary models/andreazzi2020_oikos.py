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

def pM (zi,zj, alpha=50):
    return np.exp(-alpha*(zi-zj)**2)

def pB (zi,zj, alpha=50):
    return 1/(1+np.exp(-alpha*(zi-zj)))


def diffM (S,M, phi=0.25):
    return phi*(S+M)


#%%

showfunc(pM, zj=2., xlim=(0,3))
showfunc(pB, zj=2., xlim=(0,3))
showfunc(pB, zj=2., alpha=5, xlim=(0,3))

#%% DATA LOAD
dataPath = root / 'data/dataBase'
df = pd.read_csv(dataPath / 'FW_017_03.csv', index_col=0)
#a = df.to_numpy()

np.all(df.columns == df.index)
#l[0,:,:]=(df.to_numpy()>0)+0.

b=mX.nullnet((df.to_numpy()>0)+0)
showdata(b)
#%%
#N=a.shape[0]
N=50
c=0.03 # expected connectance
ntimesteps=10000
phi=0.25
pS=[2,2.5]
xi_S=0.5 # level of environmental selection (from 0 to 1)
xi_d=1-xi_S # level of selection imposed by resource species (from 0 to 1)

theta=np.random.rand(N)*np.diff(pS)+pS[0] #values favoured by env. selection
initial_l=generateWithoutUnconnected(N,N,0.05)
initial_l=initial_l-np.diag(np.diag(initial_l))
#%

z=np.zeros((ntimesteps,N))
S=np.zeros((ntimesteps,N))
p=np.zeros((ntimesteps,N,N))
l=np.zeros((ntimesteps,N,N))
a=np.zeros((ntimesteps,N,N))
M=np.zeros((ntimesteps,N,N))


z[0,:] = theta # for simplicity
S[0,:] = xi_S*(theta-z[0,:])
p[0,:,:]=pM(z[0,:]*np.ones((N,1)), (z[0,:]*np.ones((N,1))).T)
l[0,:,:]=initial_l

lxp=l[0,:,:]*p[0,:,:]
a[0,:,:] = lxp/(lxp-np.diag(np.diag(lxp))).sum(1)[:,np.newaxis]*np.ones((1,N))
M[0,:,:]=xi_d*a[0,:,:]*(z[0,:]*np.ones((N,1)) - (z[0,:]*np.ones((N,1))).T)

#%%

a.sum(1)


for t in range(1,ntimesteps):
    z[t,:] = z[t-1,:] + phi*(S[t-1,:]+M[t-1,:,:].sum(1))
    S[t,:] = xi_S*(theta-z[t,:])
    p[t,:,:]=pM(z[t,:]*np.ones((N,1)), (z[t,:]*np.ones((N,1))).T)
        
    lxp=l[0,:,:]*p[t,:,:]
    a[t,:,:] = lxp/(lxp-np.diag(np.diag(lxp))).sum(1)[:,np.newaxis]*np.ones((1,N))
    M[t,:,:]=xi_d*a[t,:,:]*(z[t,:]*np.ones((N,1)) - (z[t,:]*np.ones((N,1))).T)

showlist(z[:40,:])
