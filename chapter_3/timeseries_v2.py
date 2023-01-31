# -*- coding: utf-8 -*-
"""The_Effect_of_Polygenic_Adaptation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/porygon-tech/ECO/blob/main/demos/The_Effect_of_Polygenic_Adaptation.ipynb
"""

#@title imports and definitions
from scipy.special import comb  
import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit

#---for obj handling---
from os import chdir
from pathlib import Path
import pickle5
import bz2
chdir('/home/roman/LAB/ECO') #this line is for Spyder IDE only
root = Path(".")
obj_path = root / 'data/obj'
img_path = root / 'gallery/timeseries'

#---for animations---
from matplotlib import rc
rc('animation', html='jshtml')
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import copy

#%%
import matplotlib.colors

def blend(image1, image2, cmap1, cmap2,norm=None):
    if norm:
        a = norm(cmap1(image1).sum(2))
        b = norm(cmap2(image2).sum(2))
    else:
        a = cmap1(image1)
        b = cmap2(image2)

    screen = 1 - (1 - a) * (1 - b)
    return screen


def rescale(arr: np.ndarray, vmin=0,vmax=1):
    re = (arr - vmin) / (vmax - vmin)
    return re
#%%

'''
cmapgrn = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "cyan"]) #seagreen also
cmapred = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "red"])

a=cmapred(mat1)
b=cmapgrn(mat2)
norm=np.linspace(0,1,100)
cmapgrn = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "cyan"])
screen = 1 - (1 - a) * (1 - b)
a.shape
'''
#%%
def bindist(n,k,p=0.5):
    return comb(n,k)*p**k*(1-p)**(n-k)

def showfunc(f,xlim=(-5,5),definition=100, **kwargs):
            x= np.linspace(xlim[0],xlim[1],definition)
            fig = plt.figure(); ax = fig.add_subplot(111)
            ax.plot(x,f(x,**kwargs))
            plt.show()

def showlist(l, distbins=False):
            fig = plt.figure(); ax = fig.add_subplot(111)
            ax.plot(np.arange(len(l)),list(l))
            plt.show()

def showdata(mat, color='magma', symmetry=False):
    mat = np.copy(mat)
    if symmetry:
        top = np.max([np.abs(np.nanmax(mat)),np.abs(np.nanmin(mat))])
        plt.imshow(mat.astype('float32'), interpolation='none', cmap='seismic',vmax=top,vmin=-top)
    else:
        plt.imshow(mat.astype('float32'), interpolation='none', cmap=color)
    plt.colorbar()
    plt.show()

def augment(x,a=10,b=5):
    return 1-a**(-b*x)
#%%
n=100
nstates=n+1
filename='oc_tensor_' + str(n) + '.obj'
with bz2.BZ2File(obj_path / filename, 'rb') as f:
	oc_tensor = pickle5.load(f)


#%%
#@title POPULATION INITIALISATION
#------------------------

ntimesteps = 200
#%%
skw=.70#np.random.rand()
v = np.zeros((ntimesteps, nstates,1))
for i in range(nstates):
    v[0,i] = bindist(n,i,skw)

#%%
def f(i):
    #return 1+n-i
    #return 2**(-i/2)
    #return n**2-i**2
    alpha=0.001; m=1/4*n
    #return (phi*(-1/2*(theta-i)**2)-np.min((phi*(-1/2*(theta-n)**2),phi*theta)))
    return np.exp(-alpha*(i-m)**2)

l = np.zeros((nstates,1))
for i in range(nstates):
    l[i] = f(i)

#------------------------
#showlist(v[0])
showlist(l)

#%%
#STANDARD RUN
phi=0.04
z = np.zeros((ntimesteps,1))
z[0]=(np.arange(nstates)*v[0].T).sum() #mean
theta=np.where(l==max(l))[0][0] #max of f (should have only one)


for t in range(1,ntimesteps):
    w = v[t-1]*l
    v[t] = ((w.T @ oc_tensor @ w) / w.sum()**2)[:,0]
    z[t]=z[t-1]+phi*(theta-z[t-1])


fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
#div = make_axes_locatable(ax)
#cax = div.append_axes('right', '5%', '5%')
#cax = plt.axes([0.85, 0.1, 0.075, 0.8]
imsh = ax.imshow(v[...,0].T, interpolation='none', cmap='magma')
ax.plot(np.arange(ntimesteps),(v[...,0]*np.arange(nstates)).mean(1)*nstates,color='yellow')
ax.plot(np.arange(ntimesteps),z,color='cyan')
fig.canvas.toolbar_visible = False
fig.canvas.header_visible = False
fig.canvas.resizable = True
fig.colorbar(imsh)
plt.show()

#=====================================================================================================
#=====================================================================================================
#%%
#@title  AFTER A CHANGE IN ENVIRONMENTAL SELECTION
v2 = np.zeros((ntimesteps, nstates,1))
l2 = np.flip(l)

#showlist(v[-1])
v2[0] = v[-1]
for t in range(1,ntimesteps):
    w2 = v2[t-1]*l2
    v2[t] = ((w2.T @ oc_tensor @ w2) / w2.sum()**2)[:,0]

#showdata(np.append(v,v2,axis=0)[...,0])
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
imsh = ax.imshow(np.append(v,v2,axis=0)[...,0], interpolation='none', cmap='magma')
fig.canvas.toolbar_visible = False
fig.canvas.header_visible = False
fig.canvas.resizable = True
fig.colorbar(imsh)
plt.show()





"""# New Section"""

s1=v[np.random.randint(ntimesteps)]
s2=v[np.random.randint(ntimesteps)]

skw_1, skw_2 = np.random.rand(2)

for i in range(nstates):
    s1[i] = bindist(n,i,skw_1)
    s2[i] = bindist(n,i,skw_2)

'''
fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(np.arange(nstates),s1)
ax.plot(np.arange(nstates),s2)
plt.show()
'''
m_comb_probs=s1@s2.T
showdata(m_comb_probs)
#%% 

v_s1 = np.zeros((ntimesteps, nstates,1))
v_s2 = np.zeros((ntimesteps, nstates,1))
l_s1 = np.zeros((ntimesteps, nstates,1))
l_s2 = np.zeros((ntimesteps, nstates,1))

l_s1[0] = s2
l_s2[0] = s1

v_s1[0] = s1
v_s2[0] = s2

for t in range(1,ntimesteps):
    
    w_s1 = v_s1[t-1]*l_s1[t-1]
    w_s2 = v_s2[t-1]*l_s2[t-1]
    v_s1[t] = ((w_s1.T @ oc_tensor @ w_s1) / w_s1.sum()**2)[:,0]
    v_s2[t] = ((w_s2.T @ oc_tensor @ w_s2) / w_s2.sum()**2)[:,0]
    #l_s1[t] = np.max(v_s2[t])-v_s2[t]
    #l_s2[t] = np.max(v_s1[t])-v_s1[t]
    l_s1[t] = 1/(1+100*v_s2[t])#*1/(1+1000*v_s1[t])
    #l_s2[t] = 1/(1+10000*v_s1[t])#*1/(1+10000*v_s2[t])
    l_s2[t] = v_s1[t]*1/(1+100*v_s2[t])

#%%
showdata(v_s1[...,0])
showdata(v_s2[...,0])


my_cmap = copy.copy(plt.cm.get_cmap('magma')) # copy the default cmap
my_cmap.set_bad((0,0,0))

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
div = make_axes_locatable(ax)
#cax = div.append_axes('right', '5%', '5%')
def frame(t):
    ax.clear()
    #plot=ax.imshow(l_matrix[:,:,w].astype('float32'), interpolation='none', cmap=my_cmap,vmax=1,vmin=10e-20,norm=LogNorm(vmin=0, vmax=1)) #
    ax.set_ylim(0,0.1)
    plot=ax.bar(np.arange(nstates),v_s1[t,:,0],width=1,alpha=0.5)
    plot+=ax.bar(np.arange(nstates),v_s2[t,:,0],width=1,alpha=0.5)
    #fig.colorbar(plot,ax=ax,cax=cax)
    return plot

anim = animation.FuncAnimation(fig, frame, frames=200, blit=False, repeat=True)

anim

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')
def frame(t):
    ax.clear()
    plot=ax.imshow(v_s1[t]@v_s2[t].T, interpolation='none', cmap='magma',vmax=0.005,vmin=0) #
    fig.colorbar(plot,ax=ax,cax=cax)
    return plot

anim = animation.FuncAnimation(fig, frame, frames=100, blit=False, repeat=True)

anim

s1=np.zeros((nstates,1))
s2=np.zeros((nstates,1))

skw_1, skw_2 = np.random.rand(2)

for i in range(nstates):
    s1[i] = bindist(n,i,skw_1)
    s2[i] = bindist(n,i,skw_2)

'''
fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(np.arange(nstates),s1)
ax.plot(np.arange(nstates),s2)
plt.show()
'''
m_comb_probs=s1@s2.T
showdata(m_comb_probs)
#%%
skw_1, skw_2 = np.random.rand(2)

for i in range(nstates):
    s1[i] = bindist(n,i,skw_1)
    s2[i] = bindist(n,i,skw_2)


v_s1 = np.zeros((ntimesteps, nstates,1))
v_s2 = np.zeros((ntimesteps, nstates,1))
l_s1 = np.zeros((ntimesteps, nstates,1))
l_s2 = np.zeros((ntimesteps, nstates,1))

l_s1[0] = s1
l_s2[0] = s2

v_s1[0] = s1
v_s2[0] = s2

for t in range(1,ntimesteps):
    
    w_s1 = v_s1[t-1]*l_s1[t-1]
    w_s2 = v_s2[t-1]*l_s2[t-1]
    v_s1[t] = ((w_s1.T @ oc_tensor @ w_s1) / w_s1.sum()**2)[:,0]
    v_s2[t] = ((w_s2.T @ oc_tensor @ w_s2) / w_s2.sum()**2)[:,0]
    #l_s1[t] = np.max(v_s2[t])-v_s2[t]
    #l_s2[t] = np.max(v_s1[t])-v_s1[t]
    l_s1[t] = 1/(1+200*v_s2[t])*1/(1+600*v_s1[t])
    #l_s2[t] = 1/(1+10000*v_s1[t])#*1/(1+10000*v_s2[t])
    #l_s2[t] = v_s1[t]*1/(1-200*(v_s2[t]-1)) # positive freq-dependent sel
    l_s2[t] = v_s1[t]*1/(1+120*v_s1[t]) # negative freq-dependent sel

#%%
'''
showdata(v_s1[...,0])
showdata(v_s2[...,0])

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
div = make_axes_locatable(ax)
#cax = div.append_axes('right', '5%', '5%')
def frame(t):
    ax.clear()
    #plot=ax.imshow(l_matrix[:,:,w].astype('float32'), interpolation='none', cmap=my_cmap,vmax=1,vmin=10e-20,norm=LogNorm(vmin=0, vmax=1)) #
    ax.set_ylim(0,0.1)
    plot=ax.bar(np.arange(nstates),v_s1[t,:,0],width=1,alpha=0.5)
    plot+=ax.bar(np.arange(nstates),v_s2[t,:,0],width=1,alpha=0.5)
    #fig.colorbar(plot,ax=ax,cax=cax)
    return plot

anim = animation.FuncAnimation(fig, frame, frames=100, blit=False, repeat=True)

anim

'''
#=====================================================================================================
#=====================================================================================================
#%%With soft interaction probabilities


import matplotlib.colors as colors


def pM (zdiffs, alpha=50):
    return np.exp(-alpha*(zdiffs)**2)

def pB (zdiffs, alpha=50):
    return 1/(1+np.exp(-alpha*(zdiffs)))

def convpM(values,nstates,alpha):
  c = np.zeros((nstates))
  for i in range(nstates):
    c = c + pM(np.arange(nstates)-i, alpha)*values[i]
  return c

def convpB(values,nstates,alpha):
  c = np.zeros((nstates))
  for i in range(nstates):
    c = c + pB(np.arange(nstates)-i, alpha)*values[i]
  return c

s1=np.zeros((nstates,1))
s2=np.zeros((nstates,1))

skw_1, skw_2 = np.random.rand(2)

for i in range(nstates):
    s1[i] = bindist(n,i,skw_1)
    s2[i] = bindist(n,i,skw_2)

'''
alpha = 0.05

pM_s1=convpM(s1,nstates,alpha)
pB_s1=convpB(s1,nstates,alpha)

showlist(s1)
showlist(pM_s1)
showlist(pB_s1)
'''

#%%

alpha=0.1
a=20
b=1

v_s1 = np.zeros((ntimesteps, nstates,1))
v_s2 = np.zeros((ntimesteps, nstates,1))
l_s1 = np.zeros((ntimesteps, nstates,1))
l_s2 = np.zeros((ntimesteps, nstates,1))

l_s1[0] = s1
l_s2[0] = s2

v_s1[0] = s1
v_s2[0] = s2

for t in range(1,ntimesteps):
    w_s1 = v_s1[t-1]*l_s1[t-1]
    w_s2 = v_s2[t-1]*l_s2[t-1]
    v_s1[t] = ((w_s1.T @ oc_tensor @ w_s1) / w_s1.sum()**2)[:,0]
    v_s2[t] = ((w_s2.T @ oc_tensor @ w_s2) / w_s2.sum()**2)[:,0]
    pM_s1=convpM(v_s1[t],nstates,alpha)[:,np.newaxis]
    pM_s2=convpM(v_s2[t],nstates,alpha)[:,np.newaxis]
    l_s1[t] = 1/(1+a*pM_s2) #* 1/(1+10*pM_s1)
    #l_s2[t] = 1/(1+10000*v_s1[t])#*1/(1+10000*v_s2[t])
    #l_s2[t] = v_s1[t]*1/(1-200*(v_s2[t]-1)) # positive freq-dependent sel
    l_s2[t] = pM_s1*b #* 1/(1+10*pM_s2) # negative freq-dependent sel


#%% ADDITIVE PLOT

def expnorm(x, a=1):
    return 1-np.exp(-a*x)
#showlist(expnorm(np.linspace(0, 1, 200),5))
def cNorm(x, k=1):
    return (k**2*x) / (1 + (-1 + k**2)*x)


mat1= (v_s1[...,0].T)
mat2= (v_s2[...,0].T)

temp_max=np.max((mat1,mat2))
temp_min=np.min((mat1,mat2))

G_r = rescale(mat1, temp_min,temp_max) #clip?
R_r = rescale(mat2, temp_min,temp_max)

cmapgrn = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "cyan"]) #seagreen also
cmapred = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "red"])

blended1=blend(image1 = G_r,
               image2 = R_r,
               cmap1 = cmapgrn,
               cmap2 = cmapred)

temp_g = 1.2
blended2=blend(image1 = cNorm(G_r,temp_g),
               image2 = cNorm(R_r,temp_g),
               cmap1 = cmapgrn,
               cmap2 = cmapred)


fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
pos = ax.imshow(blended2,interpolation='None')
fig.suptitle(r'$\alpha=$'+str(alpha)+', a='+str(a)+', b='+str(b),fontweight='bold',y=0.85)
ax.set_ylabel('Trait value')
ax.set_xlabel('Time (generations)')
plt.tight_layout()
plt.show()
#%%
fig.savefig(img_path / 'test.pdf',format='pdf')

#%%
#showdata(v_s1[...,0].T)
#showdata(v_s2[...,0].T)

palette = 'magma'
temp_g=1./1.5
mat1= (v_s1[...,0].T)
mat2= (v_s2[...,0].T)
temp_max=np.max((v_s1,v_s2))
temp_min=np.min((v_s1,v_s2))


fig = plt.figure(figsize=(8,6)); 
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
pos = ax1.imshow(mat1, interpolation='none', cmap=palette,norm=colors.PowerNorm(gamma=temp_g,vmin=temp_min,vmax=temp_max,))
ax2.imshow(      mat2, interpolation='none', cmap=palette,norm=colors.PowerNorm(gamma=temp_g,vmin=temp_min,vmax=temp_max,))
fig.colorbar(pos, ax=[ax1,ax2])
#fig.suptitle('Time evolution of coevolving trait probability mass functions',fontweight='bold',y=0.95)
fig.suptitle(r'$\alpha=$'+str(alpha)+', a='+str(a)+', b='+str(b),fontweight='bold',y=0.95)
ax1.set_ylabel('Species X')
ax2.set_ylabel('Species Y')
fig.text(0.5, 0.02, 'Time (generations)', ha='center',size=12)
fig.text(.11, 0.5, 'Trait value',       va='center', rotation='vertical',size=12)
plt.show()


#%% THREE SPECIES
skw_1, skw_2, skw_3 = np.random.rand(3)

s1=np.zeros((nstates,1))
s2=np.zeros((nstates,1))
s3=np.zeros((nstates,1))

for i in range(nstates):
    s1[i] = bindist(n,i,skw_1)
    s2[i] = bindist(n,i,skw_2)
    s3[i] = bindist(n,i,skw_3)

#%%

alpha=0.02
a=7
a1=a
a2=a
a3=a
b=1

v_s1 = np.zeros((ntimesteps, nstates,1))
v_s2 = np.zeros((ntimesteps, nstates,1))
v_s3 = np.zeros((ntimesteps, nstates,1))
l_s1 = np.zeros((ntimesteps, nstates,1))
l_s2 = np.zeros((ntimesteps, nstates,1))
l_s3 = np.zeros((ntimesteps, nstates,1))

l_s1[0] = s1
l_s2[0] = s2
l_s3[0] = s3

v_s1[0] = s1
v_s2[0] = s2
v_s3[0] = s3

for t in range(1,ntimesteps):
    w_s1 = v_s1[t-1]*l_s1[t-1]
    w_s2 = v_s2[t-1]*l_s2[t-1]
    w_s3 = v_s3[t-1]*l_s3[t-1]
    v_s1[t] = ((w_s1.T @ oc_tensor @ w_s1) / w_s1.sum()**2)[:,0]
    v_s2[t] = ((w_s2.T @ oc_tensor @ w_s2) / w_s2.sum()**2)[:,0]
    v_s3[t] = ((w_s3.T @ oc_tensor @ w_s3) / w_s3.sum()**2)[:,0]
    pM_s1=convpM(v_s1[t],nstates,alpha)[:,np.newaxis]
    pM_s2=convpM(v_s2[t],nstates,alpha)[:,np.newaxis]
    pM_s3=convpM(v_s3[t],nstates,alpha)[:,np.newaxis]
    '''
    l_s1[t] = pM_s2*b+pM_s3*b 
    l_s2[t] = 1/(1+a1*pM_s1) 
    l_s3[t] = 1/(1+a3*pM_s1) 
    '''
    l_s1[t] = pM_s2*b+pM_s3*b 
    l_s2[t] = 1/(1+a1*pM_s1) 
    l_s3[t] = 1/(1+a3*pM_s1) 

#%% ADDITIVE PLOT

mat1= (v_s1[...,0].T)
mat2= (v_s2[...,0].T)
mat3= (v_s3[...,0].T)

temp_max=np.max((mat1,mat2,mat3))
temp_min=np.min((mat1,mat2,mat3))

R_r = rescale(mat1, temp_min,temp_max) #clip?
G_r = rescale(mat2, temp_min,temp_max)
B_r = rescale(mat3, temp_min,temp_max)

cmapgrn = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "green"]) #seagreen also
cmapred = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "red"])
cmapblu = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "blue"])

blended = 1 - (1 - cmapred(R_r)) * (1 - cmapgrn(G_r)) * (1 - cmapblu(B_r))

temp_g = 1.5
blended = cNorm(blended,temp_g)


fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
pos = ax.imshow(blended,interpolation='None')
fig.suptitle(r'$\alpha=$'+str(alpha)+r'$, a_1=$'+str(a1)+r'$, a_3=$'+str(a3)+', b='+str(b),y=0.85)
ax.set_ylabel('Trait value')
ax.set_xlabel('Time (generations)')

ax.plot(t, (vrange*mat1*nstates).mean(0), color='red')
x,y=quantilebands(mat1); ax.fill(x,y,alpha=0.2,color='red')

ax.plot(t,(vrange*mat2*nstates).mean(0),color='green')
x,y=quantilebands(mat2); ax.fill(x,y,alpha=0.2,color='green')

ax.plot(t,(vrange*mat3*nstates).mean(0),color='blue')
x,y=quantilebands(mat3); ax.fill(x,y,alpha=0.2,color='blue')
plt.tight_layout()
plt.show()


#%% 


def quantilebands(mat,width=0.25):
    t=np.arange(mat.shape[1])
    return np.append(t,np.flip(t)), np.append((mat.cumsum(0)>=.5-width).argmax(0),np.flip((mat.cumsum(0)>=.5+width).argmax(0)))


def plotQuantilebands(ax, mat, widths, alpha=1, color='green'):
    for width in widths:
        x,y=quantilebands(mat,width); ax.fill(x,y,alpha=alpha/len(widths)*np.sqrt(.5-width),color=color) #alpha*(.5-width)

vrange=np.arange(nstates)[:,np.newaxis]
t=np.arange(ntimesteps)
#%% 
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
#ax.imshow(mat1,interpolation='None')

ax.plot(t, (vrange*mat1*nstates).mean(0), color='red'); plotQuantilebands(ax, mat1, [.25,.40,.49],color='red')

ax.plot(t,(vrange*mat2*nstates).mean(0),color='green'); plotQuantilebands(ax, mat2, [.25,.40,.49],color='green')

ax.plot(t,(vrange*mat3*nstates).mean(0),color='blue');  plotQuantilebands(ax, mat3, [.25,.40,.49],color='blue')
#ax.plot(t, (mat1.cumsum(0)>=.5).argmax(0))

fig.suptitle(r'$\alpha=$'+str(alpha)+r'$, a_1=$'+str(a1)+r'$, a_3=$'+str(a3)+', b='+str(b),y=1)
ax.set_ylabel('Trait value')
ax.set_xlabel('Time (generations)')
plt.tight_layout()
plt.show()


#%%
fig.savefig(img_path / 'bands.pdf',format='pdf')
#%% 

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
ax.imshow(mat1,interpolation='None')

ax.plot(t, (vrange*mat1*nstates).mean(0), color='red')
x,y=quantilebands(mat1); ax.fill(x,y,alpha=0.2,color='red')
x,y=quantilebands(mat1,0.45); ax.fill(x,y,alpha=0.1,color='red')

fig.suptitle(r'$\alpha=$'+str(alpha)+r'$, a_1=$'+str(a1)+r'$, a_3=$'+str(a3)+', b='+str(b),y=1)
ax.set_ylabel('Trait value')
ax.set_xlabel('Time (generations)')
plt.tight_layout()
plt.show()