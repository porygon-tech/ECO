#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:36:53 2024
Just some tests to fight the ubiquity of gamma and hypergeometric functions in the genetics of sexual reproduction and recombination 
@author: ubuntu
"""


from scipy.special import comb  
def bindist(n,k,p=0.5):
    return comb(n,k)*p**k*(1-p)**(n-k)



n=120
k=5
cp = 1
p=0
while p < k:
    cp*=(n-p)/(k-p)
    p+=1
cp
comb(n,k)
#%%

n=120
k=73
x=1


cp = 1
p=1
while p < x:
    cp*=(k-p)/(n-p)
    p+=1
print(comb(n,k)*cp == comb(n-x,k-x))

#%% check if the identity I found is correct

def binMix(a,b,x):
    return comb(a-x,b-x)

a=53
b=23
x=13


cp = 1
p=0
while p < x:
    cp*=(b-p)/(a-p)
    p+=1
    
binMix(a,b,x) - round(comb(a,b)*cp)
#%%
from scipy.special import gamma, factorial
#gamma(a+1) - factorial(a)

comb(a-x,b-x)

# comb(a,b) * gamma(-a)/gamma(-b) * gamma(x-b)/gamma(x-a) # gives error, as gamma for negative values does not exist

dv=1e-9 # approximation
appr = round(comb(a,b) * (gamma(-a+dv)/gamma(-b+dv) + gamma(-a-dv)/gamma(-b-dv))/2 * (gamma(x-b+dv)/gamma(x-a+dv) + gamma(x-b-dv)/gamma(x-a-dv))/2)
appr = round(comb(a,b) * (-1)**(2*(a-b))*factorial(b)/factorial(a)*factorial(a-x)/factorial(b-x) ) # exact value using residues at the poles.


comb(a-x,b-x) - appr # should be zero
comb(a-x,b-x) == round(comb(a,b) * comb(b,x) / comb(a,x)) # a nice identity I derived
#%% test another identity (II)
a=53
b=23
x=13


cp = 1
p=0
while p < x:
    cp*=(b-p)/(a-b+p+1)
    p+=1
    
res = round(cp * comb(a,b))
comb(a,b-x) - res


comb(a,b-x) - round(comb(a,b)*comb(b,x)/comb(a-b+x,x))
#%% imports 
import sys
sys.path.insert(0, "./lib")
import evo
import matriX as mx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from  matriX import showdata as sd
from scipy.special import hyp2f1
from scipy.optimize import curve_fit
mx.graphictools.inline_backend(True)
mx.graphictools.inline_backend(False)
#%%
def showsurf(mat, cmap=plt.cm.gnuplot, type = 'surf', zlim=None):
    fig = plt.figure(); ax = fig.add_subplot(projection='3d')
    x = np.arange(mat.shape[0])
    y = np.arange(mat.shape[1])
    gx,gy = np.meshgrid(x,y)
    x, y = gx.flatten(), gy.flatten()
    z=mat.flatten()
    
    if type == 'surf':
        if zlim:
            ax.axes.set_zlim3d(bottom=zlim[0], top=zlim[1]) 
        surf = ax.plot_trisurf(x,y,z, cmap=cmap, linewidth=0,antialiased = True)
        fig.colorbar(surf)
        plt.show()
    elif type == 'scatter':
        if zlim:
            ax.axes.set_zlim3d(bottom=zlim[0], top=zlim[1]) 
        scat = ax.scatter3D(x,y,z, c=z, cmap=cmap)
        fig.colorbar(scat)
        plt.show()


def showF3D(f,type='surf', rangeX=(-1,1),rangeY=(-1,1),res=20,zlim=None,cmap='jet',zscale='linear'):
    resX=res
    resY=res
    x = np.linspace(rangeX[0],rangeX[1],resX)
    y = np.linspace(rangeY[0],rangeY[1],resY)
    gx,gy = np.meshgrid(x,y)
    x, y = gx.flatten(), gy.flatten()
    z = list(map(f, x,y))
    if type == 'surf':
        fig = plt.figure(); ax = fig.add_subplot(projection='3d')
        if zlim:
            ax.axes.set_zlim3d(bottom=zlim[0], top=zlim[1]) 
        surf = ax.plot_trisurf(x,y,z, cmap=cmap, linewidth=0,antialiased = True)
        fig.colorbar(surf)
        ax.set_zscale(zscale)
        plt.show()
    elif type == 'scatter':
        fig = plt.figure(); ax = fig.add_subplot(projection='3d')
        if zlim:
            ax.axes.set_zlim3d(bottom=zlim[0], top=zlim[1]) 
        scat = ax.scatter3D(x,y,z, c=z, cmap=cmap)
        fig.colorbar(scat)
        ax.set_zscale(zscale)
        plt.show()
    elif type == 'map':
        # showdata(np.array(z).reshape((resX,resY)), color='jet')
        if zlim:
            plt.imshow(np.array(z).reshape((resX,resY)).astype('float32'), interpolation='none', cmap=cmap,vmin=zlim[0], vmax=zlim[1])
        else:
            plt.imshow(np.array(z).reshape((resX,resY)).astype('float32'), interpolation='none', cmap=cmap)


#%%
mx.graphictools.inline_backend(True)

n=100
def f(g,x):
    global n
    csum=0
    for k in range(0,n):
        csum += comb(g-2*x,k-x)
    csum=((np.max((csum,1))))**(1/n)
    # csum=csum**(1/n)
    return csum


showF3D(f, rangeX=(0,60), rangeY=(0,60),cmap='flag',res=100,type='map')
#%%
mx.graphictools.inline_backend(False)

n=200
def f(g,x):
    global n
    csum=0
    for k in range(0,n):
        csum += k * comb(g-2*x,k-x)
    csum=((np.max((csum,1))))**(1/n)
    # csum=csum**(1/n)
    return csum

showF3D(f, rangeX=(0,120), rangeY=(0,120))
'''
What happens here: 
    When calculating the value of the mean trait, the sum over k that is applied can be reduced to the value that is yielded by csum.
    When exponentiated to 1/n, its values resemble a plane or a parabolic valley.
'''
#%%

def f(g,x):
    global n
    csum=0
    for k in range(0,n):
        csum += k/n * comb(g-2*x,k-x)
    csum=((np.max((csum,1))))**(1/n)
    return csum

def f2(g,x):
    global n
    res = (2**( g - 2*x - 1)*g)/((g - x)*factorial(g - 2*x)) * factorial(g-x)/(factorial(x) * comb(g-x,x))
    return ((np.max((res,1))))**(1/n)

def f3(g,x):
    global n
    res = 2**( g - 2*x - 1)*g/(g - x)
    return ((np.max((res,1))))**(1/n)

# showF3D(f2, rangeX=(0,120), rangeY=(0,120))
# def f2(g,x):
#     global n
#     csum=0
#     for k in range(0,n):
#         denom=factorial(g-x-k) * factorial(k-x)
#         if denom !=0: 
#             csum += k/n /denom 
#     denom2 = factorial(x) * comb(g-x,x)
#     if denom2 !=0:
#         csum *= factorial(g-x)/denom2
#         csum=((np.max((csum,1))))**(1/n)
#         return csum
#     else:
#         return 0


rangeX=(0,n)
rangeY=(0,n)
res=20
resX=res
resY=res
x = np.linspace(rangeX[0],rangeX[1],resX)
y = np.linspace(rangeY[0],rangeY[1],resY)
gx,gy = np.meshgrid(x,y)
x, y = gx.flatten(), gy.flatten()
z = list(map(f, x,y))
z2 = list(map(f3, x,y))


fig = plt.figure(); ax = fig.add_subplot(projection='3d')
surf = ax.plot_trisurf(x,y,z2, cmap='jet', linewidth=0,antialiased = True)
ax.scatter3D(x,y,z)
fig.colorbar(surf)
plt.show()




g=53
x=13

factorial(g-x)/(factorial(x) * comb(g-x,x))/(factorial(g - 2*x))


#%% plane approximation
'''
This is an approximation I found to a plane when doing the n root of the sum above. No math proof yet though :(
    ALERT: this approximation works for csum +=     comb(g-2*x,k-x),
           but when we calculate        csum += k * comb(g-2*x,k-x) it does not resemble it that much.
'''
def f2(g,x):
    global n
    z=1/n*(g-2*x)+1
    return z


rangeX=(0,2*n)
rangeY=(0,n)
res=20
resX=res
resY=res
x = np.linspace(rangeX[0],rangeX[1],resX)
y = np.linspace(rangeY[0],rangeY[1],resY)
gx,gy = np.meshgrid(x,y)
x, y = gx.flatten(), gy.flatten()
z = list(map(f, x,y))
z2 = list(map(f2, x,y))


fig = plt.figure(); ax = fig.add_subplot(projection='3d')
surf = ax.plot_trisurf(x,y,z, cmap='jet', linewidth=0,antialiased = True)
ax.scatter3D(x,y,z2)
fig.colorbar(surf)
plt.show()


#%%%
'''
such approximation has a difference with the real version, that is similar to a 3d parabolic valley
'''

za= np.array(z)
z2a= np.array(z2)

(za [np.where(za!=1)].sum() - z2a[np.where(z2a>1)].sum()) / res**2

mx.showdata(za.reshape(resY,resX))

def valley(xy,a,b,c,d):
    x,y=xy
    z= (b*x - c*y + a)**2 + d
    return z
    

pos = np.where(za!=1)
diffs = list(za [pos] - z2a[pos])

pars, cov = curve_fit(f = valley,
	                      xdata = (x[pos],y[pos]),
	                      ydata = diffs,
	                      p0 = [-5,0.2,0.2,-3.5],
	                      #bounds = bounds.T,
	                      check_finite = True)

fig = plt.figure(); ax = fig.add_subplot(projection='3d')
ax.scatter3D(x[pos],y[pos],diffs)
#surf = ax.plot_trisurf(x,y,valley((x,y),*pars), cmap='jet', linewidth=0,antialiased = True)
surf = ax.plot_trisurf(x[pos],y[pos],valley((x[pos],y[pos]),*pars), cmap='jet', linewidth=0.1,antialiased = True)

fig.colorbar(surf)
plt.show()


fig = plt.figure(); ax = fig.add_subplot(projection='3d')
surf = ax.plot_trisurf(x,y,z, cmap='twilight', linewidth=0,antialiased = True)
# ax.scatter3D(x,y,z2)
ax.scatter3D(x[pos],y[pos],z2a[pos]+valley((x[pos],y[pos]),*pars))
fig.colorbar(surf)
plt.show()



#%%%
def f3(g,x):
    global n
    z= comb(g-x,n+1)*comb(n+1,x)*hyp2f1(1,n+x-g+1,n-x+2,-1)
    return z**(1/n)
    # return np.log(z)

showF3D(f3, rangeX=(0,2*n), rangeY=(0,n))


def f4(g,x):
    global n
    return f2(g,x)-f(g,x)
showF3D(f4, rangeX=(0,2*n), rangeY=(0,n))


#%%
mx.graphictools.inline_backend(True)
parlist = np.zeros((1,4))
covlist = np.zeros((1,4,4))
ns=np.arange (50,150)
for n in ns:
    print(n)
    rangeX=(0,2*n)
    rangeY=(0,n)
    res=20
    resX=res
    resY=res
    x = np.linspace(rangeX[0],rangeX[1],resX)
    y = np.linspace(rangeY[0],rangeY[1],resY)
    gx,gy = np.meshgrid(x,y)
    x, y = gx.flatten(), gy.flatten()
    z = list(map(f, x,y))
    z2 = list(map(f2, x,y))
    
    za= np.array(z)
    z2a= np.array(z2)
    
    pos = np.where(za!=1)
    diffs = list(za [pos] - z2a[pos])
    
    pars, cov = curve_fit(f = valley,
    	                      xdata = (x[pos],y[pos]),
    	                      ydata = diffs,
    	                      p0 = [-5,0.2,0.2,-3.5],
    	                      #bounds = bounds.T,
    	                      check_finite = True)
    parlist = np.append(parlist,pars[np.newaxis,...], axis=0)
    covlist = np.append(covlist, cov[np.newaxis,...], axis=0)

parlist = parlist[1:]
covlist = covlist[1:]
#%%
print(parlist[-1])
plt.plot(ns,parlist);plt.show()
covlist[0]
covlist[-1]
plt.plot(ns,list(map(lambda x: (np.triu(x)).flatten(),covlist)));plt.show()
#%%
plt.plot(ns,parlist[:,0]/parlist[:,1])
plt.plot(ns,(1-ns)*0.532)         
plt.show()

def line(x,m,n):
    return m*x+n

linepars, cov = curve_fit(f = line,
	                      xdata = ns,
	                      ydata = parlist[:,0]/parlist[:,1],
	                      p0 = [-0.5,1],
	                      #bounds = bounds.T,
	                      check_finite = True)
#%%
plt.plot(ns,parlist[:,0]/parlist[:,1])
plt.plot(ns,line(ns,*linepars))       
plt.plot(ns,line(ns,-(e+1)/14,(e-1)/2))         
plt.show()
#%%
parlist[-1]
np.round(parlist[-1],5)

#%%
plt.plot(ns, parlist[:,0]/ns,parlist[:,1])
plt.plot(ns, parlist[:,2]/ns,parlist[:,1])
plt.plot(ns, parlist[:,3]/ns,parlist[:,1])
plt.plot(ns, parlist[:,1])
# plt.plot(ns,line(ns,*linepars))       
# plt.plot(ns,line(ns,-(e+1)/14,(e-1)/2))         
plt.show()

#%%
def miprimo(x,a,b,c):
    #return (a - c) *np.exp(-(b*x)) + c
    return a*x**-b+c

parlist_mycousin  = np.zeros((1,3))
for p in range(parlist.shape[1]):
    pars_for_miprimo, cov = curve_fit(f = miprimo,
    	                      xdata = ns,
    	                      ydata = parlist[:,p],
    	                      p0 = [0,0.1,0],
    	                      #bounds = bounds.T,
    	                      check_finite = True)
    parlist_mycousin = np.append(parlist_mycousin,pars_for_miprimo[np.newaxis,...], axis=0)

parlist_mycousin = parlist_mycousin[1:]

#%%
for pars_for_miprimo in parlist_mycousin:
    plt.plot(ns,miprimo(ns,*pars_for_miprimo),linewidth=10)
plt.plot(ns,parlist)
plt.show()





























#%% hypergeom 3F2 approximation
mx.graphictools.inline_backend(False)
# mx.graphictools.inline_backend(True)

def uhh3F2(i,j):
    global n
    x=0
    csum = 0
    i=int(i)
    j=int(j)
    errsum=0
    #    for x in range(min(i,j),max(i,j)):
    for x in range(0,min(j,i)):
        if n>i+j:
            csum += comb(i, x)*comb(j, x)*(i + j)/(comb(n - i - j + x, x)*(i + j - x))
    
    #return csum**(1/n)
    return csum**(1/(i+j+1))

def f(i,j):
    global n
    if n>i+j:
        # return 1+((i*j)**2 / (6*n**3))
        return ((i*j) / (n))**(2 / (n)*np.sqrt(i*j+ 1))
    else:
        return 0


n=100
rangeX=(0,n)
rangeY=(0,n)
res=30
resX=res
resY=res
x = np.linspace(rangeX[0],rangeX[1],resX)
y = np.linspace(rangeY[0],rangeY[1],resY)
gx,gy = np.meshgrid(x,y)
x, y = gx.flatten(), gy.flatten()
z = list(map(f, x,y))
z2 = list(map(uhh3F2, x,y))


fig = plt.figure(figsize=(10,10)); ax = fig.add_subplot(projection='3d')

ax.view_init(elev=10., azim=-30)
surf = ax.plot_trisurf(x,y,z2, cmap='jet', linewidth=0, antialiased = True)
# ax.scatter3D(x,y,z)
fig.colorbar(surf)
plt.show()



#%%

za= np.array(z)
z2a= np.array(z2)

def f_fit(ij,a,b,c):
    global n
    i,j =ij
    z = np.zeros_like(i)
    out = np.sqrt(2*np.pi*i*j/a)*(i*j / np.e / b)**(i*j / c)
    mask = n>i+j
    z[mask] = out[mask]

    return z

    

pos = np.where(z2a!=0)
out = z2a[pos]

pars, cov = curve_fit(f = f_fit,
	                      xdata = (x[pos],y[pos]),
	                      ydata = out,
	                      p0 = [n,n,n],
	                      #bounds = bounds.T,
	                      check_finite = True)


#%%

z=f_fit((x,y),*pars)
z2 = list(map(uhh3F2, x,y))


fig = plt.figure(figsize=(10,10)); ax = fig.add_subplot(projection='3d')

ax.view_init(elev=10., azim=-30)
surf = ax.plot_trisurf(x,y,z2, cmap='jet', linewidth=0, antialiased = True)
ax.scatter3D(x,y,z)
fig.colorbar(surf)
plt.show()
