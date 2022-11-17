
from scipy.special import comb  
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#%%
def bindist(n,k,p=0.5):
    return comb(n,k)*p**k*(1-p)**(n-k)

def oc(v,n,i,j):
    sumvar=0
    v=int(v)
    n=int(n)
    i=int(i)
    j=int(j)
    for x in range(i+1):
        sumvar+=comb(i,x) * comb(n - i, j - x) / comb(n,j)*bindist(i + j - 2*x, v - x)
    return sumvar

#=======================================================================================================
#%%
def showlist(l, distbins=False):
            fig = plt.figure(); ax = fig.add_subplot(111)
            ax.plot(np.arange(len(l)),list(l))
            plt.show()

def showdata(mat, color=plt.cm.gnuplot, symmetry=False):
    mat = np.copy(mat)
    if symmetry:
        top = np.max([np.abs(np.nanmax(mat)),np.abs(np.nanmin(mat))])
        plt.imshow(mat.astype('float32'), interpolation='none', cmap='seismic',vmax=top,vmin=-top)
    else:
        plt.imshow(mat.astype('float32'), interpolation='none', cmap=color)
    plt.colorbar()
    plt.show()

def showF3D(f,type='surf', rangeX=(-1,1),rangeY=(-1,1),res=20,zlim=None,cmap='jet'):
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
        plt.show()
    elif type == 'scatter':
        fig = plt.figure(); ax = fig.add_subplot(projection='3d')
        if zlim:
            ax.axes.set_zlim3d(bottom=zlim[0], top=zlim[1]) 
        scat = ax.scatter3D(x,y,z, c=z, cmap=cmap)
        fig.colorbar(scat)
        plt.show()
    elif type == 'map':
        # showdata(np.array(z).reshape((resX,resY)), color='jet')
        if zlim:
            plt.imshow(np.array(z).reshape((resX,resY)).astype('float32'), interpolation='none', cmap=cmap,vmin=zlim[0], vmax=zlim[1])
        else:
            plt.imshow(np.array(z).reshape((resX,resY)).astype('float32'), interpolation='none', cmap=cmap)

 
#%%

def g(x,y):
    return np.sin(x) + np.cos(y)

showF3D(g)
showF3D(g,rangeX=(-5,5),rangeY=(-5,5))
showF3D(g,rangeX=(-5,5),rangeY=(-5,5),res=40)
showF3D(g,type='scatter', rangeX=(0,5),rangeY=(0,5))


#=======================================================================================================
#%%

nloci=100
v=20
def oc2 (i,j):
    global nloci,v
    n=nloci
    #i=
    #j=
    return oc(v,n,i,j)

#showF3D(oc2, rangeX=(0,nloci),rangeY=(0,nloci),zlim=(0,1),res=int(nloci/4))
showF3D(oc2, rangeX=(0,nloci),rangeY=(0,nloci),zlim=(0,0.5),res=int(nloci),type='map')
showF3D(oc2, rangeX=(0,nloci),rangeY=(0,nloci),             res=int(nloci),type='map')
showF3D(oc2, rangeX=(0,nloci),rangeY=(0,nloci),zlim=(0,0.000000001),res=int(nloci),type='map',cmap='jet')


#%%
'''
oc(3,10,4,5)
oc2(4,5)
np.all(list(map(lambda x: x%1==0,(v,n,i,j))))
'''
#%%
nloci=100
v=50

rangeX=(0,nloci)
rangeY=(0,nloci)
res=nloci
zlim=(0,1)
resX=res
resY=res
x = np.linspace(rangeX[0],rangeX[1],resX)
y = np.linspace(rangeY[0],rangeY[1],resY)
gx,gy = np.meshgrid(x,y)
x, y = gx.flatten(), gy.flatten()
z = list(map(oc2, x,y))
mat=np.array(z).reshape((resX,resY)).astype('float32')
#%%
showdata(mat)

slicedata = mat[10,:]
showlist(slicedata)
#%%
def norm(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def normMult(x,mu,sigma,k):
    return norm(x, mu, sigma)*k
#%%
bounds = np.array(((0, nloci),
                   (0, np.inf),
                   (0, np.inf)))

parlist = np.zeros((1,3))
covlist = np.zeros((1,3,3))

for row in range(nloci):
    slicedata = mat[row,:]
    pars, cov = curve_fit(f = normMult,
                      xdata = np.arange(slicedata.size),
                      ydata = slicedata,
                      p0 = [nloci/2,1,1],
                      bounds = bounds.T,
                      check_finite = True)

    parlist = np.append(parlist,pars[np.newaxis,:], axis=0)
    covlist = np.append(covlist, cov[np.newaxis,:], axis=0)

parlist = parlist[1:]
covlist = covlist[1:]
showlist(parlist)
showlist(parlist[:,1])
showlist(parlist[:,2])

#%%
row=60
fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(np.arange(mat[row,:].size),mat[row,:])
ax.plot(np.arange(mat[row,:].size),normMult(np.arange(mat[row,:].size), parlist[row,0],parlist[row,1],parlist[row,2]))
plt.show()






#%%
'''
def oc3 (v):
    #v=
    n=10
    i=4
    j=5
    return oc(v,n,i,j)

showlist(list(map(oc3, np.arange(10))))
'''
#%%
sigma50=parlist[:,1]
showseries(sigma50)
def parabolic(x, a, b, c):
    return a*x**2+b*x+c

series=sigma50[30:80]
p_pars, p_cov = curve_fit(f = parabolic,
                  xdata = np.arange(series.size),
                  ydata = series,
                  p0 = [1,1,1])

fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(np.arange(series.size),series)
ax.plot(np.arange(series.size),parabolic(np.arange(series.size), p_pars[0],p_pars[1],p_pars[2]))
plt.show()

#%%












#%%
nloci=100

rangeX=(0,nloci)
rangeY=(0,nloci)
resX=res
resY=res
x = np.linspace(rangeX[0],rangeX[1],resX)
y = np.linspace(rangeY[0],rangeY[1],resY)
gx,gy = np.meshgrid(x,y)
x, y = gx.flatten(), gy.flatten()

bounds = np.array(((0, nloci),
                   (0, np.inf),
                   (0, np.inf)))

sigmaseries=np.zeros((nloci,nloci))

for v in range(nloci):
    print(v)
    z = list(map(oc2, x,y))
    mat=np.array(z).reshape((resX,resY)).astype('float32')
        
    parlist = np.zeros((1,3))
    covlist = np.zeros((1,3,3))
    
    for row in range(nloci):
        slicedata = mat[row,:]
        pars, cov = curve_fit(f = normMult,
                          xdata = np.arange(slicedata.size),
                          ydata = slicedata,
                          p0 = [nloci/2,1,1],
                          bounds = bounds.T,
                          check_finite = True)
    
        parlist = np.append(parlist,pars[np.newaxis,:], axis=0)
        covlist = np.append(covlist, cov[np.newaxis,:], axis=0)
    
    parlist = parlist[1:]
    covlist = covlist[1:]
    
    sigmaseries[v,:]=parlist[:,1]
    

#%%
showdata(sigmaseries[40:60,:])
showlist(sigmaseries[60,:])

#%%
thres=.2
parabolic_params_list = np.zeros((nloci,3))
for v in range(int(nloci*.1),int(nloci*.9)):
    lb = 0 if v < nloci/2-nloci*thres else v-int(nloci/2-nloci*thres)
    ub = v+nloci-int(nloci/2+nloci*thres) if v < nloci/2+nloci*thres else nloci
    print(v,lb,ub)
    
    series=sigmaseries[v,lb:ub]
    series=sigmaseries[v,:]
    p_pars, p_cov = curve_fit(f = parabolic,
                      xdata = np.arange(series.size),
                      ydata = series,
                      p0 = [1,1,1])
    
    parabolic_params_list[v,:]=p_pars
    
    

#%%

v=20
lb = 0 if v < nloci/2-nloci*thres else v-int(nloci/2-nloci*thres)
ub = v+nloci-int(nloci/2+nloci*thres) if v < nloci/2+nloci*thres else nloci
print(v,lb,ub)

series=sigmaseries[v,lb:ub]
p_pars = parabolic_params_list[v]

fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(np.arange(series.size),series)
ax.plot(np.arange(series.size),parabolic(np.arange(series.size)+lb, p_pars[0],p_pars[1],p_pars[2]))
plt.show()



#%%
fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(np.arange(series.size),series)
ax.plot(np.arange(series.size),parabolic(np.arange(series.size), p_pars[0],p_pars[1],p_pars[2]))
plt.show()





























































