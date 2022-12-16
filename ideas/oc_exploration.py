
from scipy.special import comb  
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#%%
def bindist(n,k,p=0.5):
    return comb(n,k)*p**k*(1-p)**(n-k)

def oc(v,n,i,j):
    #prob of getting phenotype v from parent phenotypes i,j with n loci
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
'''
showF3D(g)
showF3D(g,rangeX=(-5,5),rangeY=(-5,5))
showF3D(g,rangeX=(-5,5),rangeY=(-5,5),res=40)
showF3D(g,type='scatter', rangeX=(0,5),rangeY=(0,5))
'''
def norm(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def normMult(x,mu,sigma,k):
    return norm(x, mu, sigma)*k


def augment(x,a=10,b=5):
    return 1-a**(-b*x)
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
showF3D(oc2, rangeX=(0,nloci),rangeY=(0,nloci),zlim=(0,0.5),res=int(nloci+1),type='map')
showF3D(oc2, rangeX=(0,nloci),rangeY=(0,nloci),             res=int(nloci+1),type='map')
showF3D(oc2, rangeX=(0,nloci),rangeY=(0,nloci),zlim=(0,0.000000001),res=int(nloci+1),type='map',cmap='jet')


#%%



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
res=nloci+1
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

#%%





































#%% OC TENSOR GENERATION
nloci=50
n=nloci
nstates=nloci+1
x = np.arange(nstates)
y = np.arange(nstates)
gx,gy = np.meshgrid(x,y)
x, y = gx.flatten(), gy.flatten()

n_list=np.repeat(nloci,nstates**2)
oc_tensor = np.zeros((nstates,nstates,nstates))

#
for v in range(nstates):
    print('v='+str(v))
    v_list=np.repeat(v,nstates**2)
    z = list(map(oc, v_list,n_list,x,y))
    mat=np.array(z).reshape((nstates,nstates)).astype('float32')
    oc_tensor[v,...] = mat[np.newaxis,...]


#%% PLOT 1

cx,cy,cz = list(map(np.ndarray.flatten, np.meshgrid(np.arange(nstates),np.arange(nstates),np.arange(nstates))))

fig = plt.figure(); ax = fig.add_subplot(projection='3d')
scat = ax.scatter3D(cz,cx,cy, c=oc_tensor.flatten(),alpha=0.3, s=15,cmap='gnuplot')
fig.colorbar(scat)
plt.show()


#%% PLOT 2

fig = plt.figure(); ax = fig.add_subplot(projection='3d')
#scat = ax.scatter3D(cz,cx,cy, c=(1-10**(-10*oc_tensor)).flatten(),alpha=0.9, s=40,cmap='gnuplot')
scat = ax.scatter3D(cz,cx,cy, c=augment(oc_tensor).flatten(),alpha=0.9, s=10,cmap='gnuplot')
fig.colorbar(scat)
plt.show()


#%%


showdata((1-10**(-5*oc_tensor))[25,...])

showdata(augment(oc_tensor)[20,...])

v=25; showdata(oc_tensor[v,:int(v),:])


#%% NORMAL PARAMS FITTING FOR FIXED v
'''
v=30
mat=oc_tensor[v,...]
showdata(mat)

bounds = np.array(((0, nloci),
                   (0, np.inf),
                   (0, np.inf)))

parlist = np.zeros((1,3))
covlist = np.zeros((1,3,3))

for i in range(nstates):
    slicedata = mat[i,:]
    pars, cov = curve_fit(f = normMult,
                      xdata = np.arange(slicedata.size),
                      ydata = slicedata,
                      p0 = [nstates/2,1,1],
                      bounds = bounds.T,
                      check_finite = True)

    parlist = np.append(parlist,pars[np.newaxis,:], axis=0)
    covlist = np.append(covlist, cov[np.newaxis,:], axis=0)

parlist = parlist[1:]
covlist = covlist[1:]
showlist(parlist)
showlist(parlist[:,1])
showlist(parlist[:,2])
'''

#%% ALL NORMAL PARAMS FITTING FOR ALL v,i
bounds = np.array(((0, nloci),
                   (0, np.inf),
                   (0, np.inf)))

museries=np.zeros((nstates,nstates)) #v,i
sigmaseries=np.zeros((nstates,nstates)) #v,i
kseries=np.zeros((nstates,nstates)) #v,i

for v in range(nstates):
    print('v='+str(v))
    mat=oc_tensor[v,...]
        
    parlist = np.zeros((1,3))
    covlist = np.zeros((1,3,3))
    
    for i in range(nstates):
        slicedata = mat[i,:]
        pars, cov = curve_fit(f = normMult,
                          xdata = np.arange(slicedata.size),
                          ydata = slicedata,
                          p0 = [nstates/2,1,1],
                          bounds = bounds.T,
                          check_finite = True)
    
        parlist = np.append(parlist,pars[np.newaxis,:], axis=0)
        covlist = np.append(covlist, cov[np.newaxis,:], axis=0)
    
    parlist = parlist[1:]
    covlist = covlist[1:]
    
    museries[v,:]=parlist[:,0]
    sigmaseries[v,:]=parlist[:,1]
    kseries[v,:]=parlist[:,2]

#%%
showdata(sigmaseries)
showdata(museries)
showdata(kseries)




#%% PARABOLIC FITTING FOR ALL SIGMA OF v,i
def parabolic(x, a, b, c):
    return a*x**2+b*x+c

def deg3(x, a, b, c, d):
    return a*x**3+b*x**2+c*x+d


sigmacut=sigmaseries.copy()
sigmacut[np.where(sigmacut>20)]=np.NaN
showdata(sigmacut,color='jet')
showsurf(sigmacut,cmap='jet')
showsurf(sigmacut,cmap='jet', type='scatter')


v=20
showlist(sigmacut[:,-1])


series=sigmaseries[v,:]
showlist(list(series))

#%%


p_pars, p_cov = curve_fit(f = parabolic,
                  xdata = np.arange(series.size),
                  ydata = series,
                  p0 = [1,1,1])

fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(np.arange(series.size),series)
ax.plot(np.arange(series.size),parabolic(np.arange(series.size), p_pars[0],p_pars[1],p_pars[2]))
plt.show()


'''
showlist(sigmaseries[int(nloci/2)-3,:])

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
''' 


#%%

def avg(k,x,n):
    sumvar = 0
    for v in range(n+1):
        sumvar += comb(k-2*x, v-x)
    return sumvar



def f(x, a,b,c):
    return  c / a**(b**np.log(x))


def fLL(x, a,b,c):
    return  np.log(c / a**(b**x))


#%%

n=50
k=20 # i+j
nstates=n+1
x=np.arange(nstates)
l=list(map(avg, np.repeat(k,nstates),x, np.repeat(n,nstates)))
#%%
bounds = np.array(((0, np.inf),
                   (0, np.inf),
                   (0, np.inf)))

pars_powL, cov_powL = curve_fit(f = f,
                      xdata = x[1:int(nstates/2)],
                      ydata = l[1:int(nstates/2)],
                      p0 = [4.,2.7,15e20],
                      #bounds = bounds.T,
                      check_finite = False)

print(pars_powL)
#%%
l
list(map(int,np.round(f(x,pars_powL[0],pars_powL[1],pars_powL[2]),0)))

#%%
fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(x,l)
ax.plot(x,f(x,pars_powL[0],pars_powL[1],pars_powL[2]))
plt.show()

#%%

fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(x,l)
ax.plot(x,f(x,pars_powL[0],pars_powL[1],pars_powL[2]))
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()


#%%


fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(x,f(x,4,2.7,15e20))
ax.plot(x,list(map(avg, np.repeat(10,nstates),x, np.repeat(n,nstates))))
ax.plot(x,list(map(avg, np.repeat(20,nstates),x, np.repeat(n,nstates))))
ax.plot(x,list(map(avg, np.repeat(30,nstates),x, np.repeat(n,nstates))))
ax.plot(x,list(map(avg, np.repeat(40,nstates),x, np.repeat(n,nstates))))
ax.plot(x,list(map(avg, np.repeat(50,nstates),x, np.repeat(n,nstates))))
ax.plot(x,list(map(avg, np.repeat(70,nstates),x, np.repeat(n,nstates))))
ax.plot(x,list(map(avg, np.repeat(90,nstates),x, np.repeat(n,nstates))))
ax.plot(x,list(map(avg, np.repeat(100,nstates),x, np.repeat(n,nstates))))
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()

#%%

a=pars_powL[0]
b=pars_powL[1]
c=pars_powL[2]

np.round(f(1,a,b,c),0)
l[1]
f(np.e,a,b,c)
f(np.log(np.log(c)/np.log(a))/np.log(b),a,b,c)

np.log(np.log(c)/np.log(a))/np.log(b)























#%%
n=50
nstates=n+1

mat=np.zeros((nstates,nstates))
oc_tensor_cut = oc_tensor.copy()

for i in range(nstates):
    for v in range(nstates):
        if not 1/2*i+n/8 < v < 1/2*i+n/8*3:
            oc_tensor_cut[v,i,:] = np.NAN




showdata(sigmacut*mat)

fig = plt.figure(); ax = fig.add_subplot(projection='3d')
#scat = ax.scatter3D(cz,cx,cy, c=(1-10**(-10*oc_tensor)).flatten(),alpha=0.9, s=40,cmap='gnuplot')
scat = ax.scatter3D(cz,cx,cy, c=augment(oc_tensor_cut).flatten(),alpha=0.9, s=10,cmap='gnuplot')
fig.colorbar(scat)
plt.show()




#%%  
#*bindist(i + j - 2*x, v - x)

def hypergeometricPMF(N,K,n,k):
    return comb(K,k) * comb(N - K, n - k) / comb(N,n)

def convo(v,n,i,j):
    #prob of getting phenotype v from parent phenotypes i,j with n loci
    sumvar=0
    v=int(v)
    n=int(n)
    i=int(i)
    j=int(j)
    for x in range(i+1):
        sumvar+=hypergeometricPMF(n,i,j,x)*bindist(i + j - 2*x, v - x)
    return sumvar

def averageTest(n,i,j):
    #prob of getting phenotype v from parent phenotypes i,j with n loci
    sumvar=0
    for x in range(i+1):
        #if i+j-2*x >= 0 and j-x >= 0 and n+x-i-j >= 0:
        if i+j-2*x >= 0 :
            #sumvar+= np.math.factorial(i+j-2*x)/(np.math.factorial(i-x)*np.math.factorial(j-x)*np.math.factorial(x)*np.math.factorial(n+x-i-j))
            sumvar+=hypergeometricPMF(n,i,j,x)*np.math.factorial(i+j-2*x)
    return sumvar
    


#%%  
n=100
nstates=n+1

fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(np.arange(nstates),hypergeometricPMF(n,20,10,np.arange(nstates)))
plt.show()



hypergeometricPMF(n,i,j,x)*bindist(i + j - 2*x, np.arange(nstates) - x)

i=30
j=20
x=3
fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(np.arange(nstates),hypergeometricPMF(n,i,j,x)*bindist(i + j - 2*x, np.arange(nstates) - x))
plt.show()




#%%  

n=42
nstates=n+1
mat=np.zeros((nstates,nstates))
i=np.arange(nstates)

for j in range(nstates):
    mat[:,j] = list(map(averageTest, np.repeat(n,nstates),i, np.repeat(j,nstates)))
    
showdata(mat)

#%%  
fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(i,list(map(averageTest, np.repeat(n,nstates),i, np.repeat(5,nstates))))
plt.show()









#%%  




def bindist(n,k,p=0.5):
    return comb(n,k)*p**k*(1-p)**(n-k)



skw=0.3
w = []
for i in range(n+1):
    parent.append(bindist(n,i,skw))
    #parent.append(1/(n+1))
pa
def showfunc(f,xlim=(-5,5),definition=100, **kwargs):
            x= np.linspace(xlim[0],xlim[1],definition)
            fig = plt.figure(); ax = fig.add_subplot(111)
            ax.plot(x,f(x,**kwargs))
            plt.show()
            
            

ntimesteps = 100
n=50
nstates=n+1
skw=0.7
#------------------------
def f(i):
    #return 1+n-i
    return 2**(-i/2)
    #return n**2-i**2

v = np.zeros((ntimesteps, nstates,1))
l = np.zeros((nstates,1))
for i in range(nstates):
    v[0,i] = bindist(n,i,skw)
    
for i in range(nstates):
    l[i] = f(i)

#------------------------

showlist(v[0])
showlist(l)
(s*l).sum()
n*(1-skw)+1




showlist(w)


oc_tensor[k,i,j]


w = v[0]*l
for k in range(nstates):
    v[1,k] = (w.T @ oc_tensor[k,...] @ w) / w.sum()**2

#------------------------

for t in range(1,ntimesteps):
    w = v[t-1]*l
    v[t] = ((w.T @ oc_tensor @ w) / w.sum()**2)[:,0]

oc_tensor[v,i,j]



