import evo
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle5
import bz2

from scipy.optimize import curve_fit

root = Path(".")
my_path = root / 'data/obj'


with bz2.BZ2File(my_path / 'avg.obj', 'rb') as f:
	avg = pickle5.load(f)


def ema(series, gamma = 0.1):
  #exponential moving average to smooth data
  len = series.size
  y = series.copy()
  expMA=series[0]
  for i in range(len):
    expMA = gamma*y[i] + (1-gamma)*expMA
    y[i] = expMA
  #y[0]=series[0]
  return(y)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
nindivs = 100
nloci = 20
ps = (500,500+nloci)
ntries, duration = avg.shape
#-------------------------------------

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
x=np.arange(duration)
ax.plot(x,avg.T)
plt.ylim(ps)
plt.show()



d = np.diff(avg, axis=1)
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
ax.scatter(avg[:,:-1].flatten(), d.flatten(),s=0.25)
plt.show()



avgSmooth = np.zeros(avg.shape)
for i in range(ntries):
	avgSmooth[i,:] = ema(avg[i,:],gamma=0.2)



i=3
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
x=np.arange(duration)
ax.plot(x,avg[i,:])
ax.plot(x,avgSmooth[i,:])
plt.ylim(ps)
plt.show()



ds = np.diff(avgSmooth, axis=1)
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
ax.scatter(avgSmooth[:,:-1].flatten(), d.flatten(),s=0.5)
plt.show()



fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
x=np.arange(duration)
ax.plot(x,avgSmooth.T)
plt.ylim(ps)
plt.show()



fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
ax.scatter(d.flatten()*100,  avg[:,:-1].flatten(),s=0.5)
ax.scatter(ds.flatten()*100, avgSmooth[:,:-1].flatten(),s=0.3)
ax.plot(x,avg.T,       color='black', linewidth=0.25)
ax.plot(x,avgSmooth.T, color='red',   linewidth=0.5)
plt.ylim(ps)
plt.show()





#==============================

def logistic(x,a=0,b=1,k=1):
	return 2*a - b + (2*b - 2*a) / (1 + np.exp(-k*x))

parlist = np.zeros((1,3))
covlist = np.zeros((1,3,3))

for i in range(ntries):
	pars, cov = curve_fit(f = logistic,
	                      xdata = x,
	                      ydata = avg[i,:],
	                      p0 = [ps[0], ps[1], 1],
	                      #bounds = bounds.T,
	                      check_finite = True)
	parlist = np.append(parlist,pars[np.newaxis,...], axis=0)
	covlist = np.append(covlist, cov[np.newaxis,...], axis=0)


fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
for i in range(ntries):
	ax.plot(x,avg[i,:], color='red',   linewidth=0.5)
	ax.plot(x,logistic(x, parlist[i,0],parlist[i,1],parlist[i,2]), color='blue',   linewidth=0.5)

plt.ylim(ps)
plt.show()



fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
ax.scatter(parlist[:,0],parlist[:,2])
plt.show()


	

parlist[:,2].var()
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
for i in range(ntries):
	#ax.plot(x,logistic(x, avg[i,0],ps[1]-1,parlist[:,2].mean()), color='blue',   linewidth=0.15, alpha=0.5)
	#ax.plot(x,avgSmooth[i,:],  color='red',   linewidth=0.25, alpha=0.6)
	ax.plot(x,logistic(x, parlist[i,0],parlist[i,1],parlist[i,2]), color='blue',   linewidth=0.5, alpha=0.5)
	ax.plot(x,logistic(x, parlist[i,0],parlist[i,1],parlist[:,2].mean()), color='green',   linewidth=0.5, alpha=0.5)

plt.ylim(ps)
plt.show()


