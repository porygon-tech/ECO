import numpy as np
import matplotlib.pyplot as plt


#%% ===========================================================================



def pdot (p,a,alpha_P,beta_P,gamma_P,h):
	gP_temp = np.sum(gamma_P*a.T)
	return p*(alpha_P - np.sum(beta_P*p.T) + gP_temp/(1+h*gP_temp))

def adot (a,p,alpha_A,beta_A,gamma_A,h):
	gA_temp = np.sum(gamma_A*p.T)
	return a*(alpha_A - np.sum(beta_A*a.T) + gA_temp/(1+h*gA_temp))

#%%
h=0.0001

n_P = 3
n_A = 4
delta=0.0001
duration=10
ntimesteps=int(duration/delta)
p0 = np.random.randint(50,200,n_P)[:,np.newaxis].astype('float64')
a0 = np.random.randint(50,200,n_A)[:,np.newaxis].astype('float64')

#%%
alpha_P = np.random.rand(n_P)[:,np.newaxis]*2
beta_P  = np.random.rand(n_P,n_P)*0.1
gamma_P = np.random.rand(n_P,n_A)*5

alpha_A = np.random.rand(n_A)[:,np.newaxis]*4
beta_A  = np.random.rand(n_A,n_A)*0.1
gamma_A = np.random.rand(n_A,n_P)*5

#%%
p = p0.copy()
a = a0.copy()
timeseries=np.zeros((n_P+n_A,ntimesteps))
for t in range(ntimesteps):
  p+=pdot(p,a,alpha_P,beta_P,gamma_P,h)*delta
  a+=adot(a,p,alpha_A,beta_A,gamma_A,h)*delta
  timeseries[:n_P ,t] = p.flatten()
  timeseries[ n_P:,t] = a.flatten()


fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(timeseries.T)
plt.show()

#%%
'''
We need to find the conditions (parameters) for global stability
'''
p = p0.copy()
a = a0.copy()

#B is the interaction matrix
B = np.append(np.append(beta_P,-gamma_P, axis=1), np.append(-gamma_A, beta_A,axis=1), axis=0)
v=np.append(p,a,0)
alpha_v = np.append(alpha_P,alpha_A,0)

timeseries=np.zeros((n_P+n_A,ntimesteps))
for t in range(ntimesteps):
  v+=(alpha_v - np.dot(B,v))* delta
  timeseries[:,t] = v.flatten()



fig = plt.figure(); ax = fig.add_subplot(111)
ax.plot(timeseries.T)
plt.show()