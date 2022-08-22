import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import sparse
from scipy.sparse import linalg
from scipy.optimize import minimize, fsolve

'''
Rohr, R. P., Saavedra, S., & Bascompte, J. (2014). On the structural stability of mutualistic systems. Science, 345(6195), 1253497.
'''
#%%
#=========================================================================================================
def showdata(mat, color=plt.cm.gnuplot, symmetry=False):
	mat = np.copy(mat)
	if symmetry:
		top = np.max([np.abs(np.nanmax(mat)),np.abs(np.nanmin(mat))])
		plt.imshow(mat.astype('float32'), interpolation='none', cmap='seismic',vmax=top,vmin=-top)
	else:
		plt.imshow(mat.astype('float32'), interpolation='none', cmap=color)
	plt.colorbar()
	plt.show()

def to_square(m):
	rows, cols = m.shape
	uL = np.zeros((rows,rows))
	dR = np.zeros((cols,cols))
	Um = np.concatenate((uL , m ), axis=1)
	Dm = np.concatenate((m.T, dR), axis=1)
	return(np.concatenate((Um,Dm), axis=0))

def table_to_graph(incidence, binary=False):
		p=incidence
		if binary:
			p[p>0]=1
		full = to_square(p)
		sparse_full = sparse.csr_matrix(full)
		G = nx.from_scipy_sparse_matrix(sparse_full, parallel_edges=False)
		return(G)


def showgraph(incidence):
	g=table_to_graph(incidence)
	top = np.array(g.nodes)[:incidence.shape[0]]
	pos = nx.bipartite_layout(g,top, align='horizontal')
	nx.draw(g, pos, width=incidence.flatten(), node_size=10)
	plt.show()


def spectralRnorm(incidence):
	incidence_norm = incidence/incidence.max()
	adjacency_norm = to_square(incidence_norm)
	L = sparse.csr_matrix(adjacency_norm)
	sR = sparse.linalg.eigs(L,k=1,which='LM', return_eigenvectors=False)
	return(np.abs(sR[0]) / np.sqrt(incidence.size))

def spectralR(incidence):
	adjacency= to_square(incidence)
	L = sparse.csr_matrix(adjacency)
	sR = sparse.linalg.eigs(L,k=1,which='LM', return_eigenvectors=False)
	return(np.abs(sR[0]))


def sort_degrees(a):
    s = a[:,np.argsort(a.sum(axis=0))  ]
    s = s[  np.argsort(s.sum(axis=1)),:]
    return np.flip(s)

def generateNested(m,n, strength = 0.5):
	a = np.zeros((m,n))
	a[0,:] = np.random.choice(0,1,n)
    
def is_symmetric(m):
    return (m==m.T).all()
    
'''
def is_pos_def(m):
    return np.all(np.real(np.linalg.eigvals(m)) > 0)
 
having all positive real parts of the eigenvalues is necessary but not sufficient for being positive definite
A more complete function is provided below

'''
def is_pos_def(m, maxiter=1000, z0='rand', fullresult=False):
    #tells if matrix is positive definite
    if m.shape[0] != m.shape[1]:
        raise Exception("Matrix is not square") 
    elif is_symmetric(m): #symmetry testing
        return np.all(np.linalg.eigvals(m) > 0)
    else:
        def f(z):
            z=np.array(list(z))[:,np.newaxis]
            return np.dot(np.dot(z.T, m),z)
        if z0=='rand':
            z0 = list(np.random.rand(m.shape[0]))
        #bounds = np.repeat([[0,None]],repeats=m.shape[0],axis=0)
        #constraints for a non-zero vector solution
        cons = ({'type': 'ineq', 'fun': lambda z:  np.sum(np.abs(z))})
        minV = minimize(f, z0, method='COBYLA', options={'maxiter' : maxiter},constraints=cons);
        
        if fullresult:
            return minV
        elif minV['success'] or minV['status'] == 2:
            return minV['fun']+0 > 0 
            #return minV
        else:        
            #return minV
            raise Exception(minV['message']) 


#=========================================================================================================
#%% randomised model


n_A = 1
n_P = 1
h=0 # if h=0, the model becomes linear

#alpha is the growth rate vector
alpha_A = np.random.normal(2,0.1,(n_A, 1))
alpha_P = np.random.normal(2,0.1,(n_P, 1))

#beta is the intra-guild competition matrix
beta_A  = np.random.normal(0.05,0.01,(n_A, n_A))
beta_P  = np.random.normal(0.05,0.01,(n_P, n_P))

#gamma is the mutualistic effect matrix
gamma_A = np.random.normal(1,0.5,(n_A, n_P))
gamma_P = np.random.normal(1,0.5,(n_P, n_A))
#%%
#a = 100 + np.zeros((n_A, 1))
#p = 100 + np.zeros((n_P, 1))

ia = np.random.choice(np.arange(10,100), n_A)[:,np.newaxis]
ip = np.random.choice(np.arange(10,100), n_P)[:,np.newaxis]

#%%
a = ia
p = ip

r=0.01 #resolution
duration = 2000
data=np.zeros((n_A + n_P,duration))
for t in range(duration):
	dp = p*(alpha_P - np.dot(beta_P,p) + np.dot(gamma_P,alpha_A)/(1+h*np.dot(gamma_P,alpha_A)))
	da = a*(alpha_A - np.dot(beta_A,a) + np.dot(gamma_A,alpha_P)/(1+h*np.dot(gamma_A,alpha_P)))
	a = a + r*da 
	p = p + r*dp 
	data[:,t]=np.append(a,p)


#%%
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
x=np.arange(duration)*r
ax.plot(x,data.T)
plt.show()

#%%
'''
g=table_to_graph(beta_A)
nx.draw(g, width=0.1, node_size=10)
plt.show()
'''
showgraph(gamma_A)

#%%
spectralRnorm(beta_A)

#%%=======================================================================================================
#=========================================================================================================
#=========================================================================================================
#%% simplified model


n_A = 1
n_P = 1

h       = 0   # saturating constant of the beneficial effect of mutualisms, aka handling time. If h=0, the model becomes linear
rho     = 0.2 # interspecies competition
delta   = 0.1 # mutualistic trade off
gamma_0 = 0.2 # mutualistic strength, i.e. gamma_0
'''
The mutualistic trade off (delta) modulates the extent to which a species that interacts with few other
species does it strongly, whereas a species that interacts with many partners does it weakly.
'''
#------------------------------------

beta_A  = rho + np.zeros((n_A, n_A))
beta_P  = rho + np.zeros((n_P, n_P))

for i in range(n_A):
	beta_A[i,i]=1

for i in range(n_P):
	beta_P[i,i]=1

prob=0.4
y = np.random.choice((0,1),(n_A,n_P), p=(1-prob, prob))
while np.any(y.sum(0)==0) or np.any(y.sum(1)==0):
    y = np.random.choice((0,1),(n_A,n_P), p=(1-prob, prob));
showdata(y)

#k_A=np.dot(y,np.ones(y.shape[1])) # row sums, i.e. node degrees, y.sum(1)
k_A=y.sum(1)[:,np.newaxis]
k_P=y.sum(0)[:,np.newaxis]

gamma_A = (gamma_0*y  )/k_A**delta
gamma_P = (gamma_0*y.T)/k_P**delta

#alpha is the growth rate vector
alpha_A = np.random.normal(2,0.1,(n_A, 1))
alpha_P = np.random.normal(3,0.1,(n_P, 1))

#B is the interaction matrix
B = np.append(np.append(beta_P,-gamma_P, axis=1), np.append(-gamma_A, beta_A,axis=1), axis=0); showdata(B,symmetry=True)

#a is the growth rate vector
a = np.append(alpha_P,alpha_A,0)

#%% check for Lyapunov stability
if not (np.real(sparse.linalg.eigs(B,k=1,which='SM', return_eigenvectors=False))[-1] > 0):
    print("NOT ALL eigenvalues in the interaction matrix have positive real part, so no global stability can be guaranteed for feasible equilibrium points.")
else:
    print("ALL eigenvalues in the interaction matrix have positive real part, so a global stability can be guaranteed for feasible equilibrium points.")
#%% 
is_pos_def(B,maxiter=1000000)
#%% 
is_symmetric(B)

#%% ===========================================================================





def d(x,nPlants):
    p=x[:nPlants ]
    a=x[ nPlants:]
    a=a.reshape(a.shape[0],1)
    p=p.reshape(p.shape[0],1)
    dp = p*(alpha_P - np.dot(beta_P,p) + np.dot(gamma_P,alpha_A)/(1+h*np.dot(gamma_P,alpha_A)))
    da = a*(alpha_A - np.dot(beta_A,a) + np.dot(gamma_A,alpha_P)/(1+h*np.dot(gamma_A,alpha_P)))
    dx=np.append(dp,da)
    return dx

#%%
ia = np.random.choice(np.arange(10,100), n_A)[:,np.newaxis]
ip = np.random.choice(np.arange(10,100), n_P)[:,np.newaxis]

#%%
a = ia
p = ip

r=0.01 #resolution
duration = 50000
data=np.zeros((n_A + n_P,duration))
for t in range(duration):
    x=np.append(p,a)
    dx = d(x,n_P)
    dx = dx[:,np.newaxis]
    dp=dx[:n_P]
    da=dx[n_P:]
	
    a = a + r*da 
    p = p + r*dp 
    data[:,t]=np.append(p,a)

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
x=np.arange(duration)*r
ax.plot(data[0],data[1])
plt.show()

#%% 
ia = np.random.choice(np.arange(10,100), n_A)[:,np.newaxis]
ip = np.random.choice(np.arange(10,100), n_P)[:,np.newaxis]
fixpoint=fsolve(d,x0=np.append(ip,ia),args=n_P);fixpoint
np.isclose(d(fixpoint,n_P), [0.0, 0.0])




#%%
































#%%

ntries = 100

n_A = 1
n_P = 1

h       = 0   # saturating constant of the beneficial effect of mutualisms, aka handling time. If h=0, the model becomes linear
rho     = 0.2 # interspecies competition
delta   = 0.1 # mutualistic trade off
gamma_0 = 0.2 # mutualistic strength, i.e. gamma_0

'''
The mutualistic trade off (delta) modulates the extent to which a species that interacts with few other
species does it strongly, whereas a species that interacts with many partners does it weakly.
'''
#------------------------------------

#%% check for Lyapunov stability
if not (np.real(sparse.linalg.eigs(B,k=1,which='SM', return_eigenvectors=False))[-1] > 0):
    print("NOT ALL eigenvalues in the interaction matrix have positive real part, so no global stability can be guaranteed for feasible equilibrium points.")
else:
    print("ALL eigenvalues in the interaction matrix have positive real part, so a global stability can be guaranteed for feasible equilibrium points.")
#%% 
is_pos_def(B,maxiter=1000000)
#%% 
is_symmetric(B)













