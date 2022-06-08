import numpy as np 
import matplotlib.pyplot as plt
#from itertools import product	

def showdata(mat, color=plt.cm.gnuplot, symmetry=False):
	mat = np.copy(mat)
	if symmetry:
		top = np.max([np.abs(np.nanmax(mat)),np.abs(np.nanmin(mat))])
		plt.imshow(mat.astype('float32'), interpolation='none', cmap='seismic',vmax=top,vmin=-top)
	else:
		plt.imshow(mat.astype('float32'), interpolation='none', cmap=color)
	plt.colorbar()
	plt.show()



nindivs = 5
nloci = 5
skew=0.5
population = np.random.choice((0,1),(nindivs,nloci), p=(skew, 1-skew))
showdata(population)

m= nindivs
n= nloci

a = np.zeros((m+n,m*n))
for i in range(m):
	a[i, n*i:n*(i+1)] = 1
	#print(n*i,n*(i+1))

for j in range(n):
	for j2 in range(m):
		a[m+j , j+j2*n] = 1
		#print(m+j , j+j2*n)

indepTerms = np.append(population.sum(axis=1),population.sum(axis=0))[:,np.newaxis]
nvar = m*n
amp = np.concatenate((a,indepTerms),axis=1)




k=population.sum()
solutions = np.zeros((1,nvar))
for i in range(2**k-1, 2**nvar-2**(nvar-k)+1):
	code = format(i,'b')
	code = np.array([int(c) for c in code])
	code = np.append(np.zeros(nvar - len(code)), code)[:,np.newaxis]

	if (code.sum() == k):
	if (((a*code.T).sum(axis=1)[:,np.newaxis] != indepTerms).sum() == 0):
		solutions = np.append(solutions,code.T, axis=0)

solutions = solutions[1:,:]
print('there exist ' + str(solutions.shape[0]) + ' real solutions')



showdata((solutions.sum(axis=0)/solutions.shape[0]).reshape(nindivs,nloci),symmetry=True)

























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





class population(object):
	"""docstring for population"""
	def __init__(self, nindivs, nloci, skew=0.5):
		#super(population, self).__init__()
		self.nindivs = nindivs
		self.m = nindivs
		self.nloci = nloci
		self.n = nloci
		self.skew = skew
		self.mtx = np.random.choice((0,1),(nindivs,nloci), p=(skew, 1-skew))
		# a: the matrix of coefficients for the linear system of eqs
		self.a = np.zeros((self.m+self.n, self.m*self.n)) 
		for i in range(self.m):
			self.a[i, self.n*i:self.n*(i+1)] = 1
		for j in range(self.n):
			for j2 in range(self.m):
				self.a[self.m+j , j+j2*self.n] = 1
		# indepTerms: the vector of independent terms in the system of eqs
		self.indepTerms = np.append(self.mtx.sum(axis=1),self.mtx.sum(axis=0))[:,np.newaxis]
		# amp: extended matrix
		self.amp = np.concatenate((self.a,self.indepTerms),axis=1)
	def show(self):
		showdata(self.mtx)
	def solutionSpace(self, silent=False):
		#explores alternative matrices with same column and row sums
		k=self.mtx.sum()
		nvar = self.m*self.n
		if not silent:
			if nvar > 20:
				print('ALERT: system has {0} variables, which gives {1} possible combinations.'.format(nvar, 2**nvar))
				cont = input('This can take a while to compute. Continue? (y/n): ')
				if cont == 'y':
					pass
				else:
					return
		solutions = np.zeros((1,nvar))
		for i in range(2**k-1, 2**nvar-2**(nvar-k)+1):
			code = format(i,'b')
			code = np.array([int(c) for c in code])
			code = np.append(np.zeros(nvar - len(code)), code)[:,np.newaxis]
			if (code.sum() == k):
				if (((self.a*code.T).sum(axis=1)[:,np.newaxis] != self.indepTerms).sum() == 0):
					solutions = np.append(solutions,code.T, axis=0)
					print(code.T.astype('int'))
		solutions = solutions[1:,:]
		print('there exist ' + str(solutions.shape[0]) + ' real solutions')
		return(solutions)





#================================================================

nindivs = 5
nloci = 3

pop = population(nindivs,nloci);pop.show()

s = pop.solutionSpace()

probs = (s.sum(axis=0)/s.shape[0]).reshape(nindivs,nloci)
k = pop.mtx.sum()
c=probs.sum(axis=0)/k
r=probs.sum(axis=1)[:,np.newaxis]/k

x = np.repeat(r, nloci).flatten()
y = np.tile(  c, nindivs).flatten()
z = probs.flatten()

fig = plt.figure(); ax = fig.add_subplot(projection='3d')
ax.scatter3D(x,y, probs.flatten())
surf = ax.plot_trisurf(x,y,z, cmap=plt.cm.jet, linewidth=0)
fig.colorbar(surf)
plt.show()

#================================================================



runs = 100
nindivs = 4
nloci = 4
x = []
y = []
z = []
for _ in range(runs):
	pop = population(nindivs,nloci)
	k = pop.mtx.sum()
	s = pop.solutionSpace(silent=True)
	probs = (s.sum(axis=0)/s.shape[0]).reshape(nindivs,nloci)
	c=probs.sum(axis=0)/k
	r=probs.sum(axis=1)[:,np.newaxis]/k
	x = np.append(x, np.repeat(r, nloci).flatten())
	y = np.append(y, np.tile(  c, nindivs).flatten())
	z = np.append(z, probs.flatten())




showdata(probs,symmetry=True)
showdata(pop.mtx + probs -1,symmetry=True)

showdata((s[2:4,:].sum(axis=0)).reshape(nindivs,nloci))


# check if any row or col is all zero
#((pop.mtx.sum(axis=0)==0).sum() + (pop.mtx.sum(axis=1)==0).sum()).sum() !=0







fig = plt.figure(); ax = fig.add_subplot(projection='3d')
ax.scatter3D(x,y,z, c=z, cmap=plt.cm.jet)
#surf = ax.plot_trisurf(x,y,z, cmap=plt.cm.jet, linewidth=0)
#fig.colorbar(surf)
plt.show()



x = np.linspace(0,5,20)
y = np.linspace(0,5,20)
gx,gy = np.meshgrid(x,y)
x, y = gx.flatten(), gy.flatten()
z = np.sin(x) + np.cos(y)

fig = plt.figure(); ax = fig.add_subplot(projection='3d')
ax.scatter3D(x,y,z, c=z, cmap=plt.cm.jet)
plt.show()

fig = plt.figure(); ax = fig.add_subplot(projection='3d')
surf = ax.plot_trisurf(x,y,z, cmap=plt.cm.jet, linewidth=0)
fig.colorbar(surf)
plt.show()

