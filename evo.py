import numpy as np 
import matplotlib.pyplot as plt

#https://www.geeksforgeeks.org/decorators-in-python/
#https://www.geeksforgeeks.org/passing-function-as-an-argument-in-python/
#https://stackoverflow.com/questions/14916284/in-class-object-how-to-auto-update-attributes

def showdata(mat, color=plt.cm.gnuplot, symmetry=False):
	mat = np.copy(mat)
	if symmetry:
		top = np.max([np.abs(np.nanmax(mat)),np.abs(np.nanmin(mat))])
		plt.imshow(mat.astype('float32'), interpolation='none', cmap='seismic',vmax=top,vmin=-top)
	else:
		plt.imshow(mat.astype('float32'), interpolation='none', cmap=color)
	plt.colorbar()
	plt.show()

def pois(x,l=1):
	return (l**x * np.math.exp(-l)) / np.math.factorial(x)

def norm(x, mu, sigma):
	return 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2)



class population(object):
	"""docstring for population"""
	def __init__(self, nindivs, nloci, skew=0.5, phenoSpace = [1, 3], matrix = 'None'):
		#super(population, self).__init__()
		self._nindivs = nindivs
		self._m = nindivs
		self._nloci = nloci
		self._n = nloci
		self.phenoSpace = phenoSpace
		self.skew = skew
		if matrix == 'None':
			self.mtx = np.random.choice((0,1),(nindivs,nloci), p=(skew, 1-skew))
		else:
			self.mtx = matrix

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
		

	'''
			@property
			def list(self):
				return self._list
			@list.setter
			def list(self, val):
				self._list = val
				self._listsquare = [x**2 for x in self._list ]

			@property
			def listsquare(self):
				return self._listsquare
			@listsquare.setter
			def listsquare(self, val):
				self.list = [int(pow(x, 0.5)) for x in val]
	'''
	# a property is special attribute that can have two features:
	#   1: being read-only
	#   2: when its value is set, modifying other properties


	@property #property that modifies other attributes through its setter
	def mtx(self):
		return self._mtx # underscored attributes only should be accessed internally and never from outside

	@mtx.setter
	def mtx(self, mtx):
		self._mtx = mtx
		self._phenotypes = mtx.sum(axis=1)/self.nloci * np.diff(self.phenoSpace)[0] + self.phenoSpace[0]
		self._relativeFitnessValues = self.fitness(self.phenotypes)/self.fitness(self.phenotypes).sum()
		#setters dont need to have a return
	

	@property #read-only property
	def nindivs(self):
		return self._nindivs

	@property #read-only property
	def m(self):
		return self._m

	@property #read-only property
	def nloci(self):
		return self._nloci

	@property #read-only property
	def n(self):
		return self._n

	@property #read-only property
	def phenotypes(self):
		return self._phenotypes

	@property #read-only property
	def relativeFitnessValues(self):
		return self._relativeFitnessValues




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

	def hist(self, nbins='default'):
		if nbins == 'default':
			nbins = self.nloci + 1
		elif nbins == 'reduced':
			nbins = int(self.nloci/10)
		n, bins, patches = plt.hist(self.phenotypes, nbins, density=True, facecolor='g', alpha=0.75)
		plt.xlim(self.phenoSpace)
		plt.grid(True)
		plt.show()

	def fitness(self, x): #fitness landscape function (1 trait)
		o = np.diff(self.phenoSpace)*0.25
		return (norm(x, self.phenoSpace[0]+o, np.diff(self.phenoSpace)*0.1) + norm(x, self.phenoSpace[1]-o, np.diff(self.phenoSpace)*0.1))/2

	def showfitness(self):
		fig = plt.figure(); ax = fig.add_subplot(111)
		x=np.linspace(self.phenoSpace[0], self.phenoSpace[1], self.nloci)
		#y=norm(x,0,1)
		y=self.fitness(x)
		ax.plot(x,y)
		#ax.legend(loc='center right')
		ax.set_xlabel('phenotype value', labelpad=10)
		ax.set_ylabel('relative fitness', labelpad=10)
		ax.set_xlim(self.phenoSpace)
		plt.show()

	def set_fitnessLandscape(self, func):
		self.fitness = func
		self.relativeFitnessValues = self.fitness(self.phenotypes)/self.fitness(self.phenotypes).sum()

	def sexualPreference(self,x,y,k=1):
		return 1/(1+(x-y)**2/k) #preferring similar phenotypes

	def set_sexualPreference(self, func):
		self.sexualPreference = func

	def intraCompetition(self,x,y,k=1):
		return 1/(1+(x-y)**2/k) #similar phenotypes compete strongly

	def set_intraCompetition(self, func):
		self.intraCompetition = func

	def makeChildren(self,k, mutRate=0):
		children = np.zeros((self.nindivs,self.nloci))
		#pA = np.array([self.phenotypes]*self.nindivs).flatten()
		#pB = np.repeat(self.phenotypes, self.nindivs)
		pA,pB = np.meshgrid(self.phenotypes,self.phenotypes)
		pref=self.sexualPreference(pA,pB, k) 
		pref=pref.reshape((self.nindivs,self.nindivs))
		np.fill_diagonal(pref,0)
		pref = pref/np.sum(pref)
		#pref=np.ones((nindivs,nindivs))
		#
		rc = self.intraCompetition(pA,pB,0.005)
		rc=rc.reshape((self.nindivs,self.nindivs))
		np.fill_diagonal(rc,0)
		sat = np.sum(rc,axis=0)/np.sum(rc) #saturation
		P = ((pref * self.relativeFitnessValues*sat**2).T * self.relativeFitnessValues*sat**2).T
		P = P/np.sum(P)
		#
		#P = ((pref*relativeFitnessValues).T * relativeFitnessValues).T
		#P = P/np.sum(P)
		Pf = P.flatten()[:] 
		parents = np.random.choice(self.nindivs**2, self.nindivs, p=Pf)
		parents = np.array(np.unravel_index(parents, (self.nindivs,self.nindivs)))
		for i in range(self.nindivs):
			#ncrossovers = 1+np.random.poisson() #at least 1 crossover is forced to happen
			ncrossovers = np.random.poisson()
			crossover_pos = np.random.choice(range(1,self.nloci-2), ncrossovers)
			crossover_pos = np.sort(crossover_pos)
			crossover_pos = np.append(crossover_pos, self.nloci-1)
			start = 0
			parentSwitch = 0
			for end in crossover_pos:
				children[i,start:end] = self.mtx[parents[parentSwitch,i], start:end]
				start = end
				if parentSwitch == 0:
					parentSwitch = 1
				else:
					parentSwitch = 0
		offspring = population(self.nindivs,self.nloci, phenoSpace=self.phenoSpace, matrix=children);
		return(offspring)



	
'''
	def makeChildren(self):		
		relativeFitnessValues = self.fitness(self.phenotypes)/self.fitness(self.phenotypes).sum()
		children = np.zeros((self.nindivs,self.nloci))
		for i in range(self.nindivs):
			ncrossovers = 1+np.random.poisson() #at least 1 crossover is forced to happen
			#ncrossovers = 1+np.random.poisson(2)
			parents = np.random.choice(self.nindivs, 2, replace=False ,p=relativeFitnessValues)
			crossover_pos = np.random.choice(range(1,self.nloci-2), ncrossovers)
			crossover_pos = np.sort(crossover_pos)
			crossover_pos = np.append(crossover_pos, self.nloci-1)
			start = 0
			parentSwitch = 0
			for end in crossover_pos:
				children[i,start:end] = population[parents[parentSwitch], start:end]
				start = end
				if parentSwitch == 0:
					parentSwitch = 1
				else:
					parentSwitch = 0
		return(children)
'''

'''

try:
	undefined_variable
except NameError as e:
	# Do whatever you want if the variable is not defined yet in the program.
	print('ERROOOOR: ' + str(e))
'''
