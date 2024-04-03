import numpy as np 
import matplotlib.pyplot as plt
from copy import deepcopy
import os

I = np.newaxis

#https://www.geeksforgeeks.org/decorators-in-python/
#https://www.geeksforgeeks.org/passing-function-as-an-argument-in-python/
#https://stackoverflow.com/questions/14916284/in-class-object-how-to-auto-update-attributes

def showdata(mat, color=plt.cm.gnuplot, symmetry=False,colorbar=False):
    mat = np.copy(mat)
    if symmetry:
        top = np.max([np.abs(np.nanmax(mat)),np.abs(np.nanmin(mat))])
        plt.imshow(mat.astype('float32'), interpolation='none', cmap='seismic',vmax=top,vmin=-top)
    else:
        plt.imshow(mat.astype('float32'), interpolation='none', cmap=color)
    if colorbar: plt.colorbar()
    plt.show()

def pois(x,l=1):
    return (l**x * np.math.exp(-l)) / np.math.factorial(x)

def norm(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2)


from scipy.special import comb  
def bindist(n,k,p=0.5):
    return comb(n,k)*p**k*(1-p)**(n-k)
# '''
# def nofixation(M):
#     maf=1/2-np.abs(M.mean(0) -1/2)
#     despair=0
#     while np.any(maf==0):
#         while np.any(M.mean(0)==1):
#             maf=1/2-np.abs(M.mean(0) -1/2)
#             donor_locus = np.argmax(maf)
#             receptor_locus = np.random.choice(np.where(M.mean(0)==1)[0])
#             swapper_id = np.random.choice(np.where(M[:,donor_locus]==0)[0])
#             M[swapper_id,donor_locus   ]=1
#             M[swapper_id,receptor_locus]=0
#             print('fix 1')
#         while np.any(M.mean(0)==0):
#             maf=1/2-np.abs(M.mean(0) -1/2)
#             donor_locus = np.argmax(maf)
#             receptor_locus = np.random.choice(np.where(M.mean(0)==0)[0])
#             swapper_id = np.random.choice(np.where(M[:,donor_locus]==1)[0])
#             M[swapper_id,donor_locus   ]=0
#             M[swapper_id,receptor_locus]=1
#             print('fix 0')
#             '''
#             # if despair > 20:
#             #     M[swapper_id,receptor_locus]=1
#             # despair+=1
#             '''
#         maf=1/2-np.abs(M.mean(0) -1/2)
# '''
def mutate(mtx, rate=0.001):
    if type(mtx) == population: mtx = mtx.mtx
    mutations = np.random.choice((0,1),mtx.shape, p=(1-rate, rate))
    return(np.logical_xor(mutations, mtx))


def generate_mut_matrix(nstates,mu=0):
    n= nstates-1
    if mu==0:
        temp = np.zeros((nstates,nstates))
        np.fill_diagonal(temp,1)
        return temp
    else:
        return np.array([ list(map(lambda i: sum(list(map(lambda k: bindist(i,i-k,mu) * bindist(n-i,b-k,mu), list(range(i+1))))), list(range(nstates)))) for b in list(range(nstates))])
    
def generate_h_tensor(n):
    print('Generating inheritance tensor for n={0}...'.format(n))
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
    
    nstates=n+1
    x = np.arange(nstates)
    y = np.arange(nstates)
    gx,gy = np.meshgrid(x,y)
    x, y = gx.flatten(), gy.flatten()

    n_list=np.repeat(n,nstates**2)
    oc_tensor = np.zeros((nstates,nstates,nstates))

    for v in range(nstates):
        print('v='+str(v))
        v_list=np.repeat(v,nstates**2)
        z = list(map(oc, v_list,n_list,x,y))
        mat=np.array(z).reshape((nstates,nstates)).astype('float32')
        oc_tensor[v,...] = mat[np.newaxis,...]
    return(oc_tensor)
    

def flat_nDist(B): #copied from matriX
    '''
    keeps degree distribution along one axis, sets a random uniform to the other one
    '''
    m,n = B.shape
    B=B[:,np.argsort(B.sum(axis=0))[::-1]]
    B=B[np.argsort(B.sum(axis=1))[::-1],:]
    
    r = list(map(tuple, list(map(np.random.choice, [range(n)]*m, B.sum(axis=1).astype(int), [False]*m))))
    
    Br = np.zeros((m,n))

    for i in range(m):
        Br[(tuple([i]*len(r[i])),r[i])] = 1
    
    return(Br)


class transformations:
    def negativeSaturator( x,v=1):
        r = x/ (1-np.exp(-v*x))
        r[np.where(x==0)] = 1/v
        return r
    
#class eco:



class population(object):
    """docstring for population"""
    def __init__(self, nindivs, nloci, skew=0.5, phenoSpace = [1, 3], matrix = 'None'):
        #super(population, self).__init__()
        self._nindivs = nindivs
        self._nloci = nloci
        self.phenoSpace = phenoSpace
        self.skew = skew
        self._history = np.zeros((1,self.nloci+1))
        self._popsize_history = np.zeros(1)
        if matrix == 'None':
            self.mtx = np.random.choice((0,1),(nindivs,nloci), p=(1-skew, skew))
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
        
    # a property is special attribute that can have two features:
    #   1: being read-only
    #   2: when a new value is set, modifying other properties

    @property #property that modifies other attributes through its setter
    def mtx(self):
        return self._mtx # underscored attributes only should be accessed internally and never from outside

    @mtx.setter
    def mtx(self, mtx):
        self._mtx = mtx
        self._nindivs, self._nloci = self.mtx.shape
        self._phenotypes = mtx.sum(axis=1)/self.nloci * np.diff(self.phenoSpace)[0] + self.phenoSpace[0]
        self._fitnessValues = self.fitness(self.phenotypes)
        self._relativeFitnessValues = self.fitnessValues/self.fitnessValues.sum()
        #self._relativeFitnessValues = self.fitness(self.phenotypes)/self.fitness(self.phenotypes).sum()

        unique, counts = np.unique(self.phenotypes, return_counts=True)
        pos = (unique - self.phenoSpace[0])*self.nloci/np.diff(self.phenoSpace)[0] 
        temp=np.zeros((1,self.n+1))
        temp[0,(pos).astype(int)] = counts/self.m
        if self.history.sum() != 0:
            self.history_append(temp)
            self.popsize_append(self.nindivs)
        else:
            self.history[0,:] = temp
            self.popsize_history[0]=self.nindivs

        #setters dont need to have a return
    

    @property #read-only property
    def nindivs(self):
        return self._nindivs

    @property #read-only property
    def m(self):
        return self._nindivs

    @property #read-only property
    def nloci(self):
        return self._nloci

    @property #read-only property
    def n(self):
        return self._nloci

    @property #read-only property
    def phenotypes(self):
        return self._phenotypes

    @property #read-only property
    def fitnessValues(self):
        return self._fitnessValues
    
    @property #read-only property
    def relativeFitnessValues(self):
        return self._relativeFitnessValues
    
    @property #read-only property
    def history(self):
        return self._history
    
    @property #read-only property
    def popsize_history(self):
        return self._popsize_history
    


    def history_append(self,newhist):
        self._history = np.concatenate((self.history,newhist),axis=0)

    def popsize_append(self,m):
        self._popsize_history = np.append(self.popsize_history,m)
    

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
        return((n, bins, patches))

    def fitness(self, x): #fitness landscape function (1 trait)
        o = np.diff(self.phenoSpace)*0.25
        return (norm(x, self.phenoSpace[0]+o, np.diff(self.phenoSpace)*0.1) + norm(x, self.phenoSpace[1]-o, np.diff(self.phenoSpace)*0.1))/2

    def set_fitnessLandscape(self, func):
        if func == 'flat':
            def f(x):
                return 1.
            self.fitness = f
        else:
            self.fitness = func
        self._fitnessValues = self.fitness(self.phenotypes)
        self._relativeFitnessValues = self.fitnessValues/self.fitnessValues.sum()
        #self._relativeFitnessValues = self.fitness(self.phenotypes)/self.fitness(self.phenotypes).sum()
        #print('The fitness landscape has changed')

    def showfitness(self, distbins=False):
        
            fig = plt.figure(); ax = fig.add_subplot(111)
            n=100
            x=np.linspace(self.phenoSpace[0], self.phenoSpace[1], self.nloci)
            #y=norm(x,0,1)
            y=self.fitness(x)
            ax.plot(x,y)
            if distbins:        
                ax.hist(self.phenotypes, distbins, density=True, facecolor='r', alpha=0.5)

            ax.set_xlabel('phenotype value', labelpad=10)
            ax.set_ylabel('fitness', labelpad=10)
            ax.set_xlim(self.phenoSpace)
            plt.show()

    def reproduce(self, ngenerations, agentBased=True, fixedSize=True, 
                  verbose=False,
                  nofix=False,
                  mu=0,
                  K=1000):
        '''
        agentBased: tells if the simulation should be made with an agent-based
            model. otherwise, it will use analytical calculations
        '''
        if agentBased:
            offspring = deepcopy(self)
            m = offspring.m
            n = offspring.n
            v_genData=np.zeros((ngenerations, n+1))
            v_phen = offspring.phenotypes #self.mtx.sum(axis=1)
            unique, counts = np.unique(v_phen, return_counts=True)
            #np.asarray((unique, counts/m)).T
            
            #v_genData[0,(unique).astype(int)] = counts/m
            #for g in range(1,ngenerations):
            for g in range(0,ngenerations):
                if verbose: print('generation {0}'.format(g))
                v_phen = offspring.phenotypes
                v_fitn = offspring.relativeFitnessValues[:,np.newaxis]

                if fixedSize: 
                    temp_size=m
                else: 
                    #temp_size=int(offspring.fitnessValues.sum())
                    #r = offspring.fitnessValues.sum()/offspring.n
                    r  = offspring.fitnessValues.sum()/offspring.m
                    temp_size = int((1-1/(offspring.m * r/K+1))*K)
                    #print("new pop size: {0}".format(temp_size))
                    m = offspring.m
                v_pmatch = (v_fitn*v_fitn.T).flatten()[:]
                v_couples = np.random.choice(m**2, temp_size, p=v_pmatch)
                v_couples = np.array(np.unravel_index(v_couples, (m,m)))
                mtx_child = np.zeros((temp_size,n))

                for i in range(temp_size):
                    recomb = np.random.choice((0,1),n)
                    pA = offspring.mtx[v_couples[0,i],:]
                    pB = offspring.mtx[v_couples[1,i],:]
                    mtx_child[i,:] = np.logical_or(np.logical_and(recomb,pA),np.logical_and(np.logical_not(recomb),pB))+0
                #
                #if nofix: nofixation(mtx_child)
                if mu != 0: mtx_child = mutate(mtx_child,mu)
                if nofix: mtx_child = flat_nDist(mtx_child)
                offspring.mtx=mtx_child
                #v_phen_child = offspring.phenotypes
                #unique, counts = np.unique(v_phen_child, return_counts=True)
                #pos = (unique - offspring.phenoSpace[0])*offspring.nloci/np.diff(offspring.phenoSpace)[0] 
                #np.asarray((unique, counts/m)).T
                #v_genData[g,(pos).astype(int)] = counts/m
                #offspring.history_append(v_genData)
            
            return(offspring)

    
    def mutate(self, rate=0.001):
        mutations = np.random.choice((0,1),(self.nindivs,self.nloci), p=(1-rate, rate))
        self.mtx = np.logical_xor(mutations, self.mtx)
    
    '''
    def sexualPreference(self,x,y,k=1):
        return x*y*0.+1. #'panmictic' by default

    def set_sexualPreference(self, func):
        if func == 'panmictic':
            def f(x,y,k):
                return 1. #panmictic population
            self.sexualPreference = f
        elif func == 'similar':
            return 1/(1+(x-y)**2/k) #preferring similar phenotypes
        else:
            self.sexualPreference = func

    def intraCompetition(self,x,y,k=1):
        return 1/(1+(x-y)**2/k) #similar phenotypes have higher competition

    def set_intraCompetition(self, func):
        if func == 'flat':
            def f(x,y,k):
                return 1.
            self.intraCompetition = f
        else:
            self.intraCompetition = func

    def mutate(self, m=0.001):
        mutations = np.random.choice((0,1),(self.nindivs,self.nloci), p=(1-m, m))
        self.mtx = np.logical_xor(mutations, self.mtx)
    
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
        #
        #P = ((pref * self.relativeFitnessValues*sat**2).T * self.relativeFitnessValues*sat**2).T
        #P = P/np.sum(P)
        #
        P = ((pref*self.relativeFitnessValues).T * self.relativeFitnessValues).T
        P = P/np.sum(P)
        #
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
        #offspring = population(self.nindivs,self.nloci, phenoSpace=self.phenoSpace, matrix=children);
        offspring = deepcopy(self)
        offspring.mtx = children
        offspring.mutate(mutRate)
        return(offspring)

    def avgPhenotype(self):
        return self.phenotypes.sum()/self.nindivs
    '''




    
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
import interactors

def predict(v0,l,ntimesteps=100,h=None, mut=0., a=0., ps=None):
    """
    Parameters
    ----------
    v0 : array-like
        initial state.
    l : array-like
        absolute fitnesses.
    h : 3rd order ndarray, optional
        inheritance tensor. If nothing is provided, it tries to generate one. The default is None.
    mut : float or 2nd order ndarray, optional
        mutation matrix or mutation rate. If a matrix is not provided, it tries to generate one with the mutation rate. 
        The default is 0..
    ntimesteps : TYPE, optional
        n of generations simulated. The default is 100.

    Returns
    -------
    Predicted evolutionary history for a single population under a fixed fitness function.

    """
    v0 = np.c_[list(v0)]
    nstates=len(v0)
    if type(mut) == float:
        mut= generate_mut_matrix(nstates,mu=mut)
    if type(h) != np.ndarray:
        h = generate_h_tensor(nstates-1)
    if ps is None:
        ps=(0,nstates-1)
    states = np.linspace(ps[0],ps[1], nstates)
    statesdiff=np.outer(np.ones(nstates),states)-np.outer(states,np.ones(nstates))
    assortMat = interactors.pM(statesdiff,alpha=abs(a))
    if a<0:
        assortMat = 1 - assortMat
        
    v = np.zeros((ntimesteps+1, nstates,1))
    v[0] = np.squeeze(v0)[:,np.newaxis]
    print('Iterating...')
    for t in range(1,ntimesteps+1):
    #for t in range(1,10):
        w = v[t-1]*l
        w = ((w.T @ assortMat).T * w)/w.sum()
        v[t] = ((w.T @ h @ w) / w.sum()**2)[:,0]
        v[t] = mut @ v[t]
        print(t)
    return(v)

def holling_II(x, a=1, h=1):
    # h = saturation limit
    # a = attack rate
    # for a=1, derivative at 0 is 1 (effect is never larger than x)
    # for a=2, f(h/2)=h/2
    return (a * x) / (1 + a / h * x)
#%%


def initialize_bin_explicit(N,nloci,dev):
    #sets all distributions being binomials, only first timestep
    nstates = nloci+1
    v0 = np.zeros((N,nstates))
    #dev is equivalent to temp_thetanorm
    for species_id in range(N):
        v0[species_id] = [bindist(nloci,i,dev[species_id]) for i in range(nstates)]
    return v0

def simulate_explicit(
        v0,
        theta=None,
        ps=None,
        h=None,
        mutual_effs=None,
        ntimesteps = 50,
        alpha=0.01,
        #xi_S=None, # level of environmental selection (from 0 to 1). Overrides parameter m,
        m=None, # vector of levels of selection imposed by other species (from 0 to 1).
        D0=50,
        a=0.,
        d=0., # frequency dependence coefficient
        K=200,
        complete_output=False,
        find_fixedpoints=False,
        tolD=1e-1,
        tolZ=1e-4,
        divprint=5,
        simID=0
        ): 
    """
    Frequency-explicit coevolution
    """

    N,nstates = v0.shape
    nloci = nstates-1
    # if theta is None:
    #     xi_S=0
    if h is None:
        h = generate_h_tensor(nstates-1)
    if ps is None:
        ps=(0,nloci)
    if mutual_effs is None:
        mutual_effs = np.ones((N,N))/N**2 # ALERT: THIS IS SUPER ARBITRARY TO BE A DEFAULT
        
    if m is None:
        xi_d=0.5
        m=np.clip(np.random.normal(xi_d,0.01,(N,1)),0,1) # vector of levels of selection imposed by other species (from 0 to 1)
        
    xi_d=np.mean(m)
    xi_S=1-xi_d # xi_S is the level of environmental selection (from 0 to 1). Overrides parameter m,
    
    
    states = np.linspace(ps[0],ps[1], nstates)
    statesdiff=np.outer(np.ones(nstates),states)-np.outer(states,np.ones(nstates))
    #---------------------------------------------
    #assortative mating coefficients: if float, repeat. if list, do a matrix for each coefficient
    if hasattr(a, "__len__"): 
        assortTen = np.zeros((N,nstates,nstates))
        
        for i in range(N):
            '''
            assortMat = interactors.pM(statesdiff,alpha=abs(a[i]))
            if a[i]<0:                                                        ## a = -1/a?;assortMat = 1 - assortMat
                assortMat = 1 - assortMat
            assortTen[i] = assortMat
            '''
            if a[i]<0:  
                assortMat = 1 - interactors.pM(statesdiff,alpha=1/a[i]**2)
            else:
                assortMat =     interactors.pM(statesdiff,alpha=a[i]**2)
            
            assortTen[i] = assortMat
    else:
        assortMat = interactors.pM(statesdiff,alpha=abs(a))
        if a<0:
            assortMat = 1 - assortMat
        assortTen = np.repeat(assortMat[I,...],N,axis=0)
    #---------------------------------------------
    v = np.zeros((ntimesteps+1, N, nstates))
    v[0] = v0

    
    # K=200 # carrying capacity  
    alpha_environ=alpha#0.00001
    turnover=1 # DEPRECATED proportion of population renewed from generation to generation 

    # you can randomize theta to test how it affects
    # theta=np.random.rand(N)*np.diff(ps)+ps[0] # values favoured by env. selection

    thetadiff=np.outer(np.ones(N),states)-np.outer(theta,np.ones(nstates))

    p = np.zeros((ntimesteps+1, N, nstates))
    l = np.zeros((ntimesteps+1, N, nstates)) # fitness landscape
    D = np.zeros((ntimesteps+1, N)) # demography
    D[0]= D0
    #demoEff = 0
    

    #tol=1e-1
    window=500
    maxgen=2000
    print('Iterating...')
    # for t in range(1,ntimesteps+1):
    t=1;
    stabilized = False
    while t < ntimesteps+1:
        if find_fixedpoints:
            stabilized = np.concatenate((np.diff(D[-window-1:-1],              axis=0)**2 < tolD,
                                         np.diff(dist_averages(v,ps)[-window:],axis=0)**2 < tolZ))
            stabpercent = round(np.sum(stabilized)/np.size(stabilized)*100,4)
            
            if t==ntimesteps:
                # stabilized = [np.abs(np.diff(D[-window-1:-1],    axis=0).mean(0)) < tolD,
                #               np.abs(np.diff(dist_averages(v,ps)[-window:],axis=0).mean(0)) < tolZ]
                
                # stabilized = [sim['D'][-window-1:-1].var(0)     < tolD,
                #               sim['dist_avgs'][-window:].var(0) < tolZ]
                # if not np.all(stabilized) and t < maxgen+1:
                    
                if stabpercent < 99 and t < maxgen+1:
                
                
                    #print('{0} fixed'.format((np.abs(np.diff(D[-2:],axis=0))<tol).sum()/N))
                    v=np.append(v,np.zeros((1,N,nstates)),axis=0)
                    D=np.append(D,np.zeros((1,N)),axis=0)
                    p=np.append(p,np.zeros((1,N,nstates)),axis=0)
                    l=np.append(l,np.zeros((1,N,nstates)),axis=0)
                    ntimesteps+=1
            
            
        for species_id in range(N):
            p[t-1,species_id]=interactors.convpM(v[t-1,species_id],nstates,alpha)
            #DE = D[t-1][:,I] - (demoEff - 1) * (1 - D[t-1][:,I])
            #DE = D[t-1][:,I]/K * 10
            #DE = D[t-1][:,I]/D[t-1].sum()*100
            #DE = D[t-1][:,I] * 0.04 # mutualistic benefit of a single interaction with a partner(fixed)
        DE = D[t-1][:,I] 
        
        
        
        #   ======================================== NEWEST AND TESTED METHOD
        A = ((mutual_effs!=0)+(mutual_effs.T!=0))+0
        inter = A @ DE # Sum of individuals in neighbor nodes to each species.
        inter_sat = holling_II(inter, a=.1, h=1) # one individual from species X cannot interact with more than 100 individuals, no matter if they belong to same or different species
        # this assumption is that the individuals that saturate the interaction environment of species X only come from nodes directly linked with X.'
        # Other species don't interfere, which can be understood that they do not share the same spaces unless they are connected in the binary network.
        dec_rates=inter_sat/inter
        l[t-1] = (np.outer(dec_rates, DE) * mutual_effs) @ p[t-1]
        #   ========================================
        
        # l[t-1] = ((mutual_effs @ p[t-1]) * /(mutual_effs!=0).sum(1)) ** m * (interactors.pM(thetadiff,alpha=alpha_environ)) ** (1-m) # Leandro's suggestion + holling type II response to saturate total species effects
        #l[t-1] = ((mutual_effs @ p[t-1]) * holling_II(DE,a=.1,h=1)) ** m * (interactors.pM(thetadiff,alpha=alpha_environ)) ** (1-m) # Leandro's suggestion + holling type II response to saturate species effects
        # l[t-1] = ((mutual_effs @ p[t-1]) * DE) * m + (interactors.pM(thetadiff,alpha=alpha_environ)) * (1-m) # old procedure
        #l[t-1] = (mutual_effs @ p[t-1]) * m + (1-m)*pM(thetadiff,alpha=alpha_environ) #without demography mass interaction
        l[t-1] = transformations.negativeSaturator(l[t-1])
        # l[t-1] = np.outer(np.ones(N),f(states))
        w = v[t-1]*l[t-1]*np.exp(np.c_[d]*v[t-1]) # fitness with density dependence
        
        
        w = ((w @ assortTen)[0] * w) / np.c_[np.where(w.sum(1) == 0, 0.0001, w.sum(1))] # assortative mating effect
        
        for species_id in range(N):
            
            if w[species_id].sum() == 0:
                newgen= w[species_id] @ h @ w[species_id] / 0.0001 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            else:
                newgen= w[species_id] @ h @ w[species_id] / w[species_id].sum()**2
            
            v[t,species_id] = v[t-1,species_id]*(1-turnover) + newgen*turnover
            #v[t] = v[t] @ mut.T
            r = w[species_id].sum()
            #D[t,species_id] = (1-1/(D[t-1,species_id] * r/K+1))*K #old method
            D[t,species_id] = D[t-1,species_id] * r # NEWEST
        
        if t%int((ntimesteps)/divprint)==0:
            if find_fixedpoints:
                # print("ID: " + str(simID) + "\tt=" + str(t) + "\t" + str(round(np.sum(stabilized)/np.size(stabilized)*100,4)) + "% stabilized.")
                
                if stabpercent > 90:
                    stab_colorkey = '\033[1;31m'
                else: 
                    stab_colorkey = '\033[0m'
                
                os.system("echo \"ID: " + str(simID) + "\tt=" + str(t) + "\t" + stab_colorkey + str(stabpercent) + "%\033[0m stabilized.)\"") #should be echo -e in plain bash
                
                # ESTA NO print(str(round(np.sum(stabilized)/np.size(stabilized),4)) + " stabilized")
            else: 
                print("ID: " + str(simID) + "\tt=" + str(t))  
            
        
        t+=1
        
    if complete_output:
        print('{0} species went extinct out of {1}.'.format(((D[-1]<2).sum()),N))
        return v, D, l
    else:
        return v


class simulator(object):
    def __init__(self, 
            v0,
            ntimesteps,
            h,
            mutual_effs,
            theta,
            ps,
            alpha,
            m,
            D0,
            a,
            d,
            K,
            find_fixedpoints=False,
            simID=0):
        self.find_fixedpoints = find_fixedpoints
        self._v0=v0
        self._ntimesteps=ntimesteps
        self._h=h
        self._mutual_effs=mutual_effs
        self._theta=theta
        self._ps=ps
        self._alpha=alpha
        self._m=m
        self._D0=D0
        self._a=a
        self._d=d
        self._K=K
        self._simID=simID
        
        A_e=self._mutual_effs
        self.n_mutualisms   = int(((A_e>0) & (A_e.T>0)).sum()/2)
        self.n_competitions = int(((A_e<0) & (A_e.T<0)).sum()/2)
        self.n_predations   = ((A_e>0) & (A_e.T<0)).sum()

    def run(self,tolD=1e-1,tolZ=1e-4):
        self.v,self.D,self.l = simulate_explicit(
            v0=self._v0,
            ntimesteps=self._ntimesteps,
            h=self._h,
            mutual_effs=self._mutual_effs,
            theta=self._theta,
            ps=self._ps,
            alpha=self._alpha,
            m=self._m,
            D0=self._D0,
            a=self._a,
            d=self._d,
            K=self._K,
            complete_output=True,
            find_fixedpoints=self.find_fixedpoints,
            tolD=tolD,
            tolZ=tolZ,
            divprint=10,
            simID=self._simID
        )
        #self.fits = (self.v*self.l).sum(2)
        self.fits = (self.v*self.l*np.exp(np.c_[self._d]*self.v)).sum(2)
        
        self.dist_avgs = dist_averages(self.v,self._ps)

#%%

def dist_averages(v,phenospace=None):
    """
    Parameters
    ----------
    v : 3rd order ndarray
        Community's genetic history.
        shape: [n time steps, n species, 1 + n loci]
    phenospace : TYPE, optional
        min and max values of the trait. The default is (0, n loci).

    Returns
    -------
    avgseries : 2nd order ndarray
        timeseries of averages of the trait values.

    """
    _,_,nstates = v.shape
    if phenospace is None:
        phenospace=(0,nstates-1)
    states = np.linspace(phenospace[0],phenospace[1], nstates)
    avgseries = (v*states*nstates).mean(2)
    # mx.showlist(avgseries[:50])
    return avgseries


    
#transformations = transformations()


def getADJs(simulations, t=None, simID=None, return_gammas=False):
    I=np.newaxis
    adjs=[]
    mutu=[]
    comp=[]
    pred=[]
    gammas=[] # effect matrix without mass action

    if t is None:
        if simID is None :
            raise Exception("t or simID must be provided") 
        else:
            sim = simulations[simID]
            ntimesteps, N, nstates = sim['v'].shape; ntimesteps-=1
            for t in range(ntimesteps):

                A_e = sim['_mutual_effs']
                A=A_e != 0
                p=np.array([interactors.convpM(sim['v'][t,species_id],nstates,sim['_alpha']) for species_id in range(N)]) # equivalent to p
                k1=(A[...,np.newaxis] @ sim['v'][t,:,np.newaxis,:])
                k=(A[...,np.newaxis] @ p[:,np.newaxis,:])
                e = k * np.swapaxes(k1,0,1)
                gamma = e.sum(2)
            
                pop_weights = sim['D'][t][:,I] # * sim['_m'] 
                intensities = (np.outer(pop_weights,pop_weights)) # np.sqrt ??
                
                adj = gamma*intensities
                
                mutu.append(gamma *  ((A_e>0) & (A_e.T>0)))
                comp.append(gamma *  ((A_e<0) & (A_e.T<0)))
                pred.append(gamma * (((A_e>0) & (A_e.T<0)) | ((A_e<0) & (A_e.T>0))))
                
                adjs.append(adj)
                gammas.append(gamma)
    else:
        if simID is not None :
            raise Exception("either t or simID should be provided (not both)") 
        else:
            for sim in simulations:
                
                ntimesteps, N, nstates = sim['v'].shape; ntimesteps-=1
                
                A_e = sim['_mutual_effs']
                A=A_e != 0
                p=np.array([interactors.convpM(sim['v'][t,species_id],nstates,sim['_alpha']) for species_id in range(N)]) # equivalent to p
                k1=(A[...,np.newaxis] @ sim['v'][t,:,np.newaxis,:])
                k=(A[...,np.newaxis] @ p[:,np.newaxis,:])
                e = k * np.swapaxes(k1,0,1)
                gamma = e.sum(2)
            
                pop_weights = sim['D'][t][:,I] # * sim['_m'] 
                intensities = (np.outer(pop_weights,pop_weights)) # np.sqrt ??
                
                adj = gamma*intensities
                
                mutu.append(gamma *  ((A_e>0) & (A_e.T>0)))
                comp.append(gamma *  ((A_e<0) & (A_e.T<0)))
                pred.append(gamma * (((A_e>0) & (A_e.T<0)) | ((A_e<0) & (A_e.T>0))))
                
                adjs.append(adj)
                gammas.append(gamma)
            
    if return_gammas:
        return adjs, mutu, comp, pred, gammas
    else:
        return adjs, mutu, comp, pred

