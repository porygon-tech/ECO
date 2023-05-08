import numpy as np 
import matplotlib.pyplot as plt
from copy import deepcopy

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

from scipy.special import comb  
def bindist(n,k,p=0.5):
    return comb(n,k)*p**k*(1-p)**(n-k)

def generate_mut_matrix(nstates,mu=0):
    n= nstates-1
    if mu==0:
        temp = np.zeros((nstates,nstates))
        np.fill_diagonal(temp,1)
        return temp
    else:
        return np.array([ list(map(lambda i: sum(list(map(lambda k: bindist(i,i-k,mu) * bindist(n-i,b-k,mu), list(range(i+1))))), list(range(nstates)))) for b in list(range(nstates))])
    

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



