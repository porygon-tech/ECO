import numpy as np
#import matplotlib.pyplot as plt
#import networkx as nx
from scipy import sparse
from scipy.sparse import linalg
#from scipy.optimize import minimize


import threading
import os
from pathlib import Path
import pickle5
import bz2
#%% ===========================================================================

def task_calcMinEig(slot_object, index, nrepeats, n_A,n_P,h,rho,delta,gamma_0):

	'''
	**params
	n_A     = params.get('n_A'    , 2  )
	n_P     = params.get('n_P'    , 2  )
	h       = params.get('h'      , 0.0)
	rho     = params.get('rho'    , 0.0)
	delta   = params.get('delta'  , 0.0)
	gamma_0 = params.get('gamma_0', 0.0)
	'''
	print("Task 1 assigned to thread: {}".format(threading.current_thread().name))
	print("ID of process running task {0}: {1}".format(index, os.getpid()))

	for t in range(nrepeats):

		#beta is the intra-guild competition matrix
		beta_A  = rho + np.zeros((n_A, n_A))
		beta_P  = rho + np.zeros((n_P, n_P))

		for i in range(n_A):
			beta_A[i,i]=1

		for i in range(n_P):
			beta_P[i,i]=1

		#gamma is the mutualistic effect matrix
		prob=0.4
		y = np.random.choice((0,1),(n_A,n_P), p=(1-prob, prob))
		while np.any(y.sum(0)==0) or np.any(y.sum(1)==0):
			y = np.random.choice((0,1),(n_A,n_P), p=(1-prob, prob));

		k_A=y.sum(1)[:,np.newaxis]
		k_P=y.sum(0)[:,np.newaxis]

		gamma_A = (gamma_0*y  )/k_A**delta
		gamma_P = (gamma_0*y.T)/k_P**delta

		#B is the interaction matrix
		B = np.append(np.append(beta_P,-gamma_P, axis=1), np.append(-gamma_A, beta_A,axis=1), axis=0)

		print('\ttime {0}'.format(t))
		
		slot_object[index,t] = np.real(sparse.linalg.eigs(B,k=1,which='SM', return_eigenvectors=False))[-1]




if __name__ == "__main__":

	root = Path(".")
	my_path = root / 'data'
	#--------------------------
	

	nthreads = 200
	nrepeats = 10

	#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	eig=np.zeros((nthreads,nrepeats))
	params = np.zeros((nthreads,3))

	threads = []
	for i in range(nthreads):
		n_A     = 2
		n_P     = 2
		h       = 0.
		rho     = np.random.rand()
		delta   = np.random.rand()
		gamma_0 = np.random.rand()
		params[i,:] = (rho,delta,gamma_0)
		threads.append(threading.Thread(target=task_calcMinEig, name='t'+str(i), args=(eig, i, nrepeats,
																						n_A,
																						n_P,
																						h,
																						rho,
																						delta,
																						gamma_0,)   ))

	for i in range(nthreads):
		threads[i].start()

	for i in range(nthreads):
		threads[i].join()


	with bz2.BZ2File(my_path / 'smRealEigvals.obj', 'wb') as f:
	    pickle5.dump(eig, f)

	with bz2.BZ2File(my_path / 'params.obj', 'wb') as f:
	    pickle5.dump(params, f)



m=np.array([[ 1,  2,  2],
            [-2,  1, -2],
            [ 0,  0,  2]])
np.real(sparse.linalg.eigs(m,k=1,which='SM', return_eigenvectors=False))[-1]
np.real(np.linalg.eigvals(m))