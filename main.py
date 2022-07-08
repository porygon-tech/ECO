import evo
import numpy as np
import matplotlib.pyplot as plt
import threading
import os
from pathlib import Path
import pickle5
import bz2

'''  
def task1():
    print("Task 1 assigned to thread: {}".format(threading.current_thread().name))
    print("ID of process running task 1: {}".format(os.getpid()))
  
def task2():
    print("Task 2 assigned to thread: {}".format(threading.current_thread().name))
    print("ID of process running task 2: {}".format(os.getpid()))
'''
def task_pophistory(nindivs,nloci,skew,ps, f, duration, slot_object, index):
	pop = evo.population(nindivs,nloci, skew= skew,phenoSpace=ps);
	pop.set_fitnessLandscape(f)
	c = pop.makeChildren(k=1, mutRate=0)
	#print("Task 1 assigned to thread: {}".format(threading.current_thread().name))
	print("ID of process running task {0}: {1}".format(index, os.getpid()))
	for t in range(duration):
		print('\ttime {0}'.format(t))
		c = c.makeChildren(k=1)
		slot_object[index,t] = c.avgPhenotype()


if __name__ == "__main__":

	root = Path(".")
	my_path = root / 'data/obj'

	ntrials = 50
	duration = 200
	avg=np.zeros((ntrials,duration))
	nindivs = 100
	nloci = 20
	ps = (500,500+nloci)


	def f(x):
		return (x-ps[0]) / (ps[1]-ps[0]) # linear 

	#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	skews=np.linspace(0.01,0.99,ntrials)
	avg=np.zeros((ntrials,duration))

	threads = []
	for i in range(ntrials):
		skew = skews[i]
		threads.append(threading.Thread(target=task_pophistory, name='t'+str(i), args=(nindivs,nloci,skew,ps, f, duration, avg, i,)   ))

	for i in range(ntrials):
		threads[i].start()

	for i in range(ntrials):
		threads[i].join()


	with bz2.BZ2File(my_path / 'avg.obj', 'wb') as f:
	    pickle5.dump(avg, f)






