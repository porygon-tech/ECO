import evo
import numpy as np
import matplotlib.pyplot as plt



nindivs = 2000
nloci = 100
ps = (500,500+nloci)
pop = evo.population(nindivs,nloci, skew= 0.5,phenoSpace=ps);#pop.show()
#pop.hist()


pop.mtx 
pop.phenotypes

childpop = pop.makeChildren(10)
childpop = childpop.makeChildren(1)


pop.mtx = np.ones((nindivs,nloci))

pop.mtx 
pop.phenotypes





pop.showfitness()
pop.set_fitnessLandscape(np.cos)
pop.showfitness()

pop.hist()






nindivs = 2000
nloci = 100
pop = evo.population(nindivs,nloci, skew= 0.5,phenoSpace=(0,1));#pop.show()
pop.hist()
pop.hist('reduced')




def show3D(func, rangeX=[0,10],rangeY=[0,10],color=plt.cm.jet, resolution=50):
	resX=resY=resolution
	x = np.linspace(rangeX[0],rangeX[1],resX)
	y = np.linspace(rangeY[0],rangeY[1],resY)
	gx,gy = np.meshgrid(x,y)
	x, y = gx.flatten(), gy.flatten()
	z = func(x,y)
	fig = plt.figure(); ax = fig.add_subplot(projection='3d')
	surf = ax.plot_trisurf(x,y,z, cmap=color, linewidth=0)
	fig.colorbar(surf)
	plt.show()




def sexualPreference(x,y,k=1):
	return 1-1/(1+(x-y)**2/k) # the more different, the more attractive


show3D(sexualPreference)

def f(x,y):
	return sexualPreference(x,y, k=5)

show3D(f)
