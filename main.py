import evo
import numpy as np


nindivs = 2000
nloci = 100
pop = evo.population(nindivs,nloci, skew= 0.5,phenoSpace=(500,500+nloci));#pop.show()
#pop.hist()
pop.showfitness()
pop.setFitnessLandscape(np.cos)
pop.showfitness()



nindivs = 2000
nloci = 100
pop = evo.population(nindivs,nloci, skew= 0.5,phenoSpace=(0,1));#pop.show()
pop.hist()
pop.hist('reduced')




