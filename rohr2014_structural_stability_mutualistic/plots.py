import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle5
import bz2


root = Path(".")
my_path = root / 'data'


with bz2.BZ2File(my_path / 'smRealEigvals.obj', 'rb') as f:
	eigs = pickle5.load(f)

with bz2.BZ2File(my_path / 'params.obj', 'rb') as f:
	params = pickle5.load(f)



#figsize=(8,6)

fig = plt.figure(); ax = fig.add_subplot(projection='3d')
temp = ax.scatter3D(params[:,0],params[:,1],params[:,2], c=eigs.mean(1), cmap=plt.cm.jet)

ax.xaxis.set_rotate_label(False) 
ax.yaxis.set_rotate_label(False) 
ax.zaxis.set_rotate_label(False) 
ax.set_xlabel('$\\rho$', fontsize=12)
ax.set_ylabel('$\\delta$', fontsize=12)
ax.set_zlabel('$\\gamma_0$', fontsize=20, rotation = 0)
fig.colorbar(temp,label="real part lowest eigenvalue", orientation="horizontal")
plt.show()
#sudo ddcutil setvcp d6 04