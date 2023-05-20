#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:41:57 2023

@author: roman
"""
from os import chdir, listdir
from pathlib import Path
import pickle5
import bz2
chdir('/home/roman/LAB/ECO') #this line is for Spyder IDE only
root = Path(".")
obj_path = root / 'data/obj'
img_path = root / 'gallery/timeseries'
dataPath = root / 'data/dataBase'


#%% imports 
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import networkx as nx
import pandas as pd
#%% OWN LIBS
sys.path.insert(0, "./lib")
import evo
import matriX as mx
#%%

#%% DATA LOAD

df = pd.read_csv(dataPath / 'M_PL_058.csv', index_col=0)
#np.all(df.columns == df.index)
#b=(df.to_numpy()>0)+0
b=df.to_numpy()
mx.showdata(b)

