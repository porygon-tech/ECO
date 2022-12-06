#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:13:09 2022

@author: roman
"""
from os import chdir
chdir('/home/roman/LAB/ECO') #this line is for Spyder IDE only

import sys
sys.path.insert(0, "./lib")

import matriX as mX
import numpy as np

#%%n=3
n=3
lstableList=[]
for _ in range(10000):
    A = np.random.randint(-5,5,size=(n,n))
    A = np.triu(A) + np.triu(A,k=1).T
    #minV = is_Dstable(A,fullresult=True)
    #print minV['fun']
    if mX.is_Lyapunovstable(A):
        lstableList.append(A)

dstableList=[]
for i in range(len(lstableList)):
    if mX.is_Dstable(lstableList[i],maxiter=1000,ntries=5):
        dstableList.append(lstableList[i])

len(lstableList),len(dstableList)  #D-stable matrices are a subset of Lyapunov-stable matrices, so the size should never be bigger
#%%
n=3
dstableList=[]
for _ in range(200):
    A = np.random.randint(-2,2,size=(n,n))
    A = np.triu(A) + np.triu(A,k=1).T
    #minV = is_Dstable(A,fullresult=True)
    #print minV['fun']
    if mX.is_Dstable(A):
        dstableList.append(A)
        
lstableList=[]
for i in range(len(dstableList)):
    if mX.is_Lyapunovstable(dstableList[i]):
        lstableList.append(dstableList[i])

len(lstableList), len(dstableList)

#should be the same size, since D-stability guarantees lyapunov stability

#%%

i=0
L=lstableList[i]
mX.is_Lyapunovstable(L)
mX.is_Dstable(L)
np.linalg.eigvals(L)

#this gives more detailed results
minVs = mX.is_Dstable(L,fullresult=True); print(*minVs, sep='\n=============================================================\n')

#this shows the maximized supremum real part of the eigenvalues of the multiplication for all tries. If all of them are negative, that's a good sign! The matrix is most likely D-stable.
[-minV['fun'] for minV in minVs]

#same as before
[np.max(np.real(np.linalg.eigvals(np.dot(np.diag(minV['x']),dstableList[i])))) for minV in minVs]


i=9
D=dstableList[i];D
mX.is_Dstable(D)
np.real(np.linalg.eigvals(lstableList[i]))

minVs = mX.is_Dstable(D,fullresult=True); print(*minVs, sep='\n=============================================================\n')
[np.diag(minV['x']) for minV in minVs]
#%%
#This chunk tries many different positive diagonal matrices and checks if, multiplied by D, they yield a non-Lyapunov-stable matrix.
#the output is the supremum real part of the eigenvalues. If all of them are negative, the matrix is probably D-stable.
for _ in range(20):
    print(np.max(np.real(np.linalg.eigvals(np.dot(np.diag(np.random.rand(D.shape[0])*10*np.max(D)),D)))))

#%%



b=generateWithoutUnconnected(100,52,0.025)
b.sum(1)
showdata(b)
b.sum()/(b.shape[0]*b.shape[1])


