import numpy as np
#from scipy import sparse
#from scipy.sparse import linalg
from scipy.optimize import minimize#, fsolve
#from scipy.optimize import Bounds as scipy_bounds
#%%
'''
Please understand the code before using.
This module contains numerical optimization functions, whose output results should be treated with a certain level of skepticism.

'''
#%%

def is_symmetric(m):
    return (m==m.T).all()

def is_pos_def(m, maxiter=1000, z0='rand', fullresult=False):
    #tells if matrix is positive definite
    if m.shape[0] != m.shape[1]:
        raise Exception("Matrix is not square") 
    elif is_symmetric(m): #symmetry testing
        return np.all(np.linalg.eigvals(m) > 0)
    else:
        def f(z):
            z=np.array(list(z))[:,np.newaxis]
            return np.dot(np.dot(z.T, m),z)
        if z0=='rand':
            z0 = list(np.random.rand(m.shape[0]))
        #bounds = np.repeat([[0,None]],repeats=m.shape[0],axis=0)
        #constraints for a non-zero vector solution
        cons = ({'type': 'ineq', 'fun': lambda z:  np.sum(np.abs(z))})
        minV = minimize(f, z0, method='COBYLA', options={'maxiter' : maxiter},constraints=cons);
        
        if fullresult:
            return minV
        elif minV['success'] or minV['status'] == 2:
            return minV['fun']+0 > 0 
            #return minV
        else:        
            #return minV
            raise Exception(minV['message']) 

def is_Lyapunovstable(A):
    #tells if it is a Hurwitz matrix
    return np.all(np.real(np.linalg.eigvals(A)) < 0)
    #return np.real(sparse.linalg.eigs(A.astype('float'),which='LR',k=1,return_eigenvectors=False))[0] < 0


#%%
def is_Dstable(A, maxiter=1000, df0='rand', tol=10e-3, ntries=5, fullresult=False):
    #tells if matrix is D-stable
    lowestbound=10e-10
    if A.shape[0] != A.shape[1]:
        raise Exception("Matrix is not square") 

    else:   
        def f(df):
            D = np.diag(df)
            #return -np.real(sparse.linalg.eigs(np.dot(D,A),which='LR',k=1,return_eigenvectors=False))[0]
            return -np.max(np.real(np.linalg.eigvals(np.dot(D,A))))
        minVs=[]
        for _ in range(ntries):
            start=True
            minV={'status': None}
            while minV['status'] == 4 or start:
                start=False
                df0 = list(np.random.rand(A.shape[0])*2*np.max(A)+lowestbound)
                
                #cons = ({'type': 'ineq', 'fun': lambda df0: np.min(np.array(df0))*10e10})
                #minV = minimize(f, df0, method='COBYLA', options={'maxiter' : maxiter},constraints=cons);
                
                #L-BFGS-B  Nelder-Mead
                bounds = np.repeat([(lowestbound,None)],A.shape[0],axis=0)
                bounds = [tuple(r) for r in bounds]
                minV = minimize(f, df0, method='Nelder-Mead', options={'maxiter' : maxiter}, bounds=bounds);
                minVs.append(minV)

        if fullresult:
            return minVs
        elif minV['success'] or minV['status'] == 2:
            #minV['x'][np.where(minV['x']<0)]=0
            #return minV['fun']+0. > 0. and np.all(minV['x'] > tol) and np.all(minV['x']>0)
            #return np.all([np.all(minV['x'] > tol) for minV in minVs]) and np.all([minV['fun']+0. > 0. for minV in minVs])
            return np.all([minV['fun']+0. > 0. for minV in minVs])
        else:        
            raise Exception(minV['message']) 

#%%
'''
n=3
A = np.random.rand(n,n)*5-2
minVs = is_Dstable(A,fullresult=True, ntries=6); print(*minVs, sep='\n=============================================================\n')
[-minV['fun'] for minV in minVs]







df = list(np.random.rand(A.shape[0]))
D = np.diag(df)


np.all(np.linalg.eigvals(A) < 0)

#%%
n=3
lstableList=[]
for _ in range(10000):
    A = np.random.randint(-5,5,size=(n,n))
    A = np.triu(A) + np.triu(A,k=1).T
    #minV = is_Dstable(A,fullresult=True)
    #print minV['fun']
    if is_Lyapunovstable(A):
        lstableList.append(A)

dstableList=[]
for i in range(len(lstableList)):
    if is_Dstable(lstableList[i],maxiter=1000,ntries=5):
        dstableList.append(lstableList[i])

len(lstableList),len(dstableList)


#%%
n=3
dstableList=[]
for _ in range(200):
    A = np.random.randint(-2,2,size=(n,n))
    A = np.triu(A) + np.triu(A,k=1).T
    #minV = is_Dstable(A,fullresult=True)
    #print minV['fun']
    if is_Dstable(A):
        dstableList.append(A)
        
lstableList=[]
for i in range(len(dstableList)):
    if is_Lyapunovstable(dstableList[i]):
        lstableList.append(dstableList[i])

len(lstableList), len(dstableList)

#should be zero difference, since D-stability guarantees lyapunov stability  
#%%
i=0

is_Lyapunovstable(lstableList[i])
np.linalg.eigvals(lstableList[i])
    
minVs = is_Dstable(lstableList[i],fullresult=True); print(*minVs, sep='\n=============================================================\n')

[np.max(np.real(np.linalg.eigvals(np.dot(np.diag(minV['x']),dstableList[i])))) for minV in minVs]



is_Dstable(dstableList[i])
lstableList[i]
np.real(np.linalg.eigvals(lstableList[i]))


C=dstableList[0]
minVs = is_Dstable(C,fullresult=True); print(*minVs, sep='\n=============================================================\n')
[np.diag(minV['x']) for minV in minVs]


np.max(np.real(np.linalg.eigvals(np.dot(np.diag(np.random.rand(C.shape[0])*10*np.max(C)),C))))

'''
















































