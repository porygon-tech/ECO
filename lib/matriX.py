import numpy as np
#from scipy import sparse
#from scipy.sparse import linalg
from scipy.optimize import minimize#, fsolve
#from scipy.optimize import Bounds as scipy_bounds
import matplotlib.pyplot as plt
#%%
'''
Please understand the code before using.
This module contains numerical optimization functions, whose output results should be treated with a certain level of skepticism.

'''
#%%
def showdata(mat, color=plt.cm.gnuplot, symmetry=False,colorbar=False):
    mat = np.copy(mat)
    if symmetry:
        top = np.max([np.abs(np.nanmax(mat)),np.abs(np.nanmin(mat))])
        plt.imshow(mat.astype('float32'), interpolation='none', cmap='seismic',vmax=top,vmin=-top)
    else:
        plt.imshow(mat.astype('float32'), interpolation='none', cmap=color)
    if colorbar: plt.colorbar()
    plt.show()

def showlist(l, distbins=False):
            fig = plt.figure(); ax = fig.add_subplot(111)
            ax.plot(np.arange(len(l)),list(l))
            plt.show()

def cNorm(x, k=1):
    return (k**2*x) / (1 + (-1 + k**2)*x)

import matplotlib.colors
def rescale(arr, vmin=0,vmax=1):
    amin = np.min(arr)
    amax = np.max(arr)
    return  (arr - amin) / (amax - amin) * (vmax - vmin) +  vmin

def blendmat(mat1,mat2,mat3=None,saturation = 1.1,additive=False):
    if not mat3:
        mat3=mat2.copy()
    temp_max=np.max((mat1,mat2,mat3))
    temp_min=np.min((mat1,mat2,mat3))

    R_r = rescale(mat1, temp_min,temp_max) #clip?
    G_r = rescale(mat2, temp_min,temp_max)
    B_r = rescale(mat3, temp_min,temp_max)
    if additive:
        cmapgrn = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "green"]) #seagreen also
        cmapred = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "red"])
        cmapblu = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "blue"])
        
        blended = 1 - (1 - cmapred(R_r)) * (1 - cmapgrn(G_r)) * (1 - cmapblu(B_r))
        blended = mx.cNorm(blended,saturation)
    else:
        cmapgrn = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "magenta"]) #seagreen also
        cmapred = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "cyan"])
        cmapblu = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "yellow"])
        
        blended = (cmapred(R_r)+cmapgrn(G_r)+cmapblu(B_r))/3
        blended = mx.cNorm(blended,1/saturation)
    
    fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
    pos = ax.imshow(blended,interpolation='None')
    #fig.suptitle(r'$\alpha=$'+str(alpha)+r'$, a_{12}=$'+str(a12)+r'$, a_{13}=$'+str(a13)+', b='+str(b),y=0.75)
    #ax.set_ylim(0,n)  # decreasing time
    ax.set_ylabel('Trait value')
    ax.set_xlabel('Time (generations)')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

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

def nullnet(B):
    '''
    CAUTION: may work faster on the transpose matrix.
    if matrix dimensions are very different, make sure that m is the largest one
    '''
    m,n = B.shape
    B=B[:,np.argsort(B.sum(axis=0))[::-1]]
    B=B[np.argsort(B.sum(axis=1))[::-1],:]
    
    r = list(map(tuple, list(map(np.random.choice, [range(n)]*m, B.sum(axis=1), [False]*m))))
    
    Br = np.zeros((m,n))

    for i in range(m):
        Br[(tuple([i]*len(r[i])),r[i])] = 1
    
    colsums = Br.sum(axis=0) - B.sum(axis=0)
    initial = (colsums > 0).sum()
    while (colsums > 0).sum() > 0:
        Br=Br[:,np.argsort(Br.sum(axis=0))[::-1]] #sort columns 
        #Br=Br[np.argsort(Br.sum(axis=1))[::-1],:] # no row sorting needed
        
        colsums = Br.sum(axis=0) - B.sum(axis=0) #;colsums
        
        donor    = np.where(colsums > 0)[0] [-1] #;colsums[donor]
        acceptor = np.where(colsums < 0)[0] [ 0] #;colsums[acceptor]
        '''
        donor    = len(colsums) - 1
        acceptor = 0
        '''
 #       transfer_pos = np.array([])
        
        transfer_pos = np.where(np.logical_and((Br[:,donor]==1), (Br[:,acceptor]==0)))[0]
        
        while transfer_pos.size == 0: #this loop avoids problems when repeated columns
            if   colsums[donor-1] > 0:
                donor -= 1
            elif colsums[acceptor+1] < 0:
                acceptor += 1
            transfer_pos = np.where(np.logical_and((Br[:,donor]==1), (Br[:,acceptor]==0)))[0]
        
            
        row = np.random.choice(transfer_pos, int(min(abs(colsums[donor]),abs(colsums[acceptor]))))
        
        Br[row, donor    ] = 0
        Br[row, acceptor ] = 1
        
        #print(str(donor) + ' -> ' + str(acceptor) + '. left: ' + str(int(colsums[colsums>0].sum())))
        colsums = Br.sum(axis=0) - B.sum(axis=0)
        #print('left: ' + str(int(np.abs(colsums).sum())))
    return(Br)



def generateWithoutUnconnected(m,n,c=0.125): 
    #c is the expected connectance
    b=np.random.choice((0,1),(m,n), p=(1-c, c))
    #showdata(b)

    zero_cols=np.where(b.sum(0)==0)[0];np.random.shuffle(zero_cols)
    zero_rows=np.where(b.sum(1)==0)[0];np.random.shuffle(zero_rows)
    
    newb = b.copy()
    
    dif = len(zero_rows)-len(zero_cols)
    nreplaces=max(len(zero_rows),len(zero_cols))
    
    if dif < 0:
        zero_rows = np.append(zero_rows,np.random.choice(zero_rows,-dif))
    else:
        zero_cols = np.append(zero_cols,np.random.choice(zero_cols, dif))
        
    newb[(zero_rows,zero_cols)] =1
    #showdata(newb)
    
    for _ in range(nreplaces):
        abundant = np.where(((newb.sum(1)>1)[:,np.newaxis]*(newb.sum(0)>1)[np.newaxis,:] == 1) * (newb==1)) #positions where rows and columns have more than a single entry and b==1
        pos=np.random.choice(np.arange(len(abundant[0])),1)
        newb[(abundant[0][pos],abundant[1][pos])]=0
    
    return newb

def rmUnco(m):
    zero_rows=np.where(m.sum(1)==0)[0]
    zero_cols=np.where(m.sum(0)==0)[0]
    m = np.delete(m, zero_rows, axis=0)
    m = np.delete(m, zero_cols, axis=1)
    return m
    





def flat_nDist(B):
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

















































