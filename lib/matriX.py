import numpy as np
#from scipy import sparse
#from scipy.sparse import linalg
from scipy.optimize import minimize#, fsolve
#from scipy.optimize import Bounds as scipy_bounds
import matplotlib.pyplot as plt
import networkx as nx
#%%
'''
Please understand the code before using.
This module contains numerical optimization functions, whose output results should be treated with a certain level of skepticism.

'''
#%%
def to_square(m):
    rows, cols = m.shape
    uL = np.zeros((rows,rows))
    dR = np.zeros((cols,cols))
    Um = np.concatenate((uL , m ), axis=1)
    Dm = np.concatenate((m.T, dR), axis=1)
    return(np.concatenate((Um,Dm), axis=0))

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
    if type(mat3) == type(None):
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
        blended = cNorm(blended,saturation)
    else:
        cmapgrn = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "magenta"]) #seagreen also
        cmapred = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "cyan"])
        cmapblu = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "yellow"])
        
        blended = (cmapred(R_r)+cmapgrn(G_r)+cmapblu(B_r))/3
        blended = cNorm(blended,1/saturation)
    
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
    
def symmetric_connected_adjacency(N, c=0.5,ntries=100):
        
    initial_l=np.zeros((N,N))
    
    while initial_l.shape != rmUnco(initial_l).shape or not nx.is_connected(nx.from_numpy_array(initial_l)):
        for i in range(ntries):
            try :
                #tmp = mx.generateWithoutUnconnected(N,N,0.1)
                initial_l= generateWithoutUnconnected(N,N,c)
            except ValueError:

                print(str(i) + ' failed')
            else:
                break
        
        initial_l=np.tril(initial_l,0)+np.tril(initial_l,0).T
        initial_l=initial_l-np.diag(np.diag(initial_l))
        
    return initial_l


def triRectangular(M,N):
    if M>N:
        sub = np.zeros((M,N))
        slope = N/M
        for j in range(M):
            v=int(1+slope*j)
            sub[j,:v]=1
        sub=np.flip(sub,0)
        
    else:
        sub = np.zeros((M,N))
        slope = M/N
        for j in range(N):
            v=int(1+slope*j)
            sub[:v,j]=1
        sub=np.flip(sub,1)
        
    return(sub)


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




#%% NETWORK ANALYSIS FUNCTIONS
#find positive or negative feedbacks

def find_feedbacks(G,ftype='positive'):
    cycles = nx.simple_cycles(G)
    weighted_cycles = [cycle for cycle in cycles if np.prod(G[u][v]['weight'] for u, v in zip(cycle, cycle[1:] + cycle[:1])) != 0]
    r1=[]
    r2=[]
    
    if   ftype=='positive':
        for cycle in weighted_cycles:
            cycle_weights = []
            
            for u, v in zip(cycle, cycle[1:] + cycle[:1]):
                weight = G[u][v]['weight']
                cycle_weights.append(weight)
    
                if np.prod(cycle_weights)>0:
                    #print(f"Weights in cycle {cycle}: {cycle_weights}")
                    r1.append(cycle); r2.append(cycle_weights)
                    
    elif ftype=='negative':
        for cycle in weighted_cycles:
            cycle_weights = []
            
            for u, v in zip(cycle, cycle[1:] + cycle[:1]):
                weight = G[u][v]['weight']
                cycle_weights.append(weight)

                if np.prod(cycle_weights)<0:
                    #print(f"Weights in cycle {cycle}: {cycle_weights}")
                    r1.append(cycle); r2.append(cycle_weights)
    return [r1,r2]


#!pip install python-louvain #run this on iPython if you find problems
from community import community_louvain
from community.community_louvain import best_partition # pip3 install python-louvain
from collections import defaultdict
from networkx.algorithms.community.quality import modularity

def groupnodes(G):
    part = best_partition(G)
    inv = defaultdict(list)
    {inv[v].append(k) for k, v in part.items()}
    result = dict(inv)
    return(list(result.values()))

def mod(g):
    if type(g)==nx.classes.graph.Graph:
        comms = groupnodes(g)
        # comms = nx.community.louvain_communities(g, resolution = 1)
        mod = modularity(g, comms)
    elif type(g)==list and np.all(type(G)==nx.classes.graph.Graph for G in g):
        comms = list(map(groupnodes, g))
        mod = list(map(modularity, g, comms))
    return(mod)


from nestedness_calculator import NestednessCalculator #https://github.com/tsakim/nestedness
nodf = lambda x: NestednessCalculator(x).nodf(x)

def renormalize(vlist):
    x=vlist
    b=np.max(x)
    a=np.min(x)
    x=(x-a)/(b-a)
    return x

def spectralRnorm(a):
    a_norm = renormalize(a)
    #L = sparse.csr_matrix(a_norm)
    #sR = sparse.linalg.eigs(a_norm,k=1,which='LM', return_eigenvectors=False)
    sR = np.max(np.real(np.linalg.eigvals(a_norm)))/np.sqrt((a>0).sum())
    return(sR)


def index_mat(rows, cols):
    return [[(i, j) for j in range(cols)] for i in range(rows)]



def totext(A):
    print('\n'.join([' '.join(list(ai_.astype(int).astype(str))) for ai_ in list(A)]))


from scipy.ndimage import zoom
import contextlib
import matplotlib
import colorsys


class graphictools:
    def inline_backend(inline=True):
        if inline:
            gui = 'module://matplotlib_inline.backend_inline'
        else:
            gui = 'qt5agg'
        with contextlib.suppress(ValueError):
            matplotlib.use(gui, force=True)
        globals()['plt'] = matplotlib.pyplot
        
    def RGB(R,G,B,same=True,sat = 1, norm=False):
        if same:
            rgblist = (cNorm(renormalize((R,G,B)),sat)*255).astype('int').T
        else:
            rgblist = np.array([cNorm(renormalize(R),sat)*255,
                                cNorm(renormalize(G),sat)*255,
                                cNorm(renormalize(B),sat)*255]).astype('int').T
        return rgblist
    
    def rgb2hex(rgblist):
        colors = ['#%02x%02x%02x' % (r,g,b) for r,g,b in np.array(rgblist)]
        return(colors)
        
    def hex_color_invert_hue(hex_color):
        # Remove the '#' symbol from the hex color string
        hex_color = hex_color.lstrip('#')
    
        # Convert hex to RGB
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
        # Convert RGB to HSV
        r, g, b = [x / 255.0 for x in rgb]
        max_value = max(r, g, b)
        min_value = min(r, g, b)
        delta = max_value - min_value
    
        if delta == 0:
            hue = 0
        elif max_value == r:
            hue = 60 * (((g - b) / delta) % 6)
        elif max_value == g:
            hue = 60 * (((b - r) / delta) + 2)
        else:
            hue = 60 * (((r - g) / delta) + 4)
    
        # Invert hue
        hue = (hue + 180) % 360
    
        # Convert HSV back to RGB
        c = max_value - min_value
        x = c * (1 - abs((hue / 60) % 2 - 1))
        m = min_value
        if 0 <= hue < 60:
            r, g, b = c, x, 0
        elif 60 <= hue < 120:
            r, g, b = x, c, 0
        elif 120 <= hue < 180:
            r, g, b = 0, c, x
        elif 180 <= hue < 240:
            r, g, b = 0, x, c
        elif 240 <= hue < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
    
        r, g, b = (int((v + m) * 255) for v in (r, g, b))
    
        # Convert RGB to hex
        inverted_hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        return inverted_hex_color

    def resize_image(image, new_shape):
        """
        Resize an image represented as an m x n NumPy array to a new shape (O x P).
    
        Parameters:
            image (numpy.ndarray): The input image represented as a 2D NumPy array.
            new_shape (tuple): A tuple (O, P) specifying the new shape of the image.
    
        Returns:
            numpy.ndarray: The resized image as an O x P NumPy array.
        """
        m, n = image.shape
        O, P = new_shape
    
        # Calculate the scaling factors
        scale_factor_x = O / m
        scale_factor_y = P / n
    
        # Perform the resizing using scipy.ndimage.zoom
        resized_image = zoom(image, (scale_factor_x, scale_factor_y), order=3)
    
        return resized_image


    def get_colors(num_colors):
        colors=[]
        for i in np.arange(0., 360., 360. / num_colors):
            hue = i/360.
            lightness = (50 + np.random.rand() * 10)/100.
            saturation = (90 + np.random.rand() * 10)/100.
            colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
        return np.array(colors)

class pruning:
    def threshold(G_o,cut):
        G=G_o.copy()
        remove = [edge for edge,weight in nx.get_edge_attributes(G,'weight').items() if weight < cut]
        G.remove_edges_from(remove)
        #adjacency = nx.adjacency_matrix(G).todense()
        return G
        
class nullmodels:
    def nestedRand(N,nswaps):
        A = np.flip(np.triu(np.ones(N)),1)
        np.fill_diagonal(A,0)
        A_tmp = A.copy()
        i=0
        while i < nswaps:
            
            zero_indices = np.where(A == 0)
            one_indices  = np.where(A == 1)
            ri_0=(0,0)
            while (ri_0[0] == ri_0[1]):
                random_index = np.random.choice(len(zero_indices[0]))
                ri_0 = zero_indices[0][random_index], zero_indices[1][random_index]
            
            ri_1=(0,0)
            while (ri_1[0] == ri_1[1]):
                random_index = np.random.choice(len(one_indices[0]))
                ri_1 = one_indices[0][random_index], one_indices[1][random_index]


            A_tmp[ri_0, ri_0[::-1]] = 1
            A_tmp[ri_1, ri_1[::-1]] = 0
            
            if nx.is_connected(nx.from_numpy_array(A_tmp)):
                i+=1
                A = A_tmp.copy()
        return(A)
    
    def clusterchain(N,nclusters=2):
        N = 25
        nclusters = 4
        cuts = np.append(np.append(0,np.sort(np.random.choice(np.arange(2,N-1),nclusters-1,replace=False))),N-1)
        clustersizes =np.diff(cuts)
        clusters = [np.ones((cut,cut)) for cut in clustersizes]
    
        joint = clusters[0]
        for i in range(1,nclusters):
            joint = joingraphs(joint, clusters[i])
            joint[cuts[i],  cuts[i]-1] = 1
            joint[cuts[i]-1,cuts[i]  ] = 1
    
    
        np.fill_diagonal(joint,0)
        return joint


class ecomodels:
    def structured_triple(N=25,N_producers=10,N_consumers=12,g = np.array([-1,0.5]),producer_mutu=True,consumer_comp=False,consumer_nest=False):

        N_apex=N-N_producers-N_consumers
        g1,g2 = g

        A = np.zeros((N,N))
        A_e = A.copy()
        if producer_mutu:
            A_e[:N_producers,:N_producers]=g2
            
        if consumer_nest:
            sub = triRectangular(N_producers,N_consumers)
            
            A_e[N_producers:(N_producers+N_consumers),:N_producers]=sub.T*g2
            A_e[:N_producers,N_producers:(N_producers+N_consumers)]=sub  *g1
        else:
            A_e[N_producers:(N_producers+N_consumers),:N_producers]=g2
            A_e[:N_producers,N_producers:(N_producers+N_consumers)]=g1

        if consumer_comp:
            A_e[N_producers:(N_producers+N_consumers),N_producers:(N_producers+N_consumers)]=g1
            A_e[-N_apex:,-N_apex:]=g1

        A_e[-N_apex:,N_producers:-N_apex]=g2
        A_e[N_producers:-N_apex,-N_apex:]=g1

        np.fill_diagonal(A_e,0)
        return(A_e)

    

def swaplinks(A, nswaps, connected = False):
    A_tmp = A.copy()
    i=0
    if connected:
        while i < nswaps:
            A_tmp = __auxswaplinks(A)
            if nx.is_connected(nx.from_numpy_array(A_tmp)):
                i+=1
                A = A_tmp.copy()
    else:
        while i < nswaps:
            A_tmp = __auxswaplinks(A)
            i+=1
            A = A_tmp.copy()
    return(A)



def __auxswaplinks(A):
    A_tmp = A.copy()
    zero_indices = np.where(A == 0)
    one_indices  = np.where(A == 1)
    ri_0=(0,0)
    while (ri_0[0] == ri_0[1]):
        random_index = np.random.choice(len(zero_indices[0]))
        ri_0 = zero_indices[0][random_index], zero_indices[1][random_index]
    
    ri_1=(0,0)
    while (ri_1[0] == ri_1[1]):
        random_index = np.random.choice(len(one_indices[0]))
        ri_1 = one_indices[0][random_index], one_indices[1][random_index]

    A_tmp[ri_0, ri_0[::-1]] = 1
    A_tmp[ri_1, ri_1[::-1]] = 0
    return A_tmp

def double_edge_swap(M, nswaps=1, ntries=100):
    return nx.adjacency_matrix(nx.double_edge_swap(nx.from_numpy_array(M), nswaps, ntries)).todense()

def joingraphs(m1,m2):
	s1 = m1.shape[0]
	s2 = m2.shape[0]
	return np.concatenate((
	np.concatenate((m1 ,               np.zeros((s1,s2)) ), axis=1),
	np.concatenate((np.zeros((s2,s1)), m2                ), axis=1)), 
	axis=0)




def showF3D(f,type='surf', rangeX=(-1,1),rangeY=(-1,1),res=20,zlim=None,cmap='jet'):
    resX=res
    resY=res
    xr = np.linspace(rangeX[0],rangeX[1],resX)
    yr = np.linspace(rangeY[0],rangeY[1],resY)
    gx,gy = np.meshgrid(xr,yr)
    x, y = gx.flatten(), gy.flatten()
    z = list(map(f, x,y))
    if type == 'surf':
        fig = plt.figure(); ax = fig.add_subplot(projection='3d')
        if zlim:
            ax.axes.set_zlim3d(bottom=zlim[0], top=zlim[1]) 
        surf = ax.plot_trisurf(x,y,z, cmap=cmap, linewidth=0,antialiased = True)
        fig.colorbar(surf)
        plt.show()
    elif type == 'scatter':
        fig = plt.figure(); ax = fig.add_subplot(projection='3d')
        if zlim:
            ax.axes.set_zlim3d(bottom=zlim[0], top=zlim[1]) 
        scat = ax.scatter3D(x,y,z, c=z, cmap=cmap)
        fig.colorbar(scat)
        plt.show()
    elif type == 'map':
        # showdata(np.array(z).reshape((resX,resY)), color='jet')
        if zlim:
            plt.imshow(np.array(z).reshape((resX,resY)).astype('float32'), interpolation='none',origin='upper', cmap=cmap,vmin=zlim[0], vmax=zlim[1])
        else:
            plt.imshow(np.array(z).reshape((resX,resY)).astype('float32'), interpolation='none',origin='upper', cmap=cmap)
        plt.colorbar(label=r'$z$')

        tk = np.arange(0,res,int(res/6))
        plt.xticks(ticks=tk, labels=np.round(xr,3)[tk], rotation=45)
        plt.yticks(ticks=tk+1, labels=np.round(yr,1)[tk+1], rotation=45)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.show()

