#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:40:33 2024

@author: ubuntu
"""

from os import chdir, listdir, environ
from pathlib import Path
import pickle5
import bz2
chdir(environ['HOME'] + '/LAB/ECO') #this line is for Spyder IDE only
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
mx.graphictools.inline_backend(True)
sd = mx.showdata
I=np.newaxis
#%%

s = 3 # n species
alpha =0.
beta=1  
pR=10#phenotype range
z = np.random.rand(s)*pR
# A = np.random.randint(-2,2,size=(s,s)); np.fill_diagonal(A,0)
# A = np.random.randint(1,3,size=(s,s)); np.fill_diagonal(A,0)
A = np.ones((s,s)); np.fill_diagonal(A,0)
# A = np.random.rand(s,s)*2-1; np.fill_diagonal(A,0)
theta = z.copy()

statesdiff=np.outer(np.ones(s),z)-np.outer(z,np.ones(s))
gamma=evo.interactors.pM(statesdiff,beta)
Q = 1/(1+1/(gamma * A))
#%%
mx.graphictools.inline_backend(True)
sd(A,symmetry=True)
sd(statesdiff,symmetry=True)
sd(gamma)
sd(Q,symmetry=True, colorbar=True)


#%%
"""
z[2]-z[7] == statesdiff[7,2] # sanity check

i=5
sum(Q[i]*statesdiff[i])

#selection gradient
2*(alpha*(theta-z) + beta*(Q*statesdiff)@np.ones(s))
"""
#%%

environ=evo.interactors.pM(theta-z,alpha)
gamma=evo.interactors.pM(statesdiff,beta)
Wbar = environ * np.prod(1 + gamma * A, axis=0)


#%%
def grad(z,A,theta,alpha,beta):
    s=len(z)
    statesdiff=np.outer(np.ones(s),z)-np.outer(z,np.ones(s))
    gamma=evo.interactors.pM(statesdiff,beta)
    Q = 1/(1+1/(gamma * A))
    grad_res = 2*(alpha*(theta-z) + beta*(Q*statesdiff)@np.ones(s))
    return grad_res


#%%
from scipy.optimize import newton
#gradient of the function whose roots we're tryna find. Necessary for the Newton method
def ddlnW(z,A,theta,alpha,beta):
    s=len(z)
    statesdiff=np.outer(np.ones(s),z)-np.outer(z,np.ones(s))
    gamma=evo.interactors.pM(statesdiff,beta)
    ddlnW_res = -2 * alpha*z+((2 * beta * (1/gamma * (-1+2*statesdiff**2 * beta)-A)*A)/(1/gamma+A)**2)@np.ones(s)
    return ddlnW_res

#%%sanity check
z0 = theta + np.random.normal(0,1,s)
if 'z_root' in globals(): del(z_root)
# z_root = newton(grad, z0, args=(A,theta,alpha,beta), maxiter=2048)
z_root, converged, _ = newton(grad, z0, fprime=ddlnW, args=(A,theta,alpha,beta), maxiter=4096,full_output=True);print(np.all(converged))
z_root, converged, _ = newton(grad, z0,               args=(A,theta,alpha,beta), maxiter=4096,full_output=True);print(np.all(converged))

print(np.isclose(theta, z_root,rtol=1e-1))
print(np.round(theta - z_root, 2))
grad( z0,A,theta,alpha,beta)
ddlnW(z0,A,theta,alpha,beta)
ddlnW(z_root,A,theta,alpha,beta)
#%%

#%% ROOT FINDING
alpha =0.
beta=1  

var = 5
nsamples = 1000
points=[]
for i in range(nsamples):
    print(i)
    # z0 = theta + np.random.normal(0,var,s)
    z0 = np.random.uniform(10,size=s)-10
    if 'z_root' in globals(): del(z_root)
    z_root, converged, _  = newton(grad, z0, args=(A,theta,alpha,beta), maxiter=4096, 
                                   tol=1e-10,
                                   full_output=True)
    if np.all(converged):
        points.append(z_root)
    else:
        print("Attempt did not converge and was discarded.")

points = np.array(points)
clip = np.nanstd(points,axis=0)
points = points[np.all(np.abs(points)<clip,axis=1),:]

#%%
if s==3:
    mx.graphictools.inline_backend(False)
    fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
    scatter1=ax.scatter(points[:,0],
                        points[:,1],
                        points[:,2],
                        s=30)
    ax.set_xlabel(r"x", fontsize=16)
    ax.set_ylabel(r"y") 
    # plt.legend(handles=[scatter1, scatter2])
    plt.colorbar(scatter1)
    plt.show()

#%%

clip = np.std(points,axis=0)
points_clipped = points[np.all(np.abs(points)<clip,axis=1),:]
if s==3:
    mx.graphictools.inline_backend(False)
    fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
    scatter1=ax.scatter(points_clipped[:,0],
                        points_clipped[:,1],
                        points_clipped[:,2],
                        s=30)
    ax.set_xlabel(r"x", fontsize=16)
    ax.set_ylabel(r"y") 
    # plt.legend(handles=[scatter1, scatter2])
    plt.colorbar(scatter1)
    plt.show()
#%% ROOT FINDING (finer)
'''
points_clipped_converged = []
for i, pt in enumerate(points_clipped):
    print(round(i/len(points_clipped),4))
    z0 = pt + np.random.normal(0,1,s)
    if 'z_root' in globals(): del(z_root) 
    #tol=1e-10 for more precision
    z_root, converged, _  = newton(grad, z0, args=(A,theta,alpha,beta), maxiter=9999,tol=1e-10, full_output=True)
    if np.all(converged):
        points_clipped_converged.append(z_root)
    else:
        print("Attempt did not converge and was discarded.")
points_clipped_converged = np.array(points_clipped_converged)
#%%
if s==3:
    mx.graphictools.inline_backend(False)
    fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
    scatter1=ax.scatter(points_clipped_converged[:,0],
                        points_clipped_converged[:,1],
                        points_clipped_converged[:,2],
                        s=30)
    ax.set_xlabel(r"x", fontsize=16)
    ax.set_ylabel(r"y") 
    # plt.legend(handles=[scatter1, scatter2])
    plt.colorbar(scatter1)
    plt.show()
'''
#%%
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
pca = PCA(n_components=2)
X_r = pca.fit(points).transform(points)

mx.graphictools.inline_backend(True)
plt.scatter(*X_r.T,s=3.5)
plt.show()

hdb = HDBSCAN(min_cluster_size=10)
hdb.fit(X_r)

#%%
labels=hdb.labels_
ids=np.unique(labels)
colors = np.array([[.0,.0,.0]]) # R G B
if len(labels)>1:
    colors = np.append(colors, mx.graphictools.get_colors(len(ids)-1),axis=0)
colordict = dict(zip(ids,colors))

masks = {}
for _id in ids:
    masks[_id] = X_r[labels ==_id]

#%%

plt.figure()

lw = 2

for colorID, _Xgroup in masks.items():
    plt.scatter(
        _Xgroup[:,0], _Xgroup[:,1], color=colordict[colorID], alpha=0.8, lw=lw, label=colorID
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of IRIS dataset")
plt.show()

#%%
for colorID, _Xgroup in list(masks.items())[1:]:
    plt.scatter(
        _Xgroup[:,0], _Xgroup[:,1], color=colordict[colorID], 
        alpha=0.8, lw=lw, label=colorID, s=2
        
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("without outliers")
plt.show()


#%%
p=pca.fit(points)
p.explained_variance_ratio_
p.components_
sd(p.get_covariance(),symmetry=True)

#%%


if s==3:
    hdb = HDBSCAN(min_cluster_size=10)
    hdb.fit(points)
    labels=hdb.labels_
    ids=np.unique(labels)
    colors = np.array([[.0,.0,.0]]) # R G B
    if len(labels)>1:
        colors = np.append(colors, mx.graphictools.get_colors(len(ids)-1),axis=0)
    colordict = dict(zip(ids,colors))

    masks = {}
    for _id in ids:
        masks[_id] = points[labels ==_id]


    mx.graphictools.inline_backend(False)
    fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
    for colorID, _Xgroup in list(masks.items())[1:]:
        ax.scatter(
            *_Xgroup.T, color=colordict[colorID], 
            alpha=0.8, lw=lw, label=colorID, s=2
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    
    
    ax.set_xlabel(r"x", fontsize=16)
    ax.set_ylabel(r"y") 
    # plt.legend(handles=[scatter1, scatter2])
    plt.show()

    mx.graphictools.inline_backend(False)
    fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
    for colorID, _Xgroup in list(masks.items()):
        ax.scatter(
            *_Xgroup.T, color=colordict[colorID], 
            alpha=0.8, lw=lw, label=colorID, s=2
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    
    
    ax.set_xlabel(r"x", fontsize=16)
    ax.set_ylabel(r"y") 
    # plt.legend(handles=[scatter1, scatter2])
    plt.show()


fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    *points.T, 
    alpha=0.8, lw=lw, label=colorID, s=2
)

ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y") 
# plt.legend(handles=[scatter1, scatter2])
plt.show()
#%%
def filter_outliers_mcd(centered_points, contamination=0.1):
    '''
    this basically excludes outliers using the Minimum Covariance Determinant and mahalanobis distances.
    '''
    # Fit the Minimum Covariance Determinant estimator
    mcd = MinCovDet(support_fraction=1 - contamination)
    mcd.fit(centered_points)
    
    # Get the robust covariance matrix and mean
    robust_cov = mcd.covariance_
    robust_mean = mcd.location_
    
    # Handle non-invertible covariance matrix
    try:
        robust_cov_inv = np.linalg.inv(robust_cov)
        #If the covariance matrix isn't full rank, there exists a linear combination of your variables which has zero variance. 
        #That linear combination should always equal some constant.
        #This means your data is actually in a lower dimensional space than the current number of variables.
    except np.linalg.LinAlgError:
        print("Covariance matrix is singular, using pseudo-inverse.")
        robust_cov_inv = np.linalg.pinv(robust_cov)
    
    # Compute Mahalanobis distances
    diff = centered_points - robust_mean
    mahalanobis_distances = np.sum(diff @ robust_cov_inv * diff, axis=1)
    
    # Determine the threshold for outliers
    threshold = np.percentile(mahalanobis_distances, 100 * (1 - contamination))
    
    # Filter inliers based on the threshold
    inliers = mahalanobis_distances <= threshold
    filtered_points = centered_points[inliers]
    
    return filtered_points, inliers
#%%


from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.covariance import MinCovDet
from scipy.stats import median_abs_deviation
from scipy.spatial.distance import mahalanobis

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')

fittedlines = {}
for colorID, _Xgroup in list(masks.items())[1:]:
    ax.scatter(
            *_Xgroup.T, color=colordict[colorID], 
            alpha=0.5, lw=lw, label=colorID, s=5
        )
    
    # Step 1: Center the points by subtracting the mean
    mean = np.mean(_Xgroup, axis=0)
    centered_points = _Xgroup - mean
    
    # Step 2: Compute the covariance matrix
    cov_matrix = np.cov(centered_points, rowvar=False)
    # cov_matrix = MinCovDet(support_fraction=0.9).fit(centered_points).covariance_
    
    # Step 3: Perform PCA (eigenvalue decomposition)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 4: Sort the eigenvalues in descending order
    sorted_eigenvalues = np.flip(np.sort(eigenvalues))
    
    # Step 5: Calculate the ratio of the largest eigenvalue to the sum of the others
    linear_ratio = sorted_eigenvalues[0] / np.sum(sorted_eigenvalues[1:])
    
    #add label
    ax.text(*mean+0.5, colorID, color='black')
    
    print(colorID, linear_ratio)
    if linear_ratio > 0:
        filtered_points, inliers = filter_outliers_mcd(_Xgroup, contamination=0.1)
        X = filtered_points[:,:-1]
        y = filtered_points[:,-1]
        line_X = np.linspace(np.min(X, axis=0), np.max(X, axis=0), 100)
        
        # reg = RANSACRegressor(random_state=0,
        #                       loss = 'squared_error',
        #                       stop_probability=0.1,
        #                       residual_threshold = median_abs_deviation(y))
        # reg.fit(X, y)
        # line_y_pred = reg.predict(line_X)
        
        huber = HuberRegressor().fit(X, y) # Im happier with the Huber rather than ransac
        line_y_pred = huber.predict(line_X)
                
        ax.plot(line_X[:, 0], line_X[:, 1], line_y_pred, color='r')
        fittedlines[colorID] = np.append(line_X,line_y_pred[:,I],axis=1)
        # print(huber.score(X,y))
        


plt.legend(loc="best", shadow=False, scatterpoints=1)
ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y") 
ax.set_aspect('equal', adjustable='box')
# plt.legend(handles=[scatter1, scatter2])
plt.show()


#%%
from itertools import combinations

# combs = list(combinations(list(fittedlines.keys()),2))
combs = list(combinations(list(masks.keys())[1:],2)) # this can be set to take triplets etc

eigsdict = {}
eigVdict = {}

for groups in combs:
    # _gs = [filter_outliers_mcd(masks[colorID], contamination=0.1)[0] for colorID in groups]
    _gs = [masks[colorID] for colorID in groups]
    # _g1, _  = filter_outliers_mcd(masks[_group1], contamination=0.1)
    # _g2, _  = filter_outliers_mcd(masks[_group2], contamination=0.1)
    # _g12 = np.append(fittedlines[_group1], fittedlines[_group2],axis=0)
    # _g12 = np.append(_g1, _g2, axis=0)
    # _g12 = np.append(masks[_group1], masks[_group2],axis=0)
    _g = np.concatenate(_gs)
    
    
    # Step 1: Center the points by subtracting the mean
    # mean = np.mean(_g12, axis=0)
    # centered_points = _g12 - mean
    mean = np.mean(_g, axis=0)
    centered_points = _g - mean
    
    
    # Step 2: Compute the covariance matrix
    # cov_matrix = np.cov(centered_points, rowvar=False)
    cov_matrix = MinCovDet(support_fraction=0.99).fit(centered_points).covariance_
    
    # Step 3: Perform PCA (eigenvalue decomposition)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    
    eigsdict[groups] = sorted_eigenvalues
    eigVdict[groups] = eigenvectors[:,sorted_indices]
    # filtered_points, inliers = filter_outliers_mcd(_g12, contamination=0.1)
    
#%%
threshold = 0.9999
coplanars = []
for key, sorted_eigenvalues in list(eigsdict.items()):
    # Calculate the cumulative explained variance
    cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
    num_components = np.searchsorted(cumulative_variance, threshold) + 1
    reduced_eigenvectors = eigenvectors[:, :num_components]
    print(key, num_components)
    if num_components == 2:
        # print(eigVdict[key])
        coplanars.append(key)
        
import networkx as nx
G=nx.from_edgelist(coplanars)
nx.simple(G)
list(nx.connected_components(G))
#%%


def on_limits_change(event):
    # Get the current axis limits
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    # Remove the old vectors if they exist
    for vec in ax.vectors:
        vec.remove()

    # Clear the list to store new vectors
    ax.vectors.clear()
    # Draw new vectors and store them in the list
    ax.vectors.append(ax.quiver(0, 0, 0, x_limits[1], 0, 0, color='k', capstyle='round'))
    ax.vectors.append(ax.quiver(0, 0, 0, 0, y_limits[1], 0, color='k', capstyle='round'))
    ax.vectors.append(ax.quiver(0, 0, 0, 0, 0, z_limits[1], color='k', capstyle='round'))
    ax.vectors.append(ax.text(*[x_limits[1], 0, 0], 'x', color='k'))
    ax.vectors.append(ax.text(*[0, y_limits[1], 0], 'y', color='k'))
    ax.vectors.append(ax.text(*[0, 0, z_limits[1]], 'z', color='k'))
    # Redraw the figure
    plt.draw()
    
def mathmode(fig):
    ax = plt.gca()
    
    
    limits = np.stack((ax.get_xlim3d(), 
                       ax.get_ylim3d(), 
                       ax.get_zlim3d()))
    
    lms = limits.mean(axis=0)
    ax.set_xlim3d(lms)
    ax.set_ylim3d(lms)
    ax.set_zlim3d(lms)
    ax.set_aspect('equal', adjustable='box')
    ax.vectors = []
    fig.canvas.mpl_connect('motion_notify_event', on_limits_change)

def plot_plane(v1, v2, ax, origin=[0, 0, 0], scale=1.0):
    v1 = np.array(v1)
    v2 = np.array(v2)
    origin = np.array(origin)
    
    # Create a mesh grid for plotting the plane
    u = np.linspace(-scale, scale, 10)
    v = np.linspace(-scale, scale, 10)
    U, V = np.meshgrid(u, v)
    
    # Compute the points on the plane
    plane_points = origin[:, None, None] + U[None, :, :] * v1[:, None, None] + V[None, :, :] * v2[:, None, None]
    
    # Extract X, Y, Z coordinates for the plot
    X, Y, Z = plane_points[0], plane_points[1], plane_points[2]
    
    ax.plot_surface(X, Y, Z, alpha=0.2,rstride=1, cstride=1, shade=True, antialiased=False)
    
    # Plot the origin and vectors for reference
    ax.quiver(*origin, *v1, color='r')
    ax.quiver(*origin, *v2, color='g')

def plot_plane_standard(ax, slope_x, slope_y, intercept, rrange=10):
    # Generate grid over specified ranges
    x = np.linspace(-rrange,rrange, 10)
    y = np.linspace(-rrange,rrange, 10)
    x, y = np.meshgrid(x, y)
    
    # Equation of the plane
    z = intercept + slope_x * x + slope_y * y
    
    # Plot the surface
    ax.plot_surface(x, y, z, alpha=0.6)
#%%
fig = plt.figure(figsize=(8,8)); ax = fig.add_subplot(111, projection='3d')
for colorID, _Xgroup in list(masks.items())[1:]:
    ax.scatter(
            *_Xgroup.T, color=colordict[colorID], 
            alpha=0.5, lw=lw, label=colorID, s=5
        )
    mean = np.mean(_Xgroup, axis=0)
    ax.text(*mean+0.5, colorID, color='black')
    
groups = coplanars[np.random.choice(len(coplanars))]
vvs = eigVdict[groups][:,:2]
_gs = [masks[colorID] for colorID in groups]
_g = np.concatenate(_gs)
origin = np.mean(_g,axis=0)
# plot_plane(*vvs.T,ax, origin=origin, scale=200)

# ev,M = np.linalg.eigh(A)
# for ai in M:
#     ax.quiver(*[0,0,0], *(ai*100),color='r')
# for ai in M.T:
#     ax.quiver(*[0,0,0], *(ai*100),color='b')
    
# for groups in coplanars:
#     vvs = eigVdict[groups][:,:2]
#     _gs = [masks[colorID] for colorID in groups]
#     _g = np.concatenate(_gs)
#     origin = np.mean(_g,axis=0)
#     plot_plane(*vvs.T,ax, origin=origin, scale=200)

# plot_plane([0,1,-1],[1,1,-2],ax, origin=[0,0,0], scale=100)
# plot_plane([1,1,0],[1,1,1],ax, origin=[0,0,0], scale=100)
# plot_plane([1,-1,0],[1,1,-2],ax, origin=[0,0,0], scale=100)
plot_plane([1,1,1],[1,2,3/2],ax, origin=[0,0,0], scale=50)
# plot_plane([1,1,1],[1,3/2,2],ax, origin=[0,0,0], scale=50)
# plot_plane([1,1,1],[3/2,1,2],ax, origin=[0,0,0], scale=50)
ax.quiver(*[0,0,0],*[100,100,-200])
ax.quiver(*[0,0,0],*[-100,-100,-100])


dig=np.linspace(0,-100,100)
ax.plot(*(dig,dig), dig)
plt.legend(loc="best", shadow=False, scatterpoints=1)
ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y") 

ax.set_proj_type('ortho')
# Connect the event to the callback
mathmode(fig)


plt.show()

#%%

for key, sorted_eigenvalues in list(eigsdict.items()):
    # Calculate the cumulative explained variance
    eigsratio = np.diag(np.outer(sorted_eigenvalues, 1/sorted_eigenvalues),1)
    print(key, eigsratio)

    

#fittedlines[colorID] 



#%%

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    *masks[1].T, 
    alpha=0.8, lw=lw, label=colorID, s=2
)
ax.set_aspect('equal', adjustable='box')
plt.show()


np.diag(np.outer(sorted_eigenvalues, 1/sorted_eigenvalues),1)


mcd = MinCovDet().fit(X)
robust_cov = mcd.covariance_

MinCovDet(support_fraction=0.9).fit(centered_points).covariance_
np.cov(centered_points, rowvar=False)

n_samples, n_features = centered_points.shape
(n_samples + n_features + 1) / (2 * n_samples)
#%%
        # mask = np.isclose(denom, 0, atol=nslv)
        
        # patch    = np.round(nslv     / (1-np.exp(-v*nslv)), rond)
        
        # # if not hasattr(x, "__len__"):
        # #     print("EH")
        # # else:
        # #     print("ta bien................................")
        # r = np.zeros_like(x)
        # r[ mask] = patch
        # r[~mask] = np.round(x[~mask] / denom[~mask],        rond)

#%%
def rr(arr):
    #takes random row
    r = np.random.randint(arr.shape[0])
    return(arr[r])
#%%
colorID = 8
direction_vector = np.diff(fittedlines[colorID][0:2],axis=0)
slope = direction_vector / np.linalg.norm(direction_vector)
intercept = fittedlines[colorID][0]
ideal_slope = np.array([(1/s)**(1/2)]*s)
np.isclose(slope, ideal_slope,atol=1e-3)

intercept+ideal_slope*-intercept[0]/(1/s)**(1/2)
#%%
#norm vector from one line to the other, assuming their director vectors are (1,1,1)
#p1 = rr(fittedlines[3]) 
nreps = 1000
estimates= []
for colorID in [3,4,5,6,7,8]:
    estimates.append(0)
    for _ in range(nreps):
        p1 = np.array([0,0,0])
        p2 = rr(fittedlines[8])
        v = p1-p2
        vproj = v - np.mean(v)
        vnorm = np.linalg.norm(vproj)
        # vproj_n =vproj/vnorm
        # vproj_n
        #print(vnorm)
        estimates[-1]+=vnorm
    estimates[-1] /=nreps
    
np.mean(estimates) * np.round(np.sin(np.pi/6),14)

vnorm * np.round(np.sin(np.pi/3),14)
#%%
accurates = np.array([p for p in points if np.all(grad(p, A, theta, alpha, beta) == [0]*s)])

accs_clip = np.nanstd(accurates,axis=0)
accs = accurates[np.all(np.abs(accurates)<accs_clip,axis=1),:]
#%%
r = np.mean(estimates)
ri = np.mean(estimates) * np.round(np.sin(np.pi/6),14)
np.sqrt(r**2/3)

fig = plt.figure(figsize=(8,8)); ax = fig.add_subplot(111, projection='3d')

ax.scatter(
        *accs.T, color='r', 
        alpha=0.5, lw=2, s=5
    )

ax.quiver(*[0,0,0],*[100,100,-200])
ax.quiver(*[0,0,0],*[-100,-100,-100])
# drawsphere(ax, origin=(0, 0, 0), radius=ri)

# for cID, l in fittedlines.items():
#     ax.plot(*l.T)


plot_plane_standard(ax, 1, 0, np.sqrt(2*(ri**2)), rrange=100)
# plot_plane_standard(ax, 1, 0, -np.sqrt(2*(ri**2)), rrange=100)
# plot_plane_standard(ax, 0, 1, -np.sqrt(2*(ri**2)), rrange=100)
# plot_plane_standard(ax, 0, 1, np.sqrt(2*(ri**2)), rrange=100)
plot_plane_standard(ax, 1/2,1/2, 0, rrange=100)


dig=np.linspace(0,-100,100)
ax.plot(*(dig,dig), dig)
plt.legend(loc="best", shadow=False, scatterpoints=1)
ax.set_xlabel(r"x", fontsize=16)
ax.set_ylabel(r"y") 

# ax.set_proj_type('ortho')
ax.set_proj_type('persp', focal_length=0.2) # icreased FOV for harder perspective effect
# Connect the event to the callback
mathmode(fig)


plt.show()


#%%
def drawsphere(ax, origin, radius):
    # Unpack origin
    ox, oy, oz = origin
    
    # Define u and v for parametric equations of the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    # Parametric equations for the sphere
    x = ox + radius * np.outer(np.cos(u), np.sin(v))
    y = oy + radius * np.outer(np.sin(u), np.sin(v))
    z = oz + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot the surface
    ax.plot_surface(x, y, z, color='b', alpha=0.6)
    
#%%
# zplane = np.random.rand(s)*10*[1,0,np.sqrt(2*(ri**2))]
# zplane = np.random.rand(s)*10*[2,2,7]
# zplane = np.random.rand(s)*10*[4,4,np.sqrt(2*(ri**2))]
rnd = np.random.rand()
r =  np.sqrt(2*(ri**2))
xy=(x,y)=np.random.rand(s-1)*10
# z = np.append(xy,1)@[0,1, np.sqrt(2*(ri**2))]
# z = np.append(xy,1)@[0,3, -np.sqrt(2*(ri**2))]
# z = np.append(xy,1)@[1,0, np.sqrt(2*(ri**2))]
# coords=np.append(xy,z)
# coords=np.array([x,x-r,x+r])
coords=np.array([x,0,x+r])
grad( coords, A,theta,alpha,beta)
#%%
rs = np.sqrt(r)
rs=0.12
a=1
np.sqrt(rs/4)/(1 + a*np.exp(beta*rs/4))
np.sqrt(rs)/(1 + a*np.exp(beta*rs))
2*np.exp(rs/4)+np.exp(rs)+3=0
#%%
2*np.exp(-1/4*beta*(x-y)**2)/ (-2+beta*(x-y)**2)

lt=np.unique(A[np.tril_indices(s,-1)])
ut=np.unique(A[np.triu_indices(s, 1)])
ut==lt and len(ut)==1
#%% realization
A = np.random.rand(s,s)*2-1; np.fill_diagonal(A,0)
z0 = np.random.normal(0,10,1)+np.random.normal(0,2,s)
ntimesteps = 100
zt=[]
zt.append(z0)
for t in range(1,ntimesteps):
    zt.append(grad( zt[-1],A,theta,alpha,beta))
#%% 
mx.graphictools.inline_backend(True)
zta=np.array(zt)
# plt.plot(*zta)
plt.plot(np.arange(ntimesteps),zta[:,0])
plt.plot(np.arange(ntimesteps),zta[:,1])
plt.plot(np.arange(ntimesteps),zta[:,2])
plt.show()
#%%
definition_rvs = 15
definition_x = 200
rt=np.zeros((definition_rvs,definition_x,s))

rvs = np.linspace(20,30,definition_rvs)
x=np.linspace(0,-50,definition_x)
for i,r_i in enumerate(rvs):
    l_coords=np.array([x,x*0,x+r_i]).T 
    l_grads=[grad(coords, A,theta,alpha,beta) for coords in l_coords]
    rt[i] = l_grads
    
rta=np.array(rt)

#%%
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
for line in range(definition_rvs):
    plt.plot(x,rta[line,:,2])
plt.show()


#%%
w=10
res=100
A = np.random.rand(s,s)*4-1; np.fill_diagonal(A,0)
fx = lambda x,z: grad(np.array([x,x*0,z]).T, A,theta,alpha,beta)[0]
fy = lambda x,z: grad(np.array([x,x*0,z]).T, A,theta,alpha,beta)[1]
fz = lambda x,z: grad(np.array([x,x*0,z]).T, A,theta,alpha,beta)[2]
mx.showF3D(fx,type = 'map',
           rangeX=(-w,w),
           rangeY=(-w,w),
           res=res,cmap='seismic')
mx.showF3D(fy,type = 'map',
           rangeX=(-w,w),
           rangeY=(-w,w),
           res=res,cmap='seismic')
mx.showF3D(fz,type = 'map',
           rangeX=(-w,w),
           rangeY=(-w,w),
           res=res,cmap='seismic')

#%%
w=5
res=100
# f = lambda x,z: np.abs(fx(x,z))+np.abs(fy(x,z))+np.abs(fz(x,z))
f = lambda x,z: np.log(1+np.abs(fx(x,z))+np.abs(fy(x,z))+np.abs(fz(x,z)))
mx.showF3D(f,type = 'map',
           rangeX=(-w,w),
           rangeY=(-w,w),
           #res=res,cmap='gist_stern')
           res=res,cmap='jet')

#%%


#%%
i=np.random.randint(definition_rvs)

fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
plt.plot(x,rta[i,:,])
plt.scatter(-rvs[i],0)
plt.show()

#%%
fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
plt.plot(rvs,rta[:,0])
plt.plot(rvs,rta[:,1])
plt.plot(rvs,rta[:,2])
plt.show()