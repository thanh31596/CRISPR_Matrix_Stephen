#!/usr/bin/env python
# coding: utf-8

# In[1]:
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy import sparse
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

import pandas as pd
import numpy as np
import networkx as nx
from numpy import linalg as LA
#from sklearn.metrics.pairwise import cosine_similarity

# In[74]:


def create_hyperparameter(alpha=[0.0, 0.003, 600, 9000],  lam1=[1], lam2=[10,0.001,  1, 100] , lamz=[10,0.001,  1, 100] , othorZ=[False], othorH=[False], normZ=[True], normH=[True],normH2=[True]):
    # # learning_rate choices
    # alpha = [0.1]  # [ 0.0, 0.003, 600, 9000]
    # lam1 = [0.1]  # [ 0.0, 0.003, 600, 9000]
    # lam2 = [0.1]  # [ 0.0, 0.003, 600, 9000]
    # lamz = [0.1]  # [ 0.0, 0.003, 600, 9000]

    # othorH = [False, True]
    #othorZ = [False, True]
    #normZ = [False, True]
    #normH = [False, True]

    # iterations choices
    iterations = [100]

    # available combination of learning_rate and iterations

    parameters = []
    for i in alpha:
        for j in lam1:
            for k in lam2:
                for l in lamz:
                    for m in iterations:
                        for n in othorH:
                            for o in othorZ:
                                for p in normZ:
                                    for q in normH:
                                        for x in normH2:

                                            parameters.append(
                                                (i, j, k, l, m, n, o, p, q,x))
    return parameters


def c(a, b):
    return np.inner(a, b) / (LA.norm(a) * LA.norm(b))


def constructA(X, Y, k=6, option="b"):
    """
    Function nay de tao ra ma tran A theo networkx voi size la nxn. 

    Phuong phap de tao ra Laplacian graph L = D - A do la tao ra matrix A truoc theo paper 
    o 3 dang: Binary, dot-weighting and radial. Luu y rang D la 1 ma tran cheo con A se duoc tinh nhu sau: 

    + Binary:        Aij = 1 if yi = yj 
                     Aij = 0 otherwise
    + dot-weighting: Aij = x_i.T * x_j if yi = yj  
                     Aij = 0 otherwise
    + radial:        Aij = e^-(L2_norm(x_i-x_j)^2/2*sigma) if yi = yj     
    + k: so cluster      
    """

    g = np.concatenate((X.T, Y.T), axis=1)
    #W = np.zeros((g.shape[0], g.shape[0]))

    sigma = np.var(X)
    if option == 'b':
        #                     # W[i][j]=np.exp(-LA.norm(np.abs(g[i][:-1]-g[j][:-1]))**2,(2*sigma**2))
        knn_dist_graph = kneighbors_graph(X.T,
                                          n_neighbors=k,
                                          mode='distance',
                                          metric='euclidean',
                                          n_jobs=6)

        similarity_graph = sparse.csr_matrix(knn_dist_graph.shape)
        nonzeroindices = knn_dist_graph.nonzero()
        similarity_graph[nonzeroindices] = 1  # doan nay thay bang 0 la dc
        W = 0.5 * (similarity_graph + similarity_graph.T)
        W = W.todense()
        #W = similarity_graph.todense()
    if option == 'r':
        #         for i in range(X.T.shape[0]):
        #             for j in range(i):
        #                 if g[i][-1] == g[j][-1]:
        #                     #print("gi_array ",np.abs(g[i][:-1]))

        #                     W[i][j] = np.exp(-LA.norm(np.abs(g[i][:-1] -
        #                                      g[j][:-1]))**2/(2*sigma**2))
        #                     # W[i][j]=np.exp(-LA.norm(np.abs(g[i][:-1]-g[j][:-1]))**2,(2*sigma**2))
        knn_dist_graph = kneighbors_graph(X.T,
                                          n_neighbors=k,
                                          mode='distance',
                                          metric='euclidean',
                                          n_jobs=6)

        #similarity_graph = knn_dist_graph

        similarity_graph = sparse.csr_matrix(knn_dist_graph.shape)
        nonzeroindices = knn_dist_graph.nonzero()
        similarity_graph[nonzeroindices] = np.exp(-np.asarray(
            knn_dist_graph[nonzeroindices])**2 / 2.0 * sigma**2)
        similarity_graph = knn_dist_graph
        W = 0.5 * (similarity_graph + similarity_graph.T)
        W = W.todense()

    if option == 'd':
        # for i in range(X.T.shape[0]):
        #     for j in range(i):
        #         if g[i][-1] == g[j][-1]:
        #             W[i][j] = c([g[i][:-1]], [g[j][:-1]])
        knn_dist_graph = kneighbors_graph(X.T,
                                          n_neighbors=k,
                                          mode='distance',
                                          metric='euclidean',
                                          n_jobs=6)

        similarity_graph = knn_dist_graph
        W = 0.5 * (similarity_graph + similarity_graph.T)
        W = W.todense()
        print("w shape: ", W.shape)
    degree_matrix = W.sum(axis=1)
    diagonal_matrix = np.diag(np.asarray(degree_matrix).reshape(W.shape[0],))
    return W, diagonal_matrix


def constructD(A):
    """
    Function tao ra D voi size nxn
    Nen nho rang D la 1 ma tran cheo duoc cau tao boi A, vay nen: 
    Djj = Σ Wjk
    """
    degree_matrix = A.sum(axis=1)
    diagonal_matrix = np.diag(np.asarray(degree_matrix).reshape(A.shape[0],))
    return diagonal_matrix


def constructL(A, D):
    """
    Make laplacian using KNN neighbors with: 
    L = D-A
    """
    L = D-A
    return L


def get_error(matrix, layer, alpha, m, lam1, lam2, lamz, Lz):
    t2=0
    t3=0
    t5=0
    t6=0
    t1 =[]
    t4 = []
    for j in layer:

        t1.append(LA.norm(
            matrix.X - np.dot(matrix.collector['Theta'][j], matrix.collector['H'][j]))**2)
        t2 += (LA.norm(matrix.Y - np.dot(matrix.collector['U'][j], np.dot(
            matrix.collector['R'][j], matrix.collector['H'][j])))**2)/m
        t3 += np.trace(np.dot(matrix.collector['H'][j], np.dot(
            matrix.L, matrix.collector['H'][j].T)))
        t4.append(LA.norm(
            matrix.X2 - np.dot(matrix.collector['Theta'][j], matrix.collector['H2'][j]))**2)
        t5 += np.trace(np.dot(matrix.collector['H2'][j], np.dot(
            matrix.L2, matrix.collector['H2'][j].T)))
        t6 += np.trace(np.dot(matrix.collector['Theta'][j].T, np.dot(
            Lz, matrix.collector['Theta'][j])))


    error = np.min(t1) + (alpha*t2) + lam1*t3 + np.min(t4) + lam2*t5 + lamz*t6
    error = np.nan_to_num(error)
    return error


def constructA_Z(X, Y, k=6, option="b"):

    g = np.concatenate((X.T, Y.T), axis=1)
    print("g shape: {}".format(g.shape))
    #W = np.zeros((X.shape[0], X.shape[0]))

    sigma = np.var(X)
    if option == 'b':
        #                     # W[i][j]=np.exp(-LA.norm(np.abs(g[i][:-1]-g[j][:-1]))**2,(2*sigma**2))
        knn_dist_graph = kneighbors_graph(X,
                                          n_neighbors=k,
                                          mode='distance',
                                          metric='euclidean',
                                          n_jobs=6)

        similarity_graph = sparse.csr_matrix(knn_dist_graph.shape)
        nonzeroindices = knn_dist_graph.nonzero()
        similarity_graph[nonzeroindices] = 1  # doan nay thay bang 0 la dc

        W = 0.5 * (similarity_graph + similarity_graph.T)
        W = W.todense()
    elif option == 'r':
        # for i in range(X.T.shape[1]):
        #     for j in range(i):
        #         if g[i][-1] == g[j][-1]:
        #             W[i][j] = np.exp(-LA.norm(np.abs(g[i][:-1] -
        #                              g[j][:-1]))**2/(2*sigma**2))
        #             # W[i][j]=np.e**LA.norm(div0(-LA.norm(np.abs(g[i][:-1]-g[j][:-1]))**2,(2*sigma**2)))
        #             # W[i][j]=np.exp(-LA.norm(np.abs(g[i][:-1]-g[j][:-1]))**2,(2*sigma**2))
        knn_dist_graph = kneighbors_graph(X,
                                          n_neighbors=k,
                                          mode='distance',
                                          metric='euclidean',
                                          n_jobs=6)

        #similarity_graph = knn_dist_graph

        similarity_graph = sparse.csr_matrix(knn_dist_graph.shape)
        nonzeroindices = knn_dist_graph.nonzero()
        similarity_graph[nonzeroindices] = np.exp(-np.asarray(
            knn_dist_graph[nonzeroindices])**2 / 2.0 * sigma**2)  # doan nay thay bang 0 la dc
        W = 0.5 * (similarity_graph + similarity_graph.T)
        W = W.todense()

    elif option == 'd':
        knn_dist_graph = kneighbors_graph(X,
                                          n_neighbors=k,
                                          mode='distance',
                                          metric='euclidean',
                                          n_jobs=6)

        similarity_graph = sparse.csr_matrix(knn_dist_graph.shape)
        nonzeroindices = knn_dist_graph.nonzero()
        similarity_graph[nonzeroindices] = np.exp(-np.asarray(
            knn_dist_graph[nonzeroindices])**2 / 2.0 * sigma**2)

        W = 0.5 * (similarity_graph + similarity_graph.T)
        W = W.todense()
    I = np.ones((X.shape[0], X.shape[0])) - np.eye(X.shape[0], X.shape[0])
    W = W * I

    degree_matrix = W.sum(axis=1)
    diagonal_matrix = np.diag(np.asarray(degree_matrix).reshape(W.shape[0],))
    return W, diagonal_matrix


# In[ ]:
def div0(a, b, fill=0):
    """ a / b, divide by 0 -> `fill`
        div0( [-1, 0, 1], 0, fill=np.nan) -> [nan nan nan]
        div0( 1, 0, fill=np.inf ) -> inf
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        if np.isscalar(c):
            if np.isfinite(c):
                return c
            else:
                c = np.nan_to_num(c, neginf=1, posinf=1, nan=1)
                return c
                # return 0
        else:
            if np.all(np.isfinite(c)):
                return c
            else:
                c = np.nan_to_num(c, neginf=1, posinf=1, nan=1)
                return c
                # return 0

# %%
