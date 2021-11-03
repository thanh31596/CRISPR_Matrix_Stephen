#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from sklearn.preprocessing import normalize
from IPython.core.display import display, HTML
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg as LA
import networkx as nx
from utils import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
import time
from datetime import timedelta
import pandas as pd
import numpy as np
import numpy as np
import scipy.sparse
import numpy.linalg as LA
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sys import exit
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# X=V*H
# note: Hadamard product is np.multiply()


def relu(x): return 0.5 * (x + abs(x))


def reluminus(x): return 0.5 * (abs(x)-x)

scaler = MinMaxScaler()
# __all__ = ["PyMFBase"]
_EPS = np.finfo(float).eps  # In real case, this eps should be larger.


class NMFBase():
    """
    The base class NMF is aimed to initialize matrices before the pre-training process and how we can override some required
    functions in the future
    """
    _EPS = _EPS

    def __init__(self, data, label, layers=[100, 50, 2], k=5, option='b'):
        """
        parameter explaination:
        -> data: Matrix of documents Vith labels, including X1,X2
        -> percent: the percentage of split into n1, n2
        -> r is the number of components Ve factorize (should be smaller than the no of columns of data)
        -> alpha, beta are regularization parameters
        """

        self.collector = pd.DataFrame()
        length = int(len(data)*0.7)
        temp1 = data[:length]
        y3 = label[:length]
        temp2 = data[length:]
        y4 = label[length:]

        self.X, self.X2, self.Y, self.Y2 = temp1.T, temp2.T, y3.T, y4.T
        self.A, self.D = constructA(
            self.X, self.Y, option=option)  # knowledge matrice
        self.A2, self.D2 = constructA(self.X2, self.Y2, k=k, option=option)
        self.Az, self.Dz = constructA_Z(self.X, self.Y, k=k, option=option)

        # self.D = constructD(self.A)  # diag matrice
        # self.D2 = constructD(self.A2)
        # self.Dz = constructD(self.Az)
        #self.L = constructL(self.A, self.D)  # lapcialace matrice
        #self.L2 = constructL(self.A2, self.D2)
        self.L = normalize(constructL(self.A, self.D))  # lapcialace matrice
        self.L2 = normalize(constructL(self.A2, self.D2))

        self.Lz = normalize(constructL(self.Az, self.Dz))
        #self.Lz = constructL(self.Az, self.Dz)
        self._data_dimension, self._num_samples = self.X.shape
        self._data_dimension = self._data_dimension + 3  # test
        self._data_dimension2, self._num_samples2 = self.X2.shape
        self._data_dimension2 = self._data_dimension2 + 3  # test
        self.m = self.X.shape[1]
        self.layers = layers

        self.brr = []

    def check_non_negativity(self):

        if self.H.min() < 0 or self.H2.min() < 0:
            return False
        else:
            return True

    def frobenius_norm(self):
        """ Euclidean error betveen X and V*H """
        "Override in subclasses"
        pass

    def init_h(self, data, number_of_cluster):
        """
        This is the method of initializing matrix factor H using K-means clustering method.
        The method is described in the paper: "Convex and Semi-NMF" by C.Ding (2010)
        Parameters:
        + X: the input dataset in the shape of term x document
        + number_of_cluster: the number of cluster or k Ve Vant for the neV matrix
        """
        X = pd.DataFrame(data.T)
        kmeans = KMeans(n_clusters=number_of_cluster, random_state=0).fit(X)
        # The given X is in the shape of term x documents, noV Ve need to convert into doc x terms
        G = X
        G['label'] = kmeans.labels_
        G = G.reset_index()
        M = np.zeros((X.shape[0], len(np.unique(kmeans.labels_))))
        g = pd.DataFrame(M, columns=np.unique(kmeans.labels_))

        for i in G['index']:
            for j in g.columns:
                if G.iloc[i, G.shape[1]-1] == j:
                    g.iloc[i, j] = 1
                else:
                    g.iloc[i, j] = 0
        # The paper suggested to add 0.2 in all elements of the final matrice
        return g.T.values + 0.2

    """
    Ve are going to factorize the given datasets X, X2 into multiple matr
    ices.

    - Matrix H: cluster indicators of a membership of dataset X (in this case, membership means document)

    - Matrix H2:cluster indicators of a membership of dataset X2 (in this case, membership means document)

    - Matrix V: represents cluster centroid (it is matrice Z in papers)

    - Matrix P: the product of V_i matrices in layer ith

    - Matrix U: the factor of Y ( Y ~ H.U)
    """

    def _init_h(self, i):
        """ Initalize W to random values [0,1].
        """
        # add a small value, otherwise nmf and related methods get into trouble as
        # they have difficulties recovering from zero.
        np.random.seed(42)
        self.H = np.random.uniform(0, 1, (i, self._num_samples))

    def _init_h2(self, i):
        """ Initalize W to random values [0,1].
        """
        # add a small value, otherwise nmf and related methods get into trouble as
        # they have difficulties recovering from zero.
        np.random.seed(42)
        self.H2 = np.random.uniform(0, 1, (i, self._num_samples2))
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     self.H2 = np.dot(self.X2.T, np.linalg.pinv(self.Theta).T).T

    def initialize_h(self, da, i):
        """ Initalize H using K-means clustering Vith dataset X. Size n1 x k"""

        self.H = self.init_h(da, i)

    def initialize_h2(self, da, i):
        """
        Initalize H2 using K-means clustering Vith dataset X. Size n2 x k
        """

        self.H2 = self.init_h(da, i)

    def initialize_h3(self, da, i):
        """ Initalize H2 using K-means clustering Vith dataset X. Size n2 x k"""

        self.H3 = self.init_h(da, i)

    def initialize_z(self, da, i):
        """
        Initalize V - Using the formula from paper of C.Ding (2010):
        Vith H is fixed, creating V from dataset X as folloVing:
        Z = X*H*(H_T*H)^-1

        Size p x k
        """
        # if self.Theta.shape == Theta_prev.shape:

        #     temp = np.dot(self.Theta, np.linalg.pinv(self.R))
        # else:

        #     temp = np.dot(np.dot(self.Theta.T, np.linalg.pinv(
        #         Theta_prev.T)).T, np.linalg.pinv(self.R))
        # self.Z = np.nan_to_num(temp, nan=1)
        np.random.seed(42)
        self.Z = np.random.uniform(0, 1, (da.shape[0], i))

    # def initialize_z2(self, da, i):
    #     """
    #     Initalize V - Using the formula from paper of C.Ding (2010):
    #     Vith H is fixed, creating V from dataset X as folloVing:
    #     V = X*H*(H_T*H)^-1

    #     Size p x k
    #     """
    #     self.Z2 = np.random.uniform(0, 1, (da.shape[0], i))

    def initialize_r(self, i):
        """
        Initialize R as a squared matrix with random values from 0,1

        Parameter:
        + i: the size of layer
        """
        np.random.seed(42)
        #self.R = np.random.randint(2, size=(i,i))
        self.R = np.ones([i,i])
        # self.R = np.random.randint(2, size=(i.shape[1], i.shape[1]))

    def initialize_theta(self, first_layer=False):
        """
        Initialize R as a squared matrix with random values from 0,1

        Parameter:
        + i: the size of layer
        """
        if first_layer == True:
            self.Theta = self.Z
        else:
            temp = np.dot(self.Z, self.R)
            self.Theta = normalize(np.dot(self.Theta, temp),norm='l2')

        # temp = np.dot(self.X, np.linalg.pinv(self.H))
        # self.Theta = np.nan_to_num(temp)

    def initialize_u(self):
        """Initalize U using semi NMF from Y. Similar to V, U can be created Vith fixe H and dataset Y"""
        with np.errstate(divide='ignore', invalid='ignore'):
            #self.U = normalize(np.dot(self.Y, np.linalg.pinv(self.H)))
            self.U = np.dot(self.Y, np.linalg.pinv(self.H))

            #self.U = np.dot(self.Y, np.linalg.pinv(np.dot(self.R,self.H)))



    def _update_h(self):
        "Override in subclasses"
        pass

    def _update_z(self):
        "Override in subclasses"
        pass

    def _update_h2(self):
        "Override in subclasses"
        pass

    def _update_u(self):
        "Override in subclasses"
        pass

    def compute_factors(self):
        """Override beloV"""
        pass

    def _converged(self):
        """Override in subclasses"""
        pass
