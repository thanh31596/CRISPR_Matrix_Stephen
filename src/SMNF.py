#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from sklearn.preprocessing import normalize
from NMFbase import *
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
from copy import deepcopy
import scipy.sparse
import numpy.linalg as LA
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sys import exit
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
import pandas as pd

# X=V*H
# note: Hadamard product is np.multiply()


def relu(x): return 0.5 * (x + abs(x))


def reluminus(x): return 0.5 * (abs(x)-x)


#__all__ = ["PyMFBase"]
_EPS = np.finfo(float).eps  # In real case, this eps should be larger.
scaler = MinMaxScaler()

class SNMF(NMFBase):
    """
    Semi-NMF: 
    Vith H can be mixed signs, V must be nonnegative an
    """

    def _update_z(self, H_prev):

        Z1 = np.dot(H_prev,self.H.T)  # 100x946 x 946x3800 = 100x3800
        # print("H1:{}".format(H1.shape))
        Z2 = np.dot(self.H, self.H.T)  # 100x946 x 946x100 = 100x100
        if np.linalg.det(Z2) == 0:  # check if the product is a singular matrix
            self.Z = np.dot(Z1,np.linalg.pinv(Z2)) #I CHANGE THE ORDER OF Z1 and Z2
            # print("H2nghichdao:{}".format(np.linalg.pinv(H2).shape))
        else:
            self.Z = np.dot(Z1,np.linalg.inv(Z2) )
            # print("H2nghichdao:{}".format(np.linalg.inv(H2).shape))
        self.Z = np.nan_to_num(self.Z,neginf=0, posinf=0, nan=0)
  
    def _update_h(self, H_prev):  # layer 1 = 946x100, layer 2 = 100x 50, layer 3 = 50x2
        def separate_positive(m):
            return (np.abs(m) + m) / 2.0

        def separate_negative(m):
            return (np.abs(m) - m) / 2.0

        XV = np.dot(H_prev.T, self.Z)  # 946x3800 x 100x3800 = 946x100

        VV = np.dot(self.Z.T, self.Z)  # 100x3800 x 3800x100 =100x100

        VV_pos = separate_positive(VV)
        VV_neg = separate_negative(VV)

        XV_pos = separate_positive(XV)  # 3800x100

        V1 = (XV_pos + np.dot(self.H.T, VV_neg))

        XV_neg = separate_negative(XV)
        V2 = (XV_neg + np.dot(self.H.T, VV_pos))

        self.H *= np.sqrt(V1 / V2).T
        self.H = np.nan_to_num(self.H,neginf=0, posinf=0, nan=0)
    def _update_h2(self, H2_prev):  # layer 1 = 946x100, layer 2 = 100x 50, layer 3 = 50x2
        def separate_positive(m):
            return (np.abs(m) + m) / 2.0

        def separate_negative(m):
            return (np.abs(m) - m) / 2.0

        XV = np.dot(H2_prev.T, self.Z)  # 946x3800 x 100x3800 = 946x100

        VV = np.dot(self.Z.T, self.Z)  # 100x3800 x 3800x100 =100x100

        VV_pos = separate_positive(VV)
        VV_neg = separate_negative(VV)

        XV_pos = separate_positive(XV)  # 3800x100

        V1 = (XV_pos + np.dot(self.H2.T, VV_neg))

        XV_neg = separate_negative(XV)
        V2 = (XV_neg + np.dot(self.H2.T, VV_pos))

        self.H2 *= np.sqrt(V1 / V2).T
        self.H2 = np.nan_to_num(self.H2,neginf=0, posinf=0, nan=0)

    def compute_factors(self, random_init=False):

        if random_init == False:
            if not hasattr(self, 'Z'):
                NMFBase.initialize_z(self, self.X, self.layers[0])

            if not hasattr(self, 'H'):
                NMFBase.initialize_h(self, self.X, self.layers[0])

            if not hasattr(self, 'H2'):
                NMFBase.initialize_h2(self, self.X2, self.layers[0])

            if not hasattr(self, 'R'):
                NMFBase.initialize_r(self,self.Z.shape[1])
            if not hasattr(self, 'U'):
                NMFBase.initialize_u(self)
            if not hasattr(self, 'Theta'):
                self.Theta = 1

        else:

            if not hasattr(self, 'H'):
                NMFBase._init_h(self, self.layers[0])
            if not hasattr(self, 'Z'):
                NMFBase.initialize_z(self, self.X, self.layers[0])
            if not hasattr(self, 'Theta'):
                self.Theta = 1
            if not hasattr(self, 'H2'):
                NMFBase._init_h2(self, self.layers[0])
            if not hasattr(self, 'R'):
                NMFBase.initialize_r(self,self.Z.shape[1])
            if not hasattr(self, 'U'):
                NMFBase.initialize_u(self)


        if self.check_non_negativity()==True:
            pass
        else:
            self.H = relu(self.H)
            self.H2 = relu(self.H2)
        # self.R = self.initialize_r(self.Z.shape[1])
        # self.U = self.initialize_u()
        H_prev = self.X
        H2_prev = self.X2
        # Theta_prev = deepcopy(self.Theta)
        # self.initialize_z(Theta_prev)
        #print("H shape: {}".format(self.H.shape))
        #print("X shape: {}".format(self.X.shape))
        #print("X2 shape: {}".format(self.X2.shape))
        #print("Z shape: {}".format(self.Z.shape))
        dic = {'Z': {}, 'H': {},  'H2': {}, 'U': {},
               'R': {}, 'I': {}, 'Theta': {}, 'P': {}, 'Q': {}, 'P2': {}}
        self.layers.append(1)  # sua sau
        with np.errstate(divide='ignore', invalid='ignore'):

            for i in self.layers[1:]:
                self._update_h(H_prev)
                #self.H = normalize(self.H, axis=1, norm='l1')
                dic['H'].update({i: self.H})

                self._update_z(H_prev)
                dic['Z'].update({i: self.Z})
 # to convert from m x l => l x m
                self._update_h2(H2_prev)
                #self.H2 = normalize(self.H2, axis=1, norm='l1')
                dic['H2'].update({i: self.H2})

                
                dic['R'].update({i: self.R})
                # self._update_u(H3_prev)
                dic['U'].update({i: self.U})  # to convert from k x l => l x k

                # self._update_z2(H2_prev)
                #self.Z2 = normalize(self.Z2, axis=1, norm='l1')
                #dic['Z2'].update({i: self.Z2})

                #self.I = np.dot(np.dot(self.H, self.D), self.H.T)
                #dic['I'].update({i: self.I})
                self.initialize_theta()
                dic['Theta'].update({i: self.Theta})
                self.P = np.dot(self.Theta.T, self.X)
                # dic['P'].update({i:self.scaler.fit_transform(self.P)})
                dic['P'].update({i: self.P})
                self.P2 = np.dot(self.Theta.T, self.X2)
                dic['P2'].update({i: self.P2})
                # dic['P2'].update({i:self.scaler.fit_transform(self.P2)})
                self.Q = np.dot(self.Theta.T, self.Theta)
                dic['Q'].update({i: self.Q})
                # dic['Q'].update({i:self.scaler.fit_transform(self.Q)})
                H_prev = self.H
                H2_prev = self.H2

                # collect data
                self.collector = pd.DataFrame(dic).reset_index()
                if random_init == False:

                    self.initialize_h(H_prev, i)
                    self.initialize_z(H_prev, i)

                    self.initialize_h2(H2_prev, i)
                    #self.initialize_z2(H2_prev, i)

                    # self.initialize_h3(H3_prev.T,i)

                else:

                    self._init_h(i)
                    self.initialize_z(H_prev, i)

                    # self.initialize_theta()

                    self._init_h2(i)
                #self.initialize_z2(H2_prev, i)

                # self.initialize_h3(H3_prev.T,i)
                # self.initialize_u()
                self.initialize_r(self.Z.shape[1])
                self.initialize_u()

# %%
