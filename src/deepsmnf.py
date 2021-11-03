#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from numpy.linalg.linalg import _norm_dispatcher
from sklearn.preprocessing import normalize
from copy import deepcopy
from deepsmnf import *
from SMNF import *

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg as LA

from utils import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
import time
from datetime import timedelta
import pandas as pd
import numpy as np


import numpy.linalg as LA
from scipy.stats import entropy


from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
import pandas as pd


# X=V*H
# note: Hadamard product is np.multiply()
scaler = MinMaxScaler()

# __all__ = ["PyMFBase"]
_EPS = np.finfo(float).eps  # In real case, this eps should be larger.


class Deep:
    def __init__(self, snmf, alpha=100, iteration=100, lam1=0.1, lam2=0.1, lamz=0.1, normH=False, normH2=True, normZ=False, othorZ=False, othorH=False):

        self.matrix = snmf
        self.alpha = alpha
        self.lam1 = lam1
        self.lam2 = lam2
        self.lamz = lamz
        self.othorZ = othorZ
        self.othorH = othorH
        self.iteration = iteration
        self.normH = normH
        self.normZ = normZ
        self.normH2 = normH2

    def relu(x):
        return 0.5 * (x + abs(x))

    def reluminus(x):

        return 0.5 * (abs(x)-x)

    def converged(ferr, i):
        """
        If the optimization of the approximation is beloV the machine precision,
        return True.
        Parameters
        ----------
            i   : index of the update step
        Returns
        -------
            converged : boolean
        """
        k = 10**-6

        if ferr[i-1]-ferr[i] < k*np.maximum(1, ferr[i-1]):
            return True
        else:
            return False

    def run_main(self):
        matrix = self.matrix
        m = matrix.X.shape[1]
        #m = matrix.X.shape[0]
        l = [0, 1, 2]
        iternum = 0
        brr = []
        arr = []
        rate = []
        alpha = self.alpha
        lam1 = self.lam1
        lam2 = self.lam2
        lamz = self.lamz
        othorZ = self.othorZ
        othorH = self.othorH
        iteration = self.iteration
        normZ = self.normZ
        normH = self.normH
        normH2 = self.normH2
        # temporary solution
        
        while iternum < iteration:

            for i in l:

                # Update I:
                # matrix.collector['I'][i] = np.dot(
                #     matrix.collector['H'][i], np.dot(matrix.D, matrix.collector['H'][i].T))
                # Update theta
                if i == 0:
                    # matrix.collector['Theta'][i] = normalize(np.dot(
                    #     matrix.collector['Z'][i], matrix.collector['R'][i]),norm='l1')
                    matrix.collector['Theta'][i] = np.dot(
                        matrix.collector['Z'][i], matrix.collector['R'][i])
                else:
                    # matrix.collector['Theta'][i] = normalize(np.dot(
                    #     matrix.collector['Theta'][i-1], np.dot(matrix.collector['Z'][i], matrix.collector['R'][i])),norm='l1')
                    matrix.collector['Theta'][i] = np.dot(
                        matrix.collector['Theta'][i-1], np.dot(matrix.collector['Z'][i], matrix.collector['R'][i]))
                matrix.collector['Q'][i] = normalize(np.dot(
                    matrix.collector['Theta'][i].T, matrix.collector['Theta'][i]),norm='l1')
                matrix.collector['P'][i] = normalize(np.dot(
                    matrix.collector['Theta'][i].T, matrix.X),norm='l1')
                matrix.collector['P2'][i] = normalize(np.dot(
                    matrix.collector['Theta'][i].T, matrix.X2),norm='l1')

                Pneg = reluminus(matrix.collector['P'][i])
                Ppos = relu(matrix.collector['P'][i])
                P2neg = reluminus(matrix.collector['P2'][i])
                P2pos = relu(matrix.collector['P2'][i])
                Qneg = reluminus(matrix.collector['Q'][i])
                Qpos = relu(matrix.collector['Q'][i])
                if othorH == True:
                    # Update H
                    An = np.dot(matrix.collector['Theta'][i].T, np.dot(matrix.X, matrix.collector['H'][i].T)) - np.dot(matrix.collector['Theta'][i].T, np.dot(matrix.collector['Theta'][i], np.dot(matrix.collector['H'][i], matrix.collector['H'][i].T))) + np.dot(matrix.collector['R'][i].T, np.dot(matrix.collector['U'][i].T, np.dot(matrix.Y, matrix.collector['H'][i].T)))*(
                        alpha/m) - np.dot(matrix.collector['R'][i].T, np.dot(matrix.collector['U'][i].T, np.dot(matrix.collector['U'][i], np.dot(matrix.collector['R'][i], np.dot(matrix.collector['H'][i], matrix.collector['H'][i].T)))))*(alpha/m) + lam1*np.dot(matrix.collector['H'][i], np.dot(matrix.A, matrix.collector['H'][i].T))

                    H1up = Ppos + np.dot(Qneg, matrix.collector['H'][i]) + (alpha/m)*np.dot(np.dot(matrix.collector['R'][i].T, matrix.collector['U'][i].T), matrix.Y) + lam1*np.dot(
                        matrix.collector['H'][i], matrix.A) + np.dot(reluminus(An), np.dot(matrix.collector['H'][i], matrix.D))

                    H1un = Pneg + np.dot(Qpos, matrix.collector['H'][i]) + (alpha/m)*np.dot(np.dot(np.dot(np.dot(matrix.collector['R'][i].T, matrix.collector['U'][i].T),
                                                                                                          matrix.collector['U'][i]), matrix.collector['R'][i]), matrix.collector['H'][i]) + np.dot(relu(An), np.dot(matrix.collector['H'][i], matrix.D))
                    if normH == False:
                        matrix.collector['H'][i] = np.multiply(
                            matrix.collector['H'][i], np.sqrt(np.abs(div0(H1up, H1un))))
                    else:
                        matrix.collector['H'][i] = normalize(np.multiply(
                            matrix.collector['H'][i], np.sqrt(np.abs(div0(H1up, H1un)))),norm='l1')

                    # Update H2
                    An2 = np.dot(matrix.collector['Theta'][i].T, np.dot(matrix.X2, matrix.collector['H2'][i].T)) - np.dot(matrix.collector['Theta'][i].T, np.dot(matrix.collector['Theta'][i], np.dot(
                        matrix.collector['H2'][i], matrix.collector['H2'][i].T))) + lam2*np.dot(matrix.collector['H2'][i], np.dot(matrix.A2, matrix.collector['H2'][i].T))

                    H2up = P2pos + np.dot(Qneg, matrix.collector['H2'][i]) + lam2*np.dot(
                        matrix.collector['H2'][i], matrix.A2) + np.dot(reluminus(An2), np.dot(matrix.collector['H2'][i], matrix.D2))
                    H2un = P2neg + np.dot(Qpos, matrix.collector['H2'][i]) + np.dot(
                        relu(An2), np.dot(matrix.collector['H2'][i], matrix.D2))
                    if normH2 == False:
                        matrix.collector['H2'][i] = np.multiply(
                            matrix.collector['H2'][i], np.sqrt(np.abs(div0(H2up, H2un))))
                    else:
                        matrix.collector['H2'][i] = normalize(np.multiply(
                            matrix.collector['H2'][i], np.sqrt(np.abs(div0(H2up, H2un)))),norm='l1')
                else:
                    # Update H1
                    H1up = Ppos + np.dot(Qneg, matrix.collector['H'][i]) + (alpha/m)*np.dot(np.dot(
                        matrix.collector['R'][i].T, matrix.collector['U'][i].T), matrix.Y) + lam1*np.dot(matrix.collector['H'][i], reluminus(matrix.L))  # Da thay doi Y.T => Y
                    H1un = Pneg + np.dot(Qpos, matrix.collector['H'][i]) + (alpha/m)*np.dot(np.dot(np.dot(np.dot(matrix.collector['R'][i].T, matrix.collector['U'][i].T),
                                                                                                          matrix.collector['U'][i]), matrix.collector['R'][i]), matrix.collector['H'][i]) + lam1*np.dot(matrix.collector['H'][i], relu(matrix.L))
                    if normH == False:
                        matrix.collector['H'][i] = np.multiply(
                            matrix.collector['H'][i], np.sqrt(np.abs(div0(H1up, H1un))))
                    else:
                        matrix.collector['H'][i] = normalize(np.multiply(
                            matrix.collector['H'][i], np.sqrt(np.abs(div0(H1up, H1un)))),norm='l1')
                        # Update H2
                    H2up = P2pos + np.dot(Qneg, matrix.collector['H2'][i]) + lam2*np.dot(
                        matrix.collector['H2'][i], reluminus(matrix.L2))
                    H2un = P2neg + np.dot(Qpos, matrix.collector['H2'][i]) + lam2*np.dot(
                        matrix.collector['H2'][i], relu(matrix.L2))
                    if normH2 == False:
                        matrix.collector['H2'][i] = np.multiply(
                            matrix.collector['H2'][i], np.sqrt(np.abs(div0(H2up, H2un))))
                    else:
                        matrix.collector['H2'][i] = normalize(np.multiply(
                            matrix.collector['H2'][i], np.sqrt(np.abs(div0(H2up, H2un)))),norm='l1')

                if othorZ == True:
                    # Update Z
                    if i > 0:

                        Pz = np.dot(matrix.collector['Theta'][i-1].T, np.dot(matrix.X, np.dot(matrix.collector['H'][i].T, matrix.collector['R'][i].T))) + np.dot(
                            matrix.collector['Theta'][i-1].T, np.dot(matrix.X2, np.dot(matrix.collector['H2'][i].T, matrix.collector['R'][i].T)))

                        Qz = np.dot(matrix.collector['Theta'][i-1].T, np.dot(matrix.collector['Theta'][i], np.dot(matrix.collector['H'][i], np.dot(matrix.collector['H'][i].T, matrix.collector['R'][i].T)))) + np.dot(
                            matrix.collector['Theta'][i-1].T, np.dot(matrix.collector['Theta'][i], np.dot(matrix.collector['H2'][i], np.dot(matrix.collector['H2'][i].T, matrix.collector['R'][i].T))))

                        M = np.dot(matrix.collector['Theta'][i-1].T, np.dot(matrix.Az, np.dot(
                            matrix.collector['Theta'][i], matrix.collector['R'][i].T)))

                        Anz = np.dot(Pz, matrix.collector['Z'][i].T) - np.dot(
                            Qz,  matrix.collector['Z'][i].T) + lamz*np.dot(M, matrix.collector['Z'][i].T)

                        N = np.dot(Anz, np.dot(matrix.collector['Theta'][i-1].T, np.dot(
                            matrix.Dz, np.dot(matrix.collector['Theta'][i], matrix.collector['R'][i].T))))

                    else:

                        Pz = np.dot(1, np.dot(matrix.X, np.dot(matrix.collector['H'][i].T, matrix.collector['R'][i].T))) + np.dot(
                            1, np.dot(matrix.X2, np.dot(matrix.collector['H2'][i].T, matrix.collector['R'][i].T)))

                        Qz = np.dot(1, np.dot(matrix.collector['Theta'][i], np.dot(matrix.collector['H'][i], np.dot(matrix.collector['H'][i].T, matrix.collector['R'][i].T)))) + np.dot(
                            1, np.dot(matrix.collector['Theta'][i], np.dot(matrix.collector['H2'][i], np.dot(matrix.collector['H2'][i].T, matrix.collector['R'][i].T))))

                        M = np.dot(1, np.dot(matrix.Az, np.dot(
                            matrix.collector['Theta'][i], matrix.collector['R'][i].T)))

                        Anz = np.dot(Pz, matrix.collector['Z'][i].T) - np.dot(
                            Qz,  matrix.collector['Z'][i].T) + lamz*np.dot(M, matrix.collector['Z'][i].T)

                        N = np.dot(1, np.dot(matrix.Dz, np.dot(Anz, np.dot(
                            matrix.collector['Theta'][i], matrix.collector['R'][i].T))))

                    Zup = relu(Pz) + reluminus(Qz) + \
                        lamz*relu(M) + reluminus(N)
                    Zun = reluminus(Pz) + relu(Qz) + lamz * \
                        reluminus(M) + relu(N)
                    if normZ == False:
                        matrix.collector['Z'][i] = np.multiply(
                            matrix.collector['Z'][i], np.sqrt(np.abs(div0(Zup, Zun))))
                    else:
                        matrix.collector['Z'][i] = normalize(np.multiply(
                            matrix.collector['Z'][i], np.sqrt(np.abs(div0(Zup, Zun)))),norm='l1')
                else:
                    # Update Z
                    if i > 0:
                        T = np.dot(matrix.collector['Theta'][i-1].T, np.dot(matrix.Lz, np.dot(
                            matrix.collector['Theta'][i], matrix.collector['R'][i].T)))

                        Pz = np.dot(matrix.collector['Theta'][i-1].T, np.dot(matrix.X, np.dot(matrix.collector['H'][i].T, matrix.collector['R'][i].T))) + np.dot(
                            matrix.collector['Theta'][i-1].T, np.dot(matrix.X2, np.dot(matrix.collector['H2'][i].T, matrix.collector['R'][i].T)))
                        Qz = np.dot(matrix.collector['Theta'][i-1].T, np.dot(matrix.collector['Theta'][i], np.dot(matrix.collector['H'][i], np.dot(matrix.collector['H'][i].T, matrix.collector['R'][i].T)))) + np.dot(
                            matrix.collector['Theta'][i-1].T, np.dot(matrix.collector['Theta'][i], np.dot(matrix.collector['H2'][i], np.dot(matrix.collector['H2'][i].T, matrix.collector['R'][i].T))))
                    else:
                        T = np.dot(matrix.Lz, np.dot(
                            matrix.collector['Theta'][i], matrix.collector['R'][i].T))
                        Pz = np.dot(1, np.dot(matrix.X, np.dot(matrix.collector['H'][i].T, matrix.collector['R'][i].T))) + np.dot(
                            1, np.dot(matrix.X2, np.dot(matrix.collector['H2'][i].T, matrix.collector['R'][i].T)))
                        Qz = np.dot(1, np.dot(matrix.collector['Theta'][i], np.dot(matrix.collector['H'][i], np.dot(matrix.collector['H'][i].T, matrix.collector['R'][i].T)))) + np.dot(
                            1, np.dot(matrix.collector['Theta'][i], np.dot(matrix.collector['H2'][i], np.dot(matrix.collector['H2'][i].T, matrix.collector['R'][i].T))))
                    Zup = relu(Pz) + reluminus(Qz) + lamz*reluminus(T)
                    Zun = reluminus(Pz) + relu(Qz) + lamz*relu(T)
                    if normZ == False:
                        matrix.collector['Z'][i] = np.multiply(
                            matrix.collector['Z'][i], np.sqrt(np.abs(div0(Zup, Zun))))
                    else:
                        matrix.collector['Z'][i] = normalize(np.multiply(
                            matrix.collector['Z'][i], np.sqrt(np.abs(div0(Zup, Zun)))),norm='l1')

                # Update R
                if i > 0:

                    K = np.dot(matrix.collector['Z'][i].T, np.dot(matrix.collector['Theta'][i-1].T, np.dot(matrix.X, matrix.collector['H'][i].T))) + np.dot(matrix.collector['Z'][i].T, np.dot(
                        matrix.collector['Theta'][i-1].T, np.dot(matrix.X2, matrix.collector['H2'][i].T))) + np.dot(matrix.collector['U'][i].T, np.dot(matrix.Y, matrix.collector['H'][i].T))*(alpha/m)

                    F = np.dot(matrix.collector['Z'][i].T, np.dot(matrix.collector['Theta'][i-1].T, np.dot(matrix.collector['Theta'][i], np.dot(matrix.collector['H'][i], matrix.collector['H'][i].T)))) + np.dot(matrix.collector['Z'][i].T, np.dot(matrix.collector['Theta'][i-1].T, np.dot(
                        matrix.collector['Theta'][i], np.dot(matrix.collector['H2'][i], matrix.collector['H2'][i].T)))) + np.dot(matrix.collector['U'][i].T, np.dot(matrix.collector['U'][i], np.dot(matrix.collector['R'][i], np.dot(matrix.collector['H'][i], matrix.collector['H'][i].T))))*(alpha/m)
                    matrix.collector['R'][i] = normalize(matrix.collector['R'][i]*np.sqrt(
                        np.abs(div0((relu(K) + reluminus(F)), (reluminus(K) + relu(F))))))
                    #print("This is R: ", matrix.collector['R'][i])
                else:
                    K = np.dot(matrix.collector['Z'][i].T, np.dot(1, np.dot(matrix.X, matrix.collector['H'][i].T))) + np.dot(matrix.collector['Z'][i].T, np.dot(
                        1, np.dot(matrix.X2, matrix.collector['H2'][i].T))) + np.dot(matrix.collector['U'][i].T, np.dot(matrix.Y, matrix.collector['H'][i].T))*(alpha/m)

                    F = np.dot(matrix.collector['Z'][i].T, np.dot(1, np.dot(matrix.collector['Theta'][i], np.dot(matrix.collector['H'][i], matrix.collector['H'][i].T)))) + np.dot(matrix.collector['Z'][i].T, np.dot(1, np.dot(matrix.collector['Theta'][i], np.dot(
                        matrix.collector['H2'][i], matrix.collector['H2'][i].T)))) + np.dot(matrix.collector['U'][i].T, np.dot(matrix.collector['U'][i], np.dot(matrix.collector['R'][i], np.dot(matrix.collector['H'][i], matrix.collector['H'][i].T))))*(alpha/m)
                    matrix.collector['R'][i] = normalize(matrix.collector['R'][i]*np.sqrt(
                        np.abs(div0((relu(K) + reluminus(F)), (reluminus(K) + relu(F))))),norm='l1')

                # Update U
                Up = np.dot(matrix.Y, np.dot(
                    matrix.collector['H'][i].T, matrix.collector['R'][i].T))

                Un = np.dot(matrix.collector['U'][i], np.dot(matrix.collector['R'][i], np.dot(
                    matrix.collector['R'][i].T, np.dot(matrix.collector['H'][i], matrix.collector['H'][i].T))))

                matrix.collector['U'][i] = normalize(np.multiply(
                    matrix.collector['U'][i], np.sqrt(np.abs(div0(Up, Un)))),norm='l1')

            ma = matrix
            error = get_error(ma, l, alpha, m, lam1, lam2, lamz, matrix.Lz)
            #ma = 0
            matrix.brr.append(error)
            iternum += 1
            # print(error)
            k = 10**-7
            # Convergence
            if iternum > 2:

                print("converging")
                print("Error: ",matrix.brr[iternum-1])
                print("Difference: {}".format(
                    matrix.brr[iternum-2]-matrix.brr[iternum-1]))
                # if matrix.brr[iternum-2]-matrix.brr[iternum-1] == 0:
                # pass
                # k*matrix.brr[iternum-2]:
                if matrix.brr[iternum-2]-matrix.brr[iternum-1] <= k*np.maximum(0, matrix.brr[i-2]):
                #if matrix.brr[iternum-1]<100:
                    # adjust the error measure
                    print("Difference: {}".format(
                        matrix.brr[iternum-2]-matrix.brr[iternum-1]))
                    print(
                        "Converged at the {0}th iteration".format(iternum-1))
                    return matrix
                else:
                    pass 
            else:
                continue

0# %%
