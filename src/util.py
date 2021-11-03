from __future__ import division
import numpy as np
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
#import xgboost as xgb
from Bio.SeqUtils import MeltingTemp as mt
from Bio.Seq import Seq
import math
import pickle
from sklearn.metrics import roc_auc_score
import copy
from mlxtend.plotting import plot_decision_regions
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
  
#from svmutil import *
import csv
import sys
def countGC(s, start=0):
    """compute the GC counts and GC contents of a sequences from (start,end)

    Args:
        s ([type]): single guide RNA sequence
        start ([type]): the position of the examined nucleaotise, default is 0


    Returns:
        [type]: [description]
    """
    end = len(s)
    GCcounts = len(s[start:end].replace('A', '').replace('T', ''))
    GCcontent = GCcounts / len(s)
    return GCcounts, GCcontent

def ncf(s, order):
    """nucleotide composition features and position nucletide binary features

    Args:
        s ([str]): sequence
        order ([int]): 1,2,3: A,AA,AAA

    Returns:
        int: binary (vectorized): nucletide composition features
        int: position nucletide binary features
    """

    t_list = ["A", "G", "C", "T"]
    L = len(s)
    if order == 1:
        nc = t_list
    elif order == 2:
        nc = [m + n for m in ["A", "G", "C", "T"] for n in ["A", "G", "C", "T"]]
    elif order == 3:
        nc = [m + n + k for m in ["A", "G", "C", "T"] for n in ["A", "G", "C", "T"] for k in ["A", "G", "C", "T"]]
    nc_f = np.zeros((1, 4 ** order))

    pos_fea = np.zeros((1, 1))
    for i in range(0, L - order + 1):
        pos = np.zeros((1, 4 ** order))
        for j in range(0, len(nc)):
            if s[i:i + order] == nc[j]:
                nc_f[0][j] = nc_f[0][j] + 1
                pos[0][j] = 1
                pos_fea = np.hstack((pos_fea, pos))

    n = len(pos_fea[0])
    pos_fea = pos_fea[0][1:n]
    return nc_f, pos_fea

def TM_cal(s):
    """computing tm features, s should be 30-mer, but because of our data limitation, 
    s is 20-mer

    Args:
        s ([str]): sequence

    Returns:
        [array]: Thermo region
    """
    # 
    end = len(s)
    TM_region = np.zeros((1, 4))
    s_20 = s[4:24]
    s_30 = s
    tm_sub1 = mt.Tm_NN(Seq(s_20[2:7]))
    tm_sub2 = mt.Tm_NN(Seq(s_20[7:15]))
    tm_sub3 = mt.Tm_NN(Seq(s_20[15:20]))
    tm_sub4 = mt.Tm_NN(Seq(s_30[0:30]))

    TM_region[0][0] = tm_sub1
    TM_region[0][1] = tm_sub2
    TM_region[0][2] = tm_sub3
    # TM_region[0][3] = tm_sub4

    return TM_region
def binary_seqs(obs_seqs, t=1):
    """Vectorizing sequence

    Args:
        obs_seqs ([str]): observed sequences
        t ([int]): order

    Returns:
        [array]: alphaStat (vector of sequence: A = 0, C =1, G=2,T=3)
    
    Note: 
        REMEMBER TO TRANSPOSE IT AS INPUT IN THE DATAFRAME
    """
    nts = ['A', 'C', 'G', 'T']
    N = len(obs_seqs)
    n = len(obs_seqs[0])
    nc = []
    if t == 1:
        nc = nts
    elif t == 2:
        nc = [m + l for m in nts for l in nts]

    alpStat = np.zeros((N, 1 + n - t))
    for i in range(0, N):
        seq = obs_seqs[i]
        for j in range(0, 1 + n - t):
            mer = seq[j:j + t]
            ind = nc.index(mer)
            alpStat[i, j] = ind

    return alpStat

def TranEmis(alpStat, t, au=5, ad=5):
    N = len(alpStat[:, 0])
    n = len(alpStat[0, :])
    b = np.zeros((1, n + 2))
    c = np.zeros((n + 1, 1))
    d = np.eye(n + 1)
    Tr = np.hstack((c, d))
    Tr = np.vstack((Tr, b))
    tr = Tr.copy()
    tr[tr == 0] = -0.1
    tr[tr == 1] = -1

    Et = np.zeros((n + 2, 4 ** t))
    et = Et.copy()
    et[et == 0] = -0.01
    for i in range(1, n + 1):
        for j in range(0, 4 ** t):
            count = np.where(alpStat[:, i - 1] == j)
            p = (len(count[0]) + au) / (N + ad)
            if p == 0:
                et[i, j] = -0.01
            else:
                et[i, j] = math.log(p)

    return tr, et

def visualize_decision(matrix,matrix_):
    X2 = matrix.X2.T
    X1 = matrix.X.T
    Y2 = np.array(matrix.Y2[0].T).astype('int')
    Y1 = np.array(matrix.Y[0].T).astype('int')
    test_class_labels = np.unique(np.array(Y2))
    c1 = KMeans(n_clusters=2)
    c2 = KMeans(2)
    c3 = KMeans(n_clusters=2)
    c4 = KMeans(n_clusters=2)
    # SVD 
    svd = TruncatedSVD(n_components=2)
    svd_train = svd.fit_transform(X1)
    svd_embedding = svd.fit_transform(X2)
    c1.fit(svd_train,Y1)
    label_svd =np.array(c1.predict(svd_train)).astype('int')
    # label_svd =np.array(c1.predict(svd_embedding)

    pca = PCA(n_components=2)
    decomposed_embeddings = pca.fit_transform(X2)
    pca_train = svd.fit_transform(X1)
    c2.fit(pca_train,Y1)
    label_pca =np.array(c2.predict(pca_train)).astype('int')
    # label_pca =np.array(c2.predict(decomposed_embeddings)

    deep_semi = matrix_.collector.H2[2].T
    deep_train = matrix_.collector.H[2].T
    c3.fit(deep_train,Y1)
    label_deep =np.array(c3.predict(deep_train)).astype('int')
    # label_deep =np.array(c3.predict(deep_semi)
    only_semi = matrix.collector.H2[2].T
    only_semi_train = matrix.collector.H[2].T
    c4.fit(only_semi_train,Y1)
    label_semi=np.array(c4.predict(only_semi_train)).astype('int')
    step = 10
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Decision Graph for Dimension Reduction Techniques using Kmeans', fontsize=16)
    plt.title("TESTING")
    for label in test_class_labels:
        decomposed_embeddings_class = pca_train[label_pca == label]
        svd_embedding_class = svd_train[label_svd == label]
        deep_semi_class = deep_train[label_deep == label]
        only_semi_class = only_semi_train[label_semi == label]
        plt.subplot(1,4,1)
        plot_decision_regions(np.array(pca_train*10),Y1,clf=c1)
        plt.title('before training (only use PCA)')
        plt.legend()

        plt.subplot(1,4,2)
        plot_decision_regions(np.array(deep_train*200),label_deep,clf=c2)
        plt.title('after Deep SMNF')
        plt.legend()
        
        plt.subplot(1,4,3)
        plot_decision_regions(np.array(only_semi_train*100),Y1,clf=c3)
        plt.title('only use SMNF')
        plt.legend()

        plt.subplot(1,4,4)
        plot_decision_regions(np.array(svd_train),Y1,clf=c4)
        plt.title('only use SVD')
        plt.legend()
    plt.show()
def visualize_scatter(matrix,matrix_):
    X2 = matrix.X2.T
    X1 = matrix.X.T
    Y2 = matrix.Y2[0].T
    Y1 = matrix.Y[0].T
    test_class_labels = np.unique(np.array(Y2))
    c1 = KMeans(n_clusters=2)
    c2 = KMeans(n_clusters=2)
    c3 = KMeans(n_clusters=2)
    c4 = KMeans(n_clusters=2)
    # SVD 
    svd = TruncatedSVD(n_components=2)
    svd_train = svd.fit_transform(X1)
    svd_embedding = svd.fit_transform(X2)
    c1.fit(svd_train,Y1)
    label_svd = c1.predict(svd_train)
    # label_svd = c1.predict(svd_embedding)

    pca = PCA(n_components=2)
    decomposed_embeddings = pca.fit_transform(X2)
    pca_train = svd.fit_transform(X1)
    c2.fit(pca_train,Y1)
    label_pca = c2.predict(pca_train)
    # label_pca = c2.predict(decomposed_embeddings)

    deep_semi = matrix_.collector.H2[2].T
    deep_train = matrix_.collector.H[2].T
    c3.fit(deep_train,Y1)
    label_deep = c3.predict(deep_train)
    # label_deep = c3.predict(deep_semi)
    only_semi = matrix.collector.H2[2].T
    only_semi_train = matrix.collector.H[2].T
    c4.fit(only_semi_train,Y1)
    label_semi=c4.predict(only_semi_train)
    # label_semi=c4.predict(only_semi)
    step = 10
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('Kmeans Prediction over different Dimesion Reduction techniques', fontsize=16)
    for label in test_class_labels:
        # decomposed_embeddings_class = decomposed_embeddings[label_pca == label]
        # svd_embedding_class = svd_embedding[label_svd == label]
        # deep_semi_class = deep_semi[label_deep == label]
        # only_semi_class = only_semi[label_semi == label]        
        decomposed_embeddings_class = pca_train[Y1 == label]
        svd_embedding_class = svd_train[Y1 == label]
        deep_semi_class = deep_train[Y1== label]
        only_semi_class = only_semi_train[Y1 == label]
        # decomposed_embeddings_class = pca_train[label_pca == label]
        # svd_embedding_class = svd_train[label_svd == label]
        # deep_semi_class = deep_train[label_deep == label]
        # only_semi_class = only_semi_train[label_semi == label]
        plt.subplot(1,4,1)
        plt.scatter(decomposed_embeddings_class[::step, 1], decomposed_embeddings_class[::step, 0], label=str(label))
        plt.title('before training (only use PCA)')
        plt.legend()

        plt.subplot(1,4,2)
        plt.scatter(deep_semi_class[::step,1], deep_semi_class[::step,0],label=str(label))
        plt.title('after Deep SMNF')
        plt.legend()
        
        plt.subplot(1,4,3)
        plt.scatter(only_semi_class[::step, 1], only_semi_class[::step, 0], label=str(label))
        plt.title('only use SMNF')
        plt.legend()

        plt.subplot(1,4,4)
        plt.scatter(svd_embedding_class[::step, 1], svd_embedding_class[::step, 0], label=str(label))
        plt.title('only use SVD')
        plt.legend()
    plt.show()
    #DSNMF 
def evaluate_dimension(method,a,b,matrix):
    """Function aims to evaluate PCA, DSNMF ,SVD

    Args:
        method (str): d = DSNMF, p = PCA, s = SVD
    """
    # Initialize SVM
    dataset=[]
    metric = [] 
    y_true = matrix.Y[0].flatten()
    clf = KMeans(n_clusters=2)
    if method == 'd':
        clf.fit(a,matrix.Y[0])
        y_pred = clf.predict(a)
    elif method =='p':
        pca = PCA(n_components=2)
        temp = pca.fit_transform(matrix.X.T)
        temp2= pca.fit_transform(matrix.X2.T)
        clf.fit(temp,matrix.Y[0])
        y_pred = clf.predict(temp)
    elif method == "s": 
        svd =  TruncatedSVD(n_components = 2)
        temp = svd.fit_transform(matrix.X.T)
        temp2= svd.fit_transform(matrix.X2.T)
        clf.fit(temp,matrix.Y[0])
        y_pred = clf.predict(temp)
    metric = get_evaluation(y_pred,y_true)
    return metric
def get_evaluation(y_pred,y_true):
    metric = []
    tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
    class_error = (fn+fp)/(tn+fp+fn+tp)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = accuracy_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)
    nmi = normalized_mutual_info_score(y_true,y_pred)
    metric.append(class_error)
    metric.append(precision)
    metric.append(recall)
    metric.append(accuracy)
    metric.append(f1)
    metric.append(nmi)
    return metric