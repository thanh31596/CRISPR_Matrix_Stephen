from numpy.core.numeric import False_
from sklearn.metrics import f1_score, accuracy_score
from copy import deepcopy
from deepsmnf import *
from NMFbase import *
from SMNF import *
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
if __name__ == "__main__":

    # Processing data from twitter
    # X,Y=twitter_processing(X_train)
    #df = pd.read_excel(path)
    df = pd.read_csv('/home/stephenvu/Documents/Dr.KhanhLuongQUT/Dr_Luong_bigUpdate/QMI_train_and_test_tfidf.csv')  # got it
    df2 = pd.read_csv('/home/stephenvu/Documents/Dr.KhanhLuongQUT/Dr_Luong_bigUpdate/label.csv')[:5013]
    py = pd.get_dummies(df2['labels'])
    s = pd.concat([df, py], join='outer', axis=1)
    layers = [100, 50, 2]  # manually assign the list of layers
    s = s[s.iloc[:, :-2].sum(axis=1) > 1]
    df_ = s.iloc[:, :-2].values
    Y = s.iloc[:, -2:].values

    # Perform pre-training data
    matrix = SNMF(df_, Y, layers=layers, k=30, option='r')
    matrix.compute_factors(random_init=True)
    # a = matrix.collector['H'][2].T
    # b = matrix.collector['H2'][2].T
    # cs_ = KMeans(n_clusters=2, init='k-means++', random_state=42).fit(a)
    # cs_3 = KMeans(n_clusters=2, init='k-means++', random_state=42).fit(b)
    # #curr_accuracy = accuracy_score(matrix_.Y[1], cs_.labels_)
    # print("Training:{}".format(accuracy_score(matrix.Y[1], cs_.labels_)))
    # print("Testing:{}".format(accuracy_score(matrix.Y2[1], cs_3.labels_)))
    # unique, counts = np.unique(cs_.labels_, return_counts=True)
    # unique1, counts1 = np.unique(cs_3.labels_, return_counts=True)
    # print(dict(zip(unique, counts)))
    # print(dict(zip(unique1, counts1)))
    matrix.collector.drop(columns=['index'], inplace=True)
    # matrix_copy = deepcopy(matrix)
    # deepMatrix = Deep(matrix_copy, alpha=100,  iteration=100, lam1=1, lam2=10, lamz=10, normH=True, normH2=True,normZ=True, othorZ=False, othorH=False)

    # matrix_ = deepMatrix.run_main()

    # a = matrix_.collector['H'][2].T
    # b = matrix_.collector['H2'][2].T
    # cs = KMeans(n_clusters=2,  random_state=42).fit(a)
    # cs3 = KMeans(n_clusters=2,  random_state=42).fit(b)
    # #curr_accuracy = accuracy_score(matrix_.Y[1], cs.labels_)
    # print("Training:{}".format(accuracy_score(matrix_.Y[0], cs.labels_)))
    # print("Testing:{}".format(accuracy_score(matrix_.Y2[0], cs3.labels_)))
    # unique, counts = np.unique(cs.labels_, return_counts=True)
    # unique1, counts1 = np.unique(cs3.labels_, return_counts=True)
    # print(dict(zip(unique, counts)))
    # print(dict(zip(unique1, counts1)))

    # # --------------------------------------------------
    # #Create some array for later output
    draw = []
    draw2 = []
    errors = []
    f1train = []
    f1test = []
    iteration_ = [50]
    # Create hyperparameter set
    #k_ = [5, 10, 30, 50, 100]
    alpha_ = [100 ]  # [ 0.0, 0.003, 600, 9000]
    lam1_ = [1]  # [ 0.0, 0.003, 600, 9000]
    lam2_ = [10,0.1,1,100]  # [ 0.0, 0.003, 600, 9000]
    lamz_ = [10,0.1,1,100]  # [ 0.0, 0.003, 600, 9000]
    normH_ = [True]
    normH2_ = [True]
    normZ_ = [True]
    othorH_ = [False]
    othorZ_ = [False]
    # Create deep copy instance to avoid overwritten values from iteration
    parameterSet = create_hyperparameter(
        alpha_, lam1_, lam2_, lamz_,   othorH_, othorZ_, normZ_, normH_,normH2_)

    # Create dataframe output
    result = pd.DataFrame(parameterSet, columns=['alpha',  'lam1', 'lam2', 'lamz','iteration',  'othorH', 'othorZ','normH','normH2', 'normZ', ])

    count = 0
    # print(parameterSet)
    for k in range(len(parameterSet)):
        # matrix=0
        c = deepcopy(matrix)

        print("Model setting iteration: {}".format(count))
        count += 1
        max_accuracy = 0
        
        alpha = parameterSet[k][0]

        lam1 = parameterSet[k][1]

        lam2 = parameterSet[k][2]

        lamz = parameterSet[k][3]
        iteration = parameterSet[k][4]
        othorH = parameterSet[k][5]
        othorZ = parameterSet[k][6]
        normZ = parameterSet[k][7]
        normH = parameterSet[k][8]
        normH2=parameterSet[k][9]
        print("Parameter set: alpha = {0}, iteration ={1},lam1={2},lam2={3},lamz={4},normz={5},normh={6},othorZ={7},othorH={8}".format(alpha,iteration,lam1,lam2,lamz,normZ,normH,othorZ,othorH))
        deepMatrix = Deep(c, alpha,  iteration, lam1,
                          lam2, lamz, normH, normZ, othorZ, othorH)

        matrix_ = deepMatrix.run_main()

        # errors.append(matrix_.brr)

        a = matrix_.collector['H'][2].T
        b = matrix_.collector['H2'][2].T
        cs = KMeans(n_clusters=2, random_state=42).fit(a)
        cs3 = KMeans(n_clusters=2, random_state=42).fit(b)
        f1train.append(f1_score(matrix_.Y[0], cs.labels_))
        f1test.append(f1_score(matrix_.Y2[0], cs3.labels_))
        curr_accuracy = accuracy_score(matrix_.Y[0], cs.labels_)
        print("Training:{}".format(accuracy_score(matrix_.Y[0], cs.labels_)))
        print("Testing:{}".format(accuracy_score(matrix_.Y2[0], cs3.labels_)))
        unique, counts = np.unique(cs.labels_, return_counts=True)
        print(dict(zip(unique, counts)))
        unique2, counts2 = np.unique(cs3.labels_, return_counts=True)
        print(dict(zip(unique2, counts2)))
        draw2.append(accuracy_score(matrix_.Y2[0], cs3.labels_))
        draw.append(curr_accuracy)

        if max_accuracy < curr_accuracy:
            max_accuracy = curr_accuracy
    result['F1_train'] = f1train
    result['F1_test'] = f1test
    result['accuracy_training'] = draw
    result['accuracy_test'] = draw2
    print("Maximum accuracy achieved by our model through grid searching : ", max_accuracy)

    # elapsed_time_secs = time.time() - start_time
    # msg = "Execution took: %s secs (Wall clock time)" % timedelta(
    #     seconds=round(elapsed_time_secs))

    result.to_excel('hopeless.xlsx')
