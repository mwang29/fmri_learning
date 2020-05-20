import numpy as np
import csv
import h5py
import pickle
import scipy
from pyriemann.utils.mean import mean_covariance
import sklearn.datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

np.seterr(divide='ignore', invalid='ignore')


def get_data():
    '''
    Extracts FCs for each parcellation and stores into a dictionary of dictionaries
    '''
    all_parc = {}
    # Yeo ordering
    for p in np.arange(100, 600, 100):
        temp_parc = {}
        filename = f'../data/FC_all_vec_424_schaefer_subc_{p}.mat'
        f = h5py.File(filename, 'r')
        for k, v in f.items():
            temp_parc[k] = np.array(v)
        all_parc[p] = temp_parc
    return all_parc


def q1invm(q1, eig_thresh=0):
    U, S, V = scipy.linalg.svd(q1)
    s = np.diag(S)
    s[s < eig_thresh] = eig_thresh
    S = np.diag(s ** (-1 / 2))
    Q1_inv_sqrt = U * S * np.transpose(V)
    Q1_inv_sqrt = (Q1_inv_sqrt + np.transpose(Q1_inv_sqrt)) / 2
    return Q1_inv_sqrt


def qlog(q):
    U, S, V = scipy.linalg.svd(q)
    s = np.diag(S)
    S = np.diag(np.log(s))
    Q = U * S * np.transpose(V)
    return Q


def tangential(all_FC, ref):
    # Regularization for riemann
    if ref in ['riemann', 'kullback_sym', 'logeuclid']:
        print("Adding regularization!")
        eye_mat = np.eye(all_FC.shape[1])
        scaling_mat = np.repeat(eye_mat[None, ...], all_FC.shape[0], axis=0)
        all_FC += scaling_mat
    Cg = mean_covariance(all_FC, metric=ref)
    Q1_inv_sqrt = q1invm(Cg)
    Q = Q1_inv_sqrt @ all_FC @ Q1_inv_sqrt
    tangent_FC = np.array([qlog(a) for a in Q])
    return tangent_FC


def pca_recon(FC, pctComp=None):
    '''
    Reconstructs FC based on number of principle components
    '''
    if pctComp is None:
        return FC
    FC = np.reshape(FC, (FC.shape[0], -1))
    nComp = int(FC.shape[0] * pctComp)
    mu = np.mean(FC, axis=0)
    pca_rest = sklearn.decomposition.PCA()
    pca_rest.fit(FC)
    SCORES = pca_rest.transform(FC)[:, :nComp]
    COEFFS = pca_rest.components_[:nComp, :]
    FC_recon = np.dot(SCORES, COEFFS)
    del SCORES, COEFFS
    FC_recon += mu
    FC_recon = np.reshape(FC_recon, (FC.shape[0], 374, 374))
    return FC_recon


def utri2mat(utri):
    n = int(-1 + np.sqrt(1 + 8 * len(utri))) // 2
    iu1 = np.triu_indices(n)
    ret = np.empty((n, n))
    ret[iu1] = utri
    ret.T[iu1] = utri
    return ret


if __name__ == '__main__':
    with open('../data/schaefer.pickle', 'rb') as f:
        all_parc = pickle.load(f)
    nSubj = int(all_parc[100]['FC_all_vec'].shape[0] / 16)
    nFCs = int(all_parc[100]['FC_all_vec'].shape[0])
    classifier = 'task'
    if classifier == 'task':
        labels = np.tile(np.repeat(np.arange(0, 8), nSubj), 2)
        indices = np.random.permutation(nSubj)
        train_idx = indices[:int(0.80 * nSubj)]
        test_idx = indices[int(0.8 * nSubj):]
        train_idx_all, test_idx_all = np.empty(
            0, dtype=int), np.empty(0, dtype=int)
        for fc in np.arange(0, 16):
            train_idx_all = np.concatenate(
                (train_idx_all, (fc * nSubj) + train_idx)).astype(int)
            test_idx_all = np.concatenate(
                (test_idx_all, (fc * nSubj) + test_idx)).astype(int)
        train_idx = train_idx_all
        test_idx = test_idx_all
    elif classifier == 'subject':
        labels = np.tile(np.tile(np.arange(0, nSubj), 8), 2)
        indices = np.random.permutation(all_parc[100]['FC_all_vec'].shape[0])
        train_idx = indices[:int(0.80 * all_parc[100]['FC_all_vec'].shape[0])]
        test_idx = indices[int(0.80 * all_parc[100]['FC_all_vec'].shape[0]):]
    else:
        pass

    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    accuracies = {}

    for parc in np.arange(100, 600, 100):
        print(f'Using Schaefer {parc} parcellation...')
        testFCs = all_parc[parc]['FC_all_vec'][::2]
        retest_FCs = all_parc[parc]['FC_all_vec'][1::2]
        reordered_FCs = np.concatenate([testFCs, retest_FCs])
        mat_FCs = np.zeros((nFCs, parc + 13, parc + 13))
        for i in np.arange(0, nFCs):
            mat_FCs[i] = utri2mat(reordered_FCs[i])
        for ref in ['euclid', 'harmonic', 'none']:
            print(f'Testing {ref}...')
            if ref != 'none':
                all_FC = tangential(mat_FCs, ref)
                train_FCs = np.zeros(
                    (len(train_idx), reordered_FCs.shape[1]), dtype=np.float32)
                for idx, mat in enumerate(all_FC[train_idx]):
                    train_FCs[idx] = mat[np.triu_indices(mat.shape[0], k=0)]
                test_FCs = np.zeros(
                    (len(test_idx), reordered_FCs.shape[1]), dtype=np.float32)
                for idx, mat in enumerate(all_FC[test_idx]):
                    test_FCs[idx] = mat[np.triu_indices(mat.shape[0], k=0)]
            else:
                train_FCs = reordered_FCs[train_idx]
                test_FCs = reordered_FCs[test_idx]
            print(f'Fitting KNN...')
            neigh = KNeighborsClassifier(n_neighbors=30, metric='correlation')
            neigh.fit(train_FCs, train_labels)
            predicted = neigh.predict(test_FCs)
            acc = accuracy_score(test_labels, predicted)
            print(f'Accuracy: {acc}')
            accuracies[f"{ref}_{parc}"] = acc

    a_file = open(f"../results/schaefer_{parc}_{classifier}.csv", "w")

    writer = csv.writer(a_file)
    for key, value in accuracies.items():
        writer.writerow([key, value])

    a_file.close()
