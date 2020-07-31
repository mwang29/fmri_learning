import numpy as np
import warnings
import scipy
import pickle
from scipy import linalg
import scipy.io as sio
from pyriemann.utils.mean import mean_covariance
from pyriemann.estimation import Covariances
import sklearn.datasets
import sklearn.decomposition
import sys
import os
import h5py

# Disable

def utri2mat(utri):
    '''
    Converts upper triangular back to matrix form and fills in main diagonal with 1s
    '''
    n = int(-1 + np.sqrt(1 + 8 * len(utri))) // 2
    iu1 = np.triu_indices(n+1,1)
    ret = np.empty((n+1, n+1))
    ret[iu1] = utri
    ret.T[iu1] = utri
    np.fill_diagonal(ret, 1)
    return ret

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore


def enablePrint():
    sys.stdout = sys.__stdout__


np.seterr(divide='ignore', invalid='ignore')


def get_glasser():
    '''
    Navigates through file tree and extracts FCs with optional reconstruction
    '''
    # Yeo ordering
    fname = '../data/100_unrelated.csv'
    yeo = True
    if yeo:
        yeo_order = list(sio.loadmat("../data/yeo_RS7_N374.mat",
                                     squeeze_me=True,
                                     struct_as_record=False)['yeoOrder'] - 1)
    # Load subject ID and task names
    subjectids = np.loadtxt(fname, dtype=np.int)
    nSubj = len(subjectids)
    tasks = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR',
             'rfMRI_REST2_RL', 'tfMRI_EMOTION_LR', 'tfMRI_EMOTION_RL',
             'tfMRI_GAMBLING_LR', 'tfMRI_GAMBLING_RL', 'tfMRI_LANGUAGE_LR',
             'tfMRI_LANGUAGE_RL', 'tfMRI_MOTOR_LR', 'tfMRI_MOTOR_RL',
             'tfMRI_RELATIONAL_LR', 'tfMRI_RELATIONAL_RL', 'tfMRI_SOCIAL_LR',
             'tfMRI_SOCIAL_RL', 'tfMRI_WM_LR', 'tfMRI_WM_RL']
    M = {}
    # Walk through file tree and extract FCs
    for task in tasks:
        masterFC_dir = '../data/results_SIFT2'
        restingstatename = 'fMRI/' + task + '/FC/FC_glasser_subc_GS_bp_z.mat'
        task_matrices = []
        for subject in subjectids:
            filename = masterFC_dir + '/' + \
                str(subject) + '/' + restingstatename
            mat = sio.loadmat(filename, squeeze_me=True,
                              struct_as_record=False)
            A_orig = mat['FC']
            if yeo:
                A_orig = A_orig[np.ix_(yeo_order, yeo_order)]
            np.fill_diagonal(A_orig, 1)
            task_matrices.append(A_orig)
        M[task] = np.array(task_matrices)
    test = np.concatenate((M['rfMRI_REST1_LR'], M['tfMRI_EMOTION_LR'],
                           M['tfMRI_GAMBLING_LR'], M['tfMRI_LANGUAGE_LR'],
                           M['tfMRI_MOTOR_LR'], M['tfMRI_RELATIONAL_LR'],
                           M['tfMRI_SOCIAL_LR'], M['tfMRI_WM_LR']))
    retest = np.concatenate((M['rfMRI_REST1_RL'], M['tfMRI_EMOTION_RL'],
                             M['tfMRI_GAMBLING_RL'], M['tfMRI_LANGUAGE_RL'],
                             M['tfMRI_MOTOR_RL'], M['tfMRI_RELATIONAL_RL'],
                             M['tfMRI_SOCIAL_RL'], M['tfMRI_WM_RL']))
    del M
    all_FC = np.concatenate((test, retest))
    del test, retest
    return all_FC, nSubj


def get_schaefer(parc):
    with open(f'../data/schaefer{parc}.pickle', 'rb') as f:
        all_FC = pickle.load(f)
    nSubj = int(all_FC.shape[0] / 16)
    return all_FC, nSubj


def get_twins(parc, twin='DZ'):
    '''
    Navigates through file tree and extracts test/retest FCs 
    '''
    master_dir = '../data/twins'
    tasks = ['rest', 'emotion', 'gambling', 'language', 'motor', 'relational', 'social', 'wm']
    FC, test, retest = {}, {}, {}
    for task in tasks:
        task_dir = master_dir + f'/{task.upper()}/origmat_{twin}_schaefer{parc}_tests.mat'
        f = h5py.File(task_dir, 'r')
        for k, v in f.items():
            test[task] = np.array(v)
        task_dir = master_dir + f'/{task.upper()}/origmat_{twin}_schaefer{parc}_retests.mat'
        f = h5py.File(task_dir, 'r')
        for k, v in f.items():
            retest[task] = np.array(v)
        FC[task] = np.concatenate((test[task], retest[task])) 
    return FC




def q1invm(q1, eig_thresh=0):
    q1 += np.eye(q1.shape[0])
    U, S, V = scipy.linalg.svd(q1)
    S = np.diag(S ** (-1 / 2))
    Q1_inv_sqrt = U @ S @ V
    Q1_inv_sqrt = (Q1_inv_sqrt + np.transpose(Q1_inv_sqrt)) / 2
    return Q1_inv_sqrt


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
    tangent_FC = np.empty(Q.shape)
    for idx, fc in enumerate(Q):
        if idx % 100 == 0:
            print(f'{idx}/{Q.shape[0]}')

        tangent_FC[idx] = linalg.logm(fc)
        enablePrint()
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


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parcellation = input('Parcellation: ')
    if parcellation.lower() == 'schaefer':
        for parc in np.arange(100, 400, 100):
            for ref in ['euclid', 'riemann', 'kullback_sym', 'harmonic', 'logeuclid']:
                print(f'{parc}:{ref}')
                # Navigate tree and get raw correlation FC matrices
                all_FC, nSubj = get_schaefer(parc)
                print("All FCs successfully loaded!\n")
                tangent_FCs = tangential(all_FC, ref)
                with open(f'../data/tangent_fcs/schaefer{parc}_{ref}.pickle', 'wb') as f:
                    pickle.dump(tangent_FCs, f, protocol=4)
    elif parcellation.lower() == 'glasser':
        for ref in ['euclid', 'riemann', 'kullback_sym', 'harmonic', 'logeuclid']:
            print(ref)
            # Navigate tree and get raw correlation FC matrices
            all_FC, nSubj = get_glasser()
            print("All FCs successfully loaded!\n")
            tangent_FCs = tangential(all_FC, ref)
            with open(f'../data/tangent_fcs/glasser_{ref}.pickle', 'wb') as f:
                pickle.dump(tangent_FCs, f, protocol=4)
    elif parcellation.lower() == 'twins':
        for ref in ['logeuclid']:
            print(f'Using {ref} reference')
            for parc in np.arange(100, 500, 100):
                print(f'{parc} Region Parcellation')
                for twin in ['DZ', 'MZ']:
                    FCs = get_twins(parc, twin)
                    for task in ['rest', 'emotion', 'gambling', 'language', 'motor', 'relational', 'social', 'wm']:
                        FC = np.zeros((FCs[task].shape[0], parc+14, parc+14))
                        for idx, utri in enumerate(FCs[task]):
                            FC[idx] = utri2mat(utri)
                        tangent_FCs = tangential(FC, ref)
                        with open(f'../data/tangent_fcs/twins/{task}/{parc}_{twin}_{ref}.pickle', 'wb') as f:
                            pickle.dump(tangent_FCs, f, protocol=4)
        pass
    else:
        print("Error: Choose 'Glasser' or 'Schaefer'.\n")
