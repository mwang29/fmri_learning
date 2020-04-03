import numpy as np
import scipy.io as sio

fname = '100_unrelated.csv'
subjectids = np.loadtxt(fname, dtype=np.int)
tasks = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR',
         'rfMRI_REST2_RL', 'tfMRI_EMOTION_LR', 'tfMRI_EMOTION_RL',
         'tfMRI_GAMBLING_LR', 'tfMRI_GAMBLING_RL', 'tfMRI_LANGUAGE_LR',
         'tfMRI_LANGUAGE_RL', 'tfMRI_MOTOR_LR', 'tfMRI_MOTOR_RL',
         'tfMRI_RELATIONAL_LR', 'tfMRI_RELATIONAL_RL', 'tfMRI_SOCIAL_LR',
         'tfMRI_SOCIAL_RL', 'tfMRI_WM_LR', 'tfMRI_WM_RL']

for task in tasks:
    masterFC_dir = 'results_SIFT2'
    restingstatename = 'fMRI/' + task + '/FC/FC_glasser_subc_GS_bp_z.mat'
    all_matrices = []
    for subject in subjectids:
        filename = masterFC_dir + '/' + str(subject) + '/' + restingstatename
        mat = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
        A_orig = mat['FC']
        A_orig = (A_orig + A_orig.T) / 2
        np.fill_diagonal(A_orig, 0)
        all_matrices.append(A_orig)
    # This is the 100 by 374 by 374 matrices of 100 unrelated subjects
    all_matrices = np.array(all_matrices)
    print(all_matrices.shape)
