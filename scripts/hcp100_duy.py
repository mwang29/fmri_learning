import scipy.io as sio
import numpy as np

fname = '100_unrelated.csv'
subjectids = np.loadtxt(fname, dtype=np.int)
masterFC_dir = '/Users/duyduong/Dropbox/Purdue University/Duy Goni/LLNL/2019/Codes/codes/results_SIFT2'
restingstatename = 'fMRI/rfMRI_REST2_RL/FC/FC_glasser_subc_GS_bp_z.mat'
all_matrices = []
for subject in subjectids:
    filename = masterFC_dir + '/' + str(subject) + '/' + restingstatename
    mat = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    A_orig = mat['FC']
    A_orig = (A_orig + A_orig.T) / 2
    np.fill_diagonal(A_orig, 1)
    all_matrices.append(A_orig)
# This is the 100 by 374 by 374 matrices of 100 unrelated subjects
all_matrices = np.array(all_matrices)
print(all_matrices.shape)
