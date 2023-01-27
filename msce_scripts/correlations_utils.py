
import numpy as np
import os
import sys

MAX_FLOAT = sys.float_info.max

# -----------------------------------------------------------------------------

def calc_CorrMatr(strnames, cluster_info, syscmd_corr, prefix = './'):

    n_cluster = len(cluster_info[:, 0])
    n_str = len(strnames)
    correlation_matrix = MAX_FLOAT*np.ones([n_str, n_cluster])
    cwd = os.getcwd()
    for I in range(n_str):
        os.chdir(prefix + strnames[I])
        # print('  ' + os.getcwd())

        if os.path.exists('correlations.out'):
            correlation = np.loadtxt('correlations.out', float)
        else:
            os.system(syscmd_corr)
            correlation = np.loadtxt('correlations.out', float)
        # end if

        # print('    Number of correlations: {}'.format(np.shape(correlation)))
        # print(correlation)

        # print(I, n_cluster, np.shape(cluster_info), np.shape(correlation))

        # multiply correlation by the multiplicity
        if n_cluster == len(correlation):
            for J in range(len(correlation)):
                correlation_matrix[I, J] = correlation[J]*cluster_info[J, 0]
            # end of for-J
        else:
            raise ValueError('Incompatible dimensions of cluster_info and correlation: {} and {}'.format(np.shape(cluster_info), np.shape(correlation)))
            # print('Generate correlations for {} ...'.format(strnames[I]))
            # os.system(syscmd_corr)
            # correlation = np.loadtxt('correlations.out', float)
        # end of if

        os.chdir(cwd)
    # end of for-f

    count = 0
    for row in range(n_str):
        for col in range(n_cluster):
            if correlation_matrix[row, col] == MAX_FLOAT: count += 1
    if count > 0: raise ValueError('{} values of correlation matrix are not assigned'.format(count))

    # print('cluster_info = \n', cluster_info)
    # print('correlation_matrix = ', np.shape(correlation_matrix))
    # print(correlation_matrix)

    return correlation_matrix

# -----------------------------------------------------------------------------




