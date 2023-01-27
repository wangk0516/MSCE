#!/home/kw7rr/anaconda3/bin/python

import numpy as np
import random

import os
import sys
from datetime import datetime
import shutil

# ---------- Import modules in kwlib ------------------------------------------

sys.path.append("/home/kw7rr/msce_scripts")
from atat_io import read_atat_latt
from cluster_utils import read_cluster_info
from folder_utils import get_folder_names
from regression_utils_cse import CE_Expansion_FitAll
from input_utils import read_CE_params

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    start_time = datetime.now()
    timestr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    CEOptions = read_CE_params(filename = 'CE_Params.in')
    CEOptions['alpha'] = 1E0
    CEOptions['NumAtten'] = 2

    CEOptions['Optimizer'] = 'L-BFGS-B'
    CEOptions['LossFunction'] = 'FITALL-CSE-L1'
    CEOptions['lambda'] = 4.0
    CEOptions['MinOptions'] = {'ftol': 2.22E-9, 'gtol': 1.0E-9, 'maxiter': 1E6, 'maxfun': 1E6, 'eps': 1e-09}
    CEOptions['TimeStr'] = timestr

    logfile = open("MSCE-Alpha-{:.6E}-Log-WT.out".format(CEOptions['alpha']), "w+")
    logfile.write("\n")

    cwd = os.getcwd()
    logfile.write('  Current working directory: {}\n'.format(cwd))
    # ---------------------------------------

    rcut = CEOptions['Cut-Off']
    cmd_genclust = "~/bin/atat_exec_kw_hex_MultiAtten/corrdump -clus -l=lat.in -2={:.6f} -3={:.6f} -4={:.6f} -5={:.6f} -6={:.6f}".format(rcut[0], rcut[1], rcut[2], rcut[3], rcut[4])
    logfile.write("  Command to generate clusters:\n  {}\n".format(cmd_genclust))

    # equiv_clus_folder = "equivalent_clusters"
    # if os.path.exists(equiv_clus_folder): shutil.rmtree(equiv_clus_folder)
    # os.mkdir(equiv_clus_folder)
    os.system(cmd_genclust)

    ClusterCollection = read_cluster_info(filename = CEOptions['ClusterFile'])
    NumClusters = ClusterCollection['NumClusters']
    ClusterInfo = ClusterCollection['ClusterInfo']
    MaxDiameter = ClusterCollection['MaxDiameter']
    r_2, r_3, r_4, r_5, r_6 = MaxDiameter[2:] + CEOptions['EPS_LENGTH']

    logfile.write("  NumClusters = {}\n".format(NumClusters))

    logfile.write("  Numbers of pairs, triplets, quaduplets, site-5 and site-6 clusters: {}, {}, {}, {}, {}\n".format(NumClusters[2], 
                  NumClusters[3], NumClusters[4], NumClusters[5], NumClusters[6]))
    logfile.write("  Cut-off distances for pairs, triplets and quaduplets: {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}\n".format(r_2,
                  r_3, r_4, r_5, r_6))

    cmd_correlation = "~/bin/atat_exec_kw_hex_MultiAtten/corrdump -sig=9 -cf=../clusters.out -l=../lat.in -s=str.out " \
                    + "-2={:.6f} -3={:.6f} -4={:.6f} -5={:.6f} -6={:.6f} > correlations.out".format(r_2, r_3, r_4, r_5, r_6) 
    logfile.write("  Command for corrdump: \n    {}\n".format(cmd_correlation))
    logfile.write("\n")

    # Initial fitting parameters
    if os.path.exists("initial_params.out"):
        CEOptions["InitialFittingParams"] = np.loadtxt('initial_params.out', float)
    else:
        InitialParams = np.zeros([np.sum(NumClusters) +  CEOptions['NumAtten']])
        for K in range(len(InitialParams)):
            if K < CEOptions['NumAtten']:
                InitialParams[K] += 10**random.uniform(-2, 1)
            else:
                InitialParams[K] += random.uniform(0.0001, 0.001)
        # end for-K
        CEOptions["InitialFittingParams"] = InitialParams
    # end if

    LatticeInfo = read_atat_latt(CEOptions['LatticeFile'])
    LatticeInfo['syminfo'] = 'hex'
    CEOptions['ParentLattInfo'] = LatticeInfo

    directions = np.loadtxt(CEOptions["DirectionFile"], float)
    cse = np.loadtxt(CEOptions["CSECurveFile"], float)
    CEOptions['Directions'] = directions
    CEOptions['CSECurves'] = cse

    Training_Folders = get_folder_names(incl = [CEOptions['StructureFile'], CEOptions['EnergyFile']],
                               excl = ['error'])

    Testing_Folders = get_folder_names(incl = [CEOptions['StructureFile'], CEOptions['EnergyFile'], 'error'],
                               excl = [])

    CEOptions['Training_Weights'] = np.ones([len(Training_Folders)])

    logfile.write('  Number of training structures: {}\n'.format(len(Training_Folders)))
    logfile.write('  Number of tesing structures: {}\n'.format(len(Testing_Folders)))
    logfile.write("\n")

    CEOptions['Training_Folders'] = Training_Folders
    CEOptions['Testing_Folders'] = Testing_Folders
    logfile.flush()

    CE_Expansion_FitAll(LatticeInfo, ClusterCollection, cmd_correlation, CEOptions, logfile)
    logfile.write("\n")
    logfile.flush()

    # ---------------------------------------
    time_elapsed = datetime.now() - start_time
    logfile.write('  Time elapsed (hh:mm:ss) {}\n\n'.format(time_elapsed))
    logfile.close()

