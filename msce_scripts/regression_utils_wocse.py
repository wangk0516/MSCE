
import numpy as np
import numpy.linalg as LA
import scipy.optimize as optimization
import random

from datetime import datetime
import os
import sys

from correlations_utils import calc_CorrMatr
from energy_utils import get_formation_energies
from energy_utils import get_compositions
from folder_utils import get_folder_names
from output_utils import write_output

MAX_FLOAT = sys.float_info.max

# -----------------------------------------------------------------------------

def CE_Expansion_FitAll_CEREG(ParentLattInfo, ClusterCollection, syscmd_corr, CEOptions, logfile):

    def optimize_LossFunction_FitAll(CorrMatr, ClusterCollection, energies_latt, natom_latt, CEOptions, logfile):

        if CEOptions['LossFunction'] in ['CE-REG-PAIRS', 'CE-REG-ALLCLUST', 'CE-REG-ALLCLUST-L1']:

            FittingParams_guess = CEOptions["InitialFittingParams"]
            Weights = CEOptions['Training_Weights']
            options = CEOptions["MinOptions"]
            opt_results = optimization.minimize(calc_fitting_error_FitAll,
                                                FittingParams_guess,
                                                args = (CorrMatr,
                                                        ClusterCollection,
                                                        energies_latt,
                                                        Weights,
                                                        CEOptions,
                                                        logfile),
                                                method = CEOptions['Optimizer'],
                                                options = options)
            logfile.write("  Minimization algorithm: {}; Successful? {}\n".format(CEOptions['Optimizer'], opt_results.success))
            logfile.write("  options = {}\n".format(options))
            logfile.write("\n")
            FittingParams = opt_results.x
            Residual = opt_results.fun
            logfile.write("  Residual Error = {:+15.12E}\n".format(Residual))
            logfile.write("  opt_results = \n{}\n".format(opt_results))

            ECI = FittingParams[:]

            return ECI
        # end if

    # -------------------------------------------------------------------------
    start_time = datetime.now()

    natom_latt = ParentLattInfo['sitenum']

    ClusterInfo = ClusterCollection['ClusterInfo']
    NumClusters = ClusterCollection['NumClusters']

    Training_Folders = CEOptions['Training_Folders']
    Training_CorrMatr = calc_CorrMatr(Training_Folders, ClusterInfo, syscmd_corr)
    Training_Concentrations, Training_energies_atom = get_formation_energies(Training_Folders, energy_file = CEOptions['EnergyFile'])
    Training_energies_latt = Training_energies_atom*natom_latt
    logfile.write("  Shape of Training_Concentrations: {}\n".format(np.shape(Training_Concentrations)))
    logfile.write("  Shape of Training_energies_atom: {}\n".format(np.shape(Training_energies_atom)))
    logfile.write("  Shape of Training_CorrMatr: {}\n\n".format(np.shape(Training_CorrMatr)))

    Testing_Folders = CEOptions['Testing_Folders']
    Testing_CorrMatr = calc_CorrMatr(Testing_Folders, ClusterInfo, syscmd_corr)
    Testing_Concentrations, Testing_energies_atom = get_formation_energies(Testing_Folders, energy_file = CEOptions['EnergyFile'])
    Testing_energies_latt = Testing_energies_atom*natom_latt
    logfile.write("  Shape of Testing_Concentrations: {}\n".format(np.shape(Testing_Concentrations)))
    logfile.write("  Shape of Testing_energies_atom: {}\n".format(np.shape(Testing_energies_atom)))
    logfile.write("  Shape of Testing_CorrMatr: {}\n".format(np.shape(Testing_CorrMatr)))

    logfile.write("  Cluster Expansion Parameters: \n")
    for item in CEOptions: logfile.write("    {}:  {}\n".format(item, CEOptions[item]))
    logfile.write("\n")

    time_elapsed = datetime.now() - start_time
    logfile.write('  Time elapsed collecting structual and energetic info (hh:mm:ss) {}\n\n'.format(time_elapsed))
    logfile.flush()

    if CEOptions['Optimizer'] == 'PINV' and CEOptions['LossFunction'] == 'MSE':

        pinvCorrMatr = LA.pinv(Training_CorrMatr)
        ECI = np.matmul(pinvCorrMatr, Training_energies_latt)
        pred_energies_latt = np.matmul(Training_CorrMatr, ECI)
        pred_energies_atom = pred_energies_latt/natom_latt
        MSE = np.sum((pred_energies_atom - Training_energies_atom)**2)/len(Training_energies_atom)
        RMSE = np.sqrt(MSE)
        logfile.write('  RMSE of Training: {:+.9f}\n'.format(RMSE))

        if CEOptions['CALC-CV'] == 'TRUE':
            LOOCV = calc_LOOCV(Training_CorrMatr, Training_energies_latt, ClusterCollection,
                               ParentLattInfo, CEOptions, CEOptions['K-Fold'], logfile)
            logfile.write("  CV = {:+.9f}\n".format(LOOCV))
        # end if

    elif (CEOptions['Optimizer'] in ['BFGS', 'L-BFGS-B']) and (CEOptions['LossFunction'] in ['CE-REG-PAIRS', 'CE-REG-ALLCLUST', 'CE-REG-ALLCLUST-L1']):
        print("LossFunction = {}".format(CEOptions['LossFunction']))

        Training_Weights = CEOptions['Training_Weights']

        ECI = optimize_LossFunction_FitAll(Training_CorrMatr, ClusterCollection,
                                           Training_energies_latt, natom_latt, CEOptions, logfile)
        time_elapsed = datetime.now() - start_time
        logfile.write('  Time elapsed in optimize_LossFunction_FitAll (hh:mm:ss) {}\n'.format(time_elapsed))
        logfile.flush()

        CEOptions["InitialFittingParams"] = ECI[:]

        np.savetxt("REGCE-ECI.out", ECI, fmt = "%16.9E")
        logfile.write('ECI = \n')
        for ieci in range(len(ECI)): logfile.write("{:>16.9E}\n".format(ECI[ieci]))
        logfile.write('\n\n')

        pred_energies_latt = np.matmul(Training_CorrMatr, ECI)
        pred_energies_atom = pred_energies_latt/natom_latt
        pred_err = pred_energies_atom - Training_energies_atom
        MSE = np.sum((pred_err)**2)/len(pred_err)
        RMSE = np.sqrt(MSE)
        logfile.write('  RMSE of Training: {:+.9f}\n\n'.format(RMSE))
        logfile.flush()

        Training_File = open("REGCE-Training.out", "w+")
        for istr in range(len(Training_Folders)):
            for jspec in range(len(Training_Concentrations[0, :])):
                Training_File.write(" {:>12.9f} ".format(Training_Concentrations[istr, jspec]))
            # end for-jspec
            Training_File.write(" {:>12.9f} {:>12.9f} {:>12.9f} {:.3f} {}\n".format(
                                Training_energies_atom[istr],
                                pred_energies_atom[istr],
                                pred_err[istr],
                                Training_Weights[istr],
                                Training_Folders[istr]))
        # end for-istr
        Training_File.close()

        Testing_pred_energy_latt = np.matmul(Testing_CorrMatr, ECI)
        Testing_pred_error_atom = (Testing_energies_latt - Testing_pred_energy_latt)/natom_latt
        Testing_File = open("REGCE-Testing.out", "w+")
        for istr in range(len(Testing_Folders)):
            for jspec in range(len(Testing_Concentrations[0, :])):
                Testing_File.write(" {:>12.9f} ".format(Testing_Concentrations[istr, jspec]))
            # end for-jspec

            Testing_File.write(" {:>12.9f} {:>12.9f} {:>12.9f} {}\n".format(
                                Testing_energies_atom[istr],
                                Testing_pred_error_atom[istr],
                                Testing_pred_error_atom[istr],
                                Testing_Folders[istr]))

        # end-istr
        Testing_File.close()

        if CEOptions['CALC-CV'] == 'TRUE':
            LOOCV = calc_LOOCV(Training_CorrMatr, Training_energies_latt, ClusterCollection,
                               ParentLattInfo, CEOptions, CEOptions['K-Fold'], logfile)
            logfile.write("  CV = {:+.9f}\n".format(LOOCV))
        # end if

    else:
        raise ValueError('Method {} with loss function {} is not implemented yet.'.format(CEOptions['Optimizer'], CEOptions['LossFunction']))
    # end of if

    return 0

def calc_fitting_error_FitAll(FittingParams, CorrMatr, ClusterCollection, energies, weights, CEOptions, logfile):
    # print("Call calc_fitting_error_FitAll ...")

    ParentLattInfo = CEOptions['ParentLattInfo']
    natom_latt = ParentLattInfo['sitenum']

    ECI = FittingParams[:]

    lam = CEOptions['lambda']

    if CEOptions['LossFunction'] == 'CE-REG-PAIRS':

        pred_energies = np.matmul(CorrMatr, ECI)

        pair_clusters = ClusterCollection['pair_clusters']
        NumClusters = ClusterCollection['NumClusters']
        pair_ECI = ECI[2:np.sum(NumClusters[:3])]

        M = 0.0
        NF = 0.0    # Normalizing factor
        for ipair in range(NumClusters[2]):
            rr = pair_clusters[ipair, 1]
            M += 0.5*(rr**lam)*pair_clusters[ipair, 0]*(pair_ECI[ipair])**2
            NF += 0.5*(rr**lam)*pair_clusters[ipair, 0]
        # end of for-ipair

        # print(np.shape(pred_energies), np.shape(energies), np.shape(weights))

        L = np.sum(weights*(pred_energies - energies)**2)/len(energies) + CEOptions['alpha']*M/NF

        logfile.write("LossFunction = {:12.9E} \n".format(L))
        logfile.flush()
        
        return L

    elif CEOptions['LossFunction'] == 'CE-REG-ALLCLUST':

        pred_energies = np.matmul(CorrMatr, ECI)

        ClusterInfo = ClusterCollection['ClusterInfo']
        NumClusters = ClusterCollection['NumClusters']

        M = 0.0
        NF = 0.0
        for icluster in range(np.sum(NumClusters)):
            rr = ClusterInfo[icluster, 1]
            M += 0.5*(rr**lam)*ClusterInfo[icluster, 0]*(ECI[icluster]**2)
            NF += 0.5*(rr**lam)*ClusterInfo[icluster, 0]
        # end of for-icluster

        L = np.sum(weights*(pred_energies - energies)**2)/len(energies) + CEOptions['alpha']*M/NF

        logfile.write("LossFunction = {:12.9E} \n".format(L))
        logfile.flush()

        return L

    elif CEOptions['LossFunction'] == 'CE-REG-ALLCLUST-L1':

        pred_energies = np.matmul(CorrMatr, ECI)

        ClusterInfo = ClusterCollection['ClusterInfo']
        NumClusters = ClusterCollection['NumClusters']

        M = 0.0
        NF = 0.0
        for icluster in range(np.sum(NumClusters)):
            rr = ClusterInfo[icluster, 1]
            M += (rr**lam)*np.abs(ECI[icluster])
            NF += rr**lam
        # end of for-icluster

        L = np.sum(weights*(pred_energies - energies)**2)/len(energies) + CEOptions['alpha']*M/NF

        logfile.write("LossFunction = {:12.9E} \n".format(L))
        logfile.flush()

        return L

    else:
        raise ValueError('Scheme {} is not implemented yet.'.format(LossFunc))

def calc_LOOCV(CorrMatr, energies, ClusterCollection, ParentLattInfo, CEOptions, nfold, logfile):

    options = CEOptions["MinOptions"]
    All_Training_Folders = CEOptions['Training_Folders']

    NumClusters = ClusterCollection['NumClusters']
    pair_clusters = ClusterCollection['pair_clusters']
    pair_direction = np.zeros([NumClusters[2], 3])
    pair_direction[:, 0] = pair_clusters[:, 6] - pair_clusters[:, 3]
    pair_direction[:, 1] = pair_clusters[:, 7] - pair_clusters[:, 4]
    pair_direction[:, 2] = pair_clusters[:, 8] - pair_clusters[:, 5]
    pair_direction = np.matmul(pair_direction, ParentLattInfo['axes'])

    natom_latt = ParentLattInfo['sitenum']

    AllWeights = CEOptions['Training_Weights']

    LOO_pred_error = np.zeros([len(energies)])
    for K in range(nfold):
        indices = np.arange(K, len(energies), nfold, dtype = 'int')
        LOO_CorrMatr = np.delete(CorrMatr, indices, 0)
        LOO_energies = np.delete(energies, indices, 0)
        LOO_weights = np.delete(AllWeights, indices, 0) 

        if CEOptions['Optimizer'] == 'PINV' and CEOptions['LossFunction'] == 'MSE':
            LOO_ECI = np.matmul(LA.pinv(LOO_CorrMatr), LOO_energies)

        elif ((CEOptions['Optimizer'] in ['BFGS', 'L-BFGS-B'])
              and (CEOptions['LossFunction'] in ['CE-REG-PAIRS', 'CE-REG-ALLCLUST', 'CE-REG-ALLCLUST-L1'])):

            ECI_guess = CEOptions["InitialFittingParams"]
            LOO_results = optimization.minimize(calc_fitting_error_FitAll,
                                                ECI_guess,
                                                args = (LOO_CorrMatr,
                                                        ClusterCollection,
                                                        LOO_energies,
                                                        LOO_weights,
                                                        CEOptions,
                                                        logfile),
                                                method = CEOptions['Optimizer'],
                                                options = options)

            # print("  Minimization algorithm: ", CEOptions, "; Successful? ", LOO_results.success)
            LOO_ECI = LOO_results.x
            # LOO_Residual = LOO_results.fun
            # print("  Residual Error = {:.9E}".format(opt_Residual))
        else:
            raise ValueError('CEOptions {} is not implemented yet!'.format(CEOptions))
        # end of if
        LOO_pred_energy = np.matmul(CorrMatr[indices, :], LOO_ECI)
        LOO_pred_error[indices] = (energies[indices] - LOO_pred_energy)/natom_latt
        # logfile.write("  dataset: {} {:12.6f}\n".format(K, np.mean(np.abs(LOO_pred_error[indices]))))

        for ipredstr in range(len(indices)):
            logfile.write(" dataset {}: {:>12.6f} {}\n".format(K, LOO_pred_error[indices[ipredstr]],
                                                   All_Training_Folders[indices[ipredstr]]))
        # end for-ipredstr
        logfile.flush()

    # end of for-K
    LOOCV = np.sqrt(np.sum(LOO_pred_error**2)/len(LOO_pred_error))

    return LOOCV

# -----------------------------------------------------------------------------

