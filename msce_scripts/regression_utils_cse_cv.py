
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

from bz_utils import get_StructDictList 
from bz_utils import calculate_structural_CSE_StructDictList

MAX_FLOAT = sys.float_info.max

# -----------------------------------------------------------------------------

def CE_Expansion_FitAll(ParentLattInfo, ClusterCollection, syscmd_corr, CEOptions, logfile):

    def optimize_LossFunction_FitAll(CorrMatr, ClusterCollection, energies_latt, natom_latt, CEOptions, logfile):

        if CEOptions['LossFunction'] in ['FITALL-CSE-SmoothPairs', 'FITALL-CSE-SmoothAllClusters', 'FITALL-CSE-L1']:

            options = CEOptions['MinOptions']
            for item in options: logfile.write("options[{}] = {}\n".format(item, options[item]))

            FittingParams_guess = CEOptions["InitialFittingParams"]
            opt_results = optimization.minimize(calc_fitting_error_FitAll,
                                                FittingParams_guess,
                                                args = (CorrMatr,
                                                        ClusterCollection,
                                                        energies_latt,
                                                        CEOptions,
                                                        logfile),
                                                method = CEOptions['Optimizer'],
                                                options = options)
            logfile.write("  Minimization algorithm: {}; Successful? {}\n".format(CEOptions['Optimizer'], opt_results.success))
            FittingParams = opt_results.x
            Residual = opt_results.fun
            logfile.write("  Residual Error = {:+15.12E}\n".format(Residual))
            logfile.write("  opt_results = \n{}\n".format(opt_results))

            NumAtten = CEOptions['NumAtten']
            AttenCoeffs = FittingParams[:NumAtten]
            ECI = FittingParams[NumAtten:]

            return AttenCoeffs, ECI
        # end if

    # -------------------------------------------------------------------------
    start_time = datetime.now()

    natom_latt = ParentLattInfo['sitenum']

    ClusterInfo = ClusterCollection['ClusterInfo']
    NumClusters = ClusterCollection['NumClusters']
    cse = CEOptions['CSECurves']
    directions = CEOptions['Directions'] 

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

    if (CEOptions['Optimizer'] in ['BFGS', 'L-BFGS-B']) and (CEOptions['LossFunction'] in ['FITALL-CSE-SmoothPairs', 'FITALL-CSE-SmoothAllClusters', 'FITALL-CSE-L1']):
        # print("LossFunction = {}".format(CEOptions['LossFunction']))

        Training_StructDictList = get_StructDictList(Training_Folders, cse, directions, ParentLattInfo, logfile)
        CEOptions['Training_StructDictList'] = Training_StructDictList
        time_elapsed = datetime.now() - start_time
        logfile.write('  Time elapsed creating Training_StructDictList (hh:mm:ss) {}\n\n'.format(time_elapsed))

        if len(Testing_Folders) > 0:
            Testing_StructDictList = get_StructDictList(Testing_Folders, cse, directions, ParentLattInfo, logfile)
            CEOptions['Testing_StructDictList'] = Testing_StructDictList
            time_elapsed = datetime.now() - start_time
            logfile.write('  Time elapsed creating Testing_StructDictList (hh:mm:ss) {}\n\n'.format(time_elapsed))
        # end if

        Training_Weights = CEOptions['Training_Weights']
        count = 0
        MaxError = 1.0
        flag_continue = True
        while flag_continue:
            flag_continue = False

            AttenCoeffs, ECI = optimize_LossFunction_FitAll(Training_CorrMatr, ClusterCollection, Training_energies_latt, natom_latt, CEOptions, logfile)
            time_elapsed = datetime.now() - start_time
            logfile.write('  Time elapsed in optimize_LossFunction_FitAll (hh:mm:ss) {}\n'.format(time_elapsed))
            logfile.flush()

            pred_CSE_latt = calculate_structural_CSE_StructDictList(AttenCoeffs, Training_Folders, natom_latt, Training_StructDictList, logfile)
            pred_energies_latt = np.matmul(Training_CorrMatr, ECI)
            pred_energies_atom = (pred_energies_latt + pred_CSE_latt)/natom_latt
            MSE = np.sum((pred_energies_atom - Training_energies_atom)**2)/len(Training_energies_atom)
            RMSE = np.sqrt(MSE)
            logfile.write('  RMSE of Training: {:+.9f}\n'.format(RMSE))
            logfile.flush()

            Training_prederr = pred_energies_atom - Training_energies_atom

            Training_File = open("MSCE-Alpha-{:.6E}-Training_WT-{}.out".format(CEOptions['alpha'], count), "w+")
            for istr in range(len(Training_Folders)):
                for jspec in range(len(Training_Concentrations[0, :])):
                    Training_File.write(" {:>12.9f} ".format(Training_Concentrations[istr, jspec]))
                # end for-jspec
                Training_File.write(" {:>12.9f} {:>12.9f} {:>12.9f} {:>12.9f} {:.3f} {}\n".format(
                                    Training_energies_atom[istr],
                                    pred_energies_atom[istr],
                                    pred_CSE_latt[istr]/natom_latt,
                                    Training_prederr[istr],
                                    Training_Weights[istr],
                                    Training_Folders[istr]))
            # end for-istr
            Training_File.close()

            Params_File = open("MSCE-Alpha-{:.6E}-Params_WT-{}.out".format(CEOptions['alpha'], count), "w+")
            for I in range(len(AttenCoeffs)): Params_File.write(" {:+15.12E} \n".format(AttenCoeffs[I]))
            for J in range(len(ECI)): Params_File.write(" {:+15.12E} \n".format(ECI[J]))
            Params_File.close()

            Atten_File = open("MSCE-Alpha-{:.6E}-Atten_WT-{}.out".format(CEOptions['alpha'], count), "w+")
            for I in range(len(AttenCoeffs)): Atten_File.write(" {:+15.12E} \n".format(AttenCoeffs[I]))
            Atten_File.close()

            ECI_File = open("MSCE-Alpha-{:.6E}-ECI_WT-{}.out".format(CEOptions['alpha'], count), "w+")
            for J in range(len(ECI)): ECI_File.write(" {:+15.12E} \n".format(ECI[J]))
            ECI_File.close()

            # update initial fitting parameters
            InitialFittingParams = np.zeros([len(AttenCoeffs) + len(ECI)])
            InitialFittingParams[:len(AttenCoeffs)] = AttenCoeffs[:]
            InitialFittingParams[len(AttenCoeffs):] = ECI[:]
            CEOptions["InitialFittingParams"] = InitialFittingParams

            if len(Testing_Folders) > 0:

                pred_CSE_latt = calculate_structural_CSE_StructDictList(AttenCoeffs, Testing_Folders, natom_latt, Testing_StructDictList, logfile)
                pred_energies_latt = np.matmul(Testing_CorrMatr, ECI)
                pred_energies_atom = (pred_energies_latt + pred_CSE_latt)/natom_latt
                MSE = np.sum((pred_energies_atom - Testing_energies_atom)**2)/len(Testing_energies_atom)
                RMSE = np.sqrt(MSE)
                logfile.write('  RMSE of Testing: {:+.9f}\n'.format(RMSE))

                Testing_File = open("MSCE-Alpha-{:.6E}-Testing_WT-{}.out".format(CEOptions['alpha'], count), "w+")
                for istr in range(len(Testing_Folders)):
                    for jspec in range(len(Testing_Concentrations[0, :])):
                        Testing_File.write(" {:>12.9f} ".format(Testing_Concentrations[istr, jspec]))
                    # end for-jspec
                    Testing_File.write(" {:>12.9f} {:>12.9f} {:>12.9f} {:>12.9f} {}\n".format(Testing_energies_atom[istr],
                              pred_energies_atom[istr], pred_CSE_latt[istr]/natom_latt,
                              pred_energies_atom[istr] - Testing_energies_atom[istr], Testing_Folders[istr]))
                # end for-istr
                Testing_File.close()
            # end if

            # if CEOptions['CALC-CV'] == 'TRUE':
            #     LOOCV = calc_LOOCV(Training_CorrMatr, Training_energies_latt, ClusterCollection,
            #                        ParentLattInfo, CEOptions, CEOptions['K-Fold'], logfile)
            #     logfile.write("  CV = {:+.9f}\n".format(LOOCV))
            # # end if

            if CEOptions['AdjustWeights'] == 'TRUE':
                IndexMaxErr = []
                for K in range(len(Training_prederr)):
                    if np.abs(Training_prederr[K]) > CEOptions['MaxError']:
                        IndexMaxErr.append(K)
                        Training_Weights[K] += 2.0*np.floor(np.abs(Training_prederr[K])/CEOptions['MaxError'])
                        flag_continue = True
                        logfile.write("count, error, structure = {:>3d} {:>12.9f} {}\n".format(count, Training_prederr[K], Training_Folders[K]))

                    elif (Training_Folders[K] in CEOptions['GroundStateStr']) and np.abs(Training_prederr[K]) > CEOptions['GSError']:
                        Training_Weights[K] += 2.0*np.floor(np.abs(Training_prederr[K])/CEOptions['GSError'])
                        flag_continue = True
                        logfile.write("count, error, structure = {:>3d} {:>12.9f} {}\n".format(count, Training_prederr[K], Training_Folders[K]))

                    # end if
                # end for-K
                CEOptions['Training_Weights'] = Training_Weights
            # end if

            MaxError = np.amax(np.abs(Training_prederr))
            count += 1

        # end while

        if CEOptions['CALC-CV'] == 'TRUE':
            LOOCV = calc_LOOCV(Training_CorrMatr, Training_energies_latt, ClusterCollection,
                               ParentLattInfo, CEOptions, CEOptions['K-Fold'], logfile)
            logfile.write("  CV = {:+.9f}\n".format(LOOCV))
            logfile.flush()
        # end if

    else:
        raise ValueError('Method {} with loss function {} is not implemented yet.'.format(CEOptions['Optimizer'], CEOptions['LossFunction']))
    # end of if

    return 0

def calc_fitting_error_FitAll(FittingParams, CorrMatr, ClusterCollection, energies, CEOptions, logfile):

    Training_Folders = CEOptions['Training_Folders']
    directions = CEOptions['Directions']
    cse = CEOptions['CSECurves']
    ParentLattInfo = CEOptions['ParentLattInfo']
    natom_latt = ParentLattInfo['sitenum']

    Training_Weights = CEOptions['Training_Weights']

    NumAtten = CEOptions['NumAtten']
    atten_coeffs = FittingParams[:NumAtten]
    ECI = FittingParams[NumAtten:]

    Training_StructDictList = CEOptions['Training_StructDictList']
    natom_latt = ParentLattInfo['sitenum']
    pred_CSE = calculate_structural_CSE_StructDictList(atten_coeffs, Training_Folders, natom_latt, Training_StructDictList, logfile)

    lam = CEOptions['lambda']

    if CEOptions['LossFunction'] == 'FITALL-CSE-AllPairs':

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

        L = np.sum(Training_Weights*(pred_energies + pred_CSE - energies)**2)/len(energies) + CEOptions['alpha']*M/NF

        logfile.write("LossFunction = {:12.9E} \n".format(L))
        logfile.flush()

        return L

    elif CEOptions['LossFunction'] == 'FITALL-CSE-SmoothAllClusters':

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

        L = np.sum(Training_Weights*(pred_energies + pred_CSE - energies)**2)/len(energies) + CEOptions['alpha']*M/NF

        logfile.write("LossFunction = {:12.9E} \n".format(L))
        logfile.flush()

        return L

    elif CEOptions['LossFunction'] == 'FITALL-CSE-L1':

        pred_energies = np.matmul(CorrMatr, ECI)

        ClusterInfo = ClusterCollection['ClusterInfo']
        NumClusters = ClusterCollection['NumClusters']

        M = 0.0
        NF = 0.0
        for icluster in range(np.sum(NumClusters)):
            rr = ClusterInfo[icluster, 1]
            M += (rr**lam)*ClusterInfo[icluster, 0]*np.abs(ECI[icluster])
            NF += (rr**lam)*ClusterInfo[icluster, 0]
        # end of for-icluster

        L = np.sum(Training_Weights*(pred_energies + pred_CSE - energies)**2)/len(energies) + CEOptions['alpha']*M/NF

        logfile.write("LossFunction = {:12.9E} \n".format(L))
        logfile.flush()

        return L

    else:
        raise ValueError('Scheme {} is not implemented yet.'.format(LossFunc))

def calc_LOOCV(CorrMatr, energies, ClusterCollection, ParentLattInfo, CEOptions, nfold, logfile):

    options = CEOptions["MinOptions"]
    All_Training_Folders = np.array(CEOptions['Training_Folders'])
    All_Training_StructDictList = CEOptions['Training_StructDictList']

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
        LOO_Folders = np.delete(All_Training_Folders, indices, 0)

        CEOptions['Training_Weights'] = LOO_weights
        CEOptions['Training_Folders'] = LOO_Folders

        if CEOptions['Optimizer'] == 'PINV' and CEOptions['LossFunction'] == 'MSE':
            LOO_ECI = np.matmul(LA.pinv(LOO_CorrMatr), LOO_energies)

        elif ((CEOptions['Optimizer'] in ['BFGS', 'L-BFGS-B'])
              and (CEOptions['LossFunction'] in ['FITALL-CSE-SmoothPairs', 'FITALL-CSE-SmoothAllClusters', 'FITALL-CSE-L1'])):

            print("LOO_CorrMatr, LOO_energies = ", np.shape(LOO_CorrMatr), np.shape(LOO_energies))

            options = CEOptions['MinOptions']
            FittingParams_guess = CEOptions["InitialFittingParams"]
            LOO_results = optimization.minimize(calc_fitting_error_FitAll,
                                                FittingParams_guess,
                                                args = (LOO_CorrMatr,
                                                        ClusterCollection,
                                                        LOO_energies,
                                                        CEOptions,
                                                        logfile),
                                                method = CEOptions['Optimizer'],
                                                options = options)
            logfile.write("  Minimization algorithm: {}; Successful? {}\n".format(CEOptions['Optimizer'], LOO_results.success))
            FittingParams = LOO_results.x
            Residual = LOO_results.fun
            logfile.write("  Residual Error = {:+15.12E}\n".format(Residual))

            NumAtten = CEOptions['NumAtten']
            LOO_AttenCoeffs = FittingParams[:NumAtten]
            LOO_ECI = FittingParams[NumAtten:]

        else:
            raise ValueError('CEOptions {} is not implemented yet!'.format(CEOptions))
        # end of if
        LOO_pred_energy = np.matmul(CorrMatr[indices, :], LOO_ECI)

        LOO_pred_CSE_latt = calculate_structural_CSE_StructDictList(LOO_AttenCoeffs, All_Training_Folders[indices], natom_latt, All_Training_StructDictList, logfile)
        LOO_pred_error[indices] = (energies[indices] - (LOO_pred_energy + LOO_pred_CSE_latt))/natom_latt

        # logfile.write("  dataset: {} {:12.6f}\n".format(K, np.mean(np.abs(LOO_pred_error[indices]))))

        for ipredstr in range(len(indices)):
            logfile.write(" dataset {}: {:>12.9f} {:>12.9f} {:>12.9f} {}\n".format(K,
                          energies[indices[ipredstr]]/natom_latt,
                          (LOO_pred_energy[ipredstr] + LOO_pred_CSE_latt[ipredstr])/natom_latt,
                          LOO_pred_error[indices[ipredstr]],
                          All_Training_Folders[indices[ipredstr]]))
        # end for-ipredstr
        logfile.flush()

    # end of for-K

    cvfile = open("MSCE_DATASET_CV.OUT", "w+")
    for ii in range(len(All_Training_Folders)):
        cvfile.write("{:>12.9f} {}\n".format(LOO_pred_error[ii], All_Training_Folders[ii]))
    cvfile.close()

    LOOCV = np.sqrt(np.sum(LOO_pred_error**2)/len(LOO_pred_error))

    return LOOCV

# -----------------------------------------------------------------------------

