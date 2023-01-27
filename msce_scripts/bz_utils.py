
import numpy as np
import numpy.linalg as LA

import sys
import os

from datetime import datetime

import scipy.optimize as optimization

from atat_io import read_atat_str
from energy_utils import calc_X
from structure_utils import which_atom

from cse_utils import interpolate_CSE_conc_nolog
from cse_utils import calc_fitting_err
from cse_utils import get_kspace_ECI

from hex_harmonics import calc_hex_surf_value

# -----------------------------------------------------------------------------

def get_KS_ECI(folders, cse, directions, ParentLattInfo):

    ParentLattVec = np.matmul(ParentLattInfo['lattvec'], ParentLattInfo['axes'])
    natom_latt = ParentLattInfo['sitenum']

    cwd = os.getcwd()
    for K in range(len(folders)):
        os.chdir(folders[K])
        print(os.getcwd())

        kpoints = np.loadtxt('./kpoints_atat.out', float)
        kpoints = kpoints.reshape(-1, 3)

        StructInfo = read_atat_str('str.out')
        StructAtomConc = calc_X(StructInfo, reference_file = '../elemental_reference.out')

        conc_x = StructAtomConc[1]
        conc_q = 1.0 - 2.0*conc_x

        coeffs_opt = np.zeros([7])
        cse_conc = interpolate_CSE_conc_nolog(conc_x, directions, cse)
        minialgo = 'Powell'
        syminfo = ParentLattInfo['syminfo']
        opt_results = optimization.minimize(calc_fitting_err, coeffs_opt,
                                            args = (directions, cse_conc, syminfo, ParentLattVec), method = minialgo)
        coeffs_opt = opt_results.x

        keci_list = []
        for ikpt in range(len(kpoints[:, 0])):
            keci = get_kspace_ECI(coeffs_opt, kpoints[ikpt, :], conc_x)
            keci_list.append(keci)
        # end for-ikpt
        np.savetxt("KECI.OUT", np.array(keci_list), fmt = "%16.9E")

        os.chdir(cwd)
    # end for-K

    return 0

def calculate_structural_factors(ParentLattInfo, folders, elements):

    ParentLattVec = np.matmul(ParentLattInfo['lattvec'], ParentLattInfo['axes'])
    ParentAtomPos = np.matmul(ParentLattInfo['sitepos'], ParentLattInfo['axes'])

    ParentLattVec_KS = np.transpose(LA.inv(ParentLattVec))
    natom_latt = len(ParentAtomPos[:, 0])

    for f in folders:
        os.chdir(f)
        cwd = os.getcwd()
        print('  Current working directory: ' + cwd)

        StructureInfo = read_atat_str('str.out')
        StructureLattVec = np.matmul(StructureInfo['lattvec'], StructureInfo['axes'])
        StructureAtomPos = np.matmul(StructureInfo['atompos'], StructureInfo['axes'])
        vol = int(round(LA.det(StructureLattVec)/LA.det(ParentLattVec)))

        # kpoints = get_1stBZ_LatticePoint(ParentLattVec, StructureLattVec)
        kpoints = np.loadtxt('kpoints_atat.out', float, usecols = (0, 1, 2))
        kpoints = np.reshape(kpoints, (-1, 3))

        atom_list = StructureInfo['atomlist']
        atom_spins = []
        for atom in atom_list:
            if atom == elements[0]:
                atom_spins.append(-1)
            elif atom == elements[1]:
                atom_spins.append(1)
            else:
                raise ValueError('atomic label {} is not recognizable!'.format(atom))
            # end of if
        # end of for-atom
        # np.savetxt('atom_spins.out', np.asarray(atom_spins), fmt = '%+d')

        struct_fact = np.zeros([len(kpoints[:, 0]), natom_latt], dtype = complex)
        for IK in range(len(kpoints[:, 0])):
            for JA in range(len(StructureAtomPos[:, 0])):

                # find index of atom within unit cell of lattice;
                w_atom = which_atom(np.reshape(ParentAtomPos, (-1, 3)),
                                    np.reshape(StructureAtomPos[JA, :], (-1, 3)),
                                    ParentLattVec_KS)
                # print("w_atom = ", w_atom)

                if w_atom != -1: # skip spectator atoms;
                    temp_atompos = StructureAtomPos[JA, :] - ParentAtomPos[w_atom, :]
                    phase_shift = 2.0*np.pi*np.inner(kpoints[IK, :], temp_atompos)
                    struct_fact[IK, w_atom] += atom_spins[JA]*complex(np.cos(phase_shift), np.sin(phase_shift))
                # end of if
            # end of for-JA

            # normalize so that structure factor is independent of cell size;
            for JW in range(natom_latt):
                struct_fact[IK, JW] = struct_fact[IK, JW]/float(vol)
            # end of for-JW
        # end of for-IK

        SF_file = open('struct_fact.out', "w+")
        for isf in range(len(struct_fact[:, 0])):
            for jsf in range(len(struct_fact[0, :])):
                temp_SF = struct_fact[isf, jsf]
                SF_file.write(" {:18.12f} {:18.12f} ".format(temp_SF.real, temp_SF.imag))
            # end for-isf
            SF_file.write("\n")
        # end for-jsf
        SF_file.close()

        os.chdir('..')
    # end of for-K

    return 0

def get_StructDictList(folders, cse, directions, ParentLattInfo, logfile):

    start_time = datetime.now()

    ParentLattVec = np.matmul(ParentLattInfo['lattvec'], ParentLattInfo['axes'])
    natom_latt = ParentLattInfo['sitenum']

    StructDictList = {}
    cwd = os.getcwd()
    for K in range(len(folders)):
        StructDict = {}
        os.chdir(folders[K])

        # start_time = datetime.now()

        kpoints = np.loadtxt('./kpoints_atat.out', float)
        kpoints = kpoints.reshape(-1, 3)
        StructDict['KPOINTS'] = kpoints

        StructInfo = read_atat_str('str.out')
        StructLattVec = np.matmul(StructInfo['lattvec'], StructInfo['axes'])
        StructAtomPos = np.matmul(StructInfo['atompos'], StructInfo['axes'])
        StructAtomList = StructInfo['atomlist']
        StructAtomConc = calc_X(StructInfo, reference_file = '../elemental_reference.out')

        conc_x = StructAtomConc[1]
        conc_q = 1.0 - 2.0*conc_x

        # StructAtomSpins = np.loadtxt('atom_spins.out', float)

        struct_fact = np.zeros([len(kpoints[:, 0]), natom_latt], dtype = complex)
        struct_fact_array = np.loadtxt('struct_fact.out', float)
        struct_fact_array = struct_fact_array.reshape(-1, 2*natom_latt)
        for isf in range(len(kpoints[:, 0])):
            for jsf in range(natom_latt):
                temp_real = struct_fact_array[isf, jsf*natom_latt]
                temp_imag = struct_fact_array[isf, jsf*natom_latt + 1]
                struct_fact[isf, jsf] = complex(temp_real, temp_imag)
            # end for-jsf
        # end for-isf
        StructDict['STRUCT_FACT'] = struct_fact

        if not os.path.exists("KECI.OUT"):
            coeffs_opt = np.zeros([7])
            cse_conc = interpolate_CSE_conc(conc_x, directions, cse, logfile)
            minialgo = 'Powell'
            syminfo = ParentLattInfo['syminfo']
            opt_results = optimization.minimize(calc_fitting_err, coeffs_opt,
                                                args = (directions, cse_conc, syminfo, ParentLattVec), method = minialgo)
            coeffs_opt = opt_results.x

            keci_list = []
            for ikpt in range(len(kpoints[:, 0])):
                keci = get_kspace_ECI(coeffs_opt, kpoints[ikpt, :], conc_x)
                keci_list.append(keci)
            # end for-ikpt

            np.savetxt("KECI.OUT", np.array(keci_list), fmt = "%16.9E")

        # end if
        KECI = np.loadtxt("KECI.OUT", float)
        StructDict['KECI_LIST'] = np.reshape(KECI, (-1,))

        StructDictList[folders[K]] = StructDict

        time_elapsed = datetime.now() - start_time
        logfile.write('  Time elapsed in get_StructDictList (hh:mm:ss) {} {}\n'.format(time_elapsed, folders[K]))
        logfile.flush()

        os.chdir(cwd)
    # end for-K

    return StructDictList

def calculate_structural_CSE_StructDictList(atten_coeffs, folders, natom_latt, StructDictList, logfile):

    # start_time = datetime.now()

    CSE_List = np.zeros([len(folders)])
    cwd = os.getcwd()
    for K in range(len(folders)):

        StructDict = StructDictList[folders[K]]
        kpoints = StructDict['KPOINTS']
        keci_list = StructDict['KECI_LIST']
        struct_fact = StructDict['STRUCT_FACT']

        energy_CSE = 0.0
        for ikpt in range(len(kpoints[:, 0])):

            if LA.norm(kpoints[ikpt, :]) > 1E-3:
                kc = calc_hex_surf_value(atten_coeffs, kpoints[ikpt, :])
                atten_factor = np.exp(-1.0*((LA.norm(kpoints[ikpt, :]))**2.0)/(kc**2.0))
            else:
                atten_factor = 1.0
            # end if

            for jatom in range(natom_latt):
                temp_dCSE = keci_list[ikpt]*atten_factor*np.conj(struct_fact[ikpt, jatom])*struct_fact[ikpt, jatom]
                energy_CSE += temp_dCSE.real
            # end for-jatom
        # end for-ikpt

        CSE_List[K] = energy_CSE

        os.chdir(cwd)
    # end for-K

    # time_elapsed = datetime.now() - start_time

    return CSE_List

# -----------------------------------------------------------------------------

