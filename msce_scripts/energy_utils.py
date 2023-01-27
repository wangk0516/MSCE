
import numpy as np
import sys
import os

MAX_FLOAT = sys.float_info.max

from vasp_io import read_vasp_str
from atat_io import read_atat_str

# -----------------------------------------------------------------------------

def calc_EX(crysinfo, energy_file, reference_file = '../elemental_reference.out'):

    all_atomic_labels = np.loadtxt(reference_file, str, usecols = (0,))
    all_atomic_energy = np.loadtxt(reference_file, float, usecols = (1,))
    # print('all_atomic_labels = ', all_atomic_labels)
    # print('all_atomic_energy = ', all_atomic_energy)

    atomlabel = crysinfo['atomlabel']
    atomnum = crysinfo['atomnum']
    # print('atomlabel = ', atomlabel)
    # print('atomnum = ', atomnum)

    all_conc = np.zeros([len(all_atomic_labels)])
    all_atomnum = np.zeros([len(all_atomic_labels)])
    for I in range(len(all_atomic_labels)):
        for J in range(len(atomlabel)):
            if all_atomic_labels[I] == atomlabel[J]:
                all_atomnum[I] = atomnum[J]
                break
            # end of if
        # end of for-I
    # end of for-K
    # print('all_atomnum = ', all_atomnum)

    all_conc = all_atomnum/np.float(np.sum(all_atomnum))

    energy_DFT = np.loadtxt(energy_file, float)

    dEf = energy_DFT/np.sum(all_atomnum) - np.sum(all_conc*all_atomic_energy)

    return all_conc, dEf

def calc_X(crysinfo, reference_file = '../elemental_reference.out'):

    all_atomic_labels = np.loadtxt(reference_file, str, usecols = (0,))
    atomlabel = crysinfo['atomlabel']
    atomnum = crysinfo['atomnum']

    all_atomnum = np.zeros([len(all_atomic_labels)])
    for I in range(len(all_atomic_labels)):
        for J in range(len(atomlabel)):
            if all_atomic_labels[I] == atomlabel[J]:
                all_atomnum[I] = atomnum[J]
                break
            # end of if
        # end of for-I
    # end of for-K
    all_conc = all_atomnum/np.float(np.sum(all_atomnum))

    return all_conc

def get_formation_energies(strnames, energy_file, reference_file = './elemental_reference.out'):

    all_atomic_labels = np.loadtxt(reference_file, str, usecols = (0,))
    concentrations = MAX_FLOAT*np.ones([len(strnames), len(all_atomic_labels)])
    energies = MAX_FLOAT*np.ones([len(strnames)])
    for K in range(len(strnames)):
        os.chdir(strnames[K])
        # print('  ' + os.getcwd())
        struct_info = read_atat_str('str.out')
        conc, dEf = calc_EX(struct_info, energy_file)
        # dE_cs = np.loadtxt('energy_CSE.out', float)
        # dE_chem = dEf - dE_cs
        # print(conc, dEf)
        concentrations[K, :] = conc[:]
        energies[K] = dEf
        os.chdir('..')
    # end of for-K

    return concentrations, energies

def get_compositions(strnames, reference_file = './elemental_reference.out'):

    all_atomic_labels = np.loadtxt(reference_file, str, usecols = (0,))
    concentrations = MAX_FLOAT*np.ones([len(strnames), len(all_atomic_labels)])
    for K in range(len(strnames)):
        os.chdir(strnames[K])
        struct_info = read_vasp_str('CONTCAR.static')
        conc = calc_X(struct_info)
        concentrations[K, :] = conc[:]
        os.chdir('..')
    # end of for-K

    return concentrations

# -----------------------------------------------------------------------------

