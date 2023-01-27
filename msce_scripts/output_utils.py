

# -----------------------------------------------------------------------------

def write_output(CEOptions, CEInfo):

    folders = CEInfo['folders']
    CONC = CEInfo['CONC']

    ECI = CEInfo['ECI']
    ECI_File = open("CE_ECI_{}.out".format(CEOptions['LossFunction']), "w+")
    for I in range(len(ECI)):
        ECI_File.write(" {:+.9f}\n".format(ECI[I]))
    # end for-I
    ECI_File.close()

    energies = CEInfo['energies']
    energy_File = open("CE_ENERGIES_{}.out".format(CEOptions['LossFunction']), "w+")
    for J in range(len(energies[:, 0])):
        for ispec in range(len(CONC[0, :])):
            energy_File.write(" {:.9f} ".format(CONC[J, ispec]))
        # end for-ispec
        energy_File.write(" {:+.9f} {:+.9f} {:+.9f} {}\n".format(energies[J, 0], energies[J, 1],
                          energies[J, 0] - energies[J, 1], folders[J]))
    # end for-J
    energy_File.close()

    if len(energies[0, :]) == 3:
        RS_pair_energy_File = open("CE_ENERGIES_{}_RS_PAIR.out".format(CEOptions['LossFunction']), "w+")
        for J in range(len(energies[:, 0])):
            for ispec in range(len(CONC[0, :])):
                RS_pair_energy_File.write(" {:.9f} ".format(CONC[J, ispec]))
            # end for-ispec
            RS_pair_energy_File.write(" {:+.9f} {}\n".format(energies[J, 2], folders[J]))
        # end for-J
        RS_pair_energy_File.close()
    # end if

    if CEOptions['Prediction']:
        STRBNK_folders = CEInfo['STRBNK_folders']
        STRBNK_energies = CEInfo['STRBNK_energies']
        STRBNK_CONC = CEInfo['STRBNK_CONC']
        STRBNK_File = open(CEOptions['StructureBank'] + "/CE_Prediction_{}.out".format(CEOptions['LossFunction']), "w+")
        for K in range(len(STRBNK_folders)):
            for jspec in range(len(STRBNK_CONC[0, :])):
                STRBNK_File.write(" {:.9f} ".format(STRBNK_CONC[K, jspec]))
            # end for-jspec
            STRBNK_File.write(" {:+.9f} {}\n".format(STRBNK_energies[K], STRBNK_folders[K]))
        # end for-K
        STRBNK_File.close()
    # end if

    return 0

# -----------------------------------------------------------------------------

