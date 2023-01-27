
import numpy as np

###############################################################################

def fmt_str(s, fmt_type = 'front', length = 15, app_str = ' '):

    if type(s) == int or type(s) == float or type(s) == np.float64:
        fmt_string = "{:.10f}".format(s)
    elif type(s) == str:
        fmt_string = s
    else:
        raise TypeError("Cannot recognize the type of input: ", type(s))
    # end of if
    
    if fmt_type == 'front':
        while len(fmt_string) < length:
            fmt_string = app_str + fmt_string
        # end of while
    elif fmt_type == 'back':
        while len(fmt_string) < length:
            fmt_string = fmt_string + app_str
        # end of while
    # end of if
    return fmt_string

def RepresentsInt(s):
	try:
		int(s)
		return True
	except ValueError:
		return False

def read_vasp_str(fname):

    # open the vasp lattice file
    VASPLATT = open(fname, "r")
    lattice = VASPLATT.readlines()
    VASPLATT.close
    # read the comment line and the scale
    comment = lattice[0]
    scale = lattice[1].split()
    scale = float(scale[0])
    # read the lattice vectors
    vxstr = lattice[2].split()
    vystr = lattice[3].split()
    vzstr = lattice[4].split()
    lattvec = np.zeros([3, 3])
    for II in range(3): lattvec[0, II] = float(vxstr[II])
    for JJ in range(3): lattvec[1, JJ] = float(vystr[JJ])
    for KK in range(3): lattvec[2, KK] = float(vzstr[KK])

    line4th = lattice[5].split()
    if RepresentsInt(line4th[0]) == False:
        WITHLABELS = 1
    elif RepresentsInt(line4th[0]) == True:
        WITHLABELS = 0
    # end of if

    # cases are different with and without atomic labels
    if WITHLABELS == 1:
        atomlabel = lattice[4 + WITHLABELS].split()
    else:
        atomlabel = []
    # end of if
    atomnumstr = lattice[5 + WITHLABELS].split()
    atomnum = []
    for LL in range(np.size(atomnumstr)):
        atomnum.append(int(atomnumstr[LL]))
    # end of for-LL
    atomnum = np.array(atomnum)

    # get simple atom labels for the case they are missing in POSCAR
    if atomlabel == []:
        for CC in range(np.size(atomnum)): atomlabel.append("C" + str(CC + 1))
    # end of if

    # read the type of coordinates
    coordtype = lattice[6 + WITHLABELS].split()
    # read the positions of the atoms
    ntot = np.sum(atomnum)
    atompos = np.zeros([ntot, 3])
    for MM in range(ntot):
        linestr = lattice[MM + 7 + WITHLABELS].split()
        for NN in range(3):
            atompos[MM, NN] = float(linestr[NN])
        # end of for-NN
    # end of for-MM
    # define the dictionary for crystal infomation
    crysinfo = {'comment': comment, 'scale': scale, \
            'lattvec': lattvec, 'atomlabel': atomlabel, \
            'atomnum': atomnum, 'coordtype': coordtype, \
            'atompos': atompos}
    # return the results
    return crysinfo

def write_vasp_str(crysinfo, posname):
    ###########################################################################################
    # get the necessary information from the dictionary `crysinfo`
    comment = crysinfo['comment']
    scale = crysinfo['scale']
    lattvec = crysinfo['lattvec']
    atomlabel = crysinfo['atomlabel']
    atomnum = crysinfo['atomnum']
    coordtype = crysinfo['coordtype']
    atompos = crysinfo['atompos']
    ###########################################################################################
    if 'atomlist' not in crysinfo:
        atomlist = []
        for ispec in range(len(atomnum)):
            for jatom in range(atomnum[ispec]):
                atomlist.append(atomlabel[ispec])
            # end for-jatom
        # end for-ispec
    else:
        atomlist = crysinfo['atomlist']
    # print('len(atomlist) = {}'.format(len(atomlist)))
    ###########################################################################################
    # write the vasp position file
    posfile = open(posname, "w+")
    # 1st line, the comment
    if '\n' in comment:
        posfile.write("%s"%(comment))
    else:
        posfile.write("%s\n"%(comment))
    # end of if
    # 2nd line, the scale
    posfile.write("%s\n"%(fmt_str(scale)))
    # 3rd to 5th line, the lattice vectors
    posfile.write("%s\t%s\t%s\n"%(fmt_str(lattvec[0, 0]), fmt_str(lattvec[0, 1]), fmt_str(lattvec[0, 2])))
    posfile.write("%s\t%s\t%s\n"%(fmt_str(lattvec[1, 0]), fmt_str(lattvec[1, 1]), fmt_str(lattvec[1, 2])))
    posfile.write("%s\t%s\t%s\n"%(fmt_str(lattvec[2, 0]), fmt_str(lattvec[2, 1]), fmt_str(lattvec[2, 2])))
    # 6th line, the list of atom labels
    for I in range(np.size(atomlabel)):
        posfile.write("%s  "%(atomlabel[I]))
    # end of for-I
    posfile.write("\n")
    # 7th line, the corresponding number of atoms
    for J in range(np.size(atomnum)):
        posfile.write("%d  "%(atomnum[J]))
    # end of for-J
    posfile.write("\n")
    # 8th line, type of coordinates
    posfile.write("%s\n"%(coordtype[0]))
    natom = int(np.sum(atomnum))
    # the rest, coordinates of the atoms
    for K in range(natom):
        posfile.write("%s\t%s\t%s\t%s\n"%(fmt_str(atompos[K, 0]), fmt_str(atompos[K, 1]), fmt_str(atompos[K, 2]), atomlist[K]))
    # end of for-K
    posfile.close()

###############################################################################

