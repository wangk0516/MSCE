
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import os
import sys

###############################################################################

def integer_vector(vector):

    temparg = np.sort(np.abs(vector))
    for I in range(3):
        if temparg[I] > 0:
            vector = vector/temparg[I]
            break
        # end of if
    # end of for-I

    return vector

def find_perp_vectors(G_epi):

    G_epi = G_epi/LA.norm(G_epi)
    
    perp_vec_1 = np.zeros([3])
    for ax in range(3):
        tempvec = np.zeros([3])
        tempvec[ax] = 1.0
        perp_vec_1 = tempvec - G_epi*np.inner(G_epi, tempvec)/np.inner(G_epi, G_epi)
        if LA.norm(perp_vec_1) > 1E-4:
            break
        # end of if
    # end of for-axis
    perp_vec_1 = perp_vec_1/LA.norm(perp_vec_1)

    perp_vec_2 = np.cross(G_epi, perp_vec_1)

    # perp_vec_1 = integer_vector(perp_vec_1)
    # perp_vec_2 = integer_vector(perp_vec_2)

    return perp_vec_1, perp_vec_2

def calc_basis(G_epi, perp_vec_1, perp_vec_2):
    
    roto_dir = G_epi
    basis = np.zeros([3, 3])
    basis[:, 0] = roto_dir/LA.norm(roto_dir)
    basis[:, 1] = perp_vec_1/LA.norm(perp_vec_1)
    basis[:, 2] = perp_vec_2/LA.norm(perp_vec_2)
    
    return basis

def calc_angles(k):
    
    phi = np.arctan(k[1]/k[0])
    theta = np.arccos(k[2]/LA.norm(k))

    print('phi = ', phi*180/np.pi, ' deg')
    print('theta = ', theta*180/np.pi, 'deg')
    
    return theta, phi

def four2three_ind(G_epi_4ind, ratio):
    G_epi_4ind.astype(np.float)
    ratio = np.float(ratio)

    hh, kk, ii, ll = G_epi_4ind[0], G_epi_4ind[1], G_epi_4ind[2], G_epi_4ind[3]

    id3 = np.zeros([3, 1])
    trans = np.array([[1., -0.5, 0.],
                      [0., (3)**0.5/2, 0.],
                      [0., 0., ratio]])
 
    if ll == 0:
        method = 'type_1'
    elif hh == 0 and kk == 0 and ii == 0:
        method = 'type_1'
    else:
        method = 'type_2'
    # end of if

    # Given (hkil),find the plane normal [uvw] in hexagonal coordinate system
    if method == 'type_1':
        id3[0] = hh - ii
        id3[1] = kk - ii
        id3[2] = ll
    elif method == 'type_2':
        id3[0] = 1.0
        id3[1] = (hh + 2.0*kk)/(2.0*hh + kk)
        id3[2] = (3.0*ll/2.0)/(2.0*hh + kk)/(ratio**2)
    # end of if    

    # transform [uvw] in hexagonal to cartesian coordinate system
    cart3_matr = np.transpose(np.matmul(trans, id3))
    cart3 = cart3_matr[0, :]

    # cart3 = integer_vector(cart3)

    return cart3

def hex3_2_cart3(hex3, ratio):
    ratio = np.float(ratio)
    trans = np.array([[1., -0.5, 0.],
                      [0., (3)**0.5/2, 0.],
                      [0., 0., ratio]])

    #cart3_matr = np.transpose(np.matmul(trans, hex3))
    #cart3 = cart3_matr[0, :]
    cart3 = np.transpose(np.matmul(trans, hex3))
    print('cart3 = ', cart3)
    return cart3


def three2four_ind(perp_vec, ratio):
    perp_vec.astype(np.float)
    ratio = np.float(ratio)

    # print('Input perp_vec = ', perp_vec)
    # print('Input ratio = ', ratio)

    # print('start ----------------------------------------')    
    id3 = np.transpose(perp_vec)
    trans = np.array([[1., -0.5, 0.],
                      [0., (3)**0.5/2, 0.],
                      [0., 0., ratio]])

    # print('id3 = \n', id3)    
    # print('trans = \n', trans)
    # print('LA.inv(trans) = \n', LA.inv(trans))

    hex3_cart = np.transpose(np.matmul(LA.inv(trans), id3))

    # print('hex3_cart = \n', hex3_cart)

    # print('end ----------------------------------------')

    # hex3_cart = integer_vector(hex3_cart)

    uu, vv, ww = hex3_cart[0], hex3_cart[1], hex3_cart[2]

    hex3 = np.zeros([3])
    if (ww == 0) or (uu == 0 and vv == 0): 
        hex3[0] = (2.0*hex3_cart[0] - hex3_cart[1])/3.0 #This step is to change <UVW> to (hkil). For basal and prismatic plans, <uvtw> plan normal is (hkil).
        hex3[1] = (2.0*hex3_cart[1] - hex3_cart[0])/3.0
        hex3[2] = hex3_cart[2]
    else:
        hex3[0] = (2.0*uu - vv)*(1.0/2.0/ww/(ratio**2))
        hex3[1] = (2.0*vv - uu)*(1.0/2.0/ww/(ratio**2))
        hex3[2] = 1.0
    # end of if

    """ 
    temparg = np.sort(np.abs(hex3))
    for I in range(3):
        if temparg[I] > 0:
            hex3 = hex3/temparg[I]
            break 
    """

    # hex3 = integer_vector(hex3)

    return hex3

def read_vasp(fname):

    def RepresentsInt(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

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
        atomlist = lattice[4 + WITHLABELS].split()
    else:
        atomlist = []
    # end of if
    atomnumstr = lattice[5 + WITHLABELS].split()
    atomnum = []
    for LL in range(np.size(atomnumstr)):
        atomnum.append(int(atomnumstr[LL]))
    # end of for-LL
    atomnum = np.array(atomnum)

    # get simple atom labels for the case they are missing in POSCAR
    if atomlist == []:
        for CC in range(np.size(atomnum)): atomlist.append("C" + str(CC + 1))
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
            'lattvec': lattvec, 'atomlist': atomlist, \
            'atomnum': atomnum, 'coordtype': coordtype, \
            'atompos': atompos}
    # return the results
    return crysinfo

def write_vasp(crysinfo, posname):
    ###########################################################################################
    # get the necessary information from the dictionary `crysinfo`
    comment = crysinfo['comment']
    scale = crysinfo['scale']
    lattvec = crysinfo['lattvec']
    atomlist = crysinfo['atomlist']
    atomnum = crysinfo['atomnum']
    coordtype = crysinfo['coordtype']
    atompos = crysinfo['atompos']
    ###########################################################################################
    # write the vasp position file
    posfile = open(posname, "w+")
    # 1st line, the comment
    posfile.write("%s\n"%(comment))
    # 2nd line, the scale
    posfile.write("%5.8f\n"%(scale))
    # 3rd to 5th line, the lattice vectors
    posfile.write("%5.8f\t%5.8f\t%5.8f\n"%(lattvec[0, 0], lattvec[0, 1], lattvec[0, 2]))
    posfile.write("%5.8f\t%5.8f\t%5.8f\n"%(lattvec[1, 0], lattvec[1, 1], lattvec[1, 2]))
    posfile.write("%5.8f\t%5.8f\t%5.8f\n"%(lattvec[2, 0], lattvec[2, 1], lattvec[2, 2]))
    # 6th line, the list of atom labels
    for I in range(np.size(atomlist)):
        posfile.write("%s  "%(atomlist[I]))
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
        posfile.write("%5.8f\t%5.8f\t%5.8f\n"%(atompos[K, 0], atompos[K, 1], atompos[K, 2]))
    # end of for-K
    posfile.close()

###############################################################################

max_paral_stretch = [0.5, 1.75]
ngrids_perp_1 = 15
ngrids_perp_2 = 17
ngrids_paral = 19

directions = np.loadtxt('./dir_HCP.in', float)
directions = np.reshape(directions, (-1, 10))

crysinfo_AA = read_vasp('./0/CONTCAR.static')
crysinfo_BB = read_vasp('./1/CONTCAR.static')

lattvec_AA = crysinfo_AA['scale']*crysinfo_AA['lattvec']
lattvec_BB = crysinfo_BB['scale']*crysinfo_BB['lattvec']

alatt_AA, clatt_AA = lattvec_AA[0, 0], lattvec_AA[2, 2]  # [A];
alatt_BB, clatt_BB = lattvec_BB[0, 0], lattvec_BB[2, 2]  # [A];

ratio_AA = lattvec_AA[2, 2]/lattvec_AA[0, 0]
ratio_BB = lattvec_BB[2, 2]/lattvec_BB[0, 0]

for idir in range(len(directions[:, 0])):
    G_epi_4ind = directions[idir, :4]    # epitaxial direction, valid only for basal and primatic plane;
    dir_string = '_' + str(int(G_epi_4ind[0])) + '_' + str(int(G_epi_4ind[1])) + '_' \
                 + str(int(G_epi_4ind[2])) + '_' + str(int(G_epi_4ind[3]))
    print('Analyzing direction', G_epi_4ind, '... ')

    perp_vec_1_hex3 = directions[idir, 4:7]
    perp_vec_2_hex3 = directions[idir, 7:]
    
    perp_vec_1_cart3_AA = hex3_2_cart3(perp_vec_1_hex3, ratio_AA)
    perp_vec_2_cart3_AA = hex3_2_cart3(perp_vec_2_hex3, ratio_AA)
    paral_vec_cart3_AA = np.cross(perp_vec_1_cart3_AA, perp_vec_2_cart3_AA) 

    perp_vec_1_cart3_BB = hex3_2_cart3(perp_vec_1_hex3, ratio_BB)
    perp_vec_2_cart3_BB = hex3_2_cart3(perp_vec_2_hex3, ratio_BB)
    paral_vec_cart3_BB = np.cross(perp_vec_1_cart3_BB, perp_vec_2_cart3_BB)

    alatt_AA_perp_1 = alatt_AA*np.sqrt(perp_vec_1_cart3_AA[0]**2 + perp_vec_1_cart3_AA[1]**2 + perp_vec_1_cart3_AA[2]**2 ) 
    alatt_AA_perp_2 = alatt_AA*np.sqrt(perp_vec_2_cart3_AA[0]**2 + perp_vec_2_cart3_AA[1]**2 + perp_vec_2_cart3_AA[2]**2 )

    alatt_BB_perp_1 = alatt_BB*np.sqrt(perp_vec_1_cart3_BB[0]**2 + perp_vec_1_cart3_BB[1]**2 + perp_vec_1_cart3_BB[2]**2 )
    alatt_BB_perp_2 = alatt_BB*np.sqrt(perp_vec_2_cart3_BB[0]**2 + perp_vec_2_cart3_BB[1]**2 + perp_vec_2_cart3_BB[2]**2 )

    a_epi_perp_1 = np.linspace(alatt_AA_perp_1, alatt_BB_perp_1, ngrids_perp_1)
    a_epi_perp_2 = np.linspace(alatt_AA_perp_2, alatt_BB_perp_2, ngrids_perp_2)
    paral_stretch = np.linspace(max_paral_stretch[0], max_paral_stretch[1], ngrids_paral)

    # calculate basis correspoind to perp1 and perp2
    basis_AA =  calc_basis(paral_vec_cart3_AA, perp_vec_1_cart3_AA, perp_vec_2_cart3_AA)
    basis_BB =  calc_basis(paral_vec_cart3_BB, perp_vec_1_cart3_BB, perp_vec_2_cart3_BB)

    for M in range(2):
        os.chdir(str(M))

        if not os.path.exists("DIR_{}".format(dir_string)): os.mkdir("DIR_{}".format(dir_string))
        os.chdir("DIR_{}".format(dir_string))
        print(os.getcwd())

        for I in range(len(a_epi_perp_1)):
            stretch_AA_perp_1 = a_epi_perp_1[I]/alatt_AA_perp_1
            stretch_BB_perp_1 = a_epi_perp_1[I]/alatt_BB_perp_1

            for J in range(len(a_epi_perp_2)):
                stretch_AA_perp_2 = a_epi_perp_2[J]/alatt_AA_perp_2
                stretch_BB_perp_2 = a_epi_perp_2[J]/alatt_BB_perp_2

                stretch_name = "stretch_{}_{}".format(I, J)
                if not os.path.exists(stretch_name):
                    os.mkdir(stretch_name)
                    os.chdir(stretch_name)
                    print(stretch_name)

                    for K in range(len(paral_stretch)):

                        stretch = np.zeros([3, 3])
                        if M == 0:

                            stretch[0, 0] = paral_stretch[K]
                            stretch[1, 1] = stretch_AA_perp_1
                            stretch[2, 2] = stretch_AA_perp_2
                            lattvec = lattvec_AA
                            crysinfo = crysinfo_AA

                            trans_matrix = np.matmul(np.matmul(basis_AA, stretch), np.transpose(basis_AA))

                            new_lattvec_T = np.matmul(trans_matrix, np.transpose(lattvec))
                            new_lattvec = np.transpose(new_lattvec_T)

                            crysinfo['comment'] = "{}: {}_{}_{}".format(dir_string, I, J, K)
                            crysinfo['scale'] = 1.0
                            crysinfo['lattvec'] = new_lattvec

                        elif M == 1:

                            stretch[0, 0] = paral_stretch[K]
                            stretch[1, 1] = stretch_BB_perp_1
                            stretch[2, 2] = stretch_BB_perp_2
                            lattvec = lattvec_BB
                            crysinfo = crysinfo_BB

                            trans_matrix = np.matmul(np.matmul(basis_BB, stretch), np.transpose(basis_BB))

                            new_lattvec_T = np.matmul(trans_matrix, np.transpose(lattvec))
                            new_lattvec = np.transpose(new_lattvec_T)

                            crysinfo['comment'] = "{}: {}_{}_{}".format(dir_string, I, J, K)
                            crysinfo['scale'] = 1.0
                            crysinfo['lattvec'] = new_lattvec

                        # end of if
                        write_vasp(crysinfo, "POSCAR.{}".format(K))

                    # end of for-K
                    os.chdir('..')
                # end if

            # end of for-J
        # end of for-I
        os.chdir('../..')
    # end of for-M

#end of for-idir


