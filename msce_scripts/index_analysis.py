
import numpy as np
import numpy.linalg as LA

# -----------------------------------------------------------------------------

def findPlaneNormal_Hex(planeIndex, lattvec):
    """
    Function:
        Read in the 3 Miller indices in hexagonal coordinates for plane index
        and find the normal vector in cartesion coordinates, i.e., <hh, kk, ll>  ==> <uu, vv, ww>
    Input:
        planeIndex ---- 3 Miller indices in hexagonal coordinates, alpha = 120, beta = 90 and gamma = 90 deg;
        lattvec ---- 3x3 matrix for lattice vectors with each ROW for x, y, and z respectively;
    Return:
        planeNormal ---- 3 Miller indices in cartesian coordinates for plane normal vector;
    """
    planeIndex.astype(np.float)

    # hh, kk, ll = planeIndex[0], planeIndex[1], planeIndex[2]
    aa = lattvec[0, 0]  # a of HCP lattice
    cc = lattvec[2, 2]  # c of HCP lattice

    if LA.norm(aa) < 1E-3 or LA.norm(cc) < 1E-3:
        raise ValueError('Lattice parameters to small: aa = {:18.12f}, cc = {:18.12f}'.format(aa, cc))

    ghex = np.array([[2.0, 1.0, 0.0],
                     [1.0, 2.0, 0.0],
                     [0.0, 0.0, (3.0*aa**2)/(2.0*cc**2)]])
    # ghex = (2.0/3.0/(aa**2))*ghex
    planeNormal = np.matmul(ghex, planeIndex)
    # print("planeNormal = ", planeNormal)

    # planeNormal = np.zeros([3])
    # planeNormal[0] = 2.0*hh + kk
    # planeNormal[1] = hh + 2.0*kk
    # planeNormal[2] = ll*(3.0*aa**2)/(2.0*cc**2)
    # print("planeNormal = ", planeNormal)

    if LA.norm(planeNormal) < 1E-4:
        print("planeIndex = ", planeIndex)
        print("lattvec = \n", lattvec)
        print("planeNormal and norm = ", planeNormal, LA.norm(planeNormal))
        raise ValueError("zero vector encountered!")
    else:
        return planeNormal/LA.norm(planeNormal)

def findPlaneIndex_Hex(planeNormal, lattvec):

    planeNormal.astype(np.float)

    uu, vv, ww = planeNormal[0], planeNormal[1], planeNormal[2]
    aa = lattvec[0, 0]  # a of HCP lattice
    cc = lattvec[2, 2]  # c of HCP lattice

    ghex = np.array([[2.0, 1.0, 0.0],
                     [1.0, 2.0, 0.0],
                     [0.0, 0.0, (3.0*aa**2)/(2.0*cc**2)]])
    planeIndex = np.matmul(LA.inv(ghex), planeNormal)
    # print("planeIndex = ", planeIndex)

    return planeIndex

def coordTransform(indices, lattvec, method):

    aa = lattvec[0, 0]  # a of HCP lattice
    cc = lattvec[2, 2]  # c of HCP lattice

    # coordinates in hcp primitive cell
    # transposed since row vectors are transformed to column vectors
    transMatr = np.array([[1.,   -0.5,         0.   ],
                          [0.,   np.sqrt(3)/2, 0.   ],
                          [0.,   0.,           cc/aa]])

    if method == "Hex2Cart":
        cart3 = np.transpose(np.matmul(transMatr, indices))
        if LA.norm(cart3) < 1E-4:
            raise ValueError("zero vector encountered!")
        else:
            ans = cart3/LA.norm(cart3)
        # end of if
        return ans
    elif method == "Cart2Hex":
        hex3 = np.transpose(np.matmul(LA.inv(transMatr), indices))
        if LA.norm(hex3) < 1E-4:
            raise ValueError("zero vector encountered!")
        else:
            ans = hex3/LA.norm(hex3)
        # end of if
        return ans
    else:
        raise ValueError("Cannot recognize method")
    # end of if

def find_perp_vectors(G_epi):

    G_epi = G_epi/LA.norm(G_epi)

    perp_vec_1 = np.zeros([3])
    for ax in range(3):
        tempvec = np.zeros([3])
        tempvec[ax] = 1.0
        perp_vec_1 = tempvec - G_epi*np.inner(G_epi, tempvec)/np.inner(G_epi, G_epi)
        if LA.norm(perp_vec_1) > 1E-4: break
    # end of for-axis
    perp_vec_1 = perp_vec_1/LA.norm(perp_vec_1)

    perp_vec_2 = np.cross(G_epi, perp_vec_1)

    return perp_vec_1, perp_vec_2

def get_Cart3_PlaneNormal(planeIndexHex3, lattvec):

    planeNormalHex3 = findPlaneNormal_Hex(planeIndexHex3, lattvec)
    PlaneNormalCart3 = coordTransform(planeNormalHex3, lattvec, "Hex2Cart")

    return PlaneNormalCart3

def get_Hex3_planeIndex(planeNormalCart3, lattvec):

    planeNormalHex3 = coordTransform(planeNormalCart3, lattvec, "Cart2Hex")
    planeIndexHex3 = findPlaneIndex_Hex(planeNormalHex3, lattvec)

    return planeIndexHex3

def distort_cell(lattvec, G_epi, perp_V1, perp_V2, stretch_p1, stretch_p2, stretch_epi):

    strain = np.zeros([3, 3])
    strain[0, 0] = stretch_epi
    strain[1, 1] = stretch_p1
    strain[2, 2] = stretch_p2

    basis = np.zeros([3, 3])
    basis[:, 0] = G_epi/LA.norm(G_epi)
    basis[:, 1] = perp_V1/LA.norm(perp_V1)
    basis[:, 2] = perp_V2/LA.norm(perp_V2)

    deformMatr = np.matmul(np.matmul(basis, strain), LA.inv(basis))
    lattvec_dfm = np.matmul(deformMatr, lattvec)

    return lattvec_dfm

# -----------------------------------------------------------------------------

