import logging
import numpy as np


def dist(xyzs, i, j, derivative=False):
    rij = xyzs[i, :] - xyzs[j, :]
    r = np.linalg.norm(rij)
    if derivative:
        dr = np.zeros(len(xyzs)*3)
        dr[3*i:3*(i+1)] = rij/r
        dr[3*j:3*(j+1)] = -rij/r
        return r, dr
    else:
        return r


def angle(xyzs, i, j, k, derivative=False):
    """atom j is center of angle
    Follows:
    https://www.cs.utexas.edu/users/evouga/uploads/4/5/6/8/45689883/turning.pdf
    """
    rji = xyzs[i, :] - xyzs[j, :]
    rjk = xyzs[k, :] - xyzs[j, :]
    norm_rji = np.linalg.norm(rji)
    norm_rjk = np.linalg.norm(rjk)
    cross = np.cross(rji, rjk)
    z = cross/np.linalg.norm(cross)
    t = 2 * np.arctan2(
        np.dot(cross, z),
        norm_rji*norm_rjk + np.dot(rji, rjk))
    if derivative:
        dt = np.zeros(len(xyzs)*3)
        dt[3*i:3*(i+1)] = np.cross(rji, z)/norm_rji**2
        dt[3*k:3*(k+1)] = - np.cross(rjk, z)/norm_rjk**2
        dt[3*j:3*(j+1)] = - dt[3*i:3*(i+1)] - dt[3*k:3*(k+1)]
        return t, dt
    else:
        return t


def dihedral(xyzs, i, j, k, l, derivative=False):
    """Follows:
    Blondel, A. and Karplus, M., J. Comput. Chem., 17: 1132-1141. (1996)
    """
    F = xyzs[i, :] - xyzs[j, :]
    G = xyzs[j, :] - xyzs[k, :]
    H = xyzs[l, :] - xyzs[k, :]
    A = np.cross(F, G)
    B = np.cross(H, G)
    norm_G = np.linalg.norm(G)
    w = np.arctan2(
            np.dot(np.cross(B, A), G/norm_G),
            np.dot(A, B))
    if derivative:
        A_sq = np.dot(A, A)
        B_sq = np.dot(B, B)
        dw = np.zeros(len(xyzs)*3)
        dw[3*i:3*(i+1)] = - norm_G/A_sq*A
        dw[3*j:3*(j+1)] = (norm_G/A_sq*A
                           + np.dot(F, G)/(A_sq*norm_G)*A
                           - np.dot(H, G)/(B_sq*norm_G)*B)
        dw[3*k:3*(k+1)] = (np.dot(H, G)/(B_sq*norm_G)*B
                           - np.dot(F, G)/(A_sq*norm_G)*A
                           - norm_G/B_sq*B)
        dw[3*l:3*(l+1)] = norm_G/B_sq*B
        return w, dw
    else:
        return w


def find_angles_and_dihedrals(bonds):
    angles = []
    dihedrals = []
    for n, b1 in enumerate(bonds):
        for m, b2 in enumerate(bonds[n+1:]):
            i, j, k = None, None, None
            if b1[0] == b2[0]:
                i, j, k = b1[1], b1[0], b2[1]
            elif b1[1] == b2[0]:
                i, j, k = b1[0], b1[1], b2[1]
            elif b1[0] == b2[1]:
                i, j, k = b1[1], b1[0], b2[0]
            elif b1[1] == b2[1]:
                i, j, k = b1[0], b1[1], b2[0]
            if i is not None:
                angles.append((i, j, k))
                # Loop over all bonds to detect circular geometries
                for b3 in bonds[n+1:]:
                    if j not in b3:  # Ignore impropers
                        if (i == b3[0] and k != b3[1]
                                and not (k, j, i, b3[1]) in dihedrals):
                            dihedrals.append((b3[1], i, j, k))
                        elif (i == b3[1] and k != b3[0]
                                and not (k, j, i, b3[0]) in dihedrals):
                            dihedrals.append((b3[0], i, j, k))
                        elif (k == b3[0] and i != b3[1]
                                and not (b3[1], k, j, i) in dihedrals):
                            dihedrals.append((i, j, k, b3[1]))
                        elif (k == b3[1] and i != b3[0]
                                and not (b3[0], k, j, i) in dihedrals):
                            dihedrals.append((i, j, k, b3[0]))
    # for b1, b2, b3 in combinations(bonds, 3):
    #     if (len(set(b1 + b2 + b3)) == 4
    #             and len(set(b1) & set(b2) & set(b3))) == 0:
    #         print(b1, b2, b3)
    return angles, dihedrals


def to_primitives_factory(bonds):
    angles, dihedrals = find_angles_and_dihedrals(bonds)
    logging.info('Found %d angles and %d dihedrals' % (len(angles),
                                                       len(dihedrals)))
    n_q = len(bonds) + len(angles) + len(dihedrals)

    def to_primitives(atoms):
        xyzs = atoms.get_positions()
        qs = np.zeros(n_q)
        dqs = np.zeros((n_q, len(xyzs)*3))
        for i, b in enumerate(bonds):
            qs[i], dqs[i, :] = dist(xyzs, b[0], b[1], derivative=True)
        for i, a in enumerate(angles):
            j = len(bonds) + i
            qs[j], dqs[j, :] = angle(xyzs, a[0], a[1], a[2], derivative=True)
        for i, d in enumerate(dihedrals):
            j = len(bonds) + len(angles) + i
            qs[j], dqs[j, :] = dihedral(xyzs, d[0], d[1], d[2], d[3],
                                        derivative=True)
        return qs, dqs
    return to_primitives, angles, dihedrals
