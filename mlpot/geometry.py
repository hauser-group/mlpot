import logging
import numpy as np
from ase.data import covalent_radii
from itertools import combinations, permutations


def distance(atoms1, atoms2, permute=False):
    if permute:
        raise NotImplementedError()

    def align(atoms):
        """ Returns numpy array of aligned positions"""
        xyzs = atoms.get_positions()
        # Shift by center of mass
        xyzs -= atoms.get_center_of_mass()

        # Find an atom i that does is not exactly centered
        for xi in xyzs:
            if np.linalg.norm(xi) > 1e-5:
                # Rotate atom i to align with the x-axis
                x_axis = np.array([1.0, 0.0, 0.0])
                # Cross product gives the rotation axis
                n = np.cross(xi, x_axis)
                norm = np.linalg.norm(n)
                n /= norm
                # Calculate the rotation angle
                alpha = np.arctan2(norm, np.dot(xi, x_axis))
                # Build the rotation matrix
                cross_mat = np.array([[0, -n[2], n[1]],
                                      [n[2], 0, -n[0]],
                                      [-n[1], n[0], 0]])
                R = ((1 - np.cos(alpha))*np.outer(n, n)
                     + np.eye(3)*np.cos(alpha) + np.sin(alpha)*cross_mat)
                # Apply rotation
                xyzs = xyzs.dot(R.T)
                break
        # Find an atom that is not aligned with the x-axis
        for xj in xyzs:
            if np.linalg.norm(xj) - np.abs(np.dot(xj, x_axis)) > 1e-5:
                # Rotate atom j into xy plane
                y = np.array([0.0, 1.0, 0.0])
                # Position vector of the atom j without x component
                r2 = np.array([0.0, xj[1], xj[2]])
                beta = np.arctan2(np.linalg.norm(np.cross(r2, y)),
                                  np.dot(r2, y))
                # Rotation matrix about the x-axis
                R = np.array([[1.0, 0.0, 0.0],
                              [0.0, np.cos(beta), np.sin(beta)],
                              [0.0, -np.sin(beta), np.cos(beta)]])
                # Apply rotation
                xyzs = xyzs.dot(R.T)
                break
        return xyzs

    return np.sqrt(np.mean((align(atoms1) - align(atoms2))**2))


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


def inv_dist(xyzs, i, j, derivative=False):
    rij = xyzs[i, :] - xyzs[j, :]
    q = 1.0/np.linalg.norm(rij)
    if derivative:
        dq = np.zeros(len(xyzs)*3)
        dq[3*i:3*(i+1)] = -rij * q**3
        dq[3*j:3*(j+1)] = rij * q**3
        return q, dq
    else:
        return q


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


def linear_bend(xyzs, i, j, k, derivative=False):
    """atom j is the center of angle
    Follows: 'Vibrational States' by S. Califano page 88.
    atom b -> atom i
    atom a -> atom j
    atom c -> atom k
    """
    rji = xyzs[i, :] - xyzs[j, :]
    rjk = xyzs[k, :] - xyzs[j, :]
    n = np.cross(rji, rjk)
    norm_rji = np.linalg.norm(rji)
    norm_rjk = np.linalg.norm(rjk)
    Ry = n[1]/(norm_rji*norm_rjk)
    Rx = -n[0]/(norm_rji*norm_rjk)
    if derivative:
        dRy = np.zeros(len(xyzs)*3)
        dRy[3*i:3*(i+1)] = (-n[1]*rji/norm_rji**2 - np.cross([0., 1., 0.], rjk)
                            )/(norm_rji*norm_rjk)
        dRy[3*k:3*(k+1)] = (-n[1]*rjk/norm_rjk**2 + np.cross([0., 1., 0.], rji)
                            )/(norm_rji*norm_rjk)
        dRy[3*j:3*(j+1)] = - dRy[3*i:3*(i+1)] - dRy[3*k:3*(k+1)]
        dRx = np.zeros(len(xyzs)*3)
        dRx[3*i:3*(i+1)] = (n[0]*rji/norm_rji**2 + np.cross([1., 0., 0.], rjk)
                            )/(norm_rji*norm_rjk)
        dRx[3*k:3*(k+1)] = (n[0]*rjk/norm_rjk**2 - np.cross([1., 0., 0.], rji)
                            )/(norm_rji*norm_rjk)
        dRx[3*j:3*(j+1)] = - dRx[3*i:3*(i+1)] - dRx[3*k:3*(k+1)]
        return Ry, Rx, dRy, dRx
    else:
        return Ry, Rx


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


def mod_dihedral(xyzs, i, j, k, l, derivative=False):
    F = xyzs[i, :] - xyzs[j, :]
    G = xyzs[j, :] - xyzs[k, :]
    H = xyzs[l, :] - xyzs[k, :]
    A = np.cross(F, G)
    B = np.cross(H, G)
    norm_G = np.linalg.norm(G)
    w = np.arctan2(
            np.dot(np.cross(B, A), G/norm_G),
            np.dot(A, B))
    cos_w, sin_w = np.cos(w), np.sin(w)
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
        return cos_w, sin_w, -sin_w*dw, cos_w*dw
    else:
        return cos_w, sin_w


def find_connectivity(atoms, threshold=1.25):
    bonds = []
    N = len(atoms)
    xyzs = atoms.get_positions()
    types = atoms.get_atomic_numbers()

    r2 = np.zeros((N, N))
    np.fill_diagonal(r2, np.inf)

    for i in range(N):
        for j in range(i+1, N):
            r2[i, j] = r2[j, i] = np.sum((xyzs[i, :] - xyzs[j, :])**2)
            if (r2[i, j] < threshold * (covalent_radii[types[i]]
                                        + covalent_radii[types[j]])**2):
                bonds.append((i, j))

    # Check for disconnected fragments
    connected = [True] + [False]*(N-1)
    for i in range(1, N):
        # Find minimum distance of any connected atom to any
        # unconnected atom
        masked_r2 = r2[connected, :][:, np.logical_not(connected)]
        ind = np.unravel_index(np.argmin(masked_r2), masked_r2.shape)
        # those indices are not the atomic indices due to the masking
        # of the array and have to be transformed first:
        atom1 = np.arange(N)[connected][ind[0]]
        atom2 = np.arange(N)[np.logical_not(connected)][ind[1]]
        # Add the new atom to the set of connected atoms
        connected[atom2] = True
        b = tuple(sorted([atom1, atom2]))

        # Add bond if not present
        if b not in bonds:
            bonds.append(b)
    return bonds


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
    return angles, dihedrals


def find_primitives(xyzs, bonds, threshold_angle=5., use_impropers=False):
    """
    Finds primitive internals given a reference geometry and connectivity list.
    threshold_angle: threshold to determine when to replace close to linear
                     bends with a coplanar and perpendicular bend coordinate.
    use_impropers: if true one of the bends in a close to planar configuration
                   is replaced by a suitable improper dihedral.
    """
    neighbors = [[] for _ in range(len(xyzs))]
    for b in bonds:
        neighbors[b[0]].append(b[1])
        neighbors[b[1]].append(b[0])
    neighbors = list(map(sorted, neighbors))
    bends = []
    linear_bends = []
    torsions = []
    impropers = []

    def cos_angle(r1, r2):
        """Calculates the cos of the angle between two vectors"""
        return np.dot(r1, r2)/(np.linalg.norm(r1)*np.linalg.norm(r2))

    lin_thresh = np.abs(np.cos(threshold_angle*np.pi/180.))

    for a in range(len(xyzs)):
        for i, ai in enumerate(neighbors[a]):
            for aj in neighbors[a][i+1:]:
                r_ai = xyzs[ai, :] - xyzs[a, :]
                r_aj = xyzs[aj, :] - xyzs[a, :]
                cos_theta = cos_angle(r_ai, r_aj)
                if (np.abs(cos_theta) > lin_thresh) and len(neighbors[a]) == 2:
                    # Add linear bend
                    linear_bends.append((ai, a, aj))
                    # Do not add torsions on linear bends, instead try
                    # using ai and aj as center bond. This does not detect
                    # longer linear chains
                    for ak in neighbors[ai]:
                        for am in neighbors[aj]:
                            if (ak != a and am != a
                                    and (ak, ai, aj, am) not in torsions
                                    and (am, aj, ai, ak) not in torsions):
                                r_ij = xyzs[aj, :] - xyzs[ai, :]
                                r_ik = xyzs[ak, :] - xyzs[ai, :]
                                r_jm = xyzs[am, :] - xyzs[aj, :]
                                cos_t1 = np.abs(cos_angle(r_ij, r_ik))
                                cos_t2 = np.abs(cos_angle(r_ij, r_jm))
                                if cos_t1 < 0.99 and cos_t2 < 0.99:
                                    torsions.append((ak, ai, aj, am))
                else:
                    bends.append((ai, a, aj))
                    for ak in neighbors[ai]:
                        if (ak != a and ak != aj
                                and (aj, a, ai, ak) not in torsions
                                and (ak, ai, a, aj) not in torsions):
                            # Check if (ak, ai, a) is linear
                            r_ik = xyzs[ak, :] - xyzs[ai, :]
                            cos_phi = cos_angle(r_ik, r_ai)
                            if np.abs(cos_phi) < 0.99:
                                torsions.append((ak, ai, a, aj))
                    for ak in neighbors[aj]:
                        if (ak != a and ak != ai
                                and (ak, aj, a, ai) not in torsions
                                and (ai, a, aj, ak) not in torsions):
                            # Check if (a, aj, ak) is linear
                            r_jk = xyzs[ak, :] - xyzs[aj, :]
                            cos_phi = cos_angle(r_jk, r_aj)
                            if np.abs(cos_phi) < 0.99:
                                torsions.append((ai, a, aj, ak))
        if use_impropers and len(neighbors[a]) > 2:
            # Check for planar configurations:
            for (ai, aj, ak) in combinations(neighbors[a], 3):
                r_ai = xyzs[ai, :] - xyzs[a, :]
                r_aj = xyzs[aj, :] - xyzs[a, :]
                r_ak = xyzs[ak, :] - xyzs[a, :]
                n1 = np.cross(r_ai, r_aj)
                n1 /= np.linalg.norm(n1)
                n2 = np.cross(r_aj, r_ak)
                n2 /= np.linalg.norm(n2)
                n3 = np.cross(r_ak, r_ai)
                n3 /= np.linalg.norm(n3)
                if (np.abs(n1.dot(n2)) > 0.95 or np.abs(n2.dot(n3)) > 0.95
                        or np.abs(n3.dot(n1)) > 0.95):
                    # Remove bend
                    bends.remove((ai, a, aj))
                    # Try to find an improper (b, a, c, d)
                    # such neither the angle t1 between (b, a, c)
                    # nor t2 between (a, c, d) is close to linear
                    for (b, c, d) in permutations([ai, aj, ak], 3):
                        r_ab = xyzs[b, :] - xyzs[a, :]
                        r_ac = xyzs[c, :] - xyzs[a, :]
                        r_cd = xyzs[d, :] - xyzs[c, :]
                        cos_t1 = cos_angle(r_ab, r_ac)
                        cos_t2 = cos_angle(r_ac, r_cd)
                        if np.abs(cos_t1) < 0.95 and np.abs(cos_t2 < 0.95):
                            impropers.append((b, a, c, d))
                            break
                    # Break after one improper has been added
                    break
    return bends, linear_bends, torsions, impropers


def to_primitives_factory(ref_geo, bonds, use_impropers=False):
    bends, linear_bends, torsions, impropers = find_primitives(
        ref_geo, bonds, use_impropers=use_impropers)
    print('Found %d bends, %d linear_bends, %d torsions and %d impropers' % (
          len(bends), len(linear_bends), len(torsions), len(impropers)))
    n_q = (len(bonds) + len(bends) + 2*len(linear_bends)
           + len(torsions) + len(impropers))

    def to_primitives(atoms):
        xyzs = atoms.get_positions()
        qs = np.zeros(n_q)
        dqs = np.zeros((n_q, len(xyzs)*3))
        for i, b in enumerate(bonds):
            qs[i], dqs[i, :] = dist(xyzs, b[0], b[1], derivative=True)
        for i, a in enumerate(bends):
            j = len(bonds) + i
            qs[j], dqs[j, :] = angle(xyzs, a[0], a[1], a[2], derivative=True)
        for i, a in enumerate(linear_bends):
            j = len(bonds) + len(bends) + 2*i
            qs[j], qs[j+1], dqs[j, :], dqs[j+1, :] = linear_bend(
                xyzs, a[0], a[1], a[2], derivative=True)
        for i, d in enumerate(torsions + impropers):
            j = len(bonds) + len(bends) + 2*len(linear_bends) + i
            qs[j], dqs[j, :] = dihedral(xyzs, d[0], d[1], d[2], d[3],
                                        derivative=True)
        return qs, dqs
    return to_primitives, bends, linear_bends, torsions, impropers


def to_nonredundant_primitives_factory(ref_geo, bonds, use_impropers):
    bends, linear_bends, torsions, impropers = find_primitives(
        ref_geo, bonds, use_impropers=use_impropers)
    print('Found %d stretches, %d bends, %d linear bends, '
          '%d torsions and %d impropers' % (
           len(bonds), len(bends), len(linear_bends),
           len(torsions), len(impropers)))

    u = []
    active_bonds = []
    for b in bonds:
        _, ui = dist(ref_geo, b[0], b[1], derivative=True)
        for uj in u:
            ui -= uj.dot(ui)/(uj.dot(uj)) * uj
        norm = np.linalg.norm(ui)
        if norm > 1e-6:
            u.append(ui/norm)
            active_bonds.append(b)
    active_bends = []
    for a in bends:
        _, ui = angle(ref_geo, a[0], a[1], a[2], derivative=True)
        for uj in u:
            ui -= uj.dot(ui)/(uj.dot(uj)) * uj
        norm = np.linalg.norm(ui)
        if norm > 1e-6:
            u.append(ui/norm)
            active_bends.append(a)
    active_linear_bends = []
    for a in linear_bends:
        _, _, ui1, ui2 = linear_bend(ref_geo, a[0], a[1], a[2],
                                     derivative=True)
        for uj in u:
            ui1 -= uj.dot(ui1)/(uj.dot(uj)) * uj
        norm1 = np.linalg.norm(ui1)
        for uj in u:
            ui2 -= uj.dot(ui2)/(uj.dot(uj)) * uj
        norm2 = np.linalg.norm(ui2)
        if norm1 > 1e-6 and norm2 > 1e-6:
            u.append(ui1/norm1)
            u.append(ui2/norm2)
            active_linear_bends.append(a)
    active_torsions = []
    for d in torsions:
        _, ui = dihedral(ref_geo, d[0], d[1], d[2], d[3], derivative=True)
        for uj in u:
            ui -= uj.dot(ui)/(uj.dot(uj)) * uj
        norm = np.linalg.norm(ui)
        if norm > 1e-6:
            u.append(ui/norm)
            active_torsions.append(d)
    active_impropers = []
    for d in impropers:
        _, ui = dihedral(ref_geo, d[0], d[1], d[2], d[3], derivative=True)
        for uj in u:
            ui -= uj.dot(ui)/(uj.dot(uj)) * uj
        norm = np.linalg.norm(ui)
        if norm > 1e-6:
            u.append(ui/norm)
            active_impropers.append(d)

    print('Final set: %d stretches, %d bends, %d linear bends, '
          '%d torsions and %d impropers' % (
           len(active_bonds), len(active_bends), 2*len(active_linear_bends),
           len(active_torsions), len(active_impropers)))
    n_q = (len(active_bonds) + len(active_bends) + 2*len(active_linear_bends)
           + len(active_torsions) + len(active_impropers))

    def to_primitives(atoms):
        xyzs = atoms.get_positions()
        qs = np.zeros(n_q)
        dqs = np.zeros((n_q, len(xyzs)*3))
        for i, b in enumerate(active_bonds):
            qs[i], dqs[i, :] = dist(xyzs, b[0], b[1], derivative=True)
        for i, a in enumerate(active_bends):
            j = len(active_bonds) + i
            qs[j], dqs[j, :] = angle(xyzs, a[0], a[1], a[2], derivative=True)
        for i, a in enumerate(active_linear_bends):
            j = len(active_bonds) + len(active_bends) + 2*i
            qs[j], qs[j+1], dqs[j], dqs[j+1] = linear_bend(
                xyzs, a[0], a[1], a[2], derivative=True)
        for i, d in enumerate(active_torsions + active_impropers):
            j = (len(active_bonds) + len(active_bends)
                 + 2*len(active_linear_bends) + i)
            qs[j], dqs[j, :] = dihedral(xyzs, d[0], d[1], d[2], d[3],
                                        derivative=True)
        return qs, dqs
    return (to_primitives, active_bonds, active_bends, active_linear_bends,
            active_torsions, active_impropers)


def to_dic_factory(bonds, atoms_ref):
    transform = to_primitives_factory(atoms_ref.get_positions(), bonds)[0]
    # Wilson B matrix is just the derivative of q with respect to x
    _, B = transform(atoms_ref)
    # G matrix without mass weighting
    G = B.dot(B.T)
    w, v = np.linalg.eigh(G)
    # Set of nonredundant eigenvectors (eigenvalue =/= 0)
    U = v[:, w > 1e-10]

    def to_dic(atoms):
        q, dq = transform(atoms)
        return U.T.dot(q), U.T.dot(dq)
    return to_dic


def to_mass_weighted(atoms):
    xyzs = atoms.get_positions()
    masses = atoms.get_masses()
    q = xyzs.flatten() * np.repeat(np.sqrt(masses), 3)
    dq = np.eye(3*len(atoms))
    dq *= np.repeat(np.sqrt(masses), 3)[:, None]
    return q,  dq


def to_COM(atoms):
    xyzs = atoms.get_positions()
    masses = atoms.get_masses()
    total_mass = masses.sum()
    rel_masses = masses/total_mass
    com = rel_masses.dot(xyzs)
    q = (xyzs - com).flatten()
    dq = np.eye(3*len(atoms))
    dq[::3, ::3] -= rel_masses
    dq[1::3, 1::3] -= rel_masses
    dq[2::3, 2::3] -= rel_masses
    return q,  dq


def to_COM_mass_weighted(atoms):
    xyzs = atoms.get_positions()
    masses = atoms.get_masses()
    total_mass = masses.sum()
    rel_masses = masses/total_mass
    com = rel_masses.dot(xyzs)
    q = (xyzs - com).flatten() * np.repeat(np.sqrt(masses), 3)
    dq = np.eye(3*len(atoms))
    dq[::3, ::3] -= rel_masses
    dq[1::3, 1::3] -= rel_masses
    dq[2::3, 2::3] -= rel_masses
    dq *= np.repeat(np.sqrt(masses), 3)[:, None]
    return q,  dq


def to_standard_orientation(atoms):
    """
    Moves the center of mass to the origin and
    aligns the moments of interia with the axes.
    Smallest along the x-axis, largest along the
    z-axis.
    """
    xyzs = atoms.get_positions()
    masses = atoms.get_masses()
    com = masses.dot(xyzs)/masses.sum()
    xyzs -= com
    inertial_tensor = np.zeros((3, 3))
    for i in range(len(atoms)):
        inertial_tensor += masses[i]*(
            xyzs[i, :].dot(xyzs[i, :])*np.eye(3)
            - np.outer(xyzs[i, :], xyzs[i, :]))
    w, v = np.linalg.eigh(inertial_tensor)
    return xyzs.dot(v)


def to_distance_matrix(atoms):
    xyzs = atoms.get_positions()
    N = len(atoms)
    r_vec = xyzs[np.newaxis, :, :] - xyzs[:, np.newaxis, :]
    q = np.sqrt(np.sum(r_vec**2, axis=2))
    dq = (r_vec[:, :, np.newaxis, :]
          * (- np.eye(N, N)[:, np.newaxis, :, np.newaxis]
             + np.eye(N, N)[np.newaxis, :, :, np.newaxis]))
    dq[q > 0, :, :] /= q[q > 0, np.newaxis, np.newaxis]
    return q.flatten(), dq.reshape(-1, 3*N)


def to_inverse_distance_matrix(atoms):
    xyzs = atoms.get_positions()
    N = len(atoms)
    r_vec = xyzs[np.newaxis, :, :] - xyzs[:, np.newaxis, :]
    r = np.sqrt(np.sum(r_vec**2, axis=2))
    q = np.zeros((N, N))
    q[r > 0] = 1.0/r[r > 0]
    dq = (r_vec[:, :, np.newaxis, :]*q[:, :, np.newaxis, np.newaxis]**3
          * (np.eye(N, N)[:, np.newaxis, :, np.newaxis]
             - np.eye(N, N)[np.newaxis, :, :, np.newaxis]))
    return q.flatten(), dq.reshape(-1, 3*N)
