import numpy as np
from numba import njit, prange

def Faces2Edges(Faces):
    """
    Construct an edge list from a triangle mesh.

    Given a manifold triangle mesh (where every edge belongs to exactly two faces),
    this function returns an edge list where each edge is associated with
    the two adjacent faces and their relative orientations.

    Parameters
    ----------
    Faces : (Nf, 3) np.ndarray, dtype=int32
        Triangle face connectivity array. Each row contains three vertex indices
        ordered counter-clockwise (CCW) when viewed from the outer-normal side of the face.

    Returns
    -------
    Edges : (Ne, 4) np.ndarray, dtype=int32
        - Column 0: start vertex index (smaller of the two vertex indices)
        - Column 1: end vertex index (larger of the two vertex indices)
        - Column 2: face index where the edge direction (start → end) follows 
                    the CCW winding of the face
        - Column 3: face index where the edge direction (start → end) opposes 
                    the CCW winding (i.e., appears CW in that face)

    Raises
    ------
    ValueError
        If the input mesh is non-manifold (i.e., any edge is shared by fewer or
        more than two faces).

    Notes
    -----
    - Uses **int32** for all output indices to reduce memory consumption.
    - Ensure that all vertex indices in Faces is < 2,147,483,647 (max int32)
      to avoid overflow. This is sufficient for most practical meshes.
    """
    Faces    = np.asarray(Faces, dtype=np.int32);
    Nf       = Faces.shape[0];
    face_ids = np.repeat(np.arange(Nf), 3);
    v_sta    = Faces.flatten();
    v_end    = np.roll(Faces, -1, axis=1).flatten();
    ccw      = np.where(v_sta > v_end, -1, 1);
    temp     = np.stack((np.minimum(v_sta, v_end), 
                         np.maximum(v_sta, v_end),
                         face_ids, ccw), axis=1);
    sorted   = temp[np.lexsort((temp[:,3], temp[:,1], temp[:,0]))];
    Edges    = np.column_stack((sorted[1::2,:3], sorted[0::2,2]));

    return Edges;

@njit(parallel=True, fastmath=True, nogil=True)
def _compute_gravity_numba(
    P,
    A_f, B_f, C_f, hnf, mag_nf, A_dot_nf,
    A_e, B_e, mag_eAB,
    hn_pos, hn_neg, hnAB, hnBA,
    A_dot_hn_pos, A_dot_hn_neg, A_dot_hnAB, A_dot_hnBA,
    G, rho, km2m, si2mg, si2eot
):
    M = P.shape[0]
    V, gx, gy, gz, Txx, Tyy, Tzz, Txy, Txz, Tyz \
        = [np.zeros(M) for _ in range(10)]

    for i in prange(M):
        px, py, pz = P[i, 0], P[i, 1], P[i, 2]

        Vf = Ve = 0.0
        gxf = gyf = gzf = 0.0
        gxe = gye = gze = 0.0
        Txx_f = Tyy_f = Tzz_f = Txy_f = Txz_f = Tyz_f = 0.0
        Txx_e = Tyy_e = Tzz_e = Txy_e = Txz_e = Tyz_e = 0.0

        # --- Face loop ---
        Nf = A_f.shape[0]
        for j in range(Nf):
            ax = A_f[j, 0] - px; ay = A_f[j, 1] - py; az = A_f[j, 2] - pz
            bx = B_f[j, 0] - px; by = B_f[j, 1] - py; bz = B_f[j, 2] - pz
            cx = C_f[j, 0] - px; cy = C_f[j, 1] - py; cz = C_f[j, 2] - pz

            rPA = (ax*ax + ay*ay + az*az)**0.5
            rPB = (bx*bx + by*by + bz*bz)**0.5
            rPC = (cx*cx + cy*cy + cz*cz)**0.5

            rPA_dot_rPB = ax*bx + ay*by + az*bz
            rPB_dot_rPC = bx*cx + by*cy + bz*cz
            rPC_dot_rPA = cx*ax + cy*ay + cz*az

            denom = rPA*rPB*rPC + rPA*rPB_dot_rPC + rPB*rPC_dot_rPA + rPC*rPA_dot_rPB
            mixProd = A_dot_nf[j] - mag_nf[j] * (px*hnf[j,0] + py*hnf[j,1] + pz*hnf[j,2])
            wf = 2.0 * np.arctan2(mixProd, denom)
            rf_dot_hnf = mixProd / mag_nf[j]

            Vf += rf_dot_hnf * rf_dot_hnf * wf
            gxf += hnf[j,0] * rf_dot_hnf * wf
            gyf += hnf[j,1] * rf_dot_hnf * wf
            gzf += hnf[j,2] * rf_dot_hnf * wf

            Txx_f += hnf[j,0] * hnf[j,0] * wf
            Tyy_f += hnf[j,1] * hnf[j,1] * wf
            Tzz_f += hnf[j,2] * hnf[j,2] * wf
            Txy_f += hnf[j,0] * hnf[j,1] * wf
            Txz_f += hnf[j,0] * hnf[j,2] * wf
            Tyz_f += hnf[j,1] * hnf[j,2] * wf

        # --- Edge loop ---
        Ne = A_e.shape[0]
        for k in range(Ne):
            ax = A_e[k, 0] - px; ay = A_e[k, 1] - py; az = A_e[k, 2] - pz
            bx = B_e[k, 0] - px; by = B_e[k, 1] - py; bz = B_e[k, 2] - pz

            rPA = (ax*ax + ay*ay + az*az)**0.5
            rPB = (bx*bx + by*by + bz*bz)**0.5
            sum_r = rPA + rPB
            Le = np.log((sum_r + mag_eAB[k]) / (sum_r - mag_eAB[k]))

            # Dot products: re · h = A·h - P·h
            re_dot_hn_pos = A_dot_hn_pos[k] - (px * hn_pos[k,0] + py * hn_pos[k,1] + pz * hn_pos[k,2])
            re_dot_hnAB   = A_dot_hnAB[k]  - (px * hnAB[k,0]  + py * hnAB[k,1]  + pz * hnAB[k,2])
            re_dot_hn_neg = A_dot_hn_neg[k] - (px * hn_neg[k,0] + py * hn_neg[k,1] + pz * hn_neg[k,2])
            re_dot_hnBA   = A_dot_hnBA[k]  - (px * hnBA[k,0]  + py * hnBA[k,1]  + pz * hnBA[k,2])


            rEr = re_dot_hn_pos * re_dot_hnAB + re_dot_hn_neg * re_dot_hnBA

            Ve += rEr * Le
            gxe += (hn_pos[k,0] * re_dot_hnAB + hn_neg[k,0] * re_dot_hnBA) * Le
            gye += (hn_pos[k,1] * re_dot_hnAB + hn_neg[k,1] * re_dot_hnBA) * Le
            gze += (hn_pos[k,2] * re_dot_hnAB + hn_neg[k,2] * re_dot_hnBA) * Le

            Txx_e += (hn_pos[k,0] * hnAB[k,0] + hn_neg[k,0] * hnBA[k,0]) * Le
            Tyy_e += (hn_pos[k,1] * hnAB[k,1] + hn_neg[k,1] * hnBA[k,1]) * Le
            Tzz_e += (hn_pos[k,2] * hnAB[k,2] + hn_neg[k,2] * hnBA[k,2]) * Le
            Txy_e += (hn_pos[k,0] * hnAB[k,1] + hn_neg[k,0] * hnBA[k,1]) * Le
            Txz_e += (hn_pos[k,0] * hnAB[k,2] + hn_neg[k,0] * hnBA[k,2]) * Le
            Tyz_e += (hn_pos[k,1] * hnAB[k,2] + hn_neg[k,1] * hnBA[k,2]) * Le

        scale_V = 0.5 * G * rho * (km2m * km2m)
        scale_g = G * rho * km2m * si2mg
        scale_T = G * rho * si2eot

        V[i]  = scale_V * (Ve - Vf)
        gx[i] = scale_g * (gxf - gxe)
        gy[i] = scale_g * (gyf - gye)
        gz[i] = scale_g * (gzf - gze)
        Txx[i] = scale_T * (Txx_e - Txx_f)
        Tyy[i] = scale_T * (Tyy_e - Tyy_f)
        Tzz[i] = scale_T * (Tzz_e - Tzz_f)
        Txy[i] = scale_T * (Txy_e - Txy_f)
        Txz[i] = scale_T * (Txz_e - Txz_f)
        Tyz[i] = scale_T * (Tyz_e - Tyz_f)

    return V, gx, gy, gz, Txx, Tyy, Tzz, Txy, Txz, Tyz


def VecWerSch_numba(P, Q, If, rho):
    # Constants (float64)
    G = 6.67430e-11
    km2m = 1.e3
    si2mg = 1.e5
    si2eot = 1.e9

    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    If = np.asarray(If, dtype=np.int32)

    # --- Face geometry ---
    A_f = Q[If[:, 0]]
    B_f = Q[If[:, 1]]
    C_f = Q[If[:, 2]]

    AB = B_f - A_f
    BC = C_f - B_f
    nf = np.empty_like(A_f)
    nf[:, 0] = AB[:, 1] * BC[:, 2] - AB[:, 2] * BC[:, 1]
    nf[:, 1] = AB[:, 2] * BC[:, 0] - AB[:, 0] * BC[:, 2]
    nf[:, 2] = AB[:, 0] * BC[:, 1] - AB[:, 1] * BC[:, 0]

    mag_nf = np.sqrt(np.sum(nf * nf, axis=1))
    hnf = nf / mag_nf[:, None]
    A_dot_nf = np.sum(A_f * nf, axis=1)

    # --- Edge geometry ---
    Ie = Faces2Edges(If)  # (Ne, 4)
    A_e = Q[Ie[:, 0]]
    B_e = Q[Ie[:, 1]]
    eAB = B_e - A_e
    mag_eAB = np.sqrt(np.sum(eAB * eAB, axis=1))

    f_pos = Ie[:, 2]
    f_neg = Ie[:, 3]
    hn_pos = hnf[f_pos]
    hn_neg = hnf[f_neg]

    heAB = eAB / mag_eAB[:, None]

    # Manual cross: heAB × hn_pos
    hnAB = np.empty_like(hn_pos)
    hnAB[:, 0] = heAB[:,1] * hn_pos[:,2] - heAB[:,2] * hn_pos[:,1]
    hnAB[:, 1] = heAB[:,2] * hn_pos[:,0] - heAB[:,0] * hn_pos[:,2]
    hnAB[:, 2] = heAB[:,0] * hn_pos[:,1] - heAB[:,1] * hn_pos[:,0]

    # Manual cross: (-heAB) × hn_neg
    hnBA = np.empty_like(hn_neg)
    hnBA[:, 0] = -heAB[:,1] * hn_neg[:,2] + heAB[:,2] * hn_neg[:,1]
    hnBA[:, 1] = -heAB[:,2] * hn_neg[:,0] + heAB[:,0] * hn_neg[:,2]
    hnBA[:, 2] = -heAB[:,0] * hn_neg[:,1] + heAB[:,1] * hn_neg[:,0]

    A_dot_hn_pos = np.sum(A_e * hn_pos, axis=1)
    A_dot_hn_neg = np.sum(A_e * hn_neg, axis=1)
    A_dot_hnAB  = np.sum(A_e * hnAB, axis=1)
    A_dot_hnBA  = np.sum(A_e * hnBA, axis=1)

    # --- Compute ---
    return _compute_gravity_numba(
        P, A_f, B_f, C_f, hnf, mag_nf, A_dot_nf,
        A_e, B_e, mag_eAB,
        hn_pos, hn_neg, hnAB, hnBA,
        A_dot_hn_pos, A_dot_hn_neg, A_dot_hnAB, A_dot_hnBA,
        G, rho, km2m, si2mg, si2eot
    )

