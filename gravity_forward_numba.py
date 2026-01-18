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

# ------------------------------------------------------------
# Core Numba Kernel 
# ------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _compute_potential_numba(
    P, A_f, B_f, C_f, hnf, mag_nf, A_dot_nf,
    A_e, B_e, mag_eAB,
    hn_pos, hn_neg, hnAB, hnBA,
    A_dot_hn_pos, A_dot_hn_neg, A_dot_hnAB, A_dot_hnBA,
    G, rho, km2m
):
    M = P.shape[0]
    V = np.zeros(M, dtype=np.float64)

    for i in prange(M):
        px, py, pz = P[i, 0], P[i, 1], P[i, 2]
        Vf = np.float64(0.0)
        Ve = np.float64(0.0)

        # --- Face contribution (Vf) ---
        Nf = A_f.shape[0]
        for j in range(Nf):
            ax = A_f[j, 0] - px
            ay = A_f[j, 1] - py
            az = A_f[j, 2] - pz

            bx = B_f[j, 0] - px
            by = B_f[j, 1] - py
            bz = B_f[j, 2] - pz

            cx = C_f[j, 0] - px
            cy = C_f[j, 1] - py
            cz = C_f[j, 2] - pz

            rPA = np.sqrt(ax*ax + ay*ay + az*az)
            rPB = np.sqrt(bx*bx + by*by + bz*bz)
            rPC = np.sqrt(cx*cx + cy*cy + cz*cz)
            rPA_dot_rPB = ax*bx + ay*by + az*bz
            rPB_dot_rPC = bx*cx + by*cy + bz*cz
            rPC_dot_rPA = cx*ax + cy*ay + cz*az

            # Solid angle denominator
            denom = (rPA * rPB * rPC 
            + rPA * rPB_dot_rPC 
            + rPB * rPC_dot_rPA 
            + rPC * rPA_dot_rPB)

            mixProd = A_dot_nf[j] - mag_nf[j] * (px * hnf[j,0] + py * hnf[j,1] + pz * hnf[j,2])
            wf = np.float64(2.0) * np.arctan2(mixProd, denom)
            rf_dot_hnf = mixProd / mag_nf[j]
            Vf += rf_dot_hnf * rf_dot_hnf * wf

        # --- Edge contribution (Ve) ---
        Ne = A_e.shape[0]
        for k in range(Ne):
            ax = A_e[k, 0] - px
            ay = A_e[k, 1] - py
            az = A_e[k, 2] - pz

            bx = B_e[k, 0] - px
            by = B_e[k, 1] - py
            bz = B_e[k, 2] - pz

            rPA = np.sqrt(ax*ax + ay*ay + az*az)
            rPB = np.sqrt(bx*bx + by*by + bz*bz)

            sum_r = rPA + rPB
            Le = np.log((sum_r + mag_eAB[k]) / (sum_r - mag_eAB[k]))

            # Dot products: re · h = A·h - P·h
            re_dot_hn_pos = A_dot_hn_pos[k] - (px * hn_pos[k,0] + py * hn_pos[k,1] + pz * hn_pos[k,2])
            re_dot_hnAB   = A_dot_hnAB[k]  - (px * hnAB[k,0]  + py * hnAB[k,1]  + pz * hnAB[k,2])
            re_dot_hn_neg = A_dot_hn_neg[k] - (px * hn_neg[k,0] + py * hn_neg[k,1] + pz * hn_neg[k,2])
            re_dot_hnBA   = A_dot_hnBA[k]  - (px * hnBA[k,0]  + py * hnBA[k,1]  + pz * hnBA[k,2])

            Ve += (re_dot_hn_pos * re_dot_hnAB + re_dot_hn_neg * re_dot_hnBA) * Le

        V[i] = np.float64(0.5) * G * rho * (Ve - Vf) * (km2m * km2m)

    return V

# ------------------------------------------------------------
# Public API: Drop-in replacement for VecWerSch
# ------------------------------------------------------------
def VecWerSch_numba(P, Q, If, rho):
    """
    Numba-accelerated, memory-efficient drop-in replacement for VecWerSch.
    
    Computes gravitational potential using the Werner-Schmidt polyhedral method.
    
    Parameters
    ----------
    P : (M, 3) ndarray
        Observation points [km]
    Q : (Nv, 3) ndarray
        Computation points [km]
    If : (Nf, 3) ndarray, int
        Triangle faces (CCW, outward normal)
    rho : float
        Density [kg/m³]

    Returns
    -------
    V : (M,) ndarray
        Gravitational potential [m²/s²]
        
    Notes
    -----
    - Input units: km, kg/m³ --> Output: m²/s²
    - Requires manifold mesh (each edge shared by exactly two faces).
    """
    # Constants (as float64 for Numba)
    G = np.float64(6.67430e-11)
    km2m = np.float64(1e3)
    
    # Ensure proper dtypes
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    If = np.asarray(If, dtype=np.int32)

    # --- Face geometry ---
    A_f = Q[If[:, 0]]
    B_f = Q[If[:, 1]]
    C_f = Q[If[:, 2]]

    AB_f = B_f - A_f
    BC_f = C_f - B_f
    nf = np.empty_like(A_f)
    nf[:, 0] = AB_f[:, 1] * BC_f[:, 2] - AB_f[:, 2] * BC_f[:, 1]
    nf[:, 1] = AB_f[:, 2] * BC_f[:, 0] - AB_f[:, 0] * BC_f[:, 2]
    nf[:, 2] = AB_f[:, 0] * BC_f[:, 1] - AB_f[:, 1] * BC_f[:, 0]

    mag_nf = np.sqrt(np.sum(nf * nf, axis=1))
    hnf = nf / mag_nf[:, None]
    A_dot_nf = np.sum(A_f * nf, axis=1).astype(np.float64)

    # --- Edge geometry ---
    Ie = Faces2Edges(If)  # (Ne, 4): [v0, v1, f_pos, f_neg]
    A_e = Q[Ie[:, 0]]
    B_e = Q[Ie[:, 1]]
    eAB = B_e - A_e
    mag_eAB = np.sqrt(np.sum(eAB * eAB, axis=1)).astype(np.float64)
    heAB = eAB / mag_eAB[:, None]

    # Normals from adjacent faces
    f_pos = Ie[:, 2]  # face where edge is CCW
    f_neg = Ie[:, 3]  # face where edge is CW
    hn_pos = hnf[f_pos]  # h_{n,ABC}
    hn_neg = hnf[f_neg]  # h_{n,BAD}

    # Edge normals: h_{e,AB} = e_AB × h_{n,ABC}, h_{e,BA} = (-e_AB) × h_{n,BAD}
    hnAB = np.cross( heAB, hn_pos)
    hnBA = np.cross(-heAB, hn_neg)

    # Precompute A · h for all edge-related normals
    A_dot_hn_pos = np.sum(A_e * hn_pos, axis=1).astype(np.float64)
    A_dot_hn_neg = np.sum(A_e * hn_neg, axis=1).astype(np.float64)
    A_dot_hnAB  = np.sum(A_e * hnAB, axis=1).astype(np.float64)
    A_dot_hnBA  = np.sum(A_e * hnBA, axis=1).astype(np.float64)

    # --- Compute potential ---
    V = _compute_potential_numba(
        P, A_f, B_f, C_f, hnf, mag_nf, A_dot_nf,
        A_e, B_e, mag_eAB,
        hn_pos, hn_neg, hnAB, hnBA,
        A_dot_hn_pos, A_dot_hn_neg, A_dot_hnAB, A_dot_hnBA,
        G, np.float64(rho), km2m
    )

    return V.astype(np.float64)  # Match original output type