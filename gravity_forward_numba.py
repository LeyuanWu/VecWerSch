import numpy as np
from numba import njit, prange

def gsphere(xp, yp, zp, xq, yq, zq, a, rho):
    """Forward modelling gravity fields due to a sphere with uniform density

    Args:
        <xp, yp, zp> (numpy array)      : computation points
        <xq, yq, zq> (float)            : center of the sphere
        a (float)                       : radius of the sphere
        rho (float)                     : density of the sphere
        
    Units:
    G in m^3 kg^{-1} s^{-2}
    length in km
    density in kg/m^3
    gravity potential in m^2/s^2
    GV in mGal
    GGT in 1e-9/s^2 (Eotvos)

    Returns:
        dV                              : GP  in m^2/s^2
        gx, gy, gz                      : GV  in mGal
        Txx, Tyy, Tzz, Txy, Txz, Tyz    : GGT in Eotvos
    """
    G           = 6.67430e-11;
    km2m        = 1e3;
    si2mg       = 1e5;          # m/s^2 to mGal
    si2eot      = 1e9;          # 1/s^2 to 10^{-9}/s^2
    M           = rho*4/3*np.pi*a**3;
    xqp         = xp-xq;
    yqp         = yp-yq;
    zqp         = zp-zq;
    r           = np.sqrt(xqp**2 + yqp**2 + zqp**2);
    r2          = r**2;
    r3          = r**3;
    r5          = r**5;
    ma_ou       = (r>=a);
    ma_in       = np.logical_not(ma_ou);
    
    ######## * Init
    dV = np.zeros_like(xp);
    gx = np.zeros_like(xp);   gy = np.zeros_like(xp);  gz = np.zeros_like(xp);
    Txx = np.zeros_like(xp); Tyy = np.zeros_like(xp); Tzz = np.zeros_like(xp);
    Txy = np.zeros_like(xp); Txz = np.zeros_like(xp); Tyz = np.zeros_like(xp);
    ######## * Outside the sphere
    xqp_ou = xqp[ma_ou]; yqp_ou = yqp[ma_ou]; zqp_ou = zqp[ma_ou];
    r_ou  = r[ma_ou];  r2_ou = r2[ma_ou];
    r3_ou = r3[ma_ou]; r5_ou = r5[ma_ou];
    
    dV[ma_ou]     =   km2m**2 * G*M/r_ou;
    gx[ma_ou]     = - km2m*si2mg * G*M * xqp_ou/r3_ou;
    gy[ma_ou]     = - km2m*si2mg * G*M * yqp_ou/r3_ou;
    gz[ma_ou]     = - km2m*si2mg * G*M * zqp_ou/r3_ou;
    Txx[ma_ou]    = - si2eot * G*M * (r2_ou-3*xqp_ou**2)/r5_ou;
    Tyy[ma_ou]    = - si2eot * G*M * (r2_ou-3*yqp_ou**2)/r5_ou;
    Tzz[ma_ou]    = - si2eot * G*M * (r2_ou-3*zqp_ou**2)/r5_ou;
    Txy[ma_ou]    =   si2eot * G*M * (3*xqp_ou*yqp_ou)/r5_ou;
    Txz[ma_ou]    =   si2eot * G*M * (3*xqp_ou*zqp_ou)/r5_ou;
    Tyz[ma_ou]    =   si2eot * G*M * (3*yqp_ou*zqp_ou)/r5_ou;
    ######## * Inside the sphere
    dV[ma_in]     =   km2m**2 * G*M* (3*a**2-r2[ma_in])/(2*a**3);
    gx[ma_in]     = - km2m*si2mg * G*M * xqp[ma_in]/a**3;
    gy[ma_in]     = - km2m*si2mg * G*M * yqp[ma_in]/a**3;
    gz[ma_in]     = - km2m*si2mg * G*M * zqp[ma_in]/a**3;
    Txx[ma_in]    = - si2eot * G*M/a**3;
    Tyy[ma_in]    = - si2eot * G*M/a**3;
    Tzz[ma_in]    = - si2eot * G*M/a**3;
    Txy[ma_in]    =   0;
    Txz[ma_in]    =   0;
    Tyz[ma_in]    =   0;
    
    return dV, gx, gy, gz, Txx, Txy, Txz, Tyy, Tyz, Tzz;


def _Faces2Edges(Faces):
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

    return V, gx, gy, gz, Txx, Txy, Txz, Tyy, Tyz, Tzz


def VecWerSch_numba(P, Q, If, rho):
    """
    Computes the gravitational potential (and optionally field and gradient) at multiple observation points 
    due to a homogeneous polyhedral body using the vectorized formulation of the Werner-Schmidt method.

    This implementation follows the analytical expressions derived from potential theory for a constant-density 
    polyhedron. It leverages efficient NumPy vectorization to compute contributions from all faces and edges 
    simultaneously across many observation points.

    Reference:
    Werner, R.A., Scheeres, D.J. Exterior gravitation of a polyhedron derived and compared with harmonic and mascon gravitation    representations of asteroid 4769 Castalia. Celestial Mech Dyn Astr 65, 313-344 (1996). https://doi.org/10.1007/BF00053511

    Args:
        P (numpy.ndarray): Array of shape (M, 3) containing the Cartesian coordinates [x, y, z] 
                           of M observation (computation) points where the gravity field is evaluated.
        Q (numpy.ndarray): Array of shape (Nv, 3) listing the 3D coordinates of N unique vertices 
                           defining the polyhedron geometry.
        If (numpy.ndarray): Integer array of shape (Nf, 3) specifying the vertex indices of F triangular faces. 
                            Each row [i, j, k] corresponds to a face with vertices Q[i], Q[j], Q[k], 
                            oriented consistently (outward-pointing normal via right-hand rule).
        rho (float): Constant density of the polyhedron in kg/m³. 
                                     
    Returns:
        V (numpy.ndarray) : Gravitational Potential (GP) at each point in P.
        gx, gy, gz (numpy.ndarray) : Gravitational Vector (GV).
        Txx, Tyy, Tzz, Txy, Txz, Tyz (numpy.ndarray): Gravity Gradient Tensor (GGT)

    Units:
        length in km, density in kg/m^3,
        GP in m^2/s^2, GV in mGal, GGT in 1e-9/s^2 (Eotvos)
    """

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
    Ie = _Faces2Edges(If)  # (Ne, 4)
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


@njit(fastmath=True)
def spherical_edge_length_range(verts, faces):
    """
    Compute min and max great-circle edge lengths (in radians) of a spherical triangular mesh.
    
    Parameters:
        verts: (N, 3) float array, vertex coordinates (assumed normalized to unit sphere)
        faces: (M, 3) int array, vertex indices for triangular faces
    
    Returns:
        min_dist: float, minimum edge length (radians)
        max_dist: float, maximum edge length (radians)
    
    Raises:
        ValueError: If faces array is empty.
    """
    if faces.size == 0:
        raise ValueError("Faces array is empty - no edges to process!")
    
    min_dist = np.inf
    max_dist = -np.inf

    for f in range(faces.shape[0]):
        i0, i1, i2 = faces[f]
        v0 = verts[i0]
        v1 = verts[i1]
        v2 = verts[i2]

        # Edge 0-1
        d = _great_circle_distance(v0, v1)
        if d < min_dist: min_dist = d
        if d > max_dist: max_dist = d

        # Edge 1-2
        d = _great_circle_distance(v1, v2)
        if d < min_dist: min_dist = d
        if d > max_dist: max_dist = d

        # Edge 2-0
        d = _great_circle_distance(v2, v0)
        if d < min_dist: min_dist = d
        if d > max_dist: max_dist = d

    return min_dist, max_dist


@njit(fastmath=True)
def _great_circle_distance(a, b):
    cross_norm = np.sqrt(
        (a[1]*b[2] - a[2]*b[1])**2 +
        (a[2]*b[0] - a[0]*b[2])**2 +
        (a[0]*b[1] - a[1]*b[0])**2
    )
    dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    return np.arctan2(cross_norm, dot)


def rotate_vec_ten_ecef2ned(lon, lat, gx, gy, gz,
                            Txx, Txy, Txz, Tyy, Tyz, Tzz):
    """
    Rotate gravity vector and gravity gradient tensor (GGT) from ECEF (Earth-Centered,
    Earth-Fixed) Cartesian coordinates to local North-East-Down (NED) frame.

    Parameters
    ----------
    lon, lat : array_like, shape (N,)
        Longitude and latitude of observation points **in radians**.
    gx, gy, gz : array_like, shape (N,)
        Gravity vector components in ECEF.
    Txx, Txy, Txz, Tyy, Tyz, Tzz : array_like, shape (N,)
        Independent components of the GGT in ECEF (symmetric tensor).

    Returns
    -------
    gN, gE, gD : ndarray, shape (N,)
        Gravity vector in local NED frame.
    TNN, TNE, TND, TEE, TED, TDD : ndarray, shape (N,)
        Independent components of the GGT in local NED frame.
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    N = lon.shape[0]

    # Precompute trigonometric values
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # Build rotation matrix R: from ECEF to NED
    # R[i, :, :] is 3x3 matrix for point i
    R = np.empty((N, 3, 3), dtype=np.float64)

    # North row
    R[:, 0, 0] = -sin_lat * cos_lon   # d(North)/dX
    R[:, 0, 1] = -sin_lat * sin_lon   # d(North)/dY
    R[:, 0, 2] =  cos_lat             # d(North)/dZ

    # East row
    R[:, 1, 0] = -sin_lon             # d(East)/dX
    R[:, 1, 1] =  cos_lon             # d(East)/dY
    R[:, 1, 2] =  0.0                 # d(East)/dZ

    # Down row
    R[:, 2, 0] = -cos_lat * cos_lon   # d(Down)/dX
    R[:, 2, 1] = -cos_lat * sin_lon   # d(Down)/dY
    R[:, 2, 2] = -sin_lat             # d(Down)/dZ

    # --- Rotate gravity vector ---
    g_ecef = np.stack([gx, gy, gz], axis=1)  # (N, 3)
    g_ned = np.einsum('nij,nj->ni', R, g_ecef)  # (N, 3)
    gN, gE, gD = g_ned[:, 0], g_ned[:, 1], g_ned[:, 2]

    # --- Rotate GGT tensor: T_ned = R @ T_ecef @ R^T ---
    # Assemble full tensor (N, 3, 3)
    T_ecef = np.empty((N, 3, 3), dtype=np.float64)
    T_ecef[:, 0, 0] = Txx
    T_ecef[:, 0, 1] = Txy
    T_ecef[:, 0, 2] = Txz
    T_ecef[:, 1, 0] = Txy
    T_ecef[:, 1, 1] = Tyy
    T_ecef[:, 1, 2] = Tyz
    T_ecef[:, 2, 0] = Txz
    T_ecef[:, 2, 1] = Tyz
    T_ecef[:, 2, 2] = Tzz

    # Perform rotation: T' = R T R^T
    T_ned = np.einsum('nij,njk,nlk->nil', R, T_ecef, R)  # Note: R^T -> index order (nlk)

    # Extract unique components (tensor is symmetric)
    TNN = T_ned[:, 0, 0]
    TNE = T_ned[:, 0, 1]
    TND = T_ned[:, 0, 2]
    TEE = T_ned[:, 1, 1]
    TED = T_ned[:, 1, 2]
    TDD = T_ned[:, 2, 2]

    return gN, gE, gD, TNN, TNE, TND, TEE, TED, TDD


@njit(parallel=True, fastmath=True, nogil=True)
def green_third_identity_potential_numba(
    face_centers,   # (N, 3) in km
    face_normals,   # (N, 3) unitless (outward-pointing)
    face_areas,     # (N,) in km^2
    V_face,         # (N,) gravitational potential in m^2/s^2
    g_face,         # (N, 3) gravity vector in mGal
    eval_points     # (M, 3) evaluation points in km
):
    """
    Numerically evaluate Green's third identity to compute the gravitational potential 
    at external points using surface integrals over a polyhedral body (e.g., asteroid EROS).

    The identity for a harmonic function V (i.e., Laplacian of V is zero outside the mass) is:
    
        V(r0) = (1/(4*pi)) * surface_integral [
            V(r) * d/dn (1/|r - r0|) - (1/|r - r0|) * dV/dn(r)
        ] dS

    Since gravity vector g = grad(V), the normal derivative is dV/dn = g dot n.
    Substituting this gives:

        V(r0) = (1/(4*pi)) * surface_integral [
            V(r) * (R dot n) / R^3 - (g(r) dot n) / R
        ] dS,

    where R = r0 - r, R = |R|, and n is the outward unit normal.

    This implementation uses a piecewise-constant (face-center) quadrature:
    each triangular face contributes its center value times its area.

    Unit Convention (user-managed consistency):
    - Positions (face_centers, eval_points): kilometers (km)
    - Face areas: square kilometers (km^2)
    - Potential V_face: meters^2 / seconds^2 (m^2/s^2)
    - Gravity vector g_face: milligals (mGal)
    

    Parameters
    ----------
    face_centers : ndarray, shape (N, 3)
        Center coordinates of mesh faces in km.
    face_normals : ndarray, shape (N, 3)
        Outward-pointing unit normal vectors (dimensionless).
    face_areas : ndarray, shape (N,)
        Area of each face in km^2.
    V_face : ndarray, shape (N,)
        Gravitational potential at face centers in m^2/s^2.
    g_face : ndarray, shape (N, 3)
        Gravitational acceleration vector at face centers in mGal.
    eval_points : ndarray, shape (M, 3)
        Points where potential is to be computed, in km.

    Returns
    -------
    V_green : ndarray, shape (M,)
        Reconstructed gravitational potential at evaluation points in m^2/s^2.

    Notes
    -----
    - Valid only for evaluation points outside the body (where Laplacian V = 0).
    - Assumes mesh is closed and consistently oriented (outward normals).
    - Accuracy limited by face-center quadrature; suitable for validation.
    - Accelerated with Numba: parallelized over evaluation points.
    """
    N = face_centers.shape[0]
    M = eval_points.shape[0]
    inv_4pi = 1.0 / (4.0 * np.pi)

    V_green = np.empty(M, dtype=np.float64)

    # Parallel loop over evaluation points
    for i in prange(M):
        r0x = eval_points[i, 0]
        r0y = eval_points[i, 1]
        r0z = eval_points[i, 2]

        integral = 0.0

        # Loop over all faces to accumulate surface integral
        for j in range(N):
            # Vector from source (face center) to field point
            dx = r0x - face_centers[j, 0]
            dy = r0y - face_centers[j, 1]
            dz = r0z - face_centers[j, 2]

            # Squared and actual distance
            R2 = dx*dx + dy*dy + dz*dz
            R = np.sqrt(R2)
            R3 = R2 * R  # R^3

            # Normal components at face j
            nx, ny, nz = face_normals[j, 0], face_normals[j, 1], face_normals[j, 2]

            # Term 1: V * d(1/R)/dn = V * (R dot n) / R^3
            R_dot_n = dx * nx + dy * ny + dz * nz
            term1 = V_face[j] * (R_dot_n / R3) * face_areas[j] 

            # Term 2: - (1/R) * dV/dn = - (1/R) * (g dot n)  [since dV/dn = g dot n]
            gx, gy, gz = g_face[j, 0], g_face[j, 1], g_face[j, 2]
            g_dot_n = gx * nx + gy * ny + gz * nz
            term2 = - 1.e-2 * g_dot_n / R * face_areas[j] 

            # Combine terms and weight by face area
            integrand = term1 + term2
            integral += integrand 

        # Apply Green's identity scaling factor
        V_green[i] = inv_4pi * integral

    return V_green
