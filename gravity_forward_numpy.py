# %%
# ! # Setup
import numpy as np;
# %%
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

def VecWerSch(P, Q, If, rho):
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

    ######## * Constants
    G           = 6.67430e-11;
    km2m        = 1.e3;
    si2mg       = 1.e5;
    si2eot      = 1.e9;
    ######## * Computation of <FACE> contribution
    #### * Polyhedron geometry
    A           = Q[If[:,0],:];                                          # (Nf,3)
    B           = Q[If[:,1],:];                                          # (Nf,3)
    C           = Q[If[:,2],:];                                          # (Nf,3)
    rA2         = np.sum(A**2, axis=1);                                  # (Nf,)
    rB2         = np.sum(B**2, axis=1);                                  # (Nf,)
    rC2         = np.sum(C**2, axis=1);                                  # (Nf,)
    rA_dot_rB   = np.sum(A*B, axis=1);                                   # (Nf,)
    rB_dot_rC   = np.sum(B*C, axis=1);                                   # (Nf,)
    rC_dot_rA   = np.sum(C*A, axis=1);                                   # (Nf,)
    nf          = np.cross(B-A, C-B);                                    # (Nf,3)
    mag_nf      = np.sqrt(np.sum(nf**2, axis=1, keepdims=True));         # (Nf,1)
    hnf         = nf / mag_nf;                                           # (Nf,3)
    #### * Field-Source interaction
    rP2         = np.sum(P**2, axis=1);                                  # (M,)
    rP_dot_rQ   = Q @ P.T                                                # (Nv,3) @ (3,M) --> (Nv,M)
    rP_dot_rA   = rP_dot_rQ[If[:,0],:];                                  # (Nv,M) --> (Nf,M)
    rP_dot_rB   = rP_dot_rQ[If[:,1],:];                                  # (Nv,M) --> (Nf,M)
    rP_dot_rC   = rP_dot_rQ[If[:,2],:];                                  # (Nv,M) --> (Nf,M)
    rPA         = np.sqrt(rP2 + rA2[:,None] - 2 * rP_dot_rA);            # (M,), (Nf,1), (Nf,M) --> (Nf,M)
    rPB         = np.sqrt(rP2 + rB2[:,None] - 2 * rP_dot_rB);            # (M,), (Nf,1), (Nf,M) --> (Nf,M)
    rPC         = np.sqrt(rP2 + rC2[:,None] - 2 * rP_dot_rC);            # (M,), (Nf,1), (Nf,M) --> (Nf,M)
    rPA_dot_rPB = rP2 + rA_dot_rB[:,None] - (rP_dot_rA + rP_dot_rB);     # (M,), (Nf,1), (Nf,M) --> (Nf,M)
    rPB_dot_rPC = rP2 + rB_dot_rC[:,None] - (rP_dot_rB + rP_dot_rC);     # (M,), (Nf,1), (Nf,M) --> (Nf,M)
    rPC_dot_rPA = rP2 + rC_dot_rA[:,None] - (rP_dot_rC + rP_dot_rA);     # (M,), (Nf,1), (Nf,M) --> (Nf,M)
    denom       = rPA*rPB*rPC + rPA*rPB_dot_rPC \
                  + rPB*rPC_dot_rPA + rPC*rPA_dot_rPB;                   # (Nf,M)
    rP_dot_hnf  = hnf @ P.T;                                             # (Nf,3) @ (3,M) --> (Nf,M)
    mixProd     = np.sum(A*nf, axis=1, keepdims=True) \
                  - mag_nf * rP_dot_hnf;                                 # (Nf,1), (Nf,M) --> (Nf,M)
    wf          = 2 * np.arctan2(mixProd, denom);                        # (Nf,M)
    rf_dot_hnf  = mixProd / mag_nf;                                      # (Nf,M)
    Vf          = np.sum(rf_dot_hnf * rf_dot_hnf * wf, axis=0);          # (M,)
    gx_f, gy_f, gz_f = \
        np.sum(hnf[:,:,None] * (rf_dot_hnf * wf)[:,None,:], axis=0);
    Txx_f, Tyy_f, Tzz_f, Txy_f, Txz_f, Tyz_f = (
        np.sum(hnf[:, [i]] * hnf[:, [j]] * wf, axis=0)
        for i, j in [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
    )
    ######## * Computation of <EDGE> contribution
    #### * Polyhedron geometry
    Ie          = Faces2Edges(If);                                       # (Ne,4)
    A           = Q[Ie[:,0],:];                                          # (Ne,3)
    B           = Q[Ie[:,1],:];                                          # (Ne,3)
    rA2         = np.sum(A**2, axis=1);                                  # (Ne,)
    rB2         = np.sum(B**2, axis=1);                                  # (Ne,)
    eAB         = B - A;                                                 # (Ne,3)
    mag_eAB     = np.sqrt(np.sum(eAB**2, axis=1, keepdims=True));        # (Ne,1)
    heAB        = eAB / mag_eAB;                                         # (Ne,3)
    hnABC       = hnf[Ie[:,2],:];                                        # (Nf,3) --> (Ne,3)
    hnBAD       = hnf[Ie[:,3],:];                                        # (Nf,3) --> (Ne,3)
    hnAB        = np.cross( heAB, hnABC);                                # (Nf,3) --> (Ne,3)
    hnBA        = np.cross(-heAB, hnBAD);                                # (Nf,3) --> (Ne,3) 
    #### * Field-Source interaction
    rPA         = np.sqrt(rP2 + rA2[:,None] - 2 * rP_dot_rQ[Ie[:,0],:]); # (M,1), (Ne,), (Ne,M) --> (Ne,M)
    rPB         = np.sqrt(rP2 + rB2[:,None] - 2 * rP_dot_rQ[Ie[:,1],:]); # (M,1), (Ne,), (Ne,M) --> (Ne,M)
    Le          = np.log((rPA + rPB + mag_eAB) / (rPA + rPB - mag_eAB)); # (Ne,M)
    re_dot_hnf1 = np.sum(A*hnABC, axis=1, keepdims=True) \
                  - rP_dot_hnf[Ie[:,2],:];                               # (Ne,1), (Ne,M) --> (Ne,M)   
    re_dot_hnf2 = np.sum(A*hnBAD, axis=1, keepdims=True) \
                  - rP_dot_hnf[Ie[:,3],:];                               # (Ne,1), (Ne,M) --> (Ne,M)
    re_dot_hne1 = np.sum(A*hnAB, axis=1, keepdims=True) \
                  - hnAB @ P.T;                                          # (Ne,1), (Ne,M) --> (Ne,M)      
    re_dot_hne2 = np.sum(A*hnBA, axis=1, keepdims=True) \
                  - hnBA @ P.T;                                          # (Ne,1), (Ne,M) --> (Ne,M)   
    rEr         = re_dot_hnf1 * re_dot_hne1 + re_dot_hnf2 * re_dot_hne2; # (Ne,M)
    Ve          = np.sum(rEr * Le, axis=0);                              # (M,)
    gx_e, gy_e, gz_e = np.sum((hnABC[:,:,None] * re_dot_hne1[:,None,:] 
                + hnBAD[:,:,None] * re_dot_hne2[:,None,:]) 
                * Le[:,None,:], axis=0);
    Txx_e, Tyy_e, Tzz_e, Txy_e, Txz_e, Tyz_e = (
        np.sum((hnABC[:,[i]] * hnAB[:,[j]] + hnBAD[:,[i]] * hnBA[:,[j]]) * Le, axis=0)
        for i, j in [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
    )
    ######## Summation
    V = 0.5 * km2m**2 * G * rho * (Ve - Vf);
    gx = km2m * si2mg * G * rho * (gx_f - gx_e);
    gy = km2m * si2mg * G * rho * (gy_f - gy_e);
    gz = km2m * si2mg * G * rho * (gz_f - gz_e);
    gz = km2m * si2mg * G * rho * (gz_f - gz_e);
    Txx =  si2eot* G * rho * (Txx_e - Txx_f);
    Tyy =  si2eot* G * rho * (Tyy_e - Tyy_f);
    Tzz =  si2eot* G * rho * (Tzz_e - Tzz_f);
    Txy =  si2eot* G * rho * (Txy_e - Txy_f);
    Txz =  si2eot* G * rho * (Txz_e - Txz_f);
    Tyz =  si2eot* G * rho * (Tyz_e - Tyz_f);
    return V, gx, gy, gz, Txx, Tyy, Tzz, Txy, Txz, Tyz;

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
    
    return dV, gx, gy, gz, Txx, Tyy, Tzz, Txy, Txz, Tyz;

def spherical_edge_length_range(verts, faces):
    """
    Compute the minimum and maximum spherical (great-circle) edge lengths in a triangular mesh 
    with vertices on the unit sphere.

    Extracts all unique undirected edges from the triangular faces, computes the angular 
    distance (in radians) between each pair of connected vertices, and returns the smallest 
    and largest such distances.

    Parameters
    ----------
    verts : np.ndarray of shape (V, 3)
        Vertex coordinates on the **unit sphere** (each row is a 3D unit vector).
    faces : np.ndarray of shape (F, 3)
        Triangular face definitions as integer indices into `verts`.

    Returns
    -------
    min_dist : float
        Minimum spherical edge length in radians.
    max_dist : float
        Maximum spherical edge length in radians.

    Notes
    -----
    - Assumes input vertices are normalized (||v|| = 1). If not, normalize before calling.
    - Dot products are clamped to [-1, 1] to ensure numerical stability in arccos.
    - Only unique edges are considered (each edge counted once, regardless of face sharing).
    """
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ]);
    edges_sorted = np.sort(edges, axis=1);
    edges_unique = np.unique(edges_sorted, axis=0);
    v0 = verts[edges_unique[:, 0]];
    v1 = verts[edges_unique[:, 1]];
    # dots = np.einsum('ij,ij->i', v0, v1);
    # dots = np.clip(dots, -1.0, 1.0)
    # dist = np.arccos(dots)
    chord = np.linalg.norm(v1 - v0, axis=1)
    chord = np.clip(chord, 0.0, 2.0)
    dist = 2.0 * np.arcsin(0.5 * chord)
    return np.min(dist), np.max(dist)