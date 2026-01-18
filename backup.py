# %%
# ! # Setup
import numpy as np;
from gravity_forward import *;

def VecWerSch_ver0(P, Q, If, rho):
    """_summary_

    Args:
        P (_type_): _description_
        Q (_type_): _description_
        If (_type_): _description_
        rho (_type_): _description_

    Returns:
        _type_: _description_
    """
    G = 6.67430e-11;
    km2m = 1.e3;
    ######## * Computation of wf: N = Nf
    A       = Q[If[:,0],:];                                          # (N,3)
    B       = Q[If[:,1],:];                                          # (N,3)
    C       = Q[If[:,2],:];                                          # (N,3)
    rP2     = np.sum(P**2, axis=1);                                  # (M,)
    rA2     = np.sum(A**2, axis=1);                                  # (N,)
    rB2     = np.sum(B**2, axis=1);                                  # (N,)
    rC2     = np.sum(C**2, axis=1);                                  # (N,)
    rA_rB   = np.sum(A*B, axis=1);                                   # (N,)
    rB_rC   = np.sum(B*C, axis=1);                                   # (N,)
    rC_rA   = np.sum(C*A, axis=1);                                   # (N,)
    rP_rA   = P @ A.T;                                               # (M,3) @ (3,N) --> (M,N)
    rP_rB   = P @ B.T;                                               # (M,3) @ (3,N) --> (M,N)
    rP_rC   = P @ C.T;                                               # (M,3) @ (3,N) --> (M,N)
    rPA     = np.sqrt(rP2[:,np.newaxis] + rA2 - 2 * rP_rA);          # (M,1), (N,), (M,N) --> (M,N)
    rPB     = np.sqrt(rP2[:,np.newaxis] + rB2 - 2 * rP_rB);          # (M,1), (N,), (M,N) --> (M,N)
    rPC     = np.sqrt(rP2[:,np.newaxis] + rC2 - 2 * rP_rC);          # (M,1), (N,), (M,N) --> (M,N)
    rPA_rPB = rP2[:,np.newaxis] + rA_rB - (rP_rA + rP_rB);           # (M,1), (N,), (M,N) --> (M,N)
    rPB_rPC = rP2[:,np.newaxis] + rB_rC - (rP_rB + rP_rC);           # (M,1), (N,), (M,N) --> (M,N)
    rPC_rPA = rP2[:,np.newaxis] + rC_rA - (rP_rC + rP_rA);           # (M,1), (N,), (M,N) --> (M,N)
    nf      = np.cross(B-A, C-B);                                    # (N,3)                   
    mag_nf  = np.sqrt(np.sum(nf**2, axis=1));                        # (N,)
    hnf     = nf / mag_nf[:, np.newaxis];                            # (N,3)
    mixProd = np.sum(A*nf, axis=1) - P @ nf.T;                       # (N,), (M,N) --> (M,N)
    denom   = rPA*rPB*rPC + rPA*rPB_rPC + rPB*rPC_rPA + rPC*rPA_rPB; # (M,N)
    wf      = 2 * np.arctan2(mixProd, denom);                        # (M,N)
    rf_hnf  = mixProd / mag_nf;                                      # (M,N)
    Vf      = np.sum(rf_hnf * rf_hnf * wf, axis=1);                  # (M,)
    ######## * Computation of Le: N = Ne = 1.5 * Nf
    Ie      = Faces2Edges(If);
    A       = Q[Ie[:,0],:];                                          # (N,3)
    B       = Q[Ie[:,1],:];                                          # (N,3)
    mag_eAB = np.sqrt(np.sum((B-A)**2, axis=1));                     # (N,)
    heAB    = (B - A)/mag_eAB[:, np.newaxis];
    rP2     = np.sum(P**2, axis=1);                                  # (M,)
    rA2     = np.sum(A**2, axis=1);                                  # (N,)
    rB2     = np.sum(B**2, axis=1);                                  # (N,)
    rP_rA   = P @ A.T;                                               # (M,3) @ (3,N) --> (M,N)
    rP_rB   = P @ B.T;                                               # (M,3) @ (3,N) --> (M,N)
    rPA     = np.sqrt(rP2[:,np.newaxis] + rA2 - 2 * rP_rA);          # (M,1), (N,), (M,N) --> (M,N)
    rPB     = np.sqrt(rP2[:,np.newaxis] + rB2 - 2 * rP_rB);          # (M,1), (N,), (M,N) --> (M,N)
    Le      = np.log((rPA + rPB + mag_eAB) / (rPA + rPB - mag_eAB));         # (M,N)
    hnABC   = hnf[Ie[:,2],:];
    hnAB    = np.cross(heAB, hnABC);
    hnBAD   = hnf[Ie[:,3],:];
    hnBA    = np.cross(-heAB, hnBAD);
    re_hnABC = np.sum(A*hnABC, axis=1) - P @ hnABC.T;
    re_hnAB = np.sum(A*hnAB, axis=1) - P @ hnAB.T;
    re_hnBAD = np.sum(A*hnBAD, axis=1) - P @ hnBAD.T;
    re_hnBA = np.sum(A*hnBA, axis=1) - P @ hnBA.T;
    Ve = np.sum((re_hnABC * re_hnAB + re_hnBAD * re_hnBA) * Le, axis=1);
    ######## Summation
    V = 0.5 * G * rho * (Ve - Vf) * km2m**2;
    return V;