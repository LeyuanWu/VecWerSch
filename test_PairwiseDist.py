import numpy as np
import timeit

def pairwise_dists_looped(x, y):
    """  Computing pairwise distances using for-loops

     Parameters
     ----------
     x : numpy.ndarray, shape=(M, D)
     y : numpy.ndarray, shape=(N, D)

     Returns
     -------
     numpy.ndarray, shape=(M, N)
         The Euclidean distance between each pair of
         rows between `x` and `y`."""
    # `dists[i, j]` will store the Euclidean
    # distance between  `x[i]` and `y[j]`
    dists = np.empty((x.shape[0], y.shape[0]))

    for i, row_x in enumerate(x):     # loops over rows of `x`
        for j, row_y in enumerate(y): # loops over rows of `y`
            # Subtract corresponding entries of the rows,
            # squares each difference, and then sums them. This
            # exactly matches our equation for Euclidean
            # distance (we will do the square root later)
            dists[i, j] = np.sum((row_x - row_y)**2)

    # we still need to take the square root of
    # each of our numbers
    return np.sqrt(dists)

def pairwise_dists(x, y):
    """ Computing pairwise distances using memory-efficient
    vectorization.

    Parameters
    ----------
    x : numpy.ndarray, shape=(M, D)
    y : numpy.ndarray, shape=(N, D)

    Returns
    -------
    numpy.ndarray, shape=(M, N)
        The Euclidean distance between each pair of
        rows between `x` and `y`."""
    sqr_dists = -2 * np.matmul(x, y.T)
    sqr_dists +=  np.sum(x**2, axis=1)[:, np.newaxis]
    sqr_dists += np.sum(y**2, axis=1)
    return  np.sqrt(np.clip(sqr_dists, a_min=0, a_max=None))

def pairwise_cross_einsum(P, Q):
    """
    More memory-efficient using einsum
    """
    M, N = P.shape[0], Q.shape[0]
    
    # Initialize output array
    result = np.zeros((M, N, 3))
    
    # Compute each component using einsum
    # x-component: u_y*v_z - u_z*v_y
    result[..., 0] = np.einsum('ij,kj->ik', P[:, [1]], Q[:, [2]]) - \
                     np.einsum('ij,kj->ik', P[:, [2]], Q[:, [1]])
    
    # y-component: u_z*v_x - u_x*v_z
    result[..., 1] = np.einsum('ij,kj->ik', P[:, [2]], Q[:, [0]]) - \
                     np.einsum('ij,kj->ik', P[:, [0]], Q[:, [2]])
    
    # z-component: u_x*v_y - u_y*v_x
    result[..., 2] = np.einsum('ij,kj->ik', P[:, [0]], Q[:, [1]]) - \
                     np.einsum('ij,kj->ik', P[:, [1]], Q[:, [0]])
    
    return result

# %%
# ! Test
M = 10000; N=10000;
P = np.random.normal(0., 1., size=(M, 3));
Q = np.random.normal(0., 1., size=(N, 3));
# D1 = pairwise_dists_looped(P, Q);
D2 = pairwise_dists(P, Q);

# --- Timing setup ---
number = 3  # Number of times to run each function
# time1 = timeit.timeit(lambda: pairwise_dists_looped(P, Q), number=number)
time2 = timeit.timeit(lambda: pairwise_dists(P, Q), number=number)
time3 = timeit.timeit(lambda: pairwise_cross_einsum(P, Q), number=number)
# print(f"For-loops method:   {time1:.4f} seconds ({time1/number:.4f} per call)")
print(f"Memory-efficient vectorization: {time2:.4f} seconds ({time2/number:.4f} per call)")
print(f"Memory-efficient cross-product: {time3:.4f} seconds ({time3/number:.4f} per call)")
