import numpy as np
import time

# Parameters
M = 10000
N = 10000
Ne = 15000  # number of column indices to select

# Random data
A = np.random.rand(M, 3)
B = np.random.rand(3, N)

# Random column indices (values in [0, N))
col_indices = np.random.randint(0, N, size=Ne)
col_indices_sorted = np.sort(col_indices)

# --- Timing matrix multiplication ---
start = time.perf_counter()
C = A @ B  # shape (M, N)
matmul_time = time.perf_counter() - start

# --- Timing indexing ---
start = time.perf_counter()
selected = C[col_indices_sorted,:]  # shape (M, Ne)
indexing_time = time.perf_counter() - start

print(f"Matrix multiplication ({M}x3 @ 3x{N}): {matmul_time:.6f} seconds")
print(f"Indexing {Ne} columns from ({M}x{N}) matrix: {indexing_time:.6f} seconds")