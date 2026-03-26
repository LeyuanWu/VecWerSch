import numpy as np
import pyvista as pv
import time
import matplotlib.pyplot as plt
from gravity_forward_numba import _Faces2Edges, _compute_gravity_numba, _compute_gravity_numba_onthefly

# ------------------------------------------------------------------
# ASSUMPTION: Your two functions VecWerSch_numba and VecWerSch_numba_onthefly
# are already defined above this code.
# If not, paste them here first!
# ------------------------------------------------------------------

# ==================================================================
# MEMORY-ESTIMATING WRAPPERS
# ==================================================================

def estimate_memory_VecWerSch_numba(P, Q, If, rho):
    """Run VecWerSch_numba and return (runtime, estimated_memory_bytes)"""
    t0 = time.perf_counter()
    
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

    # Estimate memory: sum all intermediate arrays
    mem_bytes = (
        A_f.nbytes + B_f.nbytes + C_f.nbytes +
        AB.nbytes + BC.nbytes + nf.nbytes + mag_nf.nbytes + hnf.nbytes + A_dot_nf.nbytes +
        Ie.nbytes + A_e.nbytes + B_e.nbytes + eAB.nbytes + mag_eAB.nbytes +
        hn_pos.nbytes + hn_neg.nbytes + heAB.nbytes +
        hnAB.nbytes + hnBA.nbytes +
        A_dot_hn_pos.nbytes + A_dot_hn_neg.nbytes + A_dot_hnAB.nbytes + A_dot_hnBA.nbytes
    )

    # Now call the numba function (we don't care about output, just timing)
    G = 6.67430e-11
    km2m = 1.e3
    si2mg = 1.e5
    si2eot = 1.e9

    _compute_gravity_numba(
        P, A_f, B_f, C_f, hnf, mag_nf, A_dot_nf,
        A_e, B_e, mag_eAB,
        hn_pos, hn_neg, hnAB, hnBA,
        A_dot_hn_pos, A_dot_hn_neg, A_dot_hnAB, A_dot_hnBA,
        G, rho, km2m, si2mg, si2eot
    )

    t1 = time.perf_counter()
    return t1 - t0, mem_bytes


def estimate_memory_VecWerSch_numba_onthefly(P, Q, If, rho):
    """Run VecWerSch_numba_onthefly and return (runtime, estimated_memory_bytes)"""
    t0 = time.perf_counter()

    # --- Precompute face normals (essential) ---
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

    # --- Edge topology only (no geometry) ---
    Ie = _Faces2Edges(If)  # (Ne, 4): [v0, v1, f_ccw, f_cw]

    # Estimate memory: only store what's kept in memory before numba call
    mem_bytes = (
        A_f.nbytes + B_f.nbytes + C_f.nbytes +
        AB.nbytes + BC.nbytes + nf.nbytes + mag_nf.nbytes + hnf.nbytes + A_dot_nf.nbytes +
        Ie.nbytes
    )

    # Call numba function
    G = 6.67430e-11
    km2m = 1.e3
    si2mg = 1.e5
    si2eot = 1.e9

    _compute_gravity_numba_onthefly(
        P, Q, If, Ie, hnf, mag_nf, A_dot_nf,
        G, rho, km2m, si2mg, si2eot
    )

    t1 = time.perf_counter()
    return t1 - t0, mem_bytes


# ==================================================================
# BENCHMARK LOOP
# ==================================================================

def run_benchmarks(max_level=12, n_obs=100):
    levels = list(range(1, max_level + 1))
    rho = 2670.0  # kg/m^3

    # Observation points
    np.random.seed(0)
    P_base = np.random.randn(n_obs, 3)
    P_base = P_base / np.linalg.norm(P_base, axis=1, keepdims=True) * 1.1

    results = []

    for level in levels:
        print(f"\n--- Level {level} ---")
        try:
            mesh = pv.Icosphere(nsub=level)
            Q = mesh.points.astype(np.float64)
            If = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
            P = P_base.copy()

            print(f"Mesh: {Q.shape[0]:,} vertices, {If.shape[0]:,} faces")

            # On-the-fly
            print("  Running on-the-fly...")
            try:
                time_oft, mem_oft = estimate_memory_VecWerSch_numba_onthefly(P, Q, If, rho)
            except Exception as e:
                print(f"    Failed (on-the-fly): {e}")
                time_oft, mem_oft = None, None

            # Full precompute
            print("  Running full precompute...")
            try:
                time_full, mem_full = estimate_memory_VecWerSch_numba(P, Q, If, rho)
            except Exception as e:
                print(f"    Failed (full): {e}")
                time_full, mem_full = None, None

            results.append({
                'level': level,
                'faces': If.shape[0],
                'vertices': Q.shape[0],
                'time_full': time_full,
                'mem_full_gb': mem_full / (1024**3) if mem_full else None,
                'time_oft': time_oft,
                'mem_oft_gb': mem_oft / (1024**3) if mem_oft else None,
            })

        except Exception as e:
            print(f"Mesh creation failed at level {level}: {e}")
            results.append({
                'level': level, 'faces': 0, 'vertices': 0,
                'time_full': None, 'mem_full_gb': None,
                'time_oft': None, 'mem_oft_gb': None,
            })
            break

    return results


# ==================================================================
# PRINT TABLE (no tabulate)
# ==================================================================

def print_table(results):
    headers = ["Level", "Faces", "Vertices", "Time Full (s)", "Mem Full (GB)", "Time OnFly (s)", "Mem OnFly (GB)"]
    
    rows = []
    for r in results:
        row = [
            str(r['level']),
            f"{r['faces']:,}",
            f"{r['vertices']:,}",
            f"{r['time_full']:.2f}" if r['time_full'] is not None else "—",
            f"{r['mem_full_gb']:.2f}" if r['mem_full_gb'] is not None else "—",
            f"{r['time_oft']:.2f}" if r['time_oft'] is not None else "—",
            f"{r['mem_oft_gb']:.2f}" if r['mem_oft_gb'] is not None else "—",
        ]
        rows.append(row)
    
    col_widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) + 2 for i in range(len(headers))]
    
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print("\n" + "=" * len(header_line))
    print("BENCHMARK RESULTS (Estimated Memory)")
    print("=" * len(header_line))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("".join(item.ljust(w) for item, w in zip(row, col_widths)))


# ==================================================================
# PLOTTING
# ==================================================================

def plot_results(results):
    levels = [r['level'] for r in results]
    faces = [r['faces'] for r in results]

    time_full = [r['time_full'] for r in results]
    time_oft = [r['time_oft'] for r in results]
    mem_full = [r['mem_full_gb'] for r in results]
    mem_oft = [r['mem_oft_gb'] for r in results]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Runtime vs Level
    axs[0,0].plot(levels, time_full, 'o-', label='Full Precompute')
    axs[0,0].plot(levels, time_oft, 's--', label='On-the-Fly')
    axs[0,0].set_xlabel('Subdivision Level')
    axs[0,0].set_ylabel('Runtime (s)')
    axs[0,0].set_title('Runtime vs Level')
    axs[0,0].legend()
    axs[0,0].grid(True)

    # Memory vs Level
    axs[0,1].plot(levels, mem_full, 'o-', label='Full Precompute')
    axs[0,1].plot(levels, mem_oft, 's--', label='On-the-Fly')
    axs[0,1].set_xlabel('Subdivision Level')
    axs[0,1].set_ylabel('Estimated Memory (GB)')
    axs[0,1].set_title('Estimated Memory vs Level')
    axs[0,1].legend()
    axs[0,1].grid(True)

    # Runtime vs Faces (log-log)
    valid_full_t = [i for i,t in enumerate(time_full) if t is not None]
    valid_oft_t = [i for i,t in enumerate(time_oft) if t is not None]
    axs[1,0].loglog([faces[i] for i in valid_full_t], [time_full[i] for i in valid_full_t], 'o-', label='Full')
    axs[1,0].loglog([faces[i] for i in valid_oft_t], [time_oft[i] for i in valid_oft_t], 's--', label='On-the-Fly')
    axs[1,0].set_xlabel('Number of Faces')
    axs[1,0].set_ylabel('Runtime (s)')
    axs[1,0].set_title('Runtime vs Faces (log-log)')
    axs[1,0].legend()
    axs[1,0].grid(True)

    # Memory vs Faces (log-log)
    valid_full_m = [i for i,m in enumerate(mem_full) if m is not None]
    valid_oft_m = [i for i,m in enumerate(mem_oft) if m is not None]
    axs[1,1].loglog([faces[i] for i in valid_full_m], [mem_full[i] for i in valid_full_m], 'o-', label='Full')
    axs[1,1].loglog([faces[i] for i in valid_oft_m], [mem_oft[i] for i in valid_oft_m], 's--', label='On-the-Fly')
    axs[1,1].set_xlabel('Number of Faces')
    axs[1,1].set_ylabel('Memory (GB)')
    axs[1,1].set_title('Memory vs Faces (log-log)')
    axs[1,1].legend()
    axs[1,1].grid(True)

    plt.tight_layout()
    plt.savefig("benchmark_estimated.png", dpi=150)
    plt.show()


# ==================================================================
# RUN
# ==================================================================

if __name__ == "__main__":
    results = run_benchmarks(max_level=8, n_obs=10000)
    print_table(results)
    plot_results(results)