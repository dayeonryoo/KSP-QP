"""
Validation suite for mpc_generator.py. Run directly: python3 validate_mpc_generator.py

The generator builds the M-vehicle platoon chain system.

Checks, in order:
  1. Smoke tests: dimensions, the M-independent band bound, platoon chain
     structure (including the M=1 reduction to a single double integrator),
     x_0 block bounds always +-inf.
  2. Objective cross-check: KSP-QP vs a raw OSQP solve, on the (feasible,
     near-bound) default xbar0 at several M.
  3. LQR ground-truth check: with P the exact DARE solution (a fixed point of
     the backward Riccati recursion), the finite-horizon solution equals the
     infinite-horizon LQR solution for any N>=1 when no inequality is active.
     Checked two ways:
       - obj_val == xbar'*P*xbar + obj_const (any solver)
       - u_0* == -K @ xbar (on KSP-QP's solution x)
  4. Feasibility check across benchmark_mpc.py's default (M, N) sweep GRID,
     imported from that module (sweep_configs) so the two cannot drift apart.
     Runs the whole grid by default; --max-M caps it for a quicker run.
"""

import sys
import argparse

import numpy as np
import scipy.sparse as sp
import osqp
import ksp_qp_bind
import mpc_generator as mg
from benchmark_mpc import sweep_configs

TOL = 1e-5
FAILURES = []

# M values for the structural smoke tests (cheap -- no solve) and for the
# solve-based cross-checks (kept small: sections 2 and 3 each run a raw OSQP
# solve to 1e-9, which is the expensive part).
SMOKE_M = [1, 3, 5, 20, 50]
CROSS_CHECK_M = [1, 5, 20]

# Every constraint row touches at most this many variables, independently of M
# and N: the sparsity benchmark_mpc.py relies on (the generic 2*n_x+n_u = 5M
# bound would be vacuous at large M).
#
# Derivation: Ac's only nonzero rows are the d rows, and those hit only v
# columns, while the v rows are identically zero -- so Ac @ Ac == 0 and the ZOH
# discretization truncates exactly: Ad = I + Ts*Ac, Bd = Ts*Bc + (Ts^2/2)*Ac@Bc.
# The widest row is then a d_i dynamics row: 3 entries from Ad (d_i, v_i,
# v_{i-1}), 2 from Bd (u_i, u_{i-1}), and 1 from the x_{k+1} identity block.
MAX_ROW_NNZ = 6


def _check(name: str, cond: bool, detail: str = "") -> None:
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def solve_raw_osqp(pdd, tol=1e-9):
    n = pdd["n"]
    Q = sp.csc_matrix((pdd["Q_data"], pdd["Q_indices"], pdd["Q_indptr"]), shape=pdd["Q_shape"])
    A = sp.csc_matrix((pdd["A_data"], pdd["A_indices"], pdd["A_indptr"]), shape=pdd["A_shape"])
    B = sp.csc_matrix((pdd["B_data"], pdd["B_indices"], pdd["B_indptr"]), shape=pdd["B_shape"])
    INF = 1e30
    C = sp.vstack([A, B, sp.eye(n)], format="csc")
    bmin = np.concatenate([pdd["b"], np.clip(pdd["lw"], -INF, INF), np.clip(pdd["lx"], -INF, INF)])
    bmax = np.concatenate([pdd["b"], np.clip(pdd["uw"], -INF, INF), np.clip(pdd["ux"], -INF, INF)])
    prob = osqp.OSQP()
    prob.setup(Q, pdd["c"], C, bmin, bmax, eps_abs=tol, eps_rel=tol, max_iter=200_000, verbose=False)
    res = prob.solve()
    return res.x, res.info.obj_val + pdd["obj_const"], res.info.status


def _chain_structure_checks(M: int, sysm) -> None:
    """The platoon chain itself: d_i' = v_{i-1} - v_i, v_i' = u_i, and nothing
    else -- no vehicle couples to anything but its immediate predecessor."""
    name, Ac, Bc = sysm.name, sysm.Ac, sysm.Bc

    self_ok = all(Ac[2 * i, 2 * i + 1] == -1.0 for i in range(M))
    up_ok = all(Ac[2 * i, 2 * (i - 1) + 1] == 1.0 for i in range(1, M))
    _check(f"{name}: d_i' = v_{{i-1}} - v_i", self_ok and up_ok)
    _check(f"{name}: v_i' = u_i", all(Bc[2 * i + 1, i] == 1.0 for i in range(M)))

    # Exactly 2M-1 = M self terms + (M-1) upstream couplings; anything more
    # would mean a vehicle sees past its predecessor.
    nnz_ac = int(np.count_nonzero(Ac))
    _check(f"{name}: Ac has exactly 2M-1 nonzeros (pure chain)", nnz_ac == 2 * M - 1,
           f"nnz={nnz_ac}, expected={2 * M - 1}")
    nnz_bc = int(np.count_nonzero(Bc))
    _check(f"{name}: Bc has exactly M nonzeros (one actuator per vehicle)", nnz_bc == M,
           f"nnz={nnz_bc}, expected={M}")

    # This is what makes MAX_ROW_NNZ hold (see its comment): zero v rows =>
    # Ac@Ac == 0 => the ZOH series terminates after the linear term.
    _check(f"{name}: v rows of Ac are zero, so Ad == I + Ts*Ac exactly",
           bool(np.all(Ac[1::2, :] == 0.0)) and np.allclose(Ac @ Ac, 0.0))


def smoke_tests() -> None:
    print("\n=== 1. Smoke tests ===")
    N = 10
    for M in SMOKE_M:
        sysm = mg.platoon_system(M)
        name, n_x, n_u = sysm.name, sysm.n_x, sysm.n_u
        pdd = mg.generate_mpc_qp(sysm, N, sysm.xbar0)

        _check(f"{name}: n_x == 2M and n_u == M", n_x == 2 * M and n_u == M,
               f"n_x={n_x} n_u={n_u}")
        _check(f"{name}: n matches (N+1)n_x+Nn_u", pdd["n"] == (N + 1) * n_x + N * n_u)
        _check(f"{name}: m matches (N+1)n_x", pdd["m"] == (N + 1) * n_x)
        _check(f"{name}: l == 0 (no B needed)", pdd["l"] == 0)

        A = sp.csc_matrix((pdd["A_data"], pdd["A_indices"], pdd["A_indptr"]), shape=pdd["A_shape"])
        max_row_nnz = int(np.diff(A.tocsr().indptr).max())
        _check(f"{name}: banded, max row nnz <= {MAX_ROW_NNZ} regardless of M",
               max_row_nnz <= MAX_ROW_NNZ, f"max_row_nnz={max_row_nnz}")

        lx, ux = pdd["lx"], pdd["ux"]
        _check(f"{name}: x_0 block is +-inf (never finite)",
               bool(np.all(np.isinf(lx[:n_x])) and np.all(np.isinf(ux[:n_x]))))

        _chain_structure_checks(M, sysm)

    # M=1 is a lone vehicle behind a constant-speed leader: d_1' = v_0 - v_1 = -v_1
    # (v_0 == 0 in deviation coordinates), v_1' = u_1. Note d' = -v, not +v: d is
    # a spacing error, so a positive velocity deviation drives it down.
    plat1 = mg.platoon_system(1)
    _check("platoon M=1 reduces to a double integrator in deviation coords (d'=-v, v'=u)",
           plat1.n_x == 2 and plat1.n_u == 1
           and np.array_equal(plat1.Ac, np.array([[0., -1.], [0., 0.]]))
           and np.array_equal(plat1.Bc, np.array([[0.], [1.]])),
           f"Ac={plat1.Ac.tolist()} Bc={plat1.Bc.tolist()}")


def objective_cross_check() -> None:
    print("\n=== 2. Objective cross-check: KSP-QP vs raw OSQP ===")
    for M in CROSS_CHECK_M:
        sysm = mg.platoon_system(M)
        pdd = mg.generate_mpc_qp(sysm, N=10, xbar=sysm.xbar0)
        res = ksp_qp_bind.solve_from_data(pdd, 1e-8, 1_000_000, 30.0)
        _, obj_osqp, status_osqp = solve_raw_osqp(pdd)
        rel_diff = abs(res["obj_val"] - obj_osqp) / max(1.0, abs(obj_osqp))
        _check(f"{sysm.name}: KSP-QP obj matches OSQP (rel_diff<1e-5)", rel_diff < 1e-5,
               f"ksp={res['obj_val']:.6f} osqp={obj_osqp:.6f} ({status_osqp}) rel_diff={rel_diff:.2e}")


def lqr_ground_truth_check() -> None:
    print("\n=== 3. LQR exact ground-truth check (relaxed xbar, N=10) ===")
    for M in CROSS_CHECK_M:
        sysm = mg.platoon_system(M)
        n_x, n_u = sysm.n_x, sysm.n_u
        xbar_small = sysm.xbar0 * 1e-3
        N = 10
        pdd = mg.generate_mpc_qp(sysm, N, xbar_small)
        res = ksp_qp_bind.solve_from_data(pdd, 1e-9, 1_000_000, 30.0)

        Ad, Bd = mg.discretize(sysm.Ac, sysm.Bc, sysm.Ts)
        P = mg.terminal_cost(Ad, Bd, sysm.Q, sysm.R)
        K = mg.lqr_gain(Ad, Bd, P, sysm.R)

        obj_lqr = xbar_small @ P @ xbar_small + pdd["obj_const"]
        _check(f"{sysm.name}: obj_val == xbar'Pxbar (all-solver check)",
               abs(res["obj_val"] - obj_lqr) < 1e-6,
               f"ksp={res['obj_val']:.10f} lqr={obj_lqr:.10f} diff={abs(res['obj_val'] - obj_lqr):.2e}")

        u0_ksp = res["x"][n_x:n_x + n_u]
        u0_lqr = -K @ xbar_small
        _check(f"{sysm.name}: u_0* == -K@xbar, checked against KSP-QP's OWN x",
               np.max(np.abs(u0_ksp - u0_lqr)) < 1e-5,
               f"diff={np.max(np.abs(u0_ksp - u0_lqr)):.2e}")


def feasibility_check(max_M: int) -> None:
    print("\n=== 4. Feasibility across benchmark_mpc.py's default sweep matrix ===")
    # sweep_configs() IS benchmark_mpc's grid construction, imported rather than
    # mirrored, so the validated matrix and the benchmarked matrix cannot drift.
    configs = [(N, M) for (M, N) in sweep_configs()]

    for N, M in configs:
        if max_M and M > max_M:
            print(f"  [skip] platoon M={M} N={N}  (above --max-M={max_M})")
            continue
        sysm = mg.platoon_system(M)
        pdd = mg.generate_mpc_qp(sysm, N, sysm.xbar0)
        _, _, status = solve_raw_osqp(pdd, tol=1e-6)
        _check(f"platoon M={M} N={N}: feasible", status == "solved", status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-M", type=int, default=0,
                        help="Skip section-4 feasibility configs above this vehicle count. "
                             "Default 0 (no cap): verify benchmark_mpc.py's whole default "
                             "sweep, which is the point of the section and costs ~45s in "
                             "total. Pass e.g. --max-M 50 for a faster partial run.")
    args = parser.parse_args()

    smoke_tests()
    objective_cross_check()
    lqr_ground_truth_check()
    feasibility_check(args.max_M)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
