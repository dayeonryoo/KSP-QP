"""
Benchmark KSP-QP against QPALM on the platoon (vehicle-chain) linear Model
Predictive Control (MPC) QP (see python/mpc_generator.py).

MPC is a smooth QP: quadratic tracking cost, linear dynamics equalities, box bounds,
and no general inequality rows (l = 0). Its constraint matrix is banded, with at most
6 nonzeros per row independently of M and N.

Sweep design
------------
The platoon has two size axes:

  * M (vehicle count) grows the band width and the dense 2M x 2M DARE terminal
    block, which makes Q non-diagonal.
  * N (horizon) grows the band length, at fixed bandwidth and terminal block.

QPALM's iteration count grows with both axes while KSP-QP's stays nearly fixed, so
problems of almost the same size can favor either solver depending on (M, N). The
benchmark therefore sweeps the full (M, N) grid; --max-nz can cap it by problem size.

Scope note: KSP-QP has no warm start, so every solve is a cold start for both solvers
(QPALM's warm_start() is never called). This measures cold-solve cost on MPC QPs, not
closed-loop MPC throughput, where a warm-started QPALM would be faster.

Accuracy caveat: `pmm_tol_achieved` and `qpalm_tol_achieved` are each solver's own
residual (own definition, scaling and reformulation) and are not directly comparable:
KSP-QP's is a lifted, relative residual (the terminal block makes Q non-diagonal),
QPALM's is absolute on C = [A; I].

Trajectory design: two loop modes via --loop-mode:
  - "shared" (default): one state trajectory xbar_0..xbar_{T-1} per (M, N) config is
    precomputed by a reference OSQP solve, and both solvers are benchmarked on the
    same T QP instances. OSQP is only the trajectory generator, not a comparison arm.
  - "own": each solver closes its own feedback loop with its own u_0, which shows
    whether numerical error compounds differently.

Outputs
-------
  results/{--out prefix}_mpc_kspqp.csv, results/{--out prefix}_mpc_qpalm.csv
  one file per solver, long format: one row per (instance x repeat)

  Each solver only writes its own file. Rerunning a configuration (M, N, loop mode) replaces
  that configuration's rows in the files of the solvers run and keeps every other row, so
  KSP-QP can be rerun after every change while QPALM's results are kept. Rows of the two
  files pair up on (M, N, loop_mode, step, rep); in the "shared" loop mode every run solves
  the same instances (same reference trajectory), which xbar_norm confirms.

  After the instance columns, the files use benchmark_mm.py's column names (ssn_*, pmm_iter, ...
  for KSP-QP, qpalm_* for QPALM) and record their run's provenance. A line is printed per solve.

  mpc_kspqp.csv also has ksp_mode (pcg or direct), refactor_ratio, rho_init and KSP-QP's factorization log
  and run-time split (experimental; see include/solve_profile.hpp). With --trace, KSP-QP's per-event /
  per-SSN-iteration / per-PMM-iteration rows go to mpc_kspqp_trace_{events,ssn,pmm}.csv.

=== HOW TO RUN ===
  python3 benchmark_mpc.py --out 0926              # -> results/0926_mpc_{kspqp,qpalm}.csv
  python3 benchmark_mpc.py --solver ksp-qp --out 0926   # rerun KSP-QP only, keep 0926_mpc_qpalm.csv
  python3 benchmark_mpc.py --M 20 50 --N 200 400 --T 4 --reps 3
  python3 benchmark_mpc.py --solver ksp-qp --time-limit 60
  python3 benchmark_mpc.py --solver ksp-qp --direct   # direct solver instead of PCG
  python3 benchmark_mpc.py --solver ksp-qp --refactor-ratio 4 --out rr4   # one --out per R
  python3 benchmark_mpc.py --solver ksp-qp --rho-init 1e3 --out rho3   # rho's starting value; one --out per value
  python3 benchmark_mpc.py --loop-mode own --max-nz 20000
"""

import sys
import time
import argparse
import platform
import multiprocessing as mp
from pathlib import Path

import numpy as np
import qpalm
import osqp

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

try:
    import ksp_qp_bind
except ModuleNotFoundError:
    sys.exit(
        "Cannot find ksp_qp_bind. "
        "Build it first:\n"
        "  cd python && mkdir build && cd build\n"
        "  cmake .. && cmake --build . --config Release"
    )

from benchmark_common import (
    kspqp_to_qpalm,
    QPALM_SOLVED,
    _run_isolated,
    provenance,
    check_header,
    replace_rows,
    trace_rows,
)
import mpc_generator as mg

# Sweep grid: the full cross product, unless --max-nz caps n_z.
DEFAULT_M = [5, 10, 20, 50, 100, 200, 300]
DEFAULT_N = [20, 50, 100, 200, 400]

DEFAULT_T = 6               # rollout steps per config
DEFAULT_REPS = 4            # timed repeats per instance; repeat 0 is discarded
DEFAULT_TOL = 1e-6

SOLVERS = ["ksp-qp", "qpalm"]
SOLVER_FILES = {"ksp-qp": "kspqp", "qpalm": "qpalm"}   # results/{prefix}mpc_<tag>.csv
PREFIX = {"ksp-qp": "ssn", "qpalm": "qpalm"}           # column prefix, as in benchmark_mm.py


def sweep_configs(M_list=None, N_list=None, max_nz: int = 0):
    """(M, N) pairs of the grid, with n_z <= max_nz if max_nz > 0. Shared with
    validate_mpc_generator.py so the validated matrix and the benchmarked matrix cannot drift apart."""
    M_list = DEFAULT_M if M_list is None else M_list
    N_list = DEFAULT_N if N_list is None else N_list
    out = []
    for M in M_list:
        for N in N_list:
            n_z = M * (3 * N + 2)        # (N+1)*2M + N*M
            if max_nz and n_z > max_nz:
                continue
            out.append((M, N))
    return out


# ---------------------------------------------------------------------------
# Solver wrappers.
#
# These also return x, since the rollout needs u_0 to advance the state;
# benchmark_common.run_qpalm/run_osqp discard it.
# ---------------------------------------------------------------------------

def _solve_qpalm_x(qpalm_data: tuple, tol: float, time_limit: float, obj_const: float = 0.0):
    Q_upper, q, C, bmin, bmax, n, m_total = qpalm_data
    data = qpalm.Data(n, m_total)
    data.Q, data.q, data.A, data.bmin, data.bmax = Q_upper, q, C, bmin, bmax

    settings = qpalm.Settings()
    settings.eps_abs, settings.eps_rel = tol, tol
    settings.max_iter = 2_000_000_000
    settings.time_limit = time_limit
    settings.verbose = 0
    settings.scaling = 10

    solver = qpalm.Solver(data, settings)
    solver.solve()
    info = solver.info
    result = {
        "qpalm_status": int(info.status_val),
        "qpalm_solved": int(info.status_val == QPALM_SOLVED),
        "qpalm_time": float(info.run_time),
        "qpalm_setup_time": float(getattr(info, "setup_time", float("nan"))),
        "qpalm_solve_time": float(getattr(info, "solve_time", float("nan"))),
        "qpalm_iter": int(info.iter_out),
        "qpalm_inner_iter": int(info.iter),
        "qpalm_obj": float(info.objective) + obj_const,
        "qpalm_tol_achieved": max(float(info.pri_res_norm), float(info.dua_res_norm)),
    }
    return result, np.asarray(solver.solution.x, dtype=np.float64)


def _solve_osqp_x(qpalm_data: tuple, tol: float, time_limit: float):
    """Reference-trajectory generator only -- NOT a comparison arm."""
    Q_upper, q, C, bmin, bmax, *_ = qpalm_data
    prob = osqp.OSQP()
    prob.setup(Q_upper, q, C, bmin, bmax, eps_abs=tol, eps_rel=tol,
               max_iter=2_000_000_000, time_limit=time_limit, verbose=False, scaling=10)
    res = prob.solve()
    return np.asarray(res.x, dtype=np.float64)


def _solve_kspqp_x(pdd: dict, tol: float, time_limit: float, max_iter: int,
                   direct: bool = False, trace: bool = False, refactor_ratio: float = 1.0,
                   rho_init: float | None = None):
    r = ksp_qp_bind.solve_from_data(pdd, tol, max_iter, time_limit, direct=direct, trace=trace,
                                    refactor_ratio=refactor_ratio, rho_init=-1.0 if rho_init is None else rho_init)
    result = {
        "ssn_status": int(r["status"]),
        "ssn_solved": int(r["status"] == 0),
        "ssn_time": float(r["run_time"]),
        "ssn_setup_time": float(r["setup_time"]),
        "ssn_solve_time": float(r["solve_time"]),
        "pmm_iter": int(r["pmm_iter"]),
        "ssn_iter": int(r["ssn_iter"]),
        "krylov_iter": int(r["krylov_iter"]),
        "fact": int(r["fact"]),
        "smw_count": int(r["smw_count"]),
        "ssn_obj": float(r["obj_val"]),
        "pmm_tol_achieved": float(r["pmm_tol_achieved"]),
        "ksp_mode": "direct" if direct else "pcg",
        "refactor_ratio": refactor_ratio,
        "rho_init": "" if rho_init is None else rho_init,
        **{k: r[k] for k in KSPQP_EXTRA if k not in ("ksp_mode", "refactor_ratio", "rho_init")},
    }
    if trace:
        result["_trace"] = {t: r[f"trace_{t}"] for t in TRACE_TABLES}
    return result, np.asarray(r["x"], dtype=np.float64)


def _solve_one(solver_name, pdd, tol, time_limit, max_iter, direct=False, trace=False, refactor_ratio=1.0,
               rho_init=None):
    if solver_name == "ksp-qp":
        return _solve_kspqp_x(pdd, tol, time_limit, max_iter, direct, trace, refactor_ratio, rho_init)
    if solver_name == "qpalm":
        return _solve_qpalm_x(kspqp_to_qpalm(pdd), tol, time_limit, pdd.get("obj_const", 0.0))
    raise ValueError(f"Unknown solver_name: {solver_name}")


def _log_solve(solver_name: str, t: int, rep: int, r: dict) -> None:
    """One line per solve, as benchmark_mm.py prints."""
    if solver_name == "ksp-qp":
        status = "OPTIMAL" if r["ssn_solved"] else f"status={r['ssn_status']}"
        print(f"    step {t} rep {rep}  KSP-QP : {status:12s}  t = {r['ssn_time'] * 1000:.2f} ms  "
              f"pmm={r['pmm_iter']} ssn={r['ssn_iter']} krylov={r['krylov_iter']} "
              f"fact={r['fact']} smw={r['smw_count']}  "
              f"tol={r['pmm_tol_achieved']:.2e}  obj = {r['ssn_obj']:.6g}", flush=True)
    else:
        status = "OPTIMAL" if r["qpalm_solved"] else f"status={r['qpalm_status']}"
        print(f"    step {t} rep {rep}  QPALM  : {status:12s}  t = {r['qpalm_time'] * 1000:.2f} ms  "
              f"outer={r['qpalm_iter']} inner={r['qpalm_inner_iter']}  "
              f"tol={r['qpalm_tol_achieved']:.2e}  obj = {r['qpalm_obj']:.6g}", flush=True)


# ---------------------------------------------------------------------------
# Trajectory precomputation (once per config, via a reference OSQP solve)
# ---------------------------------------------------------------------------

def generate_trajectory(M: int, N: int, T: int) -> list:
    sysm = mg.platoon_system(M)
    xbar = sysm.xbar0.copy()
    traj = [xbar.copy()]
    for _ in range(T - 1):
        pdd = mg.generate_mpc_qp(sysm, N, xbar)
        # Must converge, not time out, or the instances would differ between runs.
        x = _solve_osqp_x(kspqp_to_qpalm(pdd), 1e-8, 300.0)
        u0 = x[sysm.n_x:sysm.n_x + sysm.n_u]
        xbar = mg.step_dynamics(sysm, xbar, u0)
        traj.append(xbar.copy())
    return traj


# ---------------------------------------------------------------------------
# Rollout worker: one subprocess per (M, N) config, since small MPC QPs solve in
# milliseconds and per-step spawning would dominate; crashes/OOM are still isolated
# per config. Both solvers run in the same worker so their order can alternate per
# repeat, avoiding thermal-drift and P-/E-core bias toward whichever runs first. That
# only applies when both are run together; a solver rerun alone is timed on its own.
# ---------------------------------------------------------------------------

def _set_high_qos():
    """Ask macOS for a user-interactive QoS class, so this process is scheduled on a
    performance core rather than an efficiency core. No-op off Darwin."""
    if platform.system() != "Darwin":
        return
    try:
        import ctypes
        libc = ctypes.CDLL("libSystem.dylib")
        libc.pthread_set_qos_class_self_np(0x21, 0)   # QOS_CLASS_USER_INTERACTIVE
    except Exception:
        pass


def _worker_rollout(sysm, N, loop_mode, xbar_source, T, reps, tol, time_limit, max_iter,
                    solvers, direct, trace, refactor_ratio, rho_init, conn):
    rows = {s: [] for s in solvers}
    traces = []  # (step and repeat, KSP-QP's trace tables) per traced solve
    try:
        _set_high_qos()

        # Spin briefly so the scheduler promotes us off an efficiency core before
        # the first timed solve.
        t_end = time.perf_counter() + 0.2
        while time.perf_counter() < t_end:
            pass

        # In "own" mode each solver walks its own trajectory, so state is per solver.
        state = {s: (None if loop_mode == "shared" else xbar_source.copy()) for s in solvers}

        for t in range(T):
            for rep in range(reps):
                # Interleave: every solver is measured at every repeat index.
                for solver_name in solvers:
                    xbar = xbar_source[t] if loop_mode == "shared" else state[solver_name]
                    pdd = mg.generate_mpc_qp(sysm, N, xbar)
                    t0 = time.perf_counter()
                    result, x = _solve_one(solver_name, pdd, tol, time_limit, max_iter, direct, trace,
                                           refactor_ratio, rho_init)
                    wall = time.perf_counter() - t0
                    if "_trace" in result:
                        traces.append(({"step": t, "rep": rep}, result.pop("_trace")))
                    _log_solve(solver_name, t, rep, result)

                    rows[solver_name].append({
                        "step": t, "rep": rep,
                        "xbar_norm": float(np.linalg.norm(xbar)),
                        f"{PREFIX[solver_name]}_wall_time": wall,
                        **result,
                    })

                    if loop_mode == "own" and rep == reps - 1 and t < T - 1:
                        u0 = x[sysm.n_x:sysm.n_x + sysm.n_u]
                        state[solver_name] = mg.step_dynamics(sysm, state[solver_name], u0)
        conn.send({"rows": rows, "traces": traces})
    except Exception as e:
        conn.send({"error": f"{type(e).__name__}: {e}"})
    conn.close()


# ---------------------------------------------------------------------------
# CSV -- one file per solver, long format, one row per (instance, repeat), with
# benchmark_mm.py's column names after the instance columns. extrasaction="ignore"
# silently drops keys missing from FIELDS. A rerun replaces whole configurations
# (CONFIG_KEY) in a solver's file.
# ---------------------------------------------------------------------------

CONFIG_KEY = ["M", "N", "loop_mode"]

INSTANCE = ["M", "N", "loop_mode", "step", "rep", "T",
            "n_x", "n_u", "n_z", "m_eq", "l_ineq", "nnz_Q", "nnz_A", "xbar_norm"]

# Results as reported by each solver, in ITS OWN residual convention.
FIELDS = {
    "ksp-qp": [*INSTANCE,
               "ssn_solved", "ssn_status", "pmm_iter", "ssn_iter", "krylov_iter", "fact", "smw_count",
               "ssn_obj", "pmm_tol_achieved", "ssn_time", "ssn_setup_time", "ssn_solve_time",
               "ssn_wall_time", "ssn_tol", "ssn_time_limit",
               "ssn_run_id", "ssn_timestamp", "ssn_host", "ssn_cpu", "ssn_git_sha"],
    "qpalm":  [*INSTANCE,
               "qpalm_solved", "qpalm_status", "qpalm_iter", "qpalm_inner_iter",
               "qpalm_obj", "qpalm_tol_achieved", "qpalm_time", "qpalm_setup_time", "qpalm_solve_time",
               "qpalm_wall_time", "qpalm_tol", "qpalm_time_limit",
               "qpalm_run_id", "qpalm_timestamp", "qpalm_host", "qpalm_cpu", "qpalm_version"],
}

# mpc_kspqp.csv also has KSP-QP's mode and its factorization log and run-time split
# (experimental; include/solve_profile.hpp): fact_<reason>, smw_update, ..., t_<phase>.
KSPQP_EXTRA = ["ksp_mode", "refactor_ratio", "rho_init", "ssn_tol_achieved", "linesearch_fail", "krylov_fail",
               "kkt_ldlt_fact", "schur_chol_fact", *ksp_qp_bind.profile_columns()]
FIELDS["ksp-qp"] += KSPQP_EXTRA

# --trace: KSP-QP's per-event / per-SSN-iteration / per-PMM-iteration rows, led by the
# configuration key and (step, rep), in mpc_kspqp_trace_<table>.csv.
TRACE_TABLES = ("events", "ssn", "pmm")

# Written for each solver of a configuration whose worker failed, as in benchmark_mm.py.
ERROR_ROW = {
    "ksp-qp": dict(ssn_status=-99, ssn_solved=0, ssn_time=np.inf, ssn_wall_time=np.inf,
                   pmm_iter=np.inf, ssn_iter=np.inf, krylov_iter=np.inf, fact=np.inf, smw_count=np.inf,
                   ssn_obj=np.nan, pmm_tol_achieved=np.nan, ssn_setup_time=np.nan, ssn_solve_time=np.nan),
    "qpalm":  dict(qpalm_status=-99, qpalm_solved=0, qpalm_time=np.inf, qpalm_wall_time=np.inf,
                   qpalm_iter=np.inf, qpalm_inner_iter=np.inf,
                   qpalm_obj=np.nan, qpalm_tol_achieved=np.nan, qpalm_setup_time=np.nan, qpalm_solve_time=np.nan),
}


def run_config(M, N, T, reps, tol, time_limit, max_iter, solvers, csv_paths,
               run_cols, cooldown=0.0, loop_mode="shared", direct=False, trace_paths=None,
               refactor_ratio=1.0, rho_init=None) -> None:
    sysm = mg.platoon_system(M)
    pdd0 = mg.generate_mpc_qp(sysm, N, sysm.xbar0)
    n_z = int(pdd0["n"])
    meta = {
        "M": M, "N": N, "n_x": sysm.n_x, "n_u": sysm.n_u,
        "n_z": n_z, "m_eq": int(pdd0["m"]), "l_ineq": int(pdd0["l"]),
        "nnz_Q": int(pdd0["Q_data"].size), "nnz_A": int(pdd0["A_data"].size),
        "loop_mode": loop_mode, "T": T,
    }
    print(f"  M={M} N={N} n_z={n_z} nnz_Q={meta['nnz_Q']} nnz_A={meta['nnz_A']} "
          f"T={T} reps={reps} loop_mode={loop_mode}", flush=True)

    xbar_source = generate_trajectory(M, N, T) if loop_mode == "shared" else sysm.xbar0

    # Pass sysm itself, so the worker doesn't solve the DARE again.
    out = _run_isolated(_worker_rollout,
                        (sysm, N, loop_mode, xbar_source, T, reps, tol, time_limit,
                         max_iter, solvers, direct, bool(trace_paths), refactor_ratio, rho_init))
    if "error" in out:
        print(f"    ERROR - {out['error']}", flush=True)
        for s in solvers:
            replace_rows(csv_paths[s], [{**meta, **run_cols[s], **ERROR_ROW[s]}],
                         CONFIG_KEY, FIELDS[s])
        return

    for s in solvers:
        replace_rows(csv_paths[s], [{**meta, **run_cols[s], **r} for r in out["rows"][s]],
                     CONFIG_KEY, FIELDS[s])
    if trace_paths and out["traces"]:
        config = {k: meta[k] for k in CONFIG_KEY}
        for table, path in trace_paths.items():
            columns = list(out["traces"][0][1][table])
            rows = [row for key, tables in out["traces"] for row in trace_rows(tables[table], {**config, **key})]
            replace_rows(path, rows, CONFIG_KEY, [*CONFIG_KEY, "step", "rep", *columns])

    # Timed repeats: discard repeat 0 (allocator / page-fault warm-up), report the
    # minimum of the rest -- the least scheduler-contaminated estimate for a
    # deterministic single-threaded kernel.
    for s in solvers:
        p = PREFIX[s]
        ts = [r[f"{p}_time"] for r in out["rows"][s] if r["rep"] > 0]
        solved = sum(r[f"{p}_solved"] for r in out["rows"][s] if r["rep"] == 0)
        n_inst = sum(1 for r in out["rows"][s] if r["rep"] == 0)
        best = min(ts) if ts else float("nan")
        print(f"    {s:8s} solved={solved}/{n_inst}  best={best * 1000:9.1f}ms  "
              f"median={float(np.median(ts)) * 1000:9.1f}ms", flush=True)

    if cooldown:
        time.sleep(cooldown)


def main() -> None:
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(HERE.parent))
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--M", type=int, nargs="+", default=DEFAULT_M,
                        help="Platoon vehicle counts (grid rows)")
    parser.add_argument("--N", type=int, nargs="+", default=DEFAULT_N,
                        help="Horizon lengths (grid columns)")
    parser.add_argument("--max-nz", type=int, default=0,
                        help="Skip (M,N) combinations above this n_z (default 0: no cap, the full grid)")
    parser.add_argument("--T", type=int, default=DEFAULT_T,
                        help="Rollout steps per configuration")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help="Timed repeats per instance; repeat 0 is discarded")
    parser.add_argument("--solver", nargs="+", default=SOLVERS, choices=SOLVERS)
    parser.add_argument("--loop-mode", default="shared", choices=["shared", "own"],
                        help="'shared' (default): both solvers replay one "
                             "OSQP-precomputed trajectory. 'own': each closes its own "
                             "feedback loop. See module docstring.")
    parser.add_argument("--cooldown", type=float, default=0.0,
                        help="Seconds to sleep between configurations, to limit "
                             "thermal drift across a long sweep")
    parser.add_argument("--direct", action="store_true",
                        help="Run KSP-QP with its experimental direct solver (LDLT on K or "
                             "Cholesky on its Schur complement) instead of PCG")
    parser.add_argument("--refactor-ratio", type=float, default=1.0, metavar="R",
                        help="Experimental, PCG: keep the preconditioner's factorization across mu/rho "
                             "changes until max(mu/mu_f, mu_f/mu) * max(rho/rho_f, rho_f/rho) exceeds R "
                             "(default: 1, every change). Rows are keyed by (M, N, loop_mode), so use one "
                             "--out per R.")
    parser.add_argument("--rho-init", type=float, default=None, metavar="RHO",
                        help="Experimental: rho's starting value (default: the solver's own). Use one --out "
                             "per value.")
    parser.add_argument("--trace", action="store_true",
                        help="Also write KSP-QP's per-event, per-SSN-iteration and per-PMM-iteration "
                             "rows to mpc_kspqp_trace_{events,ssn,pmm}.csv (experimental)")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    if not args.refactor_ratio >= 1:
        sys.exit("--refactor-ratio must be at least 1")
    if args.refactor_ratio > 1 and args.direct:
        print("WARNING: --refactor-ratio has no effect with --direct")
    if args.rho_init is not None and not args.rho_init > 0:
        sys.exit("--rho-init must be positive")

    root = Path(args.root).resolve()
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)

    max_iter = 10_000_000_000
    solvers = [s for s in SOLVERS if s in set(args.solver)]
    name_prefix = f"{args.out}_" if args.out else ""
    csv_paths = {s: result_dir / f"{name_prefix}mpc_{SOLVER_FILES[s]}.csv" for s in solvers}

    # Never mix layouts in one file -- that is how the archived CSVs ended up
    # with columns that no longer exist.
    for s, p in csv_paths.items():
        check_header(p, FIELDS[s])
    trace_paths = ({t: result_dir / f"{name_prefix}mpc_kspqp_trace_{t}.csv" for t in TRACE_TABLES}
                   if args.trace and "ksp-qp" in solvers else None)

    configs = sweep_configs(args.M, args.N, args.max_nz)

    # Provenance recorded on every row this run writes, per solver.
    prov = provenance()
    run_cols = {
        "ksp-qp": {"ssn_tol": args.tol, "ssn_time_limit": args.time_limit,
                   "ssn_run_id": prov["run_id"], "ssn_timestamp": prov["timestamp"],
                   "ssn_host": prov["host"], "ssn_cpu": prov["cpu"], "ssn_git_sha": prov["git_sha"]},
        "qpalm":  {"qpalm_tol": args.tol, "qpalm_time_limit": args.time_limit,
                   "qpalm_run_id": prov["run_id"], "qpalm_timestamp": prov["timestamp"],
                   "qpalm_host": prov["host"], "qpalm_cpu": prov["cpu"],
                   "qpalm_version": prov["qpalm_version"]},
    }

    print(f"run_id={prov['run_id']}  git={prov['git_sha']}  {len(configs)} configurations  "
          f"tol={args.tol:g}  solvers={','.join(solvers)}  "
          f"ksp-qp={'direct' if args.direct else 'pcg'}  refactor_ratio={args.refactor_ratio}  "
          f"rho_init={args.rho_init or 'default'}")
    for i, (M, N) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]")
        run_config(M, N, args.T, args.reps, args.tol, args.time_limit, max_iter,
                   solvers, csv_paths, run_cols, args.cooldown, args.loop_mode, args.direct, trace_paths,
                   args.refactor_ratio, args.rho_init)

    print("\nResults written to: " + ", ".join(str(p) for p in csv_paths.values()))


if __name__ == "__main__":
    main()
