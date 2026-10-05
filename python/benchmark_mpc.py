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
benchmark therefore sweeps the (M, N) grid, capped by problem size (--max-nz).

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
  results/{--out prefix}_mpc_sweep.csv
  long format: one row per (instance x solver x repeat)

=== HOW TO RUN ===
  python3 benchmark_mpc.py --out 0926              # -> results/0926_mpc_sweep.csv
  python3 benchmark_mpc.py --M 20 50 --N 200 400 --T 4 --reps 3
  python3 benchmark_mpc.py --solver ksp-qp --time-limit 60
  python3 benchmark_mpc.py --loop-mode own --max-nz 20000
"""

import sys
import csv
import time
import uuid
import socket
import argparse
import platform
import subprocess
import multiprocessing as mp
from datetime import datetime, timezone
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
    _write_csv,
)
import mpc_generator as mg

SCHEMA_VERSION = 2          # bump on any CSV_FIELDS change; old CSVs are not appendable

# Sweep grid. The cross product is filtered by MAX_NZ, which drops only the
# top-right corner (large M *and* large N simultaneously).
DEFAULT_M = [5, 10, 20, 50, 100, 200, 300]
DEFAULT_N = [20, 50, 100, 200, 400]
MAX_NZ = 65_000

DEFAULT_T = 6               # rollout steps per config
DEFAULT_REPS = 4            # timed repeats per instance; repeat 0 is discarded
DEFAULT_TOL = 1e-6

SOLVERS = ["ksp-qp", "qpalm"]


def sweep_configs(M_list=None, N_list=None, max_nz: int = MAX_NZ):
    """(M, N) pairs of the grid, size-capped. Shared with validate_mpc_generator.py
    so the validated matrix and the benchmarked matrix cannot drift apart."""
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
        "status": int(info.status_val),
        "solved": int(info.status_val == QPALM_SOLVED),
        "obj_val": float(info.objective) + obj_const,
        "setup_time": float(getattr(info, "setup_time", float("nan"))),
        "solve_time": float(getattr(info, "solve_time", float("nan"))),
        "run_time": float(info.run_time),
        "outer_iter": int(info.iter_out),
        "inner_iter": int(info.iter),
        "tol_achieved": max(float(info.pri_res_norm), float(info.dua_res_norm)),
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


def _solve_kspqp_x(pdd: dict, tol: float, time_limit: float, max_iter: int):
    r = ksp_qp_bind.solve_from_data(pdd, tol, max_iter, time_limit)
    result = {
        "status": int(r["status"]),
        "solved": int(r["status"] == 0),
        "obj_val": float(r["obj_val"]),
        "setup_time": float(r["setup_time"]),
        "solve_time": float(r["solve_time"]),
        "run_time": float(r["run_time"]),
        "outer_iter": int(r["pmm_iter"]),
        "inner_iter": int(r["ssn_iter"]),
        "krylov_iter": int(r["krylov_iter"]),
        "fact": int(r["fact"]),
        "smw_count": int(r["smw_count"]),
        "tol_achieved": float(r["pmm_tol_achieved"]),
    }
    return result, np.asarray(r["x"], dtype=np.float64)


def _solve_one(solver_name, pdd, tol, time_limit, max_iter):
    if solver_name == "ksp-qp":
        return _solve_kspqp_x(pdd, tol, time_limit, max_iter)
    if solver_name == "qpalm":
        return _solve_qpalm_x(kspqp_to_qpalm(pdd), tol, time_limit, pdd.get("obj_const", 0.0))
    raise ValueError(f"Unknown solver_name: {solver_name}")


# ---------------------------------------------------------------------------
# Trajectory precomputation (once per config, via a reference OSQP solve)
# ---------------------------------------------------------------------------

def generate_trajectory(M: int, N: int, T: int) -> list:
    sysm = mg.platoon_system(M)
    xbar = sysm.xbar0.copy()
    traj = [xbar.copy()]
    for _ in range(T - 1):
        pdd = mg.generate_mpc_qp(sysm, N, xbar)
        x = _solve_osqp_x(kspqp_to_qpalm(pdd), 1e-8, 30.0)
        u0 = x[sysm.n_x:sysm.n_x + sysm.n_u]
        xbar = mg.step_dynamics(sysm, xbar, u0)
        traj.append(xbar.copy())
    return traj


# ---------------------------------------------------------------------------
# Rollout worker: one subprocess per (M, N) config, since small MPC QPs solve in
# milliseconds and per-step spawning would dominate; crashes/OOM are still isolated
# per config. Both solvers run in the same worker so their order can alternate per
# repeat, avoiding thermal-drift and P-/E-core bias toward whichever runs first.
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


def _worker_rollout(M, N, loop_mode, xbar_source, T, reps, tol, time_limit, max_iter,
                    solvers, conn):
    rows = []
    try:
        _set_high_qos()
        sysm = mg.platoon_system(M)

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
                    result, x = _solve_one(solver_name, pdd, tol, time_limit, max_iter)
                    wall = time.perf_counter() - t0

                    rows.append({
                        "M": M, "N": N, "step": t, "rep": rep,
                        "solver": solver_name,
                        "xbar_norm": float(np.linalg.norm(xbar)),
                        "wall_time": wall,
                        **{k: v for k, v in result.items()},
                    })

                    if loop_mode == "own" and rep == reps - 1 and t < T - 1:
                        u0 = x[sysm.n_x:sysm.n_x + sysm.n_u]
                        state[solver_name] = mg.step_dynamics(sysm, state[solver_name], u0)
        conn.send({"rows": rows})
    except Exception as e:
        conn.send({"error": f"{type(e).__name__}: {e}"})
    conn.close()


# ---------------------------------------------------------------------------
# CSV -- long format, one row per (instance, solver, repeat).
#
# schema_version separates this from older wide CSVs (all solvers on one row), which
# are incompatible; _write_csv's extrasaction="ignore" silently drops keys missing
# from CSV_FIELDS.
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    # provenance
    "schema_version", "run_id", "timestamp", "host", "cpu", "git_sha",
    "qpalm_version", "osqp_version",
    # instance
    "M", "N", "n_x", "n_u", "n_z", "m_eq", "l_ineq", "nnz_Q", "nnz_A",
    "loop_mode", "T", "step", "xbar_norm",
    # configuration
    "solver", "tol_requested", "time_limit", "rep",
    # results (as reported by each solver, in ITS OWN residual convention)
    "status", "solved", "obj_val", "tol_achieved",
    "setup_time", "solve_time", "run_time", "wall_time",
    "outer_iter", "inner_iter", "krylov_iter", "fact", "smw_count",
]


def _provenance() -> dict:
    def _sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "cpu": _sh("sysctl -n machdep.cpu.brand_string") or platform.processor(),
        "git_sha": _sh("git rev-parse --short HEAD"),
        "qpalm_version": getattr(qpalm, "__version__", "?"),
        "osqp_version": getattr(osqp, "__version__", "?"),
    }


def run_config(M, N, T, reps, tol, time_limit, max_iter, solvers, rows, csv_path,
               prov, cooldown=0.0, loop_mode="shared") -> None:
    sysm = mg.platoon_system(M)
    pdd0 = mg.generate_mpc_qp(sysm, N, sysm.xbar0)
    n_z = int(pdd0["n"])
    meta = {
        **prov,
        "M": M, "N": N, "n_x": sysm.n_x, "n_u": sysm.n_u,
        "n_z": n_z, "m_eq": int(pdd0["m"]), "l_ineq": int(pdd0["l"]),
        "nnz_Q": int(pdd0["Q_data"].size), "nnz_A": int(pdd0["A_data"].size),
        "loop_mode": loop_mode, "T": T,
        "tol_requested": tol, "time_limit": time_limit,
    }
    print(f"  M={M} N={N} n_z={n_z} nnz_Q={meta['nnz_Q']} nnz_A={meta['nnz_A']} "
          f"T={T} reps={reps} loop_mode={loop_mode}", flush=True)

    xbar_source = generate_trajectory(M, N, T) if loop_mode == "shared" else sysm.xbar0

    out = _run_isolated(_worker_rollout,
                        (M, N, loop_mode, xbar_source, T, reps, tol, time_limit,
                         max_iter, solvers))
    if "error" in out:
        print(f"    ERROR - {out['error']}", flush=True)
        rows.append({**meta, "solver": "ERROR", "status": -99, "solved": 0})
        _write_csv(csv_path, rows, CSV_FIELDS)
        return

    for r in out["rows"]:
        rows.append({**meta, **r})

    # Timed repeats: discard repeat 0 (allocator / page-fault warm-up), report the
    # minimum of the rest -- the least scheduler-contaminated estimate for a
    # deterministic single-threaded kernel.
    for s in solvers:
        ts = [r["run_time"] for r in out["rows"] if r["solver"] == s and r["rep"] > 0]
        solved = sum(r["solved"] for r in out["rows"] if r["solver"] == s and r["rep"] == 0)
        n_inst = sum(1 for r in out["rows"] if r["solver"] == s and r["rep"] == 0)
        best = min(ts) if ts else float("nan")
        print(f"    {s:8s} solved={solved}/{n_inst}  best={best * 1000:9.1f}ms  "
              f"median={float(np.median(ts)) * 1000:9.1f}ms", flush=True)

    _write_csv(csv_path, rows, CSV_FIELDS)
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
    parser.add_argument("--max-nz", type=int, default=MAX_NZ,
                        help="Skip (M,N) combinations above this n_z (0 = no cap). "
                             "Only trims the large-M-and-large-N corner.")
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
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)

    max_iter = 10_000_000_000
    solvers = [s for s in SOLVERS if s in set(args.solver)]
    name_prefix = f"{args.out}_" if args.out else ""
    csv_path = result_dir / f"{name_prefix}mpc_sweep.csv"

    # Never append across schema versions -- that is how the archived CSVs ended up
    # with columns that no longer exist.
    if csv_path.exists():
        with open(csv_path, newline="") as fh:
            head = next(csv.reader(fh), [])
        if head != CSV_FIELDS:
            sys.exit(f"{csv_path} has an incompatible schema (expected schema_version "
                     f"{SCHEMA_VERSION}). Move it aside or pass --out <prefix>.")

    configs = sweep_configs(args.M, args.N, args.max_nz)
    prov = _provenance()
    rows: list[dict] = []

    print(f"run_id={prov['run_id']}  git={prov['git_sha']}  {len(configs)} configurations  "
          f"tol={args.tol:g}  solvers={','.join(solvers)}")
    for i, (M, N) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]")
        run_config(M, N, args.T, args.reps, args.tol, args.time_limit, max_iter,
                   solvers, rows, csv_path, prov, args.cooldown, args.loop_mode)

    print(f"\nResults written to: {csv_path}")


if __name__ == "__main__":
    main()
