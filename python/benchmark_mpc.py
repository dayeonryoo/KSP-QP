"""
Benchmark KSP-QP against QPALM on the platoon (vehicle-chain) linear Model
Predictive Control (MPC) QP (see python/mpc_generator.py).

MPC is a native smooth QP: pure quadratic tracking cost, linear dynamics equality
constraints, simple box bounds, no general inequality rows (l = 0). Its constraint
matrix is banded, with at most 6 nonzeros per row independently of M and N.

Sweep design
------------
The platoon has TWO size axes and they are not interchangeable:

  * M (vehicle count) grows the band WIDTH -- and also the dense 2M x 2M DARE
    terminal cost block, which is what forces Q to be non-diagonal.
  * N (horizon) grows the band LENGTH: more banded stage blocks, fixed bandwidth,
    fixed terminal block.

Sweeping them one at a time through the corner of the (M, N) plane is misleading.
Measured at tol 1e-6, two configurations of almost identical size land on opposite
sides of the result:

    M=5,  N=1600  (n_z=24010):  KSP-QP 700 ms  vs  QPALM  477 ms   (0.68x)
    M=20, N=400   (n_z=24040):  KSP-QP 734 ms  vs  QPALM 2060 ms   (2.81x)

KSP-QP's time barely moves between them; QPALM's quadruples, because its iteration
count grows with both axes (121 -> 288) while KSP-QP's stays essentially fixed
across the whole family (pmm 13-15, krylov 104-182). So this benchmark sweeps the
(M, N) GRID rather than two 1-D slices, capped by problem size (--max-nz).

Scope note: KSP-QP has NO warm-start capability at any level yet, so every solve
here is a cold start for BOTH solvers -- QPALM's warm_start() is deliberately never
invoked, to keep the comparison like-for-like. This measures per-instance cold-solve
cost on MPC-structured QPs; it is NOT a claim about closed-loop MPC throughput, where
a warm-started QPALM would be considerably faster.

Accuracy caveat: `pmm_tol_achieved` and `qpalm_tol_achieved` are each solver's OWN
reported residual, in its own residual definition, its own scaling and its own
reformulation of the problem. They are recorded as reported and are NOT directly
comparable to one another -- in particular KSP-QP's is a lifted, relative residual
(the MPC terminal block makes Q non-diagonal) while QPALM's is absolute on the
stacked C = [A; I] form. Any write-up using these numbers must say so.

Trajectory design: two loop modes via --loop-mode:
  - "shared" (default): one canonical state trajectory xbar_0..xbar_{T-1} is
    precomputed per (M, N) config by a reference OSQP solve, then both solvers are
    benchmarked on the identical sequence of T QP instances. OSQP appears ONLY as
    this neutral trajectory generator -- it is not a comparison arm -- so neither
    compared solver's own control choices bias the instance sequence.
  - "own": each solver closes its own feedback loop with its own u_0. Operationally
    realistic, and surfaces whether numerical error compounds differently.

Outputs
-------
  results/{--out prefix}_mpc_sweep.csv
  long format: one row per (instance x solver x repeat)

=== HOW TO RUN ===
  python3 benchmark_mpc.py --out 0926              # -> results/0926_mpc_sweep.csv
  python3 benchmark_mpc.py --M 20 50 --N 200 400 --T 4 --reps 3
  python3 benchmark_mpc.py --solver ksp-qp --time-limit 60
  python3 benchmark_mpc.py --solver ksp-qp --direct   # direct solver instead of PCG
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
# These return the solution vector x as well as the result dict; the shared
# benchmark_common.run_qpalm/run_osqp discard it, and the rollout needs u_0 to
# advance the state. They are local variants rather than modifications to that
# already-relied-upon module.
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


def _solve_kspqp_x(pdd: dict, tol: float, time_limit: float, max_iter: int,
                   direct: bool = False):
    r = ksp_qp_bind.solve_from_data(pdd, tol, max_iter, time_limit, direct=direct)
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


def _solve_one(solver_name, pdd, tol, time_limit, max_iter, direct=False):
    if solver_name == "ksp-qp":
        return _solve_kspqp_x(pdd, tol, time_limit, max_iter, direct)
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
# Rollout worker. One subprocess runs the whole rollout for ONE config: MPC QPs
# solve in milliseconds at the small end, so per-step spawn overhead would dominate.
# Crash/OOM isolation is kept, just at (M, N) granularity.
#
# Both solvers run inside the SAME worker so that solver order can be interleaved
# per repeat -- running all of solver A then all of solver B lets thermal drift and
# P-core/E-core migration bias whichever ran first.
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
                    solvers, direct, conn):
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
                    result, x = _solve_one(solver_name, pdd, tol, time_limit, max_iter, direct)
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
# The previous wide merged-by-step schema put every solver's columns on one row;
# it cannot absorb repeats without a column explosion, and _write_csv's
# extrasaction="ignore" silently drops any key missing from CSV_FIELDS, which is
# how the archived CSVs became schema-incompatible. Hence schema_version.
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
               prov, cooldown=0.0, loop_mode="shared", direct=False) -> None:
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
                         max_iter, solvers, direct))
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
    parser.add_argument("--direct", action="store_true",
                        help="Run KSP-QP with its experimental direct solver (LDLT on K or "
                             "Cholesky on its Schur complement) instead of PCG")
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
          f"tol={args.tol:g}  solvers={','.join(solvers)}  "
          f"ksp-qp={'direct' if args.direct else 'pcg'}")
    for i, (M, N) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}]")
        run_config(M, N, args.T, args.reps, args.tol, args.time_limit, max_iter,
                   solvers, rows, csv_path, prov, args.cooldown, args.loop_mode, args.direct)

    print(f"\nResults written to: {csv_path}")


if __name__ == "__main__":
    main()
