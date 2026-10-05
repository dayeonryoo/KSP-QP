"""
Benchmark KSP-QP vs QPALM vs OSQP on the feasible Netlib LP test set.

Problems are read from data/netlib-main/feasible/, which holds all 114 feasible
Netlib instances with the Kennington family blended in.

Outputs
-------
  results/comparison_netlib.csv                    - per-problem timing, iteration counts, and status
  results/performance_profile_netlib.pdf/png       - Dolan-Moré performance profile (run time)
  results/performance_profile_netlib_iters.pdf/png - Dolan-Moré performance profile (iterations)

=== HOW TO RUN FROM SCRATCH ===

Step 1 - Install Python dependencies
-------------------------------------
  pip install qpalm osqp numpy scipy matplotlib pandas

Step 2 - Build the KSP-QP Python binding
------------------------------------------
  All commands are run from the KSP-QP/python/ directory.

  mkdir build
  cd build
  cmake .. -DPython3_EXECUTABLE=$(which python3)
  cmake --build . --config Release
  cd ..

  This produces ksp_qp_bind.cpython-<tag>-darwin.so in python/.
  You only need to rebuild if the C++ solver source changes.

Step 3 - Run the benchmark
---------------------------
  python3 benchmark_netlib.py

Settings: tol = 1e-6, time limit = 60 s, max iterations = infinity.
        --root:       to change the output directory (default: results/).
        --out:        to change the output file prefix (default: comparison_netlib).
        --solver:     to select which solvers to run among ksp-qp, qpalm, osqp (default: all three).
        --tol:        to change the solver tolerance (default: 1e-6).
        --time-limit: to change the solver time limit in seconds (default: 60).
        --cooldown:   to change the cooldown time in seconds between solver runs (default: 0).
"""

import sys
import os
import time
import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Locate and import the pybind11 extension
# ---------------------------------------------------------------------------
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
    run_qpalm,
    run_osqp,
    _run_isolated,
    INF_TOL_FACTOR,
    QPALM_SOLVED,
    OSQP_SOLVED,
    _write_csv,
    _load_existing_rows,
    plot_performance_profile,
    plot_performance_profile_iters,
    plot_performance_profile_inner_iters,
)

# ---------------------------------------------------------------------------
# Problem set.
#
# The .mps files in data/netlib-main/feasible/: all 114 feasible Netlib LPs,
# including the kennington family (CRE-*, KEN-*, OSA-*, PDS-*). Filenames are
# lowercase; problem names are the uppercased stems (vtp.base.mps -> VTP.BASE).
#
# Reference objectives come from data/netlib-main/feasible_gurobi_1e-8.csv
# (Gurobi 10 at 1e-8); they differ from the published Netlib optima on a few
# problems (e.g. E226, CRE-A).
# ---------------------------------------------------------------------------
FEASIBLE_SUBDIR = "netlib-main/feasible"

# ---------------------------------------------------------------------------
# Subprocess worker functions (each spawned in a fresh process for clean RSS)
# ---------------------------------------------------------------------------

def _worker_ssn_netlib(mps_path, tol, time_limit, max_iter, conn):
    result = {}
    try:
        pd_data = ksp_qp_bind.parse_sif(mps_path)
        result["res"] = ksp_qp_bind.solve_from_data(pd_data, tol, max_iter, time_limit)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_qpalm_netlib(mps_path, tol, time_limit, eps_inf, conn):
    result = {}
    try:
        pd_data = ksp_qp_bind.parse_sif(mps_path)
        qpalm_data = kspqp_to_qpalm(pd_data)
        result["res"] = run_qpalm(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0), eps_inf)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_osqp_netlib(mps_path, tol, time_limit, eps_inf, conn):
    result = {}
    try:
        pd_data = ksp_qp_bind.parse_sif(mps_path)
        qpalm_data = kspqp_to_qpalm(pd_data)
        result["res"] = run_osqp(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0), eps_inf)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(HERE.parent),
        help="Path to the KSP-QP project root (default: parent of this script)",
    )
    parser.add_argument(
        "--tol",        type=float, default=1e-6,  help="Primal-dual tolerance (default: 1e-6)"
    )
    parser.add_argument(
        "--time-limit", type=float, default=60.0, help="Per-problem time limit in seconds (default: 60)"
    )
    parser.add_argument(
        "--out", default="", help="Prefix for output filenames (e.g. '0508' → '0508_comparison_netlib.csv')"
    )
    parser.add_argument(
        "--solver", nargs="+", default=["ksp-qp", "qpalm", "osqp"],
        choices=["ksp-qp", "qpalm", "osqp"], metavar="SOLVER",
        help="Solvers to run (default: all three). Choices: ksp-qp qpalm osqp",
    )
    parser.add_argument(
        "--cooldown", type=float, default=0.0,
        help="Seconds to sleep between problems to prevent CPU throttling (default: 0)",
    )
    cert = parser.add_mutually_exclusive_group()
    cert.add_argument(
        "--inf-tol-factor", type=float, default=INF_TOL_FACTOR, dest="inf_tol_factor",
        help=f"QPALM/OSQP eps_prim_inf = eps_dual_inf = FACTOR * tol "
             f"(default: {INF_TOL_FACTOR:g}, matching KSP-QP's eps_pinf = 1e-3 * tol, "
             f"so all three solvers test their certificates at the same ratio).",
    )
    cert.add_argument(
        "--library-default-cert-tol", action="store_true", dest="library_cert_tol",
        help="Leave QPALM/OSQP at their shipped certificate tolerances "
             "(QPALM 1e-5, OSQP 1e-4) instead of matching KSP-QP.",
    )
    mp.set_start_method("spawn", force=True)
    args = parser.parse_args()
    solvers = set(args.solver)

    root      = Path(args.root).resolve()
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)

    # (name, path) for every .mps in the feasible set, alphabetical by filename.
    data_dir = root / "data" / FEASIBLE_SUBDIR
    if not data_dir.is_dir():
        sys.exit(f"Problem directory not found: {data_dir}")
    problems: list[tuple[str, Path]] = [
        (path.stem.upper(), path) for path in sorted(data_dir.glob("*.mps"))
    ]
    if not problems:
        sys.exit(f"No .mps files found in {data_dir}")
    suffix = "netlib"
    label  = "Netlib LPs"

    tol        = args.tol
    time_limit = args.time_limit
    cooldown   = args.cooldown
    max_iter   = 10_000_000_000   # effectively infinite for KSP-QP
    eps_inf    = None if args.library_cert_tol else args.inf_tol_factor * tol

    print(f"tol = {tol:g}, time limit = {time_limit:g} s, QPALM/OSQP certificate tolerance = "
          + ("library defaults (QPALM 1e-5, OSQP 1e-4)" if eps_inf is None
             else f"{eps_inf:g} ({args.inf_tol_factor:g} * tol, matching KSP-QP)"))

    prefix = f"{args.out}_" if args.out else ""
    csv_path = result_dir / f"{prefix}comparison_{suffix}.csv"
    fieldnames = [
        "name",
        "ssn_solved",   "ssn_status",   "pmm_iter", "ssn_iter",
        "krylov_iter",  "fact",         "smw_count",
        "ssn_obj",   "pmm_tol_achieved",   "ssn_time",
        "qpalm_solved", "qpalm_status", "qpalm_iter", "qpalm_inner_iter", "qpalm_obj", "qpalm_tol_achieved", "qpalm_time",
        "osqp_solved",  "osqp_status",  "osqp_iter",                      "osqp_obj",  "osqp_tol_achieved",  "osqp_time",
    ]

    n_problems = len(problems)
    rows: list[dict] = _load_existing_rows(csv_path)

    def _flush() -> None:
        """Rewrite the CSV from `rows` — called right after every solver finishes."""
        _write_csv(csv_path, rows, fieldnames)

    for idx, (name, path) in enumerate(problems, 1):
        mps_path = str(path)
        if not os.path.exists(mps_path):
            print(f"[{idx:3d}/{n_problems}] SKIP (file not found): {name}")
            continue

        print(f"\n[{idx:3d}/{n_problems}]  {name}")
        row: dict = {"name": name}
        rows.append(row)

        # ---- KSP-QP ------------------------------------------------
        if "ksp-qp" in solvers:
            ssn_out = _run_isolated(_worker_ssn_netlib, (mps_path, tol, time_limit, max_iter))
            if "error" in ssn_out:
                print(f"  KSP-QP : ERROR — {ssn_out['error']}")
                row.update(ssn_status=-99, ssn_solved=0, ssn_time=np.inf,
                           pmm_iter=np.inf, ssn_iter=np.inf,
                           krylov_iter=np.inf, fact=np.inf, smw_count=np.inf,
                           ssn_obj=np.nan, pmm_tol_achieved=np.nan)
            else:
                r = ssn_out["res"]
                row["ssn_status"]       = r["status"]
                row["ssn_solved"]       = int(r["status"] == 0)
                row["ssn_time"]         = r["run_time"]
                row["pmm_iter"]         = r["pmm_iter"]
                row["ssn_iter"]         = r["ssn_iter"]
                row["krylov_iter"]      = r["krylov_iter"]
                row["fact"]             = r["fact"]
                row["smw_count"]        = r["smw_count"]
                row["ssn_obj"]          = r["obj_val"]
                row["pmm_tol_achieved"] = r["pmm_tol_achieved"]
                status_str = "OPTIMAL" if r["status"] == 0 else f"status={r['status']}"
                print(f"  KSP-QP : {status_str:12s}  t = {r['run_time']:.3f} s  "
                      f"pmm={r['pmm_iter']} ssn={r['ssn_iter']} "
                      f"krylov={r['krylov_iter']} fact={r['fact']} smw={r['smw_count']}  "
                      f"tol={r['pmm_tol_achieved']:.2e}  obj = {r['obj_val']:.6g}")
            _flush()
            if cooldown > 0:
                time.sleep(cooldown)

        # ---- QPALM --------------------------------------------------
        if "qpalm" in solvers:
            qpalm_out = _run_isolated(_worker_qpalm_netlib, (mps_path, tol, time_limit, eps_inf))
            if "error" in qpalm_out:
                print(f"  QPALM   : ERROR — {qpalm_out['error']}")
                row.update(qpalm_status=-99, qpalm_solved=0, qpalm_time=np.inf,
                           qpalm_iter=np.inf, qpalm_inner_iter=np.inf,
                           qpalm_obj=np.nan, qpalm_tol_achieved=np.nan)
            else:
                r = qpalm_out["res"]
                row["qpalm_status"]        = r["status"]
                row["qpalm_solved"]        = int(r["status"] == QPALM_SOLVED)
                row["qpalm_time"]          = r["run_time"]
                row["qpalm_iter"]          = r["outer_iter"]
                row["qpalm_inner_iter"]    = r["inner_iter"]
                row["qpalm_obj"]           = r["obj_val"]
                row["qpalm_tol_achieved"]  = r["tol_achieved"]
                status_str = "OPTIMAL" if r["status"] == QPALM_SOLVED else f"status={r['status']}"
                print(f"  QPALM   : {status_str:12s}  t = {r['run_time']:.3f} s  "
                      f"outer={r['outer_iter']} inner={r['inner_iter']}  "
                      f"tol={r['tol_achieved']:.2e}  obj = {r['obj_val']:.6g}")
            _flush()
            if cooldown > 0:
                time.sleep(cooldown)

        # ---- OSQP ---------------------------------------------------
        if "osqp" in solvers:
            osqp_out = _run_isolated(_worker_osqp_netlib, (mps_path, tol, time_limit, eps_inf))
            if "error" in osqp_out:
                print(f"  OSQP    : ERROR — {osqp_out['error']}")
                row.update(osqp_status=-99, osqp_solved=0, osqp_time=np.inf, osqp_iter=np.inf,
                           osqp_obj=np.nan, osqp_tol_achieved=np.nan)
            else:
                r = osqp_out["res"]
                row["osqp_status"]       = r["status"]
                row["osqp_solved"]       = int(r["status"] == OSQP_SOLVED)
                row["osqp_time"]         = r["run_time"]
                row["osqp_iter"]         = r["outer_iter"]
                row["osqp_obj"]          = r["obj_val"]
                row["osqp_tol_achieved"] = r["tol_achieved"]
                status_str = "OPTIMAL" if r["status"] == OSQP_SOLVED else f"status={r['status']}"
                print(f"  OSQP    : {status_str:12s}  t = {r['run_time']:.3f} s  "
                      f"iter = {r['outer_iter']}  tol={r['tol_achieved']:.2e}  obj = {r['obj_val']:.6g}")
            _flush()
            if cooldown > 0:
                time.sleep(cooldown)

    print(f"\nResults written to: {csv_path}")

    # ---- Performance profiles -------------------------------------------
    out_prefix = result_dir / f"{prefix}performance_profile_{suffix}"
    plot_performance_profile(csv_path, out_prefix, label, tol=tol, time_limit=time_limit, solvers=solvers)

    out_prefix_iters = result_dir / f"{prefix}performance_profile_{suffix}_iters"
    plot_performance_profile_iters(csv_path, out_prefix_iters, label, tol=tol, time_limit=time_limit, solvers=solvers)

    out_prefix_inner = result_dir / f"{prefix}performance_profile_{suffix}_inner_iters"
    plot_performance_profile_inner_iters(csv_path, out_prefix_inner, label, tol=tol, time_limit=time_limit, solvers=solvers)


if __name__ == "__main__":
    main()
