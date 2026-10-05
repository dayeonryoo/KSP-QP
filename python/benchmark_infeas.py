"""
Benchmark KSP-QP vs QPALM vs OSQP on the Netlib *infeasible* LP set.

Every problem in data/netlib-main/infeasible/ is primal infeasible, so the metric is
detection: `*_detected` records whether a solver terminated with an infeasibility
status, and the raw `*_status` column keeps primal vs dual and the failure modes
(time limit, iteration cap, false optimality).

Outputs
-------
  results/comparison_infeas.csv - per-problem detection, status, iterations, time

=== HOW TO RUN ===

  python3 benchmark_infeas.py --out final_2_0924

Settings: tol = 1e-6, time limit = 60 s, max iterations = infinity.
        --root:       project root (default: parent of this script)
        --out:        output file prefix (default: none -> comparison_infeas.csv)
        --solver:     which solvers to run among ksp-qp, qpalm, osqp (default: all)
        --tol:        solver tolerance (default: 1e-6)
        --time-limit: per-solver time limit in seconds (default: 60)
        --cooldown:   seconds between solver runs (default: 0)
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
    _write_csv,
    _load_existing_rows,
)

# ---------------------------------------------------------------------------
# Netlib infeasible LPs (all primal infeasible).
# Same set and order as `ksp_qp_netlib --set infeasible --name all` (the sorted
# directory), so the two compare row for row.
# ---------------------------------------------------------------------------
# Filenames are lowercase (.mps); the names below are the uppercase spelling used
# in the result CSVs, so paths are built as f"{name.lower()}.mps".
INFEAS_SUBDIR = "netlib-main/infeasible"

INFEAS_LPS = [
    "BGDBG1", "BGETAM", "BGINDY", "BGPRTR", "BOX1", "CERIA3D", "CHEMCOM",
    "CPLEX1", "CPLEX2", "EX72A", "EX73A", "FOREST6", "GALENET", "GOSH",
    "GRAN", "GREENBEA", "ITEST2", "ITEST6", "KLEIN1", "KLEIN2", "KLEIN3",
    "MONDOU2", "PANG", "PILOT4I", "QUAL", "REACTOR", "REFINERY", "VOL1",
    "WOODINFE",
]

# ---------------------------------------------------------------------------
# Per-solver infeasibility statuses.
# Primal and dual infeasibility both count as a detection, as in src/netlib.cpp's
# infeasible set. OSQP's "_INACCURATE" variants count too; the raw status column
# preserves which.
# ---------------------------------------------------------------------------
KSPQP_INFEAS = {-2, -3}          # TerminationStatus::{PrimalInfeasible, DualInfeasible}
QPALM_INFEAS = {-3, -4}          # qpalm.Info.{PRIMAL_INFEASIBLE, DUAL_INFEASIBLE}
OSQP_INFEAS  = {3, 4, 5, 6}      # osqp.SolverStatus.OSQP_{PRIMAL,DUAL}_INFEASIBLE[_INACCURATE]


# ---------------------------------------------------------------------------
# Subprocess worker functions (each spawned in a fresh process for clean RSS)
# ---------------------------------------------------------------------------

def _worker_ssn_infeas(mps_path, tol, time_limit, max_iter, conn):
    result = {}
    try:
        pd_data = ksp_qp_bind.parse_sif(mps_path)
        result["res"] = ksp_qp_bind.solve_from_data(pd_data, tol, max_iter, time_limit)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_qpalm_infeas(mps_path, tol, time_limit, eps_inf, conn):
    result = {}
    try:
        pd_data = ksp_qp_bind.parse_sif(mps_path)
        qpalm_data = kspqp_to_qpalm(pd_data)
        result["res"] = run_qpalm(qpalm_data, tol, time_limit,
                                  pd_data.get("obj_const", 0.0), eps_inf)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_osqp_infeas(mps_path, tol, time_limit, eps_inf, conn):
    result = {}
    try:
        pd_data = ksp_qp_bind.parse_sif(mps_path)
        qpalm_data = kspqp_to_qpalm(pd_data)
        result["res"] = run_osqp(qpalm_data, tol, time_limit,
                                 pd_data.get("obj_const", 0.0), eps_inf)
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
        "--root", default=str(HERE.parent),
        help="Path to the KSP-QP project root (default: parent of this script)",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-6, help="Primal-dual tolerance (default: 1e-6)"
    )
    parser.add_argument(
        "--time-limit", type=float, default=60.0,
        help="Per-problem time limit in seconds (default: 60)",
    )
    parser.add_argument(
        "--out", default="",
        help="Prefix for output filenames (e.g. '0508' -> '0508_comparison_infeas.csv')",
    )
    parser.add_argument(
        "--solver", nargs="+", default=["ksp-qp", "qpalm", "osqp"],
        choices=["ksp-qp", "qpalm", "osqp"], metavar="SOLVER",
        help="Solvers to run (default: all three). Choices: ksp-qp qpalm osqp",
    )
    parser.add_argument(
        "--cooldown", type=float, default=0.0,
        help="Seconds to sleep between solver runs to prevent CPU throttling (default: 0)",
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

    root       = Path(args.root).resolve()
    data_dir   = root / "data" / INFEAS_SUBDIR
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)

    tol        = args.tol
    time_limit = args.time_limit
    cooldown   = args.cooldown
    max_iter   = 10_000_000_000   # effectively infinite for KSP-QP
    eps_inf    = None if args.library_cert_tol else args.inf_tol_factor * tol

    print(f"tol = {tol:g}, time limit = {time_limit:g} s, QPALM/OSQP certificate tolerance = "
          + ("library defaults (QPALM 1e-5, OSQP 1e-4)" if eps_inf is None
             else f"{eps_inf:g} ({args.inf_tol_factor:g} * tol, matching KSP-QP)"))

    prefix = f"{args.out}_" if args.out else ""
    csv_path = result_dir / f"{prefix}comparison_infeas.csv"
    fieldnames = [
        "name",
        "ssn_detected",   "ssn_status",   "pmm_iter", "ssn_iter",
        "krylov_iter",    "fact",         "smw_count",
        "pmm_tol_achieved", "ssn_time",
        "qpalm_detected", "qpalm_status", "qpalm_iter", "qpalm_inner_iter",
        "qpalm_tol_achieved", "qpalm_time",
        "osqp_detected",  "osqp_status",  "osqp_iter",
        "osqp_tol_achieved",  "osqp_time",
    ]

    n_problems = len(INFEAS_LPS)
    rows: list[dict] = _load_existing_rows(csv_path)

    def _flush() -> None:
        """Rewrite the CSV from `rows` — called right after every solver finishes."""
        _write_csv(csv_path, rows, fieldnames)

    n_det = {"ksp-qp": 0, "qpalm": 0, "osqp": 0}

    for idx, name in enumerate(INFEAS_LPS, 1):
        mps_path = str(data_dir / f"{name.lower()}.mps")
        if not os.path.exists(mps_path):
            print(f"[{idx:3d}/{n_problems}] SKIP (file not found): {name}", flush=True)
            continue

        print(f"\n[{idx:3d}/{n_problems}]  {name}", flush=True)
        row: dict = {"name": name}
        rows.append(row)

        # ---- KSP-QP ------------------------------------------------
        if "ksp-qp" in solvers:
            out = _run_isolated(_worker_ssn_infeas, (mps_path, tol, time_limit, max_iter))
            if "error" in out:
                print(f"  KSP-QP : ERROR - {out['error']}", flush=True)
                row.update(ssn_status=-99, ssn_detected=0, ssn_time=np.inf,
                           pmm_iter=np.inf, ssn_iter=np.inf,
                           krylov_iter=np.inf, fact=np.inf, smw_count=np.inf,
                           pmm_tol_achieved=np.nan)
            else:
                r = out["res"]
                detected = int(r["status"] in KSPQP_INFEAS)
                n_det["ksp-qp"] += detected
                row["ssn_status"]       = r["status"]
                row["ssn_detected"]     = detected
                row["ssn_time"]         = r["run_time"]
                row["pmm_iter"]         = r["pmm_iter"]
                row["ssn_iter"]         = r["ssn_iter"]
                row["krylov_iter"]      = r["krylov_iter"]
                row["fact"]             = r["fact"]
                row["smw_count"]        = r["smw_count"]
                row["pmm_tol_achieved"] = r["pmm_tol_achieved"]
                print(f"  KSP-QP : {'INFEAS' if detected else 'MISS':12s}  "
                      f"status={r['status']:<3d} t = {r['run_time']:.3f} s  "
                      f"pmm={r['pmm_iter']} ssn={r['ssn_iter']} "
                      f"krylov={r['krylov_iter']} fact={r['fact']} smw={r['smw_count']}",
                      flush=True)
            _flush()
            if cooldown > 0:
                time.sleep(cooldown)

        # ---- QPALM --------------------------------------------------
        if "qpalm" in solvers:
            out = _run_isolated(_worker_qpalm_infeas, (mps_path, tol, time_limit, eps_inf))
            if "error" in out:
                print(f"  QPALM  : ERROR - {out['error']}", flush=True)
                row.update(qpalm_status=-99, qpalm_detected=0, qpalm_time=np.inf,
                           qpalm_iter=np.inf, qpalm_inner_iter=np.inf,
                           qpalm_tol_achieved=np.nan)
            else:
                r = out["res"]
                detected = int(r["status"] in QPALM_INFEAS)
                n_det["qpalm"] += detected
                row["qpalm_status"]       = r["status"]
                row["qpalm_detected"]     = detected
                row["qpalm_time"]         = r["run_time"]
                row["qpalm_iter"]         = r["outer_iter"]
                row["qpalm_inner_iter"]   = r["inner_iter"]
                row["qpalm_tol_achieved"] = r["tol_achieved"]
                print(f"  QPALM  : {'INFEAS' if detected else 'MISS':12s}  "
                      f"status={r['status']:<3d} t = {r['run_time']:.3f} s  "
                      f"outer={r['outer_iter']} inner={r['inner_iter']}", flush=True)
            _flush()
            if cooldown > 0:
                time.sleep(cooldown)

        # ---- OSQP ---------------------------------------------------
        if "osqp" in solvers:
            out = _run_isolated(_worker_osqp_infeas, (mps_path, tol, time_limit, eps_inf))
            if "error" in out:
                print(f"  OSQP   : ERROR - {out['error']}", flush=True)
                row.update(osqp_status=-99, osqp_detected=0, osqp_time=np.inf,
                           osqp_iter=np.inf, osqp_tol_achieved=np.nan)
            else:
                r = out["res"]
                detected = int(r["status"] in OSQP_INFEAS)
                n_det["osqp"] += detected
                row["osqp_status"]       = r["status"]
                row["osqp_detected"]     = detected
                row["osqp_time"]         = r["run_time"]
                row["osqp_iter"]         = r["outer_iter"]
                row["osqp_tol_achieved"] = r["tol_achieved"]
                print(f"  OSQP   : {'INFEAS' if detected else 'MISS':12s}  "
                      f"status={r['status']:<3d} t = {r['run_time']:.3f} s  "
                      f"iter={r['outer_iter']}", flush=True)
            _flush()
            if cooldown > 0:
                time.sleep(cooldown)

    print("\n" + "=" * 60)
    print(f"Infeasibility detected (out of {n_problems} primal-infeasible LPs):")
    for s in ("ksp-qp", "qpalm", "osqp"):
        if s in solvers:
            print(f"  {s:8s} {n_det[s]:3d}/{n_problems}")
    print(f"\nResults written to: {csv_path}")


if __name__ == "__main__":
    main()
