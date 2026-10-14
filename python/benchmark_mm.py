"""
Benchmark KSP-QP vs QPALM vs OSQP on the Maros-Meszaros QP test set.

Outputs
-------
  results/mm_kspqp.csv, results/mm_qpalm.csv, results/mm_osqp.csv
                                               - one file per solver: per-problem timing, iteration
                                                 counts, status, and the run's provenance
  results/performance_profile_mm.pdf/png       - Dolan-Moré performance profile (run time)
  results/performance_profile_mm_iters.pdf/png - Dolan-Moré performance profile (iterations)
  results/performance_profile_mm_inner_iters.pdf/png - Dolan-Moré performance profile (inner iterations)

  Each solver only writes its own file, and a rerun replaces the rows of the problems it solved
  while keeping the others. KSP-QP can therefore be rerun after every change while the QPALM and
  OSQP results are kept. Joining the files on "name" gives the former combined comparison table.

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
  python3 benchmark_mm.py

Settings: tol = 1e-6, time limit = 60 s, max iterations = infinity.
        --root:       to change the output directory (default: results/).
        --out:        to set an output file prefix (e.g. '0508' → '0508_mm_kspqp.csv'; default: none).
        --solver:     to select which solvers to run among ksp-qp, qpalm, osqp (default: all three).
        --problems:   to run only the named problems (default: all).
        --tol:        to change the solver tolerance (default: 1e-6).
        --time-limit: to change the solver time limit in seconds (default: 60).
        --cooldown:   to change the cooldown time in seconds between solver runs (default: 0).
        --kspqp-csv, --qpalm-csv, --osqp-csv:
                      result files of other runs to compare against in the performance profiles,
                      e.g. a KSP-QP-only run against an earlier QPALM run (default: this prefix's
                      files, where they exist).

  Example: rerun KSP-QP only, against QPALM/OSQP results kept from an earlier run:
    python3 benchmark_mm.py --solver ksp-qp --out 1015 \\
        --qpalm-csv ../results/1003_mm_qpalm.csv --osqp-csv ../results/1003_mm_osqp.csv
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
    provenance,
    check_header,
    replace_rows,
    load_solver_tables,
    plot_performance_profile,
    plot_performance_profile_iters,
    plot_performance_profile_inner_iters,
)

# ---------------------------------------------------------------------------
# Maros-Meszaros problem list  (name → reference optimal objective)
# ---------------------------------------------------------------------------
QPS = {
    "AUG2D":     1.6874118e+06,
    "AUG2DC":    1.8183681e+06,
    "AUG2DCQP":  6.4981348e+06,
    "AUG2DQP":   6.2370121e+06,
    "AUG3D":     5.5406773e+02,
    "AUG3DC":    7.7126244e+02,
    "AUG3DCQP":  9.9336215e+02,
    "AUG3DQP":   6.7523767e+02,
    "BOYD1":    -6.1735220e+07,
    "BOYD2":     2.1256767e+01,
    "CONT-050": -4.5638509e+00,
    "CONT-100": -4.6443979e+00,
    "CONT-101":  1.9552733e-01,
    "CONT-200": -4.6848759e+00,
    "CONT-201":  1.9248337e-01,
    "CONT-300":  1.9151232e-01,
    "CVXQP1L":   1.0870480e+08,
    "CVXQP1M":   1.0875116e+06,
    "CVXQP1S":   1.1590718e+04,
    "CVXQP2L":   8.1842458e+07,
    "CVXQP2M":   8.2015543e+05,
    "CVXQP2S":   8.1209405e+03,
    "CVXQP3L":   1.1571110e+08,
    "CVXQP3M":   1.3628287e+06,
    "CVXQP3S":   1.1943432e+04,
    "DPKLO1":    3.7009622e-01,
    "DTOC3":     2.3526248e+02,
    "DUAL1":     3.5012966e-02,
    "DUAL2":     3.3733676e-02,
    "DUAL3":     1.3575584e-01,
    "DUAL4":     7.4609084e-01,
    "DUALC1":    6.1552508e+03,
    "DUALC2":    3.5513077e+03,
    "DUALC5":    4.2723233e+02,
    "DUALC8":    1.8309359e+04,
    "EXDATA":   -1.4184343e+02,
    "GENHS28":   9.2717369e-01,
    "GOULDQP2":  1.8427534e-04,
    "GOULDQP3":  2.0627840e+00,
    "HS118":     6.6482045e+02,
    "HS21":     -9.9960000e+01,
    "HS268":     5.7310705e-07,
    "HS35":      1.1111111e-01,
    "HS35MOD":   2.5000000e-01,
    "HS51":      8.8817842e-16,
    "HS52":      5.3266476e+00,
    "HS53":      4.0930233e+00,
    "HS76":     -4.6818182e+00,
    "HUES-MOD":  3.4824690e+07,
    "HUESTIS":   3.4824690e+11,
    "KSIP":      5.7579794e-01,
    "LASER":     2.4096014e+06,
    "LISWET1":   3.6122402e+01,
    "LISWET10":  4.9485785e+01,
    "LISWET11":  4.9523957e+01,
    "LISWET12":  1.7369274e+03,
    "LISWET2":   2.4998076e+01,
    "LISWET3":   2.5001220e+01,
    "LISWET4":   2.5000112e+01,
    "LISWET5":   2.5034253e+01,
    "LISWET6":   2.4995748e+01,
    "LISWET7":   4.9884089e+02,
    "LISWET8":   7.1447006e+03,
    "LISWET9":   1.9632513e+03,
    "LOTSCHD":   2.3984159e+03,
    "MOSARQP1": -9.5287544e+02,
    "MOSARQP2": -1.5974821e+03,
    "POWELL20":  5.2089583e+10,
    "PRIMAL1":  -3.5012965e-02,
    "PRIMAL2":  -3.3733676e-02,
    "PRIMAL3":  -1.3575584e-01,
    "PRIMAL4":  -7.4609083e-01,
    "PRIMALC1": -6.1552508e+03,
    "PRIMALC2": -3.5513077e+03,
    "PRIMALC5": -4.2723233e+02,
    "PRIMALC8": -1.8309430e+04,
    "Q25FV47":   1.3744448e+07,
    "QADLITTL":  4.8031886e+05,
    "QAFIRO":   -1.5907818e+00,
    "QBANDM":    1.6352342e+04,
    "QBEACONF":  1.6471206e+05,
    "QBORE3D":   3.1002008e+03,
    "QBRANDY":   2.8375115e+04,
    "QCAPRI":    6.6793293e+07,
    "QE226":     2.1265343e+02,
    "QETAMACR":  8.6760370e+04,
    "QFFFFF80":  8.7314747e+05,
    "QFORPLAN":  7.4566315e+09,
    "QGFRDXPN":  1.0079059e+11,
    "QGROW15":  -1.0169364e+08,
    "QGROW22":  -1.4962895e+08,
    "QGROW7":   -4.2798714e+07,
    "QISRAEL":   2.5347838e+07,
    "QPCBLEND": -7.8425409e-03,
    "QPCBOEI1":  1.1503914e+07,
    "QPCBOEI2":  8.1719623e+06,
    "QPCSTAIR":  6.2043875e+06,
    "QPILOTNO":  4.7285869e+06,
    "QPTEST":    4.3718750e+00,
    "QRECIPE":  -2.6661600e+02,
    "QSC205":   -5.8139518e-03,
    "QSCAGR25":  2.0173794e+08,
    "QSCAGR7":   2.6865949e+07,
    "QSCFXM1":   1.6882692e+07,
    "QSCFXM2":   2.7776162e+07,
    "QSCFXM3":   3.0816355e+07,
    "QSCORPIO":  1.8805096e+03,
    "QSCRS8":    9.0456001e+02,
    "QSCSD1":    8.6666667e+00,
    "QSCSD6":    5.0808214e+01,
    "QSCSD8":    9.4076357e+02,
    "QSCTAP1":   1.4158611e+03,
    "QSCTAP2":   1.7350265e+03,
    "QSCTAP3":   1.4387547e+03,
    "QSEBA":     8.1481801e+07,
    "QSHARE1B":  7.2007832e+05,
    "QSHARE2B":  1.1703692e+04,
    "QSHELL":    1.5726368e+12,
    "QSHIP04L":  2.4200155e+06,
    "QSHIP04S":  2.4249937e+06,
    "QSHIP08L":  2.3760406e+06,
    "QSHIP08S":  2.3857289e+06,
    "QSHIP12L":  3.0188766e+06,
    "QSHIP12S":  3.0569623e+06,
    "QSIERRA":   2.3750458e+07,
    "QSTAIR":    7.9854528e+06,
    "QSTANDAT":  6.4118384e+03,
    "S268":      5.7310705e-07,
    "STADAT1":  -2.8526864e+07,
    "STADAT2":  -3.2626665e+01,
    "STADAT3":  -3.5779453e+01,
    "STCQP1":    1.5514356e+05,
    "STCQP2":    2.2327313e+04,
    "TAME":      0.0000000e+00,
    "UBH1":      1.1160008e+00,
    "VALUES":   -1.3966211e+00,
    "YAO":       1.9770426e+02,
    "ZECEVIC2": -4.1250000e+00,
}

# ---------------------------------------------------------------------------
# Per-solver result files. The column names are those of the former combined
# comparison_mm.csv (ssn_* for KSP-QP, qpalm_*, osqp_*), so joining the files on
# "name" gives the same wide table; each file also records its run's provenance.
# ---------------------------------------------------------------------------
SOLVER_FILES = {"ksp-qp": "kspqp", "qpalm": "qpalm", "osqp": "osqp"}

FIELDS = {
    "ksp-qp": ["name",
               "ssn_solved", "ssn_status", "pmm_iter", "ssn_iter", "krylov_iter", "fact", "smw_count",
               "ssn_obj", "pmm_tol_achieved", "ssn_time", "ssn_setup_time", "ssn_solve_time",
               "ssn_tol", "ssn_time_limit",
               "ssn_run_id", "ssn_timestamp", "ssn_host", "ssn_cpu", "ssn_git_sha"],
    "qpalm":  ["name",
               "qpalm_solved", "qpalm_status", "qpalm_iter", "qpalm_inner_iter",
               "qpalm_obj", "qpalm_tol_achieved", "qpalm_time", "qpalm_setup_time", "qpalm_solve_time",
               "qpalm_tol", "qpalm_time_limit", "qpalm_eps_inf",
               "qpalm_run_id", "qpalm_timestamp", "qpalm_host", "qpalm_cpu", "qpalm_version"],
    "osqp":   ["name",
               "osqp_solved", "osqp_status", "osqp_iter",
               "osqp_obj", "osqp_tol_achieved", "osqp_time", "osqp_setup_time", "osqp_solve_time",
               "osqp_tol", "osqp_time_limit", "osqp_eps_inf",
               "osqp_run_id", "osqp_timestamp", "osqp_host", "osqp_cpu", "osqp_version"],
}

# ---------------------------------------------------------------------------
# Subprocess worker functions (each spawned in a fresh process for clean RSS)
# ---------------------------------------------------------------------------

def _worker_ssn_mm(sif_path, tol, time_limit, max_iter, conn):
    result = {}
    try:
        pd_data = ksp_qp_bind.parse_sif(sif_path)
        result["res"] = ksp_qp_bind.solve_from_data(pd_data, tol, max_iter, time_limit)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_qpalm_mm(sif_path, tol, time_limit, eps_inf, conn):
    result = {}
    try:
        pd_data = ksp_qp_bind.parse_sif(sif_path)
        qpalm_data = kspqp_to_qpalm(pd_data)
        result["res"] = run_qpalm(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0), eps_inf)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_osqp_mm(sif_path, tol, time_limit, eps_inf, conn):
    result = {}
    try:
        pd_data = ksp_qp_bind.parse_sif(sif_path)
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
        "--out", default="", help="Prefix for output filenames (e.g. '0508' → '0508_mm_kspqp.csv')"
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
    parser.add_argument(
        "--problems", nargs="+", default=None, metavar="NAME",
        help="Run only these problems (default: all). Their rows are replaced; the others are kept.",
    )
    for s, tag in SOLVER_FILES.items():
        parser.add_argument(
            f"--{tag}-csv", default="", metavar="FILE",
            help=f"{s} result file to use in the performance profiles instead of this prefix's "
                 f"mm_{tag}.csv (e.g. from an earlier run)",
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
    data_dir  = root / "data" / "maros_meszaros"
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
    paths = {s: result_dir / f"{prefix}mm_{tag}.csv" for s, tag in SOLVER_FILES.items()}
    for s in solvers:
        check_header(paths[s], FIELDS[s])

    if args.problems:
        unknown = sorted(set(args.problems) - set(QPS))
        if unknown:
            sys.exit(f"Unknown problem name(s): {', '.join(unknown)}")
    names = [n for n in QPS if not args.problems or n in args.problems]

    # Provenance recorded on every row this run writes, per solver.
    prov = provenance()
    run_cols = {
        "ksp-qp": {"ssn_tol": tol, "ssn_time_limit": time_limit,
                   "ssn_run_id": prov["run_id"], "ssn_timestamp": prov["timestamp"],
                   "ssn_host": prov["host"], "ssn_cpu": prov["cpu"], "ssn_git_sha": prov["git_sha"]},
        "qpalm":  {"qpalm_tol": tol, "qpalm_time_limit": time_limit,
                   "qpalm_eps_inf": "" if eps_inf is None else eps_inf,
                   "qpalm_run_id": prov["run_id"], "qpalm_timestamp": prov["timestamp"],
                   "qpalm_host": prov["host"], "qpalm_cpu": prov["cpu"],
                   "qpalm_version": prov["qpalm_version"]},
        "osqp":   {"osqp_tol": tol, "osqp_time_limit": time_limit,
                   "osqp_eps_inf": "" if eps_inf is None else eps_inf,
                   "osqp_run_id": prov["run_id"], "osqp_timestamp": prov["timestamp"],
                   "osqp_host": prov["host"], "osqp_cpu": prov["cpu"],
                   "osqp_version": prov["osqp_version"]},
    }

    def _record(solver: str, row: dict) -> None:
        """Write this problem's row into the solver's file, replacing an earlier one."""
        replace_rows(paths[solver], [{**row, **run_cols[solver]}], ["name"], FIELDS[solver])
        if cooldown > 0:
            time.sleep(cooldown)

    n_problems = len(names)
    for idx, name in enumerate(names, 1):
        sif_path = str(data_dir / f"{name}.SIF")
        if not os.path.exists(sif_path):
            print(f"[{idx:3d}/{n_problems}] SKIP (file not found): {name}")
            continue

        print(f"\n[{idx:3d}/{n_problems}]  {name}")

        # ---- KSP-QP ------------------------------------------------
        if "ksp-qp" in solvers:
            row: dict = {"name": name}
            ssn_out = _run_isolated(_worker_ssn_mm, (sif_path, tol, time_limit, max_iter))
            if "error" in ssn_out:
                print(f"  KSP-QP : ERROR — {ssn_out['error']}")
                row.update(ssn_status=-99, ssn_solved=0, ssn_time=np.inf,
                           pmm_iter=np.inf, ssn_iter=np.inf,
                           krylov_iter=np.inf, fact=np.inf, smw_count=np.inf,
                           ssn_obj=np.nan, pmm_tol_achieved=np.nan,
                           ssn_setup_time=np.nan, ssn_solve_time=np.nan)
            else:
                r = ssn_out["res"]
                row["ssn_status"]       = r["status"]
                row["ssn_solved"]       = int(r["status"] == 0)
                row["ssn_time"]         = r["run_time"]
                row["ssn_setup_time"]   = r["setup_time"]
                row["ssn_solve_time"]   = r["solve_time"]
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
            _record("ksp-qp", row)

        # ---- QPALM --------------------------------------------------
        if "qpalm" in solvers:
            row = {"name": name}
            qpalm_out = _run_isolated(_worker_qpalm_mm, (sif_path, tol, time_limit, eps_inf))
            if "error" in qpalm_out:
                print(f"  QPALM   : ERROR — {qpalm_out['error']}")
                row.update(qpalm_status=-99, qpalm_solved=0, qpalm_time=np.inf,
                           qpalm_iter=np.inf, qpalm_inner_iter=np.inf,
                           qpalm_obj=np.nan, qpalm_tol_achieved=np.nan,
                           qpalm_setup_time=np.nan, qpalm_solve_time=np.nan)
            else:
                r = qpalm_out["res"]
                row["qpalm_status"]        = r["status"]
                row["qpalm_solved"]        = int(r["status"] == QPALM_SOLVED)
                row["qpalm_time"]          = r["run_time"]
                row["qpalm_setup_time"]    = r["setup_time"]
                row["qpalm_solve_time"]    = r["solve_time"]
                row["qpalm_iter"]          = r["outer_iter"]
                row["qpalm_inner_iter"]    = r["inner_iter"]
                row["qpalm_obj"]           = r["obj_val"]
                row["qpalm_tol_achieved"]  = r["tol_achieved"]
                status_str = "OPTIMAL" if r["status"] == QPALM_SOLVED else f"status={r['status']}"
                print(f"  QPALM   : {status_str:12s}  t = {r['run_time']:.3f} s  "
                      f"outer={r['outer_iter']} inner={r['inner_iter']}  "
                      f"tol={r['tol_achieved']:.2e}  obj = {r['obj_val']:.6g}")
            _record("qpalm", row)

        # ---- OSQP ---------------------------------------------------
        if "osqp" in solvers:
            row = {"name": name}
            osqp_out = _run_isolated(_worker_osqp_mm, (sif_path, tol, time_limit, eps_inf))
            if "error" in osqp_out:
                print(f"  OSQP    : ERROR — {osqp_out['error']}")
                row.update(osqp_status=-99, osqp_solved=0, osqp_time=np.inf, osqp_iter=np.inf,
                           osqp_obj=np.nan, osqp_tol_achieved=np.nan,
                           osqp_setup_time=np.nan, osqp_solve_time=np.nan)
            else:
                r = osqp_out["res"]
                row["osqp_status"]       = r["status"]
                row["osqp_solved"]       = int(r["status"] == OSQP_SOLVED)
                row["osqp_time"]         = r["run_time"]
                row["osqp_setup_time"]   = r["setup_time"]
                row["osqp_solve_time"]   = r["solve_time"]
                row["osqp_iter"]         = r["outer_iter"]
                row["osqp_obj"]          = r["obj_val"]
                row["osqp_tol_achieved"] = r["tol_achieved"]
                status_str = "OPTIMAL" if r["status"] == OSQP_SOLVED else f"status={r['status']}"
                print(f"  OSQP    : {status_str:12s}  t = {r['run_time']:.3f} s  "
                      f"iter = {r['outer_iter']}  tol={r['tol_achieved']:.2e}  obj = {r['obj_val']:.6g}")
            _record("osqp", row)

    print("\nResults written to: " + ", ".join(str(paths[s]) for s in SOLVER_FILES if s in solvers))

    # ---- Performance profiles -------------------------------------------
    # Every solver with a result file: the one given by --<solver>-csv, else this prefix's own.
    # Only the problems present in all of them are compared.
    explicit = {s: getattr(args, f"{tag}_csv") for s, tag in SOLVER_FILES.items()}
    profile_paths = {}
    for s in SOLVER_FILES:
        p = Path(explicit[s]) if explicit[s] else paths[s]
        if p.exists():
            profile_paths[s] = p
        elif explicit[s]:
            sys.exit(f"--{SOLVER_FILES[s]}-csv: file not found: {p}")
    if not profile_paths:
        return
    df = load_solver_tables(profile_paths, ["name"])
    print("Performance profiles over " + f"{len(df)} problems from: "
          + ", ".join(f"{s}={p}" for s, p in profile_paths.items()))
    profile_solvers = set(profile_paths)

    label = "Maros-Meszaros QPs"
    out_prefix = result_dir / f"{prefix}performance_profile_mm"
    plot_performance_profile(df, out_prefix, label, tol=tol, time_limit=time_limit, solvers=profile_solvers)

    out_prefix_iters = result_dir / f"{prefix}performance_profile_mm_iters"
    plot_performance_profile_iters(df, out_prefix_iters, label, tol=tol, time_limit=time_limit, solvers=profile_solvers)

    out_prefix_inner = result_dir / f"{prefix}performance_profile_mm_inner_iters"
    plot_performance_profile_inner_iters(df, out_prefix_inner, label, tol=tol, time_limit=time_limit, solvers=profile_solvers)


if __name__ == "__main__":
    main()
