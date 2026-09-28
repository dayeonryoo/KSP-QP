"""
Benchmark KSP-QP vs QPALM vs OSQP on L2-regularized PDE-constrained QP problems
(see pde_generator.py; problems from Pearson & Gondzio, 2017).

Three named tables are produced:

  poisson_control     2D Poisson control, control-constrained
                      (u >= 0, u <= per-beta upper bound)

  poisson_state       2D Poisson control, state-constrained
                      (-0.1 <= y <= per-beta upper bound)

  convdiff_both       2D convection-diffusion control,
                      both state and control bounds

Outputs
-------
  results/l2_poisson_control.csv
  results/l2_poisson_state.csv
  results/l2_convdiff_both.csv

=== HOW TO RUN FROM SCRATCH ===

Step 1 - Install dependencies
----------------------------
  pip install numpy scipy qpalm osqp

Step 2 - Build the KSP-QP Python binding (if not already built)
-----------------------------------------------------------------
  cd python && mkdir -p build && cd build
  cmake .. -DPython3_EXECUTABLE=$(which python3)
  cmake --build . --config Release
  cd ..

Step 3 - Run the benchmark
---------------------------
python3 benchmark_l2pde.py

Settings: tol = 1e-6, time limit = 600 s (10 min), max iterations = infinity by default.
        --root:       to change the output directory (default: results/).
        --out:        to prepend a prefix to the output filenames (default: none, i.e. l2_*.csv).
        --solver:     to select which solvers to run among ksp-qp, qpalm, osqp (default: all three).
        --table:      to select which tables to run among poisson_control, poisson_state, convdiff_both (default: all three).
        --nc:         to select which grid exponents to run for vary-n tables (default: 5 6 7 8 9 10; see sweep parameters).
        --tol:        to change the solver tolerance (default: 1e-6).
        --time-limit: to change the solver time limit in seconds (default: 600).
        --cooldown:   to change the cooldown time in seconds between solver runs (default: 0).
        --lumped-mass: 0 to use the consistent mass matrix (default), 1 to use the lumped mass matrix.
        --discretization: 'fem' (default) or 'fd' spatial discretization for the PDE operator.
"""

import sys
import math
import argparse
import multiprocessing as mp
from pathlib import Path

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

import pde_generator

from benchmark_common import (
    kspqp_to_qpalm,
    run_qpalm,
    run_osqp,
    run_solvers,
    _write_csv,
    _load_existing_rows,
)

# ---------------------------------------------------------------------------
# Sweep parameters: See (Pearson & Gondzio 2017) for the original tables and parameter choices.
# ---------------------------------------------------------------------------

DEFAULT_EPS = 0.01
INF = math.inf

# 2D Poisson control, control-constrained (0 <= u <= u_upper)
TABLE1_NC    = [5, 6, 7, 8, 9, 10]
TABLE1_BETAS = [1e-2]
TABLE1_U_UPPER = {1e-2: 1.0}
# TABLE1_BETAS = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
# TABLE1_U_UPPER = {
#     1.0: 0.01, 1e-1: 0.1, 1e-2: 1.0, 1e-3: 3.0, 1e-4: 20.0, 1e-5: 100.0, 1e-6: 300.0,
# }

# 2D Poisson control, state-constrained (-0.1 <= y <= y_upper)
TABLE2_NC    = [5, 6, 7, 8, 9, 10]
TABLE2_BETAS = [1e-2]
TABLE2_Y_UPPER = {1e-2: 0.175}
# TABLE2_BETAS = [1.0, 1e-2, 1e-4, 1e-6]
# TABLE2_Y_UPPER = {1.0: 0.002, 1e-2: 0.175, 1e-4: 0.9, 1e-6: 1.0}

# 2D convection-diffusion control, both state and control bounds
TABLE3_NC    = [5, 6, 7, 8, 9, 10]
TABLE3_BETAS = [1e-2]
TABLE3_Y_UPPER = {1e-2: 0.5}
TABLE3_U_BOUND = {1e-2: 2.0}
# TABLE3_BETAS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
# TABLE3_Y_UPPER = {1e-1: 0.2, 1e-2: 0.5, 1e-3: 0.5, 1e-4: 0.75, 1e-5: 0.75}
# TABLE3_U_BOUND = {1e-1: 0.75, 1e-2: 2.0, 1e-3: 3.0, 1e-4: 5.0, 1e-5: 6.0}

ALL_TABLES = ["poisson_control", "poisson_state", "convdiff_both"]


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _n_display(nc: int) -> int:
    """n = 2 * np (x = [y; u]), np = (2^nc+1)^2."""
    n1d = 2 ** nc + 1
    return 2 * (n1d ** 2)


def _fmt_bound(v: float) -> str:
    if v == INF:
        return "+inf"
    if v == -INF:
        return "-inf"
    return f"{v:g}"


# ---------------------------------------------------------------------------
# Subprocess worker functions
# ---------------------------------------------------------------------------

def _generate(choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, lumped_mass, discretization):
    return pde_generator.generate_pde_l2_qp(
        choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps,
        lumped_mass=lumped_mass, discretization=discretization
    )


def _worker_ssn(choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, lumped_mass, discretization,
                 tol, time_limit, max_iter, conn):
    result = {}
    try:
        pd_data = _generate(choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, lumped_mass,
                             discretization)
        result["n_vars"] = pd_data["n"]
        result["res"] = ksp_qp_bind.solve_from_data(pd_data, tol, max_iter, time_limit)
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_qpalm(choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, lumped_mass, discretization,
                   tol, time_limit, conn):
    result = {}
    try:
        pd_data = _generate(choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, lumped_mass,
                             discretization)
        qpalm_data = kspqp_to_qpalm(pd_data)
        result["res"] = run_qpalm(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0))
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


def _worker_osqp(choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, lumped_mass, discretization,
                  tol, time_limit, conn):
    result = {}
    try:
        pd_data = _generate(choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, lumped_mass,
                             discretization)
        qpalm_data = kspqp_to_qpalm(pd_data)
        result["res"] = run_osqp(qpalm_data, tol, time_limit, pd_data.get("obj_const", 0.0))
    except Exception as e:
        result["error"] = str(e)
    conn.send(result)
    conn.close()


# ---------------------------------------------------------------------------
# Core per-problem runner
# ---------------------------------------------------------------------------

def run_one(result: dict, choice: str, nc: int, beta: float,
            y_lower: float, y_upper: float, u_lower: float, u_upper: float,
            eps: float, lumped_mass: bool, discretization: str,
            tol: float, time_limit: float, max_iter: int, solvers: set,
            cooldown: float = 0.0, flush_cb=None) -> dict:
    """Solve one problem with all three solvers in isolated subprocesses."""
    n_disp = _n_display(nc)
    print(f"  {choice} nc={nc} (n={n_disp:.2e})  alpha2={beta:.0e}  "
          f"y=[{_fmt_bound(y_lower)},{_fmt_bound(y_upper)}]  "
          f"u=[{_fmt_bound(u_lower)},{_fmt_bound(u_upper)}]  "
          f"mass={'lumped' if lumped_mass else 'consistent'}  "
          f"disc={discretization}", flush=True)

    worker_args = (choice, nc, beta, y_lower, y_upper, u_lower, u_upper, eps, lumped_mass,
                   discretization)
    return run_solvers(result, worker_args, _worker_ssn, _worker_qpalm, _worker_osqp,
                       tol, time_limit, max_iter, solvers, cooldown, flush_cb)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "table", "choice", "nc", "n_display", "alpha2",
    "y_lower", "y_upper", "u_lower", "u_upper", "lumped_mass", "discretization",
    "ssn_status", "ssn_solved", "pmm_iter", "ssn_iter",
    "krylov_iter", "fact", "smw_count", "pmm_tol_achieved", "ssn_time", "ssn_obj",
    "qpalm_status", "qpalm_solved", "qpalm_iter", "qpalm_inner_iter", "qpalm_tol_achieved", "qpalm_time", "qpalm_obj",
    "osqp_status",  "osqp_solved",  "osqp_iter",  "osqp_tol_achieved",  "osqp_time",  "osqp_obj",
]


# ---------------------------------------------------------------------------
# Table runners
# ---------------------------------------------------------------------------

def _run_poisson_control(nc_list, betas, lumped_mass, discretization, tol, time_limit, max_iter,
                         result_dir, solvers, cooldown, name_prefix):
    label = "poisson_control"
    csv_path = result_dir / f"{name_prefix}l2_{label}.csv"
    rows: list[dict] = _load_existing_rows(csv_path)
    n_total = len(nc_list) * len(betas)
    done = 0

    def _flush():
        _write_csv(csv_path, rows, CSV_FIELDS)

    for nc in nc_list:
        for beta in betas:
            done += 1
            u_upper = TABLE1_U_UPPER[beta]
            print(f"\n[{label}  {done}/{n_total}]")
            row = {"table": label, "choice": "poisson", "nc": nc,
                   "n_display": _n_display(nc), "alpha2": beta,
                   "y_lower": -INF, "y_upper": INF, "u_lower": 0.0, "u_upper": u_upper,
                   "lumped_mass": int(lumped_mass), "discretization": discretization}
            rows.append(row)
            run_one(row, "poisson", nc, beta, -INF, INF, 0.0, u_upper, DEFAULT_EPS, lumped_mass,
                    discretization, tol, time_limit, max_iter, solvers, cooldown, flush_cb=_flush)
    print(f"  Saved: {csv_path}")
    return rows


def _run_poisson_state(nc_list, betas, lumped_mass, discretization, tol, time_limit, max_iter,
                       result_dir, solvers, cooldown, name_prefix):
    label = "poisson_state"
    csv_path = result_dir / f"{name_prefix}l2_{label}.csv"
    rows: list[dict] = _load_existing_rows(csv_path)
    n_total = len(nc_list) * len(betas)
    done = 0

    def _flush():
        _write_csv(csv_path, rows, CSV_FIELDS)

    for nc in nc_list:
        for beta in betas:
            done += 1
            y_upper = TABLE2_Y_UPPER[beta]
            print(f"\n[{label}  {done}/{n_total}]")
            row = {"table": label, "choice": "poisson_state", "nc": nc,
                   "n_display": _n_display(nc), "alpha2": beta,
                   "y_lower": -0.1, "y_upper": y_upper, "u_lower": -INF, "u_upper": INF,
                   "lumped_mass": int(lumped_mass), "discretization": discretization}
            rows.append(row)
            run_one(row, "poisson_state", nc, beta, -0.1, y_upper, -INF, INF, DEFAULT_EPS,
                    lumped_mass, discretization, tol, time_limit, max_iter, solvers, cooldown,
                    flush_cb=_flush)
    print(f"  Saved: {csv_path}")
    return rows


def _run_convdiff_both(nc_list, betas, lumped_mass, discretization, tol, time_limit, max_iter,
                       result_dir, solvers, cooldown, name_prefix):
    label = "convdiff_both"
    csv_path = result_dir / f"{name_prefix}l2_{label}.csv"
    rows: list[dict] = _load_existing_rows(csv_path)
    n_total = len(nc_list) * len(betas)
    done = 0

    def _flush():
        _write_csv(csv_path, rows, CSV_FIELDS)

    for nc in nc_list:
        for beta in betas:
            done += 1
            y_upper = TABLE3_Y_UPPER[beta]
            u_bound = TABLE3_U_BOUND[beta]
            print(f"\n[{label}  {done}/{n_total}]")
            row = {"table": label, "choice": "convdiff", "nc": nc,
                   "n_display": _n_display(nc), "alpha2": beta,
                   "y_lower": 0.0, "y_upper": y_upper,
                   "u_lower": -u_bound, "u_upper": u_bound,
                   "lumped_mass": int(lumped_mass), "discretization": discretization}
            rows.append(row)
            run_one(row, "convdiff", nc, beta, 0.0, y_upper, -u_bound, u_bound, DEFAULT_EPS,
                    lumped_mass, discretization, tol, time_limit, max_iter, solvers, cooldown,
                    flush_cb=_flush)
    print(f"  Saved: {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",       default=str(HERE.parent))
    parser.add_argument("--tol",        type=float, default=1e-6)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--nc",         type=int,   nargs="+", default=None,
                        help="Grid exponents (overrides each table's default nc list)")
    parser.add_argument("--table",      nargs="+",  default=ALL_TABLES,
                        choices=ALL_TABLES, metavar="TABLE",
                        help=f"Which tables to run (default: all). "
                             f"Choices: {' '.join(ALL_TABLES)}")
    parser.add_argument("--solver",     nargs="+",  default=["ksp-qp", "qpalm", "osqp"],
                        choices=["ksp-qp", "qpalm", "osqp"], metavar="SOLVER",
                        help="Solvers to run (default: all three). Choices: ksp-qp qpalm osqp")
    parser.add_argument("--cooldown",   type=float, default=0.0,
                        help="Seconds to sleep between solver runs to prevent CPU throttling (default: 0)")
    parser.add_argument("--lumped-mass", type=int, default=0, choices=[0, 1], metavar="{0,1}",
                        help="0 to use the consistent mass matrix (default), "
                             "1 to use the lumped mass matrix. Ignored (always lumped) "
                             "when --discretization fd is used.")
    parser.add_argument("--discretization", default="fem", choices=["fem", "fd"],
                        help="Spatial discretization for the PDE operator: 'fem' (default, "
                             "Q1 finite elements) or 'fd' (5-point finite-difference stencil "
                             "with first-order upwind convection).")
    parser.add_argument("--out",       default="",
                        help="Prefix for output filenames (e.g. '0727' -> '0727_l2_poisson_control.csv')")
    args = parser.parse_args()

    root       = Path(args.root).resolve()
    result_dir = root / "results"
    result_dir.mkdir(exist_ok=True)

    tol        = args.tol
    time_limit = args.time_limit
    cooldown   = args.cooldown
    max_iter   = 10_000_000_000
    tables     = set(args.table)
    solvers    = set(args.solver)
    lumped_mass = bool(args.lumped_mass)
    discretization = args.discretization
    name_prefix = f"{args.out}_" if args.out else ""

    if "poisson_control" in tables:
        nc_list = args.nc or TABLE1_NC
        _run_poisson_control(nc_list, TABLE1_BETAS, lumped_mass, discretization, tol, time_limit,
                             max_iter, result_dir, solvers, cooldown, name_prefix)

    if "poisson_state" in tables:
        nc_list = args.nc or TABLE2_NC
        _run_poisson_state(nc_list, TABLE2_BETAS, lumped_mass, discretization, tol, time_limit,
                           max_iter, result_dir, solvers, cooldown, name_prefix)

    if "convdiff_both" in tables:
        nc_list = args.nc or TABLE3_NC
        _run_convdiff_both(nc_list, TABLE3_BETAS, lumped_mass, discretization, tol, time_limit,
                           max_iter, result_dir, solvers, cooldown, name_prefix)


if __name__ == "__main__":
    main()
