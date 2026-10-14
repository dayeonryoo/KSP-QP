"""
Shared helpers for benchmark_*.py scripts (benchmark_mm.py, benchmark_mpc.py,
benchmark_netlib.py, benchmark_infeas.py, benchmark_l2pde.py):

  - QPALM / OSQP imports (with install hints) and status constants
  - KSPQPdata -> QPALM/OSQP conversion and solver wrappers
  - crash-tolerant subprocess isolation
  - the per-problem three-solver runner used by the PDE benchmark
  - CSV read/write helpers, including the per-solver result files of benchmark_mm.py and
    benchmark_mpc.py (one file per solver, rows replaced by key on a rerun) and their join
  - run provenance (run id, time, host, git revision, solver versions)
"""

import os
import sys
import csv
import time
import uuid
import socket
import platform
import subprocess
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import pandas as pd


def _import_or_exit(module_name: str, hint: str):
    try:
        return __import__(module_name)
    except ModuleNotFoundError:
        sys.exit(hint)


qpalm = _import_or_exit("qpalm", "Cannot find qpalm. Install it with: pip install qpalm")
osqp  = _import_or_exit("osqp",  "Cannot find osqp. Install it with: pip install osqp")

QPALM_SOLVED = qpalm.Info.SOLVED   # == 1
OSQP_SOLVED  = 1                   # osqp.constant("OSQP_SOLVED")

# Infeasibility certificate tolerance as a multiple of tol, mirroring KSP-QP's
# eps_pinf = eps_dinf = 1e-3 * tol (include/ksp_qp.hpp), so all three solvers test
# their Farkas certificates at the same ratio. QPALM/OSQP default to fixed constants
# (eps_prim_inf = 1e-5 / 1e-4), which are looser than the accuracy asked for once
# eps_abs/eps_rel are tightened; run_qpalm/run_osqp take eps_inf to keep the ratio.
INF_TOL_FACTOR = 1e-3


# ---------------------------------------------------------------------------
# Convert KSPQPdata dict (from ksp_qp_bind) to QPALM/OSQP inputs.
#
# KSPQPdata form:  min ½ xᵀQx + cᵀx   s.t.  Ax = b,  lw ≤ Bx ≤ uw,  lx ≤ x ≤ ux
# QPALM/OSQP form: min ½ xᵀQx + qᵀx   s.t.  bmin ≤ Cx ≤ bmax
#
# Stacking:  C = [A; B; Iₙ],  bmin/bmax accordingly. The identity block covers every
# variable, free ones included: QPALM/OSQP conventionally receive a full constraint
# matrix.
# ---------------------------------------------------------------------------

def _make_upper_triangular(Q):
    """Return upper-triangular part of a symmetric Q.

    Works regardless of whether Q is stored fully, as lower-triangular only,
    or as upper-triangular only. Uses COO deduplication to avoid double-counting.
    """
    Q_coo = Q.tocoo()
    entries: dict[tuple[int, int], float] = {}
    for r, c, v in zip(Q_coo.row.tolist(), Q_coo.col.tolist(), Q_coo.data.tolist()):
        key = (min(r, c), max(r, c))
        entries.setdefault(key, v)   # first occurrence wins
    if not entries:
        return sp.csc_matrix(Q.shape)
    rows, cols, vals = zip(*((k[0], k[1], v) for k, v in entries.items()))
    return sp.coo_matrix((vals, (rows, cols)), shape=Q.shape).tocsc()


def kspqp_to_qpalm(pd: dict):
    """Build (Q_upper, q, C, bmin, bmax, n, m_total) from a KSPQPdata dict."""
    n, ell = pd["n"], pd["l"]
    INF = 1e30   # QPALM/OSQP treat values beyond this as infinite

    Q = sp.csc_matrix((pd["Q_data"], pd["Q_indices"], pd["Q_indptr"]), shape=pd["Q_shape"])
    A = sp.csc_matrix((pd["A_data"], pd["A_indices"], pd["A_indptr"]), shape=pd["A_shape"])
    B = sp.csc_matrix((pd["B_data"], pd["B_indices"], pd["B_indptr"]), shape=pd["B_shape"])

    Q_upper = _make_upper_triangular(Q)
    q = np.asarray(pd["c"], dtype=np.float64)

    row_blocks  = [A]
    bmin_blocks = [np.asarray(pd["b"],  dtype=np.float64)]
    bmax_blocks = [np.asarray(pd["b"],  dtype=np.float64)]

    if ell > 0:
        row_blocks.append(B)
        bmin_blocks.append(np.asarray(pd["lw"], dtype=np.float64))
        bmax_blocks.append(np.asarray(pd["uw"], dtype=np.float64))

    row_blocks.append(sp.eye(n, format="csc"))
    bmin_blocks.append(np.asarray(pd["lx"], dtype=np.float64))
    bmax_blocks.append(np.asarray(pd["ux"], dtype=np.float64))

    C    = sp.vstack(row_blocks, format="csc")
    bmin = np.clip(np.concatenate(bmin_blocks), -INF, INF)
    bmax = np.clip(np.concatenate(bmax_blocks), -INF, INF)

    return Q_upper, q, C, bmin, bmax, n, int(C.shape[0])


# ---------------------------------------------------------------------------
# Solver wrappers
# ---------------------------------------------------------------------------

def run_qpalm(qpalm_data: tuple, tol: float, time_limit: float, obj_const: float = 0.0,
              eps_inf: float | None = None) -> dict:
    """Run QPALM on a problem already converted via kspqp_to_qpalm.

    eps_inf sets the primal/dual infeasibility *certificate* tolerance. Left as
    None, QPALM keeps its shipped default (eps_prim_inf = eps_dual_inf = 1e-5),
    which is decoupled from eps_abs/eps_rel: at the library's own defaults
    (eps_abs = 1e-4) the certificate is 10x tighter than the optimality test,
    but once eps_abs is overridden to a smaller tol the relationship inverts.
    Pass a value to restore a fixed certificate-to-optimality ratio.
    """
    Q_upper, q, C, bmin, bmax, n, m_total = qpalm_data

    data      = qpalm.Data(n, m_total)
    data.Q    = Q_upper
    data.q    = q
    data.A    = C
    data.bmin = bmin
    data.bmax = bmax

    settings              = qpalm.Settings()
    settings.eps_abs      = tol
    settings.eps_rel      = tol
    settings.max_iter     = 2_000_000_000   # effectively infinite
    settings.time_limit   = time_limit
    settings.verbose      = 0               # silent
    settings.scaling      = 10              # default Ruiz scaling passes
    if eps_inf is not None:
        settings.eps_prim_inf = eps_inf
        settings.eps_dual_inf = eps_inf

    solver = qpalm.Solver(data, settings)   # setup: Ruiz scaling + factorisation
    solver.solve()

    info = solver.info
    return {
        "status":       int(info.status_val),
        # QPALM's Data has no constant-term field; it only ever reports 0.5 x'Qx + q'x,
        # so obj_const is added back explicitly to compare against KSP-QP's obj_val.
        "obj_val":      float(info.objective) + obj_const,
        "setup_time":   float(info.setup_time),
        "solve_time":   float(info.solve_time),
        "run_time":     float(info.run_time),
        "outer_iter":   int(info.iter_out),
        "inner_iter":   int(info.iter),
        "tol_achieved": max(float(info.pri_res_norm), float(info.dua_res_norm)),
    }


def run_osqp(qpalm_data: tuple, tol: float, time_limit: float, obj_const: float = 0.0,
             eps_inf: float | None = None) -> dict:
    """Run OSQP on a problem already converted via kspqp_to_qpalm.

    eps_inf sets the primal/dual infeasibility *certificate* tolerance. Left as
    None, OSQP keeps its shipped default (eps_prim_inf = eps_dual_inf = 1e-4),
    decoupled from eps_abs/eps_rel in the same way QPALM's is -- see run_qpalm.
    """
    Q_upper, q, C, bmin, bmax, *_ = qpalm_data

    inf_kwargs = {} if eps_inf is None else {
        "eps_prim_inf": eps_inf, "eps_dual_inf": eps_inf,
    }

    prob = osqp.OSQP()
    prob.setup(                       # setup: scaling + factorisation
        Q_upper, q, C, bmin, bmax,
        eps_abs    = tol,
        eps_rel    = tol,
        max_iter   = 2_000_000_000,   # effectively infinite
        time_limit = time_limit,
        verbose    = False,
        scaling    = 10,
        **inf_kwargs,
    )
    res = prob.solve()

    info = res.info
    return {
        "status":       int(info.status_val),
        # OSQP's setup() has no constant-term argument so obj_const is added back explicitly.
        "obj_val":      float(info.obj_val) + obj_const,
        "setup_time":   float(info.setup_time),
        "solve_time":   float(info.solve_time),
        "run_time":     float(info.run_time),
        "outer_iter":   int(info.iter),
        "inner_iter":   0,
        "tol_achieved": max(float(info.prim_res), float(info.dual_res)),
    }


# ---------------------------------------------------------------------------
# Crash-tolerant subprocess isolation
# ---------------------------------------------------------------------------

def _run_isolated(target_func, args: tuple) -> dict:
    """Spawn a fresh process, run target_func(*args, conn), return the sent dict.

    If the worker dies without sending a result (segfault, OOM-kill by the
    OS, etc.), returns an {"error": ...} dict instead of raising.
    """
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(target=target_func, args=(*args, child_conn))
    p.start()
    child_conn.close()
    try:
        out = parent_conn.recv()
    except EOFError:
        p.join()
        return {"error": f"worker died without result (exitcode={p.exitcode}, "
                          f"likely OOM-killed or crashed)"}
    p.join()
    return out


# ---------------------------------------------------------------------------
# Per-problem three-solver runner (shared by the PDE-constrained benchmarks)
# ---------------------------------------------------------------------------

def run_solvers(result: dict, worker_args: tuple,
                worker_ssn, worker_qpalm, worker_osqp,
                tol: float, time_limit: float, max_iter: int, solvers: set,
                cooldown: float = 0.0, flush_cb=None) -> dict:
    """Solve one problem with all three solvers in isolated subprocesses.

    worker_args are the problem-defining positional args for worker_ssn/
    worker_qpalm/worker_osqp, passed ahead of (tol, time_limit[, max_iter], conn).
    """
    if "ksp-qp" in solvers:
        ssn_out = _run_isolated(worker_ssn, (*worker_args, tol, time_limit, max_iter))
        if "error" in ssn_out:
            print(f"    KSP-QP  ERROR — {ssn_out['error']}")
            result.update(ssn_status=-99, ssn_solved=0, ssn_time=float("inf"),
                          pmm_iter=-1, ssn_iter=-1,
                          krylov_iter=-1, fact=-1, smw_count=-1,
                          pmm_tol_achieved=float("nan"),
                          ssn_obj=float("nan"))
        else:
            r = ssn_out["res"]
            result.update(
                ssn_status=r["status"], ssn_solved=int(r["status"] == 0),
                ssn_time=r["run_time"],
                pmm_iter=r["pmm_iter"], ssn_iter=r["ssn_iter"],
                krylov_iter=r["krylov_iter"], fact=r["fact"], smw_count=r["smw_count"],
                pmm_tol_achieved=r["pmm_tol_achieved"],
                ssn_obj=r["obj_val"],
            )
            ok = "OK" if r["status"] == 0 else f"status={r['status']}"
            print(f"    KSP-QP  {ok:8s}  t={r['run_time']:.2f}s  "
                  f"{r['pmm_iter']}({r['ssn_iter']})[tol={r['pmm_tol_achieved']:.2e}]  "
                  f"krylov={r['krylov_iter']} fact={r['fact']} smw={r['smw_count']}")
        if flush_cb is not None:
            flush_cb()
        if cooldown > 0:
            time.sleep(cooldown)

    if "qpalm" in solvers:
        qpalm_out = _run_isolated(worker_qpalm, (*worker_args, tol, time_limit))
        if "error" in qpalm_out:
            print(f"    QPALM    ERROR — {qpalm_out['error']}")
            result.update(qpalm_status=-99, qpalm_solved=0, qpalm_time=float("inf"),
                          qpalm_iter=-1, qpalm_inner_iter=-1, qpalm_obj=float("nan"),
                          qpalm_tol_achieved=float("nan"))
        else:
            r = qpalm_out["res"]
            result.update(
                qpalm_status=r["status"], qpalm_solved=int(r["status"] == QPALM_SOLVED),
                qpalm_time=r["run_time"], qpalm_iter=r["outer_iter"],
                qpalm_inner_iter=r["inner_iter"], qpalm_obj=r["obj_val"],
                qpalm_tol_achieved=r["tol_achieved"],
            )
            ok = "OK" if r["status"] == QPALM_SOLVED else f"status={r['status']}"
            print(f"    QPALM    {ok:8s}  t={r['run_time']:.2f}s  "
                  f"{r['outer_iter']}({r['inner_iter']})[tol={r['tol_achieved']:.2e}]")
        if flush_cb is not None:
            flush_cb()
        if cooldown > 0:
            time.sleep(cooldown)

    if "osqp" in solvers:
        osqp_out = _run_isolated(worker_osqp, (*worker_args, tol, time_limit))
        if "error" in osqp_out:
            print(f"    OSQP     ERROR — {osqp_out['error']}")
            result.update(osqp_status=-99, osqp_solved=0, osqp_time=float("inf"),
                          osqp_iter=-1, osqp_obj=float("nan"),
                          osqp_tol_achieved=float("nan"))
        else:
            r = osqp_out["res"]
            result.update(
                osqp_status=r["status"], osqp_solved=int(r["status"] == OSQP_SOLVED),
                osqp_time=r["run_time"], osqp_iter=r["outer_iter"], osqp_obj=r["obj_val"],
                osqp_tol_achieved=r["tol_achieved"],
            )
            ok = "OK" if r["status"] == OSQP_SOLVED else f"status={r['status']}"
            print(f"    OSQP     {ok:8s}  t={r['run_time']:.2f}s  iter={r['outer_iter']}  "
                  f"tol={r['tol_achieved']:.2e}")
        if flush_cb is not None:
            flush_cb()
        if cooldown > 0:
            time.sleep(cooldown)

    return result


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Rewrite the whole CSV from `rows`. Called after every solver finishes.

    Written to a temporary file and moved into place, so a crash mid-write leaves the old file intact.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _load_existing_rows(path: Path) -> list[dict]:
    """Load previously written rows so a rerun appends instead of overwriting."""
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Per-solver result files (benchmark_mm.py, benchmark_mpc.py)
#
# Each solver writes its own CSV, so QPALM's results can be kept while KSP-QP is rerun.
# A rerun replaces the rows of the problems/instances it solved and keeps every other row;
# load_solver_tables() joins the files back on their key.
# ---------------------------------------------------------------------------

def provenance() -> dict:
    """Identifies a benchmark run: id, UTC start time, host, CPU, git revision (-dirty if uncommitted
    changes), solver versions."""
    def _sh(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""
    return {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "cpu": _sh(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
        "git_sha": _sh(["git", "-C", str(Path(__file__).parent), "describe", "--always", "--dirty"]),
        "qpalm_version": getattr(qpalm, "__version__", "?"),
        "osqp_version": getattr(osqp, "__version__", "?"),
    }


def check_header(path: Path, fieldnames: list[str]) -> None:
    """Exit if `path` exists with a header other than `fieldnames`; files of another schema are never mixed."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, newline="") as fh:
        head = next(csv.reader(fh), [])
    if head != fieldnames:
        sys.exit(f"{path} has a different header than this script writes. "
                 f"Move it aside or pass --out <prefix>.")


def replace_rows(path: Path, new_rows: list[dict], key: list[str], fieldnames: list[str]) -> None:
    """Write `new_rows` into the CSV at `path`, replacing every existing row whose `key` matches one of theirs.

    The replacement goes where the first replaced row was, so the file keeps its order; rows with a
    new key are appended. A key may cover several rows (e.g. every step and repeat of an MPC
    configuration), and all of them are replaced together. Keys are compared as strings, since rows
    read back from the file are strings.
    """
    path = Path(path)
    check_header(path, fieldnames)
    existing = _load_existing_rows(path)

    def k(row: dict) -> tuple:
        return tuple(str(row.get(f, "")) for f in key)

    groups: dict[tuple, list[dict]] = {}
    for r in new_rows:
        groups.setdefault(k(r), []).append(r)

    out, written = [], set()
    for r in existing:
        rk = k(r)
        if rk not in groups:
            out.append(r)
        elif rk not in written:
            out.extend(groups[rk])
            written.add(rk)
    for rk, rows in groups.items():
        if rk not in written:
            out.extend(rows)
    _write_csv(path, out, fieldnames)


def load_solver_tables(paths: dict, key: list[str], how: str = "inner") -> pd.DataFrame:
    """Join per-solver result files on `key` into one frame, in the order of `paths` ({solver: path}).

    how="inner" keeps the problems every file has; "outer" keeps all. Non-key columns present in
    more than one file get a "_<solver>" suffix.
    """
    frames = {s: pd.read_csv(p) for s, p in paths.items()}
    counts: dict[str, int] = {}
    for df in frames.values():
        for c in df.columns:
            counts[c] = counts.get(c, 0) + 1
    merged = None
    for s, df in frames.items():
        df = df.rename(columns={c: f"{c}_{s}" for c in df.columns if c not in key and counts[c] > 1})
        merged = df if merged is None else merged.merge(df, on=key, how=how)
    return merged

