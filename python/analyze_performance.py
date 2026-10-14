"""
Performance summary of KSP-QP against QPALM on the Maros-Meszaros QPs (MM) and the platoon MPC
problems, from the benchmarks' per-solver result files and KSP-QP's traces (experimental; branch
direct-vs-pcg, see include/solve_profile.hpp).

MM, over the problems in both result files:
  - robustness: the problems each solver solves, and KSP-QP's failures with where their time went,
    each classed as cost-bound or iteration-bound (see failure_verdicts());
  - run time and iteration counts against QPALM, with solve time = inner iterations x time per inner
    iteration, overall and by problem class: diagonal Q; lifted (non-diagonal Q, solved through the
    lifting of Appendix A.7); and, among the lifted ones, the Q* problems built from Netlib LPs;
  - the time per inner iteration against the problem size;
  - where KSP-QP's time goes: phases, factorizations by reason, SMW updates, CG iterations;
  - what-if bounds: the run-time ratio to QPALM if a cost were removed from KSP-QP's run;
  - the PMM and SSN behaviour, and the accuracy of the objective values.
MPC, over the (M, N) grid: robustness, timing noise over the repeats, run-time ratios, power-law fits
of the iterations, time per inner iteration and run time in M and N, phases, factorizations and
what-if bounds.

An MM problem's class comes from KSP-QP's trace: a lifted problem has m + n equality rows (s - n_act_W
of its events), so without the trace files MM is not split by class.

=== HOW TO RUN ===
  python3 analyze_performance.py --prefix ../results/1014_baseline/1014
  python3 analyze_performance.py --prefix ../results/1015 --qpalm-prefix ../results/1003 --sets mm \\
      --out ../results/1015_perf

--prefix P reads P_{mm,mpc}_kspqp.csv, P_{mm,mpc}_qpalm.csv and KSP-QP's traces
P_{mm,mpc}_kspqp_trace_{events,ssn,pmm}.csv (benchmark_*.py --trace); what needs a missing file is
skipped. --qpalm-prefix reads QPALM's files from another run (default: --prefix). --out DIR writes the
per-problem (mm_problems.csv) and per-configuration (mpc_configs.csv) tables.
"""

import re
import ast
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from analyze_profile import SSN_PHASES, _geomean, load_mm, load_mpc

# Disjoint phases of the solve time (with t_setup they add up to run_time; see solve_profile.hpp).
SOLVE_PHASES = [*SSN_PHASES, "t_pmm_other"]

KSPQP_STATUS = {-99: "error", -3: "DualInfeasible", -2: "PrimalInfeasible", -1: "NumericalError",
                0: "Optimal", 1: "MaxPmmIterations", 2: "MaxSsnIterations", 3: "TimeLimit",
                4: "Interrupted"}  # include/solution.hpp; -99 is a benchmark error row
QPALM_STATUS = {-99: "error", 0: "error", 1: "solved", 2: "dual terminated", -2: "max iterations",
                -3: "primal infeasible", -4: "dual infeasible", -5: "time limit", -6: "cancelled",
                -10: "unsolved"}  # QPALM's constants.h

# Maros-Meszaros QPs built from Netlib LPs: "Q" + the LP's name (the QPC* problems and QPTEST excluded).
NETLIB_QP = re.compile(r"Q(?!PC|PTEST)")
CLASSES = ["diagonal Q", "lifted, other", "lifted, Q*"]

SIZE_BINS = [0, 1e3, 1e4, 1e5, np.inf]
SIZE_LABELS = ["<1e3", "1e3-1e4", "1e4-1e5", ">=1e5"]

FAILURE_FACTOR = 10          # iteration-bound: at least this many times the reference SSN iterations
MU_MAX = RHO_MAX = 1e7       # Table 4

INSTANCE = ["M", "N", "loop_mode", "step"]
EVENT_COLS = ["name", "pmm_iter", "ssn_iter", "target", "action", "reason", "analyzed", "rank", "s",
              "t_assembly", "t_analyze", "t_factorize", "t_smw_setup"]
SSN_COLS = ["name", "pmm_iter", "ssn_iter", "n_act_W", "setup", "reason", "krylov"]
PMM_COLS = ["name", "pmm_iter", "ssn_iters", "ssn_status", "mu", "rho", "rule"]
TRACE_STR = ["name", "loop_mode", "target", "action", "reason", "setup", "rule"]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _title(text: str) -> None:
    print(f"\n{'=' * 100}\n{text}\n{'=' * 100}")


def _heading(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def _table(df: pd.DataFrame, **kw) -> None:
    print(df.to_string(float_format=lambda v: f"{v:.3g}", **kw))


def _percent(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.map("{:.0%}".format))


def _ratios(r: pd.Series) -> str:
    """Geometric mean of KSP-QP / QPALM ratios, their median, and how often KSP-QP is lower."""
    r = r[np.isfinite(r) & (r > 0)]
    return f"{_geomean(r):6.2f}   (median {r.median():.2f}; KSP-QP lower on {int((r < 1).sum())}/{len(r)})"


def _provenance(k: pd.DataFrame, q: pd.DataFrame) -> None:
    def values(df: pd.DataFrame, cols: list[str]) -> str:
        return ", ".join(f"{c}={'/'.join(map(str, df[c].dropna().unique()))}" for c in cols if c in df)
    print("KSP-QP:", values(k, ["ssn_git_sha", "ksp_mode", "ssn_tol", "ssn_time_limit", "ssn_run_id", "ssn_cpu"]))
    print("QPALM: ", values(q, ["qpalm_version", "qpalm_tol", "qpalm_time_limit", "qpalm_run_id", "qpalm_cpu"]))


def _read_trace(prefix: Path, kind: str, table: str, cols: list[str]) -> pd.DataFrame | None:
    path = Path(f"{prefix}_{kind}_kspqp_trace_{table}.csv")
    if not path.exists():
        print(f"note: {path} not found; what needs it is skipped", file=sys.stderr)
        return None
    return pd.read_csv(path, usecols=lambda c: c in cols, dtype={c: str for c in TRACE_STR if c in cols},
                       keep_default_na=False, na_values=["", "nan"], low_memory=False)


def _with_fact_time(events: pd.DataFrame) -> pd.DataFrame:
    """The factorization events, with their total time t (assembly, symbolic analysis, numeric factorization)."""
    f = events[events["action"] == "factorize"]
    return f.assign(t=f["t_assembly"] + f["t_analyze"] + f["t_factorize"])


# ---------------------------------------------------------------------------
# MM: one row per problem
# ---------------------------------------------------------------------------

def mm_reference_objectives() -> dict[str, float]:
    """The reference optimal objectives (QPS in benchmark_mm.py), read without importing it (it needs ksp_qp_bind)."""
    tree = ast.parse((HERE / "benchmark_mm.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "QPS" for t in node.targets):
            return ast.literal_eval(node.value)
    return {}


def lifted_from_trace(m: pd.DataFrame, events: pd.DataFrame, ssn: pd.DataFrame) -> pd.Series:
    """1 if KSP-QP solved the lifted problem (non-diagonal Q, Appendix A.7), 0 if not, NaN without a trace.
    G has the equality rows of the solved problem (m + n if lifted, m otherwise) and the active rows of B,
    so they are s - n_act_W at any SSN iteration with an event."""
    key = ["name", "pmm_iter", "ssn_iter"]
    j = events[[*key, "s"]].merge(ssn[[*key, "n_act_W"]], on=key)
    eq_rows = (j["s"] - j["n_act_W"]).groupby(j["name"]).min()
    d = m.set_index("name")
    eq = eq_rows.reindex(d.index)
    lifted = pd.Series(np.nan, index=d.index)
    lifted[eq == d["m"] + d["n"]] = 1.0
    lifted[eq == d["m"]] = 0.0
    return lifted


def _typical_ssn(m: pd.DataFrame) -> float:
    return float(m.loc[m["solved_k"] == 1, "inner_iter_k"].quantile(0.9))


def failure_verdicts(m: pd.DataFrame) -> pd.Series:
    """Why KSP-QP did not solve a problem: cost-bound if it ran out of time before doing the reference number of
    SSN iterations (QPALM's inner iterations, or, if QPALM failed too, the 90th percentile of KSP-QP's SSN
    iterations on the problems it solves); iteration-bound if it did FAILURE_FACTOR times as many or reached an
    iteration limit; both in between. Other statuses are given by name; solved problems get ''."""
    ref = m["inner_iter_q"].where(m["solved_q"] == 1, _typical_ssn(m))
    verdict = np.select([m["ssn_status"].isin([1, 2]), m["inner_iter_k"] < ref,
                         m["inner_iter_k"] >= FAILURE_FACTOR * ref],
                        ["iteration-bound (limit)", "cost-bound", "iteration-bound"], default="both")
    status = m["ssn_status"].map(lambda s: KSPQP_STATUS.get(int(s), str(s)))
    return pd.Series(np.where(m["solved_k"] == 1, "",
                              np.where(m["ssn_status"].isin([1, 2, 3]), verdict, status)), index=m.index)


def load_mm_table(prefix: Path, qpalm_prefix: Path, events: pd.DataFrame | None,
                  ssn: pd.DataFrame | None) -> pd.DataFrame:
    """Per problem in both MM result files: load_mm()'s columns (suffix _k: KSP-QP, _q: QPALM), the problem
    class, ratios, per-SSN-iteration counts, the time of the mu-triggered refactorizations and the objective
    errors."""
    qpath = Path(f"{qpalm_prefix}_mm_qpalm.csv")
    m, _ = load_mm(Path(f"{prefix}_mm_kspqp.csv"), qpath)
    m = m.merge(pd.read_csv(qpath, usecols=["name", "qpalm_status", "qpalm_obj"]), on="name", how="left")
    m["both"] = (m["solved_k"] == 1) & (m["solved_q"] == 1)

    ssn_iters = m["inner_iter_k"].where(m["inner_iter_k"] > 0)
    m["per_iter_k"] = m["solve_time_k"] / ssn_iters
    m["per_iter_q"] = m["solve_time_q"] / m["inner_iter_q"].where(m["inner_iter_q"] > 0)
    m["time_ratio"] = m["run_time_k"] / m["run_time_q"]
    m["solve_ratio"] = m["solve_time_k"] / m["solve_time_q"]
    m["inner_ratio"] = m["inner_iter_k"] / m["inner_iter_q"]
    m["per_iter_ratio"] = m["per_iter_k"] / m["per_iter_q"]
    for col, count in [("cg_per_ssn", "krylov_iter"), ("fact_per_ssn", "fact"), ("smw_per_ssn", "smw_count")]:
        m[col] = m[count] / ssn_iters
    size_cols = ["n", "nnz_Q", "nnz_A", "nnz_B"]
    m["size"] = m[size_cols].sum(axis=1) if set(size_cols) <= set(m.columns) else np.nan

    if events is not None and ssn is not None and {"n", "m"} <= set(m.columns):
        m["lifted"] = m["name"].map(lifted_from_trace(m, events, ssn))
    else:
        m["lifted"] = np.nan
    q_star = m["name"].map(lambda s: bool(NETLIB_QP.match(s)))
    m["class"] = np.select([m["lifted"] == 0, (m["lifted"] == 1) & ~q_star, (m["lifted"] == 1) & q_star],
                           CLASSES, default="unknown")

    if events is not None:
        f = _with_fact_time(events)
        m["t_fact_mu"] = m["name"].map(f[f["reason"] == "mu_rho"].groupby("name")["t"].sum()).fillna(0.0)

    f_ref = m["name"].map(mm_reference_objectives())
    scale = np.maximum(1.0, f_ref.abs())
    m["obj_err_k"] = (m["ssn_obj"] - f_ref).abs() / scale
    m["obj_err_q"] = (m["qpalm_obj"] - f_ref).abs() / scale
    m["failure"] = failure_verdicts(m)
    return m


# ---------------------------------------------------------------------------
# MM report
# ---------------------------------------------------------------------------

def mm_robustness(m: pd.DataFrame) -> None:
    _heading("Robustness")
    k, q = m["solved_k"] == 1, m["solved_q"] == 1
    print(f"{len(m)} problems: solved by KSP-QP {int(k.sum())}, by QPALM {int(q.sum())}, "
          f"by both {int((k & q).sum())}, by neither {int((~k & ~q).sum())}")
    print(f"  QPALM only:  {', '.join(m.loc[q & ~k, 'name']) or '-'}")
    print(f"  KSP-QP only: {', '.join(m.loc[k & ~q, 'name']) or '-'}")
    if m["class"].ne("unknown").any():
        by = m.groupby("class").agg(problems=("name", "size"), solved_by_KSP_QP=("solved_k", "sum"),
                                    solved_by_QPALM=("solved_q", "sum"))
        _table(by.reindex([c for c in [*CLASSES, "unknown"] if c in by.index]))
        print("(diagonal Q: solved as given; lifted: non-diagonal Q, solved through the lifting of Appendix A.7; "
              "Q*: built from a Netlib LP)")

    f = m[~k]
    if f.empty:
        return
    share = f[SOLVE_PHASES].div(f["run_time_k"], axis=0)
    top = share.fillna(-1.0).idxmax(axis=1).str.removeprefix("t_")
    qpalm = np.where(f["solved_q"] == 1,
                     f["inner_iter_q"].map("{:.0f} its".format) + ", " + f["run_time_q"].map("{:.2g} s".format),
                     f["qpalm_status"].map(lambda s: QPALM_STATUS.get(int(s), str(s))))
    tab = pd.DataFrame({
        "class": f["class"], "time_s": f["run_time_k"], "PMM": f["outer_iter_k"], "SSN": f["inner_iter_k"],
        "ms_per_SSN": 1e3 * f["solve_time_k"] / f["inner_iter_k"], "CG_per_SSN": f["cg_per_ssn"],
        "top_phase": top + " " + share.max(axis=1).map("{:.0%}".format),
        "analysis": (f["t_analyze"] / f["run_time_k"]).map("{:.0%}".format),
        "residual": f["pmm_tol_achieved"], "QPALM": qpalm, "verdict": f["failure"]})
    tab.index = f["name"]
    statuses = sorted(set(f["ssn_status"].map(lambda s: KSPQP_STATUS.get(int(s), str(s)))))
    print(f"\nKSP-QP's failures (statuses: {', '.join(statuses)}):")
    _table(tab.sort_values(["verdict", "time_s"]))
    print(f"verdict: cost-bound = fewer SSN iterations than QPALM's inner iterations (if QPALM failed too: than "
          f"KSP-QP's 90th percentile on solved problems, {_typical_ssn(m):.0f}); iteration-bound = "
          f">= {FAILURE_FACTOR}x as many, or an iteration limit; both = in between")


def mm_comparison(m: pd.DataFrame) -> None:
    b = m[m["both"]]
    _heading(f"Run time and iterations against QPALM ({len(b)} problems both solve)")
    print("KSP-QP / QPALM, geometric mean:")
    for label, r in [("run time", b["time_ratio"]), ("setup time", b["setup_time_k"] / b["setup_time_q"]),
                     ("solve time", b["solve_ratio"]), ("outer iterations", b["outer_iter_k"] / b["outer_iter_q"]),
                     ("inner iterations", b["inner_ratio"]), ("time per inner iteration", b["per_iter_ratio"])]:
        print(f"  {label:26s} {_ratios(r)}")
    t = b["time_ratio"]
    print(f"  run time within 2x of QPALM's on {int((t <= 2).sum())}, within 10x on {int((t <= 10).sum())} of {len(b)}")
    print(f"  solve time = inner iterations x time per inner iteration: {_geomean(b['solve_ratio']):.2f} = "
          f"{_geomean(b['inner_ratio']):.2f} x {_geomean(b['per_iter_ratio']):.2f}")


def mm_by_class(m: pd.DataFrame) -> None:
    b = m[m["both"]]
    if b["class"].eq("unknown").all():
        print("\nnote: no trace, so no split into diagonal-Q and lifted problems")
        return
    _heading("By problem class (problems both solve; ratios KSP-QP / QPALM, geometric means)")
    groups = [(c, b[b["class"] == c]) for c in [*CLASSES, "unknown"] if (b["class"] == c).any()]
    rows = {cls: {"problems": len(g), "run_time": _geomean(g["time_ratio"]), "inner_its": _geomean(g["inner_ratio"]),
                  "per_inner_it": _geomean(g["per_iter_ratio"]), "CG_per_SSN": g["cg_per_ssn"].median(),
                  "fact_per_SSN": g["fact_per_ssn"].median(), "SMW_per_SSN": g["smw_per_ssn"].median(),
                  "KSP_QP_faster": int((g["time_ratio"] < 1).sum())}
            for cls, g in [*groups, ("all", b)]}
    _table(pd.DataFrame(rows).T)
    print("(CG, fact and SMW per SSN iteration: medians over the problems)")


def mm_size(m: pd.DataFrame) -> None:
    b = m[m["both"] & m["size"].notna()]
    if b.empty:
        print("\nnote: no size columns (n, nnz_Q, nnz_A, nnz_B) in the result file; size analysis skipped")
        return
    _heading("Time per inner iteration against size = nnz(Q) + nnz(A) + nnz(B) + n (problems both solve)")
    b = b.assign(bin=pd.cut(b["size"], SIZE_BINS, labels=SIZE_LABELS, right=False))
    classes = CLASSES if b["class"].ne("unknown").any() else []
    rows = {}
    for label, g in b.groupby("bin", observed=True):
        row = {"problems": len(g), "KSP_QP_us": 1e6 * g["per_iter_k"].median(),
               "QPALM_us": 1e6 * g["per_iter_q"].median(), "ratio": _geomean(g["per_iter_ratio"])}
        for cls in classes:
            gc = g[g["class"] == cls]
            row[f"ratio {cls}"] = _geomean(gc["per_iter_ratio"]) if len(gc) else np.nan
        rows[label] = row
    _table(pd.DataFrame(rows).T)
    print("(us: median time per inner iteration; ratio: geometric mean of KSP-QP / QPALM)")
    big = b[b["size"] >= 1e4]
    if len(big) >= 5:
        x = np.log10(big["size"])
        slopes = [np.polyfit(x, np.log10(big[c]), 1)[0] for c in ("per_iter_k", "per_iter_q")]
        print(f"log-log slope of the time per inner iteration over the {len(big)} problems with size >= 1e4: "
              f"KSP-QP {slopes[0]:.2f}, QPALM {slopes[1]:.2f}")


def phase_shares(df: pd.DataFrame) -> pd.Series:
    """Mean over the rows of each phase's share of the solve time."""
    return df[SOLVE_PHASES].div(df[SOLVE_PHASES].sum(axis=1), axis=0).mean()


def mm_phases(m: pd.DataFrame) -> None:
    b = m[m["both"]]
    _heading("Where KSP-QP's solve time goes (problems both solve; mean of per-problem shares)")
    classes = CLASSES if b["class"].ne("unknown").any() else []
    cols = {cls: phase_shares(b[b["class"] == cls]) for cls in classes if (b["class"] == cls).any()}
    cols["all"] = phase_shares(b)
    cols["all, time-weighted"] = b[SOLVE_PHASES].sum() / b[SOLVE_PHASES].sum().sum()
    tab = pd.DataFrame(cols).rename(index=lambda c: c.removeprefix("t_"))
    print(_percent(tab).to_string())
    fact = b["t_fact"].where(b["t_fact"] > 0)
    print(f"symbolic analysis: {b['t_analyze'].sum() / b['t_fact'].sum():.0%} of the factorization time in total, "
          f"{(b['t_analyze'] / fact).median():.0%} on the median problem")


def factorization_report(events: pd.DataFrame, ssn: pd.DataFrame | None, classes: pd.Series | None) -> None:
    """Factorizations by reason, SMW updates, and PCG after SMW updates vs refactorizations, from KSP-QP's events
    and SSN rows (with a 'name' column per problem or configuration)."""
    _heading("Factorizations and SMW updates (trace)")
    f = _with_fact_time(events)
    if f.empty:
        print("no factorizations")
        return
    by = f.groupby(["target", "reason"]).agg(count=("t", "size"), time_s=("t", "sum"),
                                            problems=("name", "nunique"), with_analysis=("analyzed", "mean"))
    by.insert(1, "count_share", by["count"] / by["count"].sum())
    by.insert(3, "time_share", by["time_s"] / by["time_s"].sum())
    _table(by.sort_values("time_s", ascending=False))

    u = events[events["action"] == "smw_update"]
    if len(u):
        f_med = f[f["target"] == "prec"].groupby("name")["t"].median()
        rel = u["t_smw_setup"] / u["name"].map(f_med)
        rank_bin = pd.cut(u["rank"], [-1, 5, 10, 20, 35, 50, np.inf], labels=["0-5", "6-10", "11-20", "21-35", "36-50", ">50"])
        by_rank = rel.groupby(rank_bin, observed=True).median()
        print(f"SMW updates: {len(u)}, rank median {u['rank'].median():.0f}, 90th pct {u['rank'].quantile(.9):.0f}; "
              "setup time / the problem's median factorization time, by rank: "
              + ", ".join(f"{r} {v:.2f}" for r, v in by_rank.items()))
    rk = f[f["reason"] == "rank"]
    if len(rk):
        print(f"rank-limit refactorizations: requested rank median {rk['rank'].median():.0f}, quartiles "
              f"{rk['rank'].quantile(.25):.0f}-{rk['rank'].quantile(.75):.0f}; {(rk['rank'] <= 100).mean():.0%} "
              f"<= 100, {(rk['rank'] <= 200).mean():.0%} <= 200")
    sg = f[f["reason"] == "singular"]
    if len(sg):
        lifted = (f" ({sg['name'].map(classes).str.startswith('lifted').mean():.0%} in lifted problems)"
                  if classes is not None else "")
        print(f"singular-capacitance refactorizations: {len(sg)} in {sg['name'].nunique()} problems{lifted}, "
              f"at rank median {sg['rank'].median():.0f}")

    if ssn is not None:
        kind = np.select([ssn["setup"] == "smw_update",
                          (ssn["setup"] == "factorize") & ssn["reason"].isin(["rank", "singular", "pcg_fail"])],
                         ["smw", "fact"], default="")
        cg = ssn.assign(kind=kind)[kind != ""].groupby(["name", "kind"])["krylov"].mean().unstack()
        if {"smw", "fact"} <= set(cg.columns):
            pair = cg.dropna(subset=["smw", "fact"])
            print(f"CG iterations per SSN iteration, within the {len(pair)} problems/configurations with both: median "
                  f"{pair['smw'].median():.2f} after an SMW update vs {pair['fact'].median():.2f} after an active-set "
                  f"refactorization (rank, singular, pcg_fail); higher after SMW on {int((pair['smw'] > pair['fact']).sum())}")


def what_if_report(b: pd.DataFrame, groups: pd.Series | None) -> None:
    """Run-time ratio to QPALM if part of KSP-QP's run changed and the rest were kept: bounds on what removing a
    cost could gain, and the run time with QPALM's cost per inner iteration or with QPALM's iteration count."""
    _heading("What-if bounds: run time ratio to QPALM if part of KSP-QP's run changed (the rest kept)")
    base = b["run_time_k"]
    per_cg = b["t_krylov"] / b["krylov_iter"].where(b["krylov_iter"] > 0)
    one_cg = np.minimum(b["t_krylov"], (per_cg * b["inner_iter_k"]).fillna(b["t_krylov"]))
    scenarios = {"as is": base, "no symbolic analysis": base - b["t_analyze"]}
    if "t_fact_mu" in b:
        scenarios["no mu-triggered refactorizations"] = base - b["t_fact_mu"]
    scenarios.update({
        "factorizations free": base - b["t_fact"],
        "factorizations and SMW setups free": base - b["t_fact"] - b["t_lowrank"],
        "at most one CG iteration per SSN iteration": base - b["t_krylov"] + one_cg,
        "KSP-QP's iterations at QPALM's cost each": b["setup_time_k"] + b["inner_iter_k"] * b["per_iter_q"],
        "QPALM's iterations at KSP-QP's cost each": b["setup_time_k"] + b["inner_iter_q"] * b["per_iter_k"],
    })
    names = [c for c in [*CLASSES, "unknown"] if groups is not None and (groups == c).any()]
    rows = {}
    for label, t in scenarios.items():
        r = t / b["run_time_q"]
        row = {"all": f"{_geomean(r):.2f} ({int((r < 1).sum())}/{len(r)})"}
        for g in names:
            rg = r[groups == g]
            row[g] = f"{_geomean(rg):.2f} ({int((rg < 1).sum())}/{len(rg)})"
        rows[label] = row
    print(pd.DataFrame(rows).T.to_string())
    print("(geometric mean of KSP-QP / QPALM, and on how many KSP-QP would be faster)")


def pmm_report(pmm: pd.DataFrame, names: set[str]) -> None:
    _heading("PMM and SSN behaviour (problems both solve; per-problem statistics, PMM trace)")
    p = pmm[pmm["name"].isin(names)]
    by = p.groupby("name")
    per = pd.DataFrame({
        "PMM iterations": by.size(),
        "SSN iterations per PMM iteration": by["ssn_iters"].mean(),
        "share of SSN solves stopped at j_max": (p["ssn_status"] == 1).groupby(p["name"]).mean(),
        "share of SSN solves stagnated": (p["ssn_status"] == 3).groupby(p["name"]).mean(),
        "share of PMM iterations at mu_max": (p["mu"] >= 0.999 * MU_MAX).groupby(p["name"]).mean(),
        "share of PMM iterations with rho < rho_max": (p["rho"] < 0.999 * RHO_MAX).groupby(p["name"]).mean(),
    })
    _table(pd.DataFrame({"median": per.median(), "90th pct": per.quantile(0.9), "problems > 0": (per > 0).sum()}))
    rules = pd.crosstab(p["name"], p["rule"], normalize="index").mean().sort_values(ascending=False)
    print("PMM parameter update, mean share per problem: " + ", ".join(f"{r} {s:.0%}" for r, s in rules.items()))


def mm_accuracy(m: pd.DataFrame) -> None:
    if m["obj_err_k"].isna().all():
        return
    _heading("Accuracy of the objective: |f - f_ref| / max(1, |f_ref|), f_ref from benchmark_mm.py")
    rows = {}
    for label, sel, col in [("KSP-QP, solved by it", m["solved_k"] == 1, "obj_err_k"),
                            ("QPALM, solved by it", m["solved_q"] == 1, "obj_err_q"),
                            ("KSP-QP, solved by both", m["both"], "obj_err_k"),
                            ("QPALM, solved by both", m["both"], "obj_err_q")]:
        e = m.loc[sel, col].dropna()
        rows[label] = {"problems": len(e), "median": e.median(), "90th pct": e.quantile(0.9),
                       "> 1e-4": int((e > 1e-4).sum()), "max": e.max(),
                       "worst": m.loc[e.idxmax(), "name"] if len(e) else ""}
    _table(pd.DataFrame(rows).T)


def mm_report(prefix: Path, qpalm_prefix: Path) -> pd.DataFrame | None:
    kpath, qpath = Path(f"{prefix}_mm_kspqp.csv"), Path(f"{qpalm_prefix}_mm_qpalm.csv")
    if not (kpath.exists() and qpath.exists()):
        print(f"note: {kpath} or {qpath} not found; MM skipped", file=sys.stderr)
        return None
    events = _read_trace(prefix, "mm", "events", EVENT_COLS)
    ssn = _read_trace(prefix, "mm", "ssn", SSN_COLS)
    pmm = _read_trace(prefix, "mm", "pmm", PMM_COLS)
    m = load_mm_table(prefix, qpalm_prefix, events, ssn)

    _title("Maros-Meszaros QPs")
    _provenance(pd.read_csv(kpath), pd.read_csv(qpath))
    mm_robustness(m)
    mm_comparison(m)
    mm_by_class(m)
    mm_size(m)
    mm_phases(m)
    both = set(m.loc[m["both"], "name"])
    if events is not None:
        classes = m.set_index("name")["class"]
        factorization_report(events[events["name"].isin(both)],
                             None if ssn is None else ssn[ssn["name"].isin(both)], classes)
    b = m[m["both"]]
    what_if_report(b, None if b["class"].eq("unknown").all() else b["class"])
    if pmm is not None:
        pmm_report(pmm, both)
    mm_accuracy(m)
    return m


# ---------------------------------------------------------------------------
# MPC report: one row per (M, N, loop mode), summed over the steps both solve
# ---------------------------------------------------------------------------

def _power_law(g: pd.DataFrame, col: str) -> tuple[float, float, float]:
    """Exponents of M and N, and R^2, of a least-squares fit log y = a + b log M + c log N."""
    X = np.column_stack([np.ones(len(g)), np.log(g["M"]), np.log(g["N"])])
    y = np.log(g[col].to_numpy(dtype=float))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1.0 - np.sum((y - X @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
    return coef[1], coef[2], r2


def mpc_timing(raw_k: pd.DataFrame, raw_q: pd.DataFrame) -> None:
    for label, raw, t, its in [("KSP-QP", raw_k, "ssn_time", ["pmm_iter", "ssn_iter", "krylov_iter", "fact"]),
                               ("QPALM", raw_q, "qpalm_time", ["qpalm_iter", "qpalm_inner_iter"])]:
        timed = raw[raw["rep"] > 0].groupby(INSTANCE)[t]
        if timed.ngroups == 0:
            print(f"  {label}: no timed repeats (rep > 0)")
            continue
        spread = timed.std() / timed.mean()
        warmup = raw[raw["rep"] == 0].set_index(INSTANCE)[t] / timed.min()
        same = bool((raw.groupby(INSTANCE)[its].nunique() == 1).all().all())
        print(f"  {label:7s} spread over the timed repeats (std / mean): median {spread.median():.1%}, "
              f"90th pct {spread.quantile(.9):.1%}; repeat 0 / fastest: median {warmup.median():.2f}; "
              f"iteration counts identical over repeats: {'yes' if same else 'no'}")


def mpc_report(prefix: Path, qpalm_prefix: Path) -> pd.DataFrame | None:
    kpath, qpath = Path(f"{prefix}_mpc_kspqp.csv"), Path(f"{qpalm_prefix}_mpc_qpalm.csv")
    if not (kpath.exists() and qpath.exists()):
        print(f"note: {kpath} or {qpath} not found; MPC skipped", file=sys.stderr)
        return None
    raw_k, raw_q = pd.read_csv(kpath), pd.read_csv(qpath)
    cfg, used = load_mpc(kpath, qpath)
    config = ["M", "N", "loop_mode"]
    cfg = cfg.merge(raw_k.groupby(config, as_index=False)[["n_z", "m_eq", "nnz_Q", "nnz_A"]].first(), on=config)
    steps = cfg["steps"]
    cfg["inner_k"], cfg["inner_q"] = cfg["inner_iter_k"] / steps, cfg["inner_iter_q"] / steps
    cfg["time_k"], cfg["time_q"] = cfg["run_time_k"] / steps, cfg["run_time_q"] / steps
    cfg["per_iter_k"] = cfg["solve_time_k"] / cfg["inner_iter_k"]
    cfg["per_iter_q"] = cfg["solve_time_q"] / cfg["inner_iter_q"]
    cfg["time_ratio"] = cfg["run_time_k"] / cfg["run_time_q"]
    cfg["inner_ratio"] = cfg["inner_iter_k"] / cfg["inner_iter_q"]
    cfg["per_iter_ratio"] = cfg["per_iter_k"] / cfg["per_iter_q"]
    cfg["cg_per_ssn"] = cfg["krylov_iter"] / cfg["inner_iter_k"]
    cfg["fact_per_ssn"] = cfg["fact"] / cfg["inner_iter_k"]

    # KSP-QP's trace, restricted to the repeats the configuration sums use.
    keys = [*INSTANCE, "rep"]
    used_keys = pd.MultiIndex.from_frame(used[keys])

    def used_rows(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None:
            return None
        df = df[pd.MultiIndex.from_frame(df[keys]).isin(used_keys)].copy()
        df["name"] = df["M"].astype(str) + "x" + df["N"].astype(str) + "/" + df["loop_mode"]
        return df

    events = used_rows(_read_trace(prefix, "mpc", "events", [*EVENT_COLS, *keys]))
    ssn = used_rows(_read_trace(prefix, "mpc", "ssn", [*SSN_COLS, *keys]))
    if events is not None:
        f = _with_fact_time(events)
        cfg["t_fact_mu"] = cfg["name"].map(f[f["reason"] == "mu_rho"].groupby("name")["t"].sum()).fillna(0.0)

    _title("Platoon MPC")
    _provenance(raw_k, raw_q)
    _heading("Robustness and timing")
    k0, q0 = raw_k[raw_k["rep"] == 0], raw_q[raw_q["rep"] == 0]
    print(f"{len(raw_k[config].drop_duplicates())} configurations, {len(k0)} instances: solved by KSP-QP "
          f"{int(k0['ssn_solved'].sum())}, by QPALM {int(q0['qpalm_solved'].sum())}")
    mpc_timing(raw_k, raw_q)

    for mode, g in cfg.groupby("loop_mode"):
        _heading(f"Run time and iterations against QPALM, loop mode '{mode}' ({len(g)} configurations; "
                 f"sums over the steps both solve, fastest timed repeat)")
        print("KSP-QP / QPALM, geometric mean over the configurations:")
        for label, r in [("run time", g["time_ratio"]), ("setup time", g["setup_time_k"] / g["setup_time_q"]),
                         ("outer iterations", g["outer_iter_k"] / g["outer_iter_q"]),
                         ("inner iterations", g["inner_ratio"]), ("time per inner iteration", g["per_iter_ratio"])]:
            print(f"  {label:26s} {_ratios(r)}")
        print("Run time ratio KSP-QP / QPALM by vehicle count M (rows) and horizon N (columns):")
        _table(g.pivot(index="M", columns="N", values="time_ratio"))
        if g["M"].nunique() > 1 and g["N"].nunique() > 1:
            print("Power-law fits over the grid, y ~ M^a N^b (per instance):")
            rows = {}
            for label, k_col, q_col in [("inner iterations", "inner_k", "inner_q"),
                                        ("time per inner iteration", "per_iter_k", "per_iter_q"),
                                        ("run time", "time_k", "time_q")]:
                (ak, bk, rk), (aq, bq, rq) = _power_law(g, k_col), _power_law(g, q_col)
                rows[label] = {"KSP-QP a": ak, "KSP-QP b": bk, "KSP-QP R2": rk, "QPALM a": aq, "QPALM b": bq, "QPALM R2": rq}
            _table(pd.DataFrame(rows).T)

        _heading("Where KSP-QP's solve time goes, by M (share of the solve time summed over N)")
        ph = g.groupby("M")[[*SOLVE_PHASES, "t_analyze", "krylov_iter", "fact", "inner_iter_k"]].sum()
        tab = _percent(ph[SOLVE_PHASES].div(ph[SOLVE_PHASES].sum(axis=1), axis=0)
                       .rename(columns=lambda c: c.removeprefix("t_")))
        tab["analysis/fact"] = (ph["t_analyze"] / ph["t_fact"]).map("{:.0%}".format)
        tab["CG/SSN"] = (ph["krylov_iter"] / ph["inner_iter_k"]).map("{:.1f}".format)
        tab["fact/SSN"] = (ph["fact"] / ph["inner_iter_k"]).map("{:.2f}".format)
        print(tab.to_string())
        if events is not None:
            factorization_report(events[events["loop_mode"] == mode],
                                 None if ssn is None else ssn[ssn["loop_mode"] == mode], None)
        what_if_report(g, None)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", required=True, type=Path,
                        help="Result file prefix, e.g. ../results/1014 for ../results/1014_mm_kspqp.csv")
    parser.add_argument("--qpalm-prefix", type=Path, default=None,
                        help="Prefix of QPALM's result files (default: --prefix)")
    parser.add_argument("--sets", nargs="+", choices=["mm", "mpc"], default=["mm", "mpc"])
    parser.add_argument("--out", type=Path, default=None,
                        help="Directory for the per-problem (MM) and per-configuration (MPC) tables")
    args = parser.parse_args()
    qpalm_prefix = args.qpalm_prefix or args.prefix
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    tables = {}
    if "mm" in args.sets:
        tables["mm_problems.csv"] = mm_report(args.prefix, qpalm_prefix)
    if "mpc" in args.sets:
        tables["mpc_configs.csv"] = mpc_report(args.prefix, qpalm_prefix)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        for file, df in tables.items():
            if df is not None:
                df.to_csv(args.out / file, index=False)
                print(f"\n{file} written to: {args.out}")


if __name__ == "__main__":
    main()
