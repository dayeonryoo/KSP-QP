"""
Per-iteration gap between KSP-QP and QPALM, from the benchmarks' per-solver result files
(experimental; branch direct-vs-pcg, see include/solve_profile.hpp).

For each problem (MM) or (M, N) configuration (MPC) solved by both solvers, it reports:
  - the run time, setup and solve time, and iteration counts of each solver;
  - each solver's solve time per inner iteration (KSP-QP: per SSN iteration, QPALM: per inner
    iteration) and their ratio -- the per-iteration gap;
  - KSP-QP's time per SSN iteration split by phase (t_prep, t_fact, t_lowrank, t_krylov, ...),
    and its factorizations and SMW updates per SSN iteration;
then, over all of them, geometric means of the ratios, KSP-QP's time by phase, and its
factorizations by reason (and, given its trace, the factorization time by reason).

=== HOW TO RUN ===
  python3 analyze_profile.py mm --kspqp ../results/1015_mm_kspqp.csv --qpalm ../results/1015_mm_qpalm.csv
  python3 analyze_profile.py mpc --kspqp ../results/1015_mpc_kspqp.csv --qpalm ../results/1003_mpc_qpalm.csv \\
      --trace-prefix ../results/1015_mpc_kspqp_trace --out ../results/1015_mpc_gap.csv

--kspqp also takes a C++ driver CSV (ksp_qp_maros_meszaros/ksp_qp_netlib --out FILE) for mm.
MPC rows are timed repeats (rep > 0; rep 0 is the warm-up), the fastest per instance, summed over
the rollout steps of each configuration.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Disjoint phases of KSP-QP's solve time (with t_setup they add up to run_time); see solve_profile.hpp.
SSN_PHASES = ["t_prep", "t_fact", "t_lowrank", "t_krylov", "t_direct_solve", "t_linesearch",
              "t_update", "t_ssn_other"]
PHASES = ["t_setup", *SSN_PHASES, "t_pmm_other"]


def _fact_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("fact_") and c != "fact_analyzed"]


def _geomean(x: pd.Series) -> float:
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


# ---------------------------------------------------------------------------
# Loading: one frame per solver with common names
#   key columns, solved, run_time, setup_time, solve_time, outer_iter, inner_iter
#   (+ KSP-QP: krylov_iter, fact, smw_count, the fact_* and t_* profile columns)
# ---------------------------------------------------------------------------

def load_mm(kspqp: Path, qpalm: Path) -> tuple[pd.DataFrame, None]:
    k = pd.read_csv(kspqp)
    if "ssn_status" in k.columns:   # benchmark_mm.py's mm_kspqp.csv
        k = k.rename(columns={"ssn_solved": "solved", "ssn_time": "run_time", "ssn_setup_time": "setup_time",
                              "ssn_solve_time": "solve_time", "pmm_iter": "outer_iter", "ssn_iter": "inner_iter"})
    elif "opt_status" in k.columns:  # a C++ driver CSV
        k = k.assign(solved=(k["opt_status"] == 0).astype(int))
        k = k.rename(columns={"pmm_iter": "outer_iter", "ssn_iter": "inner_iter"})
    else:
        sys.exit(f"{kspqp}: not a KSP-QP result file")
    if "t_fact" not in k.columns:
        sys.exit(f"{kspqp}: no profile columns (t_*, fact_*); written before the factorization log was added?")
    q = pd.read_csv(qpalm).rename(columns={
        "qpalm_solved": "solved", "qpalm_time": "run_time", "qpalm_setup_time": "setup_time",
        "qpalm_solve_time": "solve_time", "qpalm_iter": "outer_iter", "qpalm_inner_iter": "inner_iter"})
    keep_q = ["name", "solved", "run_time", "setup_time", "solve_time", "outer_iter", "inner_iter"]
    return k.merge(q[keep_q], on="name", suffixes=("_k", "_q")), None


def _fastest_timed_repeat(df: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    """Per instance, the repeat with the smallest run time among rep > 0 (rep 0 if that is all there is)."""
    timed = df[df["rep"] > 0]
    if timed.empty:
        print("note: no timed repeats (rep > 0); using rep 0", file=sys.stderr)
        timed = df
    return timed.loc[timed.groupby(key)["run_time"].idxmin()]


def load_mpc(kspqp: Path, qpalm: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (M, N, loop mode): sums over the steps both solved; also the KSP-QP repeats used (for the trace)."""
    instance = ["M", "N", "loop_mode", "step"]
    k = _fastest_timed_repeat(pd.read_csv(kspqp), instance)
    q = _fastest_timed_repeat(pd.read_csv(qpalm), instance)
    if "t_fact" not in k.columns:
        sys.exit(f"{kspqp}: no profile columns (t_*, fact_*); written before the factorization log was added?")
    keep_q = [*instance, "xbar_norm", "solved", "run_time", "setup_time", "solve_time", "outer_iter", "inner_iter"]
    m = k.merge(q[keep_q], on=instance, suffixes=("_k", "_q"))
    off = ~np.isclose(m["xbar_norm_k"], m["xbar_norm_q"], rtol=1e-12, atol=0.0)
    if off.any():
        print(f"warning: {int(off.sum())} instances have different initial states (xbar_norm) in the two "
              f"files -- not the same trajectory; they are dropped", file=sys.stderr)
        m = m[~off]

    # Sum each configuration over its rollout steps (only steps both solved).
    m = m[(m["solved_k"] == 1) & (m["solved_q"] == 1)]
    used = m[[*instance, "rep"]]
    config = ["M", "N", "loop_mode"]
    summed = [c for c in m.columns if c.startswith(("t_", "fact")) or c in (
        "smw_count", "smw_update", "krylov_iter", "run_time_k", "setup_time_k", "solve_time_k",
        "outer_iter_k", "inner_iter_k", "run_time_q", "setup_time_q", "solve_time_q",
        "outer_iter_q", "inner_iter_q")]
    g = m.groupby(config)
    out = g[summed].sum()
    out["steps"] = g.size()
    out["solved_k"] = 1
    out["solved_q"] = 1
    out = out.reset_index()
    out.insert(0, "name", out["M"].astype(str) + "x" + out["N"].astype(str) + "/" + out["loop_mode"])
    return out, used


# ---------------------------------------------------------------------------
# The gap
# ---------------------------------------------------------------------------

def gap_table(m: pd.DataFrame) -> pd.DataFrame:
    """Per problem/configuration solved by both: times, iterations, per-iteration times and ratios."""
    both = m[(m["solved_k"] == 1) & (m["solved_q"] == 1)].copy()
    out = pd.DataFrame({"name": both["name"]})
    out["ksp_time"] = both["run_time_k"]
    out["qpalm_time"] = both["run_time_q"]
    out["time_ratio"] = both["run_time_k"] / both["run_time_q"]
    out["ksp_setup"] = both["setup_time_k"]
    out["qpalm_setup"] = both["setup_time_q"]
    out["ksp_outer"] = both["outer_iter_k"]
    out["qpalm_outer"] = both["outer_iter_q"]
    out["ksp_inner"] = both["inner_iter_k"]
    out["qpalm_inner"] = both["inner_iter_q"]
    out["inner_ratio"] = both["inner_iter_k"] / both["inner_iter_q"]
    out["ksp_per_iter"] = both["solve_time_k"] / both["inner_iter_k"]
    out["qpalm_per_iter"] = both["solve_time_q"] / both["inner_iter_q"]
    out["per_iter_ratio"] = out["ksp_per_iter"] / out["qpalm_per_iter"]
    ssn = both["inner_iter_k"].where(both["inner_iter_k"] > 0)
    for ph in SSN_PHASES:
        out[f"ksp_{ph}_per_iter"] = both[ph] / ssn
    out["ksp_pmm_other_per_iter"] = both["t_pmm_other"] / ssn
    out["ksp_dominant_phase"] = both[SSN_PHASES].idxmax(axis=1).str.removeprefix("t_")
    out["ksp_fact_per_iter"] = both["fact"] / ssn
    out["ksp_smw_per_iter"] = both["smw_count"] / ssn
    for c in _fact_columns(both):
        out[f"ksp_{c}"] = both[c]
    return out.reset_index(drop=True)


def fact_time_by_reason(trace_prefix: Path, names: set[str], used: pd.DataFrame | None) -> pd.DataFrame | None:
    """Factorization count and time by reason from the KSP-QP trace (events), over `names`
    (MPC: over the solves in `used`, the repeats the summary uses)."""
    path = Path(f"{trace_prefix}_events.csv")
    if not path.exists():
        print(f"note: {path} not found; skipping factorization time by reason", file=sys.stderr)
        return None
    ev = pd.read_csv(path, dtype={"reason": str}, keep_default_na=False, na_values=["", "nan"])
    if used is not None:
        ev = ev.merge(used, on=list(used.columns))
        ev["name"] = ev["M"].astype(str) + "x" + ev["N"].astype(str) + "/" + ev["loop_mode"]
    ev = ev[(ev["action"] == "factorize") & ev["name"].isin(names)]
    ev = ev.assign(t=ev["t_assembly"] + ev["t_analyze"] + ev["t_factorize"])
    by = ev.groupby(["target", "reason"]).agg(count=("t", "size"), time=("t", "sum"),
                                             with_analysis=("analyzed", "sum"))
    by["share_of_count"] = by["count"] / by["count"].sum()
    by["share_of_time"] = by["time"] / by["time"].sum()
    return by.sort_values("time", ascending=False)


def report(gap: pd.DataFrame, m: pd.DataFrame, top: int, trace_prefix: Path | None,
           used: pd.DataFrame | None) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    n_k = int((m["solved_k"] == 1).sum())
    n_q = int((m["solved_q"] == 1).sum())
    print(f"{len(m)} in both files; solved by KSP-QP {n_k}, by QPALM {n_q}, by both {len(gap)}\n")
    if gap.empty:
        return

    print("Geometric means over those solved by both (KSP-QP / QPALM):")
    for col, label in [("time_ratio", "run time"), ("inner_ratio", "inner iterations"),
                       ("per_iter_ratio", "solve time per inner iteration")]:
        print(f"  {label:32s} {_geomean(gap[col]):8.3f}   (KSP-QP lower on {int((gap[col] < 1).sum())}/{len(gap)})")
    print(f"  {'setup time':32s} {_geomean(gap['ksp_setup'] / gap['qpalm_setup']):8.3f}\n")

    both = m[(m["solved_k"] == 1) & (m["solved_q"] == 1)]
    total = both["run_time_k"].sum()
    shares = pd.Series({ph.removeprefix("t_"): both[ph].sum() / total for ph in PHASES})
    per_iter = pd.Series({ph.removeprefix("t_"): both[ph].sum() / both["inner_iter_k"].sum() for ph in PHASES})
    print("KSP-QP run time by phase (over those solved by both):")
    print(pd.DataFrame({"share": shares.round(3), "s_per_ssn_iter": per_iter}).to_string(), "\n")

    facts = both[_fact_columns(both)].sum()
    facts = facts[facts > 0].sort_values(ascending=False)
    print(f"KSP-QP factorizations by reason: {int(facts.sum())} in total, "
          f"{both['fact'].sum() / both['inner_iter_k'].sum():.2f} per SSN iteration; "
          f"SMW updates {int(both['smw_count'].sum())}")
    print((facts / facts.sum()).round(3).rename(lambda c: c.removeprefix("fact_")).to_string(), "\n")

    if trace_prefix is not None:
        by = fact_time_by_reason(trace_prefix, set(gap["name"]), used)
        if by is not None:
            print("KSP-QP factorization time by reason (trace):")
            print(by.to_string(float_format=lambda v: f"{v:.4g}"), "\n")

    cols = ["name", "time_ratio", "inner_ratio", "per_iter_ratio", "ksp_per_iter", "qpalm_per_iter",
            "ksp_dominant_phase", "ksp_fact_per_iter", "ksp_smw_per_iter", "ksp_inner", "qpalm_inner"]
    print(f"Largest per-iteration gaps (top {top}):")
    print(gap.sort_values("per_iter_ratio", ascending=False)[cols].head(top).to_string(
        index=False, float_format=lambda v: f"{v:.4g}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark", choices=["mm", "mpc"])
    parser.add_argument("--kspqp", required=True, type=Path, help="KSP-QP result file (mm_kspqp.csv / mpc_kspqp.csv)")
    parser.add_argument("--qpalm", required=True, type=Path, help="QPALM result file (mm_qpalm.csv / mpc_qpalm.csv)")
    parser.add_argument("--trace-prefix", type=Path, default=None,
                        help="KSP-QP trace prefix (reads <prefix>_events.csv) for the factorization time by reason")
    parser.add_argument("--out", type=Path, default=None, help="Write the per-problem/configuration table here")
    parser.add_argument("--top", type=int, default=20, help="Rows of the largest per-iteration gaps to print")
    args = parser.parse_args()

    m, used = load_mm(args.kspqp, args.qpalm) if args.benchmark == "mm" else load_mpc(args.kspqp, args.qpalm)
    gap = gap_table(m)
    report(gap, m, args.top, args.trace_prefix, used)
    if args.out is not None:
        gap.to_csv(args.out, index=False)
        print(f"\nPer-{'problem' if args.benchmark == 'mm' else 'configuration'} table written to: {args.out}")


if __name__ == "__main__":
    main()
