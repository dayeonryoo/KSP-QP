"""
Compares two KSP-QP runs on the Maros-Meszaros QPs (MM) and the platoon MPC problems, from their result files and
traces (experimental; branch direct-vs-pcg): e.g. PCG against the direct solver (benchmark_*.py --direct), or a run
before and after a change. Run A is the reference, so ratios are B / A.

MM, over the problems in both runs:
  - the problems each run solves, and those only one of them solves;
  - over the problems both solve: run time, PMM and SSN iterations and time per SSN iteration, overall, by problem
    class (from the trace, as in analyze_performance.py) and by size;
  - over the problems on which both take the same numbers of PMM and SSN iterations, the run time by class and by
    A's CG iterations per SSN iteration: there the runs differ only in the cost of their iterations;
  - the phases, the time per factorization and its symbolic analysis (traces);
  - the inner (SSN) solves stopped at the iteration limit j_max (PMM traces), and the largest SSN-iteration changes;
  - with --qpalm-prefix, both runs against QPALM.
MPC, over the (M, N) grid (sums over the steps both solve, each run's fastest timed repeat):
  - robustness, and solves whose own run time disagrees with the benchmark's wall time;
  - run time, iterations and time per SSN iteration by configuration;
  - the identical-step prefix (PMM traces): the PMM iterations before the runs first take different numbers of SSN
    iterations, over which B / A is the cost of the same steps, and the mu at which the runs part;
  - the time per factorization, split into symbolic analysis and numeric factorization (traces).

=== HOW TO RUN ===
  python3 compare_runs.py --a ../results/1014_baseline/1014 --b ../results/1014_direct/1014_direct \\
      --labels pcg direct --qpalm-prefix ../results/1014_baseline/1014
  python3 compare_runs.py --a ../results/1014 --b ../results/1015 --labels before after --sets mm

--a/--b P read P_{mm,mpc}_kspqp.csv and KSP-QP's traces P_{mm,mpc}_kspqp_trace_{events,ssn,pmm}.csv (benchmark_*.py
--trace); what needs a missing file is skipped. --qpalm-prefix Q reads Q_mm_qpalm.csv. --out DIR writes the
per-problem (compare_mm.csv) and per-configuration (compare_mpc.csv) tables.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from analyze_profile import KSPQP_NAMES, QPALM_NAMES, _geomean, _fastest_timed_repeat
from analyze_performance import (CLASSES, EVENT_COLS, INSTANCE, KSPQP_STATUS, NETLIB_QP, SIZE_BINS, SIZE_LABELS,
                                 SOLVE_PHASES, _heading, _percent, _read_trace, _table, _title, _with_fact_time,
                                 lifted_from_trace)

SIZE_COLS = ["n", "m", "l", "nnz_Q", "nnz_A", "nnz_B"]
CG_BINS, CG_LABELS = [-np.inf, 1, 5, 20, np.inf], ["<=1", "1-5", "5-20", ">20"]
PMM_COLS = ["pmm_iter", "ssn_iters", "ssn_status", "mu", "t_total"]


def _ratios(r: pd.Series, b: str) -> str:
    """Geometric mean of B / A ratios, their median, and how often B is lower."""
    r = r[np.isfinite(r) & (r > 0)]
    return f"{_geomean(r):6.2f}   (median {r.median():.2f}; {b} lower on {int((r < 1).sum())}/{len(r)})"


def _provenance(label: str, df: pd.DataFrame) -> None:
    cols = [c for c in ["ssn_git_sha", "ksp_mode", "ssn_tol", "ssn_time_limit", "ssn_run_id"] if c in df]
    print(f"{label}: " + ", ".join(f"{c}={'/'.join(map(str, df[c].dropna().unique()))}" for c in cols))


def _status(s) -> str:
    return KSPQP_STATUS.get(int(s), str(s))


# ---------------------------------------------------------------------------
# MM
# ---------------------------------------------------------------------------

def load_mm_pair(a: Path, b: Path) -> pd.DataFrame:
    """Per problem in both MM result files: KSP-QP's columns with suffix _a / _b (load_mm()'s names), A's sizes."""
    ka = pd.read_csv(f"{a}_mm_kspqp.csv").rename(columns=KSPQP_NAMES)
    kb = pd.read_csv(f"{b}_mm_kspqp.csv").rename(columns=KSPQP_NAMES)
    sizes = [c for c in SIZE_COLS if c in ka]
    m = ka.merge(kb.drop(columns=[c for c in sizes if c in kb]), on="name", suffixes=("_a", "_b"))
    new = {"both": (m["solved_a"] == 1) & (m["solved_b"] == 1)}
    for run in ("a", "b"):
        iters = m[f"inner_iter_{run}"].where(m[f"inner_iter_{run}"] > 0)
        new[f"per_iter_{run}"] = m[f"solve_time_{run}"] / iters
        new[f"cg_per_ssn_{run}"] = m[f"krylov_iter_{run}"] / iters
    new["time_ratio"] = m["run_time_b"] / m["run_time_a"]
    new["pmm_ratio"] = m["outer_iter_b"] / m["outer_iter_a"]
    new["ssn_ratio"] = m["inner_iter_b"] / m["inner_iter_a"]
    new["per_iter_ratio"] = new["per_iter_b"] / new["per_iter_a"]
    new["same_iters"] = (m["inner_iter_a"] == m["inner_iter_b"]) & (m["outer_iter_a"] == m["outer_iter_b"])
    size_cols = ["n", "nnz_Q", "nnz_A", "nnz_B"]
    new["size"] = m[size_cols].sum(axis=1) if set(size_cols) <= set(m) else pd.Series(np.nan, index=m.index)
    return pd.concat([m, pd.DataFrame(new)], axis=1).copy()


def mm_classes(m: pd.DataFrame, prefixes: list[Path]) -> pd.Series:
    """Each problem's class (diagonal Q / lifted / lifted Q*) from the first run whose trace is there."""
    for prefix in prefixes:
        events = _read_trace(prefix, "mm", "events", ["name", "pmm_iter", "ssn_iter", "s"])
        ssn = _read_trace(prefix, "mm", "ssn", ["name", "pmm_iter", "ssn_iter", "n_act_W"]) if events is not None else None
        if events is not None and ssn is not None and {"n", "m"} <= set(m):
            lifted = m["name"].map(lifted_from_trace(m, events, ssn))
            q_star = m["name"].map(lambda s: bool(NETLIB_QP.match(s)))
            return pd.Series(np.select([lifted == 0, (lifted == 1) & ~q_star, (lifted == 1) & q_star],
                                       CLASSES, default="unknown"), index=m.index)
    return pd.Series("unknown", index=m.index)


def mm_robustness(m: pd.DataFrame, la: str, lb: str) -> None:
    _heading("Robustness")
    sa, sb = m["solved_a"] == 1, m["solved_b"] == 1
    print(f"{len(m)} problems: solved by {la} {int(sa.sum())}, by {lb} {int(sb.sum())}, by both {int((sa & sb).sum())}, "
          f"by neither {int((~sa & ~sb).sum())}")
    diff = m[sa != sb]
    if not diff.empty:
        tab = pd.DataFrame({"class": diff["class"],
                            f"{la}": diff["ssn_status_a"].map(_status), f"{lb}": diff["ssn_status_b"].map(_status),
                            f"time_{la}": diff["run_time_a"], f"time_{lb}": diff["run_time_b"],
                            f"SSN_{la}": diff["inner_iter_a"], f"SSN_{lb}": diff["inner_iter_b"]})
        tab.index = diff["name"]
        print("Solved by only one of them:")
        _table(tab)


def mm_both(m: pd.DataFrame, la: str, lb: str) -> None:
    b = m[m["both"]]
    _heading(f"{lb} / {la} over the {len(b)} problems both solve (geometric means)")
    for label, r in [("run time", b["time_ratio"]), ("PMM iterations", b["pmm_ratio"]),
                     ("SSN iterations", b["ssn_ratio"]), ("time per SSN iteration", b["per_iter_ratio"])]:
        print(f"  {label:24s} {_ratios(r, lb)}")
    more = b["inner_iter_b"] > b["inner_iter_a"]
    print(f"  SSN iterations: same on {int((b['inner_iter_b'] == b['inner_iter_a']).sum())}, more in {lb} on "
          f"{int(more.sum())} (at least 2x on {int((b['ssn_ratio'] >= 2).sum())}), fewer on "
          f"{int((b['inner_iter_b'] < b['inner_iter_a']).sum())}")

    def by(groups: pd.Series, order: list) -> pd.DataFrame:
        rows = {}
        for g in order:
            x = b[groups == g]
            if len(x):
                rows[g] = {"problems": len(x), "run_time": _geomean(x["time_ratio"]), "SSN_its": _geomean(x["ssn_ratio"]),
                           "per_SSN_it": _geomean(x["per_iter_ratio"]), f"CG_per_SSN_{la}": x["cg_per_ssn_a"].median(),
                           f"CG_per_SSN_{lb}": x["cg_per_ssn_b"].median(),
                           f"{lb}_faster": int((x["time_ratio"] < 1).sum())}
        return pd.DataFrame(rows).T

    if b["class"].ne("unknown").any():
        print("\nBy problem class (CG per SSN iteration: medians):")
        _table(by(b["class"], CLASSES))
    if b["size"].notna().any():
        print("\nBy size = nnz(Q) + nnz(A) + nnz(B) + n:")
        _table(by(pd.cut(b["size"], SIZE_BINS, labels=SIZE_LABELS, right=False), SIZE_LABELS))

    s = b[b["same_iters"]]
    _heading(f"Same PMM and SSN iterations in both runs ({len(s)} problems): {lb} / {la} run time, the cost of the "
             f"same iterations")
    print(f"  all {_ratios(s['time_ratio'], lb)}")
    rows = {}
    for label, groups, order in [("class", s["class"], CLASSES),
                                 (f"CG/SSN in {la}", pd.cut(s["cg_per_ssn_a"], CG_BINS, labels=CG_LABELS), CG_LABELS)]:
        for g in order:
            x = s[groups == g]
            if len(x):
                rows[(label, g)] = {"problems": len(x), "run_time": _geomean(x["time_ratio"]),
                                    f"{lb}_faster": int((x["time_ratio"] < 1).sum())}
    _table(pd.DataFrame(rows).T)

    _heading("Phases (mean of per-problem shares of the solve time, problems both solve)")
    tab = pd.DataFrame({run_label: b[[f"{c}_{run}" for c in SOLVE_PHASES]].pipe(
        lambda x: x.div(x.sum(axis=1), axis=0).mean()).set_axis([c.removeprefix("t_") for c in SOLVE_PHASES])
        for run, run_label in (("a", la), ("b", lb))})
    print(_percent(tab).to_string())
    for run, label in (("a", la), ("b", lb)):
        print(f"  {label}: factorizations per SSN iteration {(b[f'fact_{run}'] / b[f'inner_iter_{run}']).median():.2f}, "
              f"SMW updates {(b[f'smw_count_{run}'] / b[f'inner_iter_{run}']).median():.2f} (medians)")


def mm_factorizations(m: pd.DataFrame, events: dict, la: str, lb: str) -> None:
    names = set(m.loc[m["both"], "name"])
    per = {}
    _heading("Factorizations (traces; problems both solve)")
    for run, label in (("a", la), ("b", lb)):
        if events[run] is None:
            return
        f = _with_fact_time(events[run][events[run]["name"].isin(names)])
        print(f"  {label}: {len(f)} factorizations, {f['analyzed'].mean():.0%} with a symbolic analysis, which takes "
              f"{f['t_analyze'].sum() / f['t'].sum():.0%} of the factorization time; by reason: "
              + ", ".join(f"{r} {n}" for r, n in f["reason"].value_counts().items()))
        per[run] = f.groupby("name")[["t", "t_factorize"]].median()
    j = per["a"].join(per["b"], lsuffix="_a", rsuffix="_b", how="inner")
    j["ratio"], j["numeric"] = j["t_b"] / j["t_a"], j["t_factorize_b"] / j["t_factorize_a"]
    j = j.join(m.set_index("name")["class"])
    rows = {cls: {"problems": len(x), "per_fact": _geomean(x["ratio"]), "per_fact_median": x["ratio"].median(),
                  "numeric_only": _geomean(x["numeric"])}
            for cls, x in [*((c, j[j["class"] == c]) for c in CLASSES), ("all", j)] if len(x)}
    print(f"Median time per factorization of each problem, {lb} / {la} (geometric mean over the problems):")
    _table(pd.DataFrame(rows).T)


def mm_inner_limits(m: pd.DataFrame, pmm: dict, la: str, lb: str) -> None:
    if pmm["a"] is None or pmm["b"] is None:
        return
    names = set(m.loc[m["both"], "name"])
    stops = pd.DataFrame({run: (p["ssn_status"] == 1).groupby(p["name"]).sum() for run, p in pmm.items()}).fillna(0)
    stops = stops[stops.index.isin(names)]
    _heading("Inner solves stopped at the iteration limit j_max (PMM traces; problems both solve)")
    print(f"  problems with such stops: {la} {int((stops['a'] > 0).sum())}, {lb} {int((stops['b'] > 0).sum())}; "
          f"stops in total: {la} {int(stops['a'].sum())}, {lb} {int(stops['b'].sum())}; more in {lb} on "
          f"{int((stops['b'] > stops['a']).sum())} problems, fewer on {int((stops['b'] < stops['a']).sum())}")
    b = m[m["both"]].sort_values("ssn_ratio", ascending=False).head(10)
    tab = pd.DataFrame({"class": b["class"], f"PMM_{la}": b["outer_iter_a"], f"PMM_{lb}": b["outer_iter_b"],
                        f"SSN_{la}": b["inner_iter_a"], f"SSN_{lb}": b["inner_iter_b"],
                        f"jmax_stops_{la}": b["name"].map(stops["a"]), f"jmax_stops_{lb}": b["name"].map(stops["b"]),
                        "run_time_ratio": b["time_ratio"]})
    tab.index = b["name"]
    print(f"Largest SSN-iteration ratios {lb} / {la}:")
    _table(tab)


def mm_vs_qpalm(m: pd.DataFrame, qpalm_prefix: Path, la: str, lb: str) -> None:
    path = Path(f"{qpalm_prefix}_mm_qpalm.csv")
    if not path.exists():
        print(f"note: {path} not found; comparison with QPALM skipped", file=sys.stderr)
        return
    q = pd.read_csv(path).rename(columns=QPALM_NAMES)[["name", "solved", "run_time"]]
    x = m[m["both"]].merge(q[q["solved"] == 1], on="name")
    ra, rb = x["run_time_a"] / x["run_time"], x["run_time_b"] / x["run_time"]
    _heading(f"Against QPALM ({len(x)} problems solved by both runs and by QPALM; run time / QPALM's)")
    print(f"  {la:8s} {_ratios(ra, la)}\n  {lb:8s} {_ratios(rb, lb)}")
    print(f"  {la} / QPALM = ({la} / {lb}) x ({lb} / QPALM): {_geomean(ra):.2f} = "
          f"{_geomean(ra / rb):.2f} x {_geomean(rb):.2f}")


def mm_report(a: Path, b: Path, la: str, lb: str, qpalm_prefix: Path | None) -> pd.DataFrame | None:
    if not (Path(f"{a}_mm_kspqp.csv").exists() and Path(f"{b}_mm_kspqp.csv").exists()):
        print("note: an MM result file is missing; MM skipped", file=sys.stderr)
        return None
    m = load_mm_pair(a, b)
    m = m.assign(**{"class": mm_classes(m, [a, b])})
    _title(f"Maros-Meszaros QPs: {lb} against {la}")
    _provenance(la, pd.read_csv(f"{a}_mm_kspqp.csv"))
    _provenance(lb, pd.read_csv(f"{b}_mm_kspqp.csv"))
    mm_robustness(m, la, lb)
    mm_both(m, la, lb)
    events = {run: _read_trace(p, "mm", "events", EVENT_COLS) for run, p in (("a", a), ("b", b))}
    mm_factorizations(m, events, la, lb)
    pmm = {run: _read_trace(p, "mm", "pmm", ["name", *PMM_COLS]) for run, p in (("a", a), ("b", b))}
    mm_inner_limits(m, pmm, la, lb)
    if qpalm_prefix is not None:
        mm_vs_qpalm(m, qpalm_prefix, la, lb)
    return m


# ---------------------------------------------------------------------------
# MPC
# ---------------------------------------------------------------------------

def _fastest(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All rows, and per instance the fastest timed repeat (load_mpc()'s choice), with load_mm()'s names."""
    raw = pd.read_csv(path).rename(columns=KSPQP_NAMES)
    return raw, _fastest_timed_repeat(raw, INSTANCE)


def mpc_report(a: Path, b: Path, la: str, lb: str) -> pd.DataFrame | None:
    pa, pb = Path(f"{a}_mpc_kspqp.csv"), Path(f"{b}_mpc_kspqp.csv")
    if not (pa.exists() and pb.exists()):
        print("note: an MPC result file is missing; MPC skipped", file=sys.stderr)
        return None
    (raw_a, fa), (raw_b, fb) = _fastest(pa), _fastest(pb)
    _title(f"Platoon MPC: {lb} against {la}")
    _provenance(la, raw_a)
    _provenance(lb, raw_b)

    _heading("Robustness")
    for label, raw in ((la, raw_a), (lb, raw_b)):
        r0 = raw[raw["rep"] == 0]
        print(f"  {label}: solved {int(r0['solved'].sum())} of {len(r0)} instances (repeat 0); statuses over all repeats: "
              + ", ".join(f"{_status(s)} {n}" for s, n in raw["ssn_status"].value_counts().items()))
        if "ssn_wall_time" in raw:
            odd = raw[raw["run_time"] > 2 * raw["ssn_wall_time"] + 1.0]
            for _, r in odd.iterrows():
                print(f"    run time {r['run_time']:.1f} s but wall time {r['ssn_wall_time']:.1f} s at (M, N) = "
                      f"({r['M']}, {r['N']}), step {r['step']}, repeat {r['rep']} ({_status(r['ssn_status'])}): "
                      f"the solver's clock ran without the process (system sleep?)")

    keep = [*INSTANCE, "rep", "solved", "run_time", "solve_time", "outer_iter", "inner_iter", "krylov_iter", "fact",
            "n_z", "t_fact", "t_analyze", "t_krylov", "t_direct_solve"]
    x = fa[keep].merge(fb[keep], on=INSTANCE, suffixes=("_a", "_b"))
    x = x[(x["solved_a"] == 1) & (x["solved_b"] == 1)]
    config = ["M", "N", "loop_mode"]
    summed = [c for c in x.columns if c.endswith(("_a", "_b")) and not c.startswith(("rep", "solved", "n_z"))]
    cfg = x.groupby(config)[summed].sum().join(x.groupby(config)["step"].size().rename("steps")).reset_index()
    cfg["time_ratio"] = cfg["run_time_b"] / cfg["run_time_a"]
    cfg["pmm_ratio"] = cfg["outer_iter_b"] / cfg["outer_iter_a"]
    cfg["ssn_ratio"] = cfg["inner_iter_b"] / cfg["inner_iter_a"]
    cfg["per_iter_ratio"] = (cfg["solve_time_b"] / cfg["inner_iter_b"]) / (cfg["solve_time_a"] / cfg["inner_iter_a"])

    # Identical-step prefix: PMM iterations before the first with different SSN iterations or SSN status.
    pmm = {run: _read_trace(p, "mpc", "pmm", [*INSTANCE, "rep", *PMM_COLS]) for run, p in (("a", a), ("b", b))}
    if pmm["a"] is not None and pmm["b"] is not None:
        for run in ("a", "b"):  # the repeats used above
            used = x[[*INSTANCE, f"rep_{run}"]].rename(columns={f"rep_{run}": "rep"})
            pmm[run] = pmm[run].merge(used, on=[*INSTANCE, "rep"]).drop(columns="rep")
        j = (pmm["a"].merge(pmm["b"], on=[*INSTANCE, "pmm_iter"], suffixes=("_a", "_b"))
             .sort_values([*INSTANCE, "pmm_iter"]))
        same = (j["ssn_iters_a"] == j["ssn_iters_b"]) & (j["ssn_status_a"] == j["ssn_status_b"])
        j["prefix"] = same.groupby([j[c] for c in INSTANCE]).cumprod().astype(bool)
        pre = j[j["prefix"]].groupby(config)[["t_total_a", "t_total_b", "ssn_iters_a"]].sum()
        cfg = cfg.merge((pre["t_total_b"] / pre["t_total_a"]).rename("prefix_ratio").reset_index(), on=config, how="left")
        cfg = cfg.merge(pre["ssn_iters_a"].rename("prefix_ssn").reset_index(), on=config, how="left")
        parted = j[~j["prefix"]].groupby(INSTANCE).first()

    for mode, g in cfg.groupby("loop_mode"):
        _heading(f"{lb} / {la}, loop mode '{mode}' ({len(g)} configurations; sums over the steps both solve, "
                 f"fastest timed repeats)")
        for label, r in [("run time", g["time_ratio"]), ("PMM iterations", g["pmm_ratio"]),
                         ("SSN iterations", g["ssn_ratio"]), ("time per SSN iteration", g["per_iter_ratio"])]:
            print(f"  {label:24s} {_ratios(r, lb)}")
        print("Run time ratio by M (rows) and N (columns):")
        _table(g.pivot(index="M", columns="N", values="time_ratio"))
        if "prefix_ratio" in g:
            print(f"Identical-step prefix: over the PMM iterations before the runs first take different numbers of SSN "
                  f"iterations ({int(g['prefix_ssn'].sum())} SSN iterations), {lb} / {la} time "
                  f"{_ratios(g['prefix_ratio'], lb)}; range {g['prefix_ratio'].min():.2f}-{g['prefix_ratio'].max():.2f}")
            p = parted[parted.index.get_level_values("loop_mode") == mode]
            if len(p):
                print(f"  the runs part at mu {p['mu_b'].median():.3g} on the median instance (range "
                      f"{p['mu_b'].min():.3g}-{p['mu_b'].max():.3g}), and never on "
                      f"{int((g['steps'].sum()) - len(p))} of the {int(g['steps'].sum())} instances")
            stops = {run: int(((q["loop_mode"] == mode) & (q["ssn_status"] == 1)).sum()) for run, q in pmm.items()}
            print(f"  inner solves stopped at j_max: {la} {stops['a']}, {lb} {stops['b']}")

    # Time per factorization, from the events of the repeats used.
    events = {run: _read_trace(p, "mpc", "events", [*EVENT_COLS, *INSTANCE, "rep"]) for run, p in (("a", a), ("b", b))}
    if events["a"] is not None and events["b"] is not None:
        _heading("Factorizations (traces; repeats used above)")
        per = {}
        for run, label in (("a", la), ("b", lb)):
            used = x[[*INSTANCE, f"rep_{run}"]].rename(columns={f"rep_{run}": "rep"})
            f = _with_fact_time(events[run].merge(used, on=[*INSTANCE, "rep"]))
            print(f"  {label}: {len(f)} factorizations, {f['analyzed'].mean():.0%} with a symbolic analysis, which takes "
                  f"{f['t_analyze'].sum() / f['t'].sum():.0%} of the factorization time; by reason: "
                  + ", ".join(f"{r} {n}" for r, n in f["reason"].value_counts().items()))
            per[run] = f.groupby(config)[["t", "t_factorize"]].median()
        r = per["b"]["t"] / per["a"]["t"]
        rn = per["b"]["t_factorize"] / per["a"]["t_factorize"]
        print(f"  median time per factorization {lb} / {la}: geometric mean over the configurations {_geomean(r):.2f} "
              f"(range {r.min():.2f}-{r.max():.2f}); numeric factorization only {_geomean(rn):.2f}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True, type=Path, help="Prefix of run A, the reference (e.g. ../results/1014)")
    parser.add_argument("--b", required=True, type=Path, help="Prefix of run B, compared with A")
    parser.add_argument("--labels", nargs=2, default=["A", "B"], metavar=("LABEL_A", "LABEL_B"))
    parser.add_argument("--qpalm-prefix", type=Path, default=None, help="Prefix of QPALM's MM result file")
    parser.add_argument("--sets", nargs="+", choices=["mm", "mpc"], default=["mm", "mpc"])
    parser.add_argument("--out", type=Path, default=None,
                        help="Directory for the per-problem (MM) and per-configuration (MPC) tables")
    args = parser.parse_args()
    la, lb = args.labels
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    tables = {}
    if "mm" in args.sets:
        tables["compare_mm.csv"] = mm_report(args.a, args.b, la, lb, args.qpalm_prefix)
    if "mpc" in args.sets:
        tables["compare_mpc.csv"] = mpc_report(args.a, args.b, la, lb)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        for file, df in tables.items():
            if df is not None:
                df.to_csv(args.out / file, index=False)
                print(f"\n{file} written to: {args.out}")


if __name__ == "__main__":
    main()
