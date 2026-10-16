# KSP-QP: Krylov Semismooth-Newton Proximal Method of Multipliers

A C++17 solver for convex **Quadratic Programs (QPs)** and **Linear Programs (LPs)**, with Python bindings via pybind11. The solver implements a Primal-Dual Proximal Method of Multipliers (PMM) outer loop with a Semi-Smooth Newton (SSN) inner solver, accelerated by iterative Krylov methods.

---

## Problem form

KSP-QP solves problems of the form:

```
min   c^T x  +  (1/2) x^T Q x
s.t.  A x  = b
      lw  <=  B x  <=  uw
      lx  <=  x    <=  ux
```

where `Q` is a symmetric positive semi-definite matrix (set `Q = 0` for LPs).

---

## Requirements

### C++ build
- CMake >= 3.16
- A C++17-compatible compiler (GCC 8+, Clang 7+, MSVC 2019+)
- Internet access for the first build (CMake auto-downloads Eigen and GoogleTest)

### Python bindings (optional)
- Python 3.9+
- `pip install numpy scipy pandas`
- `pip install qpalm osqp` (only needed for the comparison benchmark)

---

## Getting the code

```bash
git clone https://github.com/dayeonryoo/KSP-QP
cd KSP-QP
```

---

## Building

### Option A — C++ executables

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

> **Portability note:** `CMakeLists.txt` compiles with `-march=native`, which tunes for the CPU
> doing the build and is not portable to other machines — a binary built this way can crash with
> `Illegal instruction` on a different/older CPU. This is safe as long as everyone builds from
> source on their own machine (the normal workflow here); it becomes a problem if you copy a
> compiled `build/` to another machine, bake one into a Docker image that runs elsewhere, or
> ship a prebuilt release binary. In those cases, switch `-march=native` to a portable baseline
> (e.g. `-march=x86-64-v2`) first.

This produces two executables inside `build/`:

| Executable | Source | Description |
|---|---|---|
| `ksp_qp_netlib` | `src/netlib.cpp` | Runs the solver on Netlib LPs (`.mps`) |
| `ksp_qp_maros_meszaros` | `src/maros_meszaros.cpp` | Runs the solver on Maros-Meszaros QPs (`.SIF`) |

PDE-constrained QPs are generated in Python (`python/pde_generator.py`) rather than by a C++
driver, so that `include/` holds solver code only — see
[the PDE benchmark](#pde-constrained-qp-benchmarks-smooth-l2-regularized) below.

`ksp_qp_netlib` and `ksp_qp_maros_meszaros` each take `--name`, `--tol`, `--max-iter`, and
`--time-limit` flags. `--name` picks a single problem to solve, or `all` to sweep the whole set
and write a CSV (both also take `--root` and `--in` to point at a different data directory, and
`ksp_qp_netlib` takes `--set feasible|infeasible`). Run either of them with `--help` for the
full flag list. Run them from the repo root so their default (relative) data paths resolve, or
pass `--root`. The `PrintWhen`/`PrintWhat` verbosity is the one setting still hardcoded in each
driver's `main()` — see ["Running the C++ drivers"](#running-the-c-drivers) below.

### Option B — Python bindings

All commands are run from the `python/` subdirectory.

```bash
cd python
mkdir build && cd build
cmake .. -DPython3_EXECUTABLE=$(which python3)
cmake --build . --config Release
cd ..
```

This places `ksp_qp_bind.cpython-<tag>-<platform>.so` directly in `python/`, so
`import ksp_qp_bind` works without any installation step.

---

## Quick start

### C++ — solve a problem from matrices

```cpp
#include "ksp_qp.hpp"
#include "problem.hpp"

using T   = double;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<T>;

int main() {
    int n = 2;  // variables
    int m = 1;  // equality constraints
    int l = 0;  // general inequality rows (none here)

    // Q = [[2, 0], [0, 2]]  (positive definite)
    SpMat Q(n, n);
    Q.insert(0, 0) = 2.0;
    Q.insert(1, 1) = 2.0;
    Q.makeCompressed();

    // c = [-2, -5]
    Vec c(n);
    c << -2.0, -5.0;

    // A x = b  →  x[0] + x[1] = 1
    SpMat A(m, n);
    A.insert(0, 0) = 1.0;
    A.insert(0, 1) = 1.0;
    A.makeCompressed();

    Vec b(m);
    b << 1.0;

    // B (empty — no general inequality constraints)
    SpMat B(l, n);

    // Bounds on x: 0 <= x <= inf
    Vec lx = Vec::Zero(n);
    Vec ux = Vec::Constant(n, 1e20);

    // Bounds on Bx (empty)
    Vec lw(l), uw(l);

    T   tol        = 1e-6;
    int max_iter   = 10000;
    double time_limit = 60.0;   // seconds

    Problem<T> prob(Q, A, B, c, b, /*obj_const=*/0.0,
                    lx, ux, lw, uw,
                    tol, max_iter, time_limit,
                    PrintWhen::EVERY10, PrintWhat::TUNING);

    KSP_QP<T> solver(prob);
    Solution<T> sol = solver.solve();

    sol.print_summary();
    // sol.opt == TerminationStatus::Optimal  →  optimal
    // sol.obj_val                            →  optimal objective value
    // sol.x                                  →  primal solution vector
    return 0;
}
```

### C++ — solve from an MPS/SIF file

```cpp
#include "ksp_qp.hpp"
#include "problem.hpp"
#include "mps_format_parser.hpp"

using T = double;

int main() {
    MpsFormatParser<T> parser;
    ParsedModel<T> model = parser.parse("path/to/problem.mps");
    KSPQPdata<T>   pd    = parser.to_kspqp(model);

    T   tol        = 1e-6;
    int max_iter   = 1000000;
    double time_limit = 600.0;   // seconds

    Problem<T>  prob(pd, tol, max_iter, time_limit,
                     PrintWhen::EVERY10, PrintWhat::TUNING);
    KSP_QP<T>  solver(prob);
    Solution<T> sol = solver.solve();

    sol.print_summary();
    return 0;
}
```

### Python — solve from a SIF/MPS file

```python
import sys
sys.path.insert(0, "path/to/KSP-QP/python")
import ksp_qp_bind

result = ksp_qp_bind.solve_from_sif(
    "path/to/KSP-QP/data/maros_meszaros/QAFIRO.SIF",
    tol        = 1e-6,
    max_iter   = 1_000_000_000,
    time_limit = 600.0,
)

print("Status      :", result["status"])       # 0 = optimal
print("Objective   :", result["obj_val"])
print("Solve time  :", result["run_time"], "s")  # setup + solve
print("PMM iters   :", result["pmm_iter"])
print("SSN iters   :", result["ssn_iter"])
print("Krylov iters:", result["krylov_iter"])
```

### Python — parse a SIF file and feed to another solver

```python
import sys, scipy.sparse as sp, numpy as np
sys.path.insert(0, "path/to/KSP-QP/python")
import ksp_qp_bind

pd = ksp_qp_bind.parse_sif("path/to/problem.SIF")

# Reconstruct scipy sparse matrices
Q = sp.csc_matrix((pd["Q_data"], pd["Q_indices"], pd["Q_indptr"]), shape=pd["Q_shape"])
A = sp.csc_matrix((pd["A_data"], pd["A_indices"], pd["A_indptr"]), shape=pd["A_shape"])
B = sp.csc_matrix((pd["B_data"], pd["B_indices"], pd["B_indptr"]), shape=pd["B_shape"])

c, b   = pd["c"],  pd["b"]
lx, ux = pd["lx"], pd["ux"]
lw, uw = pd["lw"], pd["uw"]
n, m, l = pd["n"], pd["m"], pd["l"]
```

---

## API reference

### C++ `Problem<T>`

| Field | Type | Description |
|---|---|---|
| `Q` | `SpMat` | `n×n` symmetric PSD quadratic cost matrix |
| `A` | `SpMat` | `m×n` equality constraint matrix |
| `B` | `SpMat` | `l×n` general inequality constraint matrix |
| `c` | `Vec` | `n`-dim linear cost vector |
| `b` | `Vec` | `m`-dim RHS of equality constraints |
| `lx`, `ux` | `Vec` | `n`-dim variable bounds |
| `lw`, `uw` | `Vec` | `l`-dim bounds on `Bx` |
| `tol` | `T` | Primal-dual termination tolerance (default `1e-6`) |
| `max_iter` | `int` | Maximum PMM outer iterations |
| `time_limit` | `double` | Wall-clock limit in seconds on setup + solve (default `600`) |
| `when` | `PrintWhen` | `NEVER` / `EVERY10` / `ALWAYS` |
| `what` | `PrintWhat` | `NONE` / `MINIMAL` / `SSN` / `TUNING` / `FULL` (see ["Tuning: printing and profiling"](#tuning-printing-and-profiling)) |

### C++ `Solution<T>`

| Field | Type | Description |
|---|---|---|
| `opt` | `TerminationStatus` | Termination status (see table below) |
| `x` | `Vec` | Primal solution |
| `y1` | `Vec` | Dual variables for `Ax = b` |
| `y2` | `Vec` | Dual variables for `lw ≤ Bx ≤ uw` |
| `z` | `Vec` | Dual variables for variable bounds |
| `obj_val` | `T` | Primal objective value |
| `pmm_iter` | `int` | PMM outer iterations performed |
| `ssn_iter` | `int` | Total SSN inner iterations |
| `krylov_iter` | `int` | Total Krylov iterations |
| `fact` | `int` | Factorizations performed (of the preconditioner; of K or S by the direct solver) |
| `smw_count` | `int` | Sherman-Morrison-Woodbury low-rank updates used instead of refactorizing (of the preconditioner; of the direct solver's factorization in direct mode) |
| `pmm_tol_achieved` | `T` | Final PMM (outer) residual |
| `ssn_tol_achieved` | `T` | Final SSN (inner) residual |
| `setup_time` | `double` | Wall-clock seconds in the `KSP_QP<T>` constructor |
| `solve_time` | `double` | Wall-clock seconds in `solve()` |
| `run_time` | `double` | `setup_time + solve_time` |
| `linesearch_fail` | `int` | Line-search failures |
| `krylov_fail` | `int` | Krylov failures (fell back to a direct factorization) |

### Linear solver for the SSN Newton systems

By default each Newton system is solved by PCG on its Schur complement
`S = G H⁻¹ Gᵀ + (1/μ) I`, preconditioned by a factorization that is reused across iterations
through Sherman-Morrison-Woodbury (SMW) low-rank updates (`include/schur_preconditioner.hpp`).
PCG stops once `‖rhs − S·dy‖₂ < 0.1 · g / μ` (`SSN::krylov_tol_factor`), where g = max(ssn_tol, ‖∇M‖)
at the current iterate. The 1/μ is there because the gradient a step leaves is about μ·Gᵀ times the
Schur residual. Iterative refinement on the KKT system then brings the Newton residual ‖J·du + ∇M‖
to `0.1 · ssn_tol` if the full step keeps the active sets (the one step that can end the SSN loop), and
to `0.1 · g` otherwise, with at most 3 corrections. Both targets are absolute, with no floor. A solve
that misses its tolerance is a PCG failure. If PCG fails, the direct solver below takes over for the
rest of the run. Two public
`KSP_QP<T>` members select the experimental direct mode instead:

| Member | Default | Description |
|---|---|---|
| `direct_solve` | `false` | Skip PCG and solve every Newton system exactly, by LDLT on the KKT matrix `K = [-H, Gᵀ; G, (1/μ) I]` or Cholesky on `S`, whichever is estimated cheaper |
| `direct_smw` | `true` | With `direct_solve`: when the active sets differ by a few rows/columns from the last factorization's (and μ, ρ are unchanged), solve through an SMW update of that factorization instead of refactorizing (`include/schur_smw_update.hpp`). An update is kept only if iterative refinement brings its relative KKT residual to `1e-10`, the refinement target of a full factorization (`SSN::direct_smw_accept_tol`); otherwise that system is refactorized |

After `solve()`, `kkt_ldlt_fact` / `schur_chol_fact` count the direct solver's factorizations
(both included in `Solution::fact`), `direct_smw_count` the Newton systems solved through an SMW
update instead (included in `Solution::smw_count`), and `direct_smw_reject` the updates that the
residual check rejected. The SMW updates are not used in the fallback after a PCG failure.

`refactor_ratio` (default `1`; experimental, PCG only) lets the preconditioner keep its factorization
across μ/ρ changes. SMW updates for active-set changes continue on top of it, built at the
factorization's own μ_f and ρ_f, and it is refactorized once max(μ/μ_f, μ_f/μ) · max(ρ/ρ_f, ρ_f/ρ)
exceeds `refactor_ratio` (`1`: at every change). That product bounds the condition number of the kept
preconditioner relative to a fresh one. A PCG failure through a preconditioner kept this way is
retried with a fresh factorization. The drivers and the MM/MPC benchmarks take it as `--refactor-ratio R`.

### Termination status codes

`opt` is a scoped enum, `TerminationStatus` (`include/solution.hpp`); the underlying
`int` values are what the result CSVs and the Python bindings report.

| `TerminationStatus` | `int` | Meaning |
|---|---|---|
| `Optimal` | `0` | Optimal solution found |
| `NumericalError` | `-1` | Numerical error (setup or solve exception) |
| `PrimalInfeasible` | `-2` | Primal infeasibility detected |
| `DualInfeasible` | `-3` | Dual infeasibility detected |
| `MaxPmmIterations` | `1` | Maximum PMM iterations reached |
| `MaxSsnIterations` | `2` | Maximum SSN iterations reached |
| `TimeLimit` | `3` | Time limit exceeded |
| `Interrupted` | `4` | Solve was interrupted before converging |

`to_string(TerminationStatus)` gives the short label used by `print_summary()`.

### Python `ksp_qp_bind`

**`solve_from_sif(filename, tol=1e-6, max_iter=1_000_000_000, time_limit=600.0, direct=False, direct_smw=True, trace=False)`**
Parse and solve a SIF/MPS file. Returns a dict with keys:
`status`, `obj_val`, `setup_time`, `solve_time`, `run_time`, `pmm_iter`, `ssn_iter`,
`krylov_iter`, `fact`, `smw_count`, `pmm_tol_achieved`, `system`, `kkt_ldlt_fact`,
`schur_chol_fact`, `direct_smw_count`, `direct_smw_reject`, `ssn_tol_achieved`, `linesearch_fail`,
`krylov_fail`, the profile keys (`profile_columns()`; see
["Profiling and refactorization log"](#profiling-and-refactorization-log-experimental)), `x`, `y1`,
`y2`, `z`. With `trace=True` it also returns `trace_events`, `trace_ssn` and `trace_pmm`, each a
dict of equal-length lists (`pandas.DataFrame(r["trace_ssn"])`).
`x` and the multipliers are returned in the original, unscaled units, so they can be
checked directly against the problem data as given. `direct` and `direct_smw` set the
`KSP_QP<T>` members of the same names (see ["Linear solver"](#linear-solver-for-the-ssn-newton-systems));
`system` is `"S"` when only PCG ran, otherwise `"D"` followed by `"K"` and/or `"S"` for the
systems the direct solver factorized.

**`solve_from_data(pd, tol=1e-6, max_iter=1_000_000_000, time_limit=600.0, trace_path="", rho_init=-1.0, direct=False, direct_smw=True, trace=False)`**
Same, but takes already-parsed problem data — the dict returned by `parse_sif()`, or the one
built by `pde_generator.py` / `mpc_generator.py`. Returns the same keys as `solve_from_sif`.
`trace_path` is diagnostic-only: when set, it writes a per-iteration CSV trace (active-set
sizes, `mu`, `rho`, residuals) and turns on per-inner-iteration reporting, which has
non-trivial overhead — leave it empty for timing runs.

**`profile_columns()`**
The profile keys returned by the two solve functions, in a fixed order (for CSV headers).

**`parse_sif(filename)`**
Parse a SIF/MPS file and return problem data as numpy arrays.
Returns a dict with keys: `n`, `m`, `l`, `Q_*`, `A_*`, `B_*`, `c`, `b`, `lx`, `ux`, `lw`, `uw`, `obj_const`.
Sparse matrices are in CSC format (`_data`, `_indices`, `_indptr`, `_shape`).

---

## Running the C++ drivers

Each driver takes a small set of `--flag value` command-line options (see `--help`), and
defaults to paths relative to the repo root — **run them from the repo root**, or pass `--root`
to point at your clone from elsewhere. No editing or rebuilding is needed just to change `tol`,
`max_iter`, `time_limit`, or (where applicable) which data file to load.

### `ksp_qp_netlib` — Netlib LPs (`src/netlib.cpp`)

```bash
./build/ksp_qp_netlib [--root DIR] [--set feasible|infeasible] [--in DIR] [--name PROBLEM|all] \
                      [--tol T] [--max-iter N] [--time-limit S] [--direct [--no-direct-smw]] \
                      [--refactor-ratio R] [--out FILE] [--trace PREFIX] [--cooldown S] [--ref FILE]
```

`--set` picks the problem set and, with it, the default input directory, default problem, and
default output CSV:

| `--set` | Directory | Default `--name` | Success criterion |
|---|---|---|---|
| `feasible` (default) | `data/netlib-main/feasible/` | `AFIRO` | Objective matches the reference value |
| `infeasible` | `data/netlib-main/infeasible/` | `KLEIN1` | Solver *detects* infeasibility |

With a problem name it solves that one LP and prints the solution summary (`--set infeasible`
also prints a feasibility breakdown of the final iterate, showing *how* the problem fails to
admit a solution). Names are case-insensitive; the `.mps` files themselves are lowercase.

Pass `--name all` to sweep the set, appending a row per problem to
`<root>/results/pcg_netlib.csv` or `<root>/results/pcg_infeas.csv`
(override with `--out`). `--direct` runs the experimental direct mode
(see ["Linear solver"](#linear-solver-for-the-ssn-newton-systems)) and writes `direct_*.csv`
instead; `--no-direct-smw` additionally turns its SMW updates off (`direct_nosmw_*.csv`).
`--refactor-ratio R` sets `refactor_ratio` (see the same section); with R > 1 the default CSV is
`pcg_rr<R>_*.csv`. A CSV or trace file written with another header is refused rather than appended to.
The sweep is the directory listing, so adding or removing an `.mps` file is all it takes to
change the set — the same convention `python/benchmark_netlib.py` uses.

Both sets write the same CSV schema (`include/record_result.hpp`); its last columns are the
profile (see ["Profiling and refactorization log"](#profiling-and-refactorization-log-experimental)),
and `--trace PREFIX` also writes the trace files `PREFIX_{events,ssn,pmm}.csv`. Two columns are read
differently per set: on the infeasible set `agree` means "infeasibility was detected" (rather
than "objective matched"), `abs_err`/`rel_err` are `-1` since there is no objective to compare,
and `diverged` is always `0` because a large residual is the expected outcome there.

Reference objectives for the feasible set are not hardcoded — they ship with the dataset in
`data/netlib-main/feasible_gurobi_1e-8.csv` (Gurobi 10 at 1e-8; override with `--ref`). Those
disagree with the long-standing published Netlib optima on a handful of problems (E226 and
CRE-A among them), so treat either source with care.

For a QPALM/OSQP comparison, use `python/benchmark_netlib.py` and `python/benchmark_infeas.py`
instead (see below).

### `ksp_qp_maros_meszaros` — Maros-Meszaros QPs (`src/maros_meszaros.cpp`)

```bash
./build/ksp_qp_maros_meszaros [--root DIR] [--in DIR] [--name PROBLEM|all] [--tol T] [--max-iter N] [--time-limit S] \
                              [--direct [--no-direct-smw]] [--refactor-ratio R] [--out FILE] [--trace PREFIX] \
                              [--cooldown S]
```

Solves `<root>/<PROBLEM>.SIF` (default: `data/maros_meszaros/AUG2DCQP.SIF`), printing the
solution summary. Pass `--name all` to sweep the full Maros-Meszaros set against its built-in
reference objectives, appending a row to `<root>/results/pcg_mm.csv` (override with
`--out`) for each — `--cooldown` (default 3s) sleeps between problems in this mode, which keeps
a long sweep from being distorted by CPU thermal throttling. `--direct`, `--no-direct-smw` and
`--refactor-ratio` work as for `ksp_qp_netlib` (default CSVs `results/direct_mm.csv` /
`results/direct_nosmw_mm.csv` / `results/pcg_rr<R>_mm.csv`),
and so do the profile columns and `--trace PREFIX`. A single-problem run also prints the profile.

For comparing against QPALM/OSQP rather than just checking against reference objectives, use
the Python benchmark instead (see below).

### PDE-constrained QPs — generated in Python

There is no C++ driver for these: the generators live in `python/pde_generator.py` (Q1 FEM and
FD assembly, built on the element kernels in `python/fem_q1.py`), so `include/` stays
solver-only. They build the L2-regularized PDE-constrained control problems of
(Pearson & Gondzio, 2017):

| Choice | Generator | Problem |
|---|---|---|
| `poisson` | `make_poisson_l2_control` | 2D Poisson control, control-constrained |
| `poisson_state` | `make_poisson_l2_state_control` | 2D Poisson control, state-constrained |
| `convdiff` | `make_convdiff_l2_control` | 2D convection-diffusion control |

Each returns a `KSPQPdata`; `.to_dict()` gives the CSC/numpy dict that
`ksp_qp_bind.solve_from_data()` and `benchmark_common.kspqp_to_qpalm()` consume.
`generate_pde_l2_qp(choice, nc, beta, ...)` wraps all three and returns that dict directly:

```python
import pde_generator, ksp_qp_bind          # from the python/ directory
pd = pde_generator.generate_pde_l2_qp("convdiff", nc=6, beta=1e-2,
                                      y_lower=0.0, y_upper=0.2,
                                      u_lower=-0.75, u_upper=0.75, eps=0.01)
res = ksp_qp_bind.solve_from_data(pd, 1e-6, 10**9, 600.0)
```

`nc` is the grid exponent (grid size `2^nc + 1` per direction). `lumped_mass` swaps the
consistent Q1 mass matrix for the lumped one, and `discretization` selects `"fem"` (default)
or `"fd"` (5-point stencil with first-order upwind convection; always lumped).

The generated QP is normalized by the mesh size `h = 2^-nc`: the objective and the interior
(non-Dirichlet) rows of the state equation are divided by `h²` (`scale_by_mesh_size`). Every
mass-matrix entry is O(h²), so without this the KKT residuals, and with them every solver's
stopping test, would shrink as O(h²), making a fixed `tol` less demanding on finer meshes. The
solution `x = [y; u]` is unchanged, but objective values, including the `*_obj` columns written
by `benchmark_l2pde.py`, come out divided by `h²`.

---

## Running the benchmarks

The Python benchmark scripts compare **KSP-QP vs QPALM vs OSQP** and live in `python/`.
Build the Python binding first (see "Building" above), then `pip install qpalm osqp numpy scipy
pandas`.

### Maros-Meszaros QP benchmark

```bash
cd python
python3 benchmark_mm.py
```

Runs the full Maros-Meszaros test set. Writes one result file per solver:
`results/mm_kspqp.csv`, `results/mm_qpalm.csv`, `results/mm_osqp.csv`.

Each solver writes only its own file, with the problem size (`n` variables, `m` equality rows,
`l` inequality rows, and `nnz_Q`, `nnz_A`, `nnz_B`, the nonzero values of Q's lower triangle, A and
B), its setup/solve times and the run's provenance (run id,
timestamp, host, CPU, git revision or solver version, tolerance, time limit). A rerun replaces the
rows of the problems it solved and keeps the others, so KSP-QP can be rerun after every change while
the QPALM and OSQP results are kept. The files share the column names of the former combined
`comparison_mm.csv` (`ssn_*`, `qpalm_*`, `osqp_*`), and joining them on `name` and the size columns
gives that table plus the problem size (`benchmark_common.load_solver_tables`).

```
--root DIR             override project root (default: parent of script)
--tol 1e-6             primal-dual tolerance
--time-limit 60        per-problem time limit in seconds
--solver {ksp-qp,qpalm,osqp} [...]   which solvers to run (default: all three)
--problems NAME [...]  run only these problems (default: all)
--out PREFIX           output file prefix (e.g. 1015 -> results/1015_mm_kspqp.csv)
--cooldown 0           seconds to sleep between problems (avoids CPU throttling)
```

For example, to rerun KSP-QP only, keeping this prefix's QPALM and OSQP files:

```bash
python3 benchmark_mm.py --solver ksp-qp --out 1015
```

### Netlib LP benchmark

```bash
cd python
python3 benchmark_netlib.py
```

Runs the full Netlib LP test set (`data/netlib-main/feasible/`, 114 instances). Writes
`results/comparison_netlib.csv`. Same flags as `benchmark_mm.py`, except `--problems`: this
benchmark still writes one combined file for all solvers.

### Netlib infeasible LP benchmark

```bash
cd python
python3 benchmark_infeas.py
```

Runs the 29 primal-infeasible Netlib instances (`data/netlib-main/infeasible/`). Every problem
is known to be infeasible, so the metric is *detection*, not solve time: a solver succeeds when
it terminates with an infeasibility status. Writes `results/comparison_infeas.csv` with a
`*_detected` column per solver and the raw `*_status` alongside, so primal-vs-dual infeasibility
and the failure modes (time limit, iteration cap, or a false claim of optimality) stay
recoverable.

```
--root DIR             override project root (default: parent of script)
--tol 1e-6             solver tolerance
--time-limit 60        per-solver time limit in seconds
--solver {ksp-qp,qpalm,osqp} [...]   default: all three
--out PREFIX           output file prefix (default: comparison_infeas)
--cooldown 0           seconds between solver runs
```

### PDE-constrained QP benchmarks (smooth, L2-regularized)

```bash
cd python
python3 benchmark_l2pde.py
```

Produces three tables — `poisson_control`, `poisson_state`, `convdiff_both` — written to
`results/l2_<table>.csv`. The problems come from `pde_generator.py`
(Pearson & Gondzio, 2017); the sweep parameters follow their tables.

```
--root DIR              override project root (default: parent of script)
--tol 1e-6              solver tolerance
--time-limit 600        per-problem time limit in seconds (10 min default)
--table {poisson_control,poisson_state,convdiff_both} [...]   default: all three
--nc N [N ...]          grid exponents to sweep (default: 5 6 7 8 9)
--solver {ksp-qp,qpalm,osqp} [...]   default: all three
--cooldown 0            seconds to sleep between problems
--lumped-mass {0,1}     0 = consistent mass matrix (default), 1 = lumped
--discretization {fem,fd}   spatial discretization (default: fem)
--out PREFIX            output file prefix
```

### MPC benchmark (platoon / vehicle-chain)

```bash
cd python
python3 benchmark_mpc.py --out 0926          # -> results/0926_mpc_kspqp.csv, results/0926_mpc_qpalm.csv
python3 benchmark_mpc.py --solver ksp-qp --out 0926   # rerun KSP-QP only; 0926_mpc_qpalm.csv is kept
```

Benchmarks KSP-QP against QPALM on the platoon linear-MPC QP built by `mpc_generator.py` — a
native smooth QP (quadratic tracking cost, linear dynamics equalities, box bounds, no general
inequality rows) whose constraint matrix is banded with at most 6 nonzeros per row.

The problem has **two** size axes and they are not interchangeable, so the sweep is a grid
rather than a curve: `M` (vehicle count) grows the band *width* and the dense `2M × 2M` DARE
terminal cost block, while `N` (horizon) grows the band *length* at fixed bandwidth. Sweeping
one at a time through the corner of the `(M, N)` plane is misleading. Output is one file per
solver, in long format — one row per (instance × repeat) — in `results/<prefix>_mpc_kspqp.csv` and
`results/<prefix>_mpc_qpalm.csv`. Rerunning a configuration `(M, N, loop mode)` replaces that
configuration's rows in the files of the solvers run and keeps every other row. Rows of the two
files pair up on `(M, N, loop_mode, step, rep)`; with `--loop-mode shared` every run solves the same
instances, which the `xbar_norm` column confirms. After the instance columns, the files use the
column names of `benchmark_mm.py` (`ssn_*`, `pmm_iter`, … for KSP-QP, `qpalm_*` for QPALM),
including the run's provenance, and a line is printed per solve.

```
--root DIR             override project root (default: parent of script)
--tol 1e-6             solver tolerance
--time-limit 300       per-problem time limit in seconds
--M 5 10 20 50 100 200 300   vehicle counts to sweep
--N 20 50 100 200 400        horizons to sweep
--max-nz 65000         skip (M, N) configurations larger than this nonzero count
--T 6                  MPC rollout steps per configuration
--reps 4               timed repeats per instance (repeat 0 is discarded as warm-up)
--solver {ksp-qp,qpalm} [...]   default: both
--loop-mode {shared,own}        default: shared
--cooldown 0           seconds to sleep between problems
--out PREFIX           output file prefix
```

`validate_mpc_generator.py` checks the generated QPs independently (DARE residual, dynamics
consistency, KKT conditions of the solved instance) — run it after changing the generator:

```bash
cd python
python3 validate_mpc_generator.py [--max-M N]
```

### Per-iteration gap with QPALM (experimental)

`analyze_profile.py` joins a KSP-QP result file with a QPALM one (MM or MPC, possibly from different
runs) and reports, for each problem or (M, N) configuration solved by both:
- the run time and iterations of each solver;
- each solver's solve time per inner iteration and their ratio;
- KSP-QP's time per SSN iteration by phase, with its factorizations and SMW updates per SSN
  iteration.

Over all of them, it reports the geometric means of the ratios, KSP-QP's time by phase and its
factorizations by reason. With `--trace-prefix`, it also gives the factorization time by reason. See
["Profiling and refactorization log"](#profiling-and-refactorization-log-experimental).

```bash
python3 analyze_profile.py mm  --kspqp ../results/1015_mm_kspqp.csv --qpalm ../results/1003_mm_qpalm.csv
python3 analyze_profile.py mpc --kspqp ../results/1015_mpc_kspqp.csv --qpalm ../results/1015_mpc_qpalm.csv \
    --trace-prefix ../results/1015_mpc_kspqp_trace --out ../results/1015_mpc_gap.csv
```

### Performance summary (experimental)

`analyze_performance.py` summarizes KSP-QP against QPALM from the MM and MPC result files of one
run (`--prefix`) and KSP-QP's traces:
- MM: the problems each solver solves, with KSP-QP's failures classed as cost-bound or
  iteration-bound; run time and iterations against QPALM, split as solve time = inner iterations ×
  time per inner iteration, overall and by problem class (diagonal Q, lifted, and the lifted Q*
  problems built from Netlib LPs); the time per inner iteration against problem size; KSP-QP's
  phases, factorizations by reason and SMW updates; what-if bounds on the run-time ratio; the PMM and
  SSN behaviour; and the accuracy of the objective values.
- MPC: robustness and timing noise over the repeats; run-time ratios over the (M, N) grid; power-law
  fits of iterations, time per inner iteration and run time in M and N; phases, factorizations and
  what-if bounds.

The class of an MM problem comes from the trace, so without `--trace` files MM is not split by class.

```bash
python3 analyze_performance.py --prefix ../results/1014_baseline/1014
python3 analyze_performance.py --prefix ../results/1015 --qpalm-prefix ../results/1003 --sets mm \
    --out ../results/1015_perf   # also writes mm_problems.csv and mpc_configs.csv
```

### Comparing two KSP-QP runs (experimental)

`compare_runs.py` compares two KSP-QP runs, A and B, from their MM and MPC result files and traces:
PCG against the direct solver (`--direct`), or a run before and after a change. It reports the
problems only one run solves; B / A for run time, PMM and SSN iterations and time per SSN iteration,
by problem class and size; the run time over the problems on which both take the same iterations
(the cost of the same steps); the time per factorization and its symbolic analysis; and the inner
solves stopped at the iteration limit. On MPC it also compares the identical-step prefix, the PMM
iterations before the runs first differ, and flags solves whose own run time disagrees with the
benchmark's wall time.

```bash
python3 compare_runs.py --a ../results/1014_baseline/1014 --b ../results/1014_direct/1014_direct \
    --labels pcg direct --qpalm-prefix ../results/1014_baseline/1014
```

---

## Tests

```bash
# C++ (GoogleTest, built by the main CMakeLists when KSP_QP_BUILD_TESTS=ON, the default)
cmake --build build
ctest --test-dir build --output-on-failure

# Python generators (unittest)
cd python
python3 -m unittest discover -s tests -v
```

The C++ suite is one binary per header under test, mirroring `include/` — a broken
`solution` test cannot stop the `ssn` tests from running, and each binary stays fast to
rebuild while iterating. The Python suite covers the PDE generators (`fem_q1.py`,
`pde_generator.py`) and a KKT-residual check.

---

## Tuning: printing and profiling

The solver has two independent tools for diagnosing/tuning performance: **runtime printing**
(what gets printed to stdout while solving, controlled by `Problem<T>`'s `when`/`what` fields)
and the **profile** (why every factorization happened and where the run time went, recorded as
data).

### Runtime printing — `PrintWhen` / `PrintWhat` (`include/printing.hpp`)

`PrintWhen` controls *how often* a line is printed per PMM iteration:

| `PrintWhen` | Behavior |
|---|---|
| `NEVER` | No output |
| `EVERY10` | Print every 10th PMM iteration |
| `ALWAYS` | Print every PMM iteration |

`PrintWhat` controls *how much* is printed on each line:

| `PrintWhat` | Columns shown |
|---|---|
| `NONE` | Nothing (overrides `PrintWhen`) |
| `MINIMAL` | Iteration counts, residuals |
| `SSN` | + Krylov/factorization counts, PMM params (`mu`, `rho`, `eps`), line-search/Krylov failures — printed at every SSN iteration, not just per PMM iteration |
| `TUNING` | Same columns as `SSN`, at the normal per-PMM-iteration cadence |
| `FULL` | + objective value, but without the Krylov/factorization/failure columns |

Set these on the `Problem<T>` constructor, e.g. `Problem<T> prob(pd, tol, max_iter, time_limit,
PrintWhen::EVERY10, PrintWhat::TUNING);`. `TUNING` is the most useful combination for tuning
PMM/SSN hyperparameters — it shows residuals, `mu`/`rho`/`eps`, and failure counts together.
Turning printing off (`PrintWhen::NEVER` or `PrintWhat::NONE`) removes the stdout overhead
entirely, which matters when timing large sweeps.

### Profiling and refactorization log (experimental)

Branch `direct-vs-pcg` only (`include/solve_profile.hpp`). Always on; its overhead is a few clock
reads per SSN iteration and two per PCG iteration.

**Per run** (`KSP_QP<T>::profile` after `solve()`): it appears as the last columns of the C++
drivers' CSVs, as keys of the Python solve functions' dict, and in `mm_kspqp.csv` /
`mpc_kspqp.csv`. It has:

- **`fact_<reason>`**: the factorizations, of the PCG preconditioner or of the direct solver, by
  why they were done. They add up to `fact`.
- **Other counts:**
  - `fact_analyzed`: the factorizations that included a symbolic analysis.
  - `smw_update` (= `smw_count`): the Newton systems solved through an SMW update instead.
  - `smw_reuse`: the systems solved with the last factorization as is.
  - `mu_rho_keep`: the μ/ρ changes absorbed by keeping the preconditioner's factorization (`refactor_ratio` > 1).
  - `pcg_fail` (= `krylov_fail`).
  - `krylov_refine`: the Krylov iterations spent in iterative refinement.
  - `ls_steepest`: the steepest-descent line-search retries.
- **`t_<phase>`**: the run time split into disjoint phases that add up to `run_time` (see below).

| reason | why the factorization was done |
|---|---|
| `initial` | first factorization of the run (direct mode: of the direct solver) |
| `mu_rho` | mu and/or rho changed since the last factorization (the preconditioner: once its ratio exceeds `refactor_ratio`) |
| `rank` | the active sets moved too far from the last factorization's for an SMW update (rank > 50) |
| `singular` | the SMW capacitance matrix was (near-)singular |
| `pcg_fail` | PCG failed through an SMW-updated or kept (stale in mu/rho) preconditioner, so it was rebuilt in full and retried |
| `suppressed` | SMW updates were off after 5 failures in a row (until the next PMM iteration) |
| `no_snapshot` | no factorization to update from (snapshot wiped after a fail streak, or invalidated) |
| `switch` | switch between Cholesky and LDLT (preconditioner callback, or the direct solver's system) |
| `direct_fallback` | PCG failed for good: first factorization of the direct solver |
| `active_set` | direct solver without SMW updates (after a PCG failure, or `--no-direct-smw`): the active sets changed |
| `smw_residual` | direct mode: an SMW-updated solve failed the 1e-10 residual check |
| `chol_failed` | Cholesky on S failed, so LDLT on K took over |

When several reasons apply, the first in the order the SMW checks run wins: a forced rebuild,
the method, the snapshot, rho, mu, the rank, then the capacitance matrix. The trace keeps the
change flags.

| phase | contents |
|---|---|
| `t_setup` | `KSP_QP` constructor (scaling, factor of Q, ...) |
| `t_prep` | building each Newton system: active sets, H, G, right-hand side, factorization choice |
| `t_fact` | full factorizations: assembly, symbolic analysis, numeric factorization |
| `t_lowrank` | SMW update setups, accepted or not |
| `t_krylov` | PCG solves (main and refinement), preconditioner applications included |
| `t_direct_solve` | the direct solver's triangular solves, plain or through an SMW update |
| `t_linesearch` | exact line search and its steepest-descent retry |
| `t_update` | iterate update, gradient, SSN termination check |
| `t_ssn_other` | rest of the Newton solves (dx recovery, refinement residuals, bookkeeping) |
| `t_pmm_other` | rest of `solve()` (residual norms, multipliers, penalties, infeasibility checks) |

The last four columns are informational parts of these phases:
- `t_analyze` is part of `t_fact`;
- `t_prec_apply`, and `t_prec_apply_smw` within it, are part of `t_krylov`;
- `t_refine` is all of iterative refinement.

**Trace** (opt-in: `KSP_QP<T>::trace.enabled = true`, the drivers' `--trace PREFIX`, Python
`trace=True`, or the benchmarks' `--trace`):

- **`events`**: one row per factorization, SMW update or reuse, kept factorization (`mu_rho_keep`),
  and PCG failure. Each row has:
  - its PMM and SSN iteration;
  - `target` (`prec` | `direct`), `action`, `reason`, `method` (`chol` | `ldlt`);
  - the SMW rank and its parts h/p/q;
  - `stale_ratio`: the ratio of the preconditioner factorization's μ/ρ to the current ones after the event (1: current);
  - mu and rho, and whether mu, rho or the active sets changed;
  - the time spent;
  - for a PCG failure, its kind, iterations and error.
- **`ssn`**: one row per SSN iteration, including one that ended at the line search without a step
  (`ls_outcome`). Each row has:
  - the active-set sizes and flips;
  - the main solve's setup and reason;
  - Krylov iterations, step size and gradient norm;
  - `krylov_rtol` and `krylov_atol`: the relative tolerance the main PCG solve was given and its absolute target
    (`nan` for direct solves);
  - `newton_res`: ‖J·du + ∇M‖ of the direction after refinement, in the norm of the gradient norm. A full step with
    unchanged active sets leaves this gradient;
  - `on_piece`: 1 if the full Newton step keeps the active sets (refined to `0.1 · ssn_tol`), 0 if not, -1 for
    direct solves;
  - the phases.
- **`pmm`**: one row per PMM iteration. Each row has:
  - the SSN status;
  - mu, rho and the SSN tolerance before and after, with the update rule
    (`x2` / `x0.5` / `x1.1` / `keep` / `stop`);
  - the four residual norms;
  - the factorizations, SMW updates and Krylov iterations in that iteration, and its time.

---

## Project structure

The solver is header-only: everything under `include/` is the library, and `src/` holds only
benchmark drivers.

```
KSP-QP/
├── include/                            # header-only solver library
│   ├── ksp_qp.hpp/.tpp                 # PMM outer loop (main solver)
│   ├── ssn.hpp/.tpp                    # semismooth Newton inner solver
│   ├── schur_operator.hpp              # Schur complement as a matrix-free linear operator
│   ├── schur_preconditioner.hpp        # preconditioner (factorization reuse, SMW low-rank updates)
│   ├── schur_smw_update.hpp            # SMW low-rank updates of the direct solver's factorization
│   ├── solve_profile.hpp               # factorization log and run-time split (experimental)
│   ├── problem.hpp                     # Problem<T>: problem data + solver settings
│   ├── solution.hpp                    # Solution<T> and the TerminationStatus enum
│   ├── ksp_qp_types.hpp                # ParsedModel / KSPQPdata data structures
│   ├── mps_format_parser.hpp/.tpp      # MPS / SIF / QPS file parser
│   ├── printing.hpp                    # PrintWhen / PrintWhat runtime printing
│   ├── record_result.hpp               # TestResult + benchmark CSV writers
│   ├── amd_ordering.hpp                # AMD ordering in 64-bit indices for the sparse factorizations
│   └── cli_args.hpp                    # minimal --flag value parsing for the drivers
├── src/                                # benchmark drivers (no solver code)
│   ├── netlib.cpp                      # Netlib LP driver (--set feasible|infeasible)
│   └── maros_meszaros.cpp              # Maros-Meszaros QP driver
├── tests/                              # GoogleTest suite, one binary per header
│   ├── test_ksp_qp.cpp
│   ├── test_ssn.cpp
│   ├── test_schur_operator.cpp
│   ├── test_schur_preconditioner.cpp
│   ├── test_schur_smw_update.cpp
│   ├── test_mps_format_parser.cpp
│   ├── test_problem.cpp
│   ├── test_solution.cpp
│   ├── test_printing.cpp
│   ├── test_amd_ordering.cpp
│   └── CMakeLists.txt
├── python/
│   ├── ksp_qp_bind.cpp                 # pybind11 bindings
│   ├── fem_q1.py                       # Q1 finite-element reference-element kernels
│   ├── pde_generator.py                # builds PDE-constrained QPs (Q1 FEM / FD)
│   ├── mpc_generator.py                # builds platoon linear-MPC QPs
│   ├── validate_mpc_generator.py       # independent checks on the generated MPC QPs
│   ├── benchmark_common.py             # shared QPALM/OSQP conversion + runner helpers
│   ├── benchmark_mm.py                 # Maros-Meszaros benchmark vs QPALM/OSQP
│   ├── benchmark_netlib.py             # Netlib LP benchmark vs QPALM/OSQP
│   ├── benchmark_infeas.py             # Netlib infeasible-set detection benchmark
│   ├── benchmark_l2pde.py              # L2 PDE-constrained benchmark vs QPALM/OSQP
│   ├── benchmark_mpc.py                # platoon MPC benchmark vs QPALM
│   ├── analyze_profile.py              # per-iteration gap vs QPALM from the MM/MPC result files
│   ├── analyze_performance.py          # performance summary vs QPALM from the MM/MPC result files
│   ├── compare_runs.py                 # two KSP-QP runs compared (e.g. PCG vs direct) from their result files
│   ├── tests/                          # unittest suite for the Python generators
│   └── CMakeLists.txt                  # Python binding build config
├── data/
│   ├── maros_meszaros/                 # Maros-Meszaros QP instances (.SIF)
│   └── netlib-main/
│       ├── feasible/                   # 114 feasible Netlib LPs (.mps, lowercase names)
│       ├── infeasible/                 # 29 primal-infeasible Netlib LPs (.mps)
│       ├── netlib_grbp/                # the feasible set presolved by Gurobi
│       └── feasible_gurobi_1e-8.csv    # reference optimal objectives
├── results/                            # generated CSVs and plots (git-ignored)
└── CMakeLists.txt                      # main build configuration
```

The Kennington family (`CRE-*`, `KEN-*`, `OSA-*`, `PDS-*`) is blended into
`data/netlib-main/feasible/` rather than kept in a separate directory.
