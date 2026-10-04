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
- `pip install numpy scipy matplotlib pandas`
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
| `what` | `PrintWhat` | `NONE` / `MINIMAL` / `SSN` / `TUNING` / `FULL` (see ["Tuning: printing and timers"](#tuning-printing-and-timers)) |

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
| `fact` | `int` | Factorizations performed (of the preconditioner; of K or S by the direct solver after a PCG failure) |
| `smw_count` | `int` | Sherman-Morrison-Woodbury low-rank updates of the preconditioner used instead of refactorizing it |
| `pmm_tol_achieved` | `T` | Final PMM (outer) residual |
| `ssn_tol_achieved` | `T` | Final SSN (inner) residual |
| `setup_time` | `double` | Wall-clock seconds in the `KSP_QP<T>` constructor |
| `solve_time` | `double` | Wall-clock seconds in `solve()` |
| `run_time` | `double` | `setup_time + solve_time` |
| `linesearch_fail` | `int` | Line-search failures |
| `krylov_fail` | `int` | Krylov failures (fell back to a direct factorization) |

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

**`solve_from_sif(filename, tol=1e-6, max_iter=1_000_000_000, time_limit=600.0)`**
Parse and solve a SIF/MPS file. Returns a dict with keys:
`status`, `obj_val`, `setup_time`, `solve_time`, `run_time`, `pmm_iter`, `ssn_iter`,
`krylov_iter`, `fact`, `smw_count`, `pmm_tol_achieved`, `system`, `kkt_ldlt_fact`,
`schur_chol_fact`, `x`, `y1`, `y2`, `z`.
`x` and the multipliers are returned in the original, unscaled units, so they can be
checked directly against the problem data as given. `system` is `"S"` when only PCG ran;
if PCG failed and the direct solver took over, it is `"D"` followed by `"K"` and/or `"S"` for
the systems it factorized (LDLT on the KKT system, Cholesky on its Schur complement), which
`kkt_ldlt_fact` / `schur_chol_fact` count.

**`solve_from_data(pd, tol=1e-6, max_iter=1_000_000_000, time_limit=600.0, trace_path="", rho_init=-1.0)`**
Same, but takes already-parsed problem data — the dict returned by `parse_sif()`, or the one
built by `pde_generator.py` / `mpc_generator.py`. Returns the same keys as `solve_from_sif`.
`trace_path` is diagnostic-only: when set, it writes a per-iteration CSV trace (active-set
sizes, `mu`, `rho`, residuals) and turns on per-inner-iteration reporting, which has
non-trivial overhead — leave it empty for timing runs.

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
                      [--tol T] [--max-iter N] [--time-limit S] [--out FILE] [--cooldown S] [--ref FILE]
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
(override with `--out`). The sweep is the directory listing, so adding or removing an `.mps`
file is all it takes to change the set — the same convention `python/benchmark_netlib.py` uses.

Both sets write the same CSV schema (`include/record_result.hpp`). Two columns are read
differently per set: on the infeasible set `agree` means "infeasibility was detected" (rather
than "objective matched"), `abs_err`/`rel_err` are `-1` since there is no objective to compare,
and `diverged` is always `0` because a large residual is the expected outcome there.

Reference objectives for the feasible set are not hardcoded — they ship with the dataset in
`data/netlib-main/feasible_gurobi_1e-8.csv` (Gurobi 10 at 1e-8; override with `--ref`). Those
disagree with the long-standing published Netlib optima on a handful of problems (E226 and
CRE-A among them), so treat either source with care.

For a QPALM/OSQP comparison with performance profiles, use `python/benchmark_netlib.py` and
`python/benchmark_infeas.py` instead (see below).

### `ksp_qp_maros_meszaros` — Maros-Meszaros QPs (`src/maros_meszaros.cpp`)

```bash
./build/ksp_qp_maros_meszaros [--root DIR] [--in DIR] [--name PROBLEM|all] [--tol T] [--max-iter N] [--time-limit S] [--out FILE] [--cooldown S]
```

Solves `<root>/<PROBLEM>.SIF` (default: `data/maros_meszaros/AUG2DCQP.SIF`), printing the
solution summary. Pass `--name all` to sweep the full Maros-Meszaros set against its built-in
reference objectives, appending a row to `<root>/results/pcg_mm.csv` (override with
`--out`) for each — `--cooldown` (default 3s) sleeps between problems in this mode, which keeps
a long sweep from being distorted by CPU thermal throttling.

For comparing against QPALM/OSQP rather than just checking against reference objectives, use
the Python benchmark instead (see below) — that's what produces performance profiles.

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
matplotlib pandas`.

### Maros-Meszaros QP benchmark

```bash
cd python
python3 benchmark_mm.py
```

Runs the full Maros-Meszaros test set. Writes `results/comparison_mm.csv` plus Dolan-Moré
performance profiles (`results/performance_profile_mm*.pdf/.png`, by time and by iteration count).

```
--root DIR             override project root (default: parent of script)
--tol 1e-6             primal-dual tolerance
--time-limit 60        per-problem time limit in seconds
--solver {ksp-qp,qpalm,osqp} [...]   which solvers to run (default: all three)
--out PREFIX           output file prefix (default: comparison_mm)
--cooldown 0           seconds to sleep between problems (avoids CPU throttling)
```

### Netlib LP benchmark

```bash
cd python
python3 benchmark_netlib.py
```

Runs the full Netlib LP test set (`data/netlib-main/feasible/`, 114 instances). Writes
`results/comparison_netlib.csv` plus Dolan-Moré performance profiles
(`results/performance_profile_netlib*.pdf/.png`). Same flags as `benchmark_mm.py`.

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
python3 benchmark_mpc.py --out 0926          # -> results/0926_mpc_sweep.csv
```

Benchmarks KSP-QP against QPALM on the platoon linear-MPC QP built by `mpc_generator.py` — a
native smooth QP (quadratic tracking cost, linear dynamics equalities, box bounds, no general
inequality rows) whose constraint matrix is banded with at most 6 nonzeros per row.

The problem has **two** size axes and they are not interchangeable, so the sweep is a grid
rather than a curve: `M` (vehicle count) grows the band *width* and the dense `2M × 2M` DARE
terminal cost block, while `N` (horizon) grows the band *length* at fixed bandwidth. Sweeping
one at a time through the corner of the `(M, N)` plane is misleading. Output is long format —
one row per (instance × solver × repeat) — in `results/<prefix>_mpc_sweep.csv`.

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

## Tuning: printing and timers

The solver has two independent knobs for diagnosing/tuning performance: **runtime printing**
(what gets printed to stdout while solving, controlled by `Problem<T>`'s `when`/`what` fields)
and a **compile-time step timer** (per-phase wall-clock breakdown of the SSN inner loop,
printed to stderr).

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

### Compile-time step timers — `SSN_ENABLE_TIMERS` (`include/ssn.hpp`)

A separate, more granular timer instruments the phases inside each SSN iteration (system prep,
linear solve, preconditioner assembly/analyze/factorize, Krylov solve, LDLT fallback,
line search, state update). It is gated by a macro so it compiles to zero overhead when off:

```cpp
// include/ssn.hpp
#ifndef SSN_ENABLE_TIMERS
#define SSN_ENABLE_TIMERS 0   // set to 1 to enable
#endif
```

Enable it either by editing that line directly, or by passing the define at configure time
without touching the source:

```bash
cmake -B build -DCMAKE_CXX_FLAGS="-DSSN_ENABLE_TIMERS=1"
cmake --build build --config Release
```

When enabled, every SSN iteration prints a line like this to **stderr** (independent of the
`PrintWhen`/`PrintWhat` settings above):

```
[Timer] ssn_iter=3 total=0.1234s | prep=0.0012 linear_solve=0.1180 (prec_setup=0.0500 [assembly=0.0100 analyze=0.0150 factorize=0.0250] krylov_solve=0.0680) linesearch=0.0030 state_update=0.0012
```

If the Krylov solve has fallen back to the direct solver, further lines report its breakdown:
`kkt_ldlt` (LDLT on the KKT system: analyze/factorize/solve) and `schur_chol` (Cholesky on the
Schur complement: assembly/analyze/factorize/solve). This is the tool to use when profiling
*where* time goes inside the solver (e.g. preconditioner factorization vs. CG iterations);
`PrintWhat::TUNING` is the tool for watching *convergence behavior* (residuals, PMM parameters)
across iterations.

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
