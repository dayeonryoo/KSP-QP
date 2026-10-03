#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <fstream>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "ksp_qp.hpp"
#include "problem.hpp"
#include "mps_format_parser.hpp"
#include "record_result.hpp"

namespace py = pybind11;
using T      = double;
using Vec    = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat  = Eigen::SparseMatrix<T>;

/*-----------------------------------------------------------------------
Helper: Eigen compressed sparse (CSC) → Python dict of numpy arrays
-----------------------------------------------------------------------*/
static py::dict eigen_sparse_to_dict(const SpMat& M_in, const std::string& key) {
    SpMat M = M_in;
    M.makeCompressed();

    int nnz  = M.nonZeros();
    int rows = (int)M.rows();
    int cols = (int)M.cols();

    py::array_t<double> data(nnz);
    py::array_t<int>    indices(nnz);
    py::array_t<int>    indptr(cols + 1);

    auto d = data.mutable_unchecked<1>();
    auto i = indices.mutable_unchecked<1>();
    auto p = indptr.mutable_unchecked<1>();

    for (int k = 0; k < nnz;    ++k) { d[k] = M.valuePtr()[k];      i[k] = M.innerIndexPtr()[k]; }
    for (int j = 0; j <= cols;  ++j)   p[j] = M.outerIndexPtr()[j];

    py::dict out;
    out[py::str(key + "_data")]    = data;
    out[py::str(key + "_indices")] = indices;
    out[py::str(key + "_indptr")]  = indptr;
    out[py::str(key + "_shape")]   = py::make_tuple(rows, cols);
    return out;
}

/*-----------------------------------------------------------------------
Helper: Eigen vector → 1D numpy array
-----------------------------------------------------------------------*/
static py::array_t<double> eigen_vec_to_array(const Vec& v) {
    py::array_t<double> arr(v.size());
    auto p = arr.mutable_unchecked<1>();
    for (int k = 0; k < (int)v.size(); ++k) p[k] = v[k];
    return arr;
}

/*-----------------------------------------------------------------------
parse_sif: parse a SIF/MPS file and return problem data as numpy arrays.

Returned dict keys:
  n, m, l                      – problem dimensions
  Q_data/indices/indptr/shape  – CSC sparse Q (may be full symmetric)
  A_data/indices/indptr/shape  – CSC sparse equality matrix
  B_data/indices/indptr/shape  – CSC sparse general-inequality matrix
  c, b                         – 1D arrays
  lx, ux, lw, uw               – 1D bound arrays
  obj_const                    – scalar constant in objective
-----------------------------------------------------------------------*/
py::dict parse_sif(const std::string& filename) {
    KSPQPdata<T> pd;
    {
        py::gil_scoped_release release;
        MpsFormatParser<T>   parser;
        ParsedModel<T> model = parser.parse(filename);
        pd = parser.to_kspqp(model);
    }

    py::dict out;
    out["n"] = pd.n;
    out["m"] = pd.m;
    out["l"] = pd.l;

    // Sparse matrices
    for (auto& [M, key] : std::vector<std::pair<const SpMat*, std::string>>{
            {&pd.Q, "Q"}, {&pd.A, "A"}, {&pd.B, "B"}}) {
        py::dict d = eigen_sparse_to_dict(*M, key);
        for (auto item : d) out[item.first] = item.second;
    }

    // Dense vectors
    out["c"]         = eigen_vec_to_array(pd.c);
    out["b"]         = eigen_vec_to_array(pd.b);
    out["lx"]        = eigen_vec_to_array(pd.lx);
    out["ux"]        = eigen_vec_to_array(pd.ux);
    out["lw"]        = eigen_vec_to_array(pd.lw);
    out["uw"]        = eigen_vec_to_array(pd.uw);
    out["obj_const"] = (double)pd.obj_const;

    return out;
}

/*-----------------------------------------------------------------------
solve_from_sif: parse a SIF/MPS file and run the KSP-QP solver.

Returns dict:
  status            – opt field (0 = optimal, <0 = infeasible, >0 = limit hit)
  obj_val           – primal objective value
  setup_time        – wall-clock time in seconds spent in the KSP_QP constructor
  solve_time        – wall-clock time in seconds spent in solve()
  run_time          – setup_time + solve_time
  pmm_iter          – PMM outer iterations
  ssn_iter          – total SSN inner iterations
  krylov_iter       – total Krylov iterations
  fact              – total number of factorizations
  smw_count         – total number of SMW low-rank updates used instead of refactorizing
                      (of the preconditioner, or of the direct solver's factorization in direct mode)
  pmm_tol_achieved  – tolerance achieved by PMM at termination
  system            – "S" (PCG only), or "D" + "K"/"S" for the systems the direct
                      solver factorized (direct mode, or after PCG failed)
  kkt_ldlt_fact     – direct-solver LDLT factorizations of the KKT system
  schur_chol_fact   – direct-solver Cholesky factorizations of the Schur complement
  direct_smw_count  – direct-mode Newton systems solved through an SMW update (included in smw_count)
  direct_smw_reject – of those, updates rejected by the residual check and refactorized
  x                 – primal solution vector (original, unscaled units)
  y1, y2, z         – multipliers for Ax = b, Bx = w and the box constraints on x,
                      in the same original, unscaled units as x
-----------------------------------------------------------------------*/
py::dict solve_from_sif(const std::string& filename,
                        double tol         = 1e-6,
                        long long max_iter = 1'000'000'000LL,
                        double time_limit  = 600.0,
                        bool direct        = false,
                        bool direct_smw    = true) {
    int opt, pmm_iter, ssn_iter, krylov_iter, fact, smw_count, kkt_ldlt_fact, schur_chol_fact;
    int direct_smw_count, direct_smw_reject;
    double obj_val, setup_time, solve_time, run_time, pmm_tol_achieved;
    std::string system;
    Vec x_sol, y1_sol, y2_sol, z_sol;
    {
        py::gil_scoped_release release;
        MpsFormatParser<T>   parser;
        ParsedModel<T> model  = parser.parse(filename);
        KSPQPdata<T>   pd     = parser.to_kspqp(model);

        Problem<T>  prob(pd, (T)tol, (int)max_iter, time_limit,
                         PrintWhen::NEVER, PrintWhat::NONE);
        KSP_QP<T>  solver(prob);
        solver.direct_solve = direct;
        solver.direct_smw   = direct_smw;
        Solution<T> sol = solver.solve();
        system           = system_label(solver.direct_solve || solver.pcg_failed,
                                        solver.kkt_ldlt_fact, solver.schur_chol_fact);
        kkt_ldlt_fact    = solver.kkt_ldlt_fact;
        schur_chol_fact  = solver.schur_chol_fact;
        direct_smw_count  = solver.direct_smw_count;
        direct_smw_reject = solver.direct_smw_reject;
        opt              = static_cast<int>(sol.opt);
        obj_val          = (double)sol.obj_val;
        setup_time       = sol.setup_time;
        solve_time       = sol.solve_time;
        run_time         = sol.run_time;
        pmm_iter         = sol.pmm_iter;
        ssn_iter         = sol.ssn_iter;
        krylov_iter      = sol.krylov_iter;
        fact             = sol.fact;
        smw_count        = sol.smw_count;
        pmm_tol_achieved = (double)sol.pmm_tol_achieved;
        x_sol            = sol.x;
        y1_sol           = sol.y1;
        y2_sol           = sol.y2;
        z_sol            = sol.z;
    }

    py::dict out;
    out["status"]           = opt;
    out["obj_val"]          = obj_val;
    out["setup_time"]       = setup_time;
    out["solve_time"]       = solve_time;
    out["run_time"]         = run_time;
    out["pmm_iter"]         = pmm_iter;
    out["ssn_iter"]         = ssn_iter;
    out["krylov_iter"]      = krylov_iter;
    out["fact"]             = fact;
    out["smw_count"]        = smw_count;
    out["pmm_tol_achieved"] = pmm_tol_achieved;
    out["system"]           = system;
    out["kkt_ldlt_fact"]    = kkt_ldlt_fact;
    out["schur_chol_fact"]  = schur_chol_fact;
    out["direct_smw_count"]  = direct_smw_count;
    out["direct_smw_reject"] = direct_smw_reject;
    out["x"]                = eigen_vec_to_array(x_sol);
    out["y1"]               = eigen_vec_to_array(y1_sol);
    out["y2"]               = eigen_vec_to_array(y2_sol);
    out["z"]                = eigen_vec_to_array(z_sol);
    return out;
}

/*-----------------------------------------------------------------------
Helper: reconstruct KSPQPdata<T> from a parse_sif dict.
-----------------------------------------------------------------------*/
static KSPQPdata<T> dict_to_kspqp(const py::dict& d) {
    KSPQPdata<T> pd;
    pd.n         = d["n"].cast<int>();
    pd.m         = d["m"].cast<int>();
    pd.l         = d["l"].cast<int>();
    pd.obj_const = d["obj_const"].cast<double>();

    auto make_sparse = [&](const std::string& key) {
        auto shape_tup   = d[py::str(key + "_shape")].cast<py::tuple>();
        int rows = shape_tup[0].cast<int>(), cols = shape_tup[1].cast<int>();
        auto data_arr    = d[py::str(key + "_data")].cast<py::array_t<double>>();
        auto indices_arr = d[py::str(key + "_indices")].cast<py::array_t<int>>();
        auto indptr_arr  = d[py::str(key + "_indptr")].cast<py::array_t<int>>();
        auto dr = data_arr.unchecked<1>();
        auto ir = indices_arr.unchecked<1>();
        auto pr = indptr_arr.unchecked<1>();
        int nnz = (int)data_arr.size();
        std::vector<Eigen::Triplet<T>> trips;
        trips.reserve(nnz);
        for (int col = 0; col < cols; ++col)
            for (int k = pr[col]; k < pr[col + 1]; ++k)
                trips.emplace_back(ir[k], col, (T)dr[k]);
        SpMat M(rows, cols);
        M.setFromTriplets(trips.begin(), trips.end());
        M.makeCompressed();
        return M;
    };

    auto make_vec = [&](const std::string& key) {
        auto arr = d[py::str(key)].cast<py::array_t<double>>();
        auto r = arr.unchecked<1>();
        Vec v(r.size());
        for (int i = 0; i < (int)r.size(); ++i) v[i] = (T)r[i];
        return v;
    };

    pd.Q  = make_sparse("Q");
    pd.A  = make_sparse("A");
    pd.B  = make_sparse("B");
    pd.c  = make_vec("c");
    pd.b  = make_vec("b");
    pd.lx = make_vec("lx");
    pd.ux = make_vec("ux");
    pd.lw = make_vec("lw");
    pd.uw = make_vec("uw");
    return pd;
}

/*-----------------------------------------------------------------------
solve_from_data: run the KSP-QP solver on already-parsed problem data.
Takes a dict as returned by parse_sif.
Returns dict with the same keys as solve_from_sif.
-----------------------------------------------------------------------*/
py::dict solve_from_data(const py::dict& pd_dict,
                         double tol         = 1e-6,
                         long long max_iter = 1'000'000'000LL,
                         double time_limit  = 600.0,
                         std::string trace_path = "",
                         double rho_init = -1.0,
                         bool direct = false,
                         bool direct_smw = true) {
    KSPQPdata<T> pd = dict_to_kspqp(pd_dict); // reads Python objects

    int opt, pmm_iter, ssn_iter, krylov_iter, fact, smw_count, kkt_ldlt_fact, schur_chol_fact;
    int direct_smw_count, direct_smw_reject;
    double obj_val, setup_time, solve_time, run_time, pmm_tol_achieved;
    std::string system;
    Vec x_sol, y1_sol, y2_sol, z_sol;
    {
        py::gil_scoped_release release;
        // trace_path is diagnostic-only: when set, writes a per-PMM-iteration and
        // per-SSN-inner-iteration CSV trace (active-set sizes, ssn_opt, mu/rho)
        // to that file. Default "" preserves prior silent behavior exactly.
        bool trace = !trace_path.empty();
        Problem<T>  prob(pd, (T)tol, (int)max_iter, time_limit,
                         trace ? PrintWhen::ALWAYS : PrintWhen::NEVER,
                         trace ? PrintWhat::SSN    : PrintWhat::NONE);
        KSP_QP<T>  solver(prob);
        // Diagnostic-only: override rho's initial value (default rho_limit, i.e. pinned at
        // its ceiling from PMM iteration 0). rho does not appear in the outer termination
        // check (compute_residual_unscaled_inf_norms/primal_infeas/dual_infeas) or in
        // objective_value(), only in update_PMM_parameters()'s own schedule and inside SSN's
        // H_diag/gradient -- so this cannot corrupt what "converged" means, only the search
        // dynamics used to get there. Sentinel -1 (default) leaves rho at its usual rho_limit
        // start, matching prior behavior exactly.
        if (rho_init > 0.0) solver.rho = (T)rho_init;
        solver.direct_solve = direct;
        solver.direct_smw   = direct_smw;
        std::ofstream trace_file;
        if (trace) {
            trace_file.open(trace_path);
            trace_file << "pmm_iter,ssn_iter,n_active_W,n_active_K,ssn_opt,mu,rho,ssn_res\n";
            solver.report_ = [&trace_file](const IterationRecord<T>& r) {
                trace_file << r.pmm_iter << "," << r.ssn_iter << ","
                          << r.n_active_W << "," << r.n_active_K << ","
                          << r.ssn_opt << "," << r.mu << "," << r.rho << "," << r.ssn_res << "\n";
            };
        }
        Solution<T> sol = solver.solve();
        if (trace) trace_file.close();
        system           = system_label(solver.direct_solve || solver.pcg_failed,
                                        solver.kkt_ldlt_fact, solver.schur_chol_fact);
        kkt_ldlt_fact    = solver.kkt_ldlt_fact;
        schur_chol_fact  = solver.schur_chol_fact;
        direct_smw_count  = solver.direct_smw_count;
        direct_smw_reject = solver.direct_smw_reject;
        opt              = static_cast<int>(sol.opt);
        obj_val          = (double)sol.obj_val;
        setup_time       = sol.setup_time;
        solve_time       = sol.solve_time;
        run_time         = sol.run_time;
        pmm_iter         = sol.pmm_iter;
        ssn_iter         = sol.ssn_iter;
        krylov_iter      = sol.krylov_iter;
        fact             = sol.fact;
        smw_count        = sol.smw_count;
        pmm_tol_achieved = (double)sol.pmm_tol_achieved;
        x_sol            = sol.x;
        y1_sol           = sol.y1;
        y2_sol           = sol.y2;
        z_sol            = sol.z;
    }

    py::dict out;
    out["status"]           = opt;
    out["obj_val"]          = obj_val;
    out["setup_time"]       = setup_time;
    out["solve_time"]       = solve_time;
    out["run_time"]         = run_time;
    out["pmm_iter"]         = pmm_iter;
    out["ssn_iter"]         = ssn_iter;
    out["krylov_iter"]      = krylov_iter;
    out["fact"]             = fact;
    out["smw_count"]        = smw_count;
    out["pmm_tol_achieved"] = pmm_tol_achieved;
    out["system"]           = system;
    out["kkt_ldlt_fact"]    = kkt_ldlt_fact;
    out["schur_chol_fact"]  = schur_chol_fact;
    out["direct_smw_count"]  = direct_smw_count;
    out["direct_smw_reject"] = direct_smw_reject;
    out["x"]                = eigen_vec_to_array(x_sol);
    out["y1"]               = eigen_vec_to_array(y1_sol);
    out["y2"]               = eigen_vec_to_array(y2_sol);
    out["z"]                = eigen_vec_to_array(z_sol);
    return out;
}

/*-----------------------------------------------------------------------
Module
-----------------------------------------------------------------------*/
PYBIND11_MODULE(ksp_qp_bind, m) {
    m.doc() = "Python bindings for the KSP-QP quadratic programming solver";

    m.def("parse_sif", &parse_sif,
          py::arg("filename"),
          R"(Parse a SIF/MPS file and return problem data as numpy arrays.

Returns a dict with keys: n, m, l, Q_*, A_*, B_*, c, b, lx, ux, lw, uw, obj_const.
The sparse matrices are in CSC format (data / indices / indptr / shape).)");

    m.def("solve_from_sif", &solve_from_sif,
          py::arg("filename"),
          py::arg("tol")        = 1e-6,
          py::arg("max_iter")   = 1'000'000'000LL,
          py::arg("time_limit") = 600.0,
          py::arg("direct")     = false,
          py::arg("direct_smw") = true,
          R"(Parse a SIF/MPS file and solve it with the KSP-QP solver.

Returns a dict with keys: status, obj_val, setup_time, solve_time, run_time, pmm_iter, ssn_iter,
krylov_iter, fact, smw_count, pmm_tol_achieved, system, kkt_ldlt_fact, schur_chol_fact,
direct_smw_count, direct_smw_reject, x, y1, y2, z (x and the multipliers y1/y2/z are returned in the
original, unscaled units, so they can be checked against the problem data as given).
status == 0  → optimal solution found
status <  0  → infeasibility detected
status >  0  → iteration / time limit reached

direct: experimental, default False. When True, skips PCG and solves the KKT system (LDLT) or
its Schur complement (Cholesky) exactly. system is "S" when only PCG ran, otherwise "D" followed
by "K" and/or "S" for the systems the direct solver factorized (direct mode, or after PCG failed);
kkt_ldlt_fact/schur_chol_fact count those factorizations.

direct_smw: default True; only used with direct=True. When True, an SSN iteration whose active
sets differ little from the last factorization's is solved through a Sherman-Morrison-Woodbury
update of that factorization instead of refactorizing; direct_smw_count counts those, and
direct_smw_reject the ones whose residual check failed and were refactorized. When False, every
change is refactorized.)");

    m.def("solve_from_data", &solve_from_data,
          py::arg("pd"),
          py::arg("tol")        = 1e-6,
          py::arg("max_iter")   = 1'000'000'000LL,
          py::arg("time_limit") = 600.0,
          py::arg("trace_path") = "",
          py::arg("rho_init") = -1.0,
          py::arg("direct") = false,
          py::arg("direct_smw") = true,
          R"(Solve with KSP-QP using already-parsed problem data (dict from parse_sif).

Returns a dict with keys: status, obj_val, setup_time, solve_time, run_time, pmm_iter, ssn_iter,
krylov_iter, fact, smw_count, pmm_tol_achieved, system, kkt_ldlt_fact, schur_chol_fact,
direct_smw_count, direct_smw_reject, x, y1, y2, z (x and the multipliers y1/y2/z are returned in the
original, unscaled units, so they can be checked against the problem data as given).

direct, direct_smw: experimental; same meaning as in solve_from_sif.

trace_path: diagnostic-only, default "" (no tracing, matches prior behavior exactly). When
set, writes a per-PMM-iteration and per-SSN-inner-iteration CSV trace to that path with columns
pmm_iter,ssn_iter,n_active_W,n_active_K,ssn_opt,mu,rho,ssn_res -- n_active_K is -1 on PMM-level
rows (not applicable there) and ssn_opt is -1 on SSN-inner-loop rows (only set once a full
solve_ssn() call returns); ssn_opt is the underlying
SSN<T>::TerminationStatus enum value (0=Optimal, 1=MaxInnerIterations, 2=LineSearchFailed,
3=Stagnated, 4=Interrupted, 5=TimeLimit). Not meant for routine use -- it turns on
PrintWhat::SSN-level per-inner-iteration reporting, which has non-trivial overhead on
problems with many SSN iterations.

rho_init: diagnostic-only override of rho's starting value (default: rho_limit, i.e. pinned
at its ceiling from PMM iteration 0). Does not appear in any outer termination check or in
objective_value(), only in update_PMM_parameters()'s own schedule and SSN's H_diag/gradient,
so it cannot corrupt what "converged" means -- only the search dynamics. Sentinel -1
(default) leaves rho at its usual start, an exact no-op.)");
}
