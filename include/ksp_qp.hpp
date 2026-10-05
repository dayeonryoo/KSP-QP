#pragma once
#include <chrono>
#include <functional>
#include <limits>
#include <optional>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "problem.hpp"
#include "solution.hpp"
#include "ssn.hpp"
#include "printing.hpp"

// =============================================================
//      min  c^T x + 0.5 x^T Q x + obj_const,
//      s.t. A x = b,
//           B x = w,
//           lx <= x <= ux,
//           lw <= w <= uw.
// =============================================================
// INPUT: Problem
// --------------------------------------------------------------
// A class containing the data of the problem to be solved:
//    .Q         -> n x n sparse symmetric positive semidefinite quadratic coefficient matrix
//                  (given as a full matrix or a lower triangular matrix)
//    .A         -> m x n sparse linear equality constraint matrix
//    .B         -> l x n sparse linear inequality constraint matrix
//    .b         -> m-dim right-hand side vector for linear equality constraints
//    .c         -> n-dim coefficient vector
//    .lx        -> n-dim lower bound vector for box constraints on x
//    .ux        -> n-dim upper bound vector for box constraints on x
//    .lw        -> l-dim lower bound vector for box constraints on Bx
//    .uw        -> l-dim upper bound vector for box constraints on Bx
//    .obj_const -> constant term in the objective function
//    .tol       -> tolerance for termination
//    .max_it    -> maximum allowed number of PMM iterations
// =============================================================
// OUTPUT: Solution
// --------------------------------------------------------------
// A class containing the solution of the KSP_QP solver:
//    .opt     -> TerminationStatus (see solution.hpp) indicating the termination status
//    .x       -> Optimal primal solution vector
//    .y1      -> Lagrangian multipliers corresponding to Ax = b
//    .y2      -> Lagrangian multipliers corresponding to Bx = w
//    .z       -> Lagrangian multipliers corresponding to box constraints on x
//    .obj_val -> Optimal objective value
//    .pmm_iter    -> number of PMM iterations performed to terminate
//    .ssn_iter    -> number of SSN iterations performed to terminate
//    .krylov_iter -> number of Krylov iterations performed to terminate
//    .fact        -> number of factorizations performed to terminate
//    .smw_count   -> number of SMW low-rank updates used instead of refactorizing (of the preconditioner
//                    in PCG mode, of the direct solver's factorization in direct mode)
//    .pmm_tol_achieved -> final tolerance achieved by PMM
//    .ssn_tol_achieved -> final tolerance achieved by SSN
//    .setup_time       -> wall-clock time in seconds spent in the KSP_QP constructor
//    .solve_time       -> wall-clock time in seconds spent in solve()
//    .run_time         -> setup_time + solve_time
//    .linesearch_fail  -> number of linesearch failures
//    .krylov_fail      -> number of Krylov failures
// --------------------------------------------------------------

template <typename T>
class KSP_QP {
public:
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using ResVec = Eigen::Matrix<T, 4, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    // Inputs:
    SpMat Q, A, B;
    Vec c, b;
    Vec lx, ux, lw, uw;
    T obj_const;

    int n, m, l;
    int N, M;   // extended dimensions for general Q (N = 2n, M = m + n)
    int Q_info; // 0 = zero; 1 = diagonal; 2 = general
    Vec Q_diag;
    SpMat L;    // Cholesky factor of general Q
    SpMat A_tr, B_tr;

    SpMat Q_ruiz, A_ruiz, B_ruiz;
    Vec problem_Q_diag, Q_diag_ruiz, c_ruiz, b_ruiz, lx_ruiz, ux_ruiz, lw_ruiz, uw_ruiz;
    Vec D1A_diag, D1B_diag, D2_diag;
    Vec D1A_ext, D2_ext; // extended to size N, M
    Vec D1A_ext_inv, D2_ext_inv, D1B_diag_inv; // precomputed reciprocals
    Vec c_orig, b_orig, lx_orig, ux_orig, lw_orig, uw_orig; // unscaled problem data for termination/infeasibility check
    Vec x_sol, y1_sol, y2_sol, z_sol;

    // Pre-allocated scratch vectors for the PMM main loop
    Vec Ax_scratch_, Bx_scratch_, Qx_scratch_;      // current iteration products
    Vec Ax_old_scratch_, Bx_old_scratch_;           // previous iteration products
    Vec Adx_scratch_, Bdx_scratch_;                 // differences
    Vec x_old_scratch_, y2_old_scratch_;            // previous iterates for infeasibility check

    // Pre-allocated scratch vectors for compute_residual_unscaled_inf_norms
    Vec A_tr_y1_scratch_, B_tr_y2_scratch_;         // size N
    Vec num_scratch_;                               // size N
    Vec proj_K_unscaled_scratch_;                   // size N
    Vec proj_W_unscaled_scratch_;                   // size l
    Vec Ax_unscaled_scratch_;                       // size M
    Vec z_unscaled_scratch_, num_unscaled_scratch_; // size N
    Vec x_unscaled_scratch_;                        // size N
    Vec Bx_unscaled_scratch_, y2_unscaled_scratch_; // size l

    T inf = std::numeric_limits<T>::infinity();
    T eps_zero = T(100) * std::numeric_limits<T>::epsilon(); // ~2.2e-14 for double

    // Ruiz scaling constants (see ruiz_scaling() in ksp_qp.tpp)
    static constexpr int kMaxRuizIter = 10;
    static constexpr T   kRuizTol     = T(1e-3);

    // set_L_from_LLT regularization constants: on a negative LDLT pivot beyond noise level,
    // the diagonal regularization is escalated by 10x, up to this many attempts.
    static constexpr int kLdltMaxAttempts = 4;
    static constexpr T   kLdltVerifyTol   = T(1e-5);

    // Constant parameters
    T tol = 1e-6;
    int max_iter = 3000;
    int ssn_max_iter = 120000;
    int ssn_max_in_iter = 50;
    T eps_limit = tol;
    T mu_limit = 1e7;
    T rho_limit = 1e7;
    T alpha = 0.95;
    double time_limit = 600.0; // in seconds,  on setup_time + solve time
    bool direct_solve = false; // experimental: skip PCG and solve by LDLT on K or Cholesky on its Schur complement
    bool direct_smw = true;    // with direct_solve: reuse the last factorization through SMW low-rank updates when the active sets change little
    int linesearch_fail = 0;

    // Primal/dual infeasibility certificate tolerances.
    T eps_pinf = 1e-3 * tol;
    T eps_dinf = 1e-3 * tol;

    // Updated parameters
    T mu0 = 1e0;
    T rho0 = 1e0;
    T mu = 1e2;
    T rho = 1e7;
    T ssn_tol = 1e-2;

    // Outputs:
    TerminationStatus opt = TerminationStatus::NumericalError;
    Vec x, y1, y2, z;
    T obj_val;
    int pmm_iter, ssn_iter;
    int krylov_iter = 0, fact = 0, krylov_fail = 0;
    T pmm_tol_achieved, ssn_tol_achieved;
    ResVec res_norms;
    bool   pcg_failed      = false; // mirrors SSN::pcg_failed: PCG failed and the direct solver took over
    int    kkt_ldlt_fact   = 0;     // mirrors SSN::kkt_ldlt_fact: direct-solver LDLT factorizations of K
    int    schur_chol_fact = 0;     // mirrors SSN::schur_chol_fact: direct-solver Cholesky factorizations of S
    int    direct_smw_count  = 0;   // mirrors SSN::direct_smw_count: SMW updates used instead of refactorizing
    int    direct_smw_reject = 0;   // mirrors SSN::direct_smw_reject: SMW-updated solves rejected and refactorized
    double setup_time      = 0.0;   // wall-clock time spent in this constructor, in seconds
    bool   setup_failed    = false; // true if an error occurred during setup

    // Clock used for setup_time/solve_time and the time_limit check in solve().
    std::function<std::chrono::steady_clock::time_point()> now_ = [] { return std::chrono::steady_clock::now(); };

    // Interruption check, polled once per PMM iteration in solve().
    std::function<bool()> interrupted_ = [] { return false; };

    // Printing
    PrintWhen when = PrintWhen::NEVER;
    PrintWhat what = PrintWhat::NONE;

    // Iteration-trace hook; defaults to the class's own print() call.
    std::function<void(const IterationRecord<T>&)> report_ = [this](const IterationRecord<T>& r) {
        print<T, Vec>(when, what, r.pmm_iter, r.ssn_iter, r.krylov_iter, r.fact, r.obj_val, r.res_norms,
                      r.ssn_res, r.mu, r.rho, r.eps, r.linesearch_fail, r.krylov_fail, r.show_pmm_iter);
    };

    // Constructor
    KSP_QP(const Problem<T>& problem)
    : tol(problem.tol), max_iter(problem.max_iter), time_limit(problem.time_limit),
      n(problem.n), m(problem.m), l(problem.l), obj_const(problem.obj_const),
      when(problem.when), what(problem.what)
    {
        auto setup_start = now_();

        try {
            get_Q_info(problem.Q);
            if (n == 0 && m == 0 && l == 0) {
                determine_dimensions(problem);
            }
            check_dimensions(problem);
            Q = problem.Q;
            // problem.Q may already be lower-only or full symmetric; Q only stores the lower-triangular part.
            if (Q.nonZeros() > 0) {
                std::vector<Eigen::Triplet<T>> Q_lower_trip;
                Q_lower_trip.reserve(Q.nonZeros());
                for (int k = 0; k < Q.outerSize(); ++k) {
                    for (typename SpMat::InnerIterator it(Q, k); it; ++it) {
                        if (it.row() >= it.col()) Q_lower_trip.emplace_back(it.row(), it.col(), it.value());
                    }
                }
                Q.resize(Q.rows(), Q.cols());
                Q.setFromTriplets(Q_lower_trip.begin(), Q_lower_trip.end());
                Q.makeCompressed();
            }
            c_orig = (problem.c.size() == 0) ? Vec::Zero(n) : problem.c;
            ruiz_scaling(problem, problem_Q_diag);
            set_default(problem);

            // Clear temporary data to save memory.
            SpMat().swap(Q_ruiz);
            SpMat().swap(A_ruiz);
            SpMat().swap(B_ruiz);
            Vec().swap(problem_Q_diag);
            Vec().swap(Q_diag_ruiz);
            Vec().swap(c_ruiz);
            Vec().swap(b_ruiz);
            Vec().swap(lx_ruiz);
            Vec().swap(ux_ruiz);
            Vec().swap(lw_ruiz);
            Vec().swap(uw_ruiz);

            initialize_sols();
            if (check_bounds()) {
                A_tr = A.transpose();
                B_tr = B.transpose();
            } else {
                setup_failed = true;
                opt = TerminationStatus::PrimalInfeasible;
            }
        } catch (const std::exception& e) {
            std::cerr << "[KSP_QP] Setup error: " << e.what() << "\n";
            setup_failed = true;
            opt = TerminationStatus::NumericalError;
        }

        auto setup_end = now_();
        setup_time = time_diff_s(setup_start, setup_end); // in seconds
    }

    void get_Q_info(const SpMat& Q);
    void determine_dimensions(const Problem<T>& problem);
    void check_dimensions(const Problem<T>& problem);
    void ruiz_scaling(const Problem<T>& problem, const Vec& Q_diag);
    void set_L_from_LLT(const SpMat& Q);
    static void build_reformulated_vecs(int n, int m, int N, int M, T inf,
                                        const Vec& c_in, const Vec& b_in, const Vec& lx_in, const Vec& ux_in,
                                        Vec& c_out, Vec& b_out, Vec& lx_out, Vec& ux_out);
    void set_default(const Problem<T>& problem);
    void initialize_sols();
    bool check_bounds();

    static inline Vec proj(const Vec& u, const Vec& lower, const Vec& upper) {
        return u.cwiseMax(lower).cwiseMin(upper);
    }
    static inline T inf_norm(const Vec& v) {
        return v.cwiseAbs().maxCoeff();
    }
    static T mat_inf_norm(const SpMat& M); // matrix infinity-norm (max abs row sum)
    ResVec compute_residual_unscaled_inf_norms(const Vec& Ax, const Vec& Bx, const Vec& Qx);
    T objective_value(const Vec& x_orig);
    void printable_sol(const Vec& x, const Vec& y1, const Vec& y2, const Vec& z);
    void update_PMM_parameters(const ResVec& res_norms, const ResVec& new_res_norms, typename SSN<T>::TerminationStatus ssn_opt, T ssn_res, int ssn_inner_iters);
    bool primal_infeas(const Vec& cert_y1, const Vec& cert_y2, const Vec& cert_z);
    bool dual_infeas(const Vec& delta_x, const Vec& Adx, const Vec& Bdx);
    void accept_ssn_iterate(const SSN<T>& NS);
    void update_multipliers_if_accurate(typename SSN<T>::TerminationStatus ssn_opt, Vec& delta_y1, Vec& delta_z);
    Solution<T> solve();
};

#include "ksp_qp.tpp"