#pragma once
#include <string>
#include <limits>
#include <functional>
#include <memory>
#include <optional>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/IterativeSolvers>
#include "amd_ordering.hpp"
#include "printing.hpp"
#include "schur_operator.hpp"
#include "schur_preconditioner.hpp"

// TIMER: master switch for per-step SSN-loop timer.
// Set to 1 (here, or via -DSSN_ENABLE_TIMERS=1) to print a step-by-step timer of solve_ssn();
// 0 compiles the timers out entirely (no overhead).
#ifndef SSN_ENABLE_TIMERS
#define SSN_ENABLE_TIMERS 0
#endif

#if SSN_ENABLE_TIMERS
#include <chrono>
#include <cstdio>

// TIMER: RAII scoped timer; accumulates elapsed wall-clock seconds into `acc` on scope exit.
struct SsnScopedTimer {
    std::chrono::steady_clock::time_point t0;
    double& acc;
    explicit SsnScopedTimer(double& acc_) : t0(std::chrono::steady_clock::now()), acc(acc_) {}
    ~SsnScopedTimer() { acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};
#define SSN_TIMER_BLOCK(acc) SsnScopedTimer _ssn_scoped_timer(acc)
#else
#define SSN_TIMER_BLOCK(acc) do {} while (0)
#endif


// ----- exact_line_search: testable function, independent of SSN class -----
template <typename T> 
struct SsnBreakpoint { T t; T slope_change; }; 

template <typename T>
struct SsnLineSearchParams {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    T mu, rho, alpha, eps_zero, eps_direction, inf;
    int Q_info, N, l;
    const Vec& lx; const Vec& ux; const Vec& lw; const Vec& uw;
    const Vec& z;         // PMM box multiplier
    const Vec& y2;        // PMM outer iterate (distinct from exact_line_search's y2_curr argument)
    const Vec& x;         // PMM outer iterate (distinct from exact_line_search's x_curr argument)
    const Vec& c;
    const Vec& A_tr_y1;
    const Vec& Q_diag;
    const Vec& grad_Atr_resp; // A_tr * (Ax_ssn_ - b), cached once per SSN iteration in
                               // make_line_search_params() -- invariant across the initial
                               // exact_line_search() call and its steepest-descent retry, both
                               // of which run against the same (unmoved) Ax_ssn_.
    const Vec& b;
};

template <typename T>
T exact_line_search(const SsnLineSearchParams<T>& p,
                     const Eigen::Matrix<T, Eigen::Dynamic, 1>& x_curr,
                     const Eigen::Matrix<T, Eigen::Dynamic, 1>& y2_curr,
                     const Eigen::Matrix<T, Eigen::Dynamic, 1>& dx,
                     const Eigen::Matrix<T, Eigen::Dynamic, 1>& dy2,
                     const Eigen::Matrix<T, Eigen::Dynamic, 1>& Ax_curr,
                     const Eigen::Matrix<T, Eigen::Dynamic, 1>& Bx_curr,
                     const Eigen::Matrix<T, Eigen::Dynamic, 1>& Adx,
                     const Eigen::Matrix<T, Eigen::Dynamic, 1>& Bdx,
                     const Eigen::Matrix<T, Eigen::Dynamic, 1>& dist_K_s,
                     const Eigen::Matrix<T, Eigen::Dynamic, 1>& dist_W_v,
                     Eigen::Matrix<T, Eigen::Dynamic, 1>& ls_s_scratch,
                     Eigen::Matrix<T, Eigen::Dynamic, 1>& ls_v_scratch,
                     Eigen::Matrix<T, Eigen::Dynamic, 1>& ls_dv_scratch,
                     std::vector<SsnBreakpoint<T>>& breakpoints_scratch);
// -------------------------------------------------------------------------------

template <typename T>
class SSN {
public:
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;
    using RowMajorSpMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;
    using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;
    using Triplet = Eigen::Triplet<T>;

    using Breakpoint = SsnBreakpoint<T>;

    // Inner (SSN-level) termination status; distinct from the outer TerminationStatus in solution.hpp.
    enum class TerminationStatus : int {
        Optimal            = 0, // genuine optimal achieved, or weaker optimal achieved by line search or stagnation check
        MaxInnerIterations = 1, // maximum number of SSN inner iterations reached without convergence
        LineSearchFailed   = 2, // line search failed even after the steepest-descent fallback
        Stagnated          = 3, // ||grad M|| stagnated for 10 iterations without a confirmed optimum
        Interrupted        = 4, // interrupted_() returned true before ssn_max_in_iter was reached
        TimeLimit          = 5, // time_limit_exceeded_() returned true before ssn_max_in_iter was reached
    };

    // Inputs
    const int Q_info;
    const Vec& Q_diag;
    const SpMat& L, A, B;
    const Vec& D2_ext_inv, D1B_diag_inv; // for unscaling the SSN termination criterion
    const Vec& c, b, lx, ux, lw, uw;
    const int n, m, N, M, l;
    Vec x, y1, y2, z;
    Vec delta_x, delta_y1, delta_y2, delta_z;
    int ssn_max_in_iter;
    T mu, rho, alpha, ssn_tol;
    T eps_pinf, eps_dinf;

    // Useful vectors and matrices
    T inf = std::numeric_limits<T>::infinity();
    T eps_zero      = T(100)  * std::numeric_limits<T>::epsilon();  // relative tolerance for boundary/slope checks
    T eps_direction = std::sqrt(std::numeric_limits<T>::epsilon()); // threshold for skipping near-zero Newton step components
    Vec ones_N, ones_M, ones_l;

    const SpMat& A_tr, B_tr;
    Vec H_diag, H_diag_inv;
    T H_diag_mu_ = T(0), H_diag_rho_ = T(0); // (mu, rho) H_diag was last built with; see prepare_newton_system().

    BoolArr active_W, active_K;
    int n_active_W, n_inactive_W;
    SpMat B_inactive_W, G, G_tr;

    RowMajorSpMat B_rm;              // Row-major B for rebuilding G.
    std::vector<Triplet> G_A_trips_; // A's contribution to G, computed once since A is const.

    // Scratch triplet buffers for rebuild_G() and solve_using_ldlt()
    std::vector<Triplet> B_inact_trips_, G_trips_;
    std::vector<Triplet> ldlt_trip_;

    // Printing
    PrintWhen when = PrintWhen::NEVER;
    PrintWhat what = PrintWhat::NONE;

    // Iteration-trace hook
    std::function<void(const IterationRecord<T>&)> report_ = [this](const IterationRecord<T>& r) {
        print<T, Vec>(when, what, r.pmm_iter, r.ssn_iter, r.krylov_iter, r.fact, r.obj_val, r.res_norms,
                      r.ssn_res, r.mu, r.rho, r.eps, r.linesearch_fail, r.krylov_fail, r.show_pmm_iter);
    };

    // Interruption and time-limit checks, both polled once per SSN inner iteration in solve_ssn().
    std::function<bool()> interrupted_ = [] { return false; };
    std::function<bool()> time_limit_exceeded_ = [] { return false; };

    // Outputs
    TerminationStatus opt;
    int iter, ssn_iter;
    T tol_achieved, obj_val;
    int krylov_iter = 0, fact = 0, smw_count = 0;
    int kkt_ldlt_fact = 0, schur_chol_fact = 0; // direct-solver factorizations of K (LDLT) and S (Cholesky); included in fact
    int linesearch_fail = 0, krylov_fail = 0;
    bool krylov_converged = true;

#if SSN_ENABLE_TIMERS
    // TIMER: wall-clock seconds per SSN-loop phase for the current SSN iteration in solve_ssn();
    // reset at the top of each iteration and printed to stderr at the end of that same iteration.
    double timer_prep           = 0.0; // preparing SSN linear system
    double timer_linear_solve   = 0.0; // solve_using_cg() + iterative refinement + recovering dy2
    double timer_prec_setup     = 0.0; // subset of timer_linear_solve: preconditioner setData()/compute()
    double timer_prec_assembly  = 0.0; // subset of timer_prec_setup: SchurPreconditioner sparse matrix assembly (P_hat / G*E*G_tr)
    double timer_prec_analyze   = 0.0; // subset of timer_prec_setup: SchurPreconditioner::analyzePattern()
    double timer_prec_factorize = 0.0; // subset of timer_prec_setup: SchurPreconditioner::factorize()
    double timer_krylov_solve   = 0.0; // subset of timer_linear_solve: cg.solve()/solveWithGuess()
    double timer_ldlt_analyze   = 0.0; // subset of timer_linear_solve: solve_using_ldlt()'s ldlt_.analyzePattern()
    double timer_ldlt_factorize = 0.0; // subset of timer_linear_solve: solve_using_ldlt()'s ldlt_.factorize()
    double timer_ldlt_solve     = 0.0; // subset of timer_linear_solve: solve_using_ldlt()'s ldlt_.solve()
    double timer_chol_assembly  = 0.0; // subset of timer_linear_solve: solve_using_chol()'s assembly of S
    double timer_chol_analyze   = 0.0; // subset of timer_linear_solve: solve_using_chol()'s chol_->analyzePattern()
    double timer_chol_factorize = 0.0; // subset of timer_linear_solve: solve_using_chol()'s chol_->factorize()
    double timer_chol_solve     = 0.0; // subset of timer_linear_solve: solve_using_chol()'s chol_->solve()
    double timer_linesearch     = 0.0; // exact_linesearch
    double timer_state_update   = 0.0; // x, y2 update + termination check
#endif

    // Kylov (conjugate gradient) parameters
    T krylov_tol = 1e-12; // Eigen's convention error is ||rhs - S*dy||_2 / ||rhs||_2.
    int krylov_max_in_iter = 100;

    // Iterative refinement parameters
    int refine_max_iter = 3;
    T refine_rel_tol = 1e-10;
    T refine_abs_tol = 1e-12;

    // Preconditioner factorization method
    bool schur_use_ldlt = false;        // choose_schur_ldlt()'s return; default=false means use Cholesky to factorize a preconditioner.
    int schur_ldlt_decisions_made_ = 0; // choose_schur_ldlt() call count; locked after the first 3.
    static constexpr double kSchurLdltRatioThreshold = 0.1; // schur_use_ldlt=true if the estimated work ratio is below this cutoff.

    using CGSolver = Eigen::ConjugateGradient<
        SchurOperator<T>,
        Eigen::Lower | Eigen::Upper,
        SchurPreconditioner<T>
    >;
    CGSolver cg;

    Vec prev_dy_;
    Vec prev_dx_primal_;
    Vec A_tr_y1_; // cached A^T y1, recomputed once per PMM iteration in update_ssn_system

    // Pre-allocated scratch vectors.
    Vec Ax_ssn_, Bx_ssn_;                     // size M, l (running SpMV products in SSN)
    Vec x_cur_, y2_cur_;                      // size N, l (SSN working iterates)
    Vec u_, v_;                               // size N, l
    BoolArr new_active_K_;                    // size N
    Vec dist_K_u_;                            // size N
    BoolArr new_active_W_;                    // size l
    Vec dist_W_v_;                            // size l
    Vec y2_active_W_, y2_inactive_W_;         // size l (set to max size)
    Vec dist_W_v_active_, dist_W_v_inactive_; // size l (set to max size)
    Vec dy2_inactive_W_;                      // size l (set to max size)
    Vec r1_, r2_;                             // size N, M+n_active_W
    Vec dxdy_;                                // size N+M+n_active_W
    Vec dx_;                                  // size N (Newton direction for x, split from dxdy_)
    Vec dy2_;                                 // size l
    Vec Adx_, Bdx_;                           // size M, l
    Vec grad_L_;                              // size N+l
    Vec grad_dist_K_, grad_res_p_;            // size N, M
    Vec grad_dist_W_;                         // size l
    Vec grad_Atr_resp_, grad_Btr_distW_, grad_A_tr_y_, grad_Qx_; // size N
    Vec ls_s_, ls_dv_;                        // size N, l
    Vec ls_v_;                                // size l
    std::vector<Breakpoint> breakpoints_;     // reused across exact_line_search calls
    Vec cg_Hinv_r1_, cg_rhs_;                 // size s = G.rows() (M+n_active_W)
    Vec cg_dx_;                               // size n = N
    Vec ldlt_solve_rhs_;                      // size n+s
    Vec Gtr_dy_;                              // size n = N (iterative_refine_dxdy scratch)
    Vec G_dx_;                                // size s = G.rows() (iterative_refine_dxdy scratch)

    // Direct solver: PCG's fallback. Factorizes the Newton system exactly, either the KKT system
    // K = [-H, G^T; G, (1/mu)I] by LDLT (solve_using_ldlt()) or its Schur complement S = G H^{-1} G^T + (1/mu)I
    // by Cholesky (solve_using_chol()), chosen by decide_direct_system(). Used permanently once PCG has failed.
    bool pcg_failed = false;             // True means PCG failed once; the direct solver handles every later Newton solve.
    bool direct_use_ldlt = false;        // decide_direct_system()'s return; true means LDLT on K, false means Cholesky on S.
    int direct_ldlt_decisions_made_ = 0; // decide_direct_system() call count; locked after the first 3.
    bool schur_chol_failed_ = false;     // True means Cholesky on S failed once; LDLT on K handles the rest of the run.
    BoolArr all_cols_;                   // All-true mask over G's columns for choose_schur_ldlt(); built on first use.

    // Cached KKT matrix K = [-H, G^T; G, (1/mu)I].
    // When active_W changes (G's sparsity changes): rebuild K from triplets.
    // When only H_diag or mu changes (diagonal-only update): set diagonal entries in-place.
    SpMat K_ldlt_;
    bool K_ldlt_built_ = false;

    // Cached flat storage indices (into K_ldlt_.valuePtr()) for the diagonal entries.
    // Lets the diagonal-only patch path in solve_using_ldlt() write via valuePtr()[idx].
    std::vector<int> ldlt_diag_top_idx_; // size n, top-left -H block
    std::vector<int> ldlt_diag_bot_idx_; // size s, bottom-right (1/mu)I block

    // Stored LDLT factorization of K.
    SimplicialLDLT64<SpMat> ldlt_;
    bool ldlt_pattern_dirty_ = true; // means K's dimension changed (n_active_W changed), requires analyzePattern.
    bool ldlt_numeric_dirty_ = true; // means K's values changed (H_diag or G rows swapped), requires factorize.

    // Cached lower triangle of the Schur complement S = G H^{-1} G^T + (1/mu)I and its Cholesky factorization.
    // Held by pointer so that a pattern change reseats a fresh solver and release_chol() frees it.
    SpMat S_chol_;
    std::unique_ptr<SimplicialLLT64<SpMat>> chol_;
    bool chol_pattern_dirty_ = true;        // means S's sparsity changed (active_W changed), requires analyzePattern.
    bool chol_numeric_dirty_ = true;        // means S's values changed (H_diag, mu or G rows), requires reassembly and factorize.
    Eigen::Index chol_nnz_at_analyze_ = -1; // S's nonzero count at the last analyzePattern; a mismatch forces re-analysis.
    Vec chol_Hinv_r1_, chol_rhs_;           // size n, s

    SSN(const int Q_info, const Vec& Q_diag, const SpMat& L,
        const SpMat& A, const SpMat& B, const SpMat& A_tr, const SpMat& B_tr,
        const Vec& c, const Vec& b,
        const Vec& D2_ext_inv, const Vec& D1B_diag_inv,
        const Vec& lx, const Vec& ux, const Vec& lw, const Vec& uw,
        int n, int m, int N, int M, int l,
        T ssn_tol, int ssn_max_in_iter, T eps_pinf, T eps_dinf,
        PrintWhen when = PrintWhen::NEVER, PrintWhat what = PrintWhat::NONE)
    : Q_info(Q_info), Q_diag(Q_diag), L(L),
      A(A), B(B), A_tr(A_tr), B_tr(B_tr), c(c), b(b),
      D2_ext_inv(D2_ext_inv), D1B_diag_inv(D1B_diag_inv),
      lx(lx), ux(ux), lw(lw), uw(uw),
      n(n), m(m), N(N), M(M), l(l),
      ssn_tol(ssn_tol), ssn_max_in_iter(ssn_max_in_iter),
      eps_pinf(eps_pinf), eps_dinf(eps_dinf),
      when(when), what(what)
    {
        ssn_iter = 0;
        delta_x = Vec::Zero(N);
        delta_y2 = Vec::Zero(l);

        // M is fixed for this SSN's lifetime.
        cg.preconditioner().set_num_equality_rows(M);

        y2_active_W_.resize(l);
        y2_inactive_W_.resize(l);
        dist_W_v_active_.resize(l);
        dist_W_v_inactive_.resize(l);
        dy2_inactive_W_.resize(l);

        grad_L_.resize(N + l);

        B_rm = B; // Row-major B

        // Cache A's triplets for G.
        G_A_trips_.reserve(A.nonZeros());
        for (int col = 0; col < A.outerSize(); ++col)
            for (typename SpMat::InnerIterator it(A, col); it; ++it)
                G_A_trips_.emplace_back(it.row(), col, it.value());

    }

    void update_ssn_system(const Vec& x, const Vec& y1, const Vec& y2, const Vec& z,
                           const Vec& delta_y1, const Vec& delta_z,
                           T mu, T rho, T alpha, int ssn_iter) {
        this->x = x;
        this->y1 = y1;
        this->y2 = y2;
        this->z = z;
        this->mu = mu;
        this->rho = rho;
        this->alpha = alpha;
        this->ssn_iter = ssn_iter;
        this->delta_y1 = delta_y1;
        this->delta_z = delta_z;

        A_tr_y1_ = A_tr * y1;        // y1 is fixed for the entire SSN run; cache A^T y1 once.
        linesearch_fail = 0;         // Reset line search failure count for this SSN iteration.
        cg.preconditioner().reset_smw_fail_streak(); // Reset SMW suppression.
    }

    template <typename Derived>
    static inline T inf_norm(const Eigen::MatrixBase<Derived>& v) {
        if (v.size() == 0) return T(0);
        return v.cwiseAbs().maxCoeff();
    }

    template <typename Derived>
    static void compute_dist_box(const Eigen::MatrixBase<Derived>& v, const Vec& lower, const Vec& upper, Vec& dist) {
        const int sz = static_cast<int>(v.size());
        dist.resize(sz);
        for (int i = 0; i < sz; ++i) {
            const T vi = v[i];
            dist[i] = vi - std::max(lower[i], std::min(upper[i], vi));
        }
    }

    static void compute_subgrad_and_dist(const Vec& u, const Vec& lower, const Vec& upper,
                                         bool include_bd, BoolArr& subgrad, Vec& dist) {
        const int sz = static_cast<int>(u.size());
        subgrad.resize(sz);
        dist.resize(sz);
        for (int i = 0; i < sz; ++i) {
            const T ui = u[i], li = lower[i], hi = upper[i];
            const T pi = std::max(li, std::min(hi, ui));
            dist[i]    = ui - pi;
            subgrad[i] = include_bd ? (ui >= li && ui <= hi) : (ui > li && ui < hi);
        }
    }
    
    const Vec& compute_grad_Lagrangian(const Vec& x_new, const Vec& y2_new, const Vec& Ax_new, const Vec& Bx_new);
    T compute_grad_Lagrangian_unscaled_inf_norm(const Vec& grad_L);
    void split_by_mask(const Vec& u, const BoolArr& mask, Vec& u_sel, Vec& u_unsel);
    void rebuild_G();
    void retrieve_row_order(const Vec& u_sel, const Vec& u_unsel, const BoolArr& mask, Vec& out);
    bool choose_schur_ldlt(const SpMat& G, const BoolArr& active_K);
    Vec solve_using_cg(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const Vec& H_diag_inv, const BoolArr& active_K, const Vec& r1, const Vec& r2, T mu, T tol, int max_iter, bool update_prec, bool G_pattern_changed, bool schur_use_ldlt);
    void decide_direct_system(const SpMat& G);
    Vec solve_direct(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const Vec& H_diag_inv, const Vec& r1, const Vec& r2);
    Vec solve_using_ldlt(const SpMat& G, const Vec& H_diag, const Vec& r1, const Vec& r2);
    Vec solve_using_chol(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const Vec& H_diag_inv, const Vec& r1, const Vec& r2);
    void release_chol();
    void iterative_refine_dxdy();
    SsnLineSearchParams<T> make_line_search_params();
    void solve_ssn(const T ssn_tol);

    // Tracks active set change; computed once per SSN iteration in prepare_newton_system() call.
    struct ActiveSetDelta { bool k_changed; bool w_changed; };

    // update_prec=true means rebuild P; prec_pattern_changed=true means analyzePattern() is needed.
    struct PrepResult { bool update_prec; bool prec_pattern_changed; };

    enum class LineSearchOutcome { Proceed, AcceptOptimal, Fail };
    struct LineSearchResult { LineSearchOutcome outcome; T tau; };

    PrepResult prepare_newton_system();
    void solve_newton_direction(bool update_prec, bool prec_pattern_changed);
    LineSearchResult line_search_with_steepest_descent_fallback(T ssn_tol);
    void update_iterate(T tau, int ssn_iter_count);
    std::optional<TerminationStatus> check_ssn_termination(T ssn_tol, int& stagnation, T& prev_tol_achieved);
};

#include "ssn.tpp"
