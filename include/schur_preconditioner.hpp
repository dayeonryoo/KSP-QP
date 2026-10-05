#pragma once
#include <cassert>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/LU>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <variant>
#include "amd_ordering.hpp"

// Timer master switch; 0 (off) by default; set via -DSSN_ENABLE_TIMERS=1.
#ifndef SSN_ENABLE_TIMERS
#define SSN_ENABLE_TIMERS 0
#endif

#if SSN_ENABLE_TIMERS
#include <chrono>

// RAII scoped timer; accumulates elapsed wall-clock seconds into `acc` on scope exit.
struct SchurPrecScopedTimer {
    std::chrono::steady_clock::time_point t0;
    double& acc;
    explicit SchurPrecScopedTimer(double& acc_) : t0(std::chrono::steady_clock::now()), acc(acc_) {}
    ~SchurPrecScopedTimer() { acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};
#define SCHUR_PREC_TIMER_BLOCK(acc) SchurPrecScopedTimer _schur_prec_scoped_timer(acc)
#else
#define SCHUR_PREC_TIMER_BLOCK(acc) do {} while (0)
#endif

// =================================================================================================
// Flag-dependency chain (SSN + SchurPreconditioner)
// -------------------------------------------------------------------------------------------------
// Flags at three layers, all driven by the active-set changes SSN::ActiveSetDelta{k_changed,
// w_changed}, computed once per SSN iteration in SSN::prepare_newton_system():
//
// Layer 1 -- SSN::prepare_newton_system()
//   SSN::PrepResult{update_prec, prec_pattern_changed}: both are (k_changed || w_changed), passed
//   through SSN::solve_newton_direction() -> SSN::solve_using_cg() -> arm().
//
// Layer 2 -- SSN's direct solver, used permanently once PCG has failed (SSN::pcg_failed): LDLT on
//            the KKT matrix (K_ldlt_) or Cholesky on its Schur complement (S_chol_), chosen by
//            SSN::decide_direct_system():
//   ldlt_pattern_dirty_ / chol_pattern_dirty_ : set on w_changed (active_W changes G's sparsity).
//   ldlt_numeric_dirty_ / chol_numeric_dirty_ : set on w_changed or whenever H_diag is recomputed
//                         (k_changed, or mu/rho drift from SSN::H_diag_mu_/H_diag_rho_). Guard
//                         factorize(), so an unchanged matrix keeps its factorization.
//   K_ldlt_built_       : K_ldlt_ exists, so a numeric-only update skips triplet reassembly.
//
// Layer 3 -- SchurPreconditioner (this class): Cholesky on P = G E G^T + (1/mu)I or LDLT on
//            P_hat = [-H_act, G_act^T; G_act, (1/mu)I]:
//   rebuild_        : compute() calls build() -- caller's `rebuild` (Layer 1's update_prec) or a
//                     row-count change.
//   pattern_dirty_  : analyzePattern() needed -- caller's `prec_pattern_changed`, a size change,
//                     or the first call.
//   numeric_dirty_  : G E G^T (or P_hat's structural blocks) must be fully rebuilt; same trigger
//                     as pattern_dirty_, not set on a mu/rho-only change.
//   mu_/rho_ vs mu_at_last_fact_/rho_at_last_fact_ : on active_K entries H_diag(i) = Q_diag(i) +
//                     1/rho, so mu and rho touch disjoint parts of P/P_hat. compute() calls build()
//                     on either change, which patches without the dirty flags:
//                       mu-only  : both variants overwrite the (1/mu)I block in place, O(s).
//                       rho-only : chol must rebuild G E G^T (E = 1/H_diag is inside the product);
//                                  ldlt patches -H_act on P_hat's diagonal in place, O(n_act).
//   use_ldlt_ / use_ldlt_at_last_fact_ : Cholesky vs LDLT, from SSN::schur_use_ldlt via arm(); a
//                     switch forces prec_pattern_changed.
//   skip_smw_ / use_smw_ / has_snapshot_ / smw_fail_streak_ : SMW control (see try_build_smw()).
//                     The update reuses P_old from the last full rebuild for every retained row and
//                     column, so smw_gate_open() rejects any rho or mu change since the snapshot.
// =================================================================================================

template <typename T>
class SchurPreconditioner {
public:
    using Scalar = T;
    using RealScalar = typename Eigen::NumTraits<T>::Real;
    using StorageIndex = Eigen::Index;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using SpMat = Eigen::SparseMatrix<T>;
    using RowMajorSpMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;
    using Triplet = Eigen::Triplet<T>;
    using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;

    // ------ Setup ------

    SchurPreconditioner() = default;

    SchurPreconditioner(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const BoolArr& active_K, T mu)
        : G_(&G), G_tr_(&G_tr), H_diag_(&H_diag), active_K_(&active_K), mu_(mu) {}

    // Sets the number of equality-constraint (A) rows once from data. 
    void set_num_equality_rows(int M) { M_rows_ = M; }

    void set_data(const SpMat& G, const SpMat& G_tr, const Vec& H_diag,
                 const BoolArr& active_K, const BoolArr& active_W,
                 const RowMajorSpMat& B_rm,
                 T mu, T rho, bool rebuild, bool prec_pattern_changed) {
        // Store data pointers and set flags for refactorization.
        G_        = &G;
        G_tr_     = &G_tr;
        H_diag_   = &H_diag;
        active_K_ = &active_K;
        active_W_ = &active_W;
        B_rm_     = &B_rm; // row-major B matrix
        mu_       = mu;
        rho_      = rho;

        // Detect if refactorization is needed.
        bool size_changed = (s_current_ != static_cast<int>(G.rows()));
        if (!pattern_analyzed_ || prec_pattern_changed || size_changed)
            pattern_dirty_ = true;

        // Detect if G E G^T needs to be recomputed.
        if (prec_pattern_changed || size_changed)
            numeric_dirty_ = true;

        // Detect if P needs to be rebuild.
        rebuild_ = rebuild || size_changed;

        // Fallback if set_num_equality_rows() wasn't called: M_rows_ = G.rows() - n_active_W.
        if (M_rows_ < 0 && static_cast<int>(G.rows()) > 0)
            M_rows_ = static_cast<int>(G.rows()) - static_cast<int>(active_W.count());
    }

    // Per-solve-attempt setup protocol. Pair with consume_fact_count_delta() after cg.compute(S).
    void arm(const SpMat& G, const SpMat& G_tr, const Vec& H_diag,
             const BoolArr& active_K, const BoolArr& active_W, const RowMajorSpMat& B_rm,
             T mu, T rho, bool rebuild, bool prec_pattern_changed, bool use_ldlt, bool force_rebuild = false) {
        if (use_ldlt != use_ldlt_)       // P and P_hat differ structurally, so the cached
            prec_pattern_changed = true; // analyzePattern() is invalid.
        set_data(G, G_tr, H_diag, active_K, active_W, B_rm, mu, rho, rebuild || force_rebuild, prec_pattern_changed);
        set_use_ldlt(use_ldlt);
        if (force_rebuild || smw_suppressed())
            force_full_rebuild();
        fact_count_at_arm_ = fact_count_;
    }

    // Returns how many factorizations happened since the last arm() call. Call once, right after cg.compute(S).
    int consume_fact_count_delta() {
        const int delta = fact_count_ - fact_count_at_arm_;
        fact_count_at_arm_ = fact_count_;
        return delta;
    }

    // On PCG failure: if the preconditioner used SMW, records the failure and returns true (retry with
    // arm(..., force_rebuild=true)); otherwise returns false.
    bool should_retry_after_failure() {
        if (!use_smw_) return false;
        record_smw_rebuild();
        return true;
    }

    // ------ Factorization ------

    template <typename MatrixType>
    SchurPreconditioner& compute(const MatrixType&) {
        bool mu_changed  = initialized_ && (mu_  != mu_at_last_fact_);
        bool rho_changed = initialized_ && (rho_ != rho_at_last_fact_);
        if (!initialized_ || rebuild_ || mu_changed || rho_changed) {
            build();
            initialized_ = true;
        }
        return *this;
    }

    // ------ Application ------

    template <typename Rhs>
    const Vec& solve(const Eigen::MatrixBase<Rhs>& b) const {
        if (info_ != Eigen::Success)
            throw std::runtime_error("SchurPreconditioner solve called after failed factorization.");
        return use_smw_ ? solve_smw(b) : solve_direct(b);
    }

    // ------ Diagnostics & SMW control ------

    // Why the most recent try_build_smw() attempt did or didn't activate SMW.
    enum class SmwRejectReason {
        None,                         // SMW was used, or no attempt has run yet.
        ForcedRebuild,                // Caller requested a full rebuild (force_full_rebuild()/arm(force_rebuild=true)).
        Suppressed,                   // Fail-streak threshold reached; SMW disabled.
        MissingData,                  // active_W/B_rm not supplied, or no equality rows tracked yet.
        FactorizationMethodChanged,   // use_ldlt_ differs from the method used at the last full factorization.
        NoSnapshot,                   // No prior full-rebuild snapshot to update from.
        RhoChangedSinceSnapshot,      // rho changed since the snapshot, so H_diag changed outside the delta.
        MuChangedSinceSnapshot,       // mu changed since the snapshot: a full-rank shift of the (1/mu)I block.
        RankExceedsThreshold,         // Active-set delta rank exceeds the SMW update-size threshold.
        ReusedSnapshot,               // Rank 0: the active sets match the snapshot, whose factorization is reused as is.
        SingularCapacitance,          // Equilibrated capacitance matrix was (near-)singular or non-finite;
                                      // fell back to full rebuild.
    };

    Eigen::ComputationInfo info() const { return info_; }
    int fact_count() const { return fact_count_; }
    int smw_count()  const { return smw_count_; }
    bool used_smw()  const { return use_smw_; }
    SmwRejectReason smw_last_reject_reason() const { return smw_last_reject_reason_; }
    int  smw_last_rank() const { return smw_last_rank_; }
    bool used_ldlt_at_last_fact() const { return use_ldlt_at_last_fact_; }

#if SSN_ENABLE_TIMERS
    // TIMER: step-by-step time spent in build().
    double assembly_time()  const { return assembly_time_; }
    double analyze_time()   const { return analyze_time_; }
    double factorize_time() const { return factorize_time_; }
#endif

    void force_full_rebuild() { skip_smw_ = true; numeric_dirty_ = true; }
    void record_smw_rebuild() {
        smw_fail_streak_++;
        smw_fail_total_++;
        if (smw_fail_streak_ >= kMaxSmwFailStreak) {
            SpMat().swap(G_old_);
            H_diag_old_.resize(0);
            active_K_old_.resize(0);
            active_W_old_.resize(0);
            snapshot_wiped_by_fail_streak_ = true;
        }
    }
    void reset_smw_fail_streak() { smw_fail_streak_ = 0; }
    bool smw_suppressed() const {
        if (smw_fail_streak_ >= kMaxSmwFailStreak) return true;
        else return false;
    }
    void set_use_ldlt(bool flag) { use_ldlt_ = flag; }

    // Free all factorization/SMW state once the caller has permanently switched away from PCG.
    void release() {
        active_solver_.template emplace<std::monostate>();

        SpMat().swap(G_old_);
        H_diag_old_.resize(0);
        active_K_old_.resize(0);
        active_W_old_.resize(0);
        has_snapshot_ = false;
        snapshot_wiped_by_fail_streak_ = false;

        std::vector<int>().swap(deleted_old_rows_);
        std::vector<int>().swap(retained_old_rows_);
        std::vector<int>().swap(retained_new_rows_);
        std::vector<int>().swap(added_new_rows_);
        std::vector<int>().swap(added_W_src_);
        std::vector<int>().swap(delta_K_idx_);
        std::vector<int>().swap(ldlt_act_idx_);
        std::vector<Triplet>().swap(ldlt_build_trips_);
        E_diag_.resize(0);
        std::vector<int>().swap(ldlt_diag_top_idx_);
        std::vector<int>().swap(ldlt_diag_bot_idx_);
        std::vector<int>().swap(diag_idx_chol_);
        V_plus_.resize(0, 0);
        Y_all_.resize(0, 0);
        S_lambda_lu_ = Eigen::FullPivLU<Mat>();
        S_lambda_eq_scale_.resize(0);

        smw_e_new_b_.resize(0);
        std::vector<int>().swap(smw_touched_);
        smw_tmp_.resize(0);
        smw_ldlt_padded_.resize(0);

        r_pad_.resize(0);
        u_base_.resize(0);
        Lambda_all_.resize(0);
        lambda_work_.resize(0);
        z_base_.resize(0);
        z_new_.resize(0);
        direct_result_.resize(0);
        ldlt_rhs_.resize(0);

        G_ = nullptr; G_tr_ = nullptr; H_diag_ = nullptr; active_K_ = nullptr; active_W_ = nullptr; B_rm_ = nullptr;

        initialized_      = false;
        rebuild_          = true;
        pattern_analyzed_ = false;
        pattern_dirty_    = true;
        numeric_dirty_       = true;
        use_smw_          = false;
        skip_smw_         = false;
        smw_last_reject_reason_ = SmwRejectReason::None;
        smw_last_rank_          = 0;
    }

    // Grants test-only access to private scratch buffers; only used from tests/test_schur_preconditioner.cpp.
    friend struct SchurPreconditionerTestPeer;

private:
    // ------ Application (private helpers behind the public solve() dispatcher) ------

    template <typename Rhs>
    const Vec& solve_direct(const Eigen::MatrixBase<Rhs>& b) const {
        if (use_ldlt_) {
            // Solve hat_P [w1; w2] = [0; b]; return w2 = P^{-1} b.
            const int s = static_cast<int>(b.size());
            ldlt_rhs_.resize(n_act_ + s);
            ldlt_rhs_.head(n_act_).setZero();
            ldlt_rhs_.tail(s) = b;
            direct_result_ = std::get<LdltSolver>(active_solver_).ldlt->solve(ldlt_rhs_).tail(s);
        } else {
            direct_result_ = std::get<CholSolver>(active_solver_).llt->solve(b);
        }
        return direct_result_;
    }

    template <typename Rhs>
    const Vec& solve_smw(const Eigen::MatrixBase<Rhs>& b) const {
        r_pad_.head(M_rows_) = b.head(M_rows_);
        for (int i = 0; i < static_cast<int>(retained_old_rows_.size()); ++i)
            r_pad_(retained_old_rows_[i]) = b(retained_new_rows_[i]);

        // u_base = P_old^-1 r_pad
        if (use_ldlt_) {
            const int s = r_pad_.size();
            ldlt_rhs_.resize(n_act_ + s);
            ldlt_rhs_.head(n_act_).setZero();
            ldlt_rhs_.tail(s) = r_pad_;
            u_base_ = std::get<LdltSolver>(active_solver_).ldlt->solve(ldlt_rhs_).tail(s);
        } else {
            u_base_ = std::get<CholSolver>(active_solver_).llt->solve(r_pad_);
        }

        // Lambda = g_all - V_all^T u_base.
        //   E_-^T u_base
        for (int k = 0; k < h_; ++k)
            Lambda_all_(k) = -u_base_(deleted_old_rows_[k]);
        //   U^T u_base
        for (int j = 0; j < p_; ++j) {
            T val = T(0);
            for (typename SpMat::InnerIterator it(G_old_, delta_K_idx_[j]); it; ++it)
                val += it.value() * u_base_(it.row());
            Lambda_all_(h_ + j) = -val;
        }
        //   V_+^T u_base - r_2
        for (int j = 0; j < q_; ++j)
            Lambda_all_(h_ + p_ + j) = b(added_new_rows_[j]);
        if (q_ > 0)
            Lambda_all_.tail(q_).noalias() -= V_plus_.transpose() * u_base_;

        // Capacitance solve through the equilibrated factorization D S_Lambda D.
        lambda_work_ = S_lambda_eq_scale_.cwiseProduct(Lambda_all_);
        Lambda_all_  = S_lambda_lu_.solve(lambda_work_);
        Lambda_all_  = S_lambda_eq_scale_.cwiseProduct(Lambda_all_);

        // z_base = u_base - Y_all * Lambda_all
        z_base_  = u_base_;
        z_base_.noalias() -= Y_all_ * Lambda_all_;

        // Assemble z_new.
        z_new_.head(M_rows_) = z_base_.head(M_rows_);
        for (int i = 0; i < static_cast<int>(retained_old_rows_.size()); ++i)
            z_new_(retained_new_rows_[i]) = z_base_(retained_old_rows_[i]);
        for (int j = 0; j < q_; ++j)
            z_new_(added_new_rows_[j]) = Lambda_all_(h_ + p_ + j);

        return z_new_;
    }

    // ------ Formation and Cholesky factorization ------

    void build() {
        use_smw_ = false;

        // Case 1. Skip build() and reuse the cached factorization, via a low-rank SMW update or,
        // if the active sets are back to the snapshot's, as is.
        if (initialized_ && info_ == Eigen::Success && try_build_smw()) {
            rebuild_ = false;
            return;
        }
        V_plus_.resize(0, 0);
        Y_all_.resize(0, 0);

        // Case 2. Full factorization.
        if (use_ldlt_) {
            if (!std::holds_alternative<LdltSolver>(active_solver_))
                active_solver_.template emplace<LdltSolver>();
            factorize_by_ldlt();
        } else {
            if (!std::holds_alternative<CholSolver>(active_solver_))
                active_solver_.template emplace<CholSolver>();
            factorize_by_chol();
        }
    }

    // Shared tail of factorize_by_ldlt/factorize_by_chol.
    template <typename FactorSolver>
    void finish_factorization(FactorSolver& solver, const SpMat& P, Eigen::Index s, bool is_ldlt,
                               bool structural_change, int n_act = -1) {
        if (pattern_dirty_) {
            {
                SCHUR_PREC_TIMER_BLOCK(analyze_time_);
                solver.analyzePattern(P);
            }
            pattern_analyzed_ = true;
            pattern_dirty_ = false;
        }
        {
            SCHUR_PREC_TIMER_BLOCK(factorize_time_);
            solver.factorize(P);
        }
        info_ = solver.info();
        fact_count_++;
#if SSN_ENABLE_TIMERS
        fprintf(stderr, "[SchurFactorize] fact=%d is_ldlt=%d reselected=%d\n",
                fact_count_, (int)is_ldlt, (int)structural_change);
#endif
        mu_at_last_fact_       = mu_;
        rho_at_last_fact_      = rho_;
        use_ldlt_at_last_fact_ = is_ldlt;
        if (n_act >= 0) n_act_ = n_act;
        s_current_ = static_cast<int>(s);
        numeric_dirty_ = false;
        if (info_ == Eigen::Success) rebuild_ = false;

        if (!smw_suppressed()) snapshot_state(structural_change);
    }

    // Build P_hat = [-H_act, G_act^T; G_act, (1/mu)I] and factorize with LDLT.
    void factorize_by_ldlt() {
        const SpMat&   G        = *G_;
        const Vec&     H_diag   = *H_diag_;
        const BoolArr& active_K = *active_K_;

        const Eigen::Index s = G.rows();
        const Eigen::Index n = G.cols();

        auto& sol = std::get<LdltSolver>(active_solver_);
        const bool structural_change = numeric_dirty_; // captured before finish_factorization() resets it.
        int n_act;
        {
            SCHUR_PREC_TIMER_BLOCK(assembly_time_);

            if (numeric_dirty_) {
                // Collect active K indices.
                ldlt_act_idx_.clear();
                ldlt_act_idx_.reserve(n);
                for (Eigen::Index i = 0; i < n; ++i)
                    if (active_K(i)) ldlt_act_idx_.push_back(static_cast<int>(i));
                n_act = static_cast<int>(ldlt_act_idx_.size());

                // Build P_hat of size (n_act + s) x (n_act + s).
                ldlt_build_trips_.clear();
                ldlt_build_trips_.reserve(n_act + 2 * static_cast<int>(G.nonZeros()) + static_cast<int>(s));

                // Top-left block: -H_act (diagonal, n_act x n_act).
                for (int k = 0; k < n_act; ++k)
                    ldlt_build_trips_.emplace_back(k, k, -H_diag(ldlt_act_idx_[k]));

                // Off-diagonal blocks: G_act (s x n_act) and G_act^T (n_act x s).
                for (int k = 0; k < n_act; ++k)
                    for (typename SpMat::InnerIterator it(G, ldlt_act_idx_[k]); it; ++it) {
                        const int row = static_cast<int>(it.row());
                        ldlt_build_trips_.emplace_back(n_act + row, k,           it.value()); // G_act
                        ldlt_build_trips_.emplace_back(k,           n_act + row, it.value()); // G_act^T
                    }

                // Bottom-right block: (1/mu) I_s.
                for (Eigen::Index i = 0; i < s; ++i)
                    ldlt_build_trips_.emplace_back(n_act + i, n_act + i, T(1) / mu_);

                sol.P_hat.resize(n_act + s, n_act + s);
                sol.P_hat.setFromTriplets(ldlt_build_trips_.begin(), ldlt_build_trips_.end());
                sol.P_hat.makeCompressed();

                // Cache each diagonal's valuePtr() index for the in-place patch path below.
                ldlt_diag_top_idx_.resize(n_act);
                for (int k = 0; k < n_act; ++k)
                    ldlt_diag_top_idx_[k] = static_cast<int>(&sol.P_hat.coeffRef(k, k) - sol.P_hat.valuePtr());
                ldlt_diag_bot_idx_.resize(s);
                for (Eigen::Index i = 0; i < s; ++i)
                    ldlt_diag_bot_idx_[i] = static_cast<int>(&sol.P_hat.coeffRef(n_act + i, n_act + i) - sol.P_hat.valuePtr());
            } else {
                // Same active set as last full build.
                // Both a mu-only and a rho-only change are patched in place without reassembling any triplet:
                //  - mu only affects the bottom-right (1/mu) I block (H_diag's mu term vanishes
                //    exactly on active_K entries, so mu never touches -H_act).
                //  - rho only affects the top-left -H_act block (H_diag(i) = Q_diag(i) + 1/rho
                //    on active_K entries).
                n_act = n_act_;
                assert(sol.P_hat.rows() == n_act + s && sol.P_hat.cols() == n_act + s);
                if (rho_ != rho_at_last_fact_)
                    for (int k = 0; k < n_act; ++k)
                        sol.P_hat.valuePtr()[ldlt_diag_top_idx_[k]] = -H_diag(ldlt_act_idx_[k]);
                for (Eigen::Index i = 0; i < s; ++i)
                    sol.P_hat.valuePtr()[ldlt_diag_bot_idx_[i]] = T(1) / mu_;
            }
        }

        if (pattern_dirty_) sol.ldlt = std::make_unique<LdltType>();

        finish_factorization(*sol.ldlt, sol.P_hat, s, /*is_ldlt=*/true, structural_change, n_act);
    }

    // Build P = G E G^T + (1/mu) I (or shift its mu diagonal), then factorize with Cholesky.
    void factorize_by_chol() {
        const SpMat&   G        = *G_;
        const SpMat&   G_tr     = *G_tr_;
        const Vec&     H_diag   = *H_diag_;
        const BoolArr& active_K = *active_K_;

        const Eigen::Index s = G.rows();
        const Eigen::Index n = G.cols();

        assert(G_tr.rows() == n);
        assert(G_tr.cols() == s);
        assert(H_diag.size() == n);
        assert(active_K.size() == n);

        auto& sol = std::get<CholSolver>(active_solver_);

        const bool rho_changed = (rho_ != rho_at_last_fact_);
        const bool structural_change = numeric_dirty_; // captured before the branch below resets it.
        {
            SCHUR_PREC_TIMER_BLOCK(assembly_time_);
            if (numeric_dirty_ || rho_changed) {
                // Rebuild G E G^T. Needed on a pattern change (numeric_dirty_) or a rho-only change.
                assert(H_diag.minCoeff() > T(0));
                E_diag_.resize(n);
                for (Eigen::Index i = 0; i < n; ++i)
                    E_diag_(i) = active_K(i) ? T(1) / H_diag(i) : T(0);

                // Scale G's columns by sqrt(E) directly (E >= 0, so sqrt is well-defined).
                SpMat G_scaled = G;
                for (Eigen::Index k = 0; k < G_scaled.outerSize(); ++k) {
                    const T scale = std::sqrt(E_diag_(k));
                    for (typename SpMat::InnerIterator it(G_scaled, k); it; ++it)
                        it.valueRef() *= scale;
                }
                G_scaled.prune(T(0)); // drop explicit zeros from inactive-K columns.

                // P = G E G^T = G_scaled * G_scaled^T, but SimplicialLLT only ever reads the lower triangle.
                sol.P.template selfadjointView<Eigen::Lower>().rankUpdate(G_scaled, T(0));
                numeric_dirty_ = false;

                // Add (1/mu) I as a sparse matrix addition.
                SpMat mu_diag(s, s);
                mu_diag.setIdentity();
                mu_diag *= T(1) / mu_;
                sol.P += mu_diag;
                sol.P.makeCompressed();

                // Cache each diagonal's valuePtr() index for the mu-only shift path below.
                diag_idx_chol_.resize(s);
                for (Eigen::Index i = 0; i < s; ++i)
                    diag_idx_chol_[i] = static_cast<int>(&sol.P.coeffRef(i, i) - sol.P.valuePtr());
            } else {
                // G E G^T unchanged; only mu changed: shift the (1/mu) I diagonal by delta.
                assert(sol.P.rows() == s && sol.P.cols() == s);
                const T delta = (mu_at_last_fact_ - mu_) / (mu_ * mu_at_last_fact_);
                for (Eigen::Index i = 0; i < s; ++i)
                    sol.P.valuePtr()[diag_idx_chol_[i]] += delta;
            }
        }

        if (pattern_dirty_) sol.llt = std::make_unique<LltType>();

        finish_factorization(*sol.llt, sol.P, s, /*is_ldlt=*/false, structural_change);
    }

    // ------ SMW low-rank update ------

    // Resize v to size and fill with zeros if it was resized. Otherwise leave it unchanged.
    static void zero_resize(Vec& v, Eigen::Index size) {
        if (v.size() != size) v.setZero(size);
    }

    // SMW Setup Phase. Returns true and arms use_smw_ iff 0 < h+p+q <= kSmwRankThreshold.
    // Also returns true, with use_smw_ left false, if h+p+q == 0: the cached factorization is reused.
    bool try_build_smw() {
        smw_last_reject_reason_ = SmwRejectReason::None;
        if (!smw_gate_open())              return false;
        if (!classify_active_set_delta())  return false;
        if (h_ + p_ + q_ == 0) {           // P equals the snapshot's P_old
            reuse_snapshot_factorization();
            return true;
        }

        const Mat M_sub = build_capacitance_setup();
        compute_y_all();
        if (!factorize_capacitance(M_sub)) return false;

        finalize_smw_success();
        return true;
    }

    // Phase 1: eligibility gate -- every reason SMW can't even be attempted this call.
    bool smw_gate_open() {
        if (skip_smw_) {
            skip_smw_ = false;
            smw_last_reject_reason_ = SmwRejectReason::ForcedRebuild;
            return false;
        }
        if (smw_fail_streak_ >= kMaxSmwFailStreak) {
            smw_last_reject_reason_ = SmwRejectReason::Suppressed;
            return false;
        }
        if (!active_W_ || !B_rm_ || M_rows_ < 0) {
            smw_last_reject_reason_ = SmwRejectReason::MissingData;
            return false;
        }
        // If the factorization method changed since last full rebuild, do refactorization instead of SMW.
        if (use_ldlt_ != use_ldlt_at_last_fact_) {
            smw_last_reject_reason_ = SmwRejectReason::FactorizationMethodChanged;
            return false;
        }
        // No usable snapshot: none taken yet, or record_smw_rebuild() wiped G_old_ after a fail streak.
        if (!has_snapshot_ || (G_old_.rows() == 0 && snapshot_wiped_by_fail_streak_)) {
            smw_last_reject_reason_ = SmwRejectReason::NoSnapshot;
            return false;
        }
        // P_old was factorized with the snapshot's rho, and H_diag(i) = Q_diag(i) + 1/rho on
        // active_K entries, so a rho change invalidates it.
        if (rho_ != rho_old_) {
            smw_last_reject_reason_ = SmwRejectReason::RhoChangedSinceSnapshot;
            return false;
        }
        // The update recomputes (1/mu) only for added W rows; retained rows inherit it from P_old.
        // A mu change is thus a full-rank shift.
        if (mu_ != mu_old_) {
            smw_last_reject_reason_ = SmwRejectReason::MuChangedSinceSnapshot;
            return false;
        }
        return true;
    }

    // Phase 2: classify the active-set delta (K flips, W row add/delete) against the snapshot;
    // reject if the resulting update rank exceeds kSmwRankThreshold.
    bool classify_active_set_delta() {
        s_old_ = static_cast<int>(G_old_.rows());
        const int N = static_cast<int>(G_old_.cols());
        const int l = static_cast<int>(active_W_->size());

        deleted_old_rows_.clear();
        retained_old_rows_.clear();
        retained_new_rows_.clear();
        added_new_rows_.clear();
        added_W_src_.clear();
        delta_K_idx_.clear();

        {
            int old_pos = M_rows_, new_pos = M_rows_;
            for (int i = 0; i < l; ++i) {
                const bool in_old = active_W_old_(i), in_new = (*active_W_)(i);
                if (in_old && in_new) {
                    retained_old_rows_.push_back(old_pos++);
                    retained_new_rows_.push_back(new_pos++);
                } else if (in_old) {
                    deleted_old_rows_.push_back(old_pos++);
                } else if (in_new) {
                    added_new_rows_.push_back(new_pos++);
                    added_W_src_.push_back(i);
                }
            }
        }

        for (int i = 0; i < N; ++i)
            if (active_K_old_(i) != (*active_K_)(i))
                delta_K_idx_.push_back(i);

        h_ = static_cast<int>(deleted_old_rows_.size());
        p_ = static_cast<int>(delta_K_idx_.size());
        q_ = static_cast<int>(added_new_rows_.size());
        const int rank = h_ + p_ + q_;
        smw_last_rank_ = rank;
        if (rank > kSmwRankThreshold) {
            smw_last_reject_reason_ = SmwRejectReason::RankExceedsThreshold;
            return false;
        }
        return true;
    }

    // Phase 3: build V_plus_ (s_old_ x q_) and M_sub (rank x rank).
    Mat build_capacitance_setup() {
        const int rank = h_ + p_ + q_;
        const int N = static_cast<int>(G_old_.cols());
        Mat M_sub = Mat::Zero(rank, rank);

        // Block 2: -C^-1 (diagonal p_×p_); C_jj = E_new[idx] - E_old[idx].
        for (int j = 0; j < p_; ++j) {
            const int idx = delta_K_idx_[j];
            M_sub(h_ + j, h_ + j) = (*active_K_)(idx) ? -(*H_diag_)(idx) : H_diag_old_(idx);
        }

        // V_plus_[:,j] = sum_{i in B_j, active_K[i]} (b_ji / H_diag[i]) * G_old_[:,i].
        // Block 3: W_+ = B_+ E_new B_+^T + (1/mu) I  (q_×q_).
        V_plus_.setZero(s_old_, q_);
        zero_resize(smw_e_new_b_, N);
        // zero_resize() keeps old contents on a same-size reuse, so this relies on the loop below
        // re-zeroing every touched entry.
        assert((smw_e_new_b_.size() == 0 || smw_e_new_b_.cwiseAbs().maxCoeff() == T(0)) &&
               "smw_e_new_b_ zero-invariant violated: dirty state leaked from a previous call.");
        for (int j = 0; j < q_; ++j) {
            smw_touched_.clear();
            for (typename RowMajorSpMat::InnerIterator it(*B_rm_, added_W_src_[j]); it; ++it) {
                const int col = it.col();
                if ((*active_K_)(col)) {
                    smw_e_new_b_(col) = it.value() / (*H_diag_)(col);
                    smw_touched_.push_back(col);
                }
            }
            for (int col : smw_touched_)
                for (typename SpMat::InnerIterator git(G_old_, col); git; ++git)
                    V_plus_(git.row(), j) += smw_e_new_b_(col) * git.value();

            for (int k = j; k < q_; ++k) {
                T val = (j == k) ? T(1) / mu_ : T(0);
                for (typename RowMajorSpMat::InnerIterator it(*B_rm_, added_W_src_[k]); it; ++it)
                    val += it.value() * smw_e_new_b_(it.col());  // smw_e_new_b_ is 0 at non-active_K
                M_sub(h_ + p_ + j, h_ + p_ + k) = val;
                if (j != k) M_sub(h_ + p_ + k, h_ + p_ + j) = val;
            }

            for (int col : smw_touched_) smw_e_new_b_(col) = T(0);
        }
        return M_sub;
    }

    // Phase 4: Y_all_ = P_old^-1 [E_-, U, V_plus_] (s_old_ × rank), column by column.
    //   E_-: one column per deleted row.
    //   U  : one column per delta_K flip.
    //   V_+: one column per added W row; dense multi-RHS solve.
    void compute_y_all() {
        const int rank = h_ + p_ + q_;
        Y_all_.resize(s_old_, rank);
        zero_resize(smw_tmp_, s_old_);
        // Same zero invariant as smw_e_new_b_ in build_capacitance_setup().
        assert((smw_tmp_.size() == 0 || smw_tmp_.cwiseAbs().maxCoeff() == T(0)) &&
               "smw_tmp_ zero-invariant violated: dirty state leaked from a previous call.");

        // Only the tail is written per solve; setZero (not zero_resize) so a same-size reuse with a
        // different n_act_ can't leave stale data in the head.
        if (use_ldlt_) smw_ldlt_padded_.setZero(n_act_ + s_old_);

        // Helper: P_old^-1 v via LLT, or (P_hat_old^-1 [0;v]).tail via LDLT.
        auto solve_base_vec = [&](const Vec& rhs) -> Vec {
            if (use_ldlt_) {
                smw_ldlt_padded_.tail(s_old_) = rhs;
                return std::get<LdltSolver>(active_solver_).ldlt->solve(smw_ldlt_padded_).tail(s_old_);
            } else {
                return std::get<CholSolver>(active_solver_).llt->solve(rhs);
            }
        };

        // Cols 0..h_-1: P_old^-1 e_{del_k}
        for (int k = 0; k < h_; ++k) {
            if (k > 0) smw_tmp_(deleted_old_rows_[k - 1]) = T(0);
            smw_tmp_(deleted_old_rows_[k]) = T(1);
            Y_all_.col(k) = solve_base_vec(smw_tmp_);
        }
        if (h_ > 0) smw_tmp_(deleted_old_rows_[h_ - 1]) = T(0);

        // Cols h_..h_+p_-1: P_old^-1 G_old_col_j
        for (int j = 0; j < p_; ++j) {
            for (typename SpMat::InnerIterator it(G_old_, delta_K_idx_[j]); it; ++it)
                smw_tmp_(it.row()) = it.value();
            Y_all_.col(h_ + j) = solve_base_vec(smw_tmp_);
            for (typename SpMat::InnerIterator it(G_old_, delta_K_idx_[j]); it; ++it)
                smw_tmp_(it.row()) = T(0);
        }

        // Cols h_+p_..rank-1: P_old^-1 V_plus_
        if (q_ > 0) {
            if (use_ldlt_) {
                Mat padded_V = Mat::Zero(n_act_ + s_old_, q_);
                padded_V.bottomRows(s_old_) = V_plus_;
                Y_all_.rightCols(q_) = std::get<LdltSolver>(active_solver_).ldlt->solve(padded_V).bottomRows(s_old_);
            } else {
                Y_all_.rightCols(q_) = std::get<CholSolver>(active_solver_).llt->solve(V_plus_);
            }
        }
    }

    // Phase 5: factor the capacitance matrix S_Lambda = M_sub - V_all^T Y_all_; if near-singular,
    // the caller falls back to a full rebuild.
    // Its blocks have very different scales (flipped columns ~ H ~ 1/rho, added rows ~ |b|^2 / H,
    // up to rho |b|^2), so a pivot test against the largest pivot alone would reject well-conditioned
    // updates. S_Lambda is first equilibrated symmetrically with tau_i = max(|M_sub_ii|,
    // |(V_all^T Y_all)_ii|), so cancellation in S_Lambda_ii still shows as a small pivot, measured
    // against max(1, largest pivot).
    bool factorize_capacitance(const Mat& M_sub) {
        const int rank = h_ + p_ + q_;
        Mat VtY = Mat::Zero(rank, rank);

        // Rows 0..h_-1: (E_-)^T Y_all_ = Y_all_.row(del_k)
        for (int k = 0; k < h_; ++k)
            VtY.row(k) = Y_all_.row(deleted_old_rows_[k]);

        // Rows h_..h_+p_-1: (G_old_col_j)^T Y_all_
        for (int j = 0; j < p_; ++j)
            for (typename SpMat::InnerIterator it(G_old_, delta_K_idx_[j]); it; ++it)
                VtY.row(h_ + j) += it.value() * Y_all_.row(it.row());

        // Rows h_+p_..rank-1: V_plus_^T Y_all_
        if (q_ > 0)
            VtY.bottomRows(q_).noalias() = V_plus_.transpose() * Y_all_;

        // Near-singular (or non-finite) capacitance matrix; fall back to full rebuild.
        auto reject = [this] {
            smw_last_reject_reason_ = SmwRejectReason::SingularCapacitance;
            return false;
        };

        Mat S_Lambda = M_sub - VtY;
        if (!S_Lambda.allFinite()) return reject();

        S_lambda_eq_scale_.resize(rank);
        for (int i = 0; i < rank; ++i) {
            const T tau = std::max(std::abs(M_sub(i, i)), std::abs(VtY(i, i)));
            if (!(tau > T(0))) return reject();
            S_lambda_eq_scale_(i) = T(1) / std::sqrt(tau);
        }
        S_Lambda = S_lambda_eq_scale_.asDiagonal() * S_Lambda * S_lambda_eq_scale_.asDiagonal();

        S_lambda_lu_.compute(S_Lambda);
        const Vec pivots = S_lambda_lu_.matrixLU().diagonal().cwiseAbs();
        const T pivot_tol = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), pivots.maxCoeff());
        if (!(pivots.minCoeff() > pivot_tol)) return reject();
        return true;
    }

    // Phase 6: SMW setup succeeded -- resize solve-time hot-loop buffers and flip on use_smw_.
    void finalize_smw_success() {
        const int s_new = s_old_ - h_ + q_;
        r_pad_.setZero(s_old_);
        u_base_.resize(s_old_);
        Lambda_all_.resize(h_ + p_ + q_);
        lambda_work_.resize(h_ + p_ + q_);
        z_base_.resize(s_old_);
        z_new_.resize(s_new);
        s_current_ = s_new;
        mu_at_last_fact_  = mu_;  // SMW's result reflects the current mu/rho, not the snapshot's;
        rho_at_last_fact_ = rho_; // keep compute()'s mu_changed/rho_changed gate in sync with that.
        use_smw_   = true;
        smw_count_++;
    }

    // Rank-0 case of try_build_smw(): the active sets are back to the snapshot's, so the cached
    // factorization is exactly the current preconditioner. Reuse it without refactorizing.
    void reuse_snapshot_factorization() {
        V_plus_.resize(0, 0);
        Y_all_.resize(0, 0);
        s_current_        = s_old_;
        mu_at_last_fact_  = mu_;   // equal to mu_old_ and rho_old_ (checked by smw_gate_open())
        rho_at_last_fact_ = rho_;
        pattern_dirty_    = false; // the cached symbolic analysis and values match P again
        numeric_dirty_    = false;
        smw_last_reject_reason_ = SmwRejectReason::ReusedSnapshot;
    }

    // Helper: snapshot last full-rebuild state for next SMW attempt.
    void snapshot_state(bool structural_change) {
        if (structural_change) {
            G_old_        = *G_;
            active_K_old_ = *active_K_;
            if (active_W_) active_W_old_ = *active_W_;
            snapshot_wiped_by_fail_streak_ = false; // only a structural snapshot restores what a fail streak wiped
        }
        H_diag_old_   = *H_diag_;
        mu_old_       = mu_;
        rho_old_      = rho_;
        has_snapshot_ = true;
    }

    // ------ Setup: external data ------
    const SpMat*         G_        = nullptr;
    const SpMat*         G_tr_     = nullptr;
    const Vec*           H_diag_   = nullptr;
    const BoolArr*       active_K_ = nullptr;
    const BoolArr*       active_W_ = nullptr;
    const RowMajorSpMat* B_rm_     = nullptr;
    T mu_  = T(1);
    T rho_ = T(1);
    int M_rows_    = -1;  // number of equality-constraint rows
    int s_current_ = -1;  // current row count of G

    // ------ Formation state ------
    bool initialized_      = false;
    bool rebuild_          = true;
    bool pattern_analyzed_ = false;
    bool pattern_dirty_    = true;
    bool numeric_dirty_       = true;
    T    mu_at_last_fact_  = T(-1);
    T    rho_at_last_fact_ = T(-1);

#if SSN_ENABLE_TIMERS
    // TIMER: cumulative wall-clock seconds spent in build().
    double assembly_time_  = 0.0;
    double analyze_time_   = 0.0;
    double factorize_time_ = 0.0;
#endif

    // ------ Cholesky factorization ------
    bool use_ldlt_ = false;
    bool use_ldlt_at_last_fact_ = false;

    // Held by pointer so a pattern rebuild starts from a fresh solver (Eigen's solvers are not
    // copy-assignable). Both use AMD in 64-bit indices (amd_ordering.hpp).
    using LltType  = SimplicialLLT64<SpMat>;
    using LdltType = SimplicialLDLT64<SpMat>;
    struct CholSolver {
        SpMat P;
        std::unique_ptr<LltType> llt;
    };
    struct LdltSolver {
        SpMat P_hat;
        std::unique_ptr<LdltType> ldlt;
    };
    std::variant<std::monostate, CholSolver, LdltSolver> active_solver_;

    int n_act_ = 0;
    Eigen::ComputationInfo info_ = Eigen::Success;
    int fact_count_ = 0;
    int fact_count_at_arm_ = 0; // fact_count_ snapshot taken by arm(); see consume_fact_count_delta()

    std::vector<int> ldlt_act_idx_;
    std::vector<Triplet> ldlt_build_trips_;
    Vec E_diag_; // factorize_by_chol()'s E diagonal (1/H_diag on active_K, 0 elsewhere)

    // Diagonal entries' indices into sol.P/sol.P_hat's valuePtr(), for the mu/rho-only patch paths:
    // O(1) writes instead of coeffRef()'s O(log nnz) search.
    std::vector<int> diag_idx_chol_;
    std::vector<int> ldlt_diag_top_idx_;
    std::vector<int> ldlt_diag_bot_idx_;

    // ------ SMW low-rank update ------
    // Control & failure tracking
    bool skip_smw_        = false;
    bool use_smw_         = false;
    int  smw_count_       = 0;
    int  smw_fail_streak_ = 0;
    int  smw_fail_total_  = 0;
    static constexpr int    kMaxSmwFailStreak = 5;
    static constexpr int    kSmwRankThreshold = 50; // try_build_smw() rejects rank > this; rank 0 reuses the snapshot.
    SmwRejectReason smw_last_reject_reason_ = SmwRejectReason::None;
    int             smw_last_rank_          = 0;

    // Snapshots from last full rebuild (input to try_build_smw)
    bool    has_snapshot_ = false; // true once snapshot_state() has run at least once
    bool    snapshot_wiped_by_fail_streak_ = false; // true iff record_smw_rebuild() wiped G_old_ --
                                                     // distinct from a legitimately empty (0-row) G_old_
    SpMat   G_old_;
    Vec     H_diag_old_;
    BoolArr active_K_old_;
    BoolArr active_W_old_;
    T       mu_old_  = T(1); // mu at the time of the snapshot; see smw_gate_open()'s mu_ != mu_old_ check.
    T       rho_old_ = T(1); // rho at the time of the snapshot; see smw_gate_open()'s rho_ != rho_old_ check.

    // Setup results (output of try_build_smw, consumed by solve)
    int s_old_ = 0, h_ = 0, p_ = 0, q_ = 0;
    std::vector<int> deleted_old_rows_;
    std::vector<int> retained_old_rows_;
    std::vector<int> retained_new_rows_;
    std::vector<int> added_new_rows_;
    std::vector<int> added_W_src_;
    std::vector<int> delta_K_idx_;
    Mat V_plus_;
    Mat Y_all_;
    Eigen::FullPivLU<Mat> S_lambda_lu_; // of the equilibrated capacitance matrix D S_Lambda D
    Vec S_lambda_eq_scale_;             // D's diagonal

    // Scratch buffers reused across try_build_smw()
    Vec smw_e_new_b_;              // sized N = G_old_.cols(); build_capacitance_setup() scratch
    std::vector<int> smw_touched_; // touched-column scratch paired with smw_e_new_b_ (no zero invariant needed)
    Vec smw_tmp_;                  // sized s_old_; compute_y_all() E_-/U basis-vector scratch
    Vec smw_ldlt_padded_;          // sized n_act_ + s_old_; compute_y_all() LDLT padded-RHS scratch

    // ------ Application: solve-time working storage ------
    mutable Vec r_pad_, u_base_, Lambda_all_, lambda_work_, z_base_, z_new_, direct_result_;
    mutable Vec ldlt_rhs_;
};
