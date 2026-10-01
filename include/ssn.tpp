#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <limits>
#include <cmath>
#include <Eigen/SparseCholesky>
#include <cassert>

template <typename T>
const typename SSN<T>::Vec& SSN<T>::compute_grad_Lagrangian(const Vec& x_new, const Vec& y2_new, const Vec& Ax_new, const Vec& Bx_new) {
    compute_dist_box(z / mu + x_new, lx, ux, grad_dist_K_);
    compute_dist_box(Bx_new + ((1 - alpha) * y2_new - y2) / mu, lw, uw, grad_dist_W_);
    grad_res_p_.noalias() = Ax_new - b;

    grad_Atr_resp_.noalias()  = A_tr * grad_res_p_;
    grad_Btr_distW_.noalias() = B_tr * grad_dist_W_;
    grad_A_tr_y_.noalias() = mu * grad_Atr_resp_ + (mu / alpha) * grad_Btr_distW_
                            + mu * grad_dist_K_ - A_tr_y1_;

    if (Q_info == 0) {
        grad_L_.head(N).noalias() = c + grad_A_tr_y_ + (x_new - x) / rho;
    } else {
        grad_Qx_.noalias() = Q_diag.cwiseProduct(x_new);
        grad_L_.head(N).noalias() = c + grad_Qx_ + grad_A_tr_y_ + (x_new - x) / rho;
    }
    grad_L_.tail(l).noalias() = ((1 - alpha) / alpha) * grad_dist_W_ + ((1 - alpha) / mu) * y2_new;

    return grad_L_;
}

template <typename T>
T SSN<T>::compute_grad_Lagrangian_unscaled_inf_norm(const Vec& grad_L) {
    // Compute the infinity norm of the unscaled gradient of the Lagrangian.
    T x_block  = inf_norm(grad_L.head(N).cwiseProduct(D2_ext_inv));
    T y2_block = inf_norm(grad_L.tail(l).cwiseProduct(D1B_diag_inv));
    return std::max(x_block, y2_block);
}

template <typename T>
void SSN<T>::split_by_mask(const Vec& u, const BoolArr& mask, Vec& u_sel, Vec& u_unsel) {
    int i_sel = 0;
    int i_unsel = 0;
    for (int i = 0; i < mask.size(); ++i) {
        if (mask(i)) {
            u_sel(i_sel++) = u(i);
        } else {
            u_unsel(i_unsel++) = u(i);
        }
    }
}

template <typename T>
void SSN<T>::rebuild_G() {
    using RIt = typename RowMajorSpMat::InnerIterator;

    const int n_act   = n_active_W;
    const int n_inact = l - n_act;

    // Partitioning B into active and inactive.
    B_inact_trips_.clear();
    B_inact_trips_.reserve(B_rm.nonZeros());

    // G = [A; active rows of B]
    G_trips_.clear();
    G_trips_.reserve(G_A_trips_.size() + B_rm.nonZeros());
    G_trips_.insert(G_trips_.end(), G_A_trips_.begin(), G_A_trips_.end());

    int i_act = 0, i_inact = 0;
    for (int i = 0; i < l; ++i) {
        if (active_W(i)) {
            for (RIt it(B_rm, i); it; ++it)
                G_trips_.emplace_back(M + i_act, it.col(), it.value());
            ++i_act;
        } else {
            for (RIt it(B_rm, i); it; ++it)
                B_inact_trips_.emplace_back(i_inact, it.col(), it.value());
            ++i_inact;
        }
    }

    B_inactive_W.resize(n_inact, N);
    B_inactive_W.setFromTriplets(B_inact_trips_.begin(), B_inact_trips_.end());
    B_inactive_W.makeCompressed();

    G.resize(M + n_act, N);
    G.setFromTriplets(G_trips_.begin(), G_trips_.end());
    G.makeCompressed();
    G_tr = G.transpose();
}

template <typename T>
void SSN<T>::retrieve_row_order(const Vec& u_sel, const Vec& u_unsel, const BoolArr& mask, Vec& out) {
    assert(u_sel.size() == mask.count());
    assert(u_unsel.size() == mask.size() - mask.count());

    out.resize(mask.size());
    int i_sel = 0, i_unsel = 0;
    for (int i = 0; i < mask.size(); ++i)
        out(i) = mask(i) ? u_sel(i_sel++) : u_unsel(i_unsel++);
}

template <typename T>
bool SSN<T>::choose_schur_ldlt(const SpMat& G, const BoolArr& active_K) {
    /*
    Compare the estimated total work required to factorize the KKT matrix K and the Schur complement S.
    Ratio = (s / (s+t)) * (|K|^2 / |S|^2), where s = G.rows() and t = n_act_K.
    |K| = nnz in the KKT matrix [-H_act_K, G_act_K^T; G_act_K, (1/mu)I] (active_K columns only).
    |S| is overestimated via the densest active_K column and the pigeonhole principle.
    Returns true (prefer LDLT on K) when ratio < kSchurLdltRatioThreshold.
    The direct solver passes an all-true mask, since the exact K and S contain every column.
    */
    const int s = G.rows();
    const int n_cols = G.cols();

    // Count active_K columns (t), their total nnz, and the densest one (hat_k, G_hat).
    long long t = 0, G_act_nnz = 0;
    int G_hat = 0, hat_k = -1;
    for (int k = 0; k < n_cols; ++k) {
        if (!active_K(k)) continue;
        ++t;
        int a_k = G.isCompressed()
                ? G.outerIndexPtr()[k + 1] - G.outerIndexPtr()[k] // nnz in column k
                : (int)G.col(k).nonZeros();
        G_act_nnz += a_k;
        if (a_k > G_hat) { G_hat = a_k; hat_k = k; }
    }

    const long long K_nnz = (t + s) + 2LL * G_act_nnz;

    // Overestimate |S|: s (diagonal) + off-diagonal from densest block + pigeonhole corrections.
    long long S_nnz = (long long)s + (long long)G_hat * G_hat - G_hat;
    for (int k = 0; k < n_cols; ++k) {
        if (!active_K(k) || k == hat_k) continue;
        int a_k = G.isCompressed()
                ? G.outerIndexPtr()[k + 1] - G.outerIndexPtr()[k]
                : (int)G.col(k).nonZeros();
        int o_k = std::max(0, G_hat + a_k - s); // Guaranteed overlap by pigeonhole
        S_nnz += (long long)a_k * a_k
                   - a_k
                   - (long long) o_k * o_k
                   + o_k;
    }

    if (S_nnz <= 0) return true;

    const double ratio = ((double)s / (t + s))
                       * ((double)K_nnz / S_nnz)
                       * ((double)K_nnz / S_nnz);

    return ratio < kSchurLdltRatioThreshold;
}

template <typename T>
typename SSN<T>::Vec SSN<T>::solve_using_cg(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const Vec& H_diag_inv,
                                            const BoolArr& active_K, const Vec& r1, const Vec& r2,
                                            T mu, T tol, int max_iter, bool update_prec, bool prec_pattern_changed,
                                            bool schur_use_ldlt) {
    using Vec = typename SSN<T>::Vec;

    // Direct solver: throughout in direct mode, and permanently once PCG has failed.
    if (direct_solve || pcg_failed)
        return solve_direct(G, G_tr, H_diag, H_diag_inv, r1, r2);

    const int s = G.rows();
    const int n = G.cols();

    // Schur complement system: S dy = G H_inv G^T dy + (1/mu) dy = G H_inv r1 + r2.
    SchurOperator<T> S(G, G_tr, H_diag_inv, mu);
    cg_Hinv_r1_.noalias() = H_diag_inv.cwiseProduct(r1);
    cg_rhs_.noalias() = G * cg_Hinv_r1_ + r2;
    const Vec& rhs = cg_rhs_;

    cg.setTolerance(tol);
    cg.setMaxIterations(max_iter);

    // Set up preconditioner and call cg.compute().
    // force_rebuild=true skips SMW and recomputes G E G^T from scratch.
    auto setup_prec = [&](bool force_rebuild) {
        SSN_TIMER_BLOCK(timer_prec_setup);
        cg.preconditioner().arm(G, G_tr, H_diag, active_K, active_W, B_rm, mu, rho,
                                update_prec, prec_pattern_changed, schur_use_ldlt, force_rebuild);
#if SSN_ENABLE_TIMERS
        // TIMER: snapshot SchurPreconditioner's cumulative phase timers so we can diff after compute().
        const double prec_assembly_before  = cg.preconditioner().assembly_time();
        const double prec_analyze_before   = cg.preconditioner().analyze_time();
        const double prec_factorize_before = cg.preconditioner().factorize_time();
#endif
        cg.compute(S);
        fact      += cg.preconditioner().consume_fact_count_delta();
        smw_count  = cg.preconditioner().smw_count();
#if SSN_ENABLE_TIMERS
        timer_prec_assembly  += cg.preconditioner().assembly_time()  - prec_assembly_before;
        timer_prec_analyze   += cg.preconditioner().analyze_time()   - prec_analyze_before;
        timer_prec_factorize += cg.preconditioner().factorize_time() - prec_factorize_before;
#endif
    };

    // Run preconditioned CG.
    // Returns false and increments krylov_fail on preconditioner failure or solver non-convergence with error > 1e-10.
    // If max_iter is reached but the error is <= 1e-10, the direction is accepted.
    auto attempt_solve = [&](Vec& dy_out) -> bool {
        if (cg.preconditioner().info() != Eigen::Success) {
            std::cout << "[PCG] CG failed due to preconditioner failure.\n";
            krylov_fail++;
            return false;
        }

        Vec dy_;
        bool warm_start = (prev_dy_.size() == s); // prev_dy_ is empty on the first SSN iteration or if the active set changed.
        {
            SSN_TIMER_BLOCK(timer_krylov_solve);
            if (warm_start) {
                dy_ = cg.solveWithGuess(rhs, prev_dy_);
            } else {
                dy_ = cg.solve(rhs);
            }
        }
        krylov_iter += cg.iterations();

        T err = cg.error();
        if (cg.info() != Eigen::Success) {
            if (err > T(1e-10)) { // Acceptable error threshold for PCG failure.
                krylov_fail++;
                return false;
            }
        }
        dy_out = std::move(dy_);
        return true;
    };

    // Release the PCG state (unused anymore) when the solve permanently switches to the direct solver.
    auto switch_to_direct = [&]() {
        krylov_converged = false;
        pcg_failed = true;
        cg.preconditioner().release();
        prev_dy_.resize(0);
        cg_Hinv_r1_.resize(0);
        cg_rhs_.resize(0);
        cg_dx_.resize(0);
    };

    // Set up and attempt to solve by PCG.
    // bad_alloc can come from build() (P_base_ = G E G^T) or from Eigen's CG iteration internals;
    // in either case, fall back to the direct solver.
    Vec dy_;
    bool ok = false;
    try {
        setup_prec(false);
        ok = attempt_solve(dy_);
    } catch (const std::bad_alloc&) {
        std::cout << "[PCG] CG failed due to bad_alloc.\n";
        krylov_fail++;
        switch_to_direct();
        return solve_direct(G, G_tr, H_diag, H_diag_inv, r1, r2);
    }

    // If CG succeeded and the preconditioner used SMW, reset the SMW fail streak.
    if (ok && cg.preconditioner().used_smw())
        cg.preconditioner().reset_smw_fail_streak();

    // If CG failed and the preconditioner used SMW, retry after a full rebuild.
    // If it already used a full factorization: fall back to the direct solver.
    if (!ok && cg.preconditioner().should_retry_after_failure()) {
        try {
            setup_prec(true);
            ok = attempt_solve(dy_);
        } catch (const std::bad_alloc&) {
            std::cout << "[PCG] CG failed due to bad_alloc.\n";
            krylov_fail++;
            switch_to_direct();
            return solve_direct(G, G_tr, H_diag, H_diag_inv, r1, r2);
        }
    }

    // If CG failed again, fall back to the direct solver.
    if (!ok) {
        std::cout << "[PCG] CG failed, falling back to the direct solver (LDLT on the KKT system or Cholesky on its Schur complement).\n";
        switch_to_direct();
        return solve_direct(G, G_tr, H_diag, H_diag_inv, r1, r2);
    }

    prev_dy_ = dy_;

    // Recover dx = H_inv (G^T dy_ - r1).
    cg_dx_.noalias() = H_diag_inv.cwiseProduct(G_tr * dy_ - r1);

    Vec result(n + s);
    result.head(n) = cg_dx_;
    result.tail(s) = dy_;
    return result;
}

template <typename T>
void SSN<T>::decide_direct_system(const SpMat& G) {
    // Same cost model and threshold as the preconditioner's choice, but over all columns:
    // the exact K and S contain every column, not just the active_K ones.
    if (all_cols_.size() != G.cols()) all_cols_ = BoolArr::Constant(G.cols(), true);
    direct_use_ldlt = choose_schur_ldlt(G, all_cols_);
    ++direct_ldlt_decisions_made_;
    if (direct_use_ldlt) release_chol(); // no-op unless switching away from the Schur complement
}

template <typename T> // Direct solver: dispatches to LDLT on K or Cholesky on S
typename SSN<T>::Vec SSN<T>::solve_direct(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const Vec& H_diag_inv,
                                          const Vec& r1, const Vec& r2) {
    // No rows in G: K = -H, so dx = -H^{-1} r1 exactly and there is nothing to factorize.
    if (G.rows() == 0) return -H_diag_inv.cwiseProduct(r1);

    // First direct solve without a decision yet (right after PCG failed): decide now.
    if (direct_ldlt_decisions_made_ == 0) decide_direct_system(G);

    if (direct_use_ldlt || schur_chol_failed_)
        return solve_using_ldlt(G, H_diag, r1, r2);
    return solve_using_chol(G, G_tr, H_diag, H_diag_inv, r1, r2);
}

template <typename T> // Direct solver: LDLT on the KKT system K
typename SSN<T>::Vec SSN<T>::solve_using_ldlt(const SpMat& G, const Vec& H_diag, const Vec& r1, const Vec& r2) {
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;

    const int s = G.rows();
    const int n = G.cols();
    const int N_tot = n + s;

    ldlt_solve_rhs_.resize(N_tot);
    ldlt_solve_rhs_ << r1, r2;

    // If stored factorization is for a different system size, force re-analyze.
    if (!ldlt_pattern_dirty_ && ldlt_.rows() > 0 && ldlt_.rows() != static_cast<Eigen::Index>(N_tot)) {
        ldlt_pattern_dirty_ = true;
        ldlt_numeric_dirty_ = true;
        K_ldlt_built_       = false;
    }

    if (ldlt_pattern_dirty_ || ldlt_numeric_dirty_) {
        if (ldlt_pattern_dirty_ || !K_ldlt_built_) {
            // Full rebuild: G's sparsity changed (active_W changed) or first call.
            // Assemble K = [-H, G^T; G, (1/mu) I] from triplets and cache it.
            ldlt_trip_.clear();
            ldlt_trip_.reserve(N_tot + 2 * G.nonZeros());

            for (int i = 0; i < n; ++i)
                ldlt_trip_.emplace_back(i, i, -H_diag(i));
            const T mu_inv = T(1) / mu;
            for (int i = 0; i < s; ++i)
                ldlt_trip_.emplace_back(n + i, n + i, mu_inv);
            for (int col = 0; col < G.outerSize(); ++col)
                for (typename SpMat::InnerIterator it(G, col); it; ++it) {
                    ldlt_trip_.emplace_back(n + it.row(), it.col(), it.value());
                    ldlt_trip_.emplace_back(it.col(), n + it.row(), it.value());
                }

            K_ldlt_.resize(N_tot, N_tot);
            K_ldlt_.setFromTriplets(ldlt_trip_.begin(), ldlt_trip_.end());
            K_ldlt_.makeCompressed();
            K_ldlt_built_ = true;

            // Cache each diagonal's flat storage index so the patch path below can write 
            // via valuePtr()[idx] (O(1)) instead of coeffRef(i,i) (O(log nnz), binary search).
            ldlt_diag_top_idx_.resize(n);
            for (int i = 0; i < n; ++i)
                ldlt_diag_top_idx_[i] = static_cast<int>(&K_ldlt_.coeffRef(i, i) - K_ldlt_.valuePtr());
            ldlt_diag_bot_idx_.resize(s);
            for (int i = 0; i < s; ++i)
                ldlt_diag_bot_idx_[i] = static_cast<int>(&K_ldlt_.coeffRef(n + i, n + i) - K_ldlt_.valuePtr());

            if (ldlt_pattern_dirty_) {
                SSN_TIMER_BLOCK(timer_ldlt_analyze);
                ldlt_.analyzePattern(K_ldlt_);
                ldlt_pattern_dirty_ = false;
            }
        } else {
            // Pattern unchanged (active_W same): only diagonal values changed (H_diag, mu).
            // Update top-left and bottom-right diagonal entries in-place via the cached flat indices above;
            // G blocks stay.
            for (int i = 0; i < n; ++i)
                K_ldlt_.valuePtr()[ldlt_diag_top_idx_[i]] = -H_diag(i);
            const T mu_inv = T(1) / mu;
            for (int i = 0; i < s; ++i)
                K_ldlt_.valuePtr()[ldlt_diag_bot_idx_[i]] = mu_inv;
        }

        {
            SSN_TIMER_BLOCK(timer_ldlt_factorize);
            ldlt_.factorize(K_ldlt_);
        }
        if (ldlt_.info() != Eigen::Success)
            throw std::runtime_error("LDLT factorization of the augmented Lagrangian system failed.");
        ldlt_numeric_dirty_ = false;
        fact++;
        kkt_ldlt_fact++;
    }

    Vec result;
    {
        SSN_TIMER_BLOCK(timer_ldlt_solve);
        result = ldlt_.solve(ldlt_solve_rhs_);
    }
    if (ldlt_.info() != Eigen::Success)
        throw std::runtime_error("Solving the augmented Lagrangian system via LDLT failed.");
    return result;
}

template <typename T> // Direct solver: Cholesky on the Schur complement S
typename SSN<T>::Vec SSN<T>::solve_using_chol(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const Vec& H_diag_inv,
                                              const Vec& r1, const Vec& r2) {
    /*
    Solve K [dx; dy] = [r1; r2] through the Schur complement S = G H^{-1} G^T + (1/mu)I (SPD):
    S dy = G H^{-1} r1 + r2, then dx = H^{-1} (G^T dy - r1).
    S's sparsity follows G's structure only, so a K-only or mu/rho change reuses the symbolic analysis
    and only reassembles and refactorizes S. If Cholesky fails, fall back to LDLT on K for the rest of the run.
    */
    using Vec = typename SSN<T>::Vec;
    using SpMat = typename SSN<T>::SpMat;

    const int s = G.rows();
    const int n = G.cols();

    // If the stored factorization is for a different system size, force re-analyze.
    if (chol_ && chol_->rows() != s) {
        chol_pattern_dirty_ = true;
        chol_numeric_dirty_ = true;
    }

    bool ok = true;
    const char* reason = "";
    if (!chol_ || chol_pattern_dirty_ || chol_numeric_dirty_) {
        try {
            {
                SSN_TIMER_BLOCK(timer_chol_assembly);
                // Free the old S first: the temporaries of the product below are the memory peak.
                SpMat().swap(S_chol_);

                // Lower triangle of S = (G H^{-1/2}) (G H^{-1/2})^T + (1/mu)I; SimplicialLLT only reads that.
                // No prune: H_diag_inv > 0, so S's pattern depends only on G's structure.
                SpMat G_scaled = G;
                for (int k = 0; k < G_scaled.outerSize(); ++k) {
                    const T scale = std::sqrt(H_diag_inv(k));
                    for (typename SpMat::InnerIterator it(G_scaled, k); it; ++it)
                        it.valueRef() *= scale;
                }
                S_chol_.template selfadjointView<Eigen::Lower>().rankUpdate(G_scaled, T(0));

                SpMat mu_diag(s, s);
                mu_diag.setIdentity();
                mu_diag *= T(1) / mu;
                S_chol_ += mu_diag;
                S_chol_.makeCompressed();
            }

            // A changed nonzero count means a changed pattern, even if no flag said so.
            if (S_chol_.nonZeros() != chol_nnz_at_analyze_) chol_pattern_dirty_ = true;

            if (!chol_ || chol_pattern_dirty_) {
                SSN_TIMER_BLOCK(timer_chol_analyze);
                chol_ = std::make_unique<Eigen::SimplicialLLT<SpMat>>();
                chol_->analyzePattern(S_chol_);
                chol_nnz_at_analyze_ = S_chol_.nonZeros();
                chol_pattern_dirty_ = false;
            }
            {
                SSN_TIMER_BLOCK(timer_chol_factorize);
                chol_->factorize(S_chol_);
            }
            fact++;
            schur_chol_fact++;
            if (chol_->info() != Eigen::Success) {
                ok = false;
                reason = "not numerically positive definite";
            } else {
                chol_numeric_dirty_ = false;
            }
        } catch (const std::bad_alloc&) {
            ok = false;
            reason = "bad_alloc";
        }
    }

    if (!ok) {
        std::cout << "[Direct] Cholesky on the Schur complement failed (" << reason
                  << "), falling back to LDLT on the KKT system.\n";
        schur_chol_failed_ = true;
        release_chol();
        return solve_using_ldlt(G, H_diag, r1, r2);
    }

    // Solve S dy = G H^{-1} r1 + r2, then recover dx = H^{-1} (G^T dy - r1).
    chol_Hinv_r1_.noalias() = H_diag_inv.cwiseProduct(r1);
    chol_rhs_.noalias() = G * chol_Hinv_r1_ + r2;
    Vec dy;
    {
        SSN_TIMER_BLOCK(timer_chol_solve);
        dy = chol_->solve(chol_rhs_);
    }
    if (chol_->info() != Eigen::Success)
        throw std::runtime_error("Solving the Schur complement system via Cholesky failed.");

    Vec result(n + s);
    result.head(n) = H_diag_inv.cwiseProduct(G_tr * dy - r1);
    result.tail(s) = dy;
    return result;
}

template <typename T>
void SSN<T>::release_chol() {
    // Free the Schur complement and its factorization; the next solve_using_chol() rebuilds both.
    chol_.reset();
    SpMat().swap(S_chol_);
    chol_Hinv_r1_.resize(0);
    chol_rhs_.resize(0);
    chol_pattern_dirty_ = true;
    chol_numeric_dirty_ = true;
    chol_nnz_at_analyze_ = -1;
}

template <typename T>
SsnLineSearchParams<T> SSN<T>::make_line_search_params() {
    // Cached once per SSN iteration as Ax_ssn_ doesn't move across a line-search attempt
    // and its steepest-descent retry. 
    grad_res_p_.noalias()    = Ax_ssn_ - b;
    grad_Atr_resp_.noalias() = A_tr * grad_res_p_;

    return SsnLineSearchParams<T>{
        mu, rho, alpha, eps_zero, eps_direction, inf,
        Q_info, N, l,
        lx, ux, lw, uw,
        z, y2, x,
        c, A_tr_y1_, Q_diag, grad_Atr_resp_, b,
    };
}

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
                    std::vector<SsnBreakpoint<T>>& breakpoints_scratch) {
    /*
    psi'(t) = <∇ M(u + t du), du>
            = eta t + zeta + mu <dist_K (s + t dx), dx> + <mu / alpha dv, dist_W (v + t dv)>,
    where eta  = <(Q + mu A^T A + I_n/rho) dx, dx> + (1 - alpha) / mu ||dy2||^2,
          zeta = <c + Q x_curr - A^T y1 + mu A^T (A x_curr - b) + (x_curr - x) / rho, dx> + (1 - alpha) / mu <y2_curr, dy2>,
          s    = z / mu + x_curr,
          v    = B x_curr + ((1 - alpha) y2_curr - y2) / mu,
          dv   = B dx + (1 - alpha) / mu dy2.
    Compute all breakpoints t and corresponding slope changes of psi, and sort them in increasing order.
    Write psi'(t) = p + m (t - t_prev).
    For each breakpoint t, if psi'(t) >= 0, return t = t_prev - p / m;
    otherwise, set p = psi'(t), t_prev = t and continue.
    */
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Breakpoint = SsnBreakpoint<T>;

    const Vec& Bx = Bx_curr;
    const T mu = p.mu, rho = p.rho, alpha = p.alpha, eps_zero = p.eps_zero,
            eps_direction = p.eps_direction, inf = p.inf;
    const int Q_info = p.Q_info, N = p.N, l = p.l;
    const Vec &lx = p.lx, &ux = p.ux, &lw = p.lw, &uw = p.uw;
    const Vec &z = p.z, &y2 = p.y2, &x = p.x, &c = p.c, &A_tr_y1_ = p.A_tr_y1, &Q_diag = p.Q_diag, &b = p.b;
    const Vec& grad_Atr_resp = p.grad_Atr_resp;

    ls_s_scratch.noalias()  = z / mu + x_curr;
    ls_v_scratch.noalias()  = Bx + ((1 - alpha) * y2_curr - y2) / mu;
    ls_dv_scratch.noalias() = Bdx + (1 - alpha) / mu * dy2;

    // eta: smooth linear term in psi(t)
    T eta = T(0);
    if (Q_info != 0) {
        eta += dx.dot(Q_diag.cwiseProduct(dx));
    }
    eta += mu * (Adx).squaredNorm();
    eta += (1 / rho) * dx.squaredNorm();
    eta += (1 - alpha) / mu * dy2.squaredNorm();

    // zeta: constant term in psi(t)
    T zeta = T(0);
    if (Q_info != 0) {
        zeta += dx.dot(Q_diag.cwiseProduct(x_curr));
    }
    zeta += dx.dot(c - A_tr_y1_ + mu * grad_Atr_resp + (x_curr - x) / rho);
    zeta += (1 - alpha) / mu * dy2.dot(y2_curr);

    // Reuse the breakpoints scratch buffer.
    breakpoints_scratch.clear();
    if (breakpoints_scratch.capacity() < static_cast<size_t>(2 * (N + l)))
        breakpoints_scratch.reserve(2 * (N + l));
    auto& breakpoints = breakpoints_scratch;

    // Build K breakpoints and accumulate initial slope m.
    T m = eta;
    for (int i = 0; i < N; ++i) {
        const T s_i = ls_s_scratch(i);
        const T dx_i = dx(i);
        const T li = lx(i), ui = ux(i);

        const T s_eps = s_i + eps_zero * dx_i; // position just after tau = 0
        if ((li > -inf && s_eps < li) || (ui < inf && s_eps > ui))
            m += mu * dx_i * dx_i;

        if (std::abs(dx_i) < eps_direction) continue;

        const T change = mu * dx_i * dx_i;
        if (li > -inf) {
            const T t_l = (li - s_i) / dx_i;
            if (t_l > eps_zero) breakpoints.push_back({t_l, dx_i > 0 ? -change : +change});
        }
        if (ui < inf) {
            const T t_u = (ui - s_i) / dx_i;
            if (t_u > eps_zero) breakpoints.push_back({t_u, dx_i > 0 ? +change : -change});
        }
    }

    // Build W breakpoints and accumulate initial slope m.
    for (int i = 0; i < l; ++i) {
        const T v_i = ls_v_scratch(i);
        const T dv_i = ls_dv_scratch(i);
        const T li = lw(i), ui = uw(i);

        const T v_eps = v_i + eps_zero * dv_i; // position just after tau = 0
        if ((li > -inf && v_eps < li) || (ui < inf && v_eps > ui))
            m += mu / alpha * dv_i * dv_i;

        if (std::abs(dv_i) < eps_direction) continue;

        const T change = mu / alpha * dv_i * dv_i;
        if (li > -inf) {
            const T t_l = (li - v_i) / dv_i;
            if (t_l > eps_zero) breakpoints.push_back({t_l, dv_i > 0 ? -change : +change});
        }
        if (ui < inf) {
            const T t_u = (ui - v_i) / dv_i;
            if (t_u > eps_zero) breakpoints.push_back({t_u, dv_i > 0 ? +change : -change});
        }
    }

    // If there is no breakpoint and the direction (dx, dy2) is nearly zero, return a full step (i.e. trivial case).
    // Note: eta is the weighted squared norm of the direction.
    if (breakpoints.empty() && eta < eps_zero) return T(1);
    // Otherwise (no breakpoints but a non-negligible direction, e.g. no finite box bounds at all), 
    // fall through to the psi'(0) check and solve psi(t) = eta/2 t^2 + zeta t + const exactly.

    // Sort breakpoints by t in ascending order.
    std::sort(breakpoints.begin(), breakpoints.end(), [](const Breakpoint& a, const Breakpoint& b){ return a.t < b.t; });

    // Merge entries with identical t.
    const T merge_tol = T(100) * std::numeric_limits<T>::epsilon();
    int n_uniq = 0;
    for (size_t i = 0; i < breakpoints.size(); ) {
        T t = breakpoints[i].t;
        T slope_change_sum = T(0);
        while (i < breakpoints.size() && std::abs(breakpoints[i].t - t) < merge_tol * std::max<T>(1, std::abs(t))) {
            slope_change_sum += breakpoints[i].slope_change;
            ++i;
        }
        breakpoints[n_uniq++] = {t, slope_change_sum};
    }
    breakpoints.resize(n_uniq);

    // Check if psi'(0) >= 0; if so, return 0 (no crossing, linesearch failed).
    T p_val = zeta;
    p_val += mu * dist_K_s.dot(dx);
    p_val += mu / alpha * dist_W_v.dot(ls_dv_scratch);
    if (p_val >= T(0)) return T(0);

    // If psi'(0) < 0, check at every breakpoint t.
    T t_prev = T(0);
    for (const Breakpoint& bp : breakpoints) {
        T t = bp.t;
        T p_t = p_val + m * (t - t_prev);
        if (p_t >= T(0)) return t_prev - p_val / std::max(m, eps_zero * (T(1) + eta));

        // Cross the breakpoint(s).
        t_prev = t;
        p_val = p_t;
        m += bp.slope_change;
    }

    // Checking the last breakpoint; m should be >= 0.
    if (m >= T(0)) {
        return t_prev - p_val / std::max(m, eps_zero * (T(1) + eta));
    }
    return T(0); // safeguard: m truly negative
}

template <typename T>
typename SSN<T>::PrepResult SSN<T>::prepare_newton_system() {
    {
    SSN_TIMER_BLOCK(timer_prep);
    u_.noalias() = z / mu + x_cur_;
    v_ = Bx_ssn_ + ((1 - alpha) * y2_cur_ - y2) / mu;

    // Clarke subgradient and distance for K and W
    compute_subgrad_and_dist(u_, lx, ux, false, new_active_K_, dist_K_u_);
    compute_subgrad_and_dist(v_, lw, uw, true,  new_active_W_, dist_W_v_);
    // W is active when v lies OUTSIDE [lw,uw]; the subgrad just written is true when INSIDE
    // (include_bd=true), so invert it to land new_active_W_ in the same polarity as active_W.
    new_active_W_ = (new_active_W_ == false);
    }

    // Detect active-set changes; recompute on any single change.
    bool first_ssn_iter = (active_K.size() == 0);
    ActiveSetDelta delta{
        /*k_changed=*/first_ssn_iter || (active_K != new_active_K_).any(),
        /*w_changed=*/first_ssn_iter || (active_W != new_active_W_).any(),
    };

    bool update_prec = delta.k_changed || delta.w_changed; // true means rebuilding prec is needed.
    bool prec_pattern_changed = delta.k_changed || delta.w_changed; // true means analyzePattern() is needed.

    // Recompute H if active_K changed or mu/rho drifted since H_diag was last built.
    bool recompute_H = delta.k_changed || (mu != H_diag_mu_) || (rho != H_diag_rho_);

    // These are for the direct solver (LDLT on K, Cholesky on S): their sparsity follows G (active_W),
    // and their values change exactly when H_diag is recomputed or G is rebuilt.
    if (delta.w_changed) ldlt_pattern_dirty_ = chol_pattern_dirty_ = true;
    if (recompute_H || delta.w_changed) ldlt_numeric_dirty_ = chol_numeric_dirty_ = true;

    if (recompute_H) {
        SSN_TIMER_BLOCK(timer_prep);
        if (delta.k_changed) {
            active_K = new_active_K_;
        }

        // H = Q + mu(I_N - P_K) + I_N / rho
        if (Q_info == 0) {
            H_diag = mu * (T(1) - active_K.cast<T>()) + T(1)/rho;
        } else {
            H_diag = Q_diag.array() + mu * (T(1) - active_K.cast<T>()) +  T(1)/rho;
        }
        H_diag = H_diag.cwiseMax(eps_zero); // safeguard for non-positive diagonal entries
        H_diag_inv = H_diag.cwiseInverse();
        H_diag_mu_  = mu;
        H_diag_rho_ = rho;
    }

    // If active_W changed, recompute G.
    if (delta.w_changed) {
        SSN_TIMER_BLOCK(timer_prep);
        prev_dy_.resize(0); // No warm-starting for CG

        active_W = new_active_W_;
        n_active_W = active_W.count();
        n_inactive_W = l - n_active_W;

        rebuild_G(); // Rebuild G = [A; active rows of B], B_inactive_W, G_tr.
    }

    {
    SSN_TIMER_BLOCK(timer_prep);
    // Compute dy2 in inactive_W: dy2_inactive_W = - (mu / alpha) * dist_W(v)(inactive_W) - y2(inactive_W).
    split_by_mask(y2_cur_, active_W, y2_active_W_, y2_inactive_W_);
    split_by_mask(dist_W_v_, active_W, dist_W_v_active_, dist_W_v_inactive_);
    dy2_inactive_W_.head(n_inactive_W).noalias() =
        -(mu / alpha) * dist_W_v_inactive_.head(n_inactive_W) - y2_inactive_W_.head(n_inactive_W);

    // Compute the RHS vector.
    if (Q_info == 0) {
        r1_ = c + mu * dist_K_u_
             - B_tr * y2_cur_ - B_inactive_W.transpose() * dy2_inactive_W_.head(n_inactive_W)
             + (x_cur_ - x) / rho;
    } else {
        r1_ = c + Q_diag.cwiseProduct(x_cur_) + mu * dist_K_u_
             - B_tr * y2_cur_ - B_inactive_W.transpose() * dy2_inactive_W_.head(n_inactive_W)
             + (x_cur_ - x) / rho;
    }
    r2_.resize(M + n_active_W);
    r2_.head(M) = y1 / mu - Ax_ssn_ + b;
    r2_.tail(n_active_W) = -dist_W_v_active_.head(n_active_W) - (alpha / mu) * y2_active_W_.head(n_active_W);
    }

    // Determines the factorization method, each locked after its first 3 decisions:
    // for the direct solver (K or S, over all columns) once it is in use, otherwise for the PCG preconditioner.
    const bool use_direct = direct_solve || pcg_failed;
    if ((delta.k_changed || delta.w_changed) && use_direct && direct_ldlt_decisions_made_ < 3) {
        SSN_TIMER_BLOCK(timer_prep);
        decide_direct_system(G);
    } else if ((delta.k_changed || delta.w_changed) && !use_direct && schur_ldlt_decisions_made_ < 3) {
        SSN_TIMER_BLOCK(timer_prep);
        schur_use_ldlt = choose_schur_ldlt(G, active_K);
        ++schur_ldlt_decisions_made_;
    }

    return {update_prec, prec_pattern_changed};
}

template <typename T>
void SSN<T>::iterative_refine_dxdy() {
    // Iterative refinement: correct the residual of K [dx;dy] = [r1_;r2_], K = [-H, G^T; G, (1/mu)I].
    const int s = static_cast<int>(r2_.size());
    const T ref_norm = std::max(inf_norm(r1_), inf_norm(r2_));
    Vec rho1(N), rho2(s);

    for (int k = 0; k < refine_max_iter; ++k) {
        const auto dx_k = dxdy_.head(N);
        const auto dy_k = dxdy_.tail(s);
        Gtr_dy_.noalias() = G_tr * dy_k;
        G_dx_.noalias()   = G * dx_k;
        rho1 = r1_ + H_diag.cwiseProduct(dx_k) - Gtr_dy_;
        rho2 = r2_ - G_dx_ - dy_k / mu;

        const T res_norm = std::max(inf_norm(rho1), inf_norm(rho2));
        if (res_norm <= std::max(refine_rel_tol * ref_norm, refine_abs_tol)) break;

        prev_dy_.resize(0); // cold-start for iterative refinement
        Vec correction = solve_using_cg(G, G_tr, H_diag, H_diag_inv, active_K, rho1, rho2,
                                        mu, krylov_tol, krylov_max_in_iter, false, false, schur_use_ldlt);
        dxdy_ += correction;
        prev_dy_ = dxdy_.tail(s); // warm-start for the next SSN iteration's main solve
    }
}

template <typename T>
void SSN<T>::solve_newton_direction(bool update_prec, bool prec_pattern_changed) {
    SSN_TIMER_BLOCK(timer_linear_solve);
    // Solve for dx and dy2_active_W.
    dxdy_ = solve_using_cg(G, G_tr, H_diag, H_diag_inv, active_K, r1_, r2_, mu, krylov_tol, krylov_max_in_iter, update_prec, prec_pattern_changed, schur_use_ldlt);

    iterative_refine_dxdy();

    // Split dxdy_ into dx_ and dy2_active_W.
    dx_ = dxdy_.head(N);
    const auto dy2_active_W = dxdy_.tail(n_active_W);

    assert(dy2_active_W.size() == n_active_W);
    assert(dy2_inactive_W_.size() >= n_inactive_W); // fixed-capacity l, valid prefix n_inactive_W
    assert(active_W.size() == l);

    // Recover dy2.
    retrieve_row_order(dy2_active_W, dy2_inactive_W_.head(n_inactive_W), active_W, dy2_);
}

template <typename T>
typename SSN<T>::LineSearchResult SSN<T>::line_search_with_steepest_descent_fallback(T ssn_tol) {

    SSN_TIMER_BLOCK(timer_linesearch);
    const SsnLineSearchParams<T> line_search_params = make_line_search_params();

    Adx_.noalias() = A * dx_;
    Bdx_.noalias() = B * dx_;
    T tau = exact_line_search(line_search_params, x_cur_, y2_cur_, dx_, dy2_,
                                Ax_ssn_, Bx_ssn_, Adx_, Bdx_,
                                dist_K_u_, dist_W_v_,
                                ls_s_, ls_v_, ls_dv_, breakpoints_);

    // If linesearch found step size <= 0 with the Newton direction, ...
    if (tau <= T(0)) {
        const Vec& grad_L = compute_grad_Lagrangian(x_cur_, y2_cur_, Ax_ssn_, Bx_ssn_);
        T grad_norm = compute_grad_Lagrangian_unscaled_inf_norm(grad_L);
        // ... check if ||∇M|| is small enough to accept optimality.
        if (grad_norm <= T(5) * ssn_tol) {
            return {LineSearchOutcome::AcceptOptimal, T(0)};
        }

        // Retry linesearch with the steepest-descent direction.
        dx_  = -grad_L.head(N);
        dy2_ = -grad_L.tail(l);
        Adx_.noalias() = A * dx_;
        Bdx_.noalias() = B * dx_;
        tau = exact_line_search(line_search_params, x_cur_, y2_cur_, dx_, dy2_,
                                Ax_ssn_, Bx_ssn_, Adx_, Bdx_,
                                dist_K_u_, dist_W_v_,
                                ls_s_, ls_v_, ls_dv_, breakpoints_);

        // If linesearch still fails, exit SSN loop and let PMM adjust penalties.
        if (tau <= T(0)) {
            linesearch_fail++;
            return {LineSearchOutcome::Fail, T(0)};
        }
    }
    return {LineSearchOutcome::Proceed, tau};
}

template <typename T>
void SSN<T>::update_iterate(T tau, int ssn_iter_count) {
    SSN_TIMER_BLOCK(timer_state_update);
    x_cur_  += tau * dx_;
    y2_cur_ += tau * dy2_;

    if (ssn_iter_count % 5 == 4) { // Reset incremental drift every 5 iterations.
        Ax_ssn_.noalias() = A * x_cur_;
        Bx_ssn_.noalias() = B * x_cur_;
    } else {
        Ax_ssn_ += tau * Adx_;
        Bx_ssn_ += tau * Bdx_;
    }

    // Compute gradient of Lagrangian at current (x, y2).
    compute_grad_Lagrangian(x_cur_, y2_cur_, Ax_ssn_, Bx_ssn_);
    tol_achieved = compute_grad_Lagrangian_unscaled_inf_norm(grad_L_);
}

template <typename T>
std::optional<typename SSN<T>::TerminationStatus> SSN<T>::check_ssn_termination(T ssn_tol, int& stagnation, T& prev_tol_achieved) {
    // Check termination criterion.
    if (tol_achieved < ssn_tol) {
        return TerminationStatus::Optimal;
    }

    // Stagnation check: if ||∇M|| fails to meaningfully improve for 10 consecutive iterations,
    // exit early and let the PMM level adjust penalties.
    if (tol_achieved >= T(0.999) * prev_tol_achieved) {
        ++stagnation;
    } else {
        stagnation = 0;
    }
    prev_tol_achieved = tol_achieved;

    if (stagnation >= 10) {
        if (tol_achieved < T(5) * ssn_tol) {
            return TerminationStatus::Optimal;
        }
        return TerminationStatus::Stagnated; // ||∇M|| stagnated; not a confirmed optimum.
    }

    return std::nullopt; // Continue iterating.
}

template <typename T>
void SSN<T>::solve_ssn(const T ssn_tol) {
    /* ----------------------------------------------
    Structure:
    Let M(u), with u = (x, y2), be the proximal augmented Lagrangian associated with the subproblem of interest.
    Until (||∇M(u_{k_j})|| < eps), for some given eps, do:
        1) Compute a Clarke subgradient J of ∇M(u_{k_j})
           and solve J du = - ∇M(u_{k_j}) for the Newton direction du;
        2) Perform exact linesearch to determine the step size alpha;
           If the linesearch fails, use gradient descent and retry the linesearch;
        3) Update the variables;
        j = j + 1;
        If the SSN iteration stagnates for 10 consecutive iterations, terminate early.
    End
    ---------------------------------------------- */
    // Intialize iteration counter and set starting points.
    x_cur_ = x;
    y2_cur_ = y2;
    int _iter = 0;
    std::optional<TerminationStatus> _opt;
    T prev_tol_achieved = inf;
    int stagnation = 0;

    // Useful matvecs
    Ax_ssn_.noalias() = A * x_cur_;
    Bx_ssn_.noalias() = B * x_cur_;

    // SSN main loop
    while (_iter < ssn_max_in_iter) {
        if (interrupted_()) { _opt = TerminationStatus::Interrupted; break; }
        if (time_limit_exceeded_()) { _opt = TerminationStatus::TimeLimit; break; }
#if SSN_ENABLE_TIMERS
        // TIMER: reset per-phase accumulators for this SSN iteration.
        timer_prep = timer_linear_solve = timer_prec_setup = timer_krylov_solve = 0.0;
        timer_prec_assembly = timer_prec_analyze = timer_prec_factorize = 0.0;
        timer_ldlt_analyze = timer_ldlt_factorize = timer_ldlt_solve = 0.0;
        timer_chol_assembly = timer_chol_analyze = timer_chol_factorize = timer_chol_solve = 0.0;
        timer_linesearch = timer_state_update = 0.0;
#endif
        auto [update_prec, prec_pattern_changed] = prepare_newton_system();
        solve_newton_direction(update_prec, prec_pattern_changed);

        LineSearchResult ls = line_search_with_steepest_descent_fallback(ssn_tol);
        if (ls.outcome == LineSearchOutcome::AcceptOptimal) { _opt = TerminationStatus::Optimal; break; }
        if (ls.outcome == LineSearchOutcome::Fail)          { _opt = TerminationStatus::LineSearchFailed; break; }

        update_iterate(ls.tau, _iter);
        _iter++;

        if (what == PrintWhat::SSN) {
            report_(IterationRecord<T>{0, ssn_iter + _iter, krylov_iter, fact,
                                        T(0), Vec(), tol_achieved, mu, rho, ssn_tol,
                                        linesearch_fail, krylov_fail, /*show_pmm_iter=*/false,
                                        n_active_W, static_cast<int>(active_K.count()), -1});
        }

#if SSN_ENABLE_TIMERS
        // TIMER: step-by-step timer for this SSN iteration.
        {
            const double total = timer_prep + timer_linear_solve + timer_linesearch + timer_state_update;
            fprintf(stderr,
                "[Timer] ssn_iter=%d total=%.4fs | prep=%.4f "
                "linear_solve=%.4f (prec_setup=%.4f [assembly=%.4f analyze=%.4f factorize=%.4f] krylov_solve=%.4f) "
                "linesearch=%.4f state_update=%.4f\n",
                ssn_iter + _iter, total, timer_prep, timer_linear_solve, timer_prec_setup,
                timer_prec_assembly, timer_prec_analyze, timer_prec_factorize, timer_krylov_solve,
                timer_linesearch, timer_state_update);

            // If the direct solver ran (direct mode, or after PCG failed), report LDLT on K and Cholesky on S.
            const double ldlt_total = timer_ldlt_analyze + timer_ldlt_factorize + timer_ldlt_solve;
            if (ldlt_total > 0.0) {
                fprintf(stderr,
                    "[Timer]   kkt_ldlt total=%.4fs | analyzePattern=%.4f factorize=%.4f solve=%.4f\n",
                    ldlt_total, timer_ldlt_analyze, timer_ldlt_factorize, timer_ldlt_solve);
            }
            const double chol_total = timer_chol_assembly + timer_chol_analyze + timer_chol_factorize + timer_chol_solve;
            if (chol_total > 0.0) {
                fprintf(stderr,
                    "[Timer]   schur_chol total=%.4fs | assembly=%.4f analyzePattern=%.4f factorize=%.4f solve=%.4f\n",
                    chol_total, timer_chol_assembly, timer_chol_analyze, timer_chol_factorize, timer_chol_solve);
            }
        }
#endif

        auto term = check_ssn_termination(ssn_tol, stagnation, prev_tol_achieved);
        if (term) { _opt = term; break; }
    }

    if (!_opt) {
        _opt = TerminationStatus::MaxInnerIterations;
    }
    x  = x_cur_;
    y2 = y2_cur_;
    opt = *_opt;
    iter = _iter;
}
