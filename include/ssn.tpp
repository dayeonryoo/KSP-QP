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

    const int n_act = n_active_W;

    // G = [A; active rows of B]
    G_trips_.clear();
    G_trips_.reserve(G_A_trips_.size() + B_rm.nonZeros());
    G_trips_.insert(G_trips_.end(), G_A_trips_.begin(), G_A_trips_.end());

    int i_act = 0;
    for (int i = 0; i < l; ++i) {
        if (active_W(i)) {
            for (RIt it(B_rm, i); it; ++it)
                G_trips_.emplace_back(M + i_act, it.col(), it.value());
            ++i_act;
        }
    }

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
    |S| is overestimated via the densest active_K column and the pigeonhole principle, capped at s^2.
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
    S_nnz = std::min<long long>(S_nnz, (long long)s * s); // S is s x s.

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

    // The absolute target as Eigen's relative tolerance.
    if (krylov_abs_tol_ > T(0)) {
        const T rhs_norm = rhs.norm();
        if (rhs_norm > T(0)) tol = krylov_abs_tol_ / rhs_norm;
    }
    if (!in_refinement_) {
        it_state_.krylov_rtol = static_cast<double>(tol);
        if (krylov_abs_tol_ > T(0)) it_state_.krylov_atol = static_cast<double>(krylov_abs_tol_);
    }
    cg.setTolerance(tol);
    cg.setMaxIterations(max_iter);

    // Set up preconditioner and call cg.compute().
    // force_rebuild=true skips SMW and recomputes G E G^T from scratch.
    auto setup_prec = [&](bool force_rebuild) {
        cg.preconditioner().arm(G, G_tr, H_diag, active_K, active_W, B_rm, mu, rho,
                                update_prec, prec_pattern_changed, schur_use_ldlt, force_rebuild);
        cg.compute(S);
        const int new_facts = cg.preconditioner().consume_fact_count_delta();
        fact      += new_facts;
        smw_count  = cg.preconditioner().smw_count();
        record_prec_build(new_facts);
    };

    // Run PCG. Returns false (and counts a krylov_fail) on preconditioner failure or on non-convergence.
    auto attempt_solve = [&](Vec& dy_out) -> bool {
        if (cg.preconditioner().info() != Eigen::Success) {
            std::cout << "[PCG] CG failed due to preconditioner failure.\n";
            krylov_fail++;
            record_pcg_fail(PcgFailKind::PrecFailed, 0, std::numeric_limits<double>::quiet_NaN());
            return false;
        }

        Vec dy_;
        bool warm_start = (prev_dy_.size() == s); // prev_dy_ is empty on the first SSN iteration or if the active set changed.
        {
            const double apply0 = cg.preconditioner().apply_time(), apply_smw0 = cg.preconditioner().apply_smw_time();
            SSN_TIMER_BLOCK(it_.krylov);
            if (warm_start) {
                dy_ = cg.solveWithGuess(rhs, prev_dy_);
            } else {
                dy_ = cg.solve(rhs);
            }
            it_.prec_apply     += cg.preconditioner().apply_time()     - apply0;
            it_.prec_apply_smw += cg.preconditioner().apply_smw_time() - apply_smw0;
        }
        krylov_iter += cg.iterations();
        (in_refinement_ ? it_state_.krylov_refine : it_state_.krylov) += static_cast<int>(cg.iterations());

        T err = cg.error();
        if (!in_refinement_) it_state_.pcg_err = static_cast<double>(err);
        if (cg.info() != Eigen::Success) {
            krylov_fail++;
            record_pcg_fail(PcgFailKind::NoConvergence, static_cast<int>(cg.iterations()), static_cast<double>(err));
            return false;
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

    // Set up and attempt to solve by PCG. On bad_alloc (building the preconditioner or inside
    // Eigen's CG), fall back to the direct solver.
    Vec dy_;
    bool ok = false;
    try {
        setup_prec(false);
        ok = attempt_solve(dy_);
    } catch (const std::bad_alloc&) {
        std::cout << "[PCG] CG failed due to bad_alloc.\n";
        krylov_fail++;
        record_pcg_fail(PcgFailKind::BadAlloc, -1, std::numeric_limits<double>::quiet_NaN());
        switch_to_direct();
        return solve_direct(G, G_tr, H_diag, H_diag_inv, r1, r2);
    }

    // If CG succeeded through an SMW update or a stale factorization, reset the SMW fail streak.
    if (ok && (cg.preconditioner().used_smw() || cg.preconditioner().is_stale()))
        cg.preconditioner().reset_smw_fail_streak();

    // If CG failed through an SMW update or a stale factorization, retry after a full rebuild.
    // If it already used a fresh full factorization: fall back to the direct solver.
    if (!ok && cg.preconditioner().should_retry_after_failure()) {
        try {
            setup_prec(true);
            ok = attempt_solve(dy_);
        } catch (const std::bad_alloc&) {
            std::cout << "[PCG] CG failed due to bad_alloc.\n";
            krylov_fail++;
            record_pcg_fail(PcgFailKind::BadAlloc, -1, std::numeric_limits<double>::quiet_NaN());
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
    const bool was_ldlt = direct_use_ldlt;
    direct_use_ldlt = choose_schur_ldlt(G, all_cols_);
    if (direct_ldlt_decisions_made_ > 0 && direct_use_ldlt != was_ldlt && kkt_ldlt_fact + schur_chol_fact > 0)
        direct_pending_reason_ = FactReason::MethodSwitch;
    ++direct_ldlt_decisions_made_;
    if (direct_use_ldlt) release_chol(); // no-op unless switching away from the Schur complement
}

template <typename T>
typename SchurSmwUpdate<T>::BaseSolve SSN<T>::direct_base_solve(bool use_ldlt) {
    // S^{-1} v through the stored factorization: Cholesky on S directly, or the trailing block of
    // K^{-1} [0; v] for LDLT on K, since (K^{-1})_{22} = S^{-1}.
    if (use_ldlt) {
        return [this](const Vec& v, Vec& out) {
            direct_smw_pad_.setZero(ldlt_.rows());
            direct_smw_pad_.tail(v.size()) = v;
            out = ldlt_.solve(direct_smw_pad_).tail(v.size());
        };
    }
    return [this](const Vec& v, Vec& out) { out = chol_->solve(v); };
}

template <typename T> // Direct solver: dispatches to an SMW update of the last factorization, LDLT on K or Cholesky on S
typename SSN<T>::Vec SSN<T>::solve_direct(const SpMat& G, const SpMat& G_tr, const Vec& H_diag, const Vec& H_diag_inv,
                                          const Vec& r1, const Vec& r2) {
    // No rows in G: K = -H, so dx = -H^{-1} r1 exactly and there is nothing to factorize.
    if (G.rows() == 0) {
        direct_smw_.deactivate();
        return -H_diag_inv.cwiseProduct(r1);
    }

    // First direct solve without a decision yet (right after PCG failed): decide now.
    if (direct_ldlt_decisions_made_ == 0) decide_direct_system(G);
    const bool use_ldlt = direct_use_ldlt || schur_chol_failed_;

    // The Newton matrix changed since the last setup. In direct mode, if the factorization in use would be
    // redone, first try an SMW update of it against its snapshot. Refinement solves skip this block and
    // reuse whatever the main solve set up.
    if (direct_stale_) {
        direct_stale_ = false;
        direct_smw_.deactivate();
        const bool refactor_pending = use_ldlt ? (ldlt_pattern_dirty_ || ldlt_numeric_dirty_)
                                               : (!chol_ || chol_pattern_dirty_ || chol_numeric_dirty_);
        if (direct_smw_enabled() && !direct_force_full_ && refactor_pending) {
            using Smw = SchurSmwUpdate<T>;
            double t_setup = 0.0;
            typename Smw::Outcome outcome;
            {
                SSN_TIMER_BLOCK(t_setup);
                outcome = direct_smw_.setup(G, H_diag, active_W, B_rm, mu, rho, use_ldlt, direct_base_solve(use_ldlt));
            }
            it_.lowrank += t_setup;
            if (outcome == Smw::Outcome::Rejected) {
                // The refactorization that follows is logged with the reason the update was refused.
                switch (direct_smw_.last_reject_reason()) {
                    case Smw::RejectReason::Suppressed:              direct_pending_reason_ = FactReason::SmwSuppressed; break;
                    case Smw::RejectReason::BaseChanged:             direct_pending_reason_ = FactReason::MethodSwitch; break;
                    case Smw::RejectReason::MuChangedSinceSnapshot:
                    case Smw::RejectReason::RhoChangedSinceSnapshot: direct_pending_reason_ = FactReason::MuRho; break;
                    case Smw::RejectReason::RankExceedsThreshold:    direct_pending_reason_ = FactReason::RankExceeded; break;
                    case Smw::RejectReason::SingularCapacitance:
                    case Smw::RejectReason::NonFinite:               direct_pending_reason_ = FactReason::SingularCapacitance; break;
                    case Smw::RejectReason::NoSnapshot: // before the first direct factorization: Initial
                        if (kkt_ldlt_fact + schur_chol_fact > 0) direct_pending_reason_ = FactReason::NoSnapshot;
                        break;
                    default:                                         direct_pending_reason_ = FactReason::Other; break;
                }
            } else {
                LinEvent<T> e = make_event(LinTarget::Direct, outcome == Smw::Outcome::Updated ? LinAction::SmwUpdate
                                                                                               : LinAction::SmwReuse);
                e.ldlt = use_ldlt;
                e.rank = direct_smw_.last_rank();
                e.h = direct_smw_.last_h();
                e.p = direct_smw_.last_p();
                e.q = direct_smw_.last_q();
                e.t_smw_setup = t_setup;
                record_event(e);
            }
            if (outcome == Smw::Outcome::Updated) {
                ++direct_smw_count;
            } else if (outcome == Smw::Outcome::ReusedSnapshot) {
                // The active sets are back to the snapshot's: the stored factorization is exact again.
                if (use_ldlt) ldlt_pattern_dirty_ = ldlt_numeric_dirty_ = false;
                else          chol_pattern_dirty_ = chol_numeric_dirty_ = false;
            }
        }
        direct_force_full_ = false;
    }

    if (direct_smw_enabled() && direct_smw_.active() && direct_smw_.rows() == G.rows() &&
        direct_smw_.snapshot_base_is_ldlt() == use_ldlt) {
        SSN_TIMER_BLOCK(it_.direct_solve);
        // Solve S dy = G H^{-1} r1 + r2 through the update, then recover dx = H^{-1} (G^T dy - r1).
        const int s = G.rows();
        const int n = G.cols();
        direct_smw_Hinv_r1_.noalias() = H_diag_inv.cwiseProduct(r1);
        direct_smw_rhs_.noalias() = G * direct_smw_Hinv_r1_ + r2;
        direct_smw_.solve(direct_smw_rhs_, direct_base_solve(use_ldlt), direct_smw_dy_);

        Vec result(n + s);
        result.head(n) = H_diag_inv.cwiseProduct(G_tr * direct_smw_dy_ - r1);
        result.tail(s) = direct_smw_dy_;
        return result;
    }
    direct_smw_.deactivate();

    if (use_ldlt)
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
        // K_ldlt_ and ldlt_ are about to change, so an SMW snapshot of the old factorization no longer holds.
        direct_smw_.invalidate();
        double t_assembly = 0.0, t_analyze = 0.0, t_factorize = 0.0;
        const bool analyzed = ldlt_pattern_dirty_;

        if (ldlt_pattern_dirty_ || !K_ldlt_built_) {
            {
            SSN_TIMER_BLOCK(t_assembly);
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

            // Cache each diagonal's valuePtr() index for the in-place patch path below.
            ldlt_diag_top_idx_.resize(n);
            for (int i = 0; i < n; ++i)
                ldlt_diag_top_idx_[i] = static_cast<int>(&K_ldlt_.coeffRef(i, i) - K_ldlt_.valuePtr());
            ldlt_diag_bot_idx_.resize(s);
            for (int i = 0; i < s; ++i)
                ldlt_diag_bot_idx_[i] = static_cast<int>(&K_ldlt_.coeffRef(n + i, n + i) - K_ldlt_.valuePtr());
            }

            if (ldlt_pattern_dirty_) {
                SSN_TIMER_BLOCK(t_analyze);
                ldlt_.analyzePattern(K_ldlt_);
                ldlt_pattern_dirty_ = false;
            }
        } else {
            SSN_TIMER_BLOCK(t_assembly);
            // Pattern unchanged (active_W same): patch the -H and (1/mu)I diagonals in place.
            for (int i = 0; i < n; ++i)
                K_ldlt_.valuePtr()[ldlt_diag_top_idx_[i]] = -H_diag(i);
            const T mu_inv = T(1) / mu;
            for (int i = 0; i < s; ++i)
                K_ldlt_.valuePtr()[ldlt_diag_bot_idx_[i]] = mu_inv;
        }

        {
            SSN_TIMER_BLOCK(t_factorize);
            ldlt_.factorize(K_ldlt_);
        }
        it_.fact += t_assembly + t_analyze + t_factorize;
        it_.analyze += t_analyze;
        if (ldlt_.info() != Eigen::Success)
            throw std::runtime_error("LDLT factorization of the augmented Lagrangian system failed.");
        ldlt_numeric_dirty_ = false;
        record_direct_fact(/*ldlt=*/true, analyzed, t_assembly, t_analyze, t_factorize);
        fact++;
        kkt_ldlt_fact++;

        // Record the factorized K for later SMW updates (direct mode only).
        if (direct_smw_enabled()) direct_smw_.snapshot(G, H_diag, active_W, mu, rho, /*base_is_ldlt=*/true);
    }

    Vec result;
    {
        SSN_TIMER_BLOCK(it_.direct_solve);
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
        // S_chol_ and chol_ are about to change, so an SMW snapshot of the old factorization no longer holds.
        direct_smw_.invalidate();
        double t_assembly = 0.0, t_analyze = 0.0, t_factorize = 0.0;

        try {
            {
                SSN_TIMER_BLOCK(t_assembly);
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

            const bool analyzed = !chol_ || chol_pattern_dirty_;
            if (analyzed) {
                SSN_TIMER_BLOCK(t_analyze);
                chol_ = std::make_unique<SimplicialLLT64<SpMat>>();
                chol_->analyzePattern(S_chol_);
                chol_nnz_at_analyze_ = S_chol_.nonZeros();
                chol_pattern_dirty_ = false;
            }
            {
                SSN_TIMER_BLOCK(t_factorize);
                chol_->factorize(S_chol_);
            }
            record_direct_fact(/*ldlt=*/false, analyzed, t_assembly, t_analyze, t_factorize);
            fact++;
            schur_chol_fact++;
            if (chol_->info() != Eigen::Success) {
                ok = false;
                reason = "not numerically positive definite";
            } else {
                chol_numeric_dirty_ = false;
                // Record the factorized S for later SMW updates (direct mode only).
                if (direct_smw_enabled()) direct_smw_.snapshot(G, H_diag, active_W, mu, rho, /*base_is_ldlt=*/false);
            }
        } catch (const std::bad_alloc&) {
            ok = false;
            reason = "bad_alloc";
        }
        it_.fact += t_assembly + t_analyze + t_factorize;
        it_.analyze += t_analyze;
    }

    if (!ok) {
        std::cout << "[Direct] Cholesky on the Schur complement failed (" << reason
                  << "), falling back to LDLT on the KKT system.\n";
        schur_chol_failed_ = true;
        release_chol();
        direct_pending_reason_ = FactReason::CholFailed;
        return solve_using_ldlt(G, H_diag, r1, r2);
    }

    // Solve S dy = G H^{-1} r1 + r2, then recover dx = H^{-1} (G^T dy - r1).
    chol_Hinv_r1_.noalias() = H_diag_inv.cwiseProduct(r1);
    chol_rhs_.noalias() = G * chol_Hinv_r1_ + r2;
    Vec dy;
    {
        SSN_TIMER_BLOCK(it_.direct_solve);
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

    // An SMW snapshot of the Cholesky factorization goes with it; one of the LDLT factorization stays valid
    // (decide_direct_system() calls this on every decision for K, not only on a switch).
    if (direct_smw_.has_snapshot() && !direct_smw_.snapshot_base_is_ldlt()) direct_smw_.invalidate();
}

template <typename T>
SsnLineSearchParams<T> SSN<T>::make_line_search_params() {
    // Ax_ssn_ is unchanged across the line search and its steepest-descent retry; compute once.
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

    // No breakpoints and a near-zero direction (eta is its weighted squared norm): take a full step.
    if (breakpoints.empty() && eta < eps_zero) return T(1);
    // No breakpoints but a non-negligible direction: the code below solves the quadratic psi exactly.

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
    SSN_TIMER_BLOCK(it_.prep);
    u_.noalias() = z / mu + x_cur_;
    v_ = Bx_ssn_ + ((1 - alpha) * y2_cur_ - y2) / mu;

    // Clarke subgradient and distance for K and W
    compute_subgrad_and_dist(u_, lx, ux, false, new_active_K_, dist_K_u_);
    compute_subgrad_and_dist(v_, lw, uw, true,  new_active_W_, dist_W_v_);
    // W is active when v lies outside [lw, uw], so invert the inside-[lw, uw] subgradient.
    new_active_W_ = (new_active_W_ == false);

    // Detect active-set changes; recompute on any single change.
    bool first_ssn_iter = (active_K.size() == 0);
    const int flips_K = first_ssn_iter ? -1 : static_cast<int>((active_K != new_active_K_).count());
    const int flips_W = first_ssn_iter ? -1 : static_cast<int>((active_W != new_active_W_).count());
    ActiveSetDelta delta{
        /*k_changed=*/first_ssn_iter || flips_K > 0,
        /*w_changed=*/first_ssn_iter || flips_W > 0,
    };

    bool update_prec = delta.k_changed || delta.w_changed; // true means rebuilding prec is needed.
    bool prec_pattern_changed = delta.k_changed || delta.w_changed; // true means analyzePattern() is needed.

    // Recompute H if active_K changed or mu/rho drifted since H_diag was last built.
    bool recompute_H = delta.k_changed || (mu != H_diag_mu_) || (rho != H_diag_rho_);

    // What changed since the previous SSN iteration, for the factorization log.
    it_state_.k_changed   = delta.k_changed;
    it_state_.w_changed   = delta.w_changed;
    it_state_.mu_changed  = (mu != H_diag_mu_);
    it_state_.rho_changed = (rho != H_diag_rho_);
    it_state_.flips_K     = flips_K;
    it_state_.flips_W     = flips_W;

    // Direct-solver flags: K's and S's sparsity follows G (active_W); their values change when
    // H_diag is recomputed or G is rebuilt.
    if (delta.w_changed) ldlt_pattern_dirty_ = chol_pattern_dirty_ = true;
    if (recompute_H || delta.w_changed) {
        ldlt_numeric_dirty_ = chol_numeric_dirty_ = true;
        direct_stale_ = true;
    }

    if (recompute_H) {
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
        prev_dy_.resize(0); // No warm-starting for CG

        active_W = new_active_W_;
        n_active_W = active_W.count();
        n_inactive_W = l - n_active_W;

        rebuild_G(); // Rebuild G = [A; active rows of B] and G_tr.
    }

    // Compute dy2 in inactive_W: dy2_inactive_W = - (mu / alpha) * dist_W(v)(inactive_W) - y2(inactive_W) = - y2(inactive_W),
    // since v lies in [lw, uw] on inactive_W, where dist_W(v) = 0 exactly.
    split_by_mask(y2_cur_, active_W, y2_active_W_, y2_inactive_W_);
    split_by_mask(dist_W_v_, active_W, dist_W_v_active_, dist_W_v_inactive_);
    dy2_inactive_W_.head(n_inactive_W).noalias() = -y2_inactive_W_.head(n_inactive_W);

    // Compute the RHS vector. Its y2 term - B^T y2 - B(inactive_W)^T dy2_inactive_W reduces to - B(active_W)^T y2(active_W),
    // and B(active_W)^T is the last n_active_W columns of G_tr.
    if (Q_info == 0) {
        r1_ = c + mu * dist_K_u_
             - G_tr.rightCols(n_active_W) * y2_active_W_.head(n_active_W)
             + (x_cur_ - x) / rho;
    } else {
        r1_ = c + Q_diag.cwiseProduct(x_cur_) + mu * dist_K_u_
             - G_tr.rightCols(n_active_W) * y2_active_W_.head(n_active_W)
             + (x_cur_ - x) / rho;
    }
    r2_.resize(M + n_active_W);
    r2_.head(M) = y1 / mu - Ax_ssn_ + b;
    r2_.tail(n_active_W) = -dist_W_v_active_.head(n_active_W) - (alpha / mu) * y2_active_W_.head(n_active_W);

    // Determines the factorization method, each locked after its first 3 decisions:
    // for the direct solver (K or S, over all columns) once it is in use, otherwise for the PCG preconditioner.
    const bool use_direct = direct_solve || pcg_failed;
    if ((delta.k_changed || delta.w_changed) && use_direct && direct_ldlt_decisions_made_ < 3) {
        decide_direct_system(G);
    } else if ((delta.k_changed || delta.w_changed) && !use_direct && schur_ldlt_decisions_made_ < 3) {
        schur_use_ldlt = choose_schur_ldlt(G, active_K);
        ++schur_ldlt_decisions_made_;
    }

    return {update_prec, prec_pattern_changed};
}

template <typename T>
typename SSN<T>::RefineResult SSN<T>::iterative_refine_dxdy(bool need_final_residual, T newton_target) {
    // Iterative refinement: correct the residual of K [dx;dy] = [r1_;r2_], K = [-H, G^T; G, (1/mu)I], until
    // newton_res <= newton_target if that is positive, else until the KKT test passes.
    // The residual after the last of refine_max_iter corrections is measured only if need_final_residual or newton_target > 0.
    SSN_TIMER_BLOCK(it_.refine);
    const int s = static_cast<int>(r2_.size());
    const T ref_norm = std::max(inf_norm(r1_), inf_norm(r2_));
    T res_norm = std::numeric_limits<T>::quiet_NaN();
    T e = std::numeric_limits<T>::quiet_NaN();
    Vec rho1(N), rho2(s);

    const bool newton = newton_target > T(0);
    const int last_k = (need_final_residual || newton) ? refine_max_iter : refine_max_iter - 1;
    for (int k = 0; k <= last_k; ++k) {
        const auto dx_k = dxdy_.head(N);
        const auto dy_k = dxdy_.tail(s);
        Gtr_dy_.noalias() = G_tr * dy_k;
        G_dx_.noalias()   = G * dx_k;
        rho1 = r1_ + H_diag.cwiseProduct(dx_k) - Gtr_dy_;
        rho2 = r2_ - G_dx_ - dy_k / mu;

        res_norm = std::max(inf_norm(rho1), inf_norm(rho2));
        if (newton) e = newton_residual_norm(rho1, rho2);
        const bool met = newton ? e <= newton_target : res_norm <= std::max(refine_rel_tol * ref_norm, refine_abs_tol);
        if (met || k == refine_max_iter) break;

        prev_dy_.resize(0); // cold-start for iterative refinement
        in_refinement_ = true;
        Vec correction = solve_using_cg(G, G_tr, H_diag, H_diag_inv, active_K, rho1, rho2,
                                        mu, krylov_tol, krylov_max_in_iter, false, false, schur_use_ldlt);
        in_refinement_ = false;
        dxdy_ += correction;
        prev_dy_ = dxdy_.tail(s); // warm-start for the next SSN iteration's main solve
    }
    return {res_norm, ref_norm, e};
}

template <typename T>
T SSN<T>::newton_residual_norm() {
    const int s = static_cast<int>(r2_.size());
    const auto dx = dxdy_.head(N);
    const auto dy = dxdy_.tail(s);
    return newton_residual_norm(r1_ + H_diag.cwiseProduct(dx) - G_tr * dy, r2_ - G * dx - dy / mu);
}

template <typename T>
T SSN<T>::newton_residual_norm(Vec rho1, Vec rho2) {
    // From the KKT residual (rho1, rho2) of dxdy_: e_x = rho1 - mu G^T D rho2, e_W = -((1 - alpha) / alpha) rho2_W,
    // with D = 1 on A rows and 1/alpha on active-W rows; e is 0 on inactive W.
    T e_W = T(0);
    for (int i = 0, k = M; i < l; ++i)
        if (active_W(i)) e_W = std::max(e_W, std::abs((1 - alpha) / alpha * rho2(k++)) * D1B_diag_inv(i));

    rho2.tail(n_active_W) /= alpha;
    rho1.noalias() -= mu * (G_tr * rho2);
    return std::max(inf_norm(rho1.cwiseProduct(D2_ext_inv)), e_W);
}

template <typename T>
bool SSN<T>::full_step_keeps_active_sets() {
    // At the full step, u = z/mu + x and v = Bx + ((1 - alpha) y2 - y2_k)/mu stay on their side (below, inside, above)
    // of their bounds, with compute_subgrad_and_dist()'s conventions: K is inside iff l < u < h, W iff l <= v <= h.
    auto side_K = [](T u, T lo, T hi) { return u <= lo ? -1 : (u >= hi ? 1 : 0); };
    auto side_W = [](T v, T lo, T hi) { return v < lo ? -1 : (v > hi ? 1 : 0); };
    const auto dx = dxdy_.head(N);
    for (int i = 0; i < N; ++i)
        if (side_K(u_(i), lx(i), ux(i)) != side_K(u_(i) + dx(i), lx(i), ux(i))) return false;
    if (l == 0) return true;

    Vec dy2(l);
    retrieve_row_order(dxdy_.tail(n_active_W), dy2_inactive_W_.head(n_inactive_W), active_W, dy2);
    const Vec v1 = v_ + B * dx + ((1 - alpha) / mu) * dy2;
    for (int i = 0; i < l; ++i)
        if (side_W(v_(i), lw(i), uw(i)) != side_W(v1(i), lw(i), uw(i))) return false;
    return true;
}

template <typename T>
void SSN<T>::solve_newton_direction(bool update_prec, bool prec_pattern_changed) {
    SSN_TIMER_BLOCK(it_.linear_solve);
    // In solve_ssn(), PCG aims at newton_res <= kappa * g, g = max(ssn_tol, ||grad M||); refinement goes on to
    // kappa * ssn_tol only if the full step keeps the active sets, the one step that can end the SSN loop.
    const bool armed = ssn_tol_cur_ > T(0) && !(direct_solve || pcg_failed);
    const T g = std::max(ssn_tol_cur_, grad_cur_);
    if (armed) krylov_abs_tol_ = krylov_tol_factor * g / mu;

    // Solve for dx and dy2_active_W.
    dxdy_ = solve_using_cg(G, G_tr, H_diag, H_diag_inv, active_K, r1_, r2_, mu, krylov_tol, krylov_max_in_iter, update_prec, prec_pattern_changed, schur_use_ldlt);

    if (armed && !pcg_failed) {
        const bool keeps = full_step_keeps_active_sets();
        it_state_.on_piece = keeps ? 1 : 0;
        const T target = krylov_tol_factor * (keeps ? ssn_tol_cur_ : g);
        krylov_abs_tol_ = target / mu;
        it_state_.newton_res = static_cast<double>(iterative_refine_dxdy(/*need_final_residual=*/false, target).newton_res);
    } else {
        // An SMW-updated direct solve is kept only if refinement brings its KKT residual within
        // direct_smw_accept_tol; otherwise refactorize and solve again (cf. the preconditioner's retry after a
        // PCG failure). A NaN residual fails the check.
        const bool used_smw = direct_smw_enabled() && direct_smw_.active();
        const RefineResult refined = iterative_refine_dxdy(/*need_final_residual=*/used_smw);
        if (used_smw) {
            if (refined.res_norm <= std::max(direct_smw_accept_tol * refined.ref_norm, refine_abs_tol)) {
                direct_smw_.reset_fail_streak();
            } else {
                ++direct_smw_reject;
                direct_smw_.record_failure();
                direct_force_full_ = direct_stale_ = true;
                direct_pending_reason_ = FactReason::SmwResidual;
                dxdy_ = solve_using_cg(G, G_tr, H_diag, H_diag_inv, active_K, r1_, r2_, mu, krylov_tol, krylov_max_in_iter, update_prec, prec_pattern_changed, schur_use_ldlt);
                iterative_refine_dxdy();
            }
        }
        if (trace_ && trace_->enabled) it_state_.newton_res = static_cast<double>(newton_residual_norm());
    }

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

    SSN_TIMER_BLOCK(it_.linesearch);
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
        it_state_.ls_steepest = true;
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
    SSN_TIMER_BLOCK(it_.update);
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
    // Initialize iteration counter and set starting points.
    x_cur_ = x;
    y2_cur_ = y2;
    int _iter = 0;
    std::optional<TerminationStatus> _opt;
    T prev_tol_achieved = inf;
    int stagnation = 0;

    // Arms solve_newton_direction()'s targets; the gradient a step leaves is about mu G^T times the Schur residual.
    ssn_tol_cur_ = ssn_tol;

    // Useful matvecs
    Ax_ssn_.noalias() = A * x_cur_;
    Bx_ssn_.noalias() = B * x_cur_;

    // ||grad M|| at the start point; tol_achieved keeps the previous solve's value until the first step.
    grad_cur_ = compute_grad_Lagrangian_unscaled_inf_norm(compute_grad_Lagrangian(x_cur_, y2_cur_, Ax_ssn_, Bx_ssn_));

    // SSN main loop
    while (_iter < ssn_max_in_iter) {
        if (interrupted_()) { _opt = TerminationStatus::Interrupted; break; }
        if (time_limit_exceeded_()) { _opt = TerminationStatus::TimeLimit; break; }
        begin_iteration(ssn_iter + _iter + 1);

        auto [update_prec, prec_pattern_changed] = prepare_newton_system();
        solve_newton_direction(update_prec, prec_pattern_changed);

        LineSearchResult ls = line_search_with_steepest_descent_fallback(ssn_tol);
        if (ls.outcome == LineSearchOutcome::AcceptOptimal) {
            end_iteration(SsnLsOutcome::AcceptOptimal, T(0), ssn_tol);
            _opt = TerminationStatus::Optimal;
            break;
        }
        if (ls.outcome == LineSearchOutcome::Fail) {
            end_iteration(SsnLsOutcome::Fail, T(0), ssn_tol);
            _opt = TerminationStatus::LineSearchFailed;
            break;
        }

        update_iterate(ls.tau, _iter);
        grad_cur_ = tol_achieved;
        _iter++;

        if (what == PrintWhat::SSN) {
            report_(IterationRecord<T>{0, ssn_iter + _iter, krylov_iter, fact,
                                        T(0), Vec(), tol_achieved, mu, rho, ssn_tol,
                                        linesearch_fail, krylov_fail, /*show_pmm_iter=*/false,
                                        n_active_W, static_cast<int>(active_K.count()), -1});
        }

        std::optional<TerminationStatus> term;
        {
            SSN_TIMER_BLOCK(it_.update);
            term = check_ssn_termination(ssn_tol, stagnation, prev_tol_achieved);
        }
        end_iteration(SsnLsOutcome::Step, ls.tau, ssn_tol);
        if (term) { _opt = term; break; }
    }

    if (!_opt) {
        _opt = TerminationStatus::MaxInnerIterations;
    }
    x  = x_cur_;
    y2 = y2_cur_;
    opt = *_opt;
    iter = _iter;
    ssn_tol_cur_ = krylov_abs_tol_ = T(0);
}

// ===================== factorization log and run-time split (solve_profile.hpp) =====================

template <typename T>
FactReason SSN<T>::prec_fact_reason(const typename SchurPreconditioner<T>::BuildInfo& b) {
    // The first check that refused an SMW update decides (smw_gate_open()'s order).
    using Reject = typename SchurPreconditioner<T>::SmwRejectReason;
    using Cause = typename SchurPreconditioner<T>::SkipCause;
    if (b.initial) return FactReason::Initial;
    switch (b.reject) {
        case Reject::ForcedRebuild:
            return b.skip_cause == Cause::Suppressed ? FactReason::SmwSuppressed : FactReason::PcgFailure;
        case Reject::Suppressed:                 return FactReason::SmwSuppressed;
        case Reject::FactorizationMethodChanged: return FactReason::MethodSwitch;
        case Reject::NoSnapshot:                 return FactReason::NoSnapshot;
        case Reject::RhoChangedSinceSnapshot:
        case Reject::MuChangedSinceSnapshot:     return FactReason::MuRho;
        case Reject::RankExceedsThreshold:       return FactReason::RankExceeded;
        case Reject::SingularCapacitance:        return FactReason::SingularCapacitance;
        default:                                 return FactReason::Other;
    }
}

template <typename T>
LinEvent<T> SSN<T>::make_event(LinTarget target, LinAction action) const {
    LinEvent<T> e;
    e.pmm_iter = pmm_index_;
    e.ssn_iter = it_state_.index;
    e.target = target;
    e.action = action;
    e.mu = mu;
    e.rho = rho;
    e.mu_changed = it_state_.mu_changed;
    e.rho_changed = it_state_.rho_changed;
    e.k_changed = it_state_.k_changed;
    e.w_changed = it_state_.w_changed;
    e.s = static_cast<int>(G.rows());
    return e;
}

template <typename T>
void SSN<T>::record_event(const LinEvent<T>& e) {
    switch (e.action) {
        case LinAction::Factorize:
            ++profile_.fact_by_reason[static_cast<int>(e.reason)];
            if (e.analyzed) ++profile_.fact_analyzed;
            ++it_state_.facts;
            break;
        case LinAction::SmwUpdate: ++profile_.smw_update; ++it_state_.smw; break;
        case LinAction::SmwReuse:  ++profile_.smw_reuse; break;
        case LinAction::PcgFail:   ++profile_.pcg_fail; break;
        case LinAction::MuRhoKeep: break; // counted in record_prec_build(), with the keeps inside SMW setups
    }
    if (e.action != LinAction::PcgFail && it_state_.setup < 0) { // the main solve's setup comes first
        it_state_.setup = static_cast<int>(e.action);
        it_state_.reason = e.reason;
        it_state_.rank = e.rank;
    }
    if (trace_ && trace_->enabled) trace_->events.push_back(e);
}

template <typename T>
void SSN<T>::record_prec_build(int new_facts) {
    // One event per factorization counted in fact (new_facts is 0 or 1), or per SMW update / reuse / keep.
    const auto& b = cg.preconditioner().last_build();
    it_.fact    += b.t_assembly + b.t_analyze + b.t_factorize;
    it_.analyze += b.t_analyze;
    it_.lowrank += b.t_smw_setup;
    if (!b.ran) return;
    if (b.kept_stale) ++profile_.mu_rho_keep;

    using Action = typename SchurPreconditioner<T>::BuildAction;
    LinAction action;
    if (new_facts > 0)                           action = LinAction::Factorize;
    else if (b.action == Action::SmwUpdated)     action = LinAction::SmwUpdate;
    else if (b.action == Action::ReusedSnapshot) action = LinAction::SmwReuse;
    else if (b.action == Action::KeptStale)      action = LinAction::MuRhoKeep;
    else return;

    LinEvent<T> e = make_event(LinTarget::Prec, action);
    if (action == LinAction::Factorize) {
        e.reason = prec_fact_reason(b);
        e.analyzed = b.analyzed;
    }
    e.ldlt = b.ldlt;
    e.rank = b.rank;
    e.h = b.h;
    e.p = b.p;
    e.q = b.q;
    e.stale_ratio = b.stale_ratio;
    e.mu_changed = b.mu_changed;   // since the preconditioner's last factorization or update
    e.rho_changed = b.rho_changed;
    e.t_assembly = b.t_assembly;
    e.t_analyze = b.t_analyze;
    e.t_factorize = b.t_factorize;
    e.t_smw_setup = b.t_smw_setup;
    record_event(e);
}

template <typename T>
void SSN<T>::record_pcg_fail(PcgFailKind kind, int iters, double err) {
    LinEvent<T> e = make_event(LinTarget::Prec, LinAction::PcgFail);
    e.ldlt = schur_use_ldlt;
    e.stale_ratio = cg.preconditioner().stale_ratio();
    e.pcg_fail = kind;
    e.pcg_iters = iters;
    e.pcg_err = err;
    record_event(e);
}

template <typename T>
void SSN<T>::record_direct_fact(bool ldlt, bool analyzed, double t_assembly, double t_analyze, double t_factorize) {
    // Called right before fact++ at each direct factorization, so the counts still exclude it.
    LinEvent<T> e = make_event(LinTarget::Direct, LinAction::Factorize);
    e.reason = consume_direct_reason();
    e.ldlt = ldlt;
    e.analyzed = analyzed;
    e.t_assembly = t_assembly;
    e.t_analyze = t_analyze;
    e.t_factorize = t_factorize;
    record_event(e);
}

template <typename T>
FactReason SSN<T>::consume_direct_reason() {
    // A reason set where the refactorization was decided comes first; then the first direct
    // factorization of the run; then what changed this SSN iteration, mu/rho before the active sets.
    const FactReason pending = direct_pending_reason_;
    direct_pending_reason_ = FactReason::Count;
    if (pending != FactReason::Count) return pending;
    if (kkt_ldlt_fact + schur_chol_fact == 0) return direct_solve ? FactReason::Initial : FactReason::DirectFallback;
    if (it_state_.mu_changed || it_state_.rho_changed) return FactReason::MuRho;
    if (it_state_.k_changed || it_state_.w_changed) return FactReason::ActiveSetChange;
    return FactReason::Other;
}

template <typename T>
void SSN<T>::begin_iteration(int ssn_index) {
    it_ = IterTimes{};
    it_state_ = IterState{};
    it_state_.index = ssn_index;
    direct_pending_reason_ = FactReason::Count;
}

template <typename T>
void SSN<T>::end_iteration(SsnLsOutcome outcome, T tau, T ssn_tol) {
    // The Newton solve's remainder: RHS, dx recovery, refinement residuals, bookkeeping.
    const double ssn_other = it_.linear_solve - (it_.fact + it_.lowrank + it_.krylov + it_.direct_solve);

    SolvePhases& t = profile_.t;
    t.prep += it_.prep;
    t.fact += it_.fact;
    t.analyze += it_.analyze;
    t.lowrank += it_.lowrank;
    t.krylov += it_.krylov;
    t.prec_apply += it_.prec_apply;
    t.prec_apply_smw += it_.prec_apply_smw;
    t.direct_solve += it_.direct_solve;
    t.refine += it_.refine;
    t.ssn_other += ssn_other;
    t.linesearch += it_.linesearch;
    t.update += it_.update;
    profile_.krylov_refine += it_state_.krylov_refine;
    if (it_state_.ls_steepest) ++profile_.ls_steepest;

    if (!trace_ || !trace_->enabled) return;
    SsnRow<T> row;
    row.pmm_iter = pmm_index_;
    row.ssn_iter = it_state_.index;
    row.mu = mu;
    row.rho = rho;
    row.ssn_tol = ssn_tol;
    row.n_act_K = static_cast<int>(active_K.count());
    row.n_act_W = n_active_W;
    row.flips_K = it_state_.flips_K;
    row.flips_W = it_state_.flips_W;
    row.setup = it_state_.setup;
    row.reason = it_state_.reason;
    row.rank = it_state_.rank;
    row.facts = it_state_.facts;
    row.smw = it_state_.smw;
    row.krylov = it_state_.krylov;
    row.krylov_refine = it_state_.krylov_refine;
    row.pcg_err = it_state_.pcg_err;
    row.krylov_rtol = it_state_.krylov_rtol;
    row.krylov_atol = it_state_.krylov_atol;
    row.newton_res = it_state_.newton_res;
    row.on_piece = it_state_.on_piece;
    row.ls_outcome = outcome;
    row.ls_steepest = it_state_.ls_steepest;
    row.tau = tau;
    row.grad_norm = outcome == SsnLsOutcome::Step ? tol_achieved : std::numeric_limits<T>::quiet_NaN();
    row.t_prep = it_.prep;
    row.t_fact = it_.fact;
    row.t_lowrank = it_.lowrank;
    row.t_krylov = it_.krylov;
    row.t_direct_solve = it_.direct_solve;
    row.t_ssn_other = ssn_other;
    row.t_linesearch = it_.linesearch;
    row.t_update = it_.update;
    row.t_prec_apply = it_.prec_apply;
    row.t_refine = it_.refine;
    trace_->ssn.push_back(row);
}
