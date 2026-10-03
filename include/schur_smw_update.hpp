#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <new>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Sparse>

// =================================================================================================
// Sherman-Morrison-Woodbury (SMW) low-rank update of the exact Schur complement
//     S = G H^{-1} G^T + (1/mu) I,   G = [A; rows of B in active_W],   H = diag(H_diag),
// used by SSN's direct solver in direct mode (SSN::direct_solve; see ssn.hpp/.tpp). It mirrors
// SchurPreconditioner's SMW (schur_preconditioner.hpp), including the equilibrated rank test on the
// capacitance matrix (see factorize_capacitance()), with two differences:
//   - weights: E = H^{-1} on ALL columns (the preconditioner keeps active_K columns only);
//   - changed columns are those with H(j) != H_old(j), whatever the reason, so c_j = 1/H_new(j) - 1/H_old(j)
//     is never 0 and the capacitance block is -1/c_j = H_old(j) H_new(j) / (H_new(j) - H_old(j)).
//
// The caller factorizes S_old at a full factorization and records it with snapshot(); later, setup()
// expresses the current S_new against that snapshot as a bordered system of rank h + p + q
// (h deleted W rows, p changed columns, q added W rows):
//     [ S_old    V_all ] [z     ]   [r_pad]        V_all = [E_-, U, V_+],  U = G_old(:, changed),
//     [ V_all^T  M_sub ] [lambda] = [g    ],       V_+ = G_old E_new B_+^T,
//     M_sub = blkdiag(0_h, -C^{-1}, W_+),  C = diag(c_j),  W_+ = B_+ E_new B_+^T + (1/mu) I_q,
// and solve() applies S_new^{-1} through the snapshot's factorization (BaseSolve), which the caller
// passes on every call: Cholesky on S_old, or the trailing block of an LDLT of K_old = [-H_old, G_old^T;
// G_old, (1/mu) I], since (K^{-1})_{22} = S^{-1}.
//
// This class does not own any factorization, decide when to attempt an update, or check the accuracy of
// its result; SSN does all of that (see SSN::solve_direct() and SSN::solve_newton_direction()).
// =================================================================================================

template <typename T>
class SchurSmwUpdate {
public:
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using SpMat = Eigen::SparseMatrix<T>;
    using RowMajorSpMat = Eigen::SparseMatrix<T, Eigen::RowMajor>;
    using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;

    // out = S_snapshot^{-1} v, both of length snapshot_rows(). Passed to every call that needs it, never stored.
    using BaseSolve = std::function<void(const Vec& v, Vec& out)>;

    enum class Outcome {
        Updated,        // S_new^{-1} is now available through solve().
        ReusedSnapshot, // Rank 0: S_new equals the snapshot's S_old, so the caller's factorization is exact as is.
        Rejected,       // See last_reject_reason(); the caller must refactorize.
    };

    // Why the most recent setup() did not produce an update.
    enum class RejectReason {
        None,                    // Updated or ReusedSnapshot, or no attempt has run yet.
        Suppressed,              // Fail streak reached kMaxFailStreak; updates are off until reset_fail_streak().
        NoSnapshot,              // No snapshot to update from (none taken yet, or invalidated since).
        InconsistentData,        // Inputs do not describe G = [A; B(active_W, :)] with the snapshot's dimensions.
        BaseChanged,             // The caller now factorizes the other matrix (LDLT on K vs Cholesky on S).
        MuChangedSinceSnapshot,  // The (1/mu) I block spans every row: a full-rank shift.
        RhoChangedSinceSnapshot, // H(i) = Q(i) + ... + 1/rho changes on every column: full rank in practice.
        RankExceedsThreshold,    // h + p + q > kMaxRank.
        SingularCapacitance,     // The equilibrated capacitance matrix is (near-)singular.
        NonFinite,               // The capacitance matrix has a non-finite entry.
        OutOfMemory,             // std::bad_alloc while building the update.
    };
    static constexpr int kNumRejectReasons = 11;

    static constexpr int kMaxRank       = 50; // setup() rejects h + p + q > this; rank 0 reuses the snapshot.
    static constexpr int kMaxFailStreak = 5;  // record_failure() calls in a row that suppress updates.

    // ------ Snapshot of the last full factorization ------

    // Sets the number of equality-constraint (A) rows, i.e. G's leading rows that never change.
    void set_num_equality_rows(int M) { M_rows_ = M; }

    // Records the matrix the caller has just factorized: S = G H^{-1} G^T + (1/mu) I, with G's trailing rows
    // being the rows of B in active_W. base_is_ldlt says whether the factorization is LDLT on K or Cholesky
    // on S. Inconsistent inputs, or running out of memory for the copies, leave no snapshot.
    void snapshot(const SpMat& G, const Vec& H_diag, const BoolArr& active_W, T mu, T rho, bool base_is_ldlt) {
        active_ = false;
        if (M_rows_ < 0 || G.rows() == 0 || H_diag.size() != G.cols() ||
            G.rows() != M_rows_ + static_cast<Eigen::Index>(active_W.count())) {
            invalidate();
            return;
        }
        try {
            G_old_ = G;
            G_old_.makeCompressed(); // classify() reads column lengths from the outer index
            H_old_ = H_diag;
            active_W_old_ = active_W;
        } catch (const std::bad_alloc&) {
            invalidate();
            return;
        }
        mu_old_ = mu;
        rho_old_ = rho;
        base_is_ldlt_old_ = base_is_ldlt;
        has_snapshot_ = true;
    }

    // Drops the snapshot (and any update built on it); the caller is about to change its factorization.
    void invalidate() {
        has_snapshot_ = false;
        active_ = false;
        SpMat().swap(G_old_);
        H_old_.resize(0);
        active_W_old_.resize(0);
    }

    bool has_snapshot() const { return has_snapshot_; }
    bool snapshot_base_is_ldlt() const { return base_is_ldlt_old_; }
    int snapshot_rows() const { return has_snapshot_ ? static_cast<int>(G_old_.rows()) : 0; }

    // ------ Setup ------

    // Builds S_new^{-1} as a low-rank update of the snapshot's S_old^{-1}. B_rm is B in row-major order
    // (the source of the added rows); base applies S_old^{-1}. Deactivates any previous update on entry.
    Outcome setup(const SpMat& G, const Vec& H_diag, const BoolArr& active_W, const RowMajorSpMat& B_rm,
                  T mu, T rho, bool base_is_ldlt, const BaseSolve& base) {
        active_ = false;
        last_rank_ = 0;
        const Outcome outcome = try_setup(G, H_diag, active_W, B_rm, mu, rho, base_is_ldlt, base);
        if (outcome == Outcome::Updated) {
            active_ = true;
            ++update_count_;
        } else if (outcome == Outcome::ReusedSnapshot) {
            ++reuse_count_;
        } else {
            ++reject_counts_[static_cast<int>(last_reject_reason_)];
        }
        return outcome;
    }

    // True iff the last setup() returned Updated and nothing has invalidated or deactivated it since.
    bool active() const { return active_; }
    void deactivate() { active_ = false; }
    int rows() const { return active_ ? s_new_ : 0; } // size of S_new while active

    // ------ Application ------

    // out = S_new^{-1} rhs, with rhs and out in G_new's row order. Requires active().
    void solve(const Vec& rhs, const BaseSolve& base, Vec& out) {
        assert(active_ && rhs.size() == s_new_);

        // r_pad: rhs in the snapshot's row order; deleted rows stay 0 (set in finalize()).
        r_pad_.head(M_rows_) = rhs.head(M_rows_);
        for (std::size_t i = 0; i < retained_old_rows_.size(); ++i)
            r_pad_(retained_old_rows_[i]) = rhs(retained_new_rows_[i]);

        // u_base = S_old^{-1} r_pad
        base(r_pad_, u_base_);

        // Lambda = g - V_all^T u_base, with g = [0_h; 0_p; rhs(added rows)].
        for (int k = 0; k < h_; ++k)
            Lambda_(k) = -u_base_(deleted_old_rows_[k]);
        for (int j = 0; j < p_; ++j) {
            T val = T(0);
            for (typename SpMat::InnerIterator it(G_old_, changed_cols_[j]); it; ++it)
                val += it.value() * u_base_(it.row());
            Lambda_(h_ + j) = -val;
        }
        for (int j = 0; j < q_; ++j)
            Lambda_(h_ + p_ + j) = rhs(added_new_rows_[j]);
        if (q_ > 0)
            Lambda_.tail(q_).noalias() -= V_plus_.transpose() * u_base_;

        // lambda = S_Lambda^{-1} Lambda through the equilibrated factorization D S_Lambda D.
        lambda_ = eq_scale_.cwiseProduct(Lambda_);
        lambda_ = lu_.solve(lambda_);
        lambda_ = eq_scale_.cwiseProduct(lambda_);

        // z_base = u_base - Y_all lambda; retained rows come from z_base, added rows from lambda's tail.
        z_base_ = u_base_;
        z_base_.noalias() -= Y_all_ * lambda_;

        out.resize(s_new_);
        out.head(M_rows_) = z_base_.head(M_rows_);
        for (std::size_t i = 0; i < retained_old_rows_.size(); ++i)
            out(retained_new_rows_[i]) = z_base_(retained_old_rows_[i]);
        for (int j = 0; j < q_; ++j)
            out(added_new_rows_[j]) = lambda_(h_ + p_ + j);
    }

    // ------ Failure tracking (the caller judges accuracy) ------

    void record_failure() { ++fail_streak_; }
    void reset_fail_streak() { fail_streak_ = 0; }
    bool suppressed() const { return fail_streak_ >= kMaxFailStreak; }
    int fail_streak() const { return fail_streak_; }

    // ------ Diagnostics ------

    RejectReason last_reject_reason() const { return last_reject_reason_; }
    int last_rank() const { return last_rank_; } // h + p + q of the last classified delta (0 if not classified)
    int last_h() const { return h_; }
    int last_p() const { return p_; }
    int last_q() const { return q_; }
    int update_count() const { return update_count_; }
    int reuse_count() const { return reuse_count_; }
    int reject_count(RejectReason reason) const { return reject_counts_[static_cast<int>(reason)]; }

    // Frees the snapshot and every buffer; counters and the fail streak are kept.
    void release() {
        invalidate();
        std::vector<int>().swap(deleted_old_rows_);
        std::vector<int>().swap(retained_old_rows_);
        std::vector<int>().swap(retained_new_rows_);
        std::vector<int>().swap(added_new_rows_);
        std::vector<int>().swap(added_W_src_);
        std::vector<int>().swap(changed_cols_);
        std::vector<int>().swap(touched_);
        M_sub_.resize(0, 0);
        V_plus_.resize(0, 0);
        Y_all_.resize(0, 0);
        VtY_.resize(0, 0);
        lu_ = Eigen::FullPivLU<Mat>();
        eq_scale_.resize(0);
        e_new_b_.resize(0);
        rhs_scratch_.resize(0);
        sol_scratch_.resize(0);
        r_pad_.resize(0);
        u_base_.resize(0);
        Lambda_.resize(0);
        lambda_.resize(0);
        z_base_.resize(0);
    }

    // Grants test-only access to private state; only used from tests/test_schur_smw_update.cpp.
    friend struct SchurSmwUpdateTestPeer;

private:
    Outcome reject(RejectReason reason) {
        last_reject_reason_ = reason;
        return Outcome::Rejected;
    }

    Outcome try_setup(const SpMat& G, const Vec& H_diag, const BoolArr& active_W, const RowMajorSpMat& B_rm,
                      T mu, T rho, bool base_is_ldlt, const BaseSolve& base) {
        last_reject_reason_ = RejectReason::None;
        // Phase 1: eligibility gate -- every reason an update can't even be attempted.
        if (suppressed())                                    return reject(RejectReason::Suppressed);
        if (!has_snapshot_)                                  return reject(RejectReason::NoSnapshot);
        if (!consistent(G, H_diag, active_W, B_rm))          return reject(RejectReason::InconsistentData);
        if (base_is_ldlt != base_is_ldlt_old_)               return reject(RejectReason::BaseChanged);
        if (mu != mu_old_)                                   return reject(RejectReason::MuChangedSinceSnapshot);
        // With columns classified by H != H_old, a rho drift is still handled exactly, but it changes every
        // column, so the rank would exceed kMaxRank anyway; reject early.
        if (rho != rho_old_)                                 return reject(RejectReason::RhoChangedSinceSnapshot);

        try {
            // Phase 2: classify the delta against the snapshot.
            classify(H_diag, active_W);
            last_rank_ = h_ + p_ + q_;
            if (last_rank_ > kMaxRank) return reject(RejectReason::RankExceedsThreshold);
            if (last_rank_ == 0) return Outcome::ReusedSnapshot; // S_new == S_old exactly

            // Phases 3-5: capacitance blocks, Y_all = S_old^{-1} V_all, and the capacitance factorization.
            build_capacitance_setup(H_diag, B_rm, mu);
            compute_y_all(base);
            const RejectReason reason = factorize_capacitance();
            if (reason != RejectReason::None) return reject(reason);

            // Phase 6: size the solve-time buffers.
            finalize();
        } catch (const std::bad_alloc&) {
            return reject(RejectReason::OutOfMemory);
        }
        return Outcome::Updated;
    }

    // The inputs must describe G = [A; B(active_W, :)] with the snapshot's column count and number of W rows.
    bool consistent(const SpMat& G, const Vec& H_diag, const BoolArr& active_W, const RowMajorSpMat& B_rm) const {
        const Eigen::Index n = G_old_.cols();
        return M_rows_ >= 0
            && G.cols() == n && H_diag.size() == n && H_old_.size() == n && B_rm.cols() == n
            && active_W.size() == active_W_old_.size() && active_W.size() == B_rm.rows()
            && G.rows() == M_rows_ + static_cast<Eigen::Index>(active_W.count())
            && G.rows() > 0;
    }

    // Phase 2: rows of G are A's M_rows_ rows, then B's active_W rows in increasing order, so W rows are
    // deleted, retained or added by comparing active_W with the snapshot's. Changed columns are those whose
    // H entry differs from the snapshot's and whose snapshot column is nonempty (an empty column of G_old
    // contributes nothing to S_old; its effect on added rows is already in V_+ and W_+).
    void classify(const Vec& H_diag, const BoolArr& active_W) {
        s_old_ = static_cast<int>(G_old_.rows());
        const int l = static_cast<int>(active_W.size());
        const int n = static_cast<int>(G_old_.cols());

        deleted_old_rows_.clear();
        retained_old_rows_.clear();
        retained_new_rows_.clear();
        added_new_rows_.clear();
        added_W_src_.clear();
        changed_cols_.clear();

        int old_pos = M_rows_, new_pos = M_rows_;
        for (int i = 0; i < l; ++i) {
            const bool in_old = active_W_old_(i), in_new = active_W(i);
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

        const auto* outer = G_old_.outerIndexPtr();
        for (int j = 0; j < n; ++j)
            if (H_diag(j) != H_old_(j) && outer[j + 1] > outer[j]) changed_cols_.push_back(j);

        h_ = static_cast<int>(deleted_old_rows_.size());
        p_ = static_cast<int>(changed_cols_.size());
        q_ = static_cast<int>(added_new_rows_.size());
    }

    // Phase 3: M_sub (rank x rank) and V_plus_ (s_old_ x q_).
    void build_capacitance_setup(const Vec& H_diag, const RowMajorSpMat& B_rm, T mu) {
        const int rank = h_ + p_ + q_;
        M_sub_.setZero(rank, rank);

        // Block 2: -C^{-1} (diagonal p_ x p_), C_jj = 1/H_new(j) - 1/H_old(j).
        for (int j = 0; j < p_; ++j) {
            const int col = changed_cols_[j];
            const T h_new = H_diag(col), h_old = H_old_(col);
            M_sub_(h_ + j, h_ + j) = h_old * h_new / (h_new - h_old);
        }

        // V_plus_[:, j] = G_old E_new b_j^T = sum_{i in b_j} (b_ji / H_diag(i)) G_old[:, i].
        // Block 3: W_+ = B_+ E_new B_+^T + (1/mu) I (q_ x q_).
        V_plus_.setZero(s_old_, q_);
        e_new_b_.setZero(G_old_.cols()); // holds E_new b_j^T for one j at a time; zero elsewhere
        for (int j = 0; j < q_; ++j) {
            touched_.clear();
            for (typename RowMajorSpMat::InnerIterator it(B_rm, added_W_src_[j]); it; ++it) {
                const int col = static_cast<int>(it.col());
                e_new_b_(col) = it.value() / H_diag(col);
                touched_.push_back(col);
            }
            for (int col : touched_)
                for (typename SpMat::InnerIterator git(G_old_, col); git; ++git)
                    V_plus_(git.row(), j) += e_new_b_(col) * git.value();

            for (int k = j; k < q_; ++k) {
                T val = (j == k) ? T(1) / mu : T(0);
                for (typename RowMajorSpMat::InnerIterator it(B_rm, added_W_src_[k]); it; ++it)
                    val += it.value() * e_new_b_(it.col());
                M_sub_(h_ + p_ + j, h_ + p_ + k) = val;
                if (j != k) M_sub_(h_ + p_ + k, h_ + p_ + j) = val;
            }

            for (int col : touched_) e_new_b_(col) = T(0);
        }
    }

    // Phase 4: Y_all_ = S_old^{-1} [E_-, U, V_plus_] (s_old_ x rank), one base solve per column.
    void compute_y_all(const BaseSolve& base) {
        const int rank = h_ + p_ + q_;
        Y_all_.resize(s_old_, rank);
        rhs_scratch_.setZero(s_old_);

        // Cols 0..h_-1: S_old^{-1} e_{del_k}
        for (int k = 0; k < h_; ++k) {
            rhs_scratch_(deleted_old_rows_[k]) = T(1);
            base(rhs_scratch_, sol_scratch_);
            Y_all_.col(k) = sol_scratch_;
            rhs_scratch_(deleted_old_rows_[k]) = T(0);
        }

        // Cols h_..h_+p_-1: S_old^{-1} G_old[:, changed_j]
        for (int j = 0; j < p_; ++j) {
            for (typename SpMat::InnerIterator it(G_old_, changed_cols_[j]); it; ++it)
                rhs_scratch_(it.row()) = it.value();
            base(rhs_scratch_, sol_scratch_);
            Y_all_.col(h_ + j) = sol_scratch_;
            for (typename SpMat::InnerIterator it(G_old_, changed_cols_[j]); it; ++it)
                rhs_scratch_(it.row()) = T(0);
        }

        // Cols h_+p_..rank-1: S_old^{-1} V_plus_[:, j]
        for (int j = 0; j < q_; ++j) {
            rhs_scratch_ = V_plus_.col(j);
            base(rhs_scratch_, sol_scratch_);
            Y_all_.col(h_ + p_ + j) = sol_scratch_;
        }
    }

    // Phase 5: factor the capacitance matrix S_Lambda = M_sub - V_all^T Y_all_.
    // Its blocks live on very different scales (changed columns ~ H_old ~ 1/rho, added rows ~ |b|^2 / H,
    // up to rho |b|^2), so a rank test relative to the largest pivot would discard well-conditioned
    // updates. Equilibrate symmetrically first: tau_i = max(|M_sub_ii|, |(V_all^T Y_all)_ii|) is the
    // scale of the two terms combined in S_Lambda_ii, so genuine cancellation still shows up as a small
    // pivot, and a lone small pivot is caught by measuring against max(1, largest pivot).
    RejectReason factorize_capacitance() {
        const int rank = h_ + p_ + q_;
        VtY_.setZero(rank, rank);

        // Rows 0..h_-1: E_-^T Y_all = Y_all.row(del_k)
        for (int k = 0; k < h_; ++k)
            VtY_.row(k) = Y_all_.row(deleted_old_rows_[k]);

        // Rows h_..h_+p_-1: G_old[:, changed_j]^T Y_all
        for (int j = 0; j < p_; ++j)
            for (typename SpMat::InnerIterator it(G_old_, changed_cols_[j]); it; ++it)
                VtY_.row(h_ + j) += it.value() * Y_all_.row(it.row());

        // Rows h_+p_..rank-1: V_plus_^T Y_all
        if (q_ > 0)
            VtY_.bottomRows(q_).noalias() = V_plus_.transpose() * Y_all_;

        Mat S_Lambda = M_sub_ - VtY_;
        if (!S_Lambda.allFinite()) return RejectReason::NonFinite;

        eq_scale_.resize(rank);
        for (int i = 0; i < rank; ++i) {
            const T tau = std::max(std::abs(M_sub_(i, i)), std::abs(VtY_(i, i)));
            if (!(tau > T(0))) return RejectReason::SingularCapacitance;
            eq_scale_(i) = T(1) / std::sqrt(tau);
        }
        S_Lambda = eq_scale_.asDiagonal() * S_Lambda * eq_scale_.asDiagonal();

        lu_.compute(S_Lambda);
        const Vec pivots = lu_.matrixLU().diagonal().cwiseAbs();
        const T pivot_tol = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(T(1), pivots.maxCoeff());
        if (!(pivots.minCoeff() > pivot_tol)) return RejectReason::SingularCapacitance;
        return RejectReason::None;
    }

    // Phase 6: size the solve-time buffers.
    void finalize() {
        const int rank = h_ + p_ + q_;
        s_new_ = s_old_ - h_ + q_;
        r_pad_.setZero(s_old_);
        u_base_.resize(s_old_);
        Lambda_.resize(rank);
        lambda_.resize(rank);
        z_base_.resize(s_old_);
    }

    // ------ Snapshot ------
    int  M_rows_ = -1;
    bool has_snapshot_ = false;
    bool base_is_ldlt_old_ = false;
    SpMat   G_old_;
    Vec     H_old_;
    BoolArr active_W_old_;
    T mu_old_ = T(1), rho_old_ = T(1);

    // ------ Update state (output of setup(), consumed by solve()) ------
    bool active_ = false;
    int s_old_ = 0, s_new_ = 0, h_ = 0, p_ = 0, q_ = 0;
    std::vector<int> deleted_old_rows_;
    std::vector<int> retained_old_rows_;
    std::vector<int> retained_new_rows_;
    std::vector<int> added_new_rows_;
    std::vector<int> added_W_src_;   // B row index of each added row
    std::vector<int> changed_cols_;  // columns with H != H_old
    Mat M_sub_;
    Mat V_plus_;
    Mat Y_all_;
    Mat VtY_;
    Eigen::FullPivLU<Mat> lu_;       // of the equilibrated capacitance matrix D S_Lambda D
    Vec eq_scale_;                   // D's diagonal

    // ------ Failure tracking and diagnostics ------
    int fail_streak_ = 0;
    RejectReason last_reject_reason_ = RejectReason::None;
    int last_rank_ = 0;
    int update_count_ = 0;
    int reuse_count_ = 0;
    std::array<int, kNumRejectReasons> reject_counts_{};

    // ------ Scratch ------
    Vec e_new_b_;                    // size n; build_capacitance_setup()
    std::vector<int> touched_;       // columns of e_new_b_ set for the current added row
    Vec rhs_scratch_, sol_scratch_;  // size s_old_; compute_y_all()
    Vec r_pad_, u_base_, Lambda_, lambda_, z_base_; // solve()
};
