#include "schur_smw_update.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <limits>
#include <new>
#include <random>
#include <vector>

namespace {

using Smw = SchurSmwUpdate<double>;
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using SpMat = Eigen::SparseMatrix<double>;
using RowMajorSpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;
using Outcome = Smw::Outcome;
using Reason = Smw::RejectReason;

constexpr double kTol = 1e-9;

SpMat DenseToSparse(const Mat& dense) {
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < dense.rows(); ++i)
    for (int j = 0; j < dense.cols(); ++j)
      if (dense(i, j) != 0.0) trips.emplace_back(i, j, dense(i, j));
  SpMat sp(dense.rows(), dense.cols());
  sp.setFromTriplets(trips.begin(), trips.end());
  sp.makeCompressed();
  return sp;
}

RowMajorSpMat DenseToSparseRowMajor(const Mat& dense) {
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < dense.rows(); ++i)
    for (int j = 0; j < dense.cols(); ++j)
      if (dense(i, j) != 0.0) trips.emplace_back(i, j, dense(i, j));
  RowMajorSpMat sp(dense.rows(), dense.cols());
  sp.setFromTriplets(trips.begin(), trips.end());
  sp.makeCompressed();
  return sp;
}

BoolArr ToBoolArr(const std::vector<bool>& v) {
  BoolArr arr(static_cast<int>(v.size()));
  for (std::size_t i = 0; i < v.size(); ++i) arr(static_cast<int>(i)) = v[i];
  return arr;
}

// H = Q + mu (1 - active_K) + 1/rho, as SSN::prepare_newton_system() builds it.
Vec MakeH(const Vec& Q, const BoolArr& active_K, double mu, double rho) {
  Vec H(Q.size());
  for (int i = 0; i < Q.size(); ++i) H(i) = Q(i) + (active_K(i) ? 0.0 : mu) + 1.0 / rho;
  return H;
}

// S = G H^{-1} G^T + (1/mu) I over all columns, computed independently via dense linear algebra.
Mat DenseExactSchur(const Mat& G, const Vec& H, double mu) {
  const int s = static_cast<int>(G.rows());
  return G * H.cwiseInverse().asDiagonal() * G.transpose() + (1.0 / mu) * Mat::Identity(s, s);
}

// G = [A; rows of B in active_W], in increasing row order (as SSN::rebuild_G() stacks it).
struct Problem {
  Mat A, B;

  int M() const { return static_cast<int>(A.rows()); }
  RowMajorSpMat B_rm() const { return DenseToSparseRowMajor(B); }

  Mat StackG(const BoolArr& active_W) const {
    Mat G(M() + active_W.count(), A.cols());
    G.topRows(M()) = A;
    int row = M();
    for (int i = 0; i < B.rows(); ++i)
      if (active_W(i)) G.row(row++) = B.row(i);
    return G;
  }
};

// The caller's factorization of the snapshot matrix: Cholesky on S, or LDLT on K = [-H, G^T; G, (1/mu) I],
// exposed as the BaseSolve that SchurSmwUpdate expects.
class BaseFactorization {
 public:
  BaseFactorization(const Mat& G, const Vec& H, double mu, bool ldlt) : ldlt_(ldlt) {
    if (ldlt) {
      const int n = static_cast<int>(G.cols()), s = static_cast<int>(G.rows());
      Mat K = Mat::Zero(n + s, n + s);
      K.topLeftCorner(n, n).diagonal() = -H;
      K.topRightCorner(n, s) = G.transpose();
      K.bottomLeftCorner(s, n) = G;
      K.bottomRightCorner(s, s) = (1.0 / mu) * Mat::Identity(s, s);
      K_ = DenseToSparse(K);
      ldl_.compute(K_);
      EXPECT_EQ(ldl_.info(), Eigen::Success);
    } else {
      S_ = DenseToSparse(DenseExactSchur(G, H, mu));
      llt_.compute(S_);
      EXPECT_EQ(llt_.info(), Eigen::Success);
    }
  }

  Smw::BaseSolve Solve() {
    return [this](const Vec& v, Vec& out) {
      if (ldlt_) {
        Vec pad = Vec::Zero(K_.rows());
        pad.tail(v.size()) = v;
        out = ldl_.solve(pad).tail(v.size());
      } else {
        out = llt_.solve(v);
      }
    };
  }

 private:
  bool ldlt_;
  SpMat S_, K_;
  Eigen::SimplicialLLT<SpMat> llt_;
  Eigen::SimplicialLDLT<SpMat> ldl_;
};

Vec TestRhs(int s) {
  Vec b(s);
  for (int i = 0; i < s; ++i) b(i) = 1.0 + 0.5 * i - 0.1 * i * i;
  return b;
}

// Solves S_new z = b through the active update and compares it with a dense solve of the exact S_new.
void ExpectSolveMatchesDense(Smw& smw, BaseFactorization& base, const Mat& G_new, const Vec& H_new, double mu,
                             double tol = kTol) {
  ASSERT_TRUE(smw.active());
  ASSERT_EQ(smw.rows(), G_new.rows());
  const Vec b = TestRhs(static_cast<int>(G_new.rows()));
  Vec got;
  smw.solve(b, base.Solve(), got);
  const Vec want = DenseExactSchur(G_new, H_new, mu).ldlt().solve(b);
  EXPECT_LE((got - want).norm(), tol * want.norm()) << "got\n" << got.transpose() << "\nwant\n" << want.transpose();
}

// A small fixed problem: N = 4 columns, one equality row, three candidate W rows.
struct Fixture {
  Problem p{(Mat(1, 4) << 1.0, 1.0, 1.0, 1.0).finished(),
            (Mat(3, 4) << 1.0, 0.0, 2.0, 0.0,
                          0.0, 1.0, 0.0, -1.0,
                          0.5, 0.0, 0.0, 3.0).finished()};
  Vec Q = (Vec(4) << 1.0, 0.5, 2.0, 0.0).finished();
  double mu = 5.0;
  double rho = 3.0;  // distinct from mu so a mu/rho argument swap would be caught

  // Takes a snapshot of (active_K, active_W) and returns its base factorization.
  BaseFactorization Snapshot(Smw& smw, const BoolArr& active_K, const BoolArr& active_W, bool ldlt) const {
    const Mat G = p.StackG(active_W);
    const Vec H = MakeH(Q, active_K, mu, rho);
    smw.set_num_equality_rows(p.M());
    smw.snapshot(DenseToSparse(G), H, active_W, mu, rho, ldlt);
    return BaseFactorization(G, H, mu, ldlt);
  }

  // Updates to (active_K, active_W); returns the outcome and checks the solve when Updated.
  Outcome UpdateAndCheck(Smw& smw, BaseFactorization& base, const BoolArr& active_K, const BoolArr& active_W,
                         bool ldlt) const {
    const Mat G = p.StackG(active_W);
    const Vec H = MakeH(Q, active_K, mu, rho);
    const Outcome out = smw.setup(DenseToSparse(G), H, active_W, p.B_rm(), mu, rho, ldlt, base.Solve());
    if (out == Outcome::Updated) ExpectSolveMatchesDense(smw, base, G, H, mu);
    return out;
  }
};

}  // namespace

// Test-only peer granting access to SchurSmwUpdate's private scratch buffers.
struct SchurSmwUpdateTestPeer {
  static Smw::Vec& e_new_b(Smw& s) { return s.e_new_b_; }
  static Smw::Vec& rhs_scratch(Smw& s) { return s.rhs_scratch_; }
  static Smw::Vec& sol_scratch(Smw& s) { return s.sol_scratch_; }
  static const std::vector<int>& changed_cols(const Smw& s) { return s.changed_cols_; }
};

// ===================== single deltas =====================

TEST(SchurSmwUpdateSingleDelta, ColumnLeavingActiveKMatchesDenseForBothBases) {
  for (bool ldlt : {false, true}) {
    Fixture f;
    Smw smw;
    const BoolArr W = ToBoolArr({true, false, true});
    BaseFactorization base = f.Snapshot(smw, ToBoolArr({true, true, true, true}), W, ldlt);
    EXPECT_EQ(f.UpdateAndCheck(smw, base, ToBoolArr({true, false, true, true}), W, ldlt), Outcome::Updated);
    EXPECT_EQ(smw.last_rank(), 1);
    EXPECT_EQ(smw.last_p(), 1);
    EXPECT_EQ(smw.last_reject_reason(), Reason::None);
  }
}

TEST(SchurSmwUpdateSingleDelta, ColumnEnteringActiveKMatchesDenseForBothBases) {
  for (bool ldlt : {false, true}) {
    Fixture f;
    Smw smw;
    const BoolArr W = ToBoolArr({false, true, true});
    BaseFactorization base = f.Snapshot(smw, ToBoolArr({false, false, true, false}), W, ldlt);
    EXPECT_EQ(f.UpdateAndCheck(smw, base, ToBoolArr({true, false, true, false}), W, ldlt), Outcome::Updated);
    EXPECT_EQ(smw.last_rank(), 1);
  }
}

TEST(SchurSmwUpdateSingleDelta, AddedWRowMatchesDenseForBothBases) {
  for (bool ldlt : {false, true}) {
    Fixture f;
    Smw smw;
    const BoolArr K = ToBoolArr({true, false, true, true});
    BaseFactorization base = f.Snapshot(smw, K, ToBoolArr({true, false, false}), ldlt);
    EXPECT_EQ(f.UpdateAndCheck(smw, base, K, ToBoolArr({true, true, false}), ldlt), Outcome::Updated);
    EXPECT_EQ(smw.last_q(), 1);
    EXPECT_EQ(smw.last_rank(), 1);
  }
}

TEST(SchurSmwUpdateSingleDelta, DeletedWRowMatchesDenseForBothBases) {
  for (bool ldlt : {false, true}) {
    Fixture f;
    Smw smw;
    const BoolArr K = ToBoolArr({true, true, false, true});
    BaseFactorization base = f.Snapshot(smw, K, ToBoolArr({true, true, true}), ldlt);
    EXPECT_EQ(f.UpdateAndCheck(smw, base, K, ToBoolArr({true, false, true}), ldlt), Outcome::Updated);
    EXPECT_EQ(smw.last_h(), 1);
    EXPECT_EQ(smw.last_rank(), 1);
  }
}

TEST(SchurSmwUpdateSingleDelta, AllWRowsDeletedLeavesOnlyTheEqualityRows) {
  for (bool ldlt : {false, true}) {
    Fixture f;
    Smw smw;
    const BoolArr K = ToBoolArr({true, true, true, true});
    BaseFactorization base = f.Snapshot(smw, K, ToBoolArr({true, true, true}), ldlt);
    EXPECT_EQ(f.UpdateAndCheck(smw, base, K, ToBoolArr({false, false, false}), ldlt), Outcome::Updated);
    EXPECT_EQ(smw.rows(), 1);
  }
}

// ===================== mixed and cumulative deltas =====================

TEST(SchurSmwUpdateMixedDelta, ColumnChangeAddAndDeleteTogetherKeepGNewRowOrder) {
  // W row 0 deleted, row 1 added, row 2 retained: the retained row moves from position 2 to 2 while the
  // added row lands at position 1, so a row-order mix-up would show in the dense comparison.
  for (bool ldlt : {false, true}) {
    Fixture f;
    Smw smw;
    BaseFactorization base =
        f.Snapshot(smw, ToBoolArr({true, true, false, true}), ToBoolArr({true, false, true}), ldlt);
    EXPECT_EQ(f.UpdateAndCheck(smw, base, ToBoolArr({false, true, false, true}), ToBoolArr({false, true, true}),
                               ldlt),
              Outcome::Updated);
    EXPECT_EQ(smw.last_h(), 1);
    EXPECT_EQ(smw.last_p(), 1);
    EXPECT_EQ(smw.last_q(), 1);
  }
}

TEST(SchurSmwUpdateMixedDelta, ConsecutiveUpdatesAreMeasuredAgainstTheOriginalSnapshot) {
  for (bool ldlt : {false, true}) {
    Fixture f;
    Smw smw;
    BaseFactorization base =
        f.Snapshot(smw, ToBoolArr({true, true, true, true}), ToBoolArr({false, false, false}), ldlt);

    EXPECT_EQ(f.UpdateAndCheck(smw, base, ToBoolArr({false, true, true, true}), ToBoolArr({false, false, false}),
                               ldlt),
              Outcome::Updated);
    EXPECT_EQ(smw.last_rank(), 1);

    // Second step adds a column change and a W row on top: rank 3 relative to the snapshot, not 2.
    EXPECT_EQ(f.UpdateAndCheck(smw, base, ToBoolArr({false, false, true, true}), ToBoolArr({false, false, true}),
                               ldlt),
              Outcome::Updated);
    EXPECT_EQ(smw.last_rank(), 3);
    EXPECT_EQ(smw.update_count(), 2);
  }
}

TEST(SchurSmwUpdateMixedDelta, SolveCanBeRepeatedWithDifferentRightHandSides) {
  Fixture f;
  Smw smw;
  BaseFactorization base =
      f.Snapshot(smw, ToBoolArr({true, false, true, true}), ToBoolArr({true, false, false}), false);
  const BoolArr K2 = ToBoolArr({true, true, true, true});
  const BoolArr W2 = ToBoolArr({true, false, true});
  const Mat G2 = f.p.StackG(W2);
  const Vec H2 = MakeH(f.Q, K2, f.mu, f.rho);
  ASSERT_EQ(smw.setup(DenseToSparse(G2), H2, W2, f.p.B_rm(), f.mu, f.rho, false, base.Solve()), Outcome::Updated);

  const Mat S2 = DenseExactSchur(G2, H2, f.mu);
  for (int trial = 0; trial < 3; ++trial) {
    const Vec b = Vec::LinSpaced(G2.rows(), -1.0 - trial, 2.0 + trial);
    Vec got;
    smw.solve(b, base.Solve(), got);
    const Vec want = S2.ldlt().solve(b);
    EXPECT_LE((got - want).norm(), kTol * want.norm());
  }
}

// ===================== rank 0 and threshold =====================

TEST(SchurSmwUpdateRank, ReturnToTheSnapshotReusesItAndDeactivates) {
  Fixture f;
  Smw smw;
  const BoolArr K1 = ToBoolArr({true, false, true, true});
  const BoolArr W1 = ToBoolArr({true, false, true});
  BaseFactorization base = f.Snapshot(smw, K1, W1, false);

  ASSERT_EQ(f.UpdateAndCheck(smw, base, ToBoolArr({true, true, true, true}), W1, false), Outcome::Updated);
  EXPECT_TRUE(smw.active());

  EXPECT_EQ(f.UpdateAndCheck(smw, base, K1, W1, false), Outcome::ReusedSnapshot);
  EXPECT_FALSE(smw.active());
  EXPECT_EQ(smw.last_rank(), 0);
  EXPECT_EQ(smw.reuse_count(), 1);
  EXPECT_EQ(smw.last_reject_reason(), Reason::None);
}

TEST(SchurSmwUpdateRank, ChangedColumnThatIsEmptyInTheSnapshotDoesNotCount) {
  // Column 2 is nonzero only in B row 0, which is not in the snapshot's G. Changing H(2) leaves S unchanged
  // (rank 0); adding row 0 at the same time needs only the added row (rank 1), with H_new(2) in W_+.
  Fixture g;
  g.p.A(0, 2) = 0.0;
  Smw smw;
  const BoolArr W1 = ToBoolArr({false, true, false});
  BaseFactorization base = g.Snapshot(smw, ToBoolArr({true, true, true, true}), W1, false);
  const BoolArr K2 = ToBoolArr({true, true, false, true});

  EXPECT_EQ(g.UpdateAndCheck(smw, base, K2, W1, false), Outcome::ReusedSnapshot);
  EXPECT_TRUE(SchurSmwUpdateTestPeer::changed_cols(smw).empty());

  EXPECT_EQ(g.UpdateAndCheck(smw, base, K2, ToBoolArr({true, true, false}), false), Outcome::Updated);
  EXPECT_EQ(smw.last_rank(), 1);
  EXPECT_EQ(smw.last_p(), 0);
}

TEST(SchurSmwUpdateRank, RankFiftyIsAcceptedAndFiftyOneRejected) {
  const int n = 60;
  Problem p{Mat::Ones(1, n), Mat::Zero(0, n)};
  const Vec Q = Vec::Ones(n);
  const double mu = 2.0, rho = 3.0;
  const BoolArr W = BoolArr::Constant(0, false);
  const RowMajorSpMat B_rm = p.B_rm();

  for (int flips : {50, 51}) {
    Smw smw;
    smw.set_num_equality_rows(1);
    const BoolArr K1 = BoolArr::Constant(n, true);
    const Mat G = p.StackG(W);
    const Vec H1 = MakeH(Q, K1, mu, rho);
    smw.snapshot(DenseToSparse(G), H1, W, mu, rho, false);
    BaseFactorization base(G, H1, mu, false);

    BoolArr K2 = K1;
    for (int j = 0; j < flips; ++j) K2(j) = false;
    const Vec H2 = MakeH(Q, K2, mu, rho);
    const Outcome out = smw.setup(DenseToSparse(G), H2, W, B_rm, mu, rho, false, base.Solve());
    EXPECT_EQ(smw.last_rank(), flips);
    if (flips <= Smw::kMaxRank) {
      ASSERT_EQ(out, Outcome::Updated);
      ExpectSolveMatchesDense(smw, base, G, H2, mu);
    } else {
      EXPECT_EQ(out, Outcome::Rejected);
      EXPECT_EQ(smw.last_reject_reason(), Reason::RankExceedsThreshold);
      EXPECT_FALSE(smw.active());
    }
  }
}

// ===================== gate =====================

TEST(SchurSmwUpdateGate, RejectsWithoutSnapshotAndAfterInvalidate) {
  Fixture f;
  Smw smw;
  smw.set_num_equality_rows(f.p.M());
  const BoolArr K = ToBoolArr({true, true, true, true});
  const BoolArr W = ToBoolArr({true, false, false});
  const Mat G = f.p.StackG(W);
  const Vec H = MakeH(f.Q, K, f.mu, f.rho);
  BaseFactorization base(G, H, f.mu, false);

  EXPECT_EQ(smw.setup(DenseToSparse(G), H, W, f.p.B_rm(), f.mu, f.rho, false, base.Solve()), Outcome::Rejected);
  EXPECT_EQ(smw.last_reject_reason(), Reason::NoSnapshot);

  smw.snapshot(DenseToSparse(G), H, W, f.mu, f.rho, false);
  EXPECT_TRUE(smw.has_snapshot());
  smw.invalidate();
  EXPECT_FALSE(smw.has_snapshot());
  EXPECT_EQ(smw.setup(DenseToSparse(G), H, W, f.p.B_rm(), f.mu, f.rho, false, base.Solve()), Outcome::Rejected);
  EXPECT_EQ(smw.last_reject_reason(), Reason::NoSnapshot);
  EXPECT_EQ(smw.reject_count(Reason::NoSnapshot), 2);
}

TEST(SchurSmwUpdateGate, InconsistentSnapshotIsNotTaken) {
  Fixture f;
  Smw smw;
  smw.set_num_equality_rows(f.p.M());
  const BoolArr W = ToBoolArr({true, false, false});
  const Mat G_other = f.p.StackG(ToBoolArr({true, true, false}));  // two W rows, but active_W has one
  smw.snapshot(DenseToSparse(G_other), MakeH(f.Q, ToBoolArr({true, true, true, true}), f.mu, f.rho), W, f.mu,
               f.rho, false);
  EXPECT_FALSE(smw.has_snapshot());
}

TEST(SchurSmwUpdateGate, RejectsMuDriftRhoDriftAndBaseChange) {
  Fixture f;
  Smw smw;
  const BoolArr K1 = ToBoolArr({true, true, true, true});
  const BoolArr W = ToBoolArr({true, false, true});
  BaseFactorization base = f.Snapshot(smw, K1, W, false);
  const BoolArr K2 = ToBoolArr({true, false, true, true});
  const SpMat G = DenseToSparse(f.p.StackG(W));
  const RowMajorSpMat B_rm = f.p.B_rm();

  EXPECT_EQ(smw.setup(G, MakeH(f.Q, K2, 2 * f.mu, f.rho), W, B_rm, 2 * f.mu, f.rho, false, base.Solve()),
            Outcome::Rejected);
  EXPECT_EQ(smw.last_reject_reason(), Reason::MuChangedSinceSnapshot);

  EXPECT_EQ(smw.setup(G, MakeH(f.Q, K2, f.mu, 2 * f.rho), W, B_rm, f.mu, 2 * f.rho, false, base.Solve()),
            Outcome::Rejected);
  EXPECT_EQ(smw.last_reject_reason(), Reason::RhoChangedSinceSnapshot);

  EXPECT_EQ(smw.setup(G, MakeH(f.Q, K2, f.mu, f.rho), W, B_rm, f.mu, f.rho, /*base_is_ldlt=*/true, base.Solve()),
            Outcome::Rejected);
  EXPECT_EQ(smw.last_reject_reason(), Reason::BaseChanged);
  EXPECT_FALSE(smw.active());
}

TEST(SchurSmwUpdateGate, RejectsInconsistentInputs) {
  Fixture f;
  Smw smw;
  const BoolArr K = ToBoolArr({true, true, true, true});
  const BoolArr W = ToBoolArr({true, false, true});
  BaseFactorization base = f.Snapshot(smw, K, W, false);
  const Vec H = MakeH(f.Q, K, f.mu, f.rho);
  const SpMat G = DenseToSparse(f.p.StackG(W));
  const RowMajorSpMat B_rm = f.p.B_rm();

  // G's row count does not match active_W.
  const SpMat G_wrong = DenseToSparse(f.p.StackG(ToBoolArr({true, true, true})));
  EXPECT_EQ(smw.setup(G_wrong, H, W, B_rm, f.mu, f.rho, false, base.Solve()), Outcome::Rejected);
  EXPECT_EQ(smw.last_reject_reason(), Reason::InconsistentData);

  // H has the wrong length.
  EXPECT_EQ(smw.setup(G, Vec::Ones(3), W, B_rm, f.mu, f.rho, false, base.Solve()), Outcome::Rejected);
  EXPECT_EQ(smw.last_reject_reason(), Reason::InconsistentData);

  // active_W does not cover B's rows.
  const BoolArr W_short = ToBoolArr({true, false});
  EXPECT_EQ(smw.setup(DenseToSparse(f.p.StackG(ToBoolArr({true, false, false}))), H, W_short, B_rm, f.mu, f.rho,
                      false, base.Solve()),
            Outcome::Rejected);
  EXPECT_EQ(smw.last_reject_reason(), Reason::InconsistentData);
  EXPECT_EQ(smw.reject_count(Reason::InconsistentData), 3);
}

// ===================== capacitance =====================

TEST(SchurSmwUpdateCapacitance, EquilibrationKeepsAWellConditionedMixedUpdateAtLargeRho) {
  // rho = 1e7 (where PMM starts it): the column-change block of the capacitance is ~1/rho while the added-row
  // block is ~rho |b|^2. Unscaled, a rank test relative to the largest pivot calls this rank 1, although
  // S_new = [[2e7, 1e7], [1e7, 1e7]] + 0.01 I has condition number ~7.
  for (bool ldlt : {false, true}) {
    Problem p{(Mat(1, 3) << 1.0, 1.0, 1.0).finished(), (Mat(1, 3) << 0.0, 1.0, 0.0).finished()};
    const Vec Q = Vec::Zero(3);
    const double mu = 1e2, rho = 1e7;
    Smw smw;
    smw.set_num_equality_rows(1);
    const BoolArr K1 = ToBoolArr({true, true, true});
    const BoolArr W1 = ToBoolArr({false});
    const Mat G1 = p.StackG(W1);
    const Vec H1 = MakeH(Q, K1, mu, rho);
    smw.snapshot(DenseToSparse(G1), H1, W1, mu, rho, ldlt);
    BaseFactorization base(G1, H1, mu, ldlt);

    const BoolArr K2 = ToBoolArr({false, true, true});
    const BoolArr W2 = ToBoolArr({true});
    const Mat G2 = p.StackG(W2);
    const Vec H2 = MakeH(Q, K2, mu, rho);
    ASSERT_EQ(smw.setup(DenseToSparse(G2), H2, W2, p.B_rm(), mu, rho, ldlt, base.Solve()), Outcome::Updated)
        << "reject reason " << static_cast<int>(smw.last_reject_reason());
    EXPECT_EQ(smw.last_rank(), 2);
    ExpectSolveMatchesDense(smw, base, G2, H2, mu, 1e-8);
  }
}

TEST(SchurSmwUpdateCapacitance, DuplicateWRowWithTinyRegularizationIsSingular) {
  // Adding a copy of an active W row makes S_new singular up to the (1/mu) I term; with mu = 1e10 the
  // equilibrated capacitance pivot is ~2/mu, far below sqrt(eps).
  Problem p{(Mat(1, 3) << 1.0, 1.0, 1.0).finished(),
            (Mat(2, 3) << 1.0, 0.0, 0.0,
                          1.0, 0.0, 0.0).finished()};
  const Vec Q = Vec::Ones(3);
  const double mu = 1e10, rho = 3.0;
  Smw smw;
  smw.set_num_equality_rows(1);
  const BoolArr K = ToBoolArr({true, true, true});
  const BoolArr W1 = ToBoolArr({true, false});
  const Mat G1 = p.StackG(W1);
  const Vec H = MakeH(Q, K, mu, rho);
  smw.snapshot(DenseToSparse(G1), H, W1, mu, rho, false);
  BaseFactorization base(G1, H, mu, false);

  const BoolArr W2 = ToBoolArr({true, true});
  EXPECT_EQ(smw.setup(DenseToSparse(p.StackG(W2)), H, W2, p.B_rm(), mu, rho, false, base.Solve()),
            Outcome::Rejected);
  EXPECT_EQ(smw.last_reject_reason(), Reason::SingularCapacitance);
  EXPECT_FALSE(smw.active());
}

TEST(SchurSmwUpdateCapacitance, NonFiniteBaseSolveIsRejected) {
  Fixture f;
  Smw smw;
  const BoolArr W = ToBoolArr({true, false, true});
  BaseFactorization base = f.Snapshot(smw, ToBoolArr({true, true, true, true}), W, false);
  const Vec H2 = MakeH(f.Q, ToBoolArr({true, false, true, true}), f.mu, f.rho);
  const Smw::BaseSolve nan_solve = [](const Vec& v, Vec& out) {
    out = Vec::Constant(v.size(), std::numeric_limits<double>::quiet_NaN());
  };
  EXPECT_EQ(smw.setup(DenseToSparse(f.p.StackG(W)), H2, W, f.p.B_rm(), f.mu, f.rho, false, nan_solve),
            Outcome::Rejected);
  EXPECT_EQ(smw.last_reject_reason(), Reason::NonFinite);
}

TEST(SchurSmwUpdateCapacitance, BadAllocDuringSetupIsRejected) {
  Fixture f;
  Smw smw;
  const BoolArr W = ToBoolArr({true, false, true});
  BaseFactorization base = f.Snapshot(smw, ToBoolArr({true, true, true, true}), W, false);
  const Vec H2 = MakeH(f.Q, ToBoolArr({true, false, true, true}), f.mu, f.rho);
  const Smw::BaseSolve throwing_solve = [](const Vec&, Vec&) { throw std::bad_alloc(); };
  EXPECT_EQ(smw.setup(DenseToSparse(f.p.StackG(W)), H2, W, f.p.B_rm(), f.mu, f.rho, false, throwing_solve),
            Outcome::Rejected);
  EXPECT_EQ(smw.last_reject_reason(), Reason::OutOfMemory);
  EXPECT_FALSE(smw.active());

  // The snapshot survives, so the next attempt with a working base solve succeeds.
  EXPECT_EQ(f.UpdateAndCheck(smw, base, ToBoolArr({true, false, true, true}), W, false), Outcome::Updated);
}

// ===================== fail streak =====================

TEST(SchurSmwUpdateFailStreak, FiveFailuresSuppressUntilReset) {
  Fixture f;
  Smw smw;
  const BoolArr W = ToBoolArr({true, false, true});
  BaseFactorization base = f.Snapshot(smw, ToBoolArr({true, true, true, true}), W, false);
  const BoolArr K2 = ToBoolArr({true, false, true, true});

  for (int i = 0; i < Smw::kMaxFailStreak - 1; ++i) smw.record_failure();
  EXPECT_FALSE(smw.suppressed());
  EXPECT_EQ(f.UpdateAndCheck(smw, base, K2, W, false), Outcome::Updated);

  smw.record_failure();
  EXPECT_TRUE(smw.suppressed());
  EXPECT_EQ(f.UpdateAndCheck(smw, base, K2, W, false), Outcome::Rejected);
  EXPECT_EQ(smw.last_reject_reason(), Reason::Suppressed);
  EXPECT_FALSE(smw.active());

  smw.reset_fail_streak();
  EXPECT_EQ(smw.fail_streak(), 0);
  EXPECT_EQ(f.UpdateAndCheck(smw, base, K2, W, false), Outcome::Updated);
}

// ===================== scratch reuse =====================

TEST(SchurSmwUpdateScratch, PoisonedScratchBuffersDoNotLeakIntoTheNextSetup) {
  Fixture f;
  Smw smw;
  BaseFactorization base =
      f.Snapshot(smw, ToBoolArr({true, true, true, true}), ToBoolArr({true, false, false}), false);
  ASSERT_EQ(f.UpdateAndCheck(smw, base, ToBoolArr({true, true, false, true}), ToBoolArr({true, true, false}), false),
            Outcome::Updated);

  SchurSmwUpdateTestPeer::e_new_b(smw).setConstant(1e30);
  SchurSmwUpdateTestPeer::rhs_scratch(smw).setConstant(-7.0);
  SchurSmwUpdateTestPeer::sol_scratch(smw).setConstant(3.0);

  EXPECT_EQ(f.UpdateAndCheck(smw, base, ToBoolArr({false, true, false, true}), ToBoolArr({false, true, true}), false),
            Outcome::Updated);
}

// ===================== randomized =====================

TEST(SchurSmwUpdateRandomized, SeededSequenceOfDeltasMatchesDenseForBothBases) {
  for (bool ldlt : {false, true}) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_real_distribution<double> q_dist(0.0, 2.0);
    std::uniform_int_distribution<int> col_dist(0, 49);
    std::uniform_int_distribution<int> row_dist(0, 24);

    const int n = 50, M = 8, l = 25;
    Problem p{Mat::Zero(M, n), Mat::Zero(l, n)};
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < n; ++j)
        if (prob(rng) < 0.3) p.A(i, j) = unit(rng);
    for (int i = 0; i < l; ++i)
      for (int j = 0; j < n; ++j)
        if (prob(rng) < 0.2) p.B(i, j) = unit(rng);
    const RowMajorSpMat B_rm = p.B_rm();
    Vec Q(n);
    for (int j = 0; j < n; ++j) Q(j) = q_dist(rng);
    const double mu = 10.0, rho = 100.0;

    BoolArr K(n), W(l);
    for (int j = 0; j < n; ++j) K(j) = prob(rng) < 0.7;
    for (int i = 0; i < l; ++i) W(i) = prob(rng) < 0.5;

    Smw smw;
    smw.set_num_equality_rows(M);
    const Mat G1 = p.StackG(W);
    const Vec H1 = MakeH(Q, K, mu, rho);
    smw.snapshot(DenseToSparse(G1), H1, W, mu, rho, ldlt);
    BaseFactorization base(G1, H1, mu, ldlt);

    for (int step = 0; step < 6; ++step) {
      for (int k = 0; k < 3; ++k) K(col_dist(rng)) ^= true;
      for (int k = 0; k < 2; ++k) W(row_dist(rng)) ^= true;
      const Mat G2 = p.StackG(W);
      const Vec H2 = MakeH(Q, K, mu, rho);
      const Outcome out = smw.setup(DenseToSparse(G2), H2, W, B_rm, mu, rho, ldlt, base.Solve());
      ASSERT_NE(out, Outcome::Rejected) << "step " << step << " reason " << static_cast<int>(smw.last_reject_reason());
      if (out == Outcome::Updated) ExpectSolveMatchesDense(smw, base, G2, H2, mu);
    }
  }
}
