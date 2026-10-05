#include "ksp_qp.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using Vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<double>;

constexpr double kTol = 1e-4;    // tolerance for end-to-end solve comparisons (solver tol default 1e-6)
constexpr double kTight = 1e-9;

SpMat DenseToSparse(const Eigen::MatrixXd& dense) {
  std::vector<Eigen::Triplet<double>> trips;
  for (int i = 0; i < dense.rows(); ++i)
    for (int j = 0; j < dense.cols(); ++j)
      if (dense(i, j) != 0.0) trips.emplace_back(i, j, dense(i, j));
  SpMat sp(dense.rows(), dense.cols());
  sp.setFromTriplets(trips.begin(), trips.end());
  sp.makeCompressed();
  return sp;
}

// Builds a Problem<double> via KSPQPdata's constructor (the only path production code uses),
// filling in every field explicitly.
Problem<double> MakeProblem(int n, int m, int l, const SpMat& Q, const SpMat& A, const SpMat& B,
                             const Vec& c, const Vec& b, double obj_const,
                             const Vec& lx, const Vec& ux, const Vec& lw, const Vec& uw,
                             double tol = 1e-8, int max_iter = 2000) {
  KSPQPdata<double> pd;
  pd.n = n;
  pd.m = m;
  pd.l = l;
  pd.Q = Q;
  pd.A = A;
  pd.B = B;
  pd.c = c;
  pd.b = b;
  pd.obj_const = obj_const;
  pd.lx = lx;
  pd.ux = ux;
  pd.lw = lw;
  pd.uw = uw;
  return Problem<double>(pd, tol, max_iter, /*time_limit=*/30.0, PrintWhen::NEVER, PrintWhat::NONE);
}

const double kInf = std::numeric_limits<double>::infinity();

}  // namespace

// ===================== get_Q_info =====================

TEST(GetQInfo, ZeroMatrixReportsQInfoZero) {
  SpMat Q(2, 2);  // no entries
  auto problem = MakeProblem(2, 0, 0, Q, SpMat(0, 2), SpMat(0, 2), Vec::Zero(2), Vec::Zero(0), 0.0,
                              Vec::Constant(2, -kInf), Vec::Constant(2, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  EXPECT_EQ(ns.Q_info, 0);
}

TEST(GetQInfo, DiagonalMatrixReportsQInfoOne) {
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 3.0, 0.0, 0.0, 4.0).finished());
  auto problem = MakeProblem(2, 0, 0, Q, SpMat(0, 2), SpMat(0, 2), Vec::Zero(2), Vec::Zero(0), 0.0,
                              Vec::Constant(2, -kInf), Vec::Constant(2, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  EXPECT_EQ(ns.Q_info, 1);
}

TEST(GetQInfo, GeneralMatrixReportsQInfoTwo) {
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 3.0, 1.0, 1.0, 4.0).finished());
  auto problem = MakeProblem(2, 0, 0, Q, SpMat(0, 2), SpMat(0, 2), Vec::Zero(2), Vec::Zero(0), 0.0,
                              Vec::Constant(2, -kInf), Vec::Constant(2, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  EXPECT_EQ(ns.Q_info, 2);
}

// ===================== determine_dimensions =====================
// Called directly on an already-constructed KSP_QP<double> built from a trivially valid problem.

namespace {

KSP_QP<double> MakeValidInstance() {
  SpMat Q = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  auto problem = MakeProblem(1, 0, 0, Q, SpMat(0, 1), SpMat(0, 1), Vec::Zero(1), Vec::Zero(0), 0.0,
                              Vec::Constant(1, -kInf), Vec::Constant(1, kInf), Vec(0), Vec(0));
  return KSP_QP<double>(problem);
}

}  // namespace

TEST(DetermineDimensions, NInfersFromCSizeWhenQIsZero) {
  KSP_QP<double> ns = MakeValidInstance();
  ASSERT_FALSE(ns.setup_failed);

  SpMat Q(0, 0);
  Vec c(3);
  c << 1.0, 2.0, 3.0;
  auto problem = MakeProblem(0, 0, 0, Q, SpMat(0, 0), SpMat(0, 0), c, Vec(0), 0.0, Vec(0), Vec(0),
                              Vec(0), Vec(0));
  ns.get_Q_info(problem.Q);
  ns.determine_dimensions(problem);
  EXPECT_EQ(ns.n, 3);
}

TEST(DetermineDimensions, NInfersFromAColsWhenCIsEmpty) {
  KSP_QP<double> ns = MakeValidInstance();
  SpMat Q(0, 0);
  SpMat A(2, 4);  // 2 rows, 4 cols; c empty
  auto problem = MakeProblem(0, 0, 0, Q, A, SpMat(0, 0), Vec(0), Vec::Zero(2), 0.0, Vec(0), Vec(0),
                              Vec(0), Vec(0));
  ns.get_Q_info(problem.Q);
  ns.determine_dimensions(problem);
  EXPECT_EQ(ns.n, 4);
}

TEST(DetermineDimensions, NInfersFromDiagonalSizeWhenQInfoIsOne) {
  KSP_QP<double> ns = MakeValidInstance();
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 5.0, 0.0, 0.0, 6.0).finished());
  auto problem = MakeProblem(0, 0, 0, Q, SpMat(0, 0), SpMat(0, 0), Vec(0), Vec(0), 0.0, Vec(0),
                              Vec(0), Vec(0), Vec(0));
  ns.get_Q_info(problem.Q);
  ASSERT_EQ(ns.Q_info, 1);
  ns.determine_dimensions(problem);
  EXPECT_EQ(ns.n, 2);
}

TEST(DetermineDimensions, NInfersFromQRowsWhenQInfoIsTwo) {
  KSP_QP<double> ns = MakeValidInstance();
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 5.0, 1.0, 1.0, 6.0).finished());
  auto problem = MakeProblem(0, 0, 0, Q, SpMat(0, 0), SpMat(0, 0), Vec(0), Vec(0), 0.0, Vec(0),
                              Vec(0), Vec(0), Vec(0));
  ns.get_Q_info(problem.Q);
  ASSERT_EQ(ns.Q_info, 2);
  ns.determine_dimensions(problem);
  EXPECT_EQ(ns.n, 2);
}

TEST(DetermineDimensions, ThrowsWhenNCannotBeInferredFromAnySource) {
  KSP_QP<double> ns = MakeValidInstance();
  SpMat Q(0, 0);
  auto problem = MakeProblem(0, 0, 0, Q, SpMat(0, 0), SpMat(0, 0), Vec(0), Vec(0), 0.0, Vec(0),
                              Vec(0), Vec(0), Vec(0));
  ns.get_Q_info(problem.Q);
  EXPECT_THROW(ns.determine_dimensions(problem), std::invalid_argument);
}

TEST(DetermineDimensions, MInfersFromARowsThenBSizeThenDefaultsToZero) {
  KSP_QP<double> ns = MakeValidInstance();
  SpMat Q(0, 0);
  {
    SpMat A(3, 2);
    auto problem = MakeProblem(0, 0, 0, Q, A, SpMat(0, 0), Vec::Zero(2), Vec(0), 0.0, Vec(0),
                                Vec(0), Vec(0), Vec(0));
    ns.get_Q_info(problem.Q);
    ns.determine_dimensions(problem);
    EXPECT_EQ(ns.m, 3);
  }
  {
    Vec c(2);
    c << 1.0, 1.0;
    Vec b(5);
    b.setZero();
    auto problem = MakeProblem(0, 0, 0, Q, SpMat(0, 0), SpMat(0, 0), c, b, 0.0, Vec(0), Vec(0),
                                Vec(0), Vec(0));
    ns.get_Q_info(problem.Q);
    ns.determine_dimensions(problem);
    EXPECT_EQ(ns.m, 5);
  }
  {
    Vec c(2);
    c << 1.0, 1.0;
    auto problem = MakeProblem(0, 0, 0, Q, SpMat(0, 0), SpMat(0, 0), c, Vec(0), 0.0, Vec(0), Vec(0),
                                Vec(0), Vec(0));
    ns.get_Q_info(problem.Q);
    ns.determine_dimensions(problem);
    EXPECT_EQ(ns.m, 0);
  }
}

TEST(DetermineDimensions, LInfersFromBRowsThenLwThenUwThenDefaultsToZero) {
  KSP_QP<double> ns = MakeValidInstance();
  SpMat Q(0, 0);
  {
    SpMat B(4, 2);
    auto problem = MakeProblem(0, 0, 0, Q, SpMat(0, 0), B, Vec::Zero(2), Vec(0), 0.0, Vec(0),
                                Vec(0), Vec(0), Vec(0));
    ns.get_Q_info(problem.Q);
    ns.determine_dimensions(problem);
    EXPECT_EQ(ns.l, 4);
  }
  {
    Vec c(2);
    c << 1.0, 1.0;
    Vec lw(6);
    lw.setZero();
    auto problem = MakeProblem(0, 0, 0, Q, SpMat(0, 0), SpMat(0, 0), c, Vec(0), 0.0, Vec(0), Vec(0),
                                lw, Vec(0));
    ns.get_Q_info(problem.Q);
    ns.determine_dimensions(problem);
    EXPECT_EQ(ns.l, 6);
  }
  {
    Vec c(2);
    c << 1.0, 1.0;
    auto problem = MakeProblem(0, 0, 0, Q, SpMat(0, 0), SpMat(0, 0), c, Vec(0), 0.0, Vec(0), Vec(0),
                                Vec(0), Vec(0));
    ns.get_Q_info(problem.Q);
    ns.determine_dimensions(problem);
    EXPECT_EQ(ns.l, 0);
  }
}

// ===================== check_dimensions =====================

TEST(CheckDimensions, ThrowsWhenAColsMismatchesN) {
  KSP_QP<double> ns = MakeValidInstance();  // ns.n == 1
  SpMat Q = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  SpMat bad_A(1, 2);  // 2 cols, but ns.n == 1
  auto problem = MakeProblem(1, 1, 0, Q, bad_A, SpMat(0, 1), Vec::Zero(1), Vec::Zero(1), 0.0,
                              Vec::Constant(1, -kInf), Vec::Constant(1, kInf), Vec(0), Vec(0));
  EXPECT_THROW(ns.check_dimensions(problem), std::invalid_argument);
}

TEST(CheckDimensions, ThrowsWhenCSizeMismatchesN) {
  KSP_QP<double> ns = MakeValidInstance();  // ns.n == 1
  SpMat Q = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  Vec bad_c(2);
  bad_c << 1.0, 2.0;
  auto problem = MakeProblem(1, 0, 0, Q, SpMat(0, 1), SpMat(0, 1), bad_c, Vec(0), 0.0,
                              Vec::Constant(1, -kInf), Vec::Constant(1, kInf), Vec(0), Vec(0));
  EXPECT_THROW(ns.check_dimensions(problem), std::invalid_argument);
}

TEST(CheckDimensions, ThrowsWhenBRowsMismatchesL) {
  KSP_QP<double> ns = MakeValidInstance();  // ns.n == 1, ns.l == 0
  SpMat Q = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  SpMat bad_B(2, 1);  // 2 rows, but ns.l == 0
  auto problem = MakeProblem(1, 0, 0, Q, SpMat(0, 1), bad_B, Vec::Zero(1), Vec(0), 0.0,
                              Vec::Constant(1, -kInf), Vec::Constant(1, kInf), Vec(0), Vec(0));
  EXPECT_THROW(ns.check_dimensions(problem), std::invalid_argument);
}

// ===================== check_bounds =====================

TEST(CheckBounds, ReturnsFalseWhenLxExceedsUx) {
  KSP_QP<double> ns = MakeValidInstance();
  ASSERT_FALSE(ns.setup_failed);
  ns.lx_orig(0) = 5.0;
  ns.ux_orig(0) = 1.0;
  EXPECT_FALSE(ns.check_bounds());
}

TEST(CheckBounds, ReturnsFalseWhenLwExceedsUw) {
  SpMat Q = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  SpMat B = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  auto problem = MakeProblem(1, 0, 1, Q, SpMat(0, 1), B, Vec::Zero(1), Vec(0), 0.0,
                              Vec::Constant(1, -kInf), Vec::Constant(1, kInf), Vec::Constant(1, -1.0),
                              Vec::Constant(1, 1.0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ns.lw_orig(0) = 5.0;
  ns.uw_orig(0) = 1.0;
  EXPECT_FALSE(ns.check_bounds());
}

TEST(CheckBounds, ReturnsTrueWhenBoundsAreConsistent) {
  KSP_QP<double> ns = MakeValidInstance();
  ASSERT_FALSE(ns.setup_failed);
  EXPECT_TRUE(ns.check_bounds());
}

TEST(KspQpConstruction, ReportsPrimalInfeasibleWhenLxExceedsUx) {
  // lx > ux is a certified empty box interval, so the constructor should detect it via
  // check_bounds() and report PrimalInfeasible directly rather than throwing/NumericalError.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  auto problem = MakeProblem(1, 0, 0, Q, SpMat(0, 1), SpMat(0, 1), Vec::Zero(1), Vec(0), 0.0,
                              Vec::Constant(1, 5.0), Vec::Constant(1, 1.0), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  EXPECT_TRUE(ns.setup_failed);
  EXPECT_EQ(ns.opt, TerminationStatus::PrimalInfeasible);

  auto sol = ns.solve();
  EXPECT_EQ(sol.opt, TerminationStatus::PrimalInfeasible);
}

// ===================== ruiz_scaling =====================
// ruiz_scaling() uses only the members n/m/l/Q_info and its `problem` argument, so these tests set
// those members on an existing instance.

TEST(RuizScaling, ReturnsImmediatelyWhenNIsZero) {
  KSP_QP<double> ns = MakeValidInstance();
  ns.n = 0;
  ns.m = 0;
  ns.l = 0;
  ns.Q_info = 0;
  auto problem = MakeProblem(0, 0, 0, SpMat(0, 0), SpMat(0, 0), SpMat(0, 0), Vec(0), Vec(0), 0.0,
                              Vec(0), Vec(0), Vec(0), Vec(0));
  Vec empty_q_diag(0);
  ns.ruiz_scaling(problem, empty_q_diag);  // must not crash on n==0
  EXPECT_EQ(ns.D2_diag.size(), 0);
}

TEST(RuizScaling, MZeroLeavesD1ADiagEmptyAndSkipsRowScalingForA) {
  KSP_QP<double> ns = MakeValidInstance();
  ns.n = 1;
  ns.m = 0;
  ns.l = 0;
  ns.Q_info = 0;
  auto problem = MakeProblem(1, 0, 0, SpMat(0, 0), SpMat(0, 1), SpMat(0, 1), Vec::Zero(1), Vec(0),
                              0.0, Vec(0), Vec(0), Vec(0), Vec(0));
  Vec empty_q_diag(0);
  ns.ruiz_scaling(problem, empty_q_diag);
  EXPECT_EQ(ns.D1A_diag.size(), 0);
  EXPECT_NEAR(ns.D2_diag(0), 1.0, 1e-12);  // no A/B entries at all -> col_max stays 1 -> no scaling
}

TEST(RuizScaling, BalancesAnUnevenSingleRowMatrixInOneIteration) {
  // A=[100,1] is a single row: hand-traced through the algorithm, this converges after exactly
  // one Ruiz pass. Iteration 1: row_max_A=[100], col_max=[100,1] (deviations large, don't
  // converge yet) -> drA=sqrt(100)=10, dc=[sqrt(100),sqrt(1)]=[10,1] -> A <- [100,1] .*
  // [drA_inv*dc_inv(0), drA_inv*dc_inv(1)] = [100*0.1*0.1, 1*0.1*1] = [1, 0.1];
  // D1A_diag=1/10=0.1; D2_diag=[1/10,1]=[0.1,1]. Iteration 2 then sees row_max_A=[1],
  // col_max=[1,1] (both already balanced) and breaks without further scaling.
  KSP_QP<double> ns = MakeValidInstance();
  ns.n = 2;
  ns.m = 1;
  ns.l = 0;
  ns.Q_info = 0;
  SpMat A = DenseToSparse((Eigen::MatrixXd(1, 2) << 100.0, 1.0).finished());
  auto problem = MakeProblem(2, 1, 0, SpMat(0, 0), A, SpMat(0, 2), Vec::Zero(2), Vec::Zero(1), 0.0,
                              Vec(0), Vec(0), Vec(0), Vec(0));
  Vec empty_q_diag(0);
  ns.ruiz_scaling(problem, empty_q_diag);

  EXPECT_NEAR(ns.D1A_diag(0), 0.1, 1e-12);
  EXPECT_NEAR(ns.D2_diag(0), 0.1, 1e-12);
  EXPECT_NEAR(ns.D2_diag(1), 1.0, 1e-12);
  Eigen::MatrixXd expected_A(1, 2);
  expected_A << 1.0, 0.1;
  EXPECT_TRUE(Eigen::MatrixXd(ns.A_ruiz).isApprox(expected_A, 1e-12));
}

// ===================== set_L_from_LLT =====================

TEST(SetLFromLLT, ProducesFactorSatisfyingLLTApproximatesQForPositiveDefiniteInput) {
  KSP_QP<double> ns = MakeValidInstance();
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 4.0, 1.0, 1.0, 3.0).finished());
  ns.set_L_from_LLT(Q);

  Eigen::MatrixXd L_dense(ns.L);
  Eigen::MatrixXd LLT = L_dense * L_dense.transpose();
  Eigen::MatrixXd Q_dense(Q);
  EXPECT_TRUE(LLT.isApprox(Q_dense, 1e-6));
}

TEST(SetLFromLLT, ThrowsOnGenuinelyIndefiniteQInsteadOfSilentlyClamping) {
  // Q=[[1,2],[2,1]] has eigenvalues {3,-1}: too indefinite for the regularization tolerance, so
  // set_L_from_LLT must throw rather than clamp the pivot and return an L with L*L^T != Q.
  KSP_QP<double> ns = MakeValidInstance();
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 1.0, 2.0, 2.0, 1.0).finished());
  EXPECT_THROW(ns.set_L_from_LLT(Q), std::runtime_error);
}

TEST(SetLFromLLT, ConstructorSetsNumericalErrorStatusOnGenuinelyIndefiniteQ) {
  // Same Q through the KSP_QP constructor: set_L_from_LLT's exception must surface as
  // setup_failed + TerminationStatus::NumericalError.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 1.0, 2.0, 2.0, 1.0).finished());
  auto problem = MakeProblem(2, 0, 0, Q, SpMat(0, 2), SpMat(0, 2), Vec::Zero(2), Vec::Zero(0), 0.0,
                              Vec::Constant(2, -kInf), Vec::Constant(2, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  EXPECT_TRUE(ns.setup_failed);
  EXPECT_EQ(ns.opt, TerminationStatus::NumericalError);
  ASSERT_EQ(ns.Q_info, 2);
}

TEST(SetLFromLLT, RecoversViaEscalatedRegularizationWhenNegativePivotIsSmallRelativeToQ) {
  // Q=[[1,b],[b,1]], b=1+1e-7, has eigenvalues {2+1e-7, -1e-7}: a tiny but real negative
  // eigenvalue, within tolerance for Q's scale (~2). The initial delta (~sqrt(eps)*scale) isn't
  // enough, so this exercises the 10x retry, which must succeed with L*L^T ~= Q.
  KSP_QP<double> ns = MakeValidInstance();
  const double b = 1.0000001;
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 1.0, b, b, 1.0).finished());
  ns.set_L_from_LLT(Q);

  Eigen::MatrixXd L_dense(ns.L);
  Eigen::MatrixXd LLT = L_dense * L_dense.transpose();
  Eigen::MatrixXd Q_dense(Q);
  EXPECT_TRUE(LLT.isApprox(Q_dense, 1e-5));
}

TEST(SetLFromLLT, ThrowsOnGenuinelyIndefiniteQRegardlessOfOverallMatrixScale) {
  // The Q of ThrowsOnGenuinelyIndefiniteQInsteadOfSilentlyClamping scaled by 1e6 must still throw:
  // the regularization scales with Q's inf-norm, not an absolute constant.
  KSP_QP<double> ns = MakeValidInstance();
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 1.0, 2.0, 2.0, 1.0).finished() * 1e6);
  EXPECT_THROW(ns.set_L_from_LLT(Q), std::runtime_error);
}

TEST(SetLFromLLT, NullDiagonalRowGetsExactlyZeroRowInL) {
  // Index 2 has a zero diagonal, which means that the entire row/col with index 2
  // are zero for a PSD matrix (Cauchy-Schwarz).
  KSP_QP<double> ns = MakeValidInstance();
  SpMat Q = DenseToSparse((Eigen::MatrixXd(3, 3) <<
                            4.0, 1.0, 0.0,
                            1.0, 3.0, 0.0,
                            0.0, 0.0, 0.0).finished());
  ns.set_L_from_LLT(Q);

  Eigen::MatrixXd L_dense(ns.L);
  EXPECT_TRUE(L_dense.row(2).isZero(0.0));

  // Row 2 is dropped from L's storage, and no other stored entry is an explicit zero.
  for (int k = 0; k < ns.L.outerSize(); ++k)
    for (SpMat::InnerIterator it(ns.L, k); it; ++it) {
      EXPECT_NE(it.row(), 2);
      EXPECT_NE(it.value(), 0.0);
    }

  Eigen::MatrixXd LLT = L_dense * L_dense.transpose();
  Eigen::MatrixXd Q_dense(Q);
  EXPECT_TRUE(LLT.isApprox(Q_dense, 1e-6));
}

TEST(SetLFromLLT, LiftedConstraintMatrixStoresNoExplicitZeros) {
  // Q_info == 2 lifts A to [A, 0; L^T, -I], so L's dropped entries must not reappear in A.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(3, 3) <<
                            4.0, 1.0, 0.0,
                            1.0, 3.0, 0.0,
                            0.0, 0.0, 0.0).finished());
  auto problem = MakeProblem(3, 0, 0, Q, SpMat(0, 3), SpMat(0, 3), Vec::Zero(3), Vec::Zero(0), 0.0,
                              Vec::Constant(3, -kInf), Vec::Constant(3, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_EQ(ns.Q_info, 2);

  for (int k = 0; k < ns.A.outerSize(); ++k)
    for (SpMat::InnerIterator it(ns.A, k); it; ++it)
      EXPECT_NE(it.value(), 0.0) << "explicit zero at (" << it.row() << ", " << it.col() << ")";
  EXPECT_EQ(ns.A.col(2).nonZeros(), 0);  // x_2 is Q's null row, so L^T has no entries in its column
}

// ===================== build_reformulated_vecs (static) =====================

TEST(BuildReformulatedVecs, NoReformulationBranchDefaultsEmptyVectorsAndBoundsIndependently) {
  Vec c_out, b_out, lx_out, ux_out;
  KSP_QP<double>::build_reformulated_vecs(2, 1, /*N=*/2, /*M=*/1, kInf, /*c_in=*/Vec(0),
                                            /*b_in=*/Vec(0), /*lx_in=*/Vec(0), /*ux_in=*/Vec(0),
                                            c_out, b_out, lx_out, ux_out);
  EXPECT_TRUE(c_out.isApprox(Vec::Zero(2)));
  EXPECT_TRUE(b_out.isApprox(Vec::Zero(1)));
  ASSERT_EQ(lx_out.size(), 2);
  ASSERT_EQ(ux_out.size(), 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(lx_out(i), -kInf);
    EXPECT_EQ(ux_out(i), kInf);
  }
}

TEST(BuildReformulatedVecs, NoReformulationBranchPassesThroughNonemptyCAndBoundsUnchanged) {
  Vec c_in(2); c_in << 1.0, 2.0;
  Vec b_in(1); b_in << 3.0;
  Vec lx_in(2); lx_in << -5.0, -6.0;
  Vec ux_in(2); ux_in << 5.0, 6.0;
  Vec c_out, b_out, lx_out, ux_out;
  KSP_QP<double>::build_reformulated_vecs(2, 1, 2, 1, kInf, c_in, b_in, lx_in, ux_in, c_out, b_out,
                                            lx_out, ux_out);
  EXPECT_TRUE(c_out.isApprox(c_in));
  EXPECT_TRUE(b_out.isApprox(b_in));
  EXPECT_TRUE(lx_out.isApprox(lx_in));
  EXPECT_TRUE(ux_out.isApprox(ux_in));
}

TEST(BuildReformulatedVecs, ReformulationBranchPadsWithZerosOrInfiniteBoundsForTheAuxiliaryBlock) {
  Vec c_in(2); c_in << 1.0, 2.0;
  Vec b_in(1); b_in << 3.0;
  Vec lx_in(2); lx_in << -5.0, -6.0;
  Vec ux_in(2); ux_in << 5.0, 6.0;
  const int n = 2, m = 1, N = 4, M = 3;  // N=2n, M=m+n
  Vec c_out, b_out, lx_out, ux_out;
  KSP_QP<double>::build_reformulated_vecs(n, m, N, M, kInf, c_in, b_in, lx_in, ux_in, c_out, b_out,
                                            lx_out, ux_out);

  ASSERT_EQ(c_out.size(), N);
  Vec expected_c(4);
  expected_c << 1.0, 2.0, 0.0, 0.0;
  EXPECT_TRUE(c_out.isApprox(expected_c));

  ASSERT_EQ(b_out.size(), M);
  Vec expected_b(3);
  expected_b << 3.0, 0.0, 0.0;
  EXPECT_TRUE(b_out.isApprox(expected_b));

  ASSERT_EQ(lx_out.size(), N);
  EXPECT_TRUE(lx_out.head(2).isApprox(lx_in));
  EXPECT_EQ(lx_out(2), -kInf);
  EXPECT_EQ(lx_out(3), -kInf);

  ASSERT_EQ(ux_out.size(), N);
  EXPECT_TRUE(ux_out.head(2).isApprox(ux_in));
  EXPECT_EQ(ux_out(2), kInf);
  EXPECT_EQ(ux_out(3), kInf);
}

TEST(BuildReformulatedVecs, ReformulationBranchDefaultsEmptyCAndBToZeroPadding) {
  const int n = 1, m = 0, N = 2, M = 1;
  Vec c_out, b_out, lx_out, ux_out;
  KSP_QP<double>::build_reformulated_vecs(n, m, N, M, kInf, Vec(0), Vec(0), Vec(0), Vec(0), c_out,
                                            b_out, lx_out, ux_out);
  EXPECT_TRUE(c_out.isApprox(Vec::Zero(N)));
  EXPECT_TRUE(b_out.isApprox(Vec::Zero(M)));  // DOES default here, unlike the N==n branch above
  ASSERT_EQ(lx_out.size(), N);
  ASSERT_EQ(ux_out.size(), N);
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(lx_out(i), -kInf);
    EXPECT_EQ(ux_out(i), kInf);
  }
}

// ===================== compute_residual_unscaled_inf_norms =====================
//
// Each case constructs a KSP_QP (bypassing solve()) and sets x/y1/y2/z and Ax/Bx/Qx by hand.
// A/B entries have magnitude 1 (or M/l = 0), so Ruiz scaling is a no-op (all D factors exactly 1)
// and the expected residuals can be derived by hand.

TEST(ComputeResidualUnscaledInfNorms, AllShortcutsFireWhenMAndLAreZero) {
  SpMat Q(1, 1);  // Q_info = 0
  auto problem = MakeProblem(1, 0, 0, Q, SpMat(0, 1), SpMat(0, 1), Vec::Zero(1), Vec::Zero(0), 0.0,
                              Vec::Constant(1, -10.0), Vec::Constant(1, 10.0), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_DOUBLE_EQ(ns.D2_diag(0), 1.0);  // confirms ruiz scaling was a no-op as designed

  ns.x = Vec(1); ns.x << 0.5;
  ns.y1 = Vec::Zero(0);
  ns.y2 = Vec::Zero(0);
  ns.z = Vec(1); ns.z << 0.2;

  auto res = ns.compute_residual_unscaled_inf_norms(Vec::Zero(0), Vec::Zero(0), Vec::Zero(1));

  EXPECT_DOUBLE_EQ(res(0), 0.0);          // res_p: M=0 shortcut
  EXPECT_NEAR(res(1), 0.2 / 1.2, 1e-12);  // res_d: num=c+z=0.2, denom=max(0,0.2)+1=1.2
  EXPECT_NEAR(res(2), 0.2 / 1.7, 1e-12);  // compl_x: |0.5-0.7|/(1+max(0.2,0.7))
  EXPECT_DOUBLE_EQ(res(3), 0.0);          // compl_w: l=0 shortcut
}

TEST(ComputeResidualUnscaledInfNorms, PrimalResidualNonzeroWhenMPositive) {
  SpMat Q(1, 1);
  SpMat A = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  Vec b(1); b << 0.3;
  auto problem = MakeProblem(1, 1, 0, Q, A, SpMat(0, 1), Vec::Zero(1), b, 0.0,
                              Vec::Constant(1, -10.0), Vec::Constant(1, 10.0), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_DOUBLE_EQ(ns.D1A_diag(0), 1.0);

  ns.x = Vec(1); ns.x << 0.5;
  ns.y1 = Vec::Zero(1);
  ns.y2 = Vec::Zero(0);
  ns.z = Vec::Zero(1);

  Vec Ax(1); Ax << 0.8;
  auto res = ns.compute_residual_unscaled_inf_norms(Ax, Vec::Zero(0), Vec::Zero(1));

  EXPECT_NEAR(res(0), 0.5 / 1.8, 1e-12);  // |0.8-0.3|/(1+max(0.8,0.3))
  EXPECT_DOUBLE_EQ(res(1), 0.0);
  EXPECT_DOUBLE_EQ(res(2), 0.0);
  EXPECT_DOUBLE_EQ(res(3), 0.0);
}

TEST(ComputeResidualUnscaledInfNorms, ComplementarityWNonzeroWhenLPositive) {
  SpMat Q(1, 1);
  SpMat B = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  auto problem = MakeProblem(1, 0, 1, Q, SpMat(0, 1), B, Vec::Zero(1), Vec::Zero(0), 0.0,
                              Vec::Constant(1, -10.0), Vec::Constant(1, 10.0),
                              Vec::Constant(1, -10.0), Vec::Constant(1, 10.0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_DOUBLE_EQ(ns.D1B_diag(0), 1.0);

  ns.x = Vec(1); ns.x << 0.5;
  ns.y1 = Vec::Zero(0);
  ns.y2 = Vec(1); ns.y2 << 0.1;
  ns.z = Vec::Zero(1);

  Vec Bx(1); Bx << 0.9;
  auto res = ns.compute_residual_unscaled_inf_norms(Vec::Zero(0), Bx, Vec::Zero(1));

  EXPECT_DOUBLE_EQ(res(0), 0.0);
  EXPECT_NEAR(res(1), 0.1 / 1.1, 1e-12);  // num=c+z-B^T y2=-0.1, denom=max(0,0,0.1)+1
  EXPECT_DOUBLE_EQ(res(2), 0.0);
  EXPECT_NEAR(res(3), 0.1 / 1.8, 1e-12);  // |0.9-0.8|/(1+max(0.1,0.8))
}

TEST(ComputeResidualUnscaledInfNorms, DualResidualIncludesQxTermWhenQInfoNonzero) {
  SpMat Q = DenseToSparse((Eigen::MatrixXd(1, 1) << 5.0).finished());  // Q_info = 1
  auto problem = MakeProblem(1, 0, 0, Q, SpMat(0, 1), SpMat(0, 1), Vec::Zero(1), Vec::Zero(0), 0.0,
                              Vec::Constant(1, -10.0), Vec::Constant(1, 10.0), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_EQ(ns.Q_info, 1);

  ns.x = Vec(1); ns.x << 0.5;
  ns.y1 = Vec::Zero(0);
  ns.y2 = Vec::Zero(0);
  ns.z = Vec(1); ns.z << 0.1;

  Vec Qx(1); Qx << 0.4;
  auto res = ns.compute_residual_unscaled_inf_norms(Vec::Zero(0), Vec::Zero(0), Qx);

  EXPECT_DOUBLE_EQ(res(0), 0.0);
  EXPECT_NEAR(res(1), 0.5 / 1.4, 1e-12);  // num=c+z+Qx=0.5, denom=max(0,0.1,0.4)+1
  EXPECT_NEAR(res(2), 0.1 / 1.6, 1e-12);  // |0.5-0.6|/(1+max(0.1,0.6))
  EXPECT_DOUBLE_EQ(res(3), 0.0);
}


TEST(ComputeResidualUnscaledInfNorms, DualResidualUnaffectedByNullDirectionMagnitudeWhenQInfoIsTwo) {
  // Index 2 has zero row/column in Q, i.e. L's row 2 is exactly zero.
  // With A = I and c, b = 0, the dual residual c+z-A_ruiz^T*y1_top-L*y_v=0 must vanish
  // regardless of x_head(2)'s magnitude.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(3, 3) <<
                            4.0, 1.0, 0.0,
                            1.0, 3.0, 0.0,
                            0.0, 0.0, 0.0).finished());
  SpMat A = DenseToSparse(Eigen::MatrixXd::Identity(3, 3));
  Vec c = Vec::Zero(3);
  Vec b = Vec::Zero(3);
  auto problem = MakeProblem(3, 3, 0, Q, A, SpMat(0, 3), c, b, 0.0,
                              Vec::Constant(3, -kInf), Vec::Constant(3, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_EQ(ns.Q_info, 2);
  ASSERT_TRUE(ns.D1A_diag.isApprox(Vec::Ones(3)));  // A=I needs no rescaling

  ns.y2 = Vec::Zero(0);
  ns.z = Vec::Zero(6);

  auto res_d_for_x2 = [&](double x2) {
    Vec x_head(3); x_head << 1.0, 1.0, x2;
    Vec v = ns.L.transpose() * x_head;  // exact primal feasibility of the lifting row
    ns.x = Vec(6); ns.x << x_head, v;
    Vec y1_top = ns.L * v;              // what the lifted x_head-stationarity forces y1_top to
    ns.y1 = Vec(6); ns.y1 << y1_top, -v;  // y_v = -v satisfies the v-block's own stationarity
    Vec Ax = ns.A * ns.x;
    Vec Qx = ns.Q_diag.cwiseProduct(ns.x);
    return ns.compute_residual_unscaled_inf_norms(Ax, Vec::Zero(0), Qx)(1);
  };

  EXPECT_NEAR(res_d_for_x2(1.0), res_d_for_x2(1e6), 1e-9);
}

// ===================== objective_value =====================

TEST(ObjectiveValue, LinearOnlyWhenQInfoIsZero) {
  SpMat Q(2, 2);  // Q_info = 0
  Vec c(2); c << 3.0, -1.0;
  auto problem = MakeProblem(2, 0, 0, Q, SpMat(0, 2), SpMat(0, 2), c, Vec(0), 7.0,
                              Vec::Constant(2, -kInf), Vec::Constant(2, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_EQ(ns.Q_info, 0);

  Vec x(2); x << 2.0, 5.0;
  // obj_const + c_orig.dot(x) = 7 + (3*2 + -1*5) = 7 + 1 = 8
  EXPECT_DOUBLE_EQ(ns.objective_value(x), 8.0);
}

TEST(ObjectiveValue, ReconstructsFullSymmetricQuadraticFromLowerTriangularStorage) {
  // A general symmetric Q (Q_info=2) is stored lower-triangular; objective_value's
  // selfadjointView<Lower>() must still give the full x^T Q x.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 4.0, 1.0, 1.0, 2.0).finished());
  Vec c(2); c << 0.0, 0.0;
  auto problem = MakeProblem(2, 0, 0, Q, SpMat(0, 2), SpMat(0, 2), c, Vec(0), 0.0,
                              Vec::Constant(2, -kInf), Vec::Constant(2, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_EQ(ns.Q_info, 2);

  Vec x(2); x << 1.0, 1.0;
  // 0.5 * x^T Q x = 0.5*(4+1+1+2) = 4.0 using the full symmetric Q -- would be 0.5*(4+1+2)=3.5 if
  // only the lower triangle (as physically stored) were used without mirroring.
  EXPECT_DOUBLE_EQ(ns.objective_value(x), 4.0);
}

// ===================== printable_sol =====================

TEST(PrintableSol, DescalesDirectlyWhenQInfoIsNotTwo) {
  SpMat Q(2, 2);  // Q_info = 0 -> N=n, no reformulation
  SpMat A = DenseToSparse((Eigen::MatrixXd(1, 2) << 1.0, 0.0).finished());
  SpMat B = DenseToSparse((Eigen::MatrixXd(1, 2) << 0.0, 1.0).finished());
  auto problem = MakeProblem(2, 1, 1, Q, A, B, Vec::Zero(2), Vec::Zero(1), 0.0,
                              Vec::Constant(2, -kInf), Vec::Constant(2, kInf),
                              Vec::Constant(1, -kInf), Vec::Constant(1, kInf));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_EQ(ns.Q_info, 0);
  ASSERT_EQ(ns.N, 2);

  ns.D2_diag = Vec(2); ns.D2_diag << 2.0, 3.0;
  ns.D1A_diag = Vec(1); ns.D1A_diag << 4.0;
  ns.D1B_diag = Vec(1); ns.D1B_diag << 5.0;

  Vec x(2); x << 1.0, 1.0;
  Vec y1(1); y1 << 1.0;
  Vec y2(1); y2 << 1.0;
  Vec z(2); z << 6.0, 9.0;

  ns.printable_sol(x, y1, y2, z);

  Vec expected_x(2); expected_x << 2.0, 3.0;  // x .* D2_diag
  Vec expected_y1(1); expected_y1 << 4.0;     // y1 .* D1A_diag
  Vec expected_y2(1); expected_y2 << 5.0;     // y2 .* D1B_diag
  Vec expected_z(2); expected_z << 3.0, 3.0;  // z ./ D2_diag
  EXPECT_TRUE(ns.x_sol.isApprox(expected_x));
  EXPECT_TRUE(ns.y1_sol.isApprox(expected_y1));
  EXPECT_TRUE(ns.y2_sol.isApprox(expected_y2));
  EXPECT_TRUE(ns.z_sol.isApprox(expected_z));
}

TEST(PrintableSol, SlicesToHeadNBeforeDescalingWhenQInfoIsTwo) {
  // Q_info=2 -> N=2n, M=m+n: x/z (and y1) live at the reformulated size and must be sliced back
  // to head(n)/head(m) BEFORE descaling, not descaled at the full reformulated size.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 2.0, 1.0, 1.0, 2.0).finished());
  auto problem = MakeProblem(2, 0, 0, Q, SpMat(0, 2), SpMat(0, 2), Vec::Zero(2), Vec(0), 0.0,
                              Vec::Constant(2, -kInf), Vec::Constant(2, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_EQ(ns.Q_info, 2);
  ASSERT_EQ(ns.n, 2);
  ASSERT_EQ(ns.N, 4);
  ASSERT_EQ(ns.M, 2);

  ns.D2_diag = Vec(2); ns.D2_diag << 2.0, 3.0;
  ns.D1A_diag = Vec(0);
  ns.D1B_diag = Vec(0);

  Vec x(4); x << 1.0, 1.0, 999.0, 999.0;  // last n entries are the auxiliary block -- must be dropped
  Vec y1(0), y2(0);
  Vec z(4); z << 6.0, 9.0, -999.0, -999.0;

  ns.printable_sol(x, y1, y2, z);

  ASSERT_EQ(ns.x_sol.size(), 2);
  Vec expected_x(2); expected_x << 2.0, 3.0;
  EXPECT_TRUE(ns.x_sol.isApprox(expected_x));
  ASSERT_EQ(ns.z_sol.size(), 2);
  Vec expected_z(2); expected_z << 3.0, 3.0;
  EXPECT_TRUE(ns.z_sol.isApprox(expected_z));
}

// ===================== update_PMM_parameters =====================

TEST(UpdatePmmParameters, OptimalScalesMuRhoUpAndShrinksSsnTolByTenPercent) {
  KSP_QP<double> ns = MakeValidInstance();
  ns.mu = 10.0; ns.rho = 20.0; ns.ssn_tol = 0.01;
  KSP_QP<double>::ResVec res_norms, new_res_norms;
  res_norms << 1.0, 1.0, 1.0, 1.0;
  new_res_norms << 1.0, 1.0, 1.0, 1.0;

  ns.update_PMM_parameters(res_norms, new_res_norms, SSN<double>::TerminationStatus::Optimal,
                            /*ssn_res=*/0.0, /*ssn_inner_iters=*/0);

  EXPECT_DOUBLE_EQ(ns.mu, 20.0);
  EXPECT_DOUBLE_EQ(ns.rho, 40.0);
  EXPECT_DOUBLE_EQ(ns.ssn_tol, 0.001);
}

TEST(UpdatePmmParameters, MuAndRhoStayClampedAtTheirLimitsOnOptimal) {
  KSP_QP<double> ns = MakeValidInstance();
  ns.mu = ns.mu_limit;  // 1e9
  ns.rho = ns.rho_limit; // 1e7
  ns.ssn_tol = 0.01;
  KSP_QP<double>::ResVec res_norms, new_res_norms;
  res_norms << 1.0, 1.0, 1.0, 1.0;
  new_res_norms << 1.0, 1.0, 1.0, 1.0;

  ns.update_PMM_parameters(res_norms, new_res_norms, SSN<double>::TerminationStatus::Optimal, 0.0, 0);

  EXPECT_DOUBLE_EQ(ns.mu, ns.mu_limit);
  EXPECT_DOUBLE_EQ(ns.rho, ns.rho_limit);
}

TEST(UpdatePmmParameters, SsnTolFloorsAtEpsLimitOnOptimal) {
  KSP_QP<double> ns = MakeValidInstance();
  ns.mu = 1.0; ns.rho = 1.0;
  ns.ssn_tol = 1e-12;  // 0.1x would be 1e-13, below eps_limit = 1e-3*tol = 1e-11
  KSP_QP<double>::ResVec res_norms, new_res_norms;
  res_norms << 1.0, 1.0, 1.0, 1.0;
  new_res_norms << 1.0, 1.0, 1.0, 1.0;

  ns.update_PMM_parameters(res_norms, new_res_norms, SSN<double>::TerminationStatus::Optimal, 0.0, 0);

  EXPECT_DOUBLE_EQ(ns.ssn_tol, ns.eps_limit);
}

TEST(UpdatePmmParameters, LineSearchFailedLoosensMuRhoAndGrowsSsnTol) {
  KSP_QP<double> ns = MakeValidInstance();
  ns.mu = 10.0; ns.rho = 20.0; ns.ssn_tol = 0.001;
  KSP_QP<double>::ResVec res_norms, new_res_norms;
  res_norms << 1.0, 1.0, 1.0, 1.0;
  new_res_norms << 5.0, 0.1, 0.1, 0.1;  // worst_res = 5.0

  ns.update_PMM_parameters(res_norms, new_res_norms, SSN<double>::TerminationStatus::LineSearchFailed,
                            /*ssn_res=*/0.0, 0);

  EXPECT_DOUBLE_EQ(ns.mu, 5.0);          // max(mu0=1, 0.5*10)
  EXPECT_DOUBLE_EQ(ns.rho, 10.0);        // max(rho0=1, 0.5*20)
  EXPECT_DOUBLE_EQ(ns.ssn_tol, 0.0011);  // min(worst_res=5.0, 1.1*0.001=0.0011, 1e-2)
}

// ===================== primal_infeas / dual_infeas =====================
// Certificates are built from the solver's own Ruiz-scaled data after construction, so they
// satisfy the formulas in ksp_qp.tpp exactly, whatever the scaling factors are.

TEST(PrimalInfeas, DetectsCertificateOnHandBuiltInfeasibleLp) {
  // min 0  s.t.  x = 1000,  x <= 0.  Genuinely infeasible.  The finite ux is what makes the
  // certificate provable: condition 2's lhs is the support function of cert_z over [lx, ux], so
  // the coordinate cert_z leans on must be bounded on the side cert_z points to (see condition 3).
  SpMat A = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  Vec b(1);
  b << 1000.0;  // large positive -- see condition-2 derivation in the test file header
  auto problem = MakeProblem(1, 1, 0, SpMat(1, 1), A, SpMat(0, 1), Vec::Zero(1), b, 0.0,
                              Vec::Constant(1, -kInf), Vec::Zero(1), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_GT(ns.b(0), 0.0);  // ruiz scaling preserves sign

  Vec cert_y1(1);
  cert_y1 << 1.0;
  Vec cert_y2(0);
  Vec cert_z = ns.A_tr * cert_y1;  // makes condition 1's lhs exactly zero

  EXPECT_TRUE(ns.primal_infeas(cert_y1, cert_y2, cert_z));
}

TEST(PrimalInfeas, RejectsCertificateLeaningOnAnInfiniteBound) {
  // Same certificate as DetectsCertificateOnHandBuiltInfeasibleLp, but ux = +inf, so the LP
  // (x = 1000, x free) is feasible. cert_z > 0 on a coordinate with no upper bound makes condition
  // 2's support function +inf, so condition 3 must abandon the certificate; treating that term as 0
  // would give lhs2 = -b*cert_y1 = -1000 < 0 and a false infeasible verdict.
  SpMat A = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  Vec b(1);
  b << 1000.0;
  auto problem = MakeProblem(1, 1, 0, SpMat(1, 1), A, SpMat(0, 1), Vec::Zero(1), b, 0.0,
                              Vec::Constant(1, -kInf), Vec::Constant(1, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);

  Vec cert_y1(1);
  cert_y1 << 1.0;
  Vec cert_y2(0);
  Vec cert_z = ns.A_tr * cert_y1;  // condition 1 holds exactly, condition 2's lhs would be -1000

  EXPECT_FALSE(ns.primal_infeas(cert_y1, cert_y2, cert_z));
}

TEST(PrimalInfeas, RejectsCertificateLeaningOnAnInfiniteBoxBound) {
  // Condition 3 on the Bx side: w is free, so a nonzero cert_y2 makes the support function over
  // [lw, uw] infinite and the certificate unusable, whichever sign cert_y2 has.  Both signs are
  // exercised because the two infinite bounds are guarded by opposite branches of condition 3.
  SpMat A = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  SpMat B = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  Vec b(1);
  b << 1000.0;
  auto problem = MakeProblem(1, 1, 1, SpMat(1, 1), A, B, Vec::Zero(1), b, 0.0,
                              Vec::Constant(1, -kInf), Vec::Zero(1),
                              Vec::Constant(1, -kInf), Vec::Constant(1, kInf));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);

  Vec cert_y1(1);
  cert_y1 << 1.0;
  for (double s : {1.0, -1.0}) {
    Vec cert_y2 = Vec::Constant(1, s);
    Vec cert_z = ns.A_tr * cert_y1 + ns.B_tr * cert_y2;  // condition 1's lhs is exactly zero
    EXPECT_FALSE(ns.primal_infeas(cert_y1, cert_y2, cert_z)) << "cert_y2 sign " << s;
  }
}

TEST(PrimalInfeas, ReturnsFalseForZeroCertificate) {
  KSP_QP<double> ns = MakeValidInstance();
  Vec cert_y1(0), cert_y2(0), cert_z(1);
  cert_z << 0.0;
  EXPECT_FALSE(ns.primal_infeas(cert_y1, cert_y2, cert_z));
}

TEST(PrimalInfeas, ReturnsFalseWhenCondition2PassesButCondition1Fails) {
  // Same infeasible LP as DetectsCertificateOnHandBuiltInfeasibleLp, but cert_z=0: condition 2
  // still passes, but condition 1's lhs1 = A^T*cert_y1 is nonzero, so condition 1 alone must
  // reject the certificate.
  SpMat A = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  Vec b(1);
  b << 1000.0;
  auto problem = MakeProblem(1, 1, 0, SpMat(1, 1), A, SpMat(0, 1), Vec::Zero(1), b, 0.0,
                              Vec::Constant(1, -kInf), Vec::Zero(1), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_GT(ns.b(0), 0.0);

  Vec cert_y1(1);
  cert_y1 << 1.0;
  Vec cert_y2(0);
  Vec cert_z = Vec::Zero(1);

  EXPECT_FALSE(ns.primal_infeas(cert_y1, cert_y2, cert_z));
}

TEST(DualInfeas, DetectsCertificateOnHandBuiltUnboundedLp) {
  Vec c(1);
  c << -1000.0;  // large negative -- unconstrained x is unbounded in this direction
  auto problem = MakeProblem(1, 0, 0, SpMat(1, 1), SpMat(0, 1), SpMat(0, 1), c, Vec(0), 0.0,
                              Vec::Constant(1, -kInf), Vec::Constant(1, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_LT(ns.c(0), 0.0);  // ruiz scaling preserves sign

  Vec delta_x(1);
  delta_x << 1.0;
  Vec Adx(0), Bdx(0);
  EXPECT_TRUE(ns.dual_infeas(delta_x, Adx, Bdx));
}

TEST(DualInfeas, ReturnsFalseWhenObjectiveConditionFails) {
  Vec c(1);
  c << 1000.0;  // positive cost in the delta_x direction -> c.dot(delta_x) > 0, fails condition 2
  auto problem = MakeProblem(1, 0, 0, SpMat(1, 1), SpMat(0, 1), SpMat(0, 1), c, Vec(0), 0.0,
                              Vec::Constant(1, -kInf), Vec::Constant(1, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);

  Vec delta_x(1);
  delta_x << 1.0;
  Vec Adx(0), Bdx(0);
  EXPECT_FALSE(ns.dual_infeas(delta_x, Adx, Bdx));
}

TEST(DualInfeas, ReturnsFalseWhenQDeltaXConditionFails) {
  SpMat Q = DenseToSparse((Eigen::MatrixXd(1, 1) << 1000.0).finished());
  Vec c(1);
  c << -1000.0;  // keeps condition 2 passing so condition 1 is isolated
  auto problem = MakeProblem(1, 0, 0, Q, SpMat(0, 1), SpMat(0, 1), c, Vec(0), 0.0,
                              Vec::Constant(1, -kInf), Vec::Constant(1, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_EQ(ns.Q_info, 1);

  Vec delta_x(1);
  delta_x << 1.0;
  Vec Adx(0), Bdx(0);
  EXPECT_FALSE(ns.dual_infeas(delta_x, Adx, Bdx));
}

TEST(DualInfeas, ReturnsFalseWhenADeltaXConditionFails) {
  SpMat A = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  Vec c(1);
  c << -1000.0;
  auto problem = MakeProblem(1, 1, 0, SpMat(1, 1), A, SpMat(0, 1), c, Vec::Zero(1), 0.0,
                              Vec::Constant(1, -kInf), Vec::Constant(1, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);

  Vec delta_x(1);
  delta_x << 1.0;
  Vec Adx(1);
  Adx << 1000.0;  // huge, decoupled from delta_x -- forces condition 3 to fail on its own
  Vec Bdx(0);
  EXPECT_FALSE(ns.dual_infeas(delta_x, Adx, Bdx));
}

TEST(DualInfeas, ReturnsFalseWhenFiniteXBoundConditionFails) {
  // With N=1 and both lx,ux finite on the only dimension, |delta_x_i|_unscaled equals
  // delta_x_inf exactly, which is always > eps_dinf*delta_x_inf whenever delta_x != 0 -- so
  // condition 4 fails here regardless of magnitude, isolating it from conditions 1-3 (Q=0, M=0).
  Vec c(1);
  c << -1000.0;
  auto problem = MakeProblem(1, 0, 0, SpMat(1, 1), SpMat(0, 1), SpMat(0, 1), c, Vec(0), 0.0,
                              Vec::Constant(1, -1.0), Vec::Constant(1, 1.0), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);

  Vec delta_x(1);
  delta_x << 1.0;
  Vec Adx(0), Bdx(0);
  EXPECT_FALSE(ns.dual_infeas(delta_x, Adx, Bdx));
}

TEST(DualInfeas, ReturnsFalseWhenFiniteBxBoundConditionFails) {
  SpMat B = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  Vec c(1);
  c << -1000.0;
  auto problem = MakeProblem(1, 0, 1, SpMat(1, 1), SpMat(0, 1), B, c, Vec(0), 0.0,
                              Vec::Constant(1, -kInf), Vec::Constant(1, kInf),
                              Vec::Constant(1, -1.0), Vec::Constant(1, 1.0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);

  Vec delta_x(1);
  delta_x << 1.0;
  Vec Adx(0), Bdx(1);
  Bdx << 5.0;  // decoupled from delta_x -- forces condition 5 to fail on its own
  EXPECT_FALSE(ns.dual_infeas(delta_x, Adx, Bdx));
}

TEST(DualInfeas, DetectsCertificateInNullspaceOfNonDiagonalQ) {
  // min -1000 x1 + 0.5 (x1 + x2)^2  s.t.  x1 + x2 = 0.  Unbounded along (1, -1), which lies in
  // null(Q) and null(A).  Non-diagonal Q lifts x to [x; v] with v = L^T x and appends rows
  // [L^T, -I] to A, but a PMM iterate only satisfies v = L^T x approximately.  delta_v is set
  // inconsistent with L^T delta_x on purpose: the certificate is on the original variables, so
  // neither delta_v nor the lifted rows of Adx may affect the verdict.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 1.0, 1.0, 1.0, 1.0).finished());
  SpMat A = DenseToSparse((Eigen::MatrixXd(1, 2) << 1.0, 1.0).finished());
  Vec c(2);
  c << -1000.0, 0.0;
  auto problem = MakeProblem(2, 1, 0, Q, A, SpMat(0, 2), c, Vec::Zero(1), 0.0,
                              Vec::Constant(2, -kInf), Vec::Constant(2, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_EQ(ns.Q_info, 2);

  Vec dx(2);
  dx << 1.0, -1.0;  // unscaled: Q dx = 0 and A dx = 0 exactly
  Vec delta_x(ns.N);
  delta_x.head(2) = dx.cwiseQuotient(ns.D2_diag);
  delta_x.tail(2) << 5.0, 5.0;  // delta_v != L^T delta_x_hat
  Vec Adx = ns.A * delta_x;     // original row is 0, lifted rows L^T delta_x_hat - delta_v are not
  Vec Bdx(0);
  EXPECT_TRUE(ns.dual_infeas(delta_x, Adx, Bdx));
}

TEST(DualInfeas, ReturnsFalseWhenNonDiagonalQDeltaXConditionFails) {
  // Same Q and c, no constraints: c^T dx = -1000 keeps condition 2 passing and conditions 3-5 are
  // vacuous, but Q dx = (2, 2), so only the final original-Q check (condition 1) can reject it.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 1.0, 1.0, 1.0, 1.0).finished());
  Vec c(2);
  c << -1000.0, 0.0;
  auto problem = MakeProblem(2, 0, 0, Q, SpMat(0, 2), SpMat(0, 2), c, Vec(0), 0.0,
                              Vec::Constant(2, -kInf), Vec::Constant(2, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_EQ(ns.Q_info, 2);

  Vec dx(2);
  dx << 1.0, 1.0;  // unscaled
  Vec delta_x(ns.N);
  delta_x.head(2) = dx.cwiseQuotient(ns.D2_diag);
  delta_x.tail(2) = ns.L.transpose() * delta_x.head(2);  // consistent lifted iterate
  Vec Adx = ns.A * delta_x;
  Vec Bdx(0);
  EXPECT_FALSE(ns.dual_infeas(delta_x, Adx, Bdx));
}

// ===================== end-to-end regressions =====================
// Hand-verified tiny QPs/LPs, solved via the public solve() entry point end-to-end.

TEST(KspQpSolveEndToEnd, UnconstrainedQuadraticMatchesClosedFormMinimizer) {
  // min 0.5*(2 x1^2 + 4 x2^2) - 4 x1 - 8 x2  =>  x* = [2, 2], obj* = -12.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 2.0, 0.0, 0.0, 4.0).finished());
  Vec c(2);
  c << -4.0, -8.0;
  auto problem = MakeProblem(2, 0, 0, Q, SpMat(0, 2), SpMat(0, 2), c, Vec(0), 0.0,
                              Vec::Constant(2, -kInf), Vec::Constant(2, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  auto sol = ns.solve();

  EXPECT_EQ(sol.opt, TerminationStatus::Optimal);
  Vec expected_x(2);
  expected_x << 2.0, 2.0;
  EXPECT_TRUE(sol.x.isApprox(expected_x, kTol));
  EXPECT_NEAR(sol.obj_val, -12.0, kTol * 10);
}

TEST(KspQpSolveEndToEnd, BoxConstrainedQpActivatesUpperBound) {
  // Same objective as above, but boxed to [-1,1]^2: separable + no coupling -> x* = [1, 1].
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 2.0, 0.0, 0.0, 4.0).finished());
  Vec c(2);
  c << -4.0, -8.0;
  auto problem = MakeProblem(2, 0, 0, Q, SpMat(0, 2), SpMat(0, 2), c, Vec(0), 0.0,
                              Vec::Constant(2, -1.0), Vec::Constant(2, 1.0), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  auto sol = ns.solve();

  EXPECT_EQ(sol.opt, TerminationStatus::Optimal);
  Vec expected_x(2);
  expected_x << 1.0, 1.0;
  EXPECT_TRUE(sol.x.isApprox(expected_x, kTol));
  EXPECT_NEAR(sol.obj_val, -9.0, kTol * 10);
}

TEST(KspQpSolveEndToEnd, EqualityConstrainedLpMatchesHandSolvedVertex) {
  // min x1 - x2, s.t. x1+x2=1, 0<=x1,x2<=1  =>  x* = [0, 1], obj* = -1.
  SpMat A = DenseToSparse((Eigen::MatrixXd(1, 2) << 1.0, 1.0).finished());
  Vec c(2);
  c << 1.0, -1.0;
  Vec b(1);
  b << 1.0;
  auto problem = MakeProblem(2, 1, 0, SpMat(2, 2), A, SpMat(0, 2), c, b, 0.0, Vec::Constant(2, 0.0),
                              Vec::Constant(2, 1.0), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  auto sol = ns.solve();

  EXPECT_EQ(sol.opt, TerminationStatus::Optimal);
  Vec expected_x(2);
  expected_x << 0.0, 1.0;
  EXPECT_TRUE(sol.x.isApprox(expected_x, kTol));
  EXPECT_NEAR(sol.obj_val, -1.0, kTol * 10);
}

TEST(KspQpSolveEndToEnd, InequalityConstrainedQpMatchesKktBySubstitution) {
  // min x^2 - 4x s.t. -100 <= x <= 0.5 (via a genuine B/W inequality block, not lx/ux directly).
  // Unconstrained minimizer x=2 violates Bx<=0.5, so the bound is active: x* = 0.5, obj* = -1.75.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(1, 1) << 2.0).finished());
  SpMat B = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  Vec c(1);
  c << -4.0;
  Vec lw(1), uw(1);
  lw << -100.0;
  uw << 0.5;
  auto problem = MakeProblem(1, 0, 1, Q, SpMat(0, 1), B, c, Vec(0), 0.0, Vec::Constant(1, -kInf),
                              Vec::Constant(1, kInf), lw, uw);
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  auto sol = ns.solve();

  EXPECT_EQ(sol.opt, TerminationStatus::Optimal);
  EXPECT_NEAR(sol.x(0), 0.5, kTol);
  EXPECT_NEAR(sol.obj_val, -1.75, kTol * 10);
}

TEST(KspQpSolveEndToEnd, DetectsPrimalInfeasibleProblem) {
  // x1 = 1 and x1 = 2 simultaneously: unconditionally infeasible regardless of the objective.
  SpMat A = DenseToSparse((Eigen::MatrixXd(2, 1) << 1.0, 1.0).finished());
  Vec b(2);
  b << 1.0, 2.0;
  auto problem = MakeProblem(1, 2, 0, SpMat(1, 1), A, SpMat(0, 1), Vec::Zero(1), b, 0.0,
                              Vec::Constant(1, -kInf), Vec::Constant(1, kInf), Vec(0), Vec(0),
                              /*tol=*/1e-8, /*max_iter=*/500);
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  auto sol = ns.solve();

  EXPECT_EQ(sol.opt, TerminationStatus::PrimalInfeasible);
}

TEST(KspQpSolveEndToEnd, DetectsDualInfeasibleUnboundedProblem) {
  // min -x, x unconstrained: unbounded below.
  Vec c(1);
  c << -1.0;
  auto problem = MakeProblem(1, 0, 0, SpMat(1, 1), SpMat(0, 1), SpMat(0, 1), c, Vec(0), 0.0,
                              Vec::Constant(1, -kInf), Vec::Constant(1, kInf), Vec(0), Vec(0),
                              /*tol=*/1e-8, /*max_iter=*/500);
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  auto sol = ns.solve();

  EXPECT_EQ(sol.opt, TerminationStatus::DualInfeasible);
}

TEST(KspQpSolveEndToEnd, HandlesEmptyEqualityAndInequalityBlocks) {
  // m=0, l=0 simultaneously: pure box-constrained QP.
  // min x^2 - 4x s.t. -1<=x<=1: unconstrained minimizer x=2 is clipped to x*=1, obj*=1-4=-3.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(1, 1) << 2.0).finished());
  Vec c(1);
  c << -4.0;
  auto problem = MakeProblem(1, 0, 0, Q, SpMat(0, 1), SpMat(0, 1), c, Vec(0), 0.0,
                              Vec::Constant(1, -1.0), Vec::Constant(1, 1.0), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  auto sol = ns.solve();

  EXPECT_EQ(sol.opt, TerminationStatus::Optimal);
  EXPECT_NEAR(sol.x(0), 1.0, kTol);
  EXPECT_NEAR(sol.obj_val, -3.0, kTol * 10);
}

TEST(KspQpSolveEndToEnd, GeneralPositiveSemidefiniteQReformulationMatchesClosedForm) {
  // Q = [[2,1],[1,2]] (general, forces Q_info==2 -> SOC-style reformulation), c=[-3,-3],
  // unconstrained.  x* = Q^-1 c-negated = solve Qx = -c = [3,3]  =>  x* = [1, 1].
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 2.0, 1.0, 1.0, 2.0).finished());
  Vec c(2);
  c << -3.0, -3.0;
  auto problem = MakeProblem(2, 0, 0, Q, SpMat(0, 2), SpMat(0, 2), c, Vec(0), 0.0,
                              Vec::Constant(2, -kInf), Vec::Constant(2, kInf), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ASSERT_EQ(ns.Q_info, 2);
  auto sol = ns.solve();

  EXPECT_EQ(sol.opt, TerminationStatus::Optimal);
  Vec expected_x(2);
  expected_x << 1.0, 1.0;
  EXPECT_TRUE(sol.x.isApprox(expected_x, kTol));
}

TEST(KspQpSolveEndToEnd, TerminatesWithTimeLimitStatusWhenInjectedClockExceedsTimeLimit) {
  // The in-loop time-limit check runs after the `pmm_tol_achieved < tol` break, so a very tight
  // tol forces a non-converged iteration. The first two clock reads (solving_start and the
  // pre-loop setup check) report no elapsed time, so the limit fires inside the first PMM iteration.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(1, 1) << 2.0).finished());
  Vec c(1);
  c << -4.0;  // x=0 (the initial iterate) is not already optimal, unlike a c=0 problem
  auto problem = MakeProblem(1, 0, 0, Q, SpMat(0, 1), SpMat(0, 1), c, Vec(0), 0.0,
                              Vec::Constant(1, -1.0), Vec::Constant(1, 1.0), Vec(0), Vec(0),
                              /*tol=*/1e-14);
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ns.time_limit = 1.0;  // seconds

  auto base_time = std::chrono::steady_clock::now();
  int calls = 0;
  ns.now_ = [&]() {
    if (calls++ < 2) return base_time;
    return base_time + std::chrono::seconds(10);  // every subsequent call reports 10s elapsed
  };

  auto sol = ns.solve();
  EXPECT_EQ(sol.opt, TerminationStatus::TimeLimit);
  EXPECT_EQ(sol.pmm_iter, 1);

  // A timed-out solve still reports its last accepted iterate (as in the Interrupted and
  // MaxSsnIterations tests): the Ruiz factors and c_orig must survive this break for the
  // post-loop printable_sol()/objective_value().
  EXPECT_EQ(sol.x.size(), 1);
  EXPECT_EQ(sol.z.size(), 1);
  EXPECT_TRUE(std::isfinite(sol.obj_val));
}

TEST(KspQpSolveEndToEnd, TimeLimitCountsSetupTime) {
  // The time limit covers setup + solve, so a setup_time already past the limit stops solve()
  // before the first PMM iteration, still reporting the initial iterate.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(1, 1) << 2.0).finished());
  Vec c(1);
  c << -4.0;
  auto problem = MakeProblem(1, 0, 0, Q, SpMat(0, 1), SpMat(0, 1), c, Vec(0), 0.0,
                              Vec::Constant(1, -1.0), Vec::Constant(1, 1.0), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ns.time_limit = 1.0;   // seconds
  ns.setup_time = 10.0;  // as if setup alone took 10s

  auto sol = ns.solve();
  EXPECT_EQ(sol.opt, TerminationStatus::TimeLimit);
  EXPECT_EQ(sol.pmm_iter, 0);
  EXPECT_EQ(sol.x.size(), 1);
  EXPECT_EQ(sol.z.size(), 1);
  EXPECT_TRUE(std::isfinite(sol.obj_val));
  EXPECT_GE(sol.run_time, 10.0);
}

TEST(KspQpSolveEndToEnd, TerminatesWithMaxSsnIterationsWhenSsnIterationBudgetIsExhausted) {
  // ssn_iter starts at 0 and only accumulates; setting ssn_max_iter=0 guarantees
  // `ssn_iter >= ssn_max_iter` fires on the very first PMM iteration regardless of the problem.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(1, 1) << 2.0).finished());
  Vec c(1);
  c << -4.0;
  auto problem = MakeProblem(1, 0, 0, Q, SpMat(0, 1), SpMat(0, 1), c, Vec(0), 0.0,
                              Vec::Constant(1, -1.0), Vec::Constant(1, 1.0), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ns.ssn_max_iter = 0;

  auto sol = ns.solve();
  EXPECT_EQ(sol.opt, TerminationStatus::MaxSsnIterations);

  // This break fires before the in-loop printable_sol()/objective_value(), so x/y1/y2/z and
  // obj_val must be filled from the last accepted iterate after the loop.
  EXPECT_EQ(sol.x.size(), 1);
  EXPECT_EQ(sol.z.size(), 1);
  EXPECT_TRUE(std::isfinite(sol.obj_val));
}

TEST(KspQpSolveEndToEnd, PopulatesSolutionWhenMaxIterIsZero) {
  // max_iter=0 skips the PMM loop, so the returned solution must be the zero-initialized iterate,
  // not size-0 vectors.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(1, 1) << 2.0).finished());
  Vec c(1);
  c << -4.0;
  auto problem = MakeProblem(1, 0, 0, Q, SpMat(0, 1), SpMat(0, 1), c, Vec(0), 0.0,
                              Vec::Constant(1, -1.0), Vec::Constant(1, 1.0), Vec(0), Vec(0),
                              /*tol=*/1e-14, /*max_iter=*/0);
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);

  auto sol = ns.solve();
  EXPECT_EQ(sol.opt, TerminationStatus::MaxPmmIterations);
  EXPECT_EQ(sol.x.size(), 1);
  EXPECT_EQ(sol.z.size(), 1);
  EXPECT_TRUE(std::isfinite(sol.obj_val));
}

TEST(KspQpSolveEndToEnd, TerminatesWithMaxPmmIterationsWhenIterationBudgetIsExhausted) {
  // tol=1e-14 makes convergence in a single PMM iteration essentially impossible for this
  // non-trivial box QP (x=0 is not already optimal), so max_iter=1 forces the loop to exhaust its
  // budget without any other break condition (infeasibility, time limit, SSN budget) firing.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(1, 1) << 2.0).finished());
  Vec c(1);
  c << -4.0;
  auto problem = MakeProblem(1, 0, 0, Q, SpMat(0, 1), SpMat(0, 1), c, Vec(0), 0.0,
                              Vec::Constant(1, -1.0), Vec::Constant(1, 1.0), Vec(0), Vec(0),
                              /*tol=*/1e-14, /*max_iter=*/1);
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);

  auto sol = ns.solve();
  EXPECT_EQ(sol.opt, TerminationStatus::MaxPmmIterations);
}

TEST(KspQpSolveEndToEnd, TerminatesWithInterruptedStatusAndReportsSolutionWhenInterruptedFlagIsSet) {
  // An always-true interrupted_ fires in the first NS.solve_ssn() call, so this reaches
  // TerminationStatus::Interrupted regardless of the problem.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(1, 1) << 2.0).finished());
  Vec c(1);
  c << -4.0;
  auto problem = MakeProblem(1, 0, 0, Q, SpMat(0, 1), SpMat(0, 1), c, Vec(0), 0.0,
                              Vec::Constant(1, -1.0), Vec::Constant(1, 1.0), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  ns.interrupted_ = [] { return true; };

  auto sol = ns.solve();
  EXPECT_EQ(sol.opt, TerminationStatus::Interrupted);

  // An interrupted solve still reports its last accepted iterate: the Ruiz factors and c_orig must
  // survive this break for the post-loop printable_sol()/objective_value().
  EXPECT_EQ(sol.x.size(), 1);
  EXPECT_EQ(sol.z.size(), 1);
  EXPECT_TRUE(std::isfinite(sol.obj_val));
}

// ===================== report_ hook =====================

TEST(ReportHook, CapturesOnePmmIterationRecordPerIterationWithMatchingFinalObjective) {
  // Same box-constrained QP as KspQpSolveEndToEnd.BoxConstrainedQpActivatesUpperBound.
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 2.0, 0.0, 0.0, 4.0).finished());
  Vec c(2);
  c << -4.0, -8.0;
  auto problem = MakeProblem(2, 0, 0, Q, SpMat(0, 2), SpMat(0, 2), c, Vec(0), 0.0,
                              Vec::Constant(2, -1.0), Vec::Constant(2, 1.0), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);

  // IterationRecord::obj_val is computed only when printing is on, and MakeProblem() turns it off;
  // this test's report_ hook needs it, so turn printing on. The final Solution's obj_val is always computed.
  ns.when = PrintWhen::ALWAYS;
  ns.what = PrintWhat::FULL;

  std::vector<IterationRecord<double>> records;
  ns.report_ = [&](const IterationRecord<double>& r) { records.push_back(r); };

  auto sol = ns.solve();

  ASSERT_FALSE(records.empty());
  EXPECT_EQ(static_cast<int>(records.size()), sol.pmm_iter);
  EXPECT_EQ(records.back().pmm_iter, sol.pmm_iter);
  EXPECT_DOUBLE_EQ(records.back().obj_val, sol.obj_val);
  // pmm_iter should be strictly increasing across the captured trace.
  for (std::size_t i = 1; i < records.size(); ++i) {
    EXPECT_GT(records[i].pmm_iter, records[i - 1].pmm_iter);
  }
}

// =====================================================================================
// Tests for solve()'s decomposed helpers: accept_ssn_iterate, update_multipliers_if_accurate.
// Both are driven directly on an already-constructed KSP_QP<double>, bypassing solve()'s loop.
// =====================================================================================

namespace {

// accept_ssn_iterate(NS) only reads NS.x/NS.y2, so a minimal SSN<double> is enough. Its backing
// storage must outlive it (SSN holds references).
struct MinimalSsnHolder {
  Vec Q_diag;
  SpMat L{0, 0};
  SpMat A{0, 1}, B{0, 1}, A_tr{1, 0}, B_tr{1, 0};
  Vec c = Vec::Zero(1), b = Vec::Zero(0);
  Vec D2_ext_inv = Vec::Ones(1), D1B_diag_inv = Vec::Ones(0);
  Vec lx = Vec::Constant(1, -1.0), ux = Vec::Constant(1, 1.0);
  Vec lw = Vec::Zero(0), uw = Vec::Zero(0);

  SSN<double> Make() {
    return SSN<double>(/*Q_info=*/0, Q_diag, L, A, B, A_tr, B_tr, c, b, D2_ext_inv, D1B_diag_inv,
                        lx, ux, lw, uw, /*n=*/1, /*m=*/0, /*N=*/1, /*M=*/0, /*l=*/0,
                        /*ssn_tol=*/1e-6, /*ssn_max_in_iter=*/50, /*eps_pinf=*/1e-6,
                        /*eps_dinf=*/1e-6);
  }
};

}  // namespace

// ===================== accept_ssn_iterate =====================

TEST(AcceptSsnIterate, ComputesAxBxQxAndDeltasWhenAllBlocksNonzero) {
  SpMat Q = DenseToSparse((Eigen::MatrixXd(2, 2) << 3.0, 0.0, 0.0, 4.0).finished());
  SpMat A = DenseToSparse((Eigen::MatrixXd(1, 2) << 1.0, 1.0).finished());
  SpMat B = DenseToSparse((Eigen::MatrixXd(1, 2) << 1.0, -1.0).finished());
  Vec c = Vec::Zero(2), b(1);
  b << 5.0;
  auto problem = MakeProblem(2, 1, 1, Q, A, B, c, b, 0.0, Vec::Constant(2, -kInf),
                              Vec::Constant(2, kInf), Vec::Constant(1, -kInf),
                              Vec::Constant(1, kInf));
  KSP_QP<double> pmm(problem);
  ASSERT_FALSE(pmm.setup_failed);
  ASSERT_EQ(pmm.Q_info, 1);

  MinimalSsnHolder holder;
  SSN<double> ns = holder.Make();
  ns.x = Vec(2);
  ns.x << 2.0, 3.0;
  ns.y2 = Vec::Zero(0);

  pmm.Ax_old_scratch_ = Vec(1);
  pmm.Ax_old_scratch_ << 1.0;
  pmm.Bx_old_scratch_ = Vec(1);
  pmm.Bx_old_scratch_ << 0.5;

  pmm.accept_ssn_iterate(ns);

  EXPECT_TRUE(pmm.x.isApprox(ns.x));
  Vec expected_Ax(1);
  expected_Ax << 5.0;  // A*x = 1*2 + 1*3
  EXPECT_TRUE(pmm.Ax_scratch_.isApprox(expected_Ax));
  Vec expected_Bx(1);
  expected_Bx << -1.0;  // B*x = 1*2 - 1*3
  EXPECT_TRUE(pmm.Bx_scratch_.isApprox(expected_Bx));
  Vec expected_Qx(2);
  expected_Qx << 6.0, 12.0;  // Q_diag=[3,4] .* x=[2,3]
  EXPECT_TRUE(pmm.Qx_scratch_.isApprox(expected_Qx));
  Vec expected_Adx(1);
  expected_Adx << 4.0;  // 5.0 - 1.0
  EXPECT_TRUE(pmm.Adx_scratch_.isApprox(expected_Adx));
  Vec expected_Bdx(1);
  expected_Bdx << -1.5;  // -1.0 - 0.5
  EXPECT_TRUE(pmm.Bdx_scratch_.isApprox(expected_Bdx));
}

TEST(AcceptSsnIterate, ZerosAxBxQxScratchWhenMAndLAreZeroAndQIsZero) {
  // Degenerate: unconstrained problem (M=0, l=0) with zero Q (Q_info=0). All three scratch
  // vectors must be reset via setZero() -- pre-seeded with garbage here to prove they're not
  // left stale by the skipped branches.
  SpMat Q(2, 2);  // no entries -> Q_info = 0
  auto problem = MakeProblem(2, 0, 0, Q, SpMat(0, 2), SpMat(0, 2), Vec::Zero(2), Vec::Zero(0), 0.0,
                              Vec::Constant(2, -kInf), Vec::Constant(2, kInf), Vec(0), Vec(0));
  KSP_QP<double> pmm(problem);
  ASSERT_FALSE(pmm.setup_failed);
  ASSERT_EQ(pmm.Q_info, 0);
  ASSERT_EQ(pmm.M, 0);
  ASSERT_EQ(pmm.l, 0);

  MinimalSsnHolder holder;
  SSN<double> ns = holder.Make();
  ns.x = Vec(2);
  ns.x << 7.0, -7.0;
  ns.y2 = Vec::Zero(0);

  pmm.Ax_scratch_ = Vec::Constant(0, 0.0);  // (already size 0; nothing to pre-seed for M=0)
  pmm.Bx_scratch_ = Vec::Constant(0, 0.0);
  pmm.Qx_scratch_ = Vec::Constant(2, 999.0);  // garbage, must be zeroed
  pmm.Ax_old_scratch_ = Vec::Zero(0);
  pmm.Bx_old_scratch_ = Vec::Zero(0);

  pmm.accept_ssn_iterate(ns);

  EXPECT_EQ(pmm.Ax_scratch_.size(), 0);
  EXPECT_EQ(pmm.Bx_scratch_.size(), 0);
  EXPECT_TRUE(pmm.Qx_scratch_.isApprox(Vec::Zero(2)));
}

TEST(AcceptSsnIterate, QxScratchStaysZeroWhenQInfoZeroEvenWithNonzeroAB) {
  // Mixed case: Q_info=0 (zero Q) but M>0, l>0 -- proves the three branches are independent.
  SpMat Q(2, 2);  // Q_info = 0
  SpMat A = DenseToSparse((Eigen::MatrixXd(1, 2) << 1.0, 1.0).finished());
  SpMat B = DenseToSparse((Eigen::MatrixXd(1, 2) << 1.0, 0.0).finished());
  Vec c = Vec::Zero(2), b(1);
  b << 0.0;
  auto problem = MakeProblem(2, 1, 1, Q, A, B, c, b, 0.0, Vec::Constant(2, -kInf),
                              Vec::Constant(2, kInf), Vec::Constant(1, -kInf),
                              Vec::Constant(1, kInf));
  KSP_QP<double> pmm(problem);
  ASSERT_FALSE(pmm.setup_failed);
  ASSERT_EQ(pmm.Q_info, 0);

  MinimalSsnHolder holder;
  SSN<double> ns = holder.Make();
  ns.x = Vec(2);
  ns.x << 1.0, 1.0;
  ns.y2 = Vec::Zero(0);
  pmm.Qx_scratch_ = Vec::Constant(2, 999.0);
  pmm.Ax_old_scratch_ = Vec::Zero(1);
  pmm.Bx_old_scratch_ = Vec::Zero(1);

  pmm.accept_ssn_iterate(ns);

  Vec expected_Ax(1);
  expected_Ax << 2.0;
  EXPECT_TRUE(pmm.Ax_scratch_.isApprox(expected_Ax));  // A/B branches ran normally
  EXPECT_TRUE(pmm.Qx_scratch_.isApprox(Vec::Zero(2)));  // Q branch independently zeroed
}

// ===================== update_multipliers_if_accurate =====================

namespace {

KSP_QP<double> MakeMultiplierUpdateProblem() {
  SpMat Q(2, 2);
  SpMat A = DenseToSparse((Eigen::MatrixXd(1, 2) << 1.0, 1.0).finished());
  Vec c = Vec::Zero(2), b(1);
  b << 3.0;
  auto problem = MakeProblem(2, 1, 0, Q, A, SpMat(0, 2), c, b, 0.0, Vec::Constant(2, -1.0),
                              Vec::Constant(2, 1.0), Vec(0), Vec(0));
  return KSP_QP<double>(problem);
}

}  // namespace

TEST(UpdateMultipliersIfAccurate, UpdatesWhenSsnOptIsZeroRegardlessOfTolRatio) {
  KSP_QP<double> pmm = MakeMultiplierUpdateProblem();
  ASSERT_FALSE(pmm.setup_failed);
  pmm.mu = 2.0;
  pmm.x = Vec(2);
  pmm.x << 0.5, 0.5;
  pmm.Ax_scratch_ = Vec(1);
  pmm.Ax_scratch_ << 1.0;  // b=[3] -> residual = 1-3 = -2
  pmm.z = Vec::Zero(2);
  pmm.y1 = Vec::Zero(1);
  pmm.ssn_tol_achieved = 1e6;   // deliberately huge
  pmm.pmm_tol_achieved = 1e-12;  // ratio condition (<=100x) would fail
  Vec delta_y1 = Vec::Zero(1), delta_z = Vec::Zero(2);

  pmm.update_multipliers_if_accurate(/*ssn_opt=*/SSN<double>::TerminationStatus::Optimal, delta_y1, delta_z);

  Vec expected_delta_y1(1);
  expected_delta_y1 << 4.0;  // -mu*(Ax-b) = -2*(1-3) = 4
  EXPECT_TRUE(delta_y1.isApprox(expected_delta_y1));
  EXPECT_TRUE(pmm.y1.isApprox(expected_delta_y1));  // y1 was 0
}

TEST(UpdateMultipliersIfAccurate, UpdatesWhenTolRatioWithinBoundEvenIfSsnOptNonzero) {
  KSP_QP<double> pmm = MakeMultiplierUpdateProblem();
  ASSERT_FALSE(pmm.setup_failed);
  pmm.mu = 1.0;
  pmm.x = Vec(2);
  pmm.x << 0.5, 0.5;
  pmm.Ax_scratch_ = Vec(1);
  pmm.Ax_scratch_ << 1.0;
  pmm.z = Vec::Zero(2);
  pmm.y1 = Vec::Zero(1);
  pmm.ssn_tol_achieved = 1e-4;
  pmm.pmm_tol_achieved = 2e-6;  // ratio = 50 <= 100 -> comfortably satisfies "<="
  Vec delta_y1 = Vec::Zero(1), delta_z = Vec::Zero(2);

  pmm.update_multipliers_if_accurate(/*ssn_opt=*/SSN<double>::TerminationStatus::MaxInnerIterations, delta_y1, delta_z);

  Vec expected_delta_y1(1);
  expected_delta_y1 << 2.0;  // -1*(1-3)
  EXPECT_TRUE(delta_y1.isApprox(expected_delta_y1));
}

TEST(UpdateMultipliersIfAccurate, UpdatesAtExactlyTheHundredTimesRatioBoundary) {
  // The existing tol-ratio test uses ratio=50 (well inside the bound); this pins the `<=` edge
  // itself at ratio == 100 exactly.
  KSP_QP<double> pmm = MakeMultiplierUpdateProblem();
  ASSERT_FALSE(pmm.setup_failed);
  pmm.mu = 1.0;
  pmm.x = Vec(2);
  pmm.x << 0.5, 0.5;
  pmm.Ax_scratch_ = Vec(1);
  pmm.Ax_scratch_ << 1.0;
  pmm.z = Vec::Zero(2);
  pmm.y1 = Vec::Zero(1);
  pmm.pmm_tol_achieved = 1e-6;
  // Computed (not two independent literals): 100*1e-6 is not bit-exactly 1e-4 in double, so
  // deriving ssn_tol_achieved this way is what actually lands exactly on the `<=` boundary.
  pmm.ssn_tol_achieved = 100.0 * pmm.pmm_tol_achieved;
  Vec delta_y1 = Vec::Zero(1), delta_z = Vec::Zero(2);

  pmm.update_multipliers_if_accurate(/*ssn_opt=*/SSN<double>::TerminationStatus::MaxInnerIterations,
                                      delta_y1, delta_z);

  Vec expected_delta_y1(1);
  expected_delta_y1 << 2.0;  // -1*(1-3)
  EXPECT_TRUE(delta_y1.isApprox(expected_delta_y1));  // <=, so exact equality still updates
}

TEST(UpdateMultipliersIfAccurate, SkipsUpdateWhenNeitherConditionHolds) {
  KSP_QP<double> pmm = MakeMultiplierUpdateProblem();
  ASSERT_FALSE(pmm.setup_failed);
  pmm.mu = 1.0;
  pmm.x = Vec(2);
  pmm.x << 0.5, 0.5;
  pmm.Ax_scratch_ = Vec(1);
  pmm.Ax_scratch_ << 1.0;
  pmm.z = Vec::Zero(2);
  Vec y1_before(1);
  y1_before << 9.0;
  pmm.y1 = y1_before;
  pmm.ssn_tol_achieved = 1.0;      // ratio 1.0 / 1e-6 = 1e6 >> 100
  pmm.pmm_tol_achieved = 1e-6;
  Vec delta_y1_before(1);
  delta_y1_before << 42.0;
  Vec delta_z_before(2);
  delta_z_before << 1.0, 2.0;
  Vec delta_y1 = delta_y1_before, delta_z = delta_z_before;

  pmm.update_multipliers_if_accurate(/*ssn_opt=*/SSN<double>::TerminationStatus::MaxInnerIterations, delta_y1, delta_z);

  // ssn_opt != 0 and ssn_tol_achieved > 100*pmm_tol_achieved -> nothing changes.
  EXPECT_TRUE(pmm.y1.isApprox(y1_before));
  EXPECT_TRUE(delta_y1.isApprox(delta_y1_before));
  EXPECT_TRUE(delta_z.isApprox(delta_z_before));
}

TEST(UpdateMultipliersIfAccurate, DeltaZEqualsMinusZWhenBoxMultiplierPlusXIsInterior) {
  // Degenerate identity: if z/mu + x lands strictly inside [lx,ux] (proj is a no-op), then
  // delta_z = mu*(x - (z/mu+x)) = -z exactly, so z_new = z + delta_z = 0.
  KSP_QP<double> pmm = MakeMultiplierUpdateProblem();
  ASSERT_FALSE(pmm.setup_failed);
  pmm.mu = 2.0;
  pmm.x = Vec(2);
  pmm.x << 0.0, 0.0;
  pmm.Ax_scratch_ = Vec(1);
  pmm.Ax_scratch_ << 3.0;  // matches b -> delta_y1 = 0, irrelevant here
  Vec z_before(2);
  z_before << 0.2, -0.2;  // z/mu + x = [0.1,-0.1], interior to lx=-1,ux=1
  pmm.z = z_before;
  pmm.y1 = Vec::Zero(1);
  Vec delta_y1 = Vec::Zero(1), delta_z = Vec::Zero(2);

  pmm.update_multipliers_if_accurate(/*ssn_opt=*/SSN<double>::TerminationStatus::Optimal, delta_y1, delta_z);

  EXPECT_TRUE(delta_z.isApprox(-z_before));
  EXPECT_TRUE(pmm.z.isApprox(Vec::Zero(2)));
}

TEST(UpdateMultipliersIfAccurate, DeltaZReflectsClippingWhenBoxMultiplierPlusXIsOutsideBounds) {
  KSP_QP<double> pmm = MakeMultiplierUpdateProblem();
  ASSERT_FALSE(pmm.setup_failed);
  pmm.mu = 1.0;
  pmm.x = Vec(2);
  pmm.x << 0.0, 0.0;
  pmm.Ax_scratch_ = Vec(1);
  pmm.Ax_scratch_ << 3.0;
  Vec z_before(2);
  z_before << 5.0, -5.0;  // z/mu + x = [5,-5], clipped to [1,-1] (lx=-1,ux=1)
  pmm.z = z_before;
  pmm.y1 = Vec::Zero(1);
  Vec delta_y1 = Vec::Zero(1), delta_z = Vec::Zero(2);

  pmm.update_multipliers_if_accurate(/*ssn_opt=*/SSN<double>::TerminationStatus::Optimal, delta_y1, delta_z);

  // delta_z = mu*(x - proj(z/mu+x, lx, ux)) = 1*([0,0] - [1,-1]) = [-1, 1]
  Vec expected_delta_z(2);
  expected_delta_z << -1.0, 1.0;
  EXPECT_TRUE(delta_z.isApprox(expected_delta_z));
  Vec expected_z(2);
  expected_z << 4.0, -4.0;  // z_before + delta_z
  EXPECT_TRUE(pmm.z.isApprox(expected_z));
}

// ===================== solve()'s primal_infeas certificate ordering =====================
// solve() must call update_multipliers_if_accurate() before primal_infeas(), so delta_y1/delta_z
// come from the same PMM step as y2 - y2_old_scratch_. This replays accept_ssn_iterate(),
// update_multipliers_if_accurate(), primal_infeas() on a hand-built infeasible certificate: that
// order detects infeasibility, while stale delta_y1/delta_z would miss it.

TEST(SolvePrimalInfeasCertificateOrdering, FreshDeltaY1AfterMultiplierUpdateDetectsInfeasibility) {
  SpMat A = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  SpMat B = DenseToSparse((Eigen::MatrixXd(1, 1) << 1.0).finished());
  Vec c = Vec::Zero(1), b(1);
  b << 1000.0;  // large positive: makes condition 2 comfortably satisfied once delta_y1 != 0.
  // w is pinned to 0 while x = 1000: infeasible, and the finite lw/uw keep condition 3 satisfied
  // for the nonzero cert_y2 below (a free w would make condition 2's support function +inf).
  auto problem = MakeProblem(1, 1, 1, SpMat(1, 1), A, B, c, b, 0.0,
                              Vec::Constant(1, -kInf), Vec::Constant(1, kInf),
                              Vec::Zero(1), Vec::Zero(1));
  KSP_QP<double> pmm(problem);
  ASSERT_FALSE(pmm.setup_failed);
  ASSERT_GT(pmm.b(0), 0.0);  // ruiz scaling preserves sign

  pmm.mu = 2.0;
  pmm.z = Vec::Zero(1);
  pmm.y1 = Vec::Zero(1);
  pmm.y2 = Vec::Constant(1, 0.3);  // previous-iteration y2
  Vec y2_old_scratch = pmm.y2;

  // The upcoming SSN solve moves x to 0, so Ax_scratch_ becomes 0 and the fresh
  // delta_y1 = -mu*(0 - b) = mu*b is nonzero and known ahead of time.
  const double expected_delta_y1 = pmm.mu * pmm.b(0);
  // Condition 1 of primal_infeas needs A_tr*cert_y1 + B_tr*cert_y2 - cert_z ~= 0; with cert_z=0
  // (delta_z comes out exactly 0 below, since x is unconstrained), solve for the cert_y2 (i.e.
  // the SSN's y2 move) that exactly cancels it against the fresh delta_y1.
  const double cert_y2 = -pmm.A_tr.coeff(0, 0) * expected_delta_y1 / pmm.B_tr.coeff(0, 0);

  MinimalSsnHolder holder;
  SSN<double> ns = holder.Make();
  ns.x = Vec::Zero(1);
  ns.y2 = Vec::Constant(1, y2_old_scratch(0) + cert_y2);

  pmm.Ax_old_scratch_ = Vec::Zero(1);
  pmm.Bx_old_scratch_ = Vec::Zero(1);
  pmm.accept_ssn_iterate(ns);  // pmm.x=0, pmm.y2=ns.y2, Ax_scratch_=Bx_scratch_=0.

  Vec delta_y1 = Vec::Zero(1), delta_z = Vec::Zero(1);
  pmm.update_multipliers_if_accurate(SSN<double>::TerminationStatus::Optimal, delta_y1, delta_z);
  ASSERT_NEAR(delta_y1(0), expected_delta_y1, 1e-9);
  ASSERT_TRUE(delta_z.isApprox(Vec::Zero(1)));  // x unconstrained -> proj is a no-op

  Vec cert_y2_vec = pmm.y2 - y2_old_scratch;

  // Fixed order: primal_infeas() sees this iteration's own (freshly-updated) delta_y1/delta_z.
  EXPECT_TRUE(pmm.primal_infeas(delta_y1, cert_y2_vec, delta_z));

  // Wrong order: delta_y1/delta_z from before update_multipliers_if_accurate() (still zero here)
  // with the same cert_y2_vec. Condition 1 no longer cancels (and condition 2 fails, since
  // -b*cert_y1 is 0), so infeasibility is missed.
  Vec stale_delta_y1 = Vec::Zero(1), stale_delta_z = Vec::Zero(1);
  EXPECT_FALSE(pmm.primal_infeas(stale_delta_y1, cert_y2_vec, stale_delta_z));
}

// ===================== direct fallback =====================
// The direct solver only takes over after PCG fails, so a problem PCG solves never touches it.

TEST(KspQpSolveEndToEnd, PcgSuccessNeverUsesTheDirectFallback) {
  SpMat A = DenseToSparse((Eigen::MatrixXd(1, 2) << 1.0, 1.0).finished());
  Vec c(2);
  c << 1.0, -1.0;
  Vec b(1);
  b << 1.0;
  auto problem = MakeProblem(2, 1, 0, SpMat(2, 2), A, SpMat(0, 2), c, b, 0.0, Vec::Constant(2, 0.0),
                              Vec::Constant(2, 1.0), Vec(0), Vec(0));
  KSP_QP<double> ns(problem);
  ASSERT_FALSE(ns.setup_failed);
  auto sol = ns.solve();

  EXPECT_EQ(sol.opt, TerminationStatus::Optimal);
  EXPECT_FALSE(ns.pcg_failed);
  EXPECT_EQ(ns.kkt_ldlt_fact, 0);
  EXPECT_EQ(ns.schur_chol_fact, 0);
  EXPECT_GT(sol.fact, 0);  // all of them preconditioner factorizations
}
