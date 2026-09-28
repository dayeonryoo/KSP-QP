#include "ssn.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace {

using Vec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<double>;
using BoolArr = Eigen::Array<bool, Eigen::Dynamic, 1>;

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

BoolArr ToBoolArr(const std::vector<bool>& v) {
  BoolArr arr(static_cast<int>(v.size()));
  for (std::size_t i = 0; i < v.size(); ++i) arr(static_cast<int>(i)) = v[i];
  return arr;
}

struct SsnFixture {
  int Q_info = 0;
  Vec Q_diag;
  SpMat L = SpMat(0, 0);
  SpMat A, B, A_tr, B_tr;
  Vec c, b;
  Vec D2_ext_inv, D1B_diag_inv;
  Vec lx, ux, lw, uw;
  int n = 0, m = 0, N = 0, M = 0, l = 0;

  // A: M x N (equality block), B: l x N (candidate inequality/"W" block).
  SsnFixture(const SpMat& A_, const SpMat& B_) : A(A_), B(B_) {
    N = static_cast<int>(A.cols());
    M = static_cast<int>(A.rows());
    l = static_cast<int>(B.rows());
    n = N;
    m = M;
    A_tr = SpMat(A.transpose());
    B_tr = SpMat(B.transpose());
    c = Vec::Zero(N);
    b = Vec::Zero(M);
    D2_ext_inv = Vec::Ones(N);
    D1B_diag_inv = Vec::Ones(l);
    lx = Vec::Constant(N, -1.0);
    ux = Vec::Constant(N, 1.0);
    lw = Vec::Constant(l, -1.0);
    uw = Vec::Constant(l, 1.0);
  }

  SSN<double> Make() const {
    return SSN<double>(Q_info, Q_diag, L, A, B, A_tr, B_tr, c, b, D2_ext_inv, D1B_diag_inv,
                        lx, ux, lw, uw, n, m, N, M, l,
                        /*ssn_tol=*/1e-6, /*ssn_max_in_iter=*/50,
                        /*eps_pinf=*/1e-6, /*eps_dinf=*/1e-6);
  }
};

// Default shape reused by RebuildG/SplitByMask tests: N=3 primal cols, M=1 equality row
// (A_row=[1,1,1]), l=2 candidate W rows (B_row0=[1,0,0], B_row1=[0,1,0]).
SpMat DefaultA() { return DenseToSparse((Eigen::MatrixXd(1, 3) << 1.0, 1.0, 1.0).finished()); }
SpMat DefaultB() {
  return DenseToSparse((Eigen::MatrixXd(2, 3) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0).finished());
}

Eigen::MatrixXd Dense(const SpMat& sp) { return Eigen::MatrixXd(sp); }

}  // namespace

// ===================== compute_subgrad_and_dist (static) =====================

TEST(SSN, ComputeSubgradAndDistInteriorPointGivesSubgradOneDistZero) {
  Vec u(2), lo(2), hi(2), dist;
  BoolArr subgrad;
  u << 0.5, -0.5;
  lo << -1.0, -1.0;
  hi << 1.0, 1.0;
  SSN<double>::compute_subgrad_and_dist(u, lo, hi, /*include_bd=*/false, subgrad, dist);
  EXPECT_TRUE(subgrad(0));
  EXPECT_TRUE(subgrad(1));
  EXPECT_DOUBLE_EQ(dist(0), 0.0);
  EXPECT_DOUBLE_EQ(dist(1), 0.0);
}

TEST(SSN, ComputeSubgradAndDistBoundaryIncludeBdTrueCountsActive) {
  Vec u(1), lo(1), hi(1), dist;
  BoolArr subgrad;
  u << -1.0;
  lo << -1.0;
  hi << 1.0;
  SSN<double>::compute_subgrad_and_dist(u, lo, hi, /*include_bd=*/true, subgrad, dist);
  EXPECT_TRUE(subgrad(0));
  EXPECT_DOUBLE_EQ(dist(0), 0.0);

  SSN<double>::compute_subgrad_and_dist(u, lo, hi, /*include_bd=*/false, subgrad, dist);
  EXPECT_FALSE(subgrad(0));  // strict inequality fails exactly at the boundary
}

TEST(SSN, ComputeSubgradAndDistOutsideBoundsGivesZeroSubgradNonzeroDist) {
  Vec u(1), lo(1), hi(1), dist;
  BoolArr subgrad;
  u << 2.0;
  lo << -1.0;
  hi << 1.0;
  SSN<double>::compute_subgrad_and_dist(u, lo, hi, false, subgrad, dist);
  EXPECT_FALSE(subgrad(0));
  EXPECT_DOUBLE_EQ(dist(0), 1.0);  // u - proj(u) = 2 - 1
}

// ===================== compute_grad_Lagrangian / compute_grad_Lagrangian_unscaled_inf_norm =====================

TEST(SSN, GradLagrangianAtInteriorZeroStateWithNoInequalityRowsIsExactlyZero) {
  // l = 0: no W/inequality block at all (distinct from the existing M=0 coverage
  // elsewhere in this file, which always keeps l=2).
  SpMat A = DenseToSparse((Eigen::MatrixXd(1, 2) << 1.0, 1.0).finished());
  SpMat B(0, 2);
  SsnFixture f(A, B);
  SSN<double> ns = f.Make();

  Vec x0 = Vec::Zero(2), y10 = Vec::Zero(1), y20 = Vec::Zero(0), z0 = Vec::Zero(2);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(2);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, /*mu=*/1.0, /*rho=*/1.0, /*alpha=*/0.95, 0);

  Vec x_new = Vec::Zero(2), y2_new = Vec::Zero(0), Ax_new = Vec::Zero(1), Bx_new = Vec::Zero(0);
  const Vec& grad_L = ns.compute_grad_Lagrangian(x_new, y2_new, Ax_new, Bx_new);

  ASSERT_EQ(grad_L.size(), 2);  // N + l = 2 + 0
  EXPECT_TRUE(grad_L.isApprox(Vec::Zero(2)));

  const double tol = ns.compute_grad_Lagrangian_unscaled_inf_norm(grad_L);
  EXPECT_DOUBLE_EQ(tol, 0.0);  // inf_norm's empty-tail-block guard exercised end-to-end
}

// ===================== split_by_mask / retrieve_row_order =====================

TEST(SSN, SplitByMaskThenRetrieveRowOrderRoundTrips) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();

  Vec u(4);
  u << 10.0, 20.0, 30.0, 40.0;
  BoolArr mask(4);
  mask << true, false, true, false;

  Vec u_sel(2), u_unsel(2);
  ns.split_by_mask(u, mask, u_sel, u_unsel);
  Vec expected_sel(2), expected_unsel(2);
  expected_sel << 10.0, 30.0;
  expected_unsel << 20.0, 40.0;
  EXPECT_TRUE(u_sel.isApprox(expected_sel));
  EXPECT_TRUE(u_unsel.isApprox(expected_unsel));

  Vec out;
  ns.retrieve_row_order(u_sel, u_unsel, mask, out);
  EXPECT_TRUE(out.isApprox(u));
}

TEST(SSN, SplitByMaskAllTrueMaskPutsEverythingInSelected) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();

  Vec u(3);
  u << 1.0, 2.0, 3.0;
  BoolArr mask(3);
  mask << true, true, true;

  Vec u_sel(3), u_unsel(0);
  ns.split_by_mask(u, mask, u_sel, u_unsel);
  EXPECT_TRUE(u_sel.isApprox(u));
  EXPECT_EQ(u_unsel.size(), 0);
}

TEST(SSN, SplitByMaskAllFalseMaskPutsEverythingInUnselected) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();

  Vec u(3);
  u << 1.0, 2.0, 3.0;
  BoolArr mask(3);
  mask << false, false, false;

  Vec u_sel(0), u_unsel(3);
  ns.split_by_mask(u, mask, u_sel, u_unsel);
  EXPECT_EQ(u_sel.size(), 0);
  EXPECT_TRUE(u_unsel.isApprox(u));
}

TEST(SSN, SplitByMaskEmptyMaskIsANoOp) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();

  Vec u(0), u_sel(0), u_unsel(0);
  BoolArr mask(0);
  ns.split_by_mask(u, mask, u_sel, u_unsel);
  EXPECT_EQ(u_sel.size(), 0);
  EXPECT_EQ(u_unsel.size(), 0);
}

// ===================== rebuild_G =====================

TEST(SSN, RebuildGStacksAWithActiveBRowsInOriginalOrder) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();

  ns.active_W = ToBoolArr({true, false});
  ns.n_active_W = 1;
  ns.n_inactive_W = 1;
  ns.rebuild_G();

  Eigen::MatrixXd expected_G(2, 3);
  expected_G << 1, 1, 1,  1, 0, 0;
  EXPECT_TRUE(Dense(ns.G).isApprox(expected_G));
  EXPECT_TRUE(Dense(ns.G).bottomRows(ns.n_active_W)
                  .isApprox((Eigen::MatrixXd(1, 3) << 1, 0, 0).finished()));
  EXPECT_TRUE(Dense(ns.B_inactive_W).isApprox((Eigen::MatrixXd(1, 3) << 0, 1, 0).finished()));
}

TEST(SSN, RebuildGHandlesAllRowsInactiveW) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();

  ns.active_W = ToBoolArr({false, false});
  ns.n_active_W = 0;
  ns.n_inactive_W = 2;
  ns.rebuild_G();

  EXPECT_TRUE(Dense(ns.G).isApprox((Eigen::MatrixXd(1, 3) << 1, 1, 1).finished()));
  EXPECT_EQ(ns.G.rows(), f.M);  // no active-W rows appended to A
  Eigen::MatrixXd expected_inactive(2, 3);
  expected_inactive << 1, 0, 0,  0, 1, 0;
  EXPECT_TRUE(Dense(ns.B_inactive_W).isApprox(expected_inactive));
}

TEST(SSN, RebuildGHandlesAllRowsActiveW) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();

  ns.active_W = ToBoolArr({true, true});
  ns.n_active_W = 2;
  ns.n_inactive_W = 0;
  ns.rebuild_G();

  Eigen::MatrixXd expected_G(3, 3);
  expected_G << 1, 1, 1,  1, 0, 0,  0, 1, 0;
  EXPECT_TRUE(Dense(ns.G).isApprox(expected_G));
  Eigen::MatrixXd expected_active(2, 3);
  expected_active << 1, 0, 0,  0, 1, 0;
  EXPECT_TRUE(Dense(ns.G).bottomRows(ns.n_active_W).isApprox(expected_active));
  EXPECT_EQ(ns.B_inactive_W.rows(), 0);
}

TEST(SSN, RebuildGHandlesZeroInequalityRowsGEqualsAOnly) {
  SpMat A = DefaultA();  // 1x3
  SpMat B(0, 3);          // l = 0
  SsnFixture f(A, B);
  SSN<double> ns = f.Make();

  ns.active_W = BoolArr(0);
  ns.n_active_W = 0;
  ns.n_inactive_W = 0;
  ns.rebuild_G();

  EXPECT_TRUE(Dense(ns.G).isApprox(Dense(A)));
  EXPECT_EQ(ns.G.rows(), f.M);  // no active-W rows appended to A
  EXPECT_EQ(ns.B_inactive_W.rows(), 0);
}

// ===================== choose_schur_ldlt =====================

TEST(SSN, ChooseSchurLdltPrefersLdltForNarrowActiveSetOnWideG) {
  // 10 equality rows, 1 active (fully dense) K column: ratio ~ 0.087 < 0.1 -> LDLT.
  const int s = 10, N = 1;
  SsnFixture f(SpMat(0, N), SpMat(0, N));
  SSN<double> ns = f.Make();

  Eigen::MatrixXd G_dense = Eigen::MatrixXd::Ones(s, N);
  const SpMat G = DenseToSparse(G_dense);
  const BoolArr active_K = ToBoolArr({true});

  EXPECT_TRUE(ns.choose_schur_ldlt(G, active_K));
}

TEST(SSN, ChooseSchurLdltPrefersSchurForSparseWideActiveSet) {
  // 3 equality rows, 10 active K columns, each with a single nonzero: ratio ~ 27.9 >> 0.1 -> Schur.
  const int s = 3, N = 10;
  SsnFixture f(SpMat(0, N), SpMat(0, N));
  SSN<double> ns = f.Make();

  Eigen::MatrixXd G_dense = Eigen::MatrixXd::Zero(s, N);
  for (int k = 0; k < N; ++k) G_dense(k % s, k) = 1.0;
  const SpMat G = DenseToSparse(G_dense);
  std::vector<bool> active_k(N, true);
  const BoolArr active_K = ToBoolArr(active_k);

  EXPECT_FALSE(ns.choose_schur_ldlt(G, active_K));
}

TEST(SSN, ChooseSchurLdltHandlesZeroRowsAndZeroActiveColumnsWithoutNaN) {
  // s = G.rows() = 0 and t = active_K.count() = 0 simultaneously
  // but `if (S_nnz <= 0) return true;` always short-circuits before s/(t+s) computation.
  const int N = 1;
  SsnFixture f(SpMat(0, N), SpMat(0, N));
  SSN<double> ns = f.Make();

  const SpMat G(0, N);
  const BoolArr active_K = ToBoolArr({false});

  EXPECT_TRUE(ns.choose_schur_ldlt(G, active_K));
}

// ===================== exact_line_search =====================
namespace {

struct LineSearchCase {
  Vec lx, ux, lw, uw, z, y2, x, c, A_tr_y1, Q_diag, grad_Atr_resp, b;
  int N, l;

  explicit LineSearchCase(int N_, int l_ = 0) : N(N_), l(l_) {
    lx = Vec::Constant(N, -1.0);
    ux = Vec::Constant(N, 1.0);
    lw = Vec::Constant(l, -1.0);
    uw = Vec::Constant(l, 1.0);
    z = Vec::Zero(N);
    y2 = Vec::Zero(l);
    x = Vec::Zero(N);
    c = Vec::Zero(N);
    A_tr_y1 = Vec::Zero(N);
    Q_diag = Vec::Zero(N);
    grad_Atr_resp = Vec::Zero(N);  // A_tr * (Ax - b): no equality constraints in any of these cases
    b = Vec::Zero(0);
  }

  SsnLineSearchParams<double> Params(double mu = 1.0, double rho = 1.0, double alpha = 1.0) const {
    return SsnLineSearchParams<double>{
        mu, rho, alpha, /*eps_zero=*/1e-12, /*eps_direction=*/1e-8,
        std::numeric_limits<double>::infinity(),
        /*Q_info=*/0, N, l, lx, ux, lw, uw, z, y2, x, c, A_tr_y1, Q_diag, grad_Atr_resp, b};
  }
};

}  // namespace

TEST(ExactLineSearch, ReturnsFullStepWhenNoBoundIsCrossed) {
  // K is unbounded, so there are no breakpoints; psi is a single global quadratic
  // psi(t) = (eta/2) t^2 + zeta t + const with eta = dx^2/rho = 4. Pick c so that
  // zeta = -4 = -eta, i.e. psi'(0) = -eta exactly, matching what an exact Newton
  // step is constructed to satisfy: psi'(1) = zeta + eta = 0. The unconstrained
  // minimizer -zeta/eta then lands exactly at tau = 1.
  LineSearchCase lc(1);
  lc.lx(0) = -std::numeric_limits<double>::infinity();
  lc.ux(0) = std::numeric_limits<double>::infinity();
  lc.c(0) = -2.0;
  const auto p = lc.Params();

  Vec x_curr = Vec::Zero(1), y2_curr = Vec::Zero(0), dx(1), dy2 = Vec::Zero(0);
  dx << 2.0;
  Vec Ax = Vec::Zero(0), Bx = Vec::Zero(0), Adx = Vec::Zero(0), Bdx = Vec::Zero(0);
  Vec dist_K_s = Vec::Zero(1), dist_W_v = Vec::Zero(0);
  Vec ls_s(1), ls_v(0), ls_dv(0);
  std::vector<SsnBreakpoint<double>> breakpoints;

  const double tau = exact_line_search(p, x_curr, y2_curr, dx, dy2, Ax, Bx, Adx, Bdx, dist_K_s,
                                        dist_W_v, ls_s, ls_v, ls_dv, breakpoints);
  EXPECT_DOUBLE_EQ(tau, 1.0);
}

TEST(ExactLineSearch, ReturnsZeroWhenNoBoundIsCrossedAndInitialDerivativeIsNonNegative) {
  // Same unbounded-K setup as above, but with psi'(0) = zeta = 0 (c = 0, dist_K_s = 0):
  // since eta > 0, psi is strictly increasing on [0, inf), so t = 0 is the true
  // minimizer even though no bound is ever crossed.
  LineSearchCase lc(1);
  lc.lx(0) = -std::numeric_limits<double>::infinity();
  lc.ux(0) = std::numeric_limits<double>::infinity();
  const auto p = lc.Params();

  Vec x_curr = Vec::Zero(1), y2_curr = Vec::Zero(0), dx(1), dy2 = Vec::Zero(0);
  dx << 2.0;
  Vec Ax = Vec::Zero(0), Bx = Vec::Zero(0), Adx = Vec::Zero(0), Bdx = Vec::Zero(0);
  Vec dist_K_s = Vec::Zero(1), dist_W_v = Vec::Zero(0);
  Vec ls_s(1), ls_v(0), ls_dv(0);
  std::vector<SsnBreakpoint<double>> breakpoints;

  const double tau = exact_line_search(p, x_curr, y2_curr, dx, dy2, Ax, Bx, Adx, Bdx, dist_K_s,
                                        dist_W_v, ls_s, ls_v, ls_dv, breakpoints);
  EXPECT_DOUBLE_EQ(tau, 0.0);
}

TEST(ExactLineSearch, ReturnsZeroWhenInitialDerivativeIsNonNegative) {
  LineSearchCase lc(1);
  lc.c(0) = 1.0;  // positive cost gradient in the dx direction -> psi'(0) >= 0
  const auto p = lc.Params();

  Vec x_curr = Vec::Zero(1), y2_curr = Vec::Zero(0), dx(1), dy2 = Vec::Zero(0);
  dx << 2.0;
  Vec Ax = Vec::Zero(0), Bx = Vec::Zero(0), Adx = Vec::Zero(0), Bdx = Vec::Zero(0);
  Vec dist_K_s = Vec::Zero(1), dist_W_v = Vec::Zero(0);
  Vec ls_s(1), ls_v(0), ls_dv(0);
  std::vector<SsnBreakpoint<double>> breakpoints;

  const double tau = exact_line_search(p, x_curr, y2_curr, dx, dy2, Ax, Bx, Adx, Bdx, dist_K_s,
                                        dist_W_v, ls_s, ls_v, ls_dv, breakpoints);
  EXPECT_DOUBLE_EQ(tau, 0.0);
}

TEST(ExactLineSearch, FindsExactBreakpointForSingleActiveBoundCrossing) {
  // s(0) = z/mu + x_curr = 0 crosses ux=1 at t = (1-0)/2 = 0.5; hand-derived tau = 0.5
  // (eta=4, zeta=-2, m=4 at t=0.5 -> p_t=0 exactly at the breakpoint).
  LineSearchCase lc(1);
  lc.c(0) = -1.0;
  const auto p = lc.Params();

  Vec x_curr = Vec::Zero(1), y2_curr = Vec::Zero(0), dx(1), dy2 = Vec::Zero(0);
  dx << 2.0;
  Vec Ax = Vec::Zero(0), Bx = Vec::Zero(0), Adx = Vec::Zero(0), Bdx = Vec::Zero(0);
  Vec dist_K_s = Vec::Zero(1), dist_W_v = Vec::Zero(0);
  Vec ls_s(1), ls_v(0), ls_dv(0);
  std::vector<SsnBreakpoint<double>> breakpoints;

  const double tau = exact_line_search(p, x_curr, y2_curr, dx, dy2, Ax, Bx, Adx, Bdx, dist_K_s,
                                        dist_W_v, ls_s, ls_v, ls_dv, breakpoints);
  EXPECT_NEAR(tau, 0.5, 1e-12);
}

TEST(ExactLineSearch, MergesCoincidentBreakpointsWithoutDoubleCountingSlope) {
  // Both variables cross their upper bound at the same t=0.5;
  // the two breakpoints must be merged into a single entry.
  LineSearchCase lc(2);
  lc.c << -1.0, -1.0;
  const auto p = lc.Params();

  Vec x_curr = Vec::Zero(2), y2_curr = Vec::Zero(0), dx(2), dy2 = Vec::Zero(0);
  dx << 2.0, 2.0;
  Vec Ax = Vec::Zero(0), Bx = Vec::Zero(0), Adx = Vec::Zero(0), Bdx = Vec::Zero(0);
  Vec dist_K_s = Vec::Zero(2), dist_W_v = Vec::Zero(0);
  Vec ls_s(2), ls_v(0), ls_dv(0);
  std::vector<SsnBreakpoint<double>> breakpoints;

  const double tau = exact_line_search(p, x_curr, y2_curr, dx, dy2, Ax, Bx, Adx, Bdx, dist_K_s,
                                        dist_W_v, ls_s, ls_v, ls_dv, breakpoints);
  EXPECT_NEAR(tau, 0.5, 1e-12);
  EXPECT_EQ(breakpoints.size(), 1u);
}

TEST(ExactLineSearch, WBoxBreakpointsAlsoContribute) {
  // K is unbounded (contributes no breakpoints); the single breakpoint at t=0.5 comes entirely
  // from the W-block (B=[1], lw=-1, uw=1), isolating that branch of the algorithm.
  LineSearchCase lc(1, /*l=*/1);
  lc.lx(0) = -std::numeric_limits<double>::infinity();
  lc.ux(0) = std::numeric_limits<double>::infinity();
  lc.c(0) = -1.0;
  const auto p = lc.Params();

  Vec x_curr = Vec::Zero(1), y2_curr = Vec::Zero(1), dx(1), dy2 = Vec::Zero(1);
  dx << 2.0;
  Vec Ax = Vec::Zero(0), Bx = Vec::Zero(1), Adx = Vec::Zero(0), Bdx(1);
  Bdx << 2.0;
  Vec dist_K_s = Vec::Zero(1), dist_W_v = Vec::Zero(1);
  Vec ls_s(1), ls_v(1), ls_dv(1);
  std::vector<SsnBreakpoint<double>> breakpoints;

  const double tau = exact_line_search(p, x_curr, y2_curr, dx, dy2, Ax, Bx, Adx, Bdx, dist_K_s,
                                        dist_W_v, ls_s, ls_v, ls_dv, breakpoints);
  EXPECT_NEAR(tau, 0.5, 1e-12);
}

TEST(ExactLineSearch, AlphaLessThanOneExercisesCrossTermsInWBlock) {
  // K unbounded (isolates the (1-alpha) cross terms to the W block);
  // alpha=0.5 so eta gets a (1-alpha)/mu*dy2^2 = 0.5 contribution
  // and zeta a (1-alpha)/mu*dy2.y2_curr = 0.5*1*(-1) = -0.5 contribution.
  // dx=0, so the only breakpoint comes from dv_i = (1-alpha)/mu*dy2 = 0.5
  // crossing uw=1 from v_i=-0.5 -> t_u=(1-(-0.5))/0.5=3.0, slope_change=+mu/alpha*dv_i^2=+0.5.
  // m=eta=0.5 throughout (nothing is currently outside), p_val=zeta=-0.5<0,
  // so the true minimizer t*=-p_val/m=1.0 lies inside the first segment (< the t=3.0 breakpoint)
  // and is recovered via the breakpoint's linear-interpolation check.
  LineSearchCase lc(1, /*l=*/1);
  lc.lx(0) = -std::numeric_limits<double>::infinity();
  lc.ux(0) = std::numeric_limits<double>::infinity();
  const auto p = lc.Params(/*mu=*/1.0, /*rho=*/1.0, /*alpha=*/0.5);

  Vec x_curr = Vec::Zero(1), y2_curr(1), dx = Vec::Zero(1), dy2(1);
  y2_curr << -1.0;
  dy2 << 1.0;
  Vec Ax = Vec::Zero(0), Bx = Vec::Zero(1), Adx = Vec::Zero(0), Bdx = Vec::Zero(1);
  Vec dist_K_s = Vec::Zero(1), dist_W_v = Vec::Zero(1);
  Vec ls_s(1), ls_v(1), ls_dv(1);
  std::vector<SsnBreakpoint<double>> breakpoints;

  const double tau = exact_line_search(p, x_curr, y2_curr, dx, dy2, Ax, Bx, Adx, Bdx, dist_K_s,
                                        dist_W_v, ls_s, ls_v, ls_dv, breakpoints);
  EXPECT_NEAR(tau, 1.0, 1e-12);
}

TEST(ExactLineSearch, OutsideToInsideCrossingDecreasesSlopeAcrossABoundary) {
  // x_curr=2 starts outside ux=1; dx=-2 drives it back toward the interior,
  // crossing ux at t=0.5 (s_i(0.5)=1) and then li=-1 at t=1.5 (s_i(1.5)=-1).
  // Hand-derived: eta=4 (from (1/rho)*dx^2), m starts at eta+mu*dx^2=8 (currently outside via ux),
  // zeta=-4, dist_K_s=1 (=s_i-proj(s_i)=2-1) so p_val=zeta+mu*dist_K_s*dx=-4-2=-6.
  // At t=0.5 (slope_change=-4, crossing into the interior): p_t=-6+8*0.5=-2<0, continue with m=4.
  // At t=1.5 (slope_change=+4, crossing back out via li): p_t=-2+4*1.0=2>=0 -> tau=0.5-(-2)/4=1.0,
  // landing inside the (0.5,1.5) interior segment as expected.
  LineSearchCase lc(1);
  const auto p = lc.Params();

  Vec x_curr(1), y2_curr = Vec::Zero(0), dx(1), dy2 = Vec::Zero(0);
  x_curr << 2.0;
  dx << -2.0;
  Vec Ax = Vec::Zero(0), Bx = Vec::Zero(0), Adx = Vec::Zero(0), Bdx = Vec::Zero(0);
  Vec dist_K_s(1);
  dist_K_s << 1.0;
  Vec dist_W_v = Vec::Zero(0);
  Vec ls_s(1), ls_v(0), ls_dv(0);
  std::vector<SsnBreakpoint<double>> breakpoints;

  const double tau = exact_line_search(p, x_curr, y2_curr, dx, dy2, Ax, Bx, Adx, Bdx, dist_K_s,
                                        dist_W_v, ls_s, ls_v, ls_dv, breakpoints);
  EXPECT_NEAR(tau, 1.0, 1e-12);
}

TEST(ExactLineSearch, MergesCoincidentKAndWBreakpoints) {
  // K crosses ux=1 at t=0.5 (slope_change=+4) and, independently, W crosses uw=1 at
  // the same t=0.5 (slope_change=+4), i.e. merging across the K/W  block boundary.
  // eta=4 (alpha=1 kills all W-alpha cross terms), zeta=dx*c=2*(-1) =-2,
  // p_val=-2 (interior at t=0, all dist_* terms 0).
  // At the single merged breakpoint (summed slope_change=+8):
  // p_t=-2+4*0.5=0>=0 -> tau=0-(-2)/4=0.5.
  LineSearchCase lc(1, /*l=*/1);
  lc.c(0) = -1.0;
  const auto p = lc.Params(/*mu=*/1.0, /*rho=*/1.0, /*alpha=*/1.0);

  Vec x_curr = Vec::Zero(1), y2_curr = Vec::Zero(1), dx(1), dy2 = Vec::Zero(1);
  dx << 2.0;
  Vec Ax = Vec::Zero(0), Bx = Vec::Zero(1), Adx = Vec::Zero(0), Bdx(1);
  Bdx << 2.0;
  Vec dist_K_s = Vec::Zero(1), dist_W_v = Vec::Zero(1);
  Vec ls_s(1), ls_v(1), ls_dv(1);
  std::vector<SsnBreakpoint<double>> breakpoints;

  const double tau = exact_line_search(p, x_curr, y2_curr, dx, dy2, Ax, Bx, Adx, Bdx, dist_K_s,
                                        dist_W_v, ls_s, ls_v, ls_dv, breakpoints);
  EXPECT_NEAR(tau, 0.5, 1e-12);
  EXPECT_EQ(breakpoints.size(), 1u);
}

TEST(ExactLineSearch, ReturnsZeroWhenInitialDerivativeIsExactlyZeroAtTheBoundary) {
  // Same shape as FindsExactBreakpointForSingleActiveBoundCrossing but c=0,  
  // so zeta=0 and (interior, dist_K_s=0) p_val=0 exactly hits the `p_val >= 0`
  // check's equality branch specifically, distinct from
  // ReturnsZeroWhenInitialDerivativeIsNonNegative which only covers p_val > 0.
  LineSearchCase lc(1);
  const auto p = lc.Params();

  Vec x_curr = Vec::Zero(1), y2_curr = Vec::Zero(0), dx(1), dy2 = Vec::Zero(0);
  dx << 2.0;
  Vec Ax = Vec::Zero(0), Bx = Vec::Zero(0), Adx = Vec::Zero(0), Bdx = Vec::Zero(0);
  Vec dist_K_s = Vec::Zero(1), dist_W_v = Vec::Zero(0);
  Vec ls_s(1), ls_v(0), ls_dv(0);
  std::vector<SsnBreakpoint<double>> breakpoints;

  const double tau = exact_line_search(p, x_curr, y2_curr, dx, dy2, Ax, Bx, Adx, Bdx, dist_K_s,
                                        dist_W_v, ls_s, ls_v, ls_dv, breakpoints);
  EXPECT_DOUBLE_EQ(tau, 0.0);
}

TEST(ExactLineSearch, ReturnsFullStepForDegenerateZeroDirection) {
  // dx=0, dy2=0 with finite box bounds on both K and W: every |dx_i| and |dv_i| is below
  // eps_direction, so no breakpoints are generated at all and eta (the weighted squared norm
  // of the direction) is exactly 0. This isolates the `breakpoints.empty() && eta < eps_zero`
  // guard from the unbounded-box fallthrough exercised by ReturnsFullStepWhenNoBoundIsCrossed.
  LineSearchCase lc(3, /*l=*/2);
  const auto p = lc.Params();

  Vec x_curr = Vec::Zero(3), y2_curr = Vec::Zero(2), dx = Vec::Zero(3), dy2 = Vec::Zero(2);
  Vec Ax = Vec::Zero(0), Bx = Vec::Zero(2), Adx = Vec::Zero(0), Bdx = Vec::Zero(2);
  Vec dist_K_s = Vec::Zero(3), dist_W_v = Vec::Zero(2);
  Vec ls_s(3), ls_v(2), ls_dv(2);
  std::vector<SsnBreakpoint<double>> breakpoints;

  const double tau = exact_line_search(p, x_curr, y2_curr, dx, dy2, Ax, Bx, Adx, Bdx, dist_K_s,
                                        dist_W_v, ls_s, ls_v, ls_dv, breakpoints);
  EXPECT_DOUBLE_EQ(tau, 1.0);
  EXPECT_TRUE(breakpoints.empty());
}

TEST(ExactLineSearch, SlopeSafeguardHandlesNearZeroAccumulatedSlopeFromRounding) {
  // All three coordinates start violating lx=0 (x_curr=-1) and move toward it (dx=1),
  // each crossing at the identical t=1 and each independently contributing mu*dx_i^2=0.1 to
  // m's initial value (three sequential += at ssn.tpp:464-465) and -0.1 to the merged
  // breakpoint's slope_change (ssn.tpp:513-524). Mathematically these cancel exactly, but
  // IEEE-754 rounding of repeated 0.1 additions (cf. the classic 0.1+0.1+0.1 != 0.3) can leave
  // the post-crossing m at exactly 0.0 or a few ULPs off zero rather than a clean value.
  // rho is huge so eta itself is negligible, isolating m to these violation/breakpoint terms.
  LineSearchCase lc(3);
  lc.lx = Vec::Constant(3, 0.0);
  lc.ux = Vec::Constant(3, std::numeric_limits<double>::infinity());
  lc.c(0) = -1.0;
  const auto p = lc.Params(/*mu=*/0.1, /*rho=*/1e18, /*alpha=*/1.0);

  Vec x_curr = Vec::Constant(3, -1.0), y2_curr = Vec::Zero(0), dx = Vec::Constant(3, 1.0),
      dy2 = Vec::Zero(0);
  Vec Ax = Vec::Zero(0), Bx = Vec::Zero(0), Adx = Vec::Zero(0), Bdx = Vec::Zero(0);
  Vec dist_K_s = Vec::Constant(3, -1.0), dist_W_v = Vec::Zero(0);
  Vec ls_s(3), ls_v(0), ls_dv(0);
  std::vector<SsnBreakpoint<double>> breakpoints;

  const double tau = exact_line_search(p, x_curr, y2_curr, dx, dy2, Ax, Bx, Adx, Bdx, dist_K_s,
                                        dist_W_v, ls_s, ls_v, ls_dv, breakpoints);
  EXPECT_TRUE(std::isfinite(tau));
  EXPECT_FALSE(std::isnan(tau));
  EXPECT_EQ(breakpoints.size(), 1u);
  EXPECT_GT(tau, 1.0);
}

TEST(ExactLineSearch, CountsOutwardMovingComponentOnABoundInTheInitialSlope) {
  // s(0) = z/mu + x_curr = 0 lies exactly on lx = 0 and dx = -1 moves it outward, so it is
  // outside K for every t > 0 and adds mu*dx^2 = 1 to the slope from t = 0 on.
  // eta = dx^2/rho = 1, zeta = c*dx = -1, dist_K_s = 0, so psi'(t) = -1 + 2t and tau = 0.5
  // (ignoring the bound would give tau = 1).
  LineSearchCase lc(1);
  lc.lx(0) = 0.0;
  lc.ux(0) = std::numeric_limits<double>::infinity();
  lc.c(0) = 1.0;
  const auto p = lc.Params();

  Vec x_curr = Vec::Zero(1), y2_curr = Vec::Zero(0), dx(1), dy2 = Vec::Zero(0);
  dx << -1.0;
  Vec Ax = Vec::Zero(0), Bx = Vec::Zero(0), Adx = Vec::Zero(0), Bdx = Vec::Zero(0);
  Vec dist_K_s = Vec::Zero(1), dist_W_v = Vec::Zero(0);
  Vec ls_s(1), ls_v(0), ls_dv(0);
  std::vector<SsnBreakpoint<double>> breakpoints;

  const double tau = exact_line_search(p, x_curr, y2_curr, dx, dy2, Ax, Bx, Adx, Bdx, dist_K_s,
                                        dist_W_v, ls_s, ls_v, ls_dv, breakpoints);
  EXPECT_NEAR(tau, 0.5, 1e-12);
}

TEST(ExactLineSearch, CountsOutwardMovingWComponentOnABoundInTheInitialSlope) {
  // W-block analogue: K is unbounded, B = [1], v(0) = B x_curr = 0 lies exactly on lw = 0 and
  // dv = B dx = -1 moves it outward. With alpha = 1: eta = 1, zeta = -1, dist_W_v = 0, and the
  // outward W component adds mu/alpha*dv^2 = 1 to the slope, so tau = 0.5.
  LineSearchCase lc(1, /*l=*/1);
  lc.lx(0) = -std::numeric_limits<double>::infinity();
  lc.ux(0) = std::numeric_limits<double>::infinity();
  lc.lw(0) = 0.0;
  lc.uw(0) = std::numeric_limits<double>::infinity();
  lc.c(0) = 1.0;
  const auto p = lc.Params();

  Vec x_curr = Vec::Zero(1), y2_curr = Vec::Zero(1), dx(1), dy2 = Vec::Zero(1);
  dx << -1.0;
  Vec Ax = Vec::Zero(0), Bx = Vec::Zero(1), Adx = Vec::Zero(0), Bdx(1);
  Bdx << -1.0;
  Vec dist_K_s = Vec::Zero(1), dist_W_v = Vec::Zero(1);
  Vec ls_s(1), ls_v(1), ls_dv(1);
  std::vector<SsnBreakpoint<double>> breakpoints;

  const double tau = exact_line_search(p, x_curr, y2_curr, dx, dy2, Ax, Bx, Adx, Bdx, dist_K_s,
                                        dist_W_v, ls_s, ls_v, ls_dv, breakpoints);
  EXPECT_NEAR(tau, 0.5, 1e-12);
}

// =====================================================================================
// Tests for solve_ssn()'s decomposed per-iteration phases
// =====================================================================================

namespace {

Eigen::MatrixXd DenseA1x3() { return Dense(DefaultA()); }
Eigen::MatrixXd DenseB2x3() { return Dense(DefaultB()); }

// Asserts dxdy_ = [dx; dy] (as split by solve_newton_direction) satisfies K*[dx;dy] = [r1_;r2_],
// K = [-H_diag, G^T; G, (1/mu)I] -- i.e. that the Newton system was actually solved, independent
// of which internal path (PCG+preconditioner vs. LDLT-on-augmented-system) produced it.
void ExpectNewtonSystemSolved(const SSN<double>& ns, double tol = 1e-8) {
  const int N = ns.N;
  const int s = static_cast<int>(ns.r2_.size());
  ASSERT_EQ(ns.dxdy_.size(), N + s);
  Vec dx = ns.dxdy_.head(N);
  Vec dy = ns.dxdy_.tail(s);
  Vec res1 = ns.r1_ + ns.H_diag.cwiseProduct(dx) - ns.G_tr * dy;
  if (res1.size() > 0) EXPECT_LT(res1.cwiseAbs().maxCoeff(), tol);
  if (s > 0) {
    Vec res2 = ns.r2_ - ns.G * dx - dy / ns.mu;
    EXPECT_LT(res2.cwiseAbs().maxCoeff(), tol);
  }
}

}  // namespace

// ===================== update_ssn_system =====================

TEST(UpdateSsnSystem, CachesATrY1ExactlyOnceMatchingIndependentMatVec) {
  SsnFixture f(DefaultA(), DefaultB());  // N=3, M=1, l=2
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3);
  Vec y10(1);
  y10 << 2.5;
  Vec y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, /*mu=*/1.0, /*rho=*/1.0, 0.95, 0);

  EXPECT_TRUE(ns.A_tr_y1_.isApprox(Dense(ns.A_tr) * y10));
}

TEST(UpdateSsnSystem, ATrY1StaysBitIdenticalAcrossRepeatedPrepareAndSolveCallsWithoutReUpdate) {
  // A_tr_y1_ is recomputed unconditionally on every update_ssn_system() call, so the "cache"
  // property under test is that it survives repeated calls to the rest of the per-SSN-iteration
  // pipeline (prepare_newton_system -> solve_newton_direction -> line_search... -> update_iterate,
  // mirroring solve_ssn()'s own loop) without that pipeline ever calling update_ssn_system() again
  // -- exactly how the real outer PMM loop uses it (called once per PMM iteration, read many times
  // across the SSN iterations within it).
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x_pmm(3);
  x_pmm << 1.0, 1.0, 1.0;
  Vec y10(1);
  y10 << 2.5;
  Vec y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x_pmm, y10, y20, z0, dy10, dz0, /*mu=*/1.0, /*rho=*/1.0, /*alpha=*/0.95, 0);
  const Vec cached = ns.A_tr_y1_;

  ns.x_cur_ = ns.x;
  ns.y2_cur_ = ns.y2;
  ns.Ax_ssn_.noalias() = ns.A * ns.x_cur_;
  ns.Bx_ssn_.noalias() = ns.B * ns.x_cur_;

  for (int i = 0; i < 3; ++i) {
    auto [update_prec, prec_pattern_changed] = ns.prepare_newton_system();
    ns.solve_newton_direction(update_prec, prec_pattern_changed);
    auto ls = ns.line_search_with_steepest_descent_fallback(/*ssn_tol=*/1e-6);
    ASSERT_NE(ls.outcome, SSN<double>::LineSearchOutcome::Fail);
    EXPECT_TRUE(ns.A_tr_y1_ == cached);  // never touched by anything in this loop
    if (ls.outcome == SSN<double>::LineSearchOutcome::AcceptOptimal) break;
    ns.update_iterate(ls.tau, i);
  }
}

TEST(UpdateSsnSystem, ATrY1RecomputesOnNextUpdateSsnSystemCallWithNewY1) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3);
  Vec y10(1);
  y10 << 2.5;
  Vec y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);
  ASSERT_TRUE(ns.A_tr_y1_.isApprox(Dense(ns.A_tr) * y10));

  Vec y1_new(1);
  y1_new << -4.0;
  ns.update_ssn_system(x0, y1_new, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 1);

  EXPECT_TRUE(ns.A_tr_y1_.isApprox(Dense(ns.A_tr) * y1_new));
}

// ===================== prepare_newton_system =====================

TEST(PrepareNewtonSystem, FirstCallAtInteriorPointReturnsFullUpdateWithZeroRhs) {
  SsnFixture f(DefaultA(), DefaultB());  // N=3, M=1, l=2
  SSN<double> ns = f.Make();

  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, /*mu=*/1.0, /*rho=*/1.0, /*alpha=*/0.95, 0);
  ns.x_cur_ = Vec::Zero(3);
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Zero(1);
  ns.Bx_ssn_ = Vec::Zero(2);

  auto result = ns.prepare_newton_system();

  EXPECT_TRUE(result.update_prec);
  EXPECT_TRUE(result.prec_pattern_changed);
  EXPECT_TRUE((ns.active_K.array() == true).all());
  EXPECT_TRUE((ns.active_W.array() == false).all());
  EXPECT_EQ(ns.n_active_W, 0);
  EXPECT_EQ(ns.n_inactive_W, 2);

  Vec expected_H_diag = Vec::Constant(3, 1.0);  // mu*(1-1) + 1/rho = 0 + 1
  EXPECT_TRUE(ns.H_diag.isApprox(expected_H_diag));

  ASSERT_EQ(ns.G.rows(), 1);  // M + n_active_W = 1 + 0
  EXPECT_TRUE(Dense(ns.G).isApprox(DenseA1x3()));

  EXPECT_TRUE(ns.r1_.isApprox(Vec::Zero(3)));
  ASSERT_EQ(ns.r2_.size(), 1);
  EXPECT_NEAR(ns.r2_(0), 0.0, 1e-12);

  EXPECT_EQ(ns.schur_ldlt_decisions_made_, 1);
}

TEST(PrepareNewtonSystem, SecondCallWithUnchangedStateSkipsPrecUpdateAndSchurLdltDecision) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);
  ns.x_cur_ = Vec::Zero(3);
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Zero(1);
  ns.Bx_ssn_ = Vec::Zero(2);
  ns.prepare_newton_system();
  ASSERT_EQ(ns.schur_ldlt_decisions_made_, 1);

  auto result = ns.prepare_newton_system();  // identical state as the previous call

  EXPECT_FALSE(result.update_prec);
  EXPECT_FALSE(result.prec_pattern_changed);
  EXPECT_EQ(ns.schur_ldlt_decisions_made_, 1);  // choose_schur_ldlt not re-invoked
}

TEST(PrepareNewtonSystem, ActiveKChangeAloneMarksPrecPatternChangedWithoutRebuildingG) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);
  ns.x_cur_ = Vec::Zero(3);
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Zero(1);
  ns.Bx_ssn_ = Vec::Zero(2);
  ns.prepare_newton_system();
  Eigen::MatrixXd G_before = Dense(ns.G);

  ns.x_cur_(0) = 5.0;  // pushes u_(0) = z(0)/mu + x_cur_(0) = 5 outside ux(0)=1
  ns.Ax_ssn_(0) = 5.0;  // = A*x_cur_ = 5+0+0

  auto result = ns.prepare_newton_system();

  EXPECT_TRUE(result.update_prec);
  EXPECT_TRUE(result.prec_pattern_changed);  // K change alone still flips the pattern flag
  EXPECT_FALSE(ns.active_K(0));
  EXPECT_TRUE(ns.active_K(1));
  EXPECT_TRUE(ns.active_K(2));
  EXPECT_TRUE((ns.active_W.array() == false).all());  // W untouched
  EXPECT_TRUE(Dense(ns.G).isApprox(G_before));  // G itself unchanged (only K changed)

  Vec expected_H_diag(3);
  expected_H_diag << 2.0, 1.0, 1.0;  // dim 0: mu*(1-0)+1/rho=2; dims 1,2: mu*(1-1)+1/rho=1
  EXPECT_TRUE(ns.H_diag.isApprox(expected_H_diag));
}

TEST(PrepareNewtonSystem, ActiveWChangeAloneRebuildsGWithoutChangingActiveK) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);
  ns.x_cur_ = Vec::Zero(3);
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Zero(1);
  ns.Bx_ssn_ = Vec::Zero(2);
  ns.prepare_newton_system();

  ns.Bx_ssn_(0) = 5.0;  // pushes v_(0) outside uw(0)=1 for W row 0 (B row0 = [1,0,0])
  ns.prev_dy_ = Vec::Ones(1);  // poison: a w_changed call must clear this CG warm-start cache

  auto result = ns.prepare_newton_system();

  EXPECT_TRUE(result.update_prec);
  EXPECT_TRUE(result.prec_pattern_changed);
  EXPECT_TRUE(ns.active_W(0));
  EXPECT_FALSE(ns.active_W(1));
  EXPECT_EQ(ns.n_active_W, 1);
  ASSERT_EQ(ns.G.rows(), 2);  // M=1 + n_active_W=1
  EXPECT_TRUE(Dense(ns.G).row(1).isApprox(DenseB2x3().row(0)));
  EXPECT_TRUE((ns.active_K.array() == true).all());  // K untouched
  EXPECT_EQ(ns.prev_dy_.size(), 0);  // w_changed invalidates the CG warm-start cache
}

TEST(PrepareNewtonSystem, HDiagRebuildsOnMuOnlyChangeWithoutActiveSetChange) {
  // Regression test: H_diag = Q + mu*(1-diag_P_K) + I/rho depends on mu, but was previously only
  // rebuilt on delta.k_changed -- a mu-only drift (which happens every PMM outer iteration) left
  // it stale. z=y2=0 in this fixture, so u_/v_ (hence the active set) are unaffected by mu,
  // isolating a pure mu-only change with k_changed/w_changed both false.
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, /*mu=*/1.0, /*rho=*/1.0, 0.95, 0);
  ns.x_cur_ = Vec::Zero(3);
  ns.x_cur_(0) = 5.0;  // pushes u_(0) outside ux(0)=1: active_K = [false, true, true]
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Constant(1, 5.0);
  ns.Bx_ssn_ = Vec::Zero(2);
  ns.prepare_newton_system();
  ASSERT_TRUE(ns.H_diag.isApprox((Vec(3) << 2.0, 1.0, 1.0).finished()));
  ASSERT_DOUBLE_EQ(ns.H_diag_mu_, 1.0);

  ns.mu = 4.0;  // no active-set-affecting change accompanies this
  auto result = ns.prepare_newton_system();

  EXPECT_FALSE(result.update_prec);          // active set genuinely unchanged
  EXPECT_FALSE(result.prec_pattern_changed);
  Vec expected_H_diag(3);
  expected_H_diag << 5.0, 1.0, 1.0;  // dim 0 (inactive_K): mu*(1-0)+1/rho=4+1=5; dims 1,2: 0+1=1
  EXPECT_TRUE(ns.H_diag.isApprox(expected_H_diag));
  EXPECT_DOUBLE_EQ(ns.H_diag_mu_, 4.0);  // bookkeeping updated so the cache stays consistent
}

TEST(PrepareNewtonSystem, HDiagRebuildsOnRhoOnlyChangeWithoutActiveSetChange) {
  // Companion to the mu-only test above.
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, /*mu=*/1.0, /*rho=*/1.0, 0.95, 0);
  ns.x_cur_ = Vec::Zero(3);
  ns.x_cur_(0) = 5.0;  // pushes u_(0) outside ux(0)=1: active_K = [false, true, true]
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Constant(1, 5.0);
  ns.Bx_ssn_ = Vec::Zero(2);
  ns.prepare_newton_system();
  ASSERT_TRUE(ns.H_diag.isApprox((Vec(3) << 2.0, 1.0, 1.0).finished()));
  ASSERT_DOUBLE_EQ(ns.H_diag_rho_, 1.0);

  ns.rho = 0.5;  // no active-set-affecting change accompanies this
  auto result = ns.prepare_newton_system();

  EXPECT_FALSE(result.update_prec);
  EXPECT_FALSE(result.prec_pattern_changed);
  Vec expected_H_diag(3);
  expected_H_diag << 3.0, 2.0, 2.0;  // dim 0: mu*1+1/rho=1+2=3; dims 1,2: mu*0+1/rho=0+2=2
  EXPECT_TRUE(ns.H_diag.isApprox(expected_H_diag));
  EXPECT_DOUBLE_EQ(ns.H_diag_rho_, 0.5);
}

TEST(PrepareNewtonSystem, AllKInactiveWhenFarOutsideBoundsHandlesZeroActiveColumnsInChooseSchurLdlt) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);
  ns.x_cur_ = Vec::Constant(3, 100.0);  // way outside ux=1 for all 3 dims
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Constant(1, 300.0);
  ns.Bx_ssn_ = Vec::Zero(2);

  ns.prepare_newton_system();

  EXPECT_TRUE((ns.active_K.array() == false).all());
  Vec expected_H_diag = Vec::Constant(3, 2.0);  // mu*(1-0)+1/rho = 1+1
  EXPECT_TRUE(ns.H_diag.isApprox(expected_H_diag));
  // choose_schur_ldlt with t=0 active columns: K_nnz=S_nnz=s=1, ratio=(s/s)*(1/1)^2=1, not <0.1.
  EXPECT_FALSE(ns.schur_use_ldlt);
}

TEST(PrepareNewtonSystem, AllWActiveWhenFarOutsideBoundsRebuildsGWithAllRowsAndEmptyInactiveBlock) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);
  ns.x_cur_ = Vec::Zero(3);
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Zero(1);
  ns.Bx_ssn_ = Vec::Constant(2, 100.0);  // both W rows far outside uw=1

  ns.prepare_newton_system();

  EXPECT_TRUE((ns.active_W.array() == true).all());
  EXPECT_EQ(ns.n_active_W, 2);
  EXPECT_EQ(ns.n_inactive_W, 0);
  EXPECT_EQ(ns.B_inactive_W.rows(), 0);
  ASSERT_EQ(ns.G.rows(), 3);  // M=1 + n_active_W=2 = full [A; B]
  Eigen::MatrixXd expected(3, 3);
  expected.topRows(1) = DenseA1x3();
  expected.bottomRows(2) = DenseB2x3();
  EXPECT_TRUE(Dense(ns.G).isApprox(expected));
}

TEST(PrepareNewtonSystem, SchurLdltDecisionLocksAfterThreeActiveSetChanges) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);
  ns.y2_cur_ = Vec::Zero(2);
  ns.Bx_ssn_ = Vec::Zero(2);

  // Four distinct x_cur_ states, alternating so active_K flips (delta.k_changed=true) every call.
  double xs[4] = {0.0, 100.0, 0.0, 100.0};
  for (int i = 0; i < 4; ++i) {
    ns.x_cur_ = Vec::Constant(3, xs[i]);
    ns.Ax_ssn_ = Vec::Constant(1, 3.0 * xs[i]);
    ns.prepare_newton_system();
  }

  EXPECT_EQ(ns.schur_ldlt_decisions_made_, 3);  // locked after the first 3 decisions
}

// ===================== solve_newton_direction =====================

TEST(SolveNewtonDirection, SatisfiesKktResidualWithPcgPath) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);
  ns.x_cur_ = Vec(3);
  ns.x_cur_ << 5.0, 0.0, 0.0;  // dim 0 outside ux=1 -> nonzero r1_
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Constant(1, 5.0);
  ns.Bx_ssn_ = Vec::Zero(2);

  auto prep = ns.prepare_newton_system();
  ASSERT_FALSE(ns.kkt_ldlt_used);  // default; exercises the PCG+preconditioner path
  ns.solve_newton_direction(prep.update_prec, prep.prec_pattern_changed);

  ExpectNewtonSystemSolved(ns);
  EXPECT_EQ(ns.dx_.size(), 3);
  EXPECT_EQ(ns.dy2_.size(), 2);
  // prev_dy_ (CG warm-start cache) must be populated with the converged direction, ground-truthed
  // against the final dxdy_ rather than only inferred from downstream KKT-residual correctness.
  const int s = static_cast<int>(ns.r2_.size());
  EXPECT_TRUE(ns.prev_dy_.isApprox(ns.dxdy_.tail(s)));
}

TEST(SolveNewtonDirection, SatisfiesKktResidualWithLdltFallbackPath) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);
  ns.x_cur_ = Vec(3);
  ns.x_cur_ << 5.0, 0.0, 0.0;
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Constant(1, 5.0);
  ns.Bx_ssn_ = Vec::Zero(2);

  auto prep = ns.prepare_newton_system();
  ns.kkt_ldlt_used = true;  // force the permanent LDLT-on-augmented-KKT-system fallback
  ns.solve_newton_direction(prep.update_prec, prep.prec_pattern_changed);

  ExpectNewtonSystemSolved(ns);
}

TEST(SolveNewtonDirection, HandlesZeroActiveWDegenerateCase) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);
  ns.x_cur_ = Vec::Zero(3);
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Zero(1);
  ns.Bx_ssn_ = Vec::Zero(2);  // interior -> n_active_W = 0

  auto prep = ns.prepare_newton_system();
  ASSERT_EQ(ns.n_active_W, 0);
  ns.solve_newton_direction(prep.update_prec, prep.prec_pattern_changed);

  ASSERT_EQ(ns.dxdy_.size(), 3 + 1);  // N + (M + n_active_W) = 3 + 1
  ExpectNewtonSystemSolved(ns);
  EXPECT_EQ(ns.dy2_.size(), 2);  // full dy2_ still recovered via the inactive-W path
}

TEST(SolveNewtonDirection, HandlesAllWActiveDegenerateCase) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);
  ns.x_cur_ = Vec::Zero(3);
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Zero(1);
  ns.Bx_ssn_ = Vec::Constant(2, 100.0);  // both W rows active -> n_active_W = l = 2

  auto prep = ns.prepare_newton_system();
  ASSERT_EQ(ns.n_active_W, 2);
  ASSERT_EQ(ns.n_inactive_W, 0);
  ns.solve_newton_direction(prep.update_prec, prep.prec_pattern_changed);

  ASSERT_EQ(ns.dxdy_.size(), 3 + 3);  // N + (M + n_active_W) = 3 + (1+2)
  ExpectNewtonSystemSolved(ns);
}

TEST(SolveNewtonDirection, HandlesNoEqualityConstraintsMEqualsZero) {
  SpMat zero_row_A(0, 3);  // M = 0
  SsnFixture f(zero_row_A, DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(0), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(0), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);
  ns.x_cur_ = Vec::Zero(3);
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Zero(0);
  ns.Bx_ssn_ = Vec::Zero(2);

  auto prep = ns.prepare_newton_system();
  ASSERT_EQ(ns.r2_.size(), 0);  // M=0, n_active_W=0
  ns.solve_newton_direction(prep.update_prec, prep.prec_pattern_changed);

  ExpectNewtonSystemSolved(ns);
}

// ===================== iterative_refine_dxdy =====================

TEST(SolveNewtonDirection, IterativeRefinementCorrectsArtificiallyInjectedError) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);
  ns.x_cur_ = Vec(3);
  ns.x_cur_ << 5.0, 0.0, 0.0;  // dim 0 outside ux=1 -> nonzero r1_, nontrivial system
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Constant(1, 5.0);
  ns.Bx_ssn_ = Vec::Zero(2);

  auto prep = ns.prepare_newton_system();
  ns.solve_newton_direction(prep.update_prec, prep.prec_pattern_changed);
  ExpectNewtonSystemSolved(ns);  // baseline: already an accurate KKT solution before we corrupt it

  const int s = static_cast<int>(ns.r2_.size());
  ns.dxdy_ += Vec::Constant(ns.dxdy_.size(), 1e-6);  // inject a small artificial error into dxdy_

  auto kkt_residual = [&](const Vec& dxdy) {
    Vec dx = dxdy.head(3), dy = dxdy.tail(s);
    Vec res1 = ns.r1_ + ns.H_diag.cwiseProduct(dx) - ns.G_tr * dy;
    Vec res2 = ns.r2_ - ns.G * dx - dy / ns.mu;
    return std::max(res1.cwiseAbs().maxCoeff(), res2.size() > 0 ? res2.cwiseAbs().maxCoeff() : 0.0);
  };

  const double res_before = kkt_residual(ns.dxdy_);
  ASSERT_GT(res_before, ns.refine_abs_tol);  // corruption is large enough to actually need refining

  // Poison prev_dy_ with a correctly-sized (so the size-only warm_start gate in solve_using_cg()
  // wouldn't reject it) but wrong vector -- a size check alone couldn't catch stale-but-right-size
  // content surviving refinement.
  ns.prev_dy_ = Vec::Constant(s, 999.0);

  ns.iterative_refine_dxdy();

  const double res_after = kkt_residual(ns.dxdy_);
  const double ref_norm = std::max(ns.r1_.cwiseAbs().maxCoeff(), ns.r2_.cwiseAbs().maxCoeff());
  EXPECT_LE(res_after, std::max(ns.refine_rel_tol * ref_norm, ns.refine_abs_tol));
  EXPECT_LT(res_after, res_before);
  // iterative_refine_dxdy() ends by setting prev_dy_ = dxdy_.tail(s) (the corrected, final
  // direction) -- not the poisoned value, and not just the last correction delta.
  EXPECT_TRUE(ns.prev_dy_.isApprox(ns.dxdy_.tail(s)));
}

// ===================== solve_using_ldlt =====================

namespace {

void ExpectKktResidualSmall(const Vec& sol, const SpMat& G, const Vec& H_diag, const Vec& r1,
                             const Vec& r2, double mu, double tol = 1e-10) {
  const int n = static_cast<int>(H_diag.size());
  const int s = static_cast<int>(r2.size());
  ASSERT_EQ(sol.size(), n + s);
  Vec dx = sol.head(n), dy = sol.tail(s);
  Vec res1 = r1 + H_diag.cwiseProduct(dx) - SpMat(G.transpose()) * dy;
  Vec res2 = r2 - G * dx - dy / mu;
  if (res1.size() > 0) EXPECT_LT(res1.cwiseAbs().maxCoeff(), tol);
  if (res2.size() > 0) EXPECT_LT(res2.cwiseAbs().maxCoeff(), tol);
}

}  // namespace

TEST(SolveUsingLdlt, KktLdltLatchSurvivesActiveSetShapeChangeAcrossCalls) {
  // Once PCG has failed anywhere, every later Newton solve for the rest of the run
  // is routed straight to solve_using_ldlt() (even after the active set has changed).
  SsnFixture f(DefaultA(), DefaultB());  // N=3, M=1, l=2
  SSN<double> ns = f.Make();
  ns.mu = 1.0;

  SpMat G1 = DefaultA();  // s=1, as if n_active_W=0
  Vec H1 = Vec::Constant(3, 2.0);
  Vec r1_1 = Vec::Zero(3), r2_1 = Vec::Zero(1);
  Vec sol1 = ns.solve_using_ldlt(G1, H1, r1_1, r2_1);
  ExpectKktResidualSmall(sol1, G1, H1, r1_1, r2_1, ns.mu);

  // Simulate prepare_newton_system() having just detected an active-set change
  // (w_changed -> ldlt_pattern_dirty_/numeric_dirty_ = true) before this call.
  ns.ldlt_pattern_dirty_ = true;
  ns.ldlt_numeric_dirty_ = true;
  SpMat G2(2, 3);  // s=2, as if n_active_W=1
  {
    std::vector<Eigen::Triplet<double>> trips = {{0, 0, 1}, {0, 1, 1}, {0, 2, 1}, {1, 0, 1}};
    G2.setFromTriplets(trips.begin(), trips.end());
    G2.makeCompressed();
  }
  Vec H2 = Vec::Constant(3, 3.0);
  Vec r1_2 = Vec::Zero(3), r2_2 = Vec::Zero(2);
  r1_2(0) = 1.0;
  Vec sol2 = ns.solve_using_ldlt(G2, H2, r1_2, r2_2);
  ExpectKktResidualSmall(sol2, G2, H2, r1_2, r2_2, ns.mu);
}

TEST(SolveUsingLdlt, ForcedReanalyzeWhenSystemSizeChangesButPatternFlagWasNotMarkedDirty) {
  // A second call with a different-shaped G, WITHOUT ldlt_pattern_dirty_ having been (re-)set,
  // must still detect the size mismatch and rebuild.
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  ns.mu = 1.0;

  SpMat G1 = DefaultA();  // s=1, N_tot=4
  Vec H1 = Vec::Constant(3, 2.0);
  Vec sol1 = ns.solve_using_ldlt(G1, H1, Vec::Zero(3), Vec::Zero(1));
  ExpectKktResidualSmall(sol1, G1, H1, Vec::Zero(3), Vec::Zero(1), ns.mu);
  ASSERT_FALSE(ns.ldlt_pattern_dirty_);  // cleared by the first call
  ASSERT_TRUE(ns.K_ldlt_built_);

  SpMat G2(2, 3);  // s=2, N_tot=5
  {
    std::vector<Eigen::Triplet<double>> trips = {{0, 0, 1}, {0, 1, 1}, {0, 2, 1}, {1, 1, 1}};
    G2.setFromTriplets(trips.begin(), trips.end());
    G2.makeCompressed();
  }
  Vec H2 = Vec::Constant(3, 5.0);
  Vec r2_2(2);
  r2_2 << 0.0, 1.0;
  Vec sol2 = ns.solve_using_ldlt(G2, H2, Vec::Zero(3), r2_2);
  ExpectKktResidualSmall(sol2, G2, H2, Vec::Zero(3), r2_2, ns.mu);

  // K_ldlt_built_ ends up true again either way (the forced-reanalyze branch resets it to false,
  // then the full-rebuild branch it triggers sets it back to true within the same call), so the
  // flag alone can't distinguish "really reassembled" from "coincidentally unchanged" -- ground-
  // truth the reassembly itself against an independently-assembled K = [-H2, G2^T; G2, (1/mu)I].
  EXPECT_TRUE(ns.K_ldlt_built_);
  Eigen::MatrixXd expected_K2 = Eigen::MatrixXd::Zero(5, 5);
  for (int i = 0; i < 3; ++i) expected_K2(i, i) = -H2(i);
  for (int i = 0; i < 2; ++i) expected_K2(3 + i, 3 + i) = 1.0 / ns.mu;
  Eigen::MatrixXd G2_dense = Dense(G2);
  expected_K2.block(3, 0, 2, 3) = G2_dense;
  expected_K2.block(0, 3, 3, 2) = G2_dense.transpose();
  EXPECT_TRUE(Dense(ns.K_ldlt_).isApprox(expected_K2));
}

TEST(SolveUsingLdlt, KLdltBuiltFalseInitiallyThenTrueAfterFirstSolve) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  ns.mu = 1.0;
  EXPECT_FALSE(ns.K_ldlt_built_);

  SpMat G1 = DefaultA();
  Vec H1 = Vec::Constant(3, 2.0);
  ns.solve_using_ldlt(G1, H1, Vec::Zero(3), Vec::Zero(1));

  EXPECT_TRUE(ns.K_ldlt_built_);
}

TEST(SolveUsingLdlt, KLdltBuiltStaysTrueAcrossDiagonalOnlyPatch) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  ns.mu = 1.0;

  SpMat G1 = DefaultA();  // s=1
  Vec H1 = Vec::Constant(3, 2.0);
  ns.solve_using_ldlt(G1, H1, Vec::Zero(3), Vec::Zero(1));
  ASSERT_TRUE(ns.K_ldlt_built_);
  ASSERT_FALSE(ns.ldlt_pattern_dirty_);

  // Same G (same pattern): only H_diag values change, as prepare_newton_system() would signal via
  // ldlt_numeric_dirty_ on a mu/rho-only drift -- this must take the diagonal-patch branch, which
  // never touches K_ldlt_built_.
  ns.ldlt_numeric_dirty_ = true;
  Vec H2 = Vec::Constant(3, 5.0);
  Vec sol2 = ns.solve_using_ldlt(G1, H2, Vec::Zero(3), Vec::Zero(1));
  ExpectKktResidualSmall(sol2, G1, H2, Vec::Zero(3), Vec::Zero(1), ns.mu);

  EXPECT_TRUE(ns.K_ldlt_built_);
}

TEST(SolveUsingLdlt, ConsumesFreshHDiagAfterMuRhoOnlyChangeBetweenPrepareCalls) {
  // End-to-end regression test: ldlt_numeric_dirty_ has always fired unconditionally on every
  // update_ssn_system() call ("mu, rho may have changed"), but before the H_diag fix,
  // prepare_newton_system() only rebuilt the H_diag member on k_changed -- so a fresh
  // refactorization could still bake in a stale H_diag (fresh mu in the (1/mu)I block, stale
  // mu/rho in the -H block). This exercises the real call sequence (update_ssn_system ->
  // prepare_newton_system -> solve_using_ldlt(ns.G, ns.H_diag, ...)) to prove H_diag is fresh by
  // the time solve_using_ldlt() consumes it.
  SsnFixture f(DefaultA(), DefaultB());  // N=3, M=1, l=2
  SSN<double> ns = f.Make();

  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, /*mu=*/1.0, /*rho=*/1.0, 0.95, 0);
  ns.x_cur_ = Vec::Zero(3);
  ns.x_cur_(0) = 5.0;  // pushes u_(0) outside ux(0)=1: active_K = [false, true, true]
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Constant(1, 5.0);
  ns.Bx_ssn_ = Vec::Zero(2);
  ns.prepare_newton_system();
  ASSERT_TRUE(ns.H_diag.isApprox((Vec(3) << 2.0, 1.0, 1.0).finished()));

  // New PMM outer iteration: mu/rho change; x_cur_/Ax_ssn_/Bx_ssn_/z/y2 (hence the active set)
  // deliberately left untouched.
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, /*mu=*/4.0, /*rho=*/0.5, 0.95, 1);
  auto result = ns.prepare_newton_system();
  EXPECT_FALSE(result.update_prec);
  EXPECT_FALSE(result.prec_pattern_changed);
  ASSERT_TRUE(ns.H_diag.isApprox((Vec(3) << 6.0, 2.0, 2.0).finished()));

  Vec sol = ns.solve_using_ldlt(ns.G, ns.H_diag, ns.r1_, ns.r2_);
  ExpectKktResidualSmall(sol, ns.G, ns.H_diag, ns.r1_, ns.r2_, ns.mu);
}

// ===================== solve_using_cg =====================

TEST(SolveUsingCg, LdltLatchPermanentlyBypassesPcgAfterAGenuinePreconditionerFailure) {
  // Force a real (not simulated) Cholesky failure inside the Schur preconditioner: with a
  // structurally singular (all-zero) 1x1 G, P = G*E*G^T + (1/mu)*I collapses to just (1/mu)*I;
  // mu < 0 then makes P = -I, which is not positive-definite, so Eigen's LLT must fail.
  SsnFixture f(DefaultA(), DefaultB());  // shape is irrelevant here; only used to construct SSN
  SSN<double> ns = f.Make();
  ns.mu = -1.0;

  SpMat G_singular(1, 1);  // structurally singular: no stored nonzero entries at all
  SpMat G_tr_singular = SpMat(G_singular.transpose());
  Vec H_diag(1);
  H_diag << 1.0;  // kept strictly positive: factorize_by_chol() asserts H_diag.minCoeff() > 0
  BoolArr active_K(1);
  active_K << true;
  Vec r1(1), r2(1);
  r1 << 0.0;
  r2 << 1.0;

  ASSERT_FALSE(ns.kkt_ldlt_used);
  Vec sol1 = ns.solve_using_cg(G_singular, G_tr_singular, H_diag, H_diag.cwiseInverse(), active_K,
                                r1, r2, ns.mu, ns.krylov_tol, ns.krylov_max_in_iter,
                                /*update_prec=*/true, /*prec_pattern_changed=*/true,
                                /*schur_use_ldlt=*/false);

  // The Cholesky preconditioner genuinely failed, so the permanent LDLT latch must now be engaged,
  // and this first call's own result must have gone through solve_using_ldlt().
  EXPECT_TRUE(ns.kkt_ldlt_used);
  EXPECT_GT(ns.krylov_fail, 0);
  EXPECT_EQ(ns.prev_dy_.size(), 0);  // switch_to_ldlt() releases the now-unused CG warm-start cache
  ASSERT_EQ(sol1.size(), 2);
  {
    Vec dx = sol1.head(1), dy = sol1.tail(1);
    Vec res1 = r1 + H_diag.cwiseProduct(dx) - G_tr_singular * dy;
    Vec res2 = r2 - G_singular * dx - dy / ns.mu;
    EXPECT_LT(res1.cwiseAbs().maxCoeff(), 1e-8);
    EXPECT_LT(res2.cwiseAbs().maxCoeff(), 1e-8);
  }

  const int krylov_fail_after_first_call = ns.krylov_fail;
  const int krylov_iter_after_first_call = ns.krylov_iter;
  const int fact_after_first_call = ns.fact;

  // A second, perfectly well-posed system (positive mu, well-conditioned G) that PCG could easily
  // solve -- if the latch is honored, solve_using_cg() must route straight to solve_using_ldlt()
  // without ever touching the preconditioner/CG machinery, so krylov_fail/krylov_iter must not move.
  ns.mu = 1.0;
  SpMat G2 = DefaultA();  // 1x3, well-conditioned equality row
  SpMat G2_tr = SpMat(G2.transpose());
  Vec H2 = Vec::Constant(3, 2.0);
  Vec r1_2 = Vec::Zero(3), r2_2 = Vec::Zero(1);
  r1_2(0) = 1.0;
  BoolArr active_K2 = BoolArr::Constant(3, true);

  Vec sol2 = ns.solve_using_cg(G2, G2_tr, H2, H2.cwiseInverse(), active_K2, r1_2, r2_2,
                                /*mu=*/1.0, ns.krylov_tol, ns.krylov_max_in_iter,
                                /*update_prec=*/true, /*prec_pattern_changed=*/true,
                                /*schur_use_ldlt=*/false);

  EXPECT_EQ(ns.krylov_fail, krylov_fail_after_first_call);  // PCG was never attempted
  EXPECT_EQ(ns.krylov_iter, krylov_iter_after_first_call);  // no CG iterations were recorded
  EXPECT_GT(ns.fact, fact_after_first_call);                // work still happened, via LDLT

  ASSERT_EQ(sol2.size(), 4);
  Vec dx2 = sol2.head(3), dy2 = sol2.tail(1);
  Vec res1_2 = r1_2 + H2.cwiseProduct(dx2) - G2_tr * dy2;
  Vec res2_2 = r2_2 - G2 * dx2 - dy2 / ns.mu;
  EXPECT_LT(res1_2.cwiseAbs().maxCoeff(), 1e-8);
  EXPECT_LT(res2_2.cwiseAbs().maxCoeff(), 1e-8);
}

// ===================== cross-path agreement =====================
// For a fixed (G, H_diag, active_K, r1, r2, mu, rho), PCG+Cholesky, PCG+LDLT, PCG+SMW (updated from
// either a Cholesky or an LDLT snapshot), and solve_using_ldlt() all solve the same augmented KKT
// system K[dx;dy]=[r1;r2] -- the preconditioner only affects PCG's convergence path, and SMW is
// just a cheaper way to reach the same preconditioning matrix as a fresh factorization. This checks
// all five routes agree with each other and each independently satisfies the KKT residual.

namespace {

void ExpectAllSolvePathsAgree(const SpMat& A, const SpMat& B, const BoolArr& active_W,
                               const BoolArr& active_K, const BoolArr& active_K_snapshot,
                               const Vec& H_diag, double mu, double rho, const SpMat& G,
                               const SpMat& G_tr, const Vec& r1, const Vec& r2,
                               double tol = 1e-8) {
  SsnFixture f(A, B);
  const Vec H_diag_inv = H_diag.cwiseInverse();

  // SSN<double> is neither copyable (cg's ConjugateGradient member isn't) nor safely returnable
  // by value from a named local (no viable move ctor either), so set up each instance in place via
  // a by-reference helper rather than a factory function that returns SSN<double>.
  auto setup_ns = [&](SSN<double>& ns) {
    ns.mu = mu;
    ns.rho = rho;
    ns.active_W = active_W;
  };

  SSN<double> ns_chol = f.Make();
  setup_ns(ns_chol);
  const Vec sol_chol = ns_chol.solve_using_cg(
      G, G_tr, H_diag, H_diag_inv, active_K, r1, r2, mu, ns_chol.krylov_tol,
      ns_chol.krylov_max_in_iter, /*update_prec=*/true, /*prec_pattern_changed=*/true,
      /*schur_use_ldlt=*/false);

  SSN<double> ns_ldlt = f.Make();
  setup_ns(ns_ldlt);
  const Vec sol_ldlt = ns_ldlt.solve_using_cg(
      G, G_tr, H_diag, H_diag_inv, active_K, r1, r2, mu, ns_ldlt.krylov_tol,
      ns_ldlt.krylov_max_in_iter, /*update_prec=*/true, /*prec_pattern_changed=*/true,
      /*schur_use_ldlt=*/true);

  // SMW-from-Cholesky: first solve builds a full-Cholesky snapshot at a neighboring active_K
  // (one entry flipped); second solve, same G/active_W/mu/rho, should reuse that snapshot via a
  // rank-1 SMW update instead of refactorizing.
  SSN<double> ns_smw_from_chol = f.Make();
  setup_ns(ns_smw_from_chol);
  ns_smw_from_chol.solve_using_cg(G, G_tr, H_diag, H_diag_inv, active_K_snapshot, r1, r2, mu,
                                   ns_smw_from_chol.krylov_tol,
                                   ns_smw_from_chol.krylov_max_in_iter, /*update_prec=*/true,
                                   /*prec_pattern_changed=*/true, /*schur_use_ldlt=*/false);
  const Vec sol_smw_chol = ns_smw_from_chol.solve_using_cg(
      G, G_tr, H_diag, H_diag_inv, active_K, r1, r2, mu, ns_smw_from_chol.krylov_tol,
      ns_smw_from_chol.krylov_max_in_iter, /*update_prec=*/true, /*prec_pattern_changed=*/false,
      /*schur_use_ldlt=*/false);
  ASSERT_TRUE(ns_smw_from_chol.cg.preconditioner().used_smw())
      << "SMW-from-Cholesky path did not actually engage SMW; the agreement check below would be "
         "vacuous.";

  // Same, updating from an LDLT snapshot instead.
  SSN<double> ns_smw_from_ldlt = f.Make();
  setup_ns(ns_smw_from_ldlt);
  ns_smw_from_ldlt.solve_using_cg(G, G_tr, H_diag, H_diag_inv, active_K_snapshot, r1, r2, mu,
                                   ns_smw_from_ldlt.krylov_tol,
                                   ns_smw_from_ldlt.krylov_max_in_iter, /*update_prec=*/true,
                                   /*prec_pattern_changed=*/true, /*schur_use_ldlt=*/true);
  const Vec sol_smw_ldlt = ns_smw_from_ldlt.solve_using_cg(
      G, G_tr, H_diag, H_diag_inv, active_K, r1, r2, mu, ns_smw_from_ldlt.krylov_tol,
      ns_smw_from_ldlt.krylov_max_in_iter, /*update_prec=*/true, /*prec_pattern_changed=*/false,
      /*schur_use_ldlt=*/true);
  ASSERT_TRUE(ns_smw_from_ldlt.cg.preconditioner().used_smw())
      << "SMW-from-LDLT path did not actually engage SMW; the agreement check below would be "
         "vacuous.";

  // Direct factorization of the full augmented KKT system -- no PCG/preconditioner at all.
  const Vec sol_direct = ns_chol.solve_using_ldlt(G, H_diag, r1, r2);

  EXPECT_TRUE(sol_chol.isApprox(sol_direct, tol))
      << "PCG+Cholesky diverged from solve_using_ldlt";
  EXPECT_TRUE(sol_ldlt.isApprox(sol_direct, tol))
      << "PCG+LDLT diverged from solve_using_ldlt";
  EXPECT_TRUE(sol_smw_chol.isApprox(sol_direct, tol))
      << "PCG+SMW(from Cholesky) diverged from solve_using_ldlt";
  EXPECT_TRUE(sol_smw_ldlt.isApprox(sol_direct, tol))
      << "PCG+SMW(from LDLT) diverged from solve_using_ldlt";

  // Ground-truth each result independently too, so a bug shared by all paths (which would still
  // leave them agreeing with each other) doesn't slip through.
  ExpectKktResidualSmall(sol_chol, G, H_diag, r1, r2, mu);
  ExpectKktResidualSmall(sol_ldlt, G, H_diag, r1, r2, mu);
  ExpectKktResidualSmall(sol_smw_chol, G, H_diag, r1, r2, mu);
  ExpectKktResidualSmall(sol_smw_ldlt, G, H_diag, r1, r2, mu);
  ExpectKktResidualSmall(sol_direct, G, H_diag, r1, r2, mu);
}

}  // namespace

TEST(AllSolvePathsAgree, AllWRowsActiveSingleKFlip) {
  const SpMat A = DefaultA();
  const SpMat B = DefaultB();
  const BoolArr active_W = ToBoolArr({true, true});
  const BoolArr active_K = ToBoolArr({true, false, true});           // index 1 inactive
  const BoolArr active_K_snapshot = ToBoolArr({true, true, true});  // single flip vs. active_K

  Vec H_diag(3);
  H_diag << 2.0, 3.0, 4.0;
  const double mu = 5.0, rho = 3.0;  // mu != rho catches an argument-order swap bug

  // G = [A_row; B_row0; B_row1] (both W rows active) -> s=3.
  const Eigen::MatrixXd G_dense =
      (Eigen::MatrixXd(3, 3) << 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0).finished();
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());

  Vec r1(3), r2(3);
  r1 << 1.0, -2.0, 0.5;
  r2 << 0.5, -1.0, 2.0;

  ExpectAllSolvePathsAgree(A, B, active_W, active_K, active_K_snapshot, H_diag, mu, rho, G, G_tr,
                            r1, r2);
}

TEST(AllSolvePathsAgree, PartialWRowsActiveSingleKFlip) {
  const SpMat A = DefaultA();
  const SpMat B = DefaultB();
  const BoolArr active_W = ToBoolArr({true, false});
  const BoolArr active_K = ToBoolArr({false, true, true});           // index 0 inactive
  const BoolArr active_K_snapshot = ToBoolArr({true, true, true});  // single flip vs. active_K

  Vec H_diag(3);
  H_diag << 3.0, 1.0, 2.0;
  const double mu = 2.0, rho = 4.0;

  // G = [A_row; B_row0] (only the first W row active) -> s=2.
  const Eigen::MatrixXd G_dense =
      (Eigen::MatrixXd(2, 3) << 1.0, 1.0, 1.0, 1.0, 0.0, 0.0).finished();
  const SpMat G = DenseToSparse(G_dense);
  const SpMat G_tr = DenseToSparse(G_dense.transpose());

  Vec r1(3), r2(2);
  r1 << -1.0, 2.0, 0.5;
  r2 << 1.5, -0.5;

  ExpectAllSolvePathsAgree(A, B, active_W, active_K, active_K_snapshot, H_diag, mu, rho, G, G_tr,
                            r1, r2);
}

// ===================== make_line_search_params =====================

TEST(MakeLineSearchParams, CachesATrTimesAxMinusBMatchingIndependentRecomputation) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);
  ns.x_cur_ = Vec::Zero(3);
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Constant(1, 2.0);  // != b (b=0), makes grad_res_p_ nonzero
  ns.Bx_ssn_ = Vec::Zero(2);

  ns.make_line_search_params();

  EXPECT_TRUE(ns.grad_Atr_resp_.isApprox(Dense(ns.A_tr) * (ns.Ax_ssn_ - ns.b)));
}

// ===================== line_search_with_steepest_descent_fallback =====================

TEST(LineSearchWithSteepestDescentFallback, ProceedOutcomeOnFirstAttemptMatchesIndependentExactLineSearchCall) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x_pmm(3);
  x_pmm << 1.0, 1.0, 1.0;
  Vec y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x_pmm, y10, y20, z0, dy10, dz0, /*mu=*/1.0, /*rho=*/1.0, /*alpha=*/0.95, 0);

  ns.x_cur_ = Vec::Zero(3);
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Zero(1);
  ns.Bx_ssn_ = Vec::Zero(2);
  ns.dist_K_u_ = Vec::Zero(3);
  ns.dist_W_v_ = Vec::Zero(2);
  ns.dx_ = Vec(3);
  ns.dx_ << 1.0, 1.0, 1.0;
  ns.dy2_ = Vec::Zero(2);

  auto ls = ns.line_search_with_steepest_descent_fallback(/*ssn_tol=*/1e-6);

  ASSERT_EQ(ls.outcome, SSN<double>::LineSearchOutcome::Proceed);
  // Hand-derived: zeta = dx.dot((x_cur_-x)/rho) = [1,1,1].[-1,-1,-1] = -3;
  // breakpoint at t=1 (upper bound crossing, slope_change=+3); 
  // eta = mu*(Adx)^2 + (1/rho)*||dx||^2 = 1*9 + 1*3 = 12;
  // p_t(1) = -3+12=9>=0 -> tau = 0 - (-3)/12 = 0.25.
  EXPECT_NEAR(ls.tau, 0.25, 1e-9);

  // Cross-check against an independent call to the separately-tested free exact_line_search().
  auto params = ns.make_line_search_params();
  Vec ls_s(3), ls_v(2), ls_dv(2);
  std::vector<SsnBreakpoint<double>> bps;
  double tau_expected = exact_line_search(params, ns.x_cur_, ns.y2_cur_, ns.dx_, ns.dy2_,
                                           ns.Ax_ssn_, ns.Bx_ssn_, ns.Adx_, ns.Bdx_, ns.dist_K_u_,
                                           ns.dist_W_v_, ls_s, ls_v, ls_dv, bps);
  EXPECT_DOUBLE_EQ(ls.tau, tau_expected);
}

TEST(LineSearchWithSteepestDescentFallback, ZeroNewtonDirectionReturnsFullStepAsProceedDegenerateCase) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);

  ns.x_cur_ = Vec::Zero(3);
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Zero(1);
  ns.Bx_ssn_ = Vec::Zero(2);
  ns.dist_K_u_ = Vec::Zero(3);
  ns.dist_W_v_ = Vec::Zero(2);
  ns.dx_ = Vec::Zero(3);   // exactly zero: below eps_direction for every component
  ns.dy2_ = Vec::Zero(2);

  auto ls = ns.line_search_with_steepest_descent_fallback(/*ssn_tol=*/1e-6);

  ASSERT_EQ(ls.outcome, SSN<double>::LineSearchOutcome::Proceed);
  EXPECT_DOUBLE_EQ(ls.tau, 1.0);
}

TEST(LineSearchWithSteepestDescentFallback, RetryWithSteepestDescentSucceedsAfterFirstAttemptFails) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x_pmm(3);
  x_pmm << 1.0, 1.0, 1.0;
  Vec y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x_pmm, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);

  ns.x_cur_ = Vec::Zero(3);
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Zero(1);
  ns.Bx_ssn_ = Vec::Zero(2);
  ns.dist_K_u_ = Vec::Zero(3);
  ns.dist_W_v_ = Vec::Zero(2);
  ns.dx_ = Vec(3);
  ns.dx_ << -0.5, -0.5, -0.5;  // ascent direction: zeta1 = dx.dot([-1,-1,-1]) = 1.5 >= 0 -> tau=0
  ns.dy2_ = Vec::Zero(2);

  // grad_Atr_resp_ is cached once per SSN iteration and read again during the retry's
  // compute_grad_Lagrangian() call inside line_search_with_steepest_descent_fallback() below --
  // capture it up front the same way that internal call does, to check it stays bit-identical
  // across the retry (true because Ax_ssn_ provably doesn't move between them).
  ns.make_line_search_params();
  const Vec grad_Atr_resp_before = ns.grad_Atr_resp_;

  auto ls = ns.line_search_with_steepest_descent_fallback(/*ssn_tol=*/1e-6);

  // grad_L at x_cur_=[0,0,0] (with x=[1,1,1], c=0, y1=0, rho=1) is [-1,-1,-1,0,0], so
  // grad_norm=1 > 5*1e-6 -> AcceptOptimal is rejected -> retries with dx=-grad_L=[1,1,1],
  // which (per the previous test) succeeds with tau=0.25.
  ASSERT_EQ(ls.outcome, SSN<double>::LineSearchOutcome::Proceed);
  EXPECT_NEAR(ls.tau, 0.25, 1e-9);
  Vec expected_dx(3);
  expected_dx << 1.0, 1.0, 1.0;
  EXPECT_TRUE(ns.dx_.isApprox(expected_dx));
  EXPECT_TRUE(ns.dy2_.isApprox(Vec::Zero(2)));
  EXPECT_EQ(ns.linesearch_fail, 0);  // retry succeeded, so no failure recorded
  EXPECT_TRUE(ns.grad_Atr_resp_ == grad_Atr_resp_before);
}

TEST(LineSearchWithSteepestDescentFallback, AcceptsOptimalWhenFirstAttemptFailsButGradientIsSmall) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x_pmm(3);
  x_pmm << 1.0, 1.0, 1.0;
  Vec y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x_pmm, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);

  ns.x_cur_ = Vec::Zero(3);
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Zero(1);
  ns.Bx_ssn_ = Vec::Zero(2);
  ns.dist_K_u_ = Vec::Zero(3);
  ns.dist_W_v_ = Vec::Zero(2);
  ns.dx_ = Vec(3);
  ns.dx_ << -0.5, -0.5, -0.5;
  ns.dy2_ = Vec::Zero(2);

  // grad_norm=1 (see previous test); ssn_tol=1.0 makes 5*ssn_tol=5 >= 1 -> accept without a step.
  auto ls = ns.line_search_with_steepest_descent_fallback(/*ssn_tol=*/1.0);

  EXPECT_EQ(ls.outcome, SSN<double>::LineSearchOutcome::AcceptOptimal);
}

TEST(LineSearchWithSteepestDescentFallback, FailsAfterBothAttemptsWhenBothDirectionsAreNonDescent) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x_pmm(3);
  x_pmm << 1.0, 1.0, 1.0;
  Vec y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x_pmm, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);

  ns.x_cur_ = Vec::Zero(3);
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Zero(1);
  ns.Bx_ssn_ = Vec::Zero(2);
  // dist_K_u_ chosen so that mu*dist_K_u_.dot(dx) pushes psi'(0) >= 0 for BOTH the first attempt's dx
  // and the retry's dx=-grad_L=[1,1,1]: with dist_K_u_=[1,1,1], p_val1 = 1.5 + 1*(-1.5) = 0 >= 0,
  // and p_val2 = -3 + 1*3 = 0 >= 0.
  ns.dist_K_u_ = Vec(3);
  ns.dist_K_u_ << 1.0, 1.0, 1.0;
  ns.dist_W_v_ = Vec::Zero(2);
  ns.dx_ = Vec(3);
  ns.dx_ << -0.5, -0.5, -0.5;
  ns.dy2_ = Vec::Zero(2);

  auto ls = ns.line_search_with_steepest_descent_fallback(/*ssn_tol=*/1e-6);

  EXPECT_EQ(ls.outcome, SSN<double>::LineSearchOutcome::Fail);
  EXPECT_DOUBLE_EQ(ls.tau, 0.0);
  EXPECT_EQ(ns.linesearch_fail, 1);
}

TEST(LineSearchWithSteepestDescentFallback, FallbackDirectionExactlyEqualsNegativeGradientAndRetrySucceeds) {
  // A heavily corrupted "Newton" direction -- huge magnitude, pointed the wrong way -- forces the
  // first linesearch attempt to reject it (psi'(0) >= 0), triggering the steepest-descent fallback.
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x_pmm(3);
  x_pmm << 1.0, 1.0, 1.0;
  Vec y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x_pmm, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);

  ns.x_cur_ = Vec::Zero(3);
  ns.y2_cur_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec::Zero(1);
  ns.Bx_ssn_ = Vec::Zero(2);
  ns.dist_K_u_ = Vec::Zero(3);
  ns.dist_W_v_ = Vec::Zero(2);
  ns.dx_ = Vec::Constant(3, -1e6);  // grossly corrupted: an ascent direction of huge magnitude
  ns.dy2_ = Vec::Zero(2);

  auto ls = ns.line_search_with_steepest_descent_fallback(/*ssn_tol=*/1e-6);

  // grad_norm=1 at this state (see the RetryWithSteepestDescentSucceeds... test above) is well
  // above 5*ssn_tol=5e-6, so AcceptOptimal is rejected and the fallback direction is exercised.
  ASSERT_EQ(ls.outcome, SSN<double>::LineSearchOutcome::Proceed);
  EXPECT_GT(ls.tau, 0.0);
  EXPECT_EQ(ns.linesearch_fail, 0);

  // The corrupted dx_/dy2_ were discarded and replaced by -grad_L; verify that exactly, against an
  // independently recomputed gradient (state is otherwise unchanged by the linesearch call itself).
  const Vec& grad_L = ns.compute_grad_Lagrangian(ns.x_cur_, ns.y2_cur_, ns.Ax_ssn_, ns.Bx_ssn_);
  EXPECT_TRUE(ns.dx_.isApprox(Vec(-grad_L.head(3))));
  EXPECT_TRUE(ns.dy2_.isApprox(Vec(-grad_L.tail(2))));
}

// ===================== update_iterate =====================

TEST(UpdateIterate, AppliesTauScaledStepUsingIncrementalAxBxUpdateOnNonMultipleOfFiveIteration) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);

  ns.x_cur_ = Vec(3);
  ns.x_cur_ << 1.0, 2.0, 3.0;
  ns.y2_cur_ = Vec(2);
  ns.y2_cur_ << 0.5, -0.5;
  ns.dx_ = Vec(3);
  ns.dx_ << 0.1, 0.2, 0.3;
  ns.dy2_ = Vec(2);
  ns.dy2_ << 0.01, -0.02;
  ns.Ax_ssn_ = Vec(1);
  ns.Ax_ssn_ << 6.0;  // = A * x_cur_ = 1+2+3
  ns.Bx_ssn_ = Vec(2);
  ns.Bx_ssn_ << 1.0, 2.0;  // = B * x_cur_ (rows e1, e2)
  ns.Adx_ = Vec(1);
  ns.Adx_ << 0.6;  // = A * dx_
  ns.Bdx_ = Vec(2);
  ns.Bdx_ << 0.1, 0.2;  // = B * dx_

  ns.update_iterate(/*tau=*/0.5, /*ssn_iter_count=*/0);  // 0 % 5 != 4 -> incremental branch

  Vec expected_x(3);
  expected_x << 1.05, 2.1, 3.15;
  EXPECT_TRUE(ns.x_cur_.isApprox(expected_x));
  Vec expected_y2(2);
  expected_y2 << 0.505, -0.51;
  EXPECT_TRUE(ns.y2_cur_.isApprox(expected_y2));
  Vec expected_Ax(1);
  expected_Ax << 6.3;  // 6.0 + 0.5*0.6, via the incremental path
  EXPECT_TRUE(ns.Ax_ssn_.isApprox(expected_Ax));
  Vec expected_Bx(2);
  expected_Bx << 1.05, 2.1;
  EXPECT_TRUE(ns.Bx_ssn_.isApprox(expected_Bx));
}

TEST(UpdateIterate, RecomputesAxBxFromScratchIgnoringStaleAdxBdxEveryFifthIteration) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);

  ns.x_cur_ = Vec(3);
  ns.x_cur_ << 1.0, 2.0, 3.0;
  ns.y2_cur_ = Vec::Zero(2);
  ns.dx_ = Vec(3);
  ns.dx_ << 0.1, 0.2, 0.3;
  ns.dy2_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec(1);
  ns.Ax_ssn_ << 6.0;
  ns.Bx_ssn_ = Vec(2);
  ns.Bx_ssn_ << 1.0, 2.0;
  // Deliberately inconsistent with A*dx_/B*dx_, to prove the recompute branch ignores them.
  ns.Adx_ = Vec(1);
  ns.Adx_ << 999.0;
  ns.Bdx_ = Vec(2);
  ns.Bdx_ << 999.0, 999.0;

  ns.update_iterate(/*tau=*/0.5, /*ssn_iter_count=*/4);  // 4 % 5 == 4 -> full recompute branch

  Vec expected_x(3);
  expected_x << 1.05, 2.1, 3.15;
  EXPECT_TRUE(ns.x_cur_.isApprox(expected_x));
  // Ax_ssn_/Bx_ssn_ must equal A*x_cur_new / B*x_cur_new.
  Vec expected_Ax(1);
  expected_Ax << 6.3;  // 1.05+2.1+3.15
  EXPECT_TRUE(ns.Ax_ssn_.isApprox(expected_Ax));
  Vec expected_Bx(2);
  expected_Bx << 1.05, 2.1;
  EXPECT_TRUE(ns.Bx_ssn_.isApprox(expected_Bx));
}

TEST(UpdateIterate, TauZeroLeavesIteratesNumericallyUnchanged) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);

  Vec x_before(3);
  x_before << 1.0, 2.0, 3.0;
  ns.x_cur_ = x_before;
  Vec y2_before(2);
  y2_before << 0.5, -0.5;
  ns.y2_cur_ = y2_before;
  ns.dx_ = Vec(3);
  ns.dx_ << 0.1, 0.2, 0.3;
  ns.dy2_ = Vec(2);
  ns.dy2_ << 0.01, -0.02;
  ns.Ax_ssn_ = Vec(1);
  ns.Ax_ssn_ << 6.0;
  ns.Bx_ssn_ = Vec(2);
  ns.Bx_ssn_ << 1.0, 2.0;
  ns.Adx_ = Vec(1);
  ns.Adx_ << 0.6;
  ns.Bdx_ = Vec(2);
  ns.Bdx_ << 0.1, 0.2;

  ns.update_iterate(/*tau=*/0.0, /*ssn_iter_count=*/0);

  EXPECT_TRUE(ns.x_cur_.isApprox(x_before));
  EXPECT_TRUE(ns.y2_cur_.isApprox(y2_before));
  Vec expected_Ax(1);
  expected_Ax << 6.0;
  EXPECT_TRUE(ns.Ax_ssn_.isApprox(expected_Ax));
}

TEST(UpdateIterate, StoresToleranceConsistentWithRecomputingGradientAtTheUpdatedIterate) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  Vec x0 = Vec::Zero(3), y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);

  ns.x_cur_ = Vec(3);
  ns.x_cur_ << 1.0, 2.0, 3.0;
  ns.y2_cur_ = Vec::Zero(2);
  ns.dx_ = Vec(3);
  ns.dx_ << 0.1, 0.2, 0.3;
  ns.dy2_ = Vec::Zero(2);
  ns.Ax_ssn_ = Vec(1);
  ns.Ax_ssn_ << 6.0;
  ns.Bx_ssn_ = Vec(2);
  ns.Bx_ssn_ << 1.0, 2.0;
  ns.Adx_ = Vec(1);
  ns.Adx_ << 0.6;
  ns.Bdx_ = Vec(2);
  ns.Bdx_ << 0.1, 0.2;

  ns.update_iterate(/*tau=*/0.5, /*ssn_iter_count=*/0);

  const Vec& grad_L = ns.compute_grad_Lagrangian(ns.x_cur_, ns.y2_cur_, ns.Ax_ssn_, ns.Bx_ssn_);
  const double expected_tol = ns.compute_grad_Lagrangian_unscaled_inf_norm(grad_L);
  EXPECT_DOUBLE_EQ(ns.tol_achieved, expected_tol);
}

// ===================== check_ssn_termination =====================

TEST(CheckSsnTermination, ReturnsOptimalWhenBelowTolerance) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  ns.tol_achieved = 1e-8;
  int stagnation = 3;
  double prev_tol_achieved = 1.0;

  const auto result = ns.check_ssn_termination(/*ssn_tol=*/1e-6, stagnation, prev_tol_achieved);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, SSN<double>::TerminationStatus::Optimal);
  EXPECT_EQ(stagnation, 3);              // untouched: returns before the stagnation logic
  EXPECT_DOUBLE_EQ(prev_tol_achieved, 1.0);
}

TEST(CheckSsnTermination, ReturnsNulloptAndResetsStagnationWhenImproving) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  ns.tol_achieved = 0.5;  // < 0.999 * 1.0
  int stagnation = 3;
  double prev_tol_achieved = 1.0;

  const auto result = ns.check_ssn_termination(/*ssn_tol=*/1e-9, stagnation, prev_tol_achieved);

  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(stagnation, 0);
  EXPECT_DOUBLE_EQ(prev_tol_achieved, 0.5);
}

TEST(CheckSsnTermination, IncrementsStagnationWhenNotImproving) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  ns.tol_achieved = 0.9995;  // >= 0.999 * 1.0
  int stagnation = 0;
  double prev_tol_achieved = 1.0;

  const auto result = ns.check_ssn_termination(/*ssn_tol=*/1e-9, stagnation, prev_tol_achieved);

  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(stagnation, 1);
  EXPECT_DOUBLE_EQ(prev_tol_achieved, 0.9995);
}

TEST(CheckSsnTermination, BoundaryAtExactlyPointNineNineNineRatioCountsAsNotImproving) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  ns.tol_achieved = 0.999;  // exactly 0.999 * 1.0 -- the ">=" comparison counts this as stagnant
  int stagnation = 0;
  double prev_tol_achieved = 1.0;

  const auto result = ns.check_ssn_termination(/*ssn_tol=*/1e-9, stagnation, prev_tol_achieved);

  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(stagnation, 1);
}

TEST(CheckSsnTermination, FirstCallWithInfinitePrevToleranceNeverCountsAsStagnant) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  ns.tol_achieved = 1e6;  // huge but finite: still < 0.999*inf
  int stagnation = 0;
  double prev_tol_achieved = ns.inf;

  const auto result = ns.check_ssn_termination(/*ssn_tol=*/1e-9, stagnation, prev_tol_achieved);

  EXPECT_FALSE(result.has_value());
  EXPECT_EQ(stagnation, 0);
  EXPECT_DOUBLE_EQ(prev_tol_achieved, 1e6);
}

TEST(CheckSsnTermination, AcceptsOptimalAfterTenStagnantIterationsIfCloseEnough) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  int stagnation = 0;
  double prev_tol_achieved = 1.0;
  const double ssn_tol = 1.0;  // 5*ssn_tol = 5, well above tol_achieved=1.0

  std::optional<SSN<double>::TerminationStatus> result;
  for (int i = 0; i < 10; ++i) {
    ns.tol_achieved = 1.0;  // never improves (>= 0.999*prev every time)
    result = ns.check_ssn_termination(ssn_tol, stagnation, prev_tol_achieved);
    if (i < 9) ASSERT_FALSE(result.has_value()) << "iteration " << i;
  }

  EXPECT_EQ(stagnation, 10);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, SSN<double>::TerminationStatus::Optimal);  // tol_achieved(1.0) < 5*ssn_tol(5.0) -> accepted as optimal
}

TEST(CheckSsnTermination, ReturnsStagnatedAfterTenStagnantIterationsIfNotCloseEnough) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  int stagnation = 0;
  double prev_tol_achieved = 1.0;
  const double ssn_tol = 1e-9;  // 5*ssn_tol negligible compared to tol_achieved=1.0

  std::optional<SSN<double>::TerminationStatus> result;
  for (int i = 0; i < 10; ++i) {
    ns.tol_achieved = 1.0;
    result = ns.check_ssn_termination(ssn_tol, stagnation, prev_tol_achieved);
    if (i < 9) ASSERT_FALSE(result.has_value()) << "iteration " << i;
  }

  EXPECT_EQ(stagnation, 10);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, SSN<double>::TerminationStatus::Stagnated);  // stagnated but not close enough to ssn_tol -> not a confirmed optimum
}

TEST(CheckSsnTermination, ZeroToleranceAchievedImmediatelyOptimal) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  ns.tol_achieved = 0.0;
  int stagnation = 5;
  double prev_tol_achieved = 1.0;

  const auto result = ns.check_ssn_termination(/*ssn_tol=*/1e-6, stagnation, prev_tol_achieved);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, SSN<double>::TerminationStatus::Optimal);
}

// ===================== solve_ssn (end-to-end) =====================

TEST(SolveSsn, MaxInnerIterationsZeroTerminatesImmediatelyWithoutUpdatingIterates) {
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns(f.Q_info, f.Q_diag, f.L, f.A, f.B, f.A_tr, f.B_tr, f.c, f.b, f.D2_ext_inv,
                  f.D1B_diag_inv, f.lx, f.ux, f.lw, f.uw, f.n, f.m, f.N, f.M, f.l,
                  /*ssn_tol=*/1e-6, /*ssn_max_in_iter=*/0,
                  /*eps_pinf=*/1e-6, /*eps_dinf=*/1e-6);

  Vec x0(3);
  x0 << 0.3, 0.3, 0.3;
  Vec y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);

  ns.solve_ssn(/*ssn_tol=*/1e-6);

  EXPECT_EQ(ns.iter, 0);
  EXPECT_EQ(ns.opt, SSN<double>::TerminationStatus::MaxInnerIterations);
  EXPECT_TRUE(ns.x.isApprox(x0));   // loop body never ran: x_cur_ was never updated
  EXPECT_TRUE(ns.y2.isApprox(y20));
}

TEST(SolveSsn, TerminatesImmediatelyWithInterruptedStatusWhenInterruptedFlagIsSet) {
  // interrupted_() is checked as the very first statement of the SSN loop body, so an
  // unconditionally-true flag stops the loop before any iteration runs -- same shape as the
  // ssn_max_in_iter=0 case above, but via the interruption mechanism instead of the iteration cap.
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  ns.interrupted_ = [] { return true; };

  Vec x0(3);
  x0 << 0.3, 0.3, 0.3;
  Vec y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);

  ns.solve_ssn(/*ssn_tol=*/1e-6);

  EXPECT_EQ(ns.iter, 0);
  EXPECT_EQ(ns.opt, SSN<double>::TerminationStatus::Interrupted);
  EXPECT_TRUE(ns.x.isApprox(x0));   // loop body never ran: x_cur_ was never updated
  EXPECT_TRUE(ns.y2.isApprox(y20));
}

TEST(SolveSsn, TerminatesImmediatelyWithTimeLimitStatusWhenTimeLimitExceededFlagIsSet) {
  // Same shape as the interrupted_ test above, but for time_limit_exceeded_: checked as the
  // second statement of the SSN loop body (after interrupted_), so an unconditionally-true flag
  // stops the loop before any iteration runs.
  SsnFixture f(DefaultA(), DefaultB());
  SSN<double> ns = f.Make();
  ns.time_limit_exceeded_ = [] { return true; };

  Vec x0(3);
  x0 << 0.3, 0.3, 0.3;
  Vec y10 = Vec::Zero(1), y20 = Vec::Zero(2), z0 = Vec::Zero(3);
  Vec dy10 = Vec::Zero(1), dz0 = Vec::Zero(3);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, 1.0, 1.0, 0.95, 0);

  ns.solve_ssn(/*ssn_tol=*/1e-6);

  EXPECT_EQ(ns.iter, 0);
  EXPECT_EQ(ns.opt, SSN<double>::TerminationStatus::TimeLimit);
  EXPECT_TRUE(ns.x.isApprox(x0));   // loop body never ran: x_cur_ was never updated
  EXPECT_TRUE(ns.y2.isApprox(y20));
}

TEST(SolveSsn, ConvergesToAnalyticMinimizerWithNoEqualityOrInequalityConstraints) {
  // N=1, M=0, l=0: pure box-constrained scalar proximal subproblem
  //   min_x  c*x + (mu/2) dist_K(z/mu+x)^2 + (1/(2 rho)) (x-x_bar)^2,
  // with c=1, z=0, mu=rho=1, x_bar=0, box=[-1,1].
  // On [-1,1] (interior formula) this is x + x^2/2, minimized at x=-1 (exactly the boundary). 
  // Checking the outside formula too ((x+1) + (x+1)^2/2 + x^2/2, derivative 2x+2) also gives 0 at x=-1,
  // so x*=-1 is the unique minimizer (both one-sided derivatives agree).
  SpMat A(0, 1), B(0, 1);  // M = 0, l = 0
  SsnFixture f(A, B);
  f.c(0) = 1.0;
  SSN<double> ns = f.Make();

  Vec x0(1);
  x0 << 0.0;
  Vec y10 = Vec::Zero(0), y20 = Vec::Zero(0), z0 = Vec::Zero(1);
  Vec dy10 = Vec::Zero(0), dz0 = Vec::Zero(1);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, /*mu=*/1.0, /*rho=*/1.0, /*alpha=*/0.95, 0);

  ns.solve_ssn(/*ssn_tol=*/1e-8);

  EXPECT_EQ(ns.opt, SSN<double>::TerminationStatus::Optimal);
  EXPECT_NEAR(ns.x(0), -1.0, 1e-6);
}

TEST(SolveSsn, ConvergesToStrictlyInteriorMinimizerWithNoEqualityOrInequalityConstraints) {
  // M=0, l=0: G is a 0-row Schur complement (G.rows() = M + n_active_W = 0), exercising the fully
  // degenerate box-only case end-to-end. Distinct from
  // ConvergesToAnalyticMinimizerWithNoEqualityOrInequalityConstraints (whose optimum sits exactly
  // on the box boundary): here c is chosen so the unconstrained minimizer already lies strictly
  // inside (-1,1), so dist_K stays 0 throughout and the solve never needs to touch a box boundary
  // at all, isolating the 0x0-Schur/LDLT-padding bookkeeping from any active-set-transition logic.
  //   min_x  c*x + (mu/2) dist_K(z/mu+x)^2 + (1/(2 rho)) (x-x_bar)^2,
  // with c=0.2, z=0, mu=rho=1, x_bar=0, box=[-1,1]: interior formula reduces to c*x + x^2/2,
  // minimized at x* = -c = -0.2 (strictly inside the box).
  SpMat A(0, 1), B(0, 1);  // M = 0, l = 0
  SsnFixture f(A, B);
  f.c(0) = 0.2;
  SSN<double> ns = f.Make();

  Vec x0(1);
  x0 << 0.0;
  Vec y10 = Vec::Zero(0), y20 = Vec::Zero(0), z0 = Vec::Zero(1);
  Vec dy10 = Vec::Zero(0), dz0 = Vec::Zero(1);
  ns.update_ssn_system(x0, y10, y20, z0, dy10, dz0, /*mu=*/1.0, /*rho=*/1.0, /*alpha=*/0.95, 0);

  ns.solve_ssn(/*ssn_tol=*/1e-8);

  EXPECT_EQ(ns.opt, SSN<double>::TerminationStatus::Optimal);
  EXPECT_NEAR(ns.x(0), -0.2, 1e-6);
  EXPECT_GT(ns.x(0), -1.0);  // strictly inside the box, not pinned to either bound
  EXPECT_LT(ns.x(0), 1.0);
  EXPECT_EQ(ns.G.rows(), 0);  // 0x0 Schur complement: M + n_active_W = 0 + 0
  EXPECT_EQ(ns.n_active_W, 0);
}
