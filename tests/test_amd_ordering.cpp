#include "amd_ordering.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace {

using SpMat = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;
using Perm = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>;

SpMat FromTriplets(int n, const std::vector<Triplet>& trips) {
  SpMat A(n, n);
  A.setFromTriplets(trips.begin(), trips.end());
  A.makeCompressed();
  return A;
}

// 5-point Laplacian on a k x k grid (SPD).
SpMat Laplacian2D(int k) {
  std::vector<Triplet> trips;
  for (int i = 0; i < k; ++i)
    for (int j = 0; j < k; ++j) {
      const int p = i * k + j;
      trips.emplace_back(p, p, 4.0);
      if (i > 0) trips.emplace_back(p, p - k, -1.0);
      if (i < k - 1) trips.emplace_back(p, p + k, -1.0);
      if (j > 0) trips.emplace_back(p, p - 1, -1.0);
      if (j < k - 1) trips.emplace_back(p, p + 1, -1.0);
    }
  return FromTriplets(k * k, trips);
}

// Quasi-definite KKT matrix [-H, G^T; G, (1/mu) I] with a seeded random sparse G (s x n).
SpMat QuasiDefiniteKkt(int n, int s, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> val(-1.0, 1.0), coin(0.0, 1.0);
  std::vector<Triplet> trips;
  for (int j = 0; j < n; ++j) trips.emplace_back(j, j, -(1.0 + coin(gen)));
  for (int i = 0; i < s; ++i) {
    trips.emplace_back(n + i, n + i, 1e-2);
    for (int j = 0; j < n; ++j)
      if (coin(gen) < 0.1) {
        const double v = val(gen);
        trips.emplace_back(n + i, j, v);
        trips.emplace_back(j, n + i, v);
      }
  }
  return FromTriplets(n + s, trips);
}

// Diagonal plus a full first row and column: node 0's degree n - 1 exceeds AMD's dense
// threshold 10 sqrt(n), so AMD sets it aside.
SpMat Arrow(int n) {
  std::vector<Triplet> trips;
  for (int i = 0; i < n; ++i) trips.emplace_back(i, i, static_cast<double>(n));
  for (int i = 1; i < n; ++i) {
    trips.emplace_back(0, i, 1.0);
    trips.emplace_back(i, 0, 1.0);
  }
  return FromTriplets(n, trips);
}

// Tridiagonal pattern whose node 3 has no diagonal entry (AMD sets such nodes aside too).
SpMat TridiagonalMissingOneDiagonal(int n) {
  std::vector<Triplet> trips;
  for (int i = 0; i < n; ++i) {
    if (i != 3) trips.emplace_back(i, i, 2.0);
    if (i > 0) {
      trips.emplace_back(i, i - 1, -1.0);
      trips.emplace_back(i - 1, i, -1.0);
    }
  }
  return FromTriplets(n, trips);
}

Eigen::VectorXi OrderWithIntAmd(const SpMat& C) {
  Perm perm;
  Eigen::AMDOrdering<int>()(C, perm);
  return perm.indices();
}

Eigen::VectorXi OrderWithAmd64(const SpMat& C) {
  Perm perm;
  AMDOrdering64<int>()(C, perm);
  return perm.indices();
}

bool IsPermutation(const Eigen::VectorXi& p) {
  std::vector<int> v(p.data(), p.data() + p.size());
  std::sort(v.begin(), v.end());
  for (int i = 0; i < static_cast<int>(v.size()); ++i)
    if (v[i] != i) return false;
  return true;
}

}  // namespace

// ===================== AMDOrdering64 vs Eigen::AMDOrdering<int> =====================

TEST(AMDOrdering64, MatchesIntAmdWhereItsHashDoesNotOverflow) {
  const std::vector<std::pair<const char*, SpMat>> cases = {
      {"2-D Laplacian", Laplacian2D(30)},
      {"quasi-definite KKT", QuasiDefiniteKkt(60, 40, 7)},
      {"arrow (dense node)", Arrow(400)},
      {"node without diagonal", TridiagonalMissingOneDiagonal(12)},
      {"1x1", FromTriplets(1, {Triplet(0, 0, 1.0)})},
  };
  for (const auto& [name, C] : cases) {
    SCOPED_TRACE(name);
    const Eigen::VectorXi expected = OrderWithIntAmd(C);
    const Eigen::VectorXi got = OrderWithAmd64(C);
    ASSERT_EQ(got.size(), C.rows());
    EXPECT_TRUE(IsPermutation(got));
    EXPECT_EQ(got, expected);
  }
}

TEST(AMDOrdering64, EmptyMatrixGivesAnEmptyPermutation) {
  EXPECT_EQ(OrderWithAmd64(SpMat(0, 0)).size(), 0);
}

TEST(AMDOrdering64, SimplicialSolversMatchTheDefaultOrdering) {
  // Same permutation, hence the same factorization and the same solution as Eigen's default ordering.
  const SpMat S = Laplacian2D(20);
  const Eigen::VectorXd b_s = Eigen::VectorXd::LinSpaced(S.rows(), -1.0, 1.0);
  Eigen::SimplicialLLT<SpMat> llt(S);
  SimplicialLLT64<SpMat> llt64(S);
  ASSERT_EQ(llt.info(), Eigen::Success);
  ASSERT_EQ(llt64.info(), Eigen::Success);
  EXPECT_EQ(llt64.permutationP().indices(), llt.permutationP().indices());
  EXPECT_EQ(Eigen::VectorXd(llt64.solve(b_s)), Eigen::VectorXd(llt.solve(b_s)));

  const SpMat K = QuasiDefiniteKkt(60, 40, 11);
  const Eigen::VectorXd b_k = Eigen::VectorXd::LinSpaced(K.rows(), 1.0, 2.0);
  Eigen::SimplicialLDLT<SpMat> ldlt(K);
  SimplicialLDLT64<SpMat> ldlt64(K);
  ASSERT_EQ(ldlt.info(), Eigen::Success);
  ASSERT_EQ(ldlt64.info(), Eigen::Success);
  EXPECT_EQ(ldlt64.permutationP().indices(), ldlt.permutationP().indices());
  const Eigen::VectorXd x = ldlt64.solve(b_k);
  EXPECT_EQ(x, Eigen::VectorXd(ldlt.solve(b_k)));
  EXPECT_LT((K * x - b_k).norm(), 1e-10 * b_k.norm());
}

// ===================== a pattern that overflows Eigen's int AMD =====================
// Nodes 0 and 1 are both adjacent to the last m nodes; every node has its diagonal. AMD first
// eliminates node n - 1 (lowest degree, last in its degree list), then hashes node 0 by the sum of
// its remaining neighbors n - m, ..., n - 2, which exceeds INT_MAX. Node 0's degree m + 1 stays
// within AMD's dense threshold 10 sqrt(n), so it is not set aside first. Never run
// Eigen::AMDOrdering<int> on this pattern: the wrapped-around hash writes out of bounds.

TEST(AMDOrdering64, OrdersAPatternWhoseIntHashOverflows) {
  const int n = 400000;
  const int m = 6000;
  const std::int64_t node0_hash = std::int64_t(m - 1) * (std::int64_t(n - m) + (n - 2)) / 2;
  ASSERT_GT(node0_hash, std::int64_t(INT_MAX));
  ASSERT_LE(m + 1, static_cast<int>(10 * std::sqrt(static_cast<double>(n))));

  std::vector<Triplet> trips;
  trips.reserve(n + 4 * m);
  for (int i = 0; i < n; ++i) trips.emplace_back(i, i, i < 2 ? 2.0 * m + 1.0 : 3.0);
  for (int j = n - m; j < n; ++j)
    for (int hub : {0, 1}) {
      trips.emplace_back(hub, j, -1.0);
      trips.emplace_back(j, hub, -1.0);
    }
  const SpMat C = FromTriplets(n, trips);

  const Eigen::VectorXi perm = OrderWithAmd64(C);
  ASSERT_EQ(perm.size(), n);
  EXPECT_TRUE(IsPermutation(perm));

  // Diagonally dominant, hence SPD: a factorization in this ordering must solve accurately.
  SimplicialLLT64<SpMat> llt(C);
  ASSERT_EQ(llt.info(), Eigen::Success);
  const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(n, -1.0, 1.0);
  const Eigen::VectorXd x = llt.solve(b);
  EXPECT_LT((C * x - b).norm(), 1e-10 * b.norm());
}
