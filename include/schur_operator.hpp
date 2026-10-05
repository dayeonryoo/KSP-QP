#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

template <typename T>
class SchurOperator;

namespace Eigen {
namespace internal {
    template <typename T>
    struct traits<SchurOperator<T>> : traits<SparseMatrix<T>> {
        using Scalar = T;
        using RealScalar = typename NumTraits<T>::Real;
        using StorageIndex = Eigen::Index;

        enum {
            ColsAtCompileTime = Dynamic,
            RowsAtCompileTime = Dynamic,
            MaxColsAtCompileTime = Dynamic,
            MaxRowsAtCompileTime = Dynamic,
            Flags = 0
        };
    };
}
}

template <typename T>
class SchurOperator : public Eigen::EigenBase<SchurOperator<T>> {
public:
    using Scalar = T;
    using RealScalar = typename Eigen::NumTraits<T>::Real;
    using StorageIndex = Eigen::Index;
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using SpMat = Eigen::SparseMatrix<T>;

    enum {
        RowsAtCompileTime = Eigen::Dynamic,
        ColsAtCompileTime = Eigen::Dynamic,
        MaxRowsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };    

    SchurOperator(const SpMat& G_, const SpMat& G_tr_, const Vec& H_diag_inv_, T mu)
        : G(G_), G_tr(G_tr_), H_diag_inv(H_diag_inv_), mu_inv(T(1) / mu) {
            m_ = G.rows(); // Schur operator is m x m
            t_.resize(G.cols());
            u_.resize(G.cols());
    }

    Eigen::Index rows() const { return m_; }
    Eigen::Index cols() const { return m_; }

    // Called once per CG iteration; t_/u_ are preallocated scratch, so only the result allocates.
    // Returned by value so two products on the same S in one expression (a*(S*v1) + b*(S*v2))
    // stay independent.
    template <typename Rhs>
    Vec operator*(const Eigen::MatrixBase<Rhs>& v) const {
        t_.noalias() = G_tr * v;
        u_.noalias() = H_diag_inv.cwiseProduct(t_);
        Vec w = G * u_;
        w.noalias() += mu_inv * v;
        return w;
    }

private:
    const SpMat& G;
    const SpMat& G_tr;
    const Vec& H_diag_inv;
    T mu_inv;
    Eigen::Index m_;
    mutable Vec t_, u_;
};