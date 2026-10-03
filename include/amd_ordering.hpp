#pragma once
#include <cstdint>
#include <Eigen/Sparse>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>

// Approximate minimum degree ordering computed in 64-bit indices, for the int-indexed sparse
// factorizations (the PCG preconditioner, the direct fallback, Q's LDLT).
//
// Eigen's AMD (OrderingMethods/Amd.h) hashes each node by a running sum of node indices held in
// the matrix's StorageIndex. With int indices that sum overflows on large patterns -- the L2-PDE
// nc = 9 preconditioner (1.58M rows, 35.6M nonzeros) reaches 3.8e9 -- and the wrapped-around hash
// indexes out of bounds. AMDOrdering64 runs the same AMD on a 64-bit copy of the pattern and casts
// the permutation back; whenever the int hash does not overflow, the permutation is identical to
// Eigen::AMDOrdering<int>'s. The copy is freed before returning, so it only raises the peak memory
// of analyzePattern().
template <typename StorageIndex>
class AMDOrdering64 {
public:
    using PermutationType = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, StorageIndex>;

    template <typename MatrixType>
    void operator()(const MatrixType& mat, PermutationType& perm) const {
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, std::int64_t> perm64;
        {
            const Eigen::SparseMatrix<typename MatrixType::Scalar, Eigen::ColMajor, std::int64_t> mat64(mat);
            Eigen::AMDOrdering<std::int64_t>()(mat64, perm64);
        }
        perm.indices() = perm64.indices().template cast<StorageIndex>();
    }
};

// Eigen::SimplicialLLT / SimplicialLDLT with their default AMD ordering run in 64-bit indices.
template <typename SpMat>
using SimplicialLLT64 = Eigen::SimplicialLLT<SpMat, Eigen::Lower, AMDOrdering64<typename SpMat::StorageIndex>>;

template <typename SpMat>
using SimplicialLDLT64 = Eigen::SimplicialLDLT<SpMat, Eigen::Lower, AMDOrdering64<typename SpMat::StorageIndex>>;
