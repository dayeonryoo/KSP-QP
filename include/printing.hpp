#pragma once
#include <iostream>
#include <iomanip>
#include <functional>
#include <chrono>
#include <Eigen/Dense>

enum class PrintWhen {
    NEVER,
    EVERY10,
    ALWAYS
};

enum class PrintWhat {
    NONE,
    MINIMAL, // iter, tol
    SSN,     // iter, tol, params, linesearch and Krylov failures; printed at every SSN iteration
    TUNING,  // iter, tol, params, linesearch and Krylov failures
    FULL,    // iter, obj_val, tol, params, linesearch and Krylov failures
};

inline double time_diff_s(const std::chrono::steady_clock::time_point& start,
                          const std::chrono::steady_clock::time_point& end) {
    using namespace std::chrono;
    return duration<double>(end - start).count();
}

// One row of the PMM/SSN iteration trace, mirroring print()'s parameter list.
template <typename T>
struct IterationRecord {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    int pmm_iter, ssn_iter, krylov_iter, fact;
    T obj_val;
    Vec res_norms;
    T ssn_res, mu, rho, eps;
    int linesearch_fail, krylov_fail;
    bool show_pmm_iter = true;
    // Active-set sizes and SSN status; -1 means "not applicable". n_active_K is set only on SSN
    // inner-loop records (ssn.tpp), ssn_opt (a cast SSN<T>::TerminationStatus) only on per-PMM records.
    int n_active_W = -1, n_active_K = -1;
    int ssn_opt = -1;
};

inline void print_header(PrintWhen when, PrintWhat what) {
    if (when == PrintWhen::NEVER || what == PrintWhat::NONE) return;

    const int w_iter = 8;
    const int w_val  = 14;

    std::cout << std::setw(w_iter) << "PMM" << std::setw(w_iter) << "SSN";
    if (what == PrintWhat::TUNING || what == PrintWhat::SSN) {
        std::cout << std::setw(w_iter) << "Krylov" << std::setw(w_iter) << "fact";
    }
    if (what == PrintWhat::FULL) {
        std::cout << std::setw(w_val) << "Objective";
    }
    std::cout << std::setw(w_val)  << "PrimalRes"
              << std::setw(w_val)  << "DualRes"
              << std::setw(w_val)  << "Comp_x"
              << std::setw(w_val)  << "Comp_Bx"
              << std::setw(w_val)  << "SSN_res";
    if (what == PrintWhat::TUNING || what == PrintWhat::FULL || what == PrintWhat::SSN) {
        std::cout << std::setw(w_val) << "mu" << std::setw(w_val)  << "rho"  << std::setw(w_val) << "eps";
        std::cout << std::setw(w_val) << "l.f.";
    }
    if (what == PrintWhat::TUNING || what == PrintWhat::SSN) {
        std::cout << std::setw(w_val) << "k.f.";
    }
    std::cout << "\n";

    std::cout << std::string(w_iter*2 + w_val*5, '-');
    if (what == PrintWhat::TUNING || what == PrintWhat::SSN) {
        std::cout << std::string(w_iter*2 + w_val*5, '-');
    } else if (what == PrintWhat::FULL) {
        std::cout << std::string(w_val*4, '-');
    }
    std::cout << "\n";
}

template <typename T, typename Vec>
inline void print(PrintWhen when, PrintWhat what, int pmm_iter, int ssn_iter, int krylov_iter, int fact, T obj_val, const Vec& res_norms, T ssn_res, T mu, T rho, T eps, int linesearch_failures, int krylov_fail, bool show_pmm_iter = true) {
    if (when == PrintWhen::NEVER || what == PrintWhat::NONE) return;
    if (when == PrintWhen::EVERY10 && pmm_iter % 10 != 0) return;

    const int w_iter = 8;
    const int w_val  = 14;

    if (show_pmm_iter) std::cout << std::setw(w_iter) << pmm_iter;
    else std::cout << std::setw(w_iter) << "";
    std::cout << std::setw(w_iter) << ssn_iter;
    if (what == PrintWhat::TUNING || what == PrintWhat::SSN) {
        std::cout << std::setw(w_iter) << krylov_iter << std::setw(w_iter) << fact;
    }
    if (what == PrintWhat::FULL) {
        std::cout << std::setw(w_val)  << std::scientific << obj_val;
    }
    if (res_norms.size() == 0) {
        // Residuals unavailable (e.g. printed from inside the SSN inner loop): leave blank.
        for (int i = 0; i < 4; ++i) std::cout << std::setw(w_val) << "";
    } else {
        T max_res = res_norms.maxCoeff();
        for (T res : res_norms) {
            if (res == max_res) std::cout << "\033[1m" << std::setw(w_val)  << res << "\033[0m";
            else std::cout << std::setw(w_val)  << res;
        }
    }
    std::cout << std::setw(w_val)  << ssn_res;
    if (what == PrintWhat::TUNING || what == PrintWhat::FULL || what == PrintWhat::SSN) {
        std::cout << std::setw(w_val)  << mu << std::setw(w_val)  << rho << std::setw(w_val) << eps;
        std::cout << std::setw(w_val) << linesearch_failures;
    }
    if (what == PrintWhat::TUNING || what == PrintWhat::SSN) {
        std::cout << std::setw(w_val) << krylov_fail;
    }
    std::cout << "\n";
}
