#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <limits>

// =============================================================
//      min  c^T x + 0.5 x^T Q x + obj_const,
//      s.t. A x = b,
//           B x = w,
//           lx <= x <= ux,
//           lw <= w <= uw.
// =============================================================

// PMM-level termination status, shared by KSP_QP::opt and Solution::opt.
enum class TerminationStatus : int {
    DualInfeasible   = -3, // termination due to dual infeasibility
    PrimalInfeasible = -2, // termination due to primal infeasibility
    NumericalError   = -1, // termination due to numerical errors (setup or solve exception)
    Optimal          =  0, // optimal solution found
    MaxPmmIterations =  1, // maximum number of PMM iterations reached
    MaxSsnIterations =  2, // maximum number of SSN iterations reached
    TimeLimit        =  3, // time limit exceeded
    Interrupted      =  4, // solve was interrupted before converging
};

// Short status label per termination status.
inline const char* to_string(TerminationStatus opt) {
    switch (opt) {
        case TerminationStatus::DualInfeasible:   return "dual infeasible";
        case TerminationStatus::PrimalInfeasible: return "primal infeasible";
        case TerminationStatus::NumericalError:   return "numerical error";
        case TerminationStatus::Optimal:          return "optimal";
        case TerminationStatus::MaxPmmIterations: return "max PMM iterations reached";
        case TerminationStatus::MaxSsnIterations: return "max SSN iterations reached";
        case TerminationStatus::TimeLimit:        return "time limit reached";
        case TerminationStatus::Interrupted:      return "interrupted";
    }
    return "unknown status";
}

template <typename T>
class Solution {
public:
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    TerminationStatus opt; // Termination status; see TerminationStatus above.

    Vec x;           // Optimal primal solution vector
    Vec y1;          // Lagrangian multipliers for Ax = b
    Vec y2;          // Lagrangian multipliers for Bx = w
    Vec z;           // Lagrangian multipliers for box constraints on x
    T obj_val;       // Optimal objective value

    int pmm_iter;    // Number of PMM iterations performed
    int ssn_iter;    // Number of SSN iterations performed
    int krylov_iter; // Number of Krylov iterations performed
    int fact;        // Number of factorizations performed
    int smw_count;   // Number of SMW low-rank updates used instead of refactorizing (preconditioner or direct solver)

    T pmm_tol_achieved; // Tolerance achieved by PMM
    T ssn_tol_achieved; // Tolerance achieved by SSN

    double setup_time;   // Wall-clock time in seconds spent in the KSP_QP constructor
    double solve_time;   // Wall-clock time in seconds spent in solve()
    double run_time;     // setup_time + solve_time
    int linesearch_fail; // Number of linesearch failures
    int krylov_fail;     // Number of Krylov failures

    Solution(TerminationStatus opt, const Vec& x, const Vec& y1, const Vec& y2, const Vec& z,
             T obj_val, int pmm_iter, int ssn_iter, int krylov_iter, int fact, int smw_count,
             T pmm_tol_achieved, T ssn_tol_achieved,
             double setup_time, double solve_time, int linesearch_fail, int krylov_fail)
    : opt(opt), x(x), y1(y1), y2(y2), z(z), obj_val(obj_val),
      pmm_iter(pmm_iter), ssn_iter(ssn_iter), krylov_iter(krylov_iter), fact(fact), smw_count(smw_count),
      pmm_tol_achieved(pmm_tol_achieved), ssn_tol_achieved(ssn_tol_achieved),
      setup_time(setup_time), solve_time(solve_time), run_time(setup_time + solve_time),
      linesearch_fail(linesearch_fail), krylov_fail(krylov_fail)
    {}

    void print_summary() const {
        std::cout << "\n";
        std::cout << "Solution Summary:" << std::endl;
        std::cout << "Termination status (opt): " << to_string(opt) << " (" << static_cast<int>(opt) << ")" << std::endl;
        std::cout << "Problem dimensions: n = " << x.size() << ", m = " << y1.size() << ", l = " << y2.size() << std::endl;

        if (opt == TerminationStatus::NumericalError) {
            std::cout << "Solver terminated due to a numerical error during the solve.\n";
            return;
        }
        if (opt == TerminationStatus::PrimalInfeasible) {
            std::cout << "Problem detected primal infeasible via its infeasibility certificate.\n";
            std::cout << "If you believe the problem is feasible, consider lowering eps_pinf directly (ksp_qp.hpp, near the 'Constant parameters' block).\n";
        } else if (opt == TerminationStatus::DualInfeasible) {
            std::cout << "Problem detected dual infeasible via its infeasibility certificate.\n";
            std::cout << "If you believe the problem is feasible, consider lowering eps_dinf directly (ksp_qp.hpp, near the 'Constant parameters' block).\n";
        } else {
            // Report the best iterate found so far, whether or not it is confirmed optimal.
            if (opt == TerminationStatus::Optimal) {
                std::cout << "Solver converged to an optimal solution.\n";
            } else if (opt == TerminationStatus::MaxPmmIterations) {
                std::cout << "Solver reached the maximum number of PMM iterations before converging.\n";
            } else if (opt == TerminationStatus::MaxSsnIterations) {
                std::cout << "Solver reached the maximum number of SSN iterations before converging.\n";
            } else if (opt == TerminationStatus::TimeLimit) {
                std::cout << "Solver reached the time limit before converging.\n";
            } else if (opt == TerminationStatus::Interrupted) {
                std::cout << "Solver was interrupted before converging.\n";
            }
            std::cout << "Objective value (obj_val): " << obj_val << std::endl;
            std::cout << "Number of PMM iterations (pmm_iter): " << pmm_iter << std::endl;
            std::cout << "Number of SSN iterations (ssn_iter): " << ssn_iter << std::endl;
            std::cout << "Number of Krylov iterations (krylov_iter): " << krylov_iter << std::endl;
            std::cout << "Number of factorizations (fact): " << fact << std::endl;
            std::cout << "Number of low-rank updates (smw_count): " << smw_count << std::endl;
            std::cout << "PMM tolerance achieved (pmm_tol_achieved): " << pmm_tol_achieved << std::endl;
            std::cout << "SSN tolerance achieved (ssn_tol_achieved): " << ssn_tol_achieved << std::endl;
        }
        std::cout << "Number of linesearch failures (linesearch_fail): " << linesearch_fail << std::endl;
        std::cout << "Number of Krylov failures (krylov_fail): " << krylov_fail << std::endl;
        std::cout << "Run time (run_time): " << run_time << " seconds\n";
    }

};
