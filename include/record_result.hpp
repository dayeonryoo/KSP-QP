#pragma once

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "problem.hpp"
#include "solve_profile.hpp"

template <typename T>
struct TestResult {
    std::string system;
    bool agree;
    int opt_status;
    bool diverged;

    std::string name;
    T abs_err;
    T rel_err;

    T obj_val;
    int pmm_iter;
    int ssn_iter;
    int krylov_iter;
    int fact;
    int smw_count;
    T pmm_tol_achieved;
    T ssn_tol_achieved;

    double run_time; // in seconds
    int linesearch_fail;
    int krylov_fail;

    // Appended after the columns above (experimental; see solve_profile.hpp).
    double setup_time = 0.0, solve_time = 0.0; // in seconds
    int kkt_ldlt_fact = 0, schur_chol_fact = 0;
    SolveProfile profile;                      // factorizations by reason, run time by phase
};

// System column: "S" if every Newton system was solved by PCG on the Schur complement; otherwise
// (direct mode, or after PCG failed) "D" followed by "K" if the KKT system was factorized (LDLT)
// and "S" if its Schur complement was (Cholesky).
inline std::string system_label(bool direct_solver_used, int kkt_ldlt_fact, int schur_chol_fact) {
    if (!direct_solver_used) return "S";
    std::string label = "D";
    if (kkt_ldlt_fact > 0)   label += "K";
    if (schur_chol_fact > 0) label += "S";
    return label;
}

template <typename T>
void print_feasibility(const KSPQPdata<T>& pd,
                       const Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
                       const T tol) {
    using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    std::cout << "\nChecking feasibility of solution x from KSP_QP solver:\n";

    std::cout << "  ||Ax - b||_2 = ";
    if (pd.A.rows() == 0) {
        std::cout << "0 (m = 0)\n";
    } else {
        std::cout << (pd.A * x - pd.b).norm() << "\n";
    }

    std::cout << "  ||Ax - b||_inf = ";
    if (pd.A.rows() == 0) {
        std::cout << "0 (m = 0)\n";
    } else {
        std::cout << (pd.A * x - pd.b).cwiseAbs().maxCoeff() << "\n";
    }

    std::cout << "\n  Elements of x outside bounds:\n";
    for (int i = 0; i < pd.c.size(); ++i) {
        if (x[i] < pd.lx[i] - tol || x[i] > pd.ux[i] + tol) {
            std::cout << "      Variable " << i
                      << " out of bounds: x = " << x[i]
                      << ", [" << pd.lx[i] << ", " << pd.ux[i] << "]\n";
        }
    }

    std::cout << "\n  Elements of Bx outside bounds:\n";
    Vec Bx = pd.B * x;
    for (int i = 0; i < pd.lw.size(); ++i) {
        T Bx_i = Bx[i];
        if (Bx_i < pd.lw[i] - tol || Bx_i > pd.uw[i] + tol) {
            std::cout << "      Variable " << i
                      << " out of bounds: Bx = " << Bx_i
                      << ", [" << pd.lw[i] << ", " << pd.uw[i] << "]\n";
        }
    }
}

inline void write_csv_header(const std::string& path) {
    namespace fs = std::filesystem;

    fs::path p(path);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }

    if (!fs::exists(p) || fs::is_empty(p)) {
        std::ofstream csv(path);
        if (!csv.is_open()) {
            throw std::runtime_error("write_csv_header: failed to open " + path + " for writing");
        }
        csv << "System,agree,opt_status,diverged,name,abs_err,rel_err,obj_val,"
            << "pmm_iter,ssn_iter,krylov_iter,fact,smw_count,pmm_tol_achieved,ssn_tol_achieved,"
            << "run_time,linesearch_fail,krylov_fail,"
            << "setup_time,solve_time,kkt_ldlt_fact,schur_chol_fact";
        SolveProfile{}.visit([&](const char* col, const auto&) { csv << ',' << col; });
        csv << "\n";
    } else {
        std::ofstream csv(path, std::ios::out | std::ios::app);
        if (!csv.is_open()) {
            throw std::runtime_error("write_csv_header: failed to open " + path + " for appending");
        }
        csv << "\n";
    }
}

template <typename T>
void append_csv_result(const std::string& path, const TestResult<T>& r) {
    std::ofstream csv(path, std::ios::out | std::ios::app);
    if (!csv.is_open()) {
        throw std::runtime_error("append_csv_result: failed to open " + path + " for appending");
    }
    csv << r.system << "," << r.agree << "," << r.opt_status << "," << r.diverged << ","
        << r.name << "," << r.abs_err << "," << r.rel_err << ","
        << r.obj_val << "," << r.pmm_iter << "," << r.ssn_iter << ","
        << r.krylov_iter << "," << r.fact << "," << r.smw_count << ","
        << r.pmm_tol_achieved << "," << r.ssn_tol_achieved << ","
        << r.run_time << "," << r.linesearch_fail << "," << r.krylov_fail;
    csv << std::setprecision(10) << "," << r.setup_time << "," << r.solve_time << ","
        << r.kkt_ldlt_fact << "," << r.schur_chol_fact;
    r.profile.visit([&](const char*, const auto& v) { csv << ',' << v; });
    csv << "\n";
}

// Trace CSVs (--trace PREFIX): PREFIX_events.csv, PREFIX_ssn.csv and PREFIX_pmm.csv get one row per
// linear-solver event, SSN iteration and PMM iteration, each led by the problem's name. They are
// appended to, with the header written when a file is new.
template <typename Row>
void append_trace_csv(const std::string& path, const std::string& name, const std::vector<Row>& rows) {
    namespace fs = std::filesystem;
    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    const bool new_file = !fs::exists(p) || fs::is_empty(p);
    std::ofstream csv(path, std::ios::out | std::ios::app);
    if (!csv.is_open()) throw std::runtime_error("append_trace_csv: failed to open " + path + " for appending");
    csv << std::setprecision(10);
    if (new_file) {
        csv << "name";
        Row{}.visit([&](const char* col, const auto&) { csv << ',' << col; });
        csv << "\n";
    }
    for (const Row& row : rows) {
        csv << name;
        row.visit([&](const char*, const auto& v) { csv << ',' << v; });
        csv << "\n";
    }
}

template <typename T>
void append_trace_csvs(const std::string& prefix, const std::string& name, const SolveTrace<T>& trace) {
    append_trace_csv(prefix + "_events.csv", name, trace.events);
    append_trace_csv(prefix + "_ssn.csv", name, trace.ssn);
    append_trace_csv(prefix + "_pmm.csv", name, trace.pmm);
}

// Prints the factorizations by reason and the run time by phase (single-problem runs).
inline void print_profile(const SolveProfile& p) {
    std::cout << "Factorizations by reason:";
    for (int i = 0; i < kNumFactReasons; ++i)
        if (p.fact_by_reason[i] > 0)
            std::cout << " " << to_string(static_cast<FactReason>(i)) << "=" << p.fact_by_reason[i];
    std::cout << "  (with symbolic analysis: " << p.fact_analyzed << "); SMW updates: " << p.smw_update
              << ", reuses: " << p.smw_reuse << ", PCG failures: " << p.pcg_fail << "\n";
    std::cout << "Run time by phase (s): setup=" << p.t.setup << " prep=" << p.t.prep << " fact=" << p.t.fact
              << " lowrank=" << p.t.lowrank << " krylov=" << p.t.krylov << " direct_solve=" << p.t.direct_solve
              << " linesearch=" << p.t.linesearch << " update=" << p.t.update << " ssn_other=" << p.t.ssn_other
              << " pmm_other=" << p.t.pmm_other << "\n";
}
