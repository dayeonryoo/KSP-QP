#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <map>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <thread>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "ksp_qp.hpp"
#include "problem.hpp"
#include "printing.hpp"
#include "mps_format_parser.hpp"
#include "record_result.hpp"
#include "cli_args.hpp"

using T = double;
using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using SpMat = Eigen::SparseMatrix<T>;
using Triplet = Eigen::Triplet<T>;

// Netlib .mps filenames are lowercase; problem names are reported uppercase
// (afiro.mps -> AFIRO), as in python/benchmark_netlib.py.
static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return s;
}

static std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });
    return s;
}

// Problem set = the .mps files in data_dir, sorted.
static std::vector<std::string> list_problems(const std::string& data_dir) {
    std::vector<std::string> stems;
    if (!std::filesystem::is_directory(data_dir)) {
        std::cerr << "ERROR: not a directory: " << data_dir << "\n";
        return stems;
    }
    for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".mps") {
            stems.push_back(entry.path().stem().string());
        }
    }
    std::sort(stems.begin(), stems.end());
    return stems;
}

// Reference objectives from a "name,status,solveTime,pobj" CSV (default: Gurobi 10 at 1e-8).
// They differ from the published Netlib optima on a few problems (e.g. E226, CRE-A).
// Returns an empty map if the file is absent.
static std::map<std::string, double> load_reference_objectives(const std::string& csv_path) {
    std::map<std::string, double> refs;
    std::ifstream csv(csv_path);
    if (!csv.is_open()) {
        std::cerr << "WARNING: reference objectives not found: " << csv_path
                  << " (sweep will report no comparison)\n";
        return refs;
    }

    std::string line;
    std::getline(csv, line); // header
    while (std::getline(csv, line)) {
        if (line.empty()) continue;
        std::istringstream fields(line);
        std::string name, status, solve_time, pobj;
        if (!std::getline(fields, name, ',')) continue;
        if (!std::getline(fields, status, ',')) continue;
        if (!std::getline(fields, solve_time, ',')) continue;
        if (!std::getline(fields, pobj, ',')) continue;
        try {
            refs[to_lower(name)] = std::stod(pobj);
        } catch (const std::exception&) {
            std::cerr << "WARNING: unparsable reference objective for " << name << "\n";
        }
    }
    return refs;
}

// ==================== Solving Netlib LPs ====================
//
// Two problem sets, selected with --set:
//   feasible   -- the 114 feasible instances, checked against reference objectives
//   infeasible -- the 29 primal-infeasible instances, checked for detected infeasibility
//
// Both write the same CSV schema (record_result.hpp); on the infeasible set
// `agree` means "infeasibility was detected".

int main(int argc, char** argv) {
    if (cli::has_flag(argc, argv, "--help") || cli::has_flag(argc, argv, "-h")) {
        std::cout <<
            "Usage: ksp_qp_netlib [--root DIR] [--set feasible|infeasible] [--in DIR] [--name PROBLEM|all]\n"
            "                     [--tol T] [--max-iter N] [--time-limit S] [--out FILE] [--cooldown S] [--ref FILE]\n"
            "  Solves one Netlib LP from ROOT/IN/PROBLEM.mps, or sweeps the whole set with --name all.\n"
            "  --root DIR       working directory; --in, --ref and --out are resolved relative to this (default: ./)\n"
            "  --set SET        which problem set to use: feasible or infeasible (default: feasible)\n"
            "  --in DIR         directory containing the .mps files, relative to --root\n"
            "                   (default: data/netlib-main/feasible/ or data/netlib-main/infeasible/, per --set)\n"
            "  --name PROBLEM   problem name, case-insensitive, without extension, or \"all\" to sweep\n"
            "                   every .mps file in the directory (default: AFIRO / KLEIN1, per --set)\n"
            "  --tol T          primal-dual tolerance (default: 1e-6)\n"
            "  --max-iter N     max PMM iterations (default: 3000)\n"
            "  --time-limit S   time limit in seconds (default: 60)\n"
            "  --out FILE       (--name all only) output CSV path, relative to --root\n"
            "                   (default: results/pcg_netlib.csv or results/pcg_infeas.csv, per --set)\n"
            "  --cooldown S     (--name all only) seconds to sleep between problems (default: 0)\n"
            "  --ref FILE       (feasible set only) reference objectives CSV, relative to --root\n"
            "                   (default: data/netlib-main/feasible_gurobi_1e-8.csv)\n";
        return 0;
    }

    std::string root = cli::get_str(argc, argv, "--root", "./");
    if (!root.empty() && root.back() != '/') root += '/';

    std::string set = to_lower(cli::get_str(argc, argv, "--set", "feasible"));
    if (set != "feasible" && set != "infeasible") {
        std::cerr << "ERROR: --set must be 'feasible' or 'infeasible' (got '" << set << "')\n";
        return 1;
    }
    const bool infeasible_set = (set == "infeasible");

    std::string default_in = infeasible_set ? "data/netlib-main/infeasible/"
                                            : "data/netlib-main/feasible/";
    std::string in_dir = cli::get_str(argc, argv, "--in", default_in);
    if (!in_dir.empty() && in_dir.back() != '/') in_dir += '/';
    std::string data_dir = root + in_dir;

    std::string default_name = infeasible_set ? "KLEIN1" : "AFIRO";
    std::string name = cli::get_str(argc, argv, "--name", default_name);

    T tol = cli::get_double(argc, argv, "--tol", 1e-6);
    double time_limit = cli::get_double(argc, argv, "--time-limit", 60.0); // in seconds
    int max_iter = cli::get_int(argc, argv, "--max-iter", 3000);

    // Unknown flags are otherwise ignored; fail loudly instead of silently running PCG.
    if (cli::has_flag(argc, argv, "--direct")) {
        std::cerr << "ERROR: --direct (forced direct factorization) is only available on branch direct-vs-pcg\n";
        return 1;
    }

    if (to_lower(name) == "all") {
        std::vector<std::string> stems = list_problems(data_dir);
        if (stems.empty()) {
            std::cerr << "ERROR: no .mps files found in " << data_dir << "\n";
            return 1;
        }

        // Reference objectives only exist for the feasible set.
        std::map<std::string, double> refs;
        if (!infeasible_set) {
            std::string ref_path = root + cli::get_str(argc, argv, "--ref",
                                                       "data/netlib-main/feasible_gurobi_1e-8.csv");
            refs = load_reference_objectives(ref_path);
        }

        PrintWhen when = PrintWhen::NEVER;
        PrintWhat what = PrintWhat::TUNING;
        int cooldown_sec = cli::get_int(argc, argv, "--cooldown", 0);

        std::string default_out = infeasible_set ? "results/pcg_infeas.csv" : "results/pcg_netlib.csv";
        std::string csv_path = root + cli::get_str(argc, argv, "--out", default_out);
        write_csv_header(csv_path);

        for (const std::string& stem : stems) {
            std::string lp_name = to_upper(stem);
            std::string filename = data_dir + stem + ".mps";

            std::cout << "\n============================================= " << lp_name
                      << " =============================================\n";
            std::time_t curr_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::cout << std::ctime(&curr_time);

            try {
                // Read problem data from the file
                MpsFormatParser<T> parser;
                ParsedModel<T> model = parser.parse(filename);
                KSPQPdata<T> pd = parser.to_kspqp(model);

                // Construct the problem and solver
                Problem<T> prob(pd, tol, max_iter, time_limit, when, what);
                KSP_QP<T> solver(prob);

                // Solve the LP
                Solution<T> sol = solver.solve();

                // Compare against the expected outcome for this set: a reference
                // objective (feasible) or a detected infeasibility (infeasible).
                bool agree = false;
                T abs_err = -1.0;
                T rel_err = -1.0;
                if (infeasible_set) {
                    agree = (sol.opt == TerminationStatus::PrimalInfeasible
                             || sol.opt == TerminationStatus::DualInfeasible);
                } else {
                    auto ref = refs.find(stem);
                    if (ref == refs.end()) {
                        std::cerr << "WARNING: no reference objective for " << lp_name << "\n";
                    } else {
                        abs_err = std::abs(sol.obj_val - ref->second);
                        rel_err = abs_err / std::abs(ref->second);
                        T err_tol = 1e-2;
                        agree = (abs_err < err_tol) || (rel_err < err_tol);
                    }
                }
                // On the infeasible set a large residual is expected, so `diverged` is feasible-set only.
                bool diverged = !infeasible_set && sol.pmm_tol_achieved > 1e0;

                // Record result
                std::string system = system_label(solver.pcg_failed, solver.kkt_ldlt_fact, solver.schur_chol_fact);

                TestResult<T> result = {
                    system,
                    agree, static_cast<int>(sol.opt), diverged, lp_name,
                    abs_err, rel_err,
                    sol.obj_val, sol.pmm_iter, sol.ssn_iter, sol.krylov_iter, sol.fact, sol.smw_count,
                    sol.pmm_tol_achieved, sol.ssn_tol_achieved,
                    sol.run_time, sol.linesearch_fail, sol.krylov_fail
                };
                append_csv_result(csv_path, result);

            } catch (const std::exception& e) {
                std::cerr << "ERROR solving " << lp_name << ": " << e.what() << "\n";
                TestResult<T> result = {
                    e.what(),
                    false, -1, false, lp_name,
                    -1.0, -1.0,
                    -1.0, -1, -1, -1, -1, -1,
                    -1.0, -1.0,
                    -1.0, -1, -1
                };
                append_csv_result(csv_path, result);
            }

            std::this_thread::sleep_for(std::chrono::seconds(cooldown_sec));
        }

        return 0;
    }

    // ---- Single-problem solve ----
    PrintWhen when = PrintWhen::ALWAYS;
    PrintWhat what = PrintWhat::TUNING;

    std::string filename = data_dir + to_lower(name) + ".mps";

    std::cout << "==================== Solving " + to_upper(name) << " ====================\n";

    MpsFormatParser<T> parser;
    ParsedModel<T> model = parser.parse(filename);
    KSPQPdata<T> pd = parser.to_kspqp(model);

    Problem<T> prob(pd, tol, max_iter, time_limit, when, what);
    KSP_QP<T> solver(prob);

    // Solve:
    Solution<T> sol = solver.solve();
    sol.print_summary();
    if (solver.pcg_failed) {
        std::cout << "Direct-solver factorizations: KKT system (LDLT) = " << solver.kkt_ldlt_fact
                  << ", Schur complement (Cholesky) = " << solver.schur_chol_fact << "\n";
    }

    // On the infeasible set, the iterate's residuals show how the problem is infeasible.
    if (infeasible_set) {
        print_feasibility(pd, sol.x, tol);
    }

    return 0;
}
