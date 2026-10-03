#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <map>
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

// ----------------------- Solving individual problem -----------------------

int main(int argc, char** argv) {
    if (cli::has_flag(argc, argv, "--help") || cli::has_flag(argc, argv, "-h")) {
        std::cout <<
            "Usage: ksp_qp_maros_meszaros [--root DIR] [--in DIR] [--name PROBLEM|all] [--tol T] [--max-iter N] [--time-limit S] [--out FILE] [--cooldown S]\n"
            "  Solves one Maros-Meszaros QP from ROOT/IN/PROBLEM.SIF, or sweeps the whole set with --name all.\n"
            "  --root DIR       working directory; --in and --out are resolved relative to this (default: ./)\n"
            "  --in DIR         directory containing the .SIF files, relative to --root (default: data/maros_meszaros/)\n"
            "  --name PROBLEM   problem name, without extension, or \"all\" to sweep every QP with a\n"
            "                   known reference objective (default: AUG2DCQP)\n"
            "  --tol T          primal-dual tolerance (default: 1e-6)\n"
            "  --max-iter N     max PMM iterations (default: 3000)\n"
            "  --time-limit S   time limit in seconds (default: 60)\n"
            "  --out FILE       (--name all only) output CSV path, relative to --root\n"
            "                   (default: results/pcg_mm.csv)\n"
            "  --cooldown S     (--name all only) seconds to sleep between problems (default: 3)\n";
        return 0;
    }

    std::string root = cli::get_str(argc, argv, "--root", "./");
    if (!root.empty() && root.back() != '/') root += '/';
    std::string in_dir = cli::get_str(argc, argv, "--in", "data/maros_meszaros/");
    if (!in_dir.empty() && in_dir.back() != '/') in_dir += '/';
    std::string data_dir = root + in_dir;
    std::string name = cli::get_str(argc, argv, "--name", "AUG2DCQP");

    T tol = cli::get_double(argc, argv, "--tol", 1e-6);
    double time_limit = cli::get_double(argc, argv, "--time-limit", 60.0); // in seconds
    int max_iter = cli::get_int(argc, argv, "--max-iter", 3000);

    // Unknown flags are otherwise ignored; fail loudly instead of silently running PCG.
    if (cli::has_flag(argc, argv, "--direct")) {
        std::cerr << "ERROR: --direct (forced direct factorization) is only available on branch direct-vs-pcg\n";
        return 1;
    }

    if (name == "all") {
        // Filenames and objective values of Maros/Meszaros QPs
        static const std::map<std::string, double> QPs = {
            {"AUG2D",      1.6874118e+06},
            {"AUG2DC",     1.8183681e+06},
            {"AUG2DCQP",   6.4981348e+06},
            {"AUG2DQP",    6.2370121e+06},
            {"AUG3D",      5.5406773e+02},
            {"AUG3DC",     7.7126244e+02},
            {"AUG3DCQP",   9.9336215e+02},
            {"AUG3DQP",    6.7523767e+02},
            {"BOYD1",     -6.1735220e+07},
            {"BOYD2",      2.1256767e+01},
            {"CONT-050",  -4.5638509e+00},
            {"CONT-100",  -4.6443979e+00},
            {"CONT-101",   1.9552733e-01},
            {"CONT-200",  -4.6848759e+00},
            {"CONT-201",   1.9248337e-01},
            {"CONT-300",   1.9151232e-01},
            {"CVXQP1L",    1.0870480e+08},
            {"CVXQP1M",    1.0875116e+06},
            {"CVXQP1S",    1.1590718e+04},
            {"CVXQP2L",    8.1842458e+07},
            {"CVXQP2M",    8.2015543e+05},
            {"CVXQP2S",    8.1209405e+03},
            {"CVXQP3L",    1.1571110e+08},
            {"CVXQP3M",    1.3628287e+06},
            {"CVXQP3S",    1.1943432e+04},
            {"DPKLO1",     3.7009622e-01},
            {"DTOC3",      2.3526248e+02},
            {"DUAL1",      3.5012966e-02},
            {"DUAL2",      3.3733676e-02},
            {"DUAL3",      1.3575584e-01},
            {"DUAL4",      7.4609084e-01},
            {"DUALC1",     6.1552508e+03},
            {"DUALC2",     3.5513077e+03},
            {"DUALC5",     4.2723233e+02},
            {"DUALC8",     1.8309359e+04},
            {"EXDATA",    -1.4184343e+02},
            {"GENHS28",    9.2717369e-01},
            {"GOULDQP2",   1.8427534e-04},
            {"GOULDQP3",   2.0627840e+00},
            {"HS118",      6.6482045e+02},
            {"HS21",      -9.9960000e+01},
            {"HS268",      5.7310705e-07},
            {"HS35",       1.1111111e-01},
            {"HS35MOD",    2.5000000e-01},
            {"HS51",       8.8817842e-16},
            {"HS52",       5.3266476e+00},
            {"HS53",       4.0930233e+00},
            {"HS76",      -4.6818182e+00},
            {"HUES-MOD",   3.4824690e+07},
            {"HUESTIS",    3.4824690e+11},
            {"KSIP",       5.7579794e-01},
            {"LASER",      2.4096014e+06},
            {"LISWET1",    3.6122402e+01},
            {"LISWET10",   4.9485785e+01},
            {"LISWET11",   4.9523957e+01},
            {"LISWET12",   1.7369274e+03},
            {"LISWET2",    2.4998076e+01},
            {"LISWET3",    2.5001220e+01},
            {"LISWET4",    2.5000112e+01},
            {"LISWET5",    2.5034253e+01},
            {"LISWET6",    2.4995748e+01},
            {"LISWET7",    4.9884089e+02},
            {"LISWET8",    7.1447006e+03},
            {"LISWET9",    1.9632513e+03},
            {"LOTSCHD",    2.3984159e+03},
            {"MOSARQP1",  -9.5287544e+02},
            {"MOSARQP2",  -1.5974821e+03},
            {"POWELL20",   5.2089583e+10},
            {"PRIMAL1",   -3.5012965e-02},
            {"PRIMAL2",   -3.3733676e-02},
            {"PRIMAL3",   -1.3575584e-01},
            {"PRIMAL4",   -7.4609083e-01},
            {"PRIMALC1",  -6.1552508e+03},
            {"PRIMALC2",  -3.5513077e+03},
            {"PRIMALC5",  -4.2723233e+02},
            {"PRIMALC8",  -1.8309430e+04},
            {"Q25FV47",    1.3744448e+07},
            {"QADLITTL",   4.8031886e+05},
            {"QAFIRO",    -1.5907818e+00},
            {"QBANDM",     1.6352342e+04},
            {"QBEACONF",   1.6471206e+05},
            {"QBORE3D",    3.1002008e+03},
            {"QBRANDY",    2.8375115e+04},
            {"QCAPRI",     6.6793293e+07},
            {"QE226",      2.1265343e+02},
            {"QETAMACR",   8.6760370e+04},
            {"QFFFFF80",   8.7314747e+05},
            {"QFORPLAN",   7.4566315e+09},
            {"QGFRDXPN",   1.0079059e+11},
            {"QGROW15",   -1.0169364e+08},
            {"QGROW22",   -1.4962895e+08},
            {"QGROW7",    -4.2798714e+07},
            {"QISRAEL",    2.5347838e+07},
            {"QPCBLEND",  -7.8425409e-03},
            {"QPCBOEI1",   1.1503914e+07},
            {"QPCBOEI2",   8.1719623e+06},
            {"QPCSTAIR",   6.2043875e+06},
            {"QPILOTNO",   4.7285869e+06},
            {"QPTEST",     4.3718750e+00},
            {"QRECIPE",   -2.6661600e+02},
            {"QSC205",    -5.8139518e-03},
            {"QSCAGR25",   2.0173794e+08},
            {"QSCAGR7",    2.6865949e+07},
            {"QSCFXM1",    1.6882692e+07},
            {"QSCFXM2",    2.7776162e+07},
            {"QSCFXM3",    3.0816355e+07},
            {"QSCORPIO",   1.8805096e+03},
            {"QSCRS8",     9.0456001e+02},
            {"QSCSD1",     8.6666667e+00},
            {"QSCSD6",     5.0808214e+01},
            {"QSCSD8",     9.4076357e+02},
            {"QSCTAP1",    1.4158611e+03},
            {"QSCTAP2",    1.7350265e+03},
            {"QSCTAP3",    1.4387547e+03},
            {"QSEBA",      8.1481801e+07},
            {"QSHARE1B",   7.2007832e+05},
            {"QSHARE2B",   1.1703692e+04},
            {"QSHELL",     1.5726368e+12},
            {"QSHIP04L",   2.4200155e+06},
            {"QSHIP04S",   2.4249937e+06},
            {"QSHIP08L",   2.3760406e+06},
            {"QSHIP08S",   2.3857289e+06},
            {"QSHIP12L",   3.0188766e+06},
            {"QSHIP12S",   3.0569623e+06},
            {"QSIERRA",    2.3750458e+07},
            {"QSTAIR",     7.9854528e+06},
            {"QSTANDAT",   6.4118384e+03},
            {"S268",       5.7310705e-07},
            {"STADAT1",   -2.8526864e+07},
            {"STADAT2",   -3.2626665e+01},
            {"STADAT3",   -3.5779453e+01},
            {"STCQP1",     1.5514356e+05},
            {"STCQP2",     2.2327313e+04},
            {"TAME",       0.0000000e+00},
            {"UBH1",       1.1160008e+00},
            {"VALUES",    -1.3966211e+00},
            {"YAO",        1.9770426e+02},
            {"ZECEVIC2",  -4.1250000e+00}
        };

        PrintWhen when = PrintWhen::NEVER;
        PrintWhat what = PrintWhat::TUNING;
        int cooldown_sec = cli::get_int(argc, argv, "--cooldown", 3);

        std::string csv_path = root + cli::get_str(argc, argv, "--out", "results/pcg_mm.csv");
        write_csv_header(csv_path);

        for (const auto& [qp_name, ref_obj_val] : QPs) {

            // Build full path and check if file exists
            std::string filename = data_dir + qp_name + ".SIF";
            if (!std::filesystem::exists(filename)) {
                std::cerr << "SKIP: File not found: " << qp_name << "\n";
                continue;
            }

            std::cout << "\n============================================= " << qp_name << " =============================================\n";
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

                // Solve the QP
                Solution<T> sol = solver.solve();

                // Compare
                T abs_err = std::abs(sol.obj_val - ref_obj_val);
                T rel_err = abs_err / std::abs(ref_obj_val);
                T err_tol = 1e-2;
                bool agree = (abs_err < err_tol) || (rel_err < err_tol);
                bool diverged = sol.pmm_tol_achieved > 1e0;

                // Record result
                std::string system = system_label(solver.pcg_failed, solver.kkt_ldlt_fact, solver.schur_chol_fact);

                TestResult<T> result = {
                    system,
                    agree, static_cast<int>(sol.opt), diverged, qp_name,
                    abs_err, rel_err,
                    sol.obj_val, sol.pmm_iter, sol.ssn_iter, sol.krylov_iter, sol.fact, sol.smw_count,
                    sol.pmm_tol_achieved, sol.ssn_tol_achieved,
                    sol.run_time, sol.linesearch_fail, sol.krylov_fail
                };
                append_csv_result(csv_path, result);

            } catch (const std::exception& e) {
                std::cerr << "ERROR solving " << qp_name << ": " << e.what() << "\n";
                TestResult<T> result = {
                    e.what(),
                    false, -1, false, qp_name,
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

    std::string filename = data_dir + name + ".SIF";

    std::cout << "==================== Solving " + name << " ====================\n";

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

    return 0;
}
