#pragma once
#include <array>
#include <chrono>
#include <limits>
#include <vector>

// =================================================================================================
// Factorization log and run-time split (experimental; branch direct-vs-pcg).
//
// SolveProfile (always on) holds a run's totals: its factorizations by reason, SMW updates, PCG
// failures, and its run time split into disjoint phases that add up to the run time.
// SolveTrace (opt-in, KSP_QP::trace.enabled) also keeps one row per linear-solver event
// (factorization, SMW update or reuse, kept factorization, PCG failure), per SSN iteration and per
// PMM iteration.
// Every row type lists its columns through visit(f), which calls f(name, value) per column, so
// the CSV writers (record_result.hpp) and the Python binding share one definition.
// =================================================================================================

// RAII wall-clock timer: adds the seconds between construction and destruction to `acc`.
struct ScopedTimer {
    std::chrono::steady_clock::time_point t0;
    double& acc;
    explicit ScopedTimer(double& acc_) : t0(std::chrono::steady_clock::now()), acc(acc_) {}
    ~ScopedTimer() { acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};

// Why a factorization (of the PCG preconditioner or of the direct solver) was computed instead of
// reusing or updating the previous one. When several apply, the first in the order the SMW
// eligibility checks run wins (a forced rebuild, the method, the snapshot, rho, mu, the rank, the
// capacitance matrix); the event's mu/rho/k/w change flags keep the others.
enum class FactReason : int {
    Initial,             // first factorization of the run (direct mode: of the direct solver)
    MuRho,               // mu and/or rho changed since the last factorization (prec: the refactor_after-th change)
    RankExceeded,        // the active sets moved too far from the last factorization's for an SMW update (rank > 50)
    SingularCapacitance, // the SMW capacitance matrix was (near-)singular or non-finite
    PcgFailure,          // PCG failed through an SMW-updated or stale preconditioner: rebuilt in full and retried
    SmwSuppressed,       // SMW updates off after 5 failures in a row (until the next PMM iteration)
    NoSnapshot,          // nothing to update from (snapshot wiped after a fail streak, or invalidated)
    MethodSwitch,        // switch between Cholesky and LDLT (preconditioner callback, or direct system)
    DirectFallback,      // PCG failed for good: first factorization of the direct solver
    ActiveSetChange,     // direct solver without SMW updates (after a PCG failure, or SMW off): active sets changed
    SmwResidual,         // direct mode: an SMW-updated solve failed the residual check
    CholFailed,          // direct solver: Cholesky on S failed, so LDLT on K takes over
    Other,
    Count,               // number of reasons; also "none" where a reason is pending
};
constexpr int kNumFactReasons = static_cast<int>(FactReason::Count);

inline const char* to_string(FactReason r) {
    static const char* const names[kNumFactReasons] = {
        "initial", "mu_rho", "rank", "singular", "pcg_fail", "suppressed", "no_snapshot",
        "switch", "direct_fallback", "active_set", "smw_residual", "chol_failed", "other"};
    const int i = static_cast<int>(r);
    return (i >= 0 && i < kNumFactReasons) ? names[i] : "";
}

// Count column of each reason in result files: "fact_" + to_string(reason).
inline const char* fact_column(FactReason r) {
    static const char* const names[kNumFactReasons] = {
        "fact_initial", "fact_mu_rho", "fact_rank", "fact_singular", "fact_pcg_fail", "fact_suppressed",
        "fact_no_snapshot", "fact_switch", "fact_direct_fallback", "fact_active_set", "fact_smw_residual",
        "fact_chol_failed", "fact_other"};
    const int i = static_cast<int>(r);
    return (i >= 0 && i < kNumFactReasons) ? names[i] : "";
}

enum class LinTarget : int { Prec, Direct };  // PCG preconditioner, or direct solver
enum class LinAction : int {
    Factorize, // full factorization (see FactReason)
    SmwUpdate, // low-rank update of the last factorization
    SmwReuse,  // the last factorization reused as is (active sets back to its own)
    PcgFail,   // a PCG solve failed (see PcgFailKind)
    MuRhoKeep, // mu/rho changed; the preconditioner's factorization was kept (KSP_QP::refactor_after)
};
enum class PcgFailKind : int { None, PrecFailed, NoConvergence, BadAlloc };

inline const char* to_string(LinTarget t) { return t == LinTarget::Prec ? "prec" : "direct"; }
inline const char* to_string(LinAction a) {
    switch (a) {
        case LinAction::Factorize: return "factorize";
        case LinAction::SmwUpdate: return "smw_update";
        case LinAction::SmwReuse:  return "smw_reuse";
        case LinAction::PcgFail:   return "pcg_fail";
        case LinAction::MuRhoKeep: return "mu_rho_keep";
    }
    return "";
}
inline const char* to_string(PcgFailKind k) {
    switch (k) {
        case PcgFailKind::None:          return "";
        case PcgFailKind::PrecFailed:    return "prec_failed";
        case PcgFailKind::NoConvergence: return "no_convergence";
        case PcgFailKind::BadAlloc:      return "bad_alloc";
    }
    return "";
}

// The rule KSP_QP::update_PMM_parameters() applied to (mu, rho) after a PMM iteration.
enum class PmmRule : int {
    Double,   // SSN optimal: x2 (capped at mu_limit / rho_limit)
    Halve,    // line search failed: x0.5 (floored at mu0 / rho0)
    Increase, // too little progress: x1.1 (capped)
    Keep,     // enough progress: unchanged
    Stop,     // no update: the run terminated in this iteration
};
inline const char* to_string(PmmRule r) {
    switch (r) {
        case PmmRule::Double:   return "x2";
        case PmmRule::Halve:    return "x0.5";
        case PmmRule::Increase: return "x1.1";
        case PmmRule::Keep:     return "keep";
        case PmmRule::Stop:     return "stop";
    }
    return "";
}

// How an SSN iteration ended: with a step, or at the line search (no step taken).
enum class SsnLsOutcome : int { Step, AcceptOptimal, Fail };
inline const char* to_string(SsnLsOutcome o) {
    switch (o) {
        case SsnLsOutcome::Step:          return "step";
        case SsnLsOutcome::AcceptOptimal: return "accept_optimal";
        case SsnLsOutcome::Fail:          return "fail";
    }
    return "";
}

// Wall-clock seconds by phase. The first ten are disjoint and add up to the run time; the last four
// are informational parts of them.
struct SolvePhases {
    double setup = 0.0;        // KSP_QP constructor (scaling, factor of Q, ...)
    double prep = 0.0;         // SSN::prepare_newton_system(): active sets, H, G, RHS, factorization choice
    double fact = 0.0;         // full factorizations: assembly, symbolic analysis, numeric factorization
    double lowrank = 0.0;      // SMW update setups (preconditioner and direct solver), accepted or not
    double krylov = 0.0;       // PCG solves (main and refinement), preconditioner applications included
    double direct_solve = 0.0; // direct solver's triangular solves, plain or through an SMW update
    double linesearch = 0.0;   // exact line search and its steepest-descent retry
    double update = 0.0;       // iterate update, gradient and SSN termination check
    double ssn_other = 0.0;    // rest of the Newton solves: RHS, dx recovery, refinement residuals, bookkeeping
    double pmm_other = 0.0;    // rest of the solve: residual norms, multipliers, penalties, infeasibility checks

    double analyze = 0.0;        // symbolic analyses (part of fact)
    double prec_apply = 0.0;     // preconditioner applications (part of krylov)
    double prec_apply_smw = 0.0; // ... of which through an SMW update
    double refine = 0.0;         // iterative refinement in total (its solves are in krylov / direct_solve)

    double ssn_total() const {
        return prep + fact + lowrank + krylov + direct_solve + linesearch + update + ssn_other;
    }
};

// Per-run totals (always collected).
struct SolveProfile {
    std::array<int, kNumFactReasons> fact_by_reason{}; // factorizations by reason; they add up to fact
    int fact_analyzed = 0; // factorizations that included a symbolic analysis
    int smw_update = 0;    // Newton systems solved through an SMW update instead of a factorization (= smw_count)
    int smw_reuse = 0;     // ... or with the last factorization as is (active sets back to its own)
    int mu_rho_keep = 0;   // mu/rho changes absorbed without refactorizing the preconditioner
    int pcg_fail = 0;      // failed PCG solves (= krylov_fail)
    int krylov_refine = 0; // Krylov iterations spent in iterative refinement (part of krylov_iter)
    int ls_steepest = 0;   // line searches retried along steepest descent
    SolvePhases t;

    int fact_total() const {
        int n = 0;
        for (int c : fact_by_reason) n += c;
        return n;
    }

    // Columns as written to result files: counts, then t_* times (the first ten add up to run_time).
    template <typename F>
    void visit(F&& f) const {
        for (int i = 0; i < kNumFactReasons; ++i) f(fact_column(static_cast<FactReason>(i)), fact_by_reason[i]);
        f("fact_analyzed", fact_analyzed);
        f("smw_update", smw_update);
        f("smw_reuse", smw_reuse);
        f("mu_rho_keep", mu_rho_keep);
        f("pcg_fail", pcg_fail);
        f("krylov_refine", krylov_refine);
        f("ls_steepest", ls_steepest);
        f("t_setup", t.setup);
        f("t_prep", t.prep);
        f("t_fact", t.fact);
        f("t_lowrank", t.lowrank);
        f("t_krylov", t.krylov);
        f("t_direct_solve", t.direct_solve);
        f("t_linesearch", t.linesearch);
        f("t_update", t.update);
        f("t_ssn_other", t.ssn_other);
        f("t_pmm_other", t.pmm_other);
        f("t_analyze", t.analyze);
        f("t_prec_apply", t.prec_apply);
        f("t_prec_apply_smw", t.prec_apply_smw);
        f("t_refine", t.refine);
    }
};

// One linear-solver event: a factorization, an SMW update or reuse, a kept factorization, or a PCG failure.
template <typename T>
struct LinEvent {
    int pmm_iter = 0;                      // PMM iteration it happened in (1-based)
    int ssn_iter = 0;                      // SSN iteration, counted over the whole run (1-based)
    LinTarget target = LinTarget::Prec;
    LinAction action = LinAction::Factorize;
    FactReason reason = FactReason::Other; // Factorize only
    bool ldlt = false;                     // LDLT (on P_hat or K) rather than Cholesky (on P or S)
    bool analyzed = false;                 // Factorize: included a symbolic analysis
    int rank = -1, h = -1, p = -1, q = -1; // SMW delta against the last factorization (-1: not classified)
    int stale = 0;                         // prec: mu/rho changes its factorization lags behind, after the event
    T mu = T(0), rho = T(0);
    bool mu_changed = false, rho_changed = false; // since the last factorization or update (prec), or SSN iteration
    bool k_changed = false, w_changed = false;    // active sets changed since the previous SSN iteration
    int s = 0;                             // rows of G
    double t_assembly = 0.0, t_analyze = 0.0, t_factorize = 0.0, t_smw_setup = 0.0;
    PcgFailKind pcg_fail = PcgFailKind::None;
    int pcg_iters = -1;
    double pcg_err = std::numeric_limits<double>::quiet_NaN();

    template <typename F>
    void visit(F&& f) const {
        f("pmm_iter", pmm_iter);
        f("ssn_iter", ssn_iter);
        f("target", to_string(target));
        f("action", to_string(action));
        f("reason", action == LinAction::Factorize ? to_string(reason) : "");
        f("method", ldlt ? "ldlt" : "chol");
        f("analyzed", int(analyzed));
        f("rank", rank);
        f("h", h);
        f("p", p);
        f("q", q);
        f("stale", stale);
        f("mu", double(mu));
        f("rho", double(rho));
        f("mu_changed", int(mu_changed));
        f("rho_changed", int(rho_changed));
        f("k_changed", int(k_changed));
        f("w_changed", int(w_changed));
        f("s", s);
        f("t_assembly", t_assembly);
        f("t_analyze", t_analyze);
        f("t_factorize", t_factorize);
        f("t_smw_setup", t_smw_setup);
        f("pcg_fail", to_string(pcg_fail));
        f("pcg_iters", pcg_iters);
        f("pcg_err", pcg_err);
    }
};

// One SSN iteration, including one that ended at the line search without a step.
template <typename T>
struct SsnRow {
    int pmm_iter = 0, ssn_iter = 0; // as in LinEvent
    T mu = T(0), rho = T(0), ssn_tol = T(0);
    int n_act_K = 0, n_act_W = 0;
    int flips_K = -1, flips_W = -1; // active-set entries that changed since the previous SSN iteration (-1: first)
    int setup = -1;                 // the main solve's setup: a LinAction, or -1 if the factorization was reused as is
    FactReason reason = FactReason::Other;
    int rank = -1;
    int facts = 0, smw = 0;         // factorizations and SMW updates in this iteration (retries included)
    int krylov = 0, krylov_refine = 0;
    double pcg_err = std::numeric_limits<double>::quiet_NaN(); // main PCG solve's relative error
    SsnLsOutcome ls_outcome = SsnLsOutcome::Step;
    bool ls_steepest = false;
    T tau = T(0), grad_norm = T(0);
    double t_prep = 0.0, t_fact = 0.0, t_lowrank = 0.0, t_krylov = 0.0, t_direct_solve = 0.0,
           t_ssn_other = 0.0, t_linesearch = 0.0, t_update = 0.0, t_prec_apply = 0.0, t_refine = 0.0;

    double t_total() const {
        return t_prep + t_fact + t_lowrank + t_krylov + t_direct_solve + t_ssn_other + t_linesearch + t_update;
    }

    template <typename F>
    void visit(F&& f) const {
        f("pmm_iter", pmm_iter);
        f("ssn_iter", ssn_iter);
        f("mu", double(mu));
        f("rho", double(rho));
        f("ssn_tol", double(ssn_tol));
        f("n_act_K", n_act_K);
        f("n_act_W", n_act_W);
        f("flips_K", flips_K);
        f("flips_W", flips_W);
        f("setup", setup < 0 ? "reused" : to_string(static_cast<LinAction>(setup)));
        f("reason", setup == static_cast<int>(LinAction::Factorize) ? to_string(reason) : "");
        f("rank", rank);
        f("facts", facts);
        f("smw", smw);
        f("krylov", krylov);
        f("krylov_refine", krylov_refine);
        f("pcg_err", pcg_err);
        f("ls_outcome", to_string(ls_outcome));
        f("ls_steepest", int(ls_steepest));
        f("tau", double(tau));
        f("grad_norm", double(grad_norm));
        f("t_total", t_total());
        f("t_prep", t_prep);
        f("t_fact", t_fact);
        f("t_lowrank", t_lowrank);
        f("t_krylov", t_krylov);
        f("t_direct_solve", t_direct_solve);
        f("t_ssn_other", t_ssn_other);
        f("t_linesearch", t_linesearch);
        f("t_update", t_update);
        f("t_prec_apply", t_prec_apply);
        f("t_refine", t_refine);
    }
};

// One PMM iteration.
template <typename T>
struct PmmRow {
    int pmm_iter = 0;
    int ssn_iters = 0;       // SSN iterations taken (with a step)
    int ssn_status = -1;     // SSN<T>::TerminationStatus
    T mu = T(0), rho = T(0), ssn_tol = T(0);                // used in this iteration
    T mu_next = T(0), rho_next = T(0), ssn_tol_next = T(0); // after the update
    PmmRule rule = PmmRule::Stop;
    T res_primal = T(0), res_dual = T(0), res_comp_x = T(0), res_comp_Bx = T(0); // unscaled, after the iteration
    bool multipliers_updated = false;
    int facts = 0, smw = 0, krylov = 0;                     // in this iteration
    double t_ssn = 0.0, t_pmm_other = 0.0;

    template <typename F>
    void visit(F&& f) const {
        f("pmm_iter", pmm_iter);
        f("ssn_iters", ssn_iters);
        f("ssn_status", ssn_status);
        f("mu", double(mu));
        f("rho", double(rho));
        f("ssn_tol", double(ssn_tol));
        f("rule", to_string(rule));
        f("mu_next", double(mu_next));
        f("rho_next", double(rho_next));
        f("ssn_tol_next", double(ssn_tol_next));
        f("res_primal", double(res_primal));
        f("res_dual", double(res_dual));
        f("res_comp_x", double(res_comp_x));
        f("res_comp_Bx", double(res_comp_Bx));
        f("multipliers_updated", int(multipliers_updated));
        f("facts", facts);
        f("smw", smw);
        f("krylov", krylov);
        f("t_total", t_ssn + t_pmm_other);
        f("t_ssn", t_ssn);
        f("t_pmm_other", t_pmm_other);
    }
};

template <typename T>
struct SolveTrace {
    bool enabled = false; // set before KSP_QP::solve() to record the rows below
    std::vector<LinEvent<T>> events;
    std::vector<SsnRow<T>> ssn;
    std::vector<PmmRow<T>> pmm;

    void clear() {
        events.clear();
        ssn.clear();
        pmm.clear();
    }
};
