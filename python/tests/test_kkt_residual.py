"""End-to-end KKT residual checks on the PDE-constrained QPs.

Independently re-derives the four KKT residual blocks of the QP

    min  c^T x + 0.5 x^T Q x + obj_const,  s.t. A x = b,  B x = w,
         lx <= x <= ux,  lw <= w <= uw

from the generated problem data and the solver's returned (x, y1, y2, z) --
a true outside-in regression check. Both are in original, unscaled units, so
they are directly comparable.

Requires the ksp_qp_bind extension; build it with:
    cd python && mkdir -p build && cd build
    cmake .. -DPython3_EXECUTABLE=$(which python3) && cmake --build . --config Release

Run from the repository root with:
    python3 -m unittest discover -s python/tests -t python
"""

import unittest

import numpy as np
import scipy.sparse as sp

from pde_generator import (
    Discretization,
    make_convdiff_l2_control,
    make_poisson_l2_control,
    make_poisson_l2_state_control,
)

try:
    import ksp_qp_bind
except ModuleNotFoundError:  # pragma: no cover - exercised only without the build
    ksp_qp_bind = None

SOLVER_TOL = 1e-7   # passed to the solver
KKT_TOL = 1e-6      # residual margin tolerance
TIME_LIMIT = 60.0
MAX_ITER = 3000
OPTIMAL = 0         # TerminationStatus::Optimal
INF = np.inf


@unittest.skipIf(ksp_qp_bind is None, "ksp_qp_bind is not built")
class KktResidualTestCase(unittest.TestCase):
    def solve_and_check_kkt(self, pb, tol=KKT_TOL, solver_tol=SOLVER_TOL):
        res = ksp_qp_bind.solve_from_data(pb.to_dict(), solver_tol, MAX_ITER, TIME_LIMIT)
        self.assertEqual(res["status"], OPTIMAL)

        x = np.asarray(res["x"])
        y1 = np.asarray(res["y1"])
        y2 = np.asarray(res["y2"])
        z = np.asarray(res["z"])
        self.assertEqual(x.size, pb.n)

        # 1. Stationarity / dual residual: c + Qx - A^T y1 - B^T y2 + z.
        r_dual = pb.Q @ x + pb.c + z
        if pb.m > 0:
            r_dual = r_dual - pb.A.T @ y1
        if pb.l > 0:
            r_dual = r_dual - pb.B.T @ y2
        self.assertLess(np.abs(r_dual).max(), tol, "dual/stationarity residual too large")

        # 2. Primal equality residual: A x - b.
        if pb.m > 0:
            r_primal = pb.A @ x - pb.b
            self.assertLess(np.abs(r_primal).max(), tol, "primal equality residual too large")

        # 3. Box complementarity on x: x - proj(x + z, lx, ux).
        r_box_x = x - np.clip(x + z, pb.lx, pb.ux)
        self.assertLess(np.abs(r_box_x).max(), tol,
                        "x-box complementarity residual too large")

        # 4. Box complementarity on Bx: Bx - proj(Bx - y2, lw, uw).
        if pb.l > 0:
            Bx = pb.B @ x
            r_box_w = Bx - np.clip(Bx - y2, pb.lw, pb.uw)
            self.assertLess(np.abs(r_box_w).max(), tol,
                            "Bx-box complementarity residual too large")
        return x


class TestKktResidualFem(KktResidualTestCase):
    def test_poisson_l2_control_satisfies_kkt(self):
        self.solve_and_check_kkt(make_poisson_l2_control(2, 1e-2))

    def test_poisson_l2_state_control_satisfies_kkt(self):
        self.solve_and_check_kkt(make_poisson_l2_state_control(2, 1.0, -0.1, 0.002))

    def test_convdiff_l2_control_satisfies_kkt(self):
        self.solve_and_check_kkt(make_convdiff_l2_control(2, 0.1, 0.0, 0.2, -0.75, 0.75))


# ===================== Discretization.FD =====================
# The FEM cases above with the FD discretization. FD's operator is mass-scaled
# (assemble_*_by_discretization) so these reach the same KKT tolerance; see
# test_poisson_l2_control_fd_is_not_degenerate below.

class TestKktResidualFd(KktResidualTestCase):
    def test_poisson_l2_control_satisfies_kkt(self):
        self.solve_and_check_kkt(
            make_poisson_l2_control(2, 1e-2, disc=Discretization.FD))

    def test_poisson_l2_state_control_satisfies_kkt(self):
        self.solve_and_check_kkt(
            make_poisson_l2_state_control(2, 1.0, -0.1, 0.002, disc=Discretization.FD))

    def test_convdiff_l2_control_satisfies_kkt(self):
        self.solve_and_check_kkt(
            make_convdiff_l2_control(2, 0.1, 0.0, 0.2, -0.75, 0.75, eps=0.01,
                                     disc=Discretization.FD))

    def test_poisson_l2_control_fd_is_not_degenerate(self):
        # With a raw FD stiffness in D_op*y = M*u, the O(1/h^2) operator and the
        # O(h^2) control mass make the state's response to control vanish as O(h^4),
        # so x=0 is numerically KKT-optimal. Assert the control uses a meaningful
        # fraction of its box instead of sitting at its lower bound.
        #
        # beta = 1e-6 is badly conditioned: with the mesh-size normalization, KSP-QP
        # bottoms out at a relative residual of ~2e-7, just above SOLVER_TOL, so this
        # test solves at 1e-6.
        u_upper = 300.0
        pb = make_poisson_l2_control(5, 1e-6, u_lower=0.0, u_upper=u_upper,
                                     disc=Discretization.FD)
        x = self.solve_and_check_kkt(pb, solver_tol=1e-6)

        n_nodes = pb.n // 2
        u_max = x[n_nodes:].max()
        # The degenerate bug produced u_max on the order of 1e-9 to 1e-10; a
        # correctly-scaled solve uses a large fraction of the [0, 300] box.
        self.assertGreater(u_max, 1.0,
                           "control collapsed near zero -- FD state/control coupling regressed")


if __name__ == "__main__":
    unittest.main()
