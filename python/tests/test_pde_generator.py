"""Unit tests for the PDE-constrained QP generators in python/pde_generator.py.

Run from the repository root with:
    python3 -m unittest discover -s python/tests -t python
"""

import unittest

import numpy as np
import scipy.sparse as sp

from pde_generator import (
    Discretization,
    GridQ1,
    apply_dirichlet_bc,
    apply_dirichlet_bc_mass,
    assemble_fd_cd,
    assemble_fd_diff,
    assemble_femq1_cd,
    assemble_femq1_diff,
    fem_boundary_nodes,
    generate_pde_l2_qp,
    is_boundary_node,
    make_convdiff_l2_control,
    make_poisson_l2_control,
    make_poisson_l2_state_control,
    make_problem_l2_from_mats,
    scale_by_mesh_size,
)

TIGHT = 1e-12
LOOSE = 1e-9   # for sums accumulated over many elements
INF = np.inf

# Single-element (nc=0) grid:
# node 0 = (0,0), node 1 = (1,0), node 2 = (0,1), node 3 = (1,1)
# in GridQ1's global order.
# The standard bilinear-Q1 unit-square local stiffness/mass matrices
# (verified by direct integration of the shape functions
# phi0=(1-x)(1-y), phi1=x(1-y), phi2=xy, phi3=(1-x)y over [0,1]^2)
# are permuted from local order (0,0),(1,0),(1,1),(0,1) into GridQ1's node numbering.
EXPECTED_A_STIFF_0 = np.array([
    [4.0 / 6, -1.0 / 6, -1.0 / 6, -2.0 / 6],
    [-1.0 / 6, 4.0 / 6, -2.0 / 6, -1.0 / 6],
    [-1.0 / 6, -2.0 / 6, 4.0 / 6, -1.0 / 6],
    [-2.0 / 6, -1.0 / 6, -1.0 / 6, 4.0 / 6],
])
EXPECTED_M_CONS_0 = np.array([
    [4.0 / 36, 2.0 / 36, 2.0 / 36, 1.0 / 36],
    [2.0 / 36, 4.0 / 36, 1.0 / 36, 2.0 / 36],
    [2.0 / 36, 1.0 / 36, 4.0 / 36, 2.0 / 36],
    [1.0 / 36, 2.0 / 36, 2.0 / 36, 4.0 / 36],
])


def dense(M):
    return np.asarray(M.todense())


def row_sums(M):
    return np.asarray(M @ np.ones(M.shape[1])).ravel()


# ===================== GridQ1 =====================

class TestGridQ1(unittest.TestCase):
    def test_single_element_grid_has_four_corner_nodes(self):
        g = GridQ1(0)
        self.assertEqual(g.n1d, 2)
        self.assertEqual(g.n_nodes, 4)
        self.assertEqual(g.nel1d, 1)
        self.assertEqual(g.nel, 1)
        np.testing.assert_allclose(g.x1d, [0.0, 1.0], atol=TIGHT)
        self.assertEqual(g.element_nodes(0, 0), [0, 1, 3, 2])

    def test_four_by_four_element_grid_has_expected_counts(self):
        g = GridQ1(2)  # n1d = 2^2 + 1 = 5
        self.assertEqual(g.n1d, 5)
        self.assertEqual(g.n_nodes, 25)
        self.assertEqual(g.nel1d, 4)
        self.assertEqual(g.nel, 16)
        np.testing.assert_allclose(g.x1d, [0.0, 0.25, 0.5, 0.75, 1.0], atol=TIGHT)

    def test_element_nodes_use_ccw_convention_matching_shape(self):
        g = GridQ1(2)  # n1d = 5, idx(i,j) = i + 5*j
        self.assertEqual(g.element_nodes(1, 1), [6, 7, 12, 11])

    def test_element_nodes_for_last_element_at_far_corner(self):
        g = GridQ1(2)  # nel1d = 4, valid ei,ej in [0,3]
        # (3,3)=18, (4,3)=19, (4,4)=24, (3,4)=23
        self.assertEqual(g.element_nodes(3, 3), [18, 19, 24, 23])

    def test_node_coords_follow_the_global_node_ordering(self):
        g = GridQ1(2)
        x, y = g.node_coords()
        for p in range(g.n_nodes):
            i, j = p % g.n1d, p // g.n1d
            self.assertAlmostEqual(x[p], g.x1d[i], delta=TIGHT)
            self.assertAlmostEqual(y[p], g.x1d[j], delta=TIGHT)


# ===================== boundary node collection =====================

class TestBoundaryNodes(unittest.TestCase):
    def test_is_boundary_node_flags_only_grid_edges(self):
        n1d = 3
        self.assertTrue(is_boundary_node(0, 0, n1d))
        self.assertTrue(is_boundary_node(2, 1, n1d))
        self.assertTrue(is_boundary_node(1, 2, n1d))
        self.assertFalse(is_boundary_node(1, 1, n1d))  # sole interior node

    def test_matches_four_times_n1d_minus_four_on_small_grid(self):
        g = GridQ1(1)  # n1d = 3, n_nodes = 9, one interior node
        bc = fem_boundary_nodes(g)
        self.assertEqual(bc.size, 8)  # all nodes except the single interior one

        interior = GridQ1.idx(1, 1, g.n1d)
        self.assertNotIn(interior, bc.tolist())
        self.assertEqual(sorted(bc.tolist()),
                         [p for p in range(g.n_nodes) if p != interior])

    def test_all_nodes_are_boundary_on_smallest_grid(self):
        g = GridQ1(0)  # n1d = 2, n_nodes = 4
        bc = fem_boundary_nodes(g)
        self.assertEqual(sorted(bc.tolist()), list(range(g.n_nodes)))


# ===================== assemble_femq1_diff =====================

class TestAssembleFemq1Diff(unittest.TestCase):
    def test_single_element_matches_hand_derived_unit_square_matrices(self):
        res = assemble_femq1_diff(GridQ1(0))
        np.testing.assert_allclose(dense(res.A_stiff), EXPECTED_A_STIFF_0, atol=TIGHT)
        np.testing.assert_allclose(dense(res.M_cons), EXPECTED_M_CONS_0, atol=TIGHT)
        self.assertEqual(res.f_rhs.size, 4)
        self.assertEqual(np.linalg.norm(res.f_rhs), 0.0)  # gauss_source is identically zero

    def test_stiffness_rows_sum_to_zero_and_mass_sums_to_domain_area(self):
        # grad(constant) = 0 (row sums of A_stiff vanish),
        # sum_ij M_ij = integral of (sum_i phi_i)^2 = integral of 1 = area(Omega).
        res = assemble_femq1_diff(GridQ1(2))  # 4x4 elements
        np.testing.assert_allclose(row_sums(res.A_stiff), 0.0, atol=LOOSE)
        self.assertAlmostEqual(res.M_cons.sum(), 1.0, delta=LOOSE)  # unit square area

    def test_lumped_mass_sums_to_domain_area_and_is_diagonal(self):
        res = assemble_femq1_diff(GridQ1(2))
        self.assertAlmostEqual(res.M_lump.sum(), 1.0, delta=LOOSE)
        off_diagonal = res.M_lump - sp.diags(res.M_lump.diagonal())
        self.assertAlmostEqual(abs(off_diagonal).max() if off_diagonal.nnz else 0.0,
                               0.0, delta=TIGHT)


# ===================== assemble_femq1_cd =====================

class TestAssembleFemq1Cd(unittest.TestCase):
    def test_diffusion_and_mass_blocks_match_assemble_femq1_diff(self):
        g = GridQ1(1)
        diff_res = assemble_femq1_diff(g)
        cd_res = assemble_femq1_cd(g)
        np.testing.assert_allclose(dense(cd_res.A_stiff), dense(diff_res.A_stiff), atol=TIGHT)
        np.testing.assert_allclose(dense(cd_res.M_cons), dense(diff_res.M_cons), atol=TIGHT)

    def test_convection_rows_sum_to_zero(self):
        # N_ij = integral phi_i (w . grad phi_j); summing over j gives
        # integral phi_i * w . grad(sum_j phi_j) = integral phi_i * w . grad(1) = 0,
        # for any wind field, independent of mesh resolution.
        res = assemble_femq1_cd(GridQ1(2))
        np.testing.assert_allclose(row_sums(res.N_conv), 0.0, atol=LOOSE)


# ===================== assemble_fd_diff / assemble_fd_cd =====================

class TestAssembleFdDiff(unittest.TestCase):
    def test_interior_stencil_matches_five_point_laplacian_on_small_grid(self):
        g = GridQ1(1)  # n1d = 3, h = 0.5
        res = assemble_fd_diff(g)
        h = g.x1d[1] - g.x1d[0]
        inv_h2 = 1.0 / (h * h)
        A = dense(res.A_stiff)
        p = GridQ1.idx(1, 1, g.n1d)  # sole interior node

        self.assertAlmostEqual(A[p, p], 4.0 * inv_h2, delta=TIGHT)
        for neighbour in [(0, 1), (2, 1), (1, 0), (1, 2)]:
            q = GridQ1.idx(neighbour[0], neighbour[1], g.n1d)
            self.assertAlmostEqual(A[p, q], -inv_h2, delta=TIGHT)
        self.assertAlmostEqual(dense(res.M_lump)[p, p], h * h, delta=TIGHT)

        # Boundary rows are left empty; apply_dirichlet_bc fills them in later.
        for bp in fem_boundary_nodes(g):
            np.testing.assert_allclose(A[bp, :], 0.0, atol=TIGHT)

        self.assertEqual(res.f_rhs.size, g.n_nodes)
        self.assertEqual(np.linalg.norm(res.f_rhs), 0.0)

    def test_mass_diagonal_uses_trapezoidal_weights_and_sums_to_domain_area(self):
        g = GridQ1(2)  # 4x4 elements, n1d = 5
        res = assemble_fd_diff(g)
        h = g.x1d[1] - g.x1d[0]
        M = dense(res.M_lump)

        # Interior node: full weight h^2.
        interior = GridQ1.idx(2, 2, g.n1d)
        self.assertAlmostEqual(M[interior, interior], h * h, delta=TIGHT)
        # Edge (non-corner) boundary node: half weight.
        edge = GridQ1.idx(2, 0, g.n1d)
        self.assertAlmostEqual(M[edge, edge], 0.5 * h * h, delta=TIGHT)
        # Corner node: quarter weight.
        corner = GridQ1.idx(0, 0, g.n1d)
        self.assertAlmostEqual(M[corner, corner], 0.25 * h * h, delta=TIGHT)

        # unit square area (trapezoidal rule is exact here)
        self.assertAlmostEqual(res.M_lump.sum(), 1.0, delta=LOOSE)


class TestAssembleFdCd(unittest.TestCase):
    def test_diffusion_and_mass_blocks_match_assemble_fd_diff(self):
        g = GridQ1(1)
        diff_res = assemble_fd_diff(g)
        cd_res = assemble_fd_cd(g)
        np.testing.assert_allclose(dense(cd_res.A_stiff), dense(diff_res.A_stiff), atol=TIGHT)
        np.testing.assert_allclose(dense(cd_res.M_lump), dense(diff_res.M_lump), atol=TIGHT)

    def test_convection_rows_sum_to_zero(self):
        # Telescoping upwind stencil: row sum
        # = (wx_p+wx_m+wy_p+wy_m) - wx_p - wx_m - wy_p - wy_m = 0
        # at every interior node, for any wind field.
        res = assemble_fd_cd(GridQ1(2))
        np.testing.assert_allclose(row_sums(res.N_conv), 0.0, atol=LOOSE)


# ===================== apply_dirichlet_bc / apply_dirichlet_bc_mass =====================

class TestApplyDirichletBc(unittest.TestCase):
    def test_folds_boundary_column_into_rhs_and_sets_identity_row(self):
        D = sp.csc_matrix(EXPECTED_A_STIFF_0)
        rhs = np.array([1.0, 2.0, 3.0, 4.0])

        D, rhs = apply_dirichlet_bc(D, rhs, [0], np.array([5.0]))

        # rhs_r <- rhs_r - D_orig(r,0) * 5, for r != 0
        self.assertAlmostEqual(rhs[0], 5.0, delta=TIGHT)
        for r in (1, 2, 3):
            self.assertAlmostEqual(rhs[r], (r + 1) - EXPECTED_A_STIFF_0[r][0] * 5.0, delta=TIGHT)

        # Row/col 0 becomes an identity row; the interior 3x3 block is untouched.
        Dd = dense(D)
        np.testing.assert_allclose(Dd[0, :], [1.0, 0.0, 0.0, 0.0], atol=TIGHT)
        np.testing.assert_allclose(Dd[:, 0], [1.0, 0.0, 0.0, 0.0], atol=TIGHT)
        np.testing.assert_allclose(Dd[1:, 1:], EXPECTED_A_STIFF_0[1:, 1:], atol=TIGHT)

    def test_empty_bc_nodes_is_a_no_op(self):
        D = sp.csc_matrix(EXPECTED_A_STIFF_0)
        rhs = np.array([1.0, 2.0, 3.0, 4.0])

        D, rhs_out = apply_dirichlet_bc(D, rhs, [], np.zeros(0))

        np.testing.assert_allclose(rhs_out, rhs, atol=TIGHT)
        np.testing.assert_allclose(dense(D), EXPECTED_A_STIFF_0, atol=TIGHT)

    def test_all_nodes_as_boundary_produces_identity(self):
        D = sp.csc_matrix(EXPECTED_A_STIFF_0)
        rhs = np.array([1.0, 2.0, 3.0, 4.0])
        bc_values = np.array([10.0, 20.0, 30.0, 40.0])

        D, rhs = apply_dirichlet_bc(D, rhs, [0, 1, 2, 3], bc_values)

        np.testing.assert_allclose(rhs, bc_values, atol=TIGHT)
        np.testing.assert_allclose(dense(D), np.eye(4), atol=TIGHT)

    def test_does_not_mutate_its_inputs(self):
        D = sp.csc_matrix(EXPECTED_A_STIFF_0)
        rhs = np.array([1.0, 2.0, 3.0, 4.0])
        apply_dirichlet_bc(D, rhs, [0], np.array([5.0]))
        np.testing.assert_allclose(dense(D), EXPECTED_A_STIFF_0, atol=TIGHT)
        np.testing.assert_allclose(rhs, [1.0, 2.0, 3.0, 4.0], atol=TIGHT)


class TestApplyDirichletBcMass(unittest.TestCase):
    def test_zeroes_boundary_rows_and_columns_entirely(self):
        M = dense(apply_dirichlet_bc_mass(sp.csc_matrix(EXPECTED_M_CONS_0), [0, 2]))

        for j in range(4):
            self.assertAlmostEqual(M[0, j], 0.0, delta=TIGHT)
            self.assertAlmostEqual(M[2, j], 0.0, delta=TIGHT)
            self.assertAlmostEqual(M[j, 0], 0.0, delta=TIGHT)
            self.assertAlmostEqual(M[j, 2], 0.0, delta=TIGHT)
        # Interior rows/cols {1,3} are untouched.
        for i in (1, 3):
            for j in (1, 3):
                self.assertAlmostEqual(M[i, j], EXPECTED_M_CONS_0[i][j], delta=TIGHT)

    def test_empty_bc_nodes_is_a_no_op(self):
        M = apply_dirichlet_bc_mass(sp.csc_matrix(EXPECTED_M_CONS_0), [])
        np.testing.assert_allclose(dense(M), EXPECTED_M_CONS_0, atol=TIGHT)

    def test_all_nodes_as_boundary_produces_zero_matrix(self):
        M = apply_dirichlet_bc_mass(sp.csc_matrix(EXPECTED_M_CONS_0), [0, 1, 2, 3])
        np.testing.assert_allclose(dense(M), np.zeros((4, 4)), atol=TIGHT)


# ===================== make_problem_l2_from_mats =====================

class TestMakeProblemL2FromMats(unittest.TestCase):
    def test_matches_hand_derived_two_node_example(self):
        # n_nodes=2, D_op = diag(2,3), M = I (a lumped mass matrix is diagonal).
        D = sp.csc_matrix(np.diag([2.0, 3.0]))
        M = sp.csc_matrix(np.eye(2))
        rhs = np.array([5.0, 7.0])
        yhat = np.array([1.0, 2.0])
        beta = 4.0

        pb = make_problem_l2_from_mats(D, M, rhs, yhat, beta, -1.0, 1.0, -2.0, 2.0)

        self.assertEqual(pb.n, 4)
        self.assertEqual(pb.m, 2)
        self.assertEqual(pb.l, 0)
        self.assertAlmostEqual(pb.obj_const, 2.5, delta=TIGHT)  # 0.5 * (1*1 + 2*2)

        np.testing.assert_allclose(pb.c, [-1.0, -2.0, 0.0, 0.0], atol=TIGHT)
        np.testing.assert_allclose(dense(pb.Q), np.diag([1.0, 1.0, beta, beta]), atol=TIGHT)
        # A = [D_op, -M]
        np.testing.assert_allclose(dense(pb.A), [[2, 0, -1, 0], [0, 3, 0, -1]], atol=TIGHT)
        np.testing.assert_allclose(pb.b, rhs, atol=TIGHT)

        self.assertEqual(pb.B.shape[0], 0)
        self.assertEqual(pb.lw.size, 0)
        self.assertEqual(pb.uw.size, 0)

        np.testing.assert_allclose(pb.lx, [-1.0, -1.0, -2.0, -2.0], atol=TIGHT)
        np.testing.assert_allclose(pb.ux, [1.0, 1.0, 2.0, 2.0], atol=TIGHT)

    def test_handles_non_symmetric_mass_matrix(self):
        D = sp.csc_matrix(np.diag([2.0, 3.0]))
        M = sp.csc_matrix(np.array([[2.0, 1.0], [3.0, 4.0]]))
        rhs = np.array([5.0, 7.0])
        yhat = np.array([1.0, 2.0])
        beta = 4.0

        pb = make_problem_l2_from_mats(D, M, rhs, yhat, beta)

        # Myhat = M*yhat = [2*1+1*2, 3*1+4*2] = [4, 11]
        self.assertAlmostEqual(pb.obj_const, 13.0, delta=TIGHT)  # 0.5*(1*4 + 2*11)
        np.testing.assert_allclose(pb.c, [-4.0, -11.0, 0.0, 0.0], atol=TIGHT)

        # y-y block is M verbatim (row i, col j preserved); u-u block is beta*M.
        expected_Q = np.array([
            [2, 1, 0, 0],
            [3, 4, 0, 0],
            [0, 0, beta * 2, beta * 1],
            [0, 0, beta * 3, beta * 4],
        ])
        np.testing.assert_allclose(dense(pb.Q), expected_Q, atol=TIGHT)

        # A = [D_op, -M]: column j of the -M block holds -M(row, j), not -M(j, row).
        np.testing.assert_allclose(dense(pb.A), [[2, 0, -2, -1], [0, 3, -3, -4]], atol=TIGHT)

    def test_default_bounds_are_free(self):
        D = sp.csc_matrix(np.diag([2.0, 3.0]))
        M = sp.csc_matrix(np.eye(2))
        pb = make_problem_l2_from_mats(D, M, np.zeros(2), np.zeros(2), 1.0)
        np.testing.assert_array_equal(pb.lx, np.full(4, -INF))
        np.testing.assert_array_equal(pb.ux, np.full(4, INF))


# ===================== scale_by_mesh_size =====================

class TestScaleByMeshSize(unittest.TestCase):
    def test_divides_objective_and_interior_rows_by_h_squared(self):
        # Node 0 is a Dirichlet node: identity row/col in D, zero row/col in M,
        # as left by apply_dirichlet_bc / apply_dirichlet_bc_mass.
        D = sp.csc_matrix(np.array([[1.0, 0.0, 0.0],
                                    [0.0, 2.0, -1.0],
                                    [0.0, -1.0, 3.0]]))
        M = sp.csc_matrix(np.diag([0.0, 1.0, 2.0]))
        pb = make_problem_l2_from_mats(D, M, np.array([5.0, 7.0, 11.0]),
                                       np.array([1.0, 2.0, 3.0]), 4.0,
                                       -1.0, 1.0, -2.0, 2.0)
        Q0, A0 = dense(pb.Q), dense(pb.A)
        c0, b0, obj0 = pb.c.copy(), pb.b.copy(), pb.obj_const

        scaled = scale_by_mesh_size(pb, [0], 0.5)  # 1/h^2 = 4

        # 1/h^2 is a power of two, so every comparison below is exact.
        np.testing.assert_array_equal(dense(scaled.Q), 4.0 * Q0)
        np.testing.assert_array_equal(scaled.c, 4.0 * c0)
        self.assertEqual(scaled.obj_const, 4.0 * obj0)
        # Dirichlet row 0 is untouched; interior rows 1, 2 are multiplied by 4.
        np.testing.assert_array_equal(dense(scaled.A)[0], A0[0])
        np.testing.assert_array_equal(dense(scaled.A)[1:], 4.0 * A0[1:])
        np.testing.assert_array_equal(scaled.b, [5.0, 28.0, 44.0])

        self.assertEqual((scaled.n, scaled.m, scaled.l), (pb.n, pb.m, pb.l))
        self.assertEqual(scaled.B.shape, pb.B.shape)
        for name in ("lx", "ux", "lw", "uw"):
            np.testing.assert_array_equal(getattr(scaled, name), getattr(pb, name))

        # The input is not modified in place.
        np.testing.assert_array_equal(dense(pb.Q), Q0)
        np.testing.assert_array_equal(dense(pb.A), A0)
        np.testing.assert_array_equal(pb.c, c0)
        np.testing.assert_array_equal(pb.b, b0)
        self.assertEqual(pb.obj_const, obj0)


# ===================== QP generators (integration smoke tests) =====================

class GeneratorTestCase(unittest.TestCase):
    def assert_boundary_rows_are_identity(self, pb, g, bc_nodes):
        """After Dirichlet elimination, each boundary node's row in A must be
        an identity row on its own y-column and zero everywhere else."""
        A = dense(pb.A)
        for p in bc_nodes:
            expected = np.zeros(pb.n)
            expected[p] = 1.0
            np.testing.assert_allclose(A[p, :], expected, atol=TIGHT,
                                       err_msg=f"boundary row {p}")
        interior = GridQ1.idx(1, 1, g.n1d)
        self.assertNotEqual(A[interior, interior], 0.0,
                            "interior diagonal should retain D_op")


class TestMakePoissonL2Control(GeneratorTestCase):
    def test_dimensions_and_zero_boundary_values(self):
        g = GridQ1(1)  # n_nodes = 9
        bc_nodes = fem_boundary_nodes(g)
        pb = make_poisson_l2_control(1, 4.0)

        self.assertEqual(pb.n, 18)
        self.assertEqual(pb.m, 9)
        self.assertEqual(pb.l, 0)
        self.assert_boundary_rows_are_identity(pb, g, bc_nodes)
        np.testing.assert_allclose(pb.b[bc_nodes], 0.0, atol=TIGHT)

    def test_fd_matches_shape_and_bc_of_fem_but_uses_a_different_operator(self):
        g = GridQ1(1)
        bc_nodes = fem_boundary_nodes(g)
        pb_fem = make_poisson_l2_control(1, 4.0)
        pb_fd = make_poisson_l2_control(1, 4.0, disc=Discretization.FD)

        self.assertEqual((pb_fd.n, pb_fd.m, pb_fd.l), (pb_fem.n, pb_fem.m, pb_fem.l))
        self.assertGreater(abs(dense(pb_fd.A) - dense(pb_fem.A)).max(), TIGHT)
        self.assert_boundary_rows_are_identity(pb_fd, g, bc_nodes)
        np.testing.assert_allclose(pb_fd.b[bc_nodes], 0.0, atol=TIGHT)

    def test_fd_ignores_lump_mass_flag_and_always_uses_lumped_mass(self):
        lumped = make_poisson_l2_control(1, 4.0, lump_mass=True, disc=Discretization.FD)
        consistent = make_poisson_l2_control(1, 4.0, lump_mass=False, disc=Discretization.FD)
        np.testing.assert_allclose(dense(lumped.Q), dense(consistent.Q), atol=TIGHT)
        np.testing.assert_allclose(dense(lumped.A), dense(consistent.A), atol=TIGHT)


class TestMakePoissonL2StateControl(GeneratorTestCase):
    def _expected_yhat(self, g):
        x, y = g.node_coords()
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def test_boundary_values_match_yhat(self):
        g = GridQ1(1)
        bc_nodes = fem_boundary_nodes(g)
        pb = make_poisson_l2_state_control(1, 4.0)

        self.assert_boundary_rows_are_identity(pb, g, bc_nodes)
        np.testing.assert_allclose(pb.b[bc_nodes], self._expected_yhat(g)[bc_nodes], atol=TIGHT)

    def test_fd_boundary_values_still_match_yhat(self):
        g = GridQ1(1)
        bc_nodes = fem_boundary_nodes(g)
        pb_fd = make_poisson_l2_state_control(1, 4.0, disc=Discretization.FD)
        np.testing.assert_allclose(pb_fd.b[bc_nodes], self._expected_yhat(g)[bc_nodes], atol=TIGHT)


class TestMakeConvdiffL2Control(GeneratorTestCase):
    def test_dimensions_and_zero_boundary_values(self):
        g = GridQ1(1)
        bc_nodes = fem_boundary_nodes(g)
        pb = make_convdiff_l2_control(1, 4.0)

        self.assertEqual(pb.n, 18)
        self.assertEqual(pb.m, 9)
        self.assert_boundary_rows_are_identity(pb, g, bc_nodes)
        np.testing.assert_allclose(pb.b[bc_nodes], 0.0, atol=TIGHT)

    def test_fd_matches_shape_and_bc_of_fem_but_uses_a_different_operator(self):
        g = GridQ1(1)
        bc_nodes = fem_boundary_nodes(g)
        pb_fem = make_convdiff_l2_control(1, 4.0)
        pb_fd = make_convdiff_l2_control(1, 4.0, eps=0.01, disc=Discretization.FD)

        self.assertEqual((pb_fd.n, pb_fd.m), (pb_fem.n, pb_fem.m))
        self.assertGreater(abs(dense(pb_fd.A) - dense(pb_fem.A)).max(), TIGHT)
        self.assert_boundary_rows_are_identity(pb_fd, g, bc_nodes)
        np.testing.assert_allclose(pb_fd.b[bc_nodes], 0.0, atol=TIGHT)

    def test_eps_scales_only_the_diffusion_part_of_the_operator(self):
        # D_op = eps*A_stiff + N_conv and interior rows are divided by h^2
        # (scale_by_mesh_size), so D(eps1) - D(eps2) = (eps1-eps2)*A_stiff/h^2 there.
        g = GridQ1(2)
        n = g.n_nodes
        h = g.x1d[1] - g.x1d[0]
        pb1 = make_convdiff_l2_control(2, 1.0, eps=0.10)
        pb2 = make_convdiff_l2_control(2, 1.0, eps=0.05)
        diff = dense(pb1.A)[:, :n] - dense(pb2.A)[:, :n]

        stiff = dense(assemble_femq1_diff(g).A_stiff)
        bc_nodes = fem_boundary_nodes(g)
        interior = np.setdiff1d(np.arange(n), bc_nodes)
        np.testing.assert_allclose(diff[np.ix_(interior, interior)],
                                   0.05 * stiff[np.ix_(interior, interior)] / h**2, atol=TIGHT)


class TestGeneratorsAreMeshNormalized(GeneratorTestCase):
    def test_lumped_mass_objective_is_the_same_on_every_mesh(self):
        # M_lump is h^2 on the interior diagonal (FEM and FD alike) and zero on
        # the boundary, so after scale_by_mesh_size Q = blkdiag(I, beta*I) and
        # c = -yhat on the interior nodes, independently of nc.
        beta = 1e-2
        for disc in Discretization:
            for nc in (2, 3, 4):
                with self.subTest(disc=disc.value, nc=nc):
                    g = GridQ1(nc)
                    n = g.n_nodes
                    bc_nodes = fem_boundary_nodes(g)
                    interior = np.setdiff1d(np.arange(n), bc_nodes)
                    x, y = g.node_coords()
                    yhat = np.exp(-64.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
                    pb = make_poisson_l2_control(nc, beta, lump_mass=True, disc=disc)

                    q = pb.Q.diagonal()
                    np.testing.assert_allclose(q[interior], 1.0, rtol=TIGHT)
                    np.testing.assert_allclose(q[n + interior], beta, rtol=TIGHT)
                    np.testing.assert_allclose(pb.c[interior], -yhat[interior], rtol=TIGHT)
                    self.assert_boundary_rows_are_identity(pb, g, bc_nodes)

    def test_consistent_mass_objective_is_the_same_on_every_mesh(self):
        # M_cons is h^2 times a fixed stencil (interior diagonal 4h^2/9), so
        # after scale_by_mesh_size its interior diagonal is 4/9 on every mesh.
        for nc in (2, 3, 4):
            with self.subTest(nc=nc):
                g = GridQ1(nc)
                interior = np.setdiff1d(np.arange(g.n_nodes), fem_boundary_nodes(g))
                pb = make_poisson_l2_control(nc, 1e-2)
                np.testing.assert_allclose(pb.Q.diagonal()[interior], 4.0 / 9.0, rtol=TIGHT)


# ===================== generate_pde_l2_qp (dict façade) =====================

class TestGeneratePdeL2Qp(unittest.TestCase):
    def test_returns_the_binding_dict_format(self):
        d = generate_pde_l2_qp("poisson", 2, 1e-2, u_lower=0.0, u_upper=1.0)
        expected_keys = {"n", "m", "l", "obj_const", "c", "b", "lx", "ux", "lw", "uw"}
        expected_keys |= {f"{k}_{s}" for k in "QAB"
                          for s in ("data", "indices", "indptr", "shape")}
        self.assertEqual(set(d), expected_keys)

        n1d = 2 ** 2 + 1
        self.assertEqual(d["n"], 2 * n1d ** 2)
        self.assertEqual(d["m"], n1d ** 2)
        self.assertEqual(d["l"], 0)
        self.assertEqual(d["Q_shape"], (d["n"], d["n"]))
        self.assertEqual(d["A_shape"], (d["m"], d["n"]))
        self.assertEqual(d["B_shape"], (0, d["n"]))
        # Index arrays must be int32: the binding casts them to C int.
        self.assertEqual(d["Q_indices"].dtype, np.int32)
        self.assertEqual(d["Q_indptr"].dtype, np.int32)

    def test_reconstructed_matrices_match_the_dataclass(self):
        d = generate_pde_l2_qp("convdiff", 2, 1e-2, eps=0.02)
        pb = make_convdiff_l2_control(2, 1e-2, eps=0.02)
        Q = sp.csc_matrix((d["Q_data"], d["Q_indices"], d["Q_indptr"]), shape=d["Q_shape"])
        A = sp.csc_matrix((d["A_data"], d["A_indices"], d["A_indptr"]), shape=d["A_shape"])
        np.testing.assert_allclose(dense(Q), dense(pb.Q), atol=TIGHT)
        np.testing.assert_allclose(dense(A), dense(pb.A), atol=TIGHT)

    def test_rejects_unknown_choice_and_discretization(self):
        with self.assertRaises(ValueError):
            generate_pde_l2_qp("laplace", 2, 1e-2)
        with self.assertRaises(ValueError):
            generate_pde_l2_qp("poisson", 2, 1e-2, discretization="spectral")

    def test_discretization_string_is_case_insensitive(self):
        lower = generate_pde_l2_qp("poisson", 2, 1e-2, discretization="fd")
        upper = generate_pde_l2_qp("poisson", 2, 1e-2, discretization="FD")
        np.testing.assert_allclose(lower["A_data"], upper["A_data"], atol=TIGHT)


if __name__ == "__main__":
    unittest.main()
