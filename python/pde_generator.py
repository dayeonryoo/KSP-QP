"""
Q1 bilinear finite-element assembly and PDE-constrained QP generation.

The L2-regularized PDE-constrained QPs produced here follow

  J. W. Pearson and J. Gondzio, "Fast interior point solution of quadratic
  programming problems arising from PDE-constrained optimization",
  Numerische Mathematik, 2017.

The global assembly and boundary-condition routines are Python translations
of the corresponding IFISS MATLAB functions:
  femq1_diff.m        - IFISS function: DJS; 4 March 2005
  femq1_cd.m          - IFISS function: DJS; 5 March 2005
Copyright (c) 2005 D.J. Silvester, H.C. Elman, A. Ramage

  nonzerobc_input.m  - IFISS function: DJS, JWP; 27 June 2012
Copyright (c) 2012 D.J. Silvester, H.C. Elman, A. Ramage, J.W. Pearson

See also fem_q1.py for the element-level kernels (shape/deriv/gauss_*)
and its own IFISS citations.

The generated problems are returned as KSPQPdata instances; call .to_dict()
to get the numpy/CSC dict that ksp_qp_bind.solve_from_data() and
benchmark_common.kspqp_to_qpalm() consume.
"""

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
import scipy.sparse as sp

import fem_q1 as fem

INF = np.inf


class Discretization(str, Enum):
    """Spatial discretization used to build the PDE operator D_op and mass
    matrix M in the QP generators below. FEM is the default; FD uses a
    standard 5-point Laplacian stiffness with first-order upwind convection
    on the same uniform GridQ1 node layout, so both share Dirichlet BC
    handling (apply_dirichlet_bc / apply_dirichlet_bc_mass) and only ever
    produce a lumped mass matrix.
    """

    FEM = "fem"
    FD = "fd"

    @classmethod
    def parse(cls, value):
        """Accepts a Discretization or a case-insensitive 'fem'/'fd' string."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            raise ValueError(f"discretization must be 'fem' or 'fd', got: {value!r}") from None


def is_boundary_node(i, j, n1d):
    """Vectorised over i/j: True on the four edges of the node grid."""
    return (i == 0) | (j == 0) | (i == n1d - 1) | (j == n1d - 1)


def fd_trapezoid_factor(idx, n1d):
    """Composite trapezoidal-rule weight along one axis: edge nodes get half
    the interior weight so that the resulting tensor-product nodal mass sums
    exactly to the domain area (used by the FD lumped mass below).
    """
    return np.where((idx == 0) | (idx == n1d - 1), 0.5, 1.0)


# -----------------------------------------------------------------------
# Q1 finite-element mesh on a uniform rectangular tensor-product grid over
# the unit square. Element connectivity is generated directly from (ei,ej)
# since we assume the mesh is always structured.
#
# n_nodes is the C++ GridQ1::np field, renamed to avoid colliding with the
# conventional numpy alias.
# -----------------------------------------------------------------------

class GridQ1:
    def __init__(self, nc):
        self.nc = nc
        self.n1d = (1 << nc) + 1        # nodes per direction
        self.n_nodes = self.n1d * self.n1d
        self.nel1d = self.n1d - 1       # elements per direction
        self.nel = self.nel1d * self.nel1d
        self.x1d = fem.uniform_1d_coords(self.n1d)

    @staticmethod
    def idx(i, j, n1d):
        return i + j * n1d

    def element_nodes(self, ei, ej):
        """Global node ids of element (ei,ej)'s 4 vertices, CCW order matching
        fem.shape's convention: (ei,ej), (ei+1,ej), (ei+1,ej+1), (ei,ej+1).
        """
        n1d = self.n1d
        return [self.idx(ei, ej, n1d), self.idx(ei + 1, ej, n1d),
                self.idx(ei + 1, ej + 1, n1d), self.idx(ei, ej + 1, n1d)]

    def node_coords(self):
        """(x, y) of every node, indexed by the global node id p = i + j*n1d."""
        return np.tile(self.x1d, self.n1d), np.repeat(self.x1d, self.n1d)


def _element_arrays(g):
    """Connectivity and vertex coordinates of every element at once.

    Returns (nodes, xl, yl) of shape (nel, 4), with elements ordered ej-major
    (ei fastest), matching the nested assembly loops of the IFISS originals.
    """
    ej, ei = np.meshgrid(np.arange(g.nel1d), np.arange(g.nel1d), indexing="ij")
    ei = ei.ravel()
    ej = ej.ravel()

    n1d = g.n1d
    nodes = np.stack([GridQ1.idx(ei, ej, n1d), GridQ1.idx(ei + 1, ej, n1d),
                      GridQ1.idx(ei + 1, ej + 1, n1d), GridQ1.idx(ei, ej + 1, n1d)], axis=1)

    x0, x1 = g.x1d[ei], g.x1d[ei + 1]
    y0, y1 = g.x1d[ej], g.x1d[ej + 1]
    xl = np.stack([x0, x1, x1, x0], axis=1)
    yl = np.stack([y0, y0, y1, y1], axis=1)
    return nodes, xl, yl


def _element_matrix_to_csr(nodes, elem_mats, n_nodes):
    """Scatter per-element 4x4 matrices (nel,4,4) into a global n x n matrix.

    Entry (il,jl) of element e lands at (nodes[e,il], nodes[e,jl]); duplicate
    (row,col) pairs are summed, which is the assembly step itself.
    """
    rows = np.repeat(nodes, fem.NUM_LOCAL_NODES, axis=1).ravel()
    cols = np.tile(nodes, (1, fem.NUM_LOCAL_NODES)).ravel()
    return sp.coo_matrix((elem_mats.ravel(), (rows, cols)),
                         shape=(n_nodes, n_nodes)).tocsc()


def _element_diag_to_csr(nodes, elem_diags, n_nodes):
    """Scatter per-element lumped (diagonal) contributions (nel,4) into a
    diagonal n x n matrix."""
    flat = nodes.ravel()
    return sp.coo_matrix((elem_diags.ravel(), (flat, flat)),
                         shape=(n_nodes, n_nodes)).tocsc()


# -----------------------------------------------------------------------
# Q1 diffusion assembly: stiffness A_stiff, consistent mass M_cons, lumped
# mass M_lump and source load f_rhs. Translation of femq1_diff.m.
#
#   A_stiff_{ij} = \int_Ω  ∇(phi_i) ⋅ ∇(phi_j)  dΩ
#   M_cons_{ij}  = \int_Ω  phi_i * phi_j        dΩ
#   f_rhs_i      = \int_Ω  phi_i * source(x)    dΩ
#
# where {phi_i} are the Q1 bilinear nodal basis functions on the element,
# evaluated at 2x2 Gauss points.
# -----------------------------------------------------------------------

@dataclass
class FemQ1DiffResult:
    A_stiff: sp.csc_matrix
    M_cons: sp.csc_matrix
    M_lump: sp.csc_matrix
    f_rhs: np.ndarray


def assemble_femq1_diff(g):
    nodes, xl, yl = _element_arrays(g)
    nloc = fem.NUM_LOCAL_NODES

    ae = np.zeros((g.nel, nloc, nloc))
    me = np.zeros((g.nel, nloc, nloc))
    me_lump = np.zeros((g.nel, nloc))
    fe = np.zeros((g.nel, nloc))

    for s, t, wt in fem.gauss_2x2():
        phi, dphidx, dphidy, jac = fem.deriv(s, t, xl, yl)
        scale = jac * wt
        src = fem.gauss_source(phi, xl, yl)

        ae += (dphidx[:, :, None] * dphidx[:, None, :] +
               dphidy[:, :, None] * dphidy[:, None, :]) * scale[:, None, None]
        me += (phi[:, None] * phi[None, :]) * scale[:, None, None]
        fe += src[:, None] * phi * scale[:, None]
        me_lump += phi * scale[:, None]

    f_rhs = np.bincount(nodes.ravel(), weights=fe.ravel(), minlength=g.n_nodes)
    return FemQ1DiffResult(
        A_stiff=_element_matrix_to_csr(nodes, ae, g.n_nodes),
        M_cons=_element_matrix_to_csr(nodes, me, g.n_nodes),
        M_lump=_element_diag_to_csr(nodes, me_lump, g.n_nodes),
        f_rhs=f_rhs,
    )


# ---------------------------------------------------------------------
# Q1 convection-diffusion assembly: adds convection matrix N_conv to the
# diffusion assembly above. Translation of femq1_cd.m. Element Peclet
# number / SUPG scaling diagnostics are omitted.
#
#   A_stiff_{ij} = \int_Ω  ∇(phi_i) ⋅ ∇(phi_j)          dΩ
#   M_cons_{ij}  = \int_Ω  phi_i * phi_j                dΩ
#   N_conv_{ij}  = \int_Ω  phi_i * ( w(x) ⋅ ∇(phi_j) )  dΩ
#
# where w(x) = (wx, wy) is the wind/transport field sampled at each Gauss
# point via fem.gauss_transprt. The resulting convection-diffusion operator
# is D_op = eps * A_stiff + N_conv.
# ---------------------------------------------------------------------

@dataclass
class FemQ1CdResult:
    A_stiff: sp.csc_matrix
    N_conv: sp.csc_matrix
    M_cons: sp.csc_matrix
    M_lump: sp.csc_matrix
    f_rhs: np.ndarray


def assemble_femq1_cd(g, wind=fem.velocity_field_w_constant):
    nodes, xl, yl = _element_arrays(g)
    nloc = fem.NUM_LOCAL_NODES

    ae = np.zeros((g.nel, nloc, nloc))
    ne = np.zeros((g.nel, nloc, nloc))
    me = np.zeros((g.nel, nloc, nloc))
    me_lump = np.zeros((g.nel, nloc))
    fe = np.zeros((g.nel, nloc))

    for s, t, wt in fem.gauss_2x2():
        phi, dphidx, dphidy, jac = fem.deriv(s, t, xl, yl)
        scale = jac * wt
        src = fem.gauss_source(phi, xl, yl)
        wx, wy = fem.gauss_transprt(phi, xl, yl, wind)

        ae += (dphidx[:, :, None] * dphidx[:, None, :] +
               dphidy[:, :, None] * dphidy[:, None, :]) * scale[:, None, None]
        me += (phi[:, None] * phi[None, :]) * scale[:, None, None]
        # ne[e,il,jl] = phi_il * (w . grad phi_jl): row index il, column index jl.
        ne += (wx[:, None, None] * phi[:, None] * dphidx[:, None, :] +
               wy[:, None, None] * phi[:, None] * dphidy[:, None, :]) * scale[:, None, None]
        fe += src[:, None] * phi * scale[:, None]
        me_lump += phi * scale[:, None]

    f_rhs = np.bincount(nodes.ravel(), weights=fe.ravel(), minlength=g.n_nodes)
    return FemQ1CdResult(
        A_stiff=_element_matrix_to_csr(nodes, ae, g.n_nodes),
        N_conv=_element_matrix_to_csr(nodes, ne, g.n_nodes),
        M_cons=_element_matrix_to_csr(nodes, me, g.n_nodes),
        M_lump=_element_diag_to_csr(nodes, me_lump, g.n_nodes),
        f_rhs=f_rhs,
    )


# -----------------------------------------------------------------------
# FD diffusion assembly: standard 5-point Laplacian stiffness A_stiff and
# diagonal lumped mass M_lump (composite-trapezoidal area weight per node,
# so it sums exactly to the domain area, same invariant as the FEM lumped
# mass) on the same uniform GridQ1 node layout used by the FEM path.
# Boundary rows of A_stiff are left empty here since apply_dirichlet_bc
# fills them in (diagonal = 1) during the shared post-assembly BC step below.
# -----------------------------------------------------------------------

@dataclass
class FdDiffResult:
    A_stiff: sp.csc_matrix
    M_lump: sp.csc_matrix
    f_rhs: np.ndarray


def _fd_node_grid(g):
    """Node index arrays in the p = i + j*n1d ordering, plus the interior mask."""
    n1d = g.n1d
    j, i = np.meshgrid(np.arange(n1d), np.arange(n1d), indexing="ij")
    i = i.ravel()
    j = j.ravel()
    p = GridQ1.idx(i, j, n1d)
    return i, j, p, ~is_boundary_node(i, j, n1d)


def _fd_lumped_mass(g, i, j, p):
    h = g.x1d[1] - g.x1d[0]
    area = h * h
    weight = area * fd_trapezoid_factor(i, g.n1d) * fd_trapezoid_factor(j, g.n1d)
    return sp.coo_matrix((weight, (p, p)), shape=(g.n_nodes, g.n_nodes)).tocsc()


def _fd_laplacian_entries(g, p_int):
    """Rows/cols/vals of the interior 5-point Laplacian stencil."""
    n1d = g.n1d
    h = g.x1d[1] - g.x1d[0]
    inv_h2 = 1.0 / (h * h)

    rows = np.concatenate([p_int] * 5)
    cols = np.concatenate([p_int, p_int - 1, p_int + 1, p_int - n1d, p_int + n1d])
    vals = np.concatenate([np.full(p_int.size, 4.0 * inv_h2)] +
                          [np.full(p_int.size, -inv_h2)] * 4)
    return rows, cols, vals


def assemble_fd_diff(g):
    i, j, p, interior = _fd_node_grid(g)
    rows, cols, vals = _fd_laplacian_entries(g, p[interior])
    A_stiff = sp.coo_matrix((vals, (rows, cols)), shape=(g.n_nodes, g.n_nodes)).tocsc()
    return FdDiffResult(A_stiff=A_stiff,
                        M_lump=_fd_lumped_mass(g, i, j, p),
                        f_rhs=np.zeros(g.n_nodes))


# -----------------------------------------------------------------------
# FD convection-diffusion assembly: adds a first-order upwind convection
# operator N_conv to the FD diffusion assembly above, wind sampled directly
# at each grid node. The resulting operator is D_op = eps * A_stiff + N_conv,
# matching the FEM composition in assemble_femq1_cd.
# -----------------------------------------------------------------------

@dataclass
class FdCdResult:
    A_stiff: sp.csc_matrix
    N_conv: sp.csc_matrix
    M_lump: sp.csc_matrix
    f_rhs: np.ndarray


def assemble_fd_cd(g, wind=fem.velocity_field_w_constant):
    i, j, p, interior = _fd_node_grid(g)
    n1d = g.n1d
    h = g.x1d[1] - g.x1d[0]
    inv_h = 1.0 / h

    p_int = p[interior]
    rows, cols, vals = _fd_laplacian_entries(g, p_int)
    A_stiff = sp.coo_matrix((vals, (rows, cols)), shape=(g.n_nodes, g.n_nodes)).tocsc()

    # Upwind: w * dy/dx |_i ~ wx_p*(y_i - y_{i-1})/h - wx_m*(y_{i+1} - y_i)/h,
    # and likewise in y; so both neighbor coefficients carry a minus sign.
    wx, wy = wind(g.x1d[i[interior]], g.x1d[j[interior]])
    wx_p, wx_m = np.maximum(wx, 0.0), np.maximum(-wx, 0.0)
    wy_p, wy_m = np.maximum(wy, 0.0), np.maximum(-wy, 0.0)

    n_rows = np.concatenate([p_int] * 5)
    n_cols = np.concatenate([p_int, p_int - 1, p_int + 1, p_int - n1d, p_int + n1d])
    n_vals = np.concatenate([(wx_p + wx_m + wy_p + wy_m) * inv_h,
                             -wx_p * inv_h, -wx_m * inv_h,
                             -wy_p * inv_h, -wy_m * inv_h])
    N_conv = sp.coo_matrix((n_vals, (n_rows, n_cols)), shape=(g.n_nodes, g.n_nodes)).tocsc()

    return FdCdResult(A_stiff=A_stiff, N_conv=N_conv,
                      M_lump=_fd_lumped_mass(g, i, j, p),
                      f_rhs=np.zeros(g.n_nodes))


# -----------------------------------------------------------------------
# Dispatches diffusion / convection-diffusion assembly to FEM or FD based
# on `disc`, so the QP generators below only branch once per operator. FD
# always uses its lumped mass (there is no FD analogue of the consistent
# Q1 mass matrix), so `lump_mass` only affects the FEM path.
#
# D_op feeds into the shared PDE constraint D_op*y - M*u = rhs (see
# make_problem_l2_from_mats), which encodes the FEM weak form K*y = M*u.
# FD's strong-form Laplacian/convection assembly instead represents the
# pointwise equation D_op*y = u (no mass weighting on u), and FD's stiffness
# is O(1/h^2) rather than FEM's O(1) -- so passing FD's raw D_op through the
# same M*u convention would silently divide the control's influence on the
# state by an extra O(h^2) per stage. To reuse the shared constraint assembly
# unchanged, the FD operator is mass-scaled here (M_lump * D_op_raw), which is
# algebraically equivalent to the strong-form equation (mass is
# diagonal/invertible) and also renormalizes FD's stiffness down to FEM's
# O(1) magnitude.
# -----------------------------------------------------------------------

def assemble_diff_by_discretization(g, disc, lump_mass):
    """Returns (D_op, M)."""
    if Discretization.parse(disc) is Discretization.FD:
        res = assemble_fd_diff(g)
        return (res.M_lump @ res.A_stiff).tocsc(), res.M_lump
    res = assemble_femq1_diff(g)
    return res.A_stiff, (res.M_lump if lump_mass else res.M_cons)


def assemble_cd_by_discretization(g, disc, lump_mass, eps, wind=fem.velocity_field_w_constant):
    """Returns (D_op, M)."""
    if Discretization.parse(disc) is Discretization.FD:
        res = assemble_fd_cd(g, wind)
        return (res.M_lump @ (eps * res.A_stiff + res.N_conv)).tocsc(), res.M_lump
    res = assemble_femq1_cd(g, wind)
    return (eps * res.A_stiff + res.N_conv).tocsc(), (res.M_lump if lump_mass else res.M_cons)


# -----------------------------------------------------------------------
# Dirichlet boundary conditions via row/col elimination on the fully
# assembled global operator. Translation of nonzerobc_input.m. Applied as
# a separate post-assembly step.
#
# For boundary nodes p with prescribed value g_p = bc_values[p]:
#   rhs_r  <-  rhs_r - sum_{p in bc_nodes} D_op(r,p) * g_p    for r not in bc_nodes
#   D_op(p, :) = D_op(:, p) = 0,  D_op(p, p) = 1              for p in bc_nodes
#   rhs_p  <-  g_p                                            for p in bc_nodes
#
# i.e. known boundary columns are folded into the RHS of the interior
# equations, then boundary rows/cols are replaced by identity rows so that
# solving D_op y = rhs directly yields y_p = g_p at the boundary.
# -----------------------------------------------------------------------

def apply_dirichlet_bc(D_op, rhs, bc_nodes, bc_values):
    """Returns (D_op, rhs) with the boundary conditions eliminated. Inputs are
    not modified in place."""
    n = D_op.shape[0]
    bc_nodes = np.asarray(bc_nodes, dtype=np.int64)
    rhs = np.array(rhs, dtype=float, copy=True)

    is_bc = np.zeros(n, dtype=bool)
    bc_value_at = np.zeros(n)
    is_bc[bc_nodes] = True
    bc_value_at[bc_nodes] = bc_values

    coo = D_op.tocsc().tocoo()
    r, c, v = coo.row, coo.col, coo.data

    # fold known boundary columns into the RHS before elimination
    fold = (~is_bc[r]) & is_bc[c]
    np.add.at(rhs, r[fold], -v[fold] * bc_value_at[c[fold]])

    # zero boundary rows/cols, keep interior entries, set diagonal = 1
    keep = (~is_bc[r]) & (~is_bc[c])
    rows = np.concatenate([r[keep], bc_nodes])
    cols = np.concatenate([c[keep], bc_nodes])
    vals = np.concatenate([v[keep], np.ones(bc_nodes.size)])
    D_new = sp.coo_matrix((vals, (rows, cols)), shape=D_op.shape).tocsc()

    rhs[bc_nodes] = bc_value_at[bc_nodes]
    return D_new, rhs


def apply_dirichlet_bc_mass(M, bc_nodes):
    """Zeroes boundary rows/cols of a mass matrix (no RHS coupling)."""
    n = M.shape[0]
    is_bc = np.zeros(n, dtype=bool)
    is_bc[np.asarray(bc_nodes, dtype=np.int64)] = True

    coo = M.tocsc().tocoo()
    keep = (~is_bc[coo.row]) & (~is_bc[coo.col])
    return sp.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])),
                         shape=M.shape).tocsc()


def fem_boundary_nodes(g):
    """Boundary node ids of a GridQ1, in increasing node order."""
    n1d = g.n1d
    p = np.arange(g.n_nodes)
    return p[is_boundary_node(p % n1d, p // n1d, n1d)]


# -----------------------------------------------------------------------
# Problem container, matching the C++ KSPQPdata layout:
#
#   min  c^T x + 0.5 x^T Q x + obj_const
#   s.t. A x = b,  B x = w,  lx <= x <= ux,  lw <= w <= uw
# -----------------------------------------------------------------------

@dataclass
class KSPQPdata:
    n: int
    m: int
    l: int
    Q: sp.csc_matrix
    A: sp.csc_matrix
    B: sp.csc_matrix
    c: np.ndarray
    b: np.ndarray
    lx: np.ndarray
    ux: np.ndarray
    lw: np.ndarray
    uw: np.ndarray
    obj_const: float

    def to_dict(self):
        """CSC/numpy dict in the format ksp_qp_bind.solve_from_data() and
        benchmark_common.kspqp_to_qpalm() consume (same as parse_sif's output).
        """
        out = {"n": int(self.n), "m": int(self.m), "l": int(self.l),
               "obj_const": float(self.obj_const)}
        for M, key in ((self.Q, "Q"), (self.A, "A"), (self.B, "B")):
            M = M.tocsc()
            out[f"{key}_data"] = np.asarray(M.data, dtype=np.float64)
            out[f"{key}_indices"] = np.asarray(M.indices, dtype=np.int32)
            out[f"{key}_indptr"] = np.asarray(M.indptr, dtype=np.int32)
            out[f"{key}_shape"] = (int(M.shape[0]), int(M.shape[1]))
        for name in ("c", "b", "lx", "ux", "lw", "uw"):
            out[name] = np.asarray(getattr(self, name), dtype=np.float64)
        return out


# -----------------------------------------------------------------------
# L2-regularized PDE-constrained QP:
#
#   x = [y; u]
#   min  0.5 (y - yhat)^T M (y - yhat) + 0.5 * beta * u^T M u
#   s.t. D_op y - M u = rhs
#        y_lower <= y <= y_upper
#        u_lower <= u <= u_upper
#
# (Pearson & Gondzio, 2017)
#
# Note: B is an empty 0 x n matrix, i.e. l = 0.
# -----------------------------------------------------------------------

def make_problem_l2_from_mats(D_op, M, rhs, yhat, beta,
                              y_lower=-INF, y_upper=INF,
                              u_lower=-INF, u_upper=INF):
    rhs = np.asarray(rhs, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    n_nodes = rhs.size
    nx = 2 * n_nodes  # [y; u]

    M = M.tocsc()
    D_op = D_op.tocsc()

    Myhat = M @ yhat
    obj_const = 0.5 * float(yhat @ Myhat)

    c = np.zeros(nx)
    c[:n_nodes] = -Myhat
    # c on u is zero.

    # Q = [[M      0   ],
    #      [0, beta * M]]
    Mc = M.tocoo()
    Q = sp.coo_matrix(
        (np.concatenate([Mc.data, beta * Mc.data]),
         (np.concatenate([Mc.row, Mc.row + n_nodes]),
          np.concatenate([Mc.col, Mc.col + n_nodes]))),
        shape=(nx, nx)).tocsc()

    # A = [D_op, -M], b = rhs
    Dc = D_op.tocoo()
    A = sp.coo_matrix(
        (np.concatenate([Dc.data, -Mc.data]),
         (np.concatenate([Dc.row, Mc.row]),
          np.concatenate([Dc.col, Mc.col + n_nodes]))),
        shape=(n_nodes, nx)).tocsc()

    # No B x = w block.
    B = sp.csc_matrix((0, nx))

    # Bounds on x = [y; u]
    lx = np.concatenate([np.full(n_nodes, y_lower), np.full(n_nodes, u_lower)])
    ux = np.concatenate([np.full(n_nodes, y_upper), np.full(n_nodes, u_upper)])

    return KSPQPdata(n=nx, m=n_nodes, l=0, Q=Q, A=A, B=B, c=c, b=rhs.copy(),
                     lx=lx, ux=ux, lw=np.zeros(0), uw=np.zeros(0),
                     obj_const=obj_const)


# -----------------------------------------------------------------------
# Mesh-size normalization, applied to the assembled QP as a final step:
#
#   Q, c, obj_const  <-  (Q, c, obj_const) / h^2
#   A(r,:), b_r      <-  (A(r,:), b_r) / h^2       for r not in bc_nodes
#
# On the uniform grid (h = 2^-nc) every entry of M is O(h^2) -- exactly h^2
# on the lumped interior diagonal, h^2 times a fixed stencil for the
# consistent mass -- and hence so are Q and c. An interior row
# D_op*y - M*u - rhs of the state equation is O(h^2) as well at smooth
# (y, u), since D_op*y ~ h^2 * (D y)(x_p) there. Left as is, every KKT
# residual block shrinks as O(h^2) under refinement, so a fixed solver tol
# becomes less demanding as the mesh is refined; dividing by h^2 brings them
# all to the mesh-independent O(1) scale of the continuous optimality system.
# The Dirichlet rows y_p = g_p set by apply_dirichlet_bc are already O(1)
# and are left alone.
#
# The primal solution x = [y; u] is unchanged. The multipliers change as
# z -> z / h^2 and, on the Dirichlet rows only, y1 -> y1 / h^2; the interior
# entries of y1 (the discrete adjoint) are unchanged. h is a power of two, so
# the scaling is exact in floating point.
# -----------------------------------------------------------------------

def scale_by_mesh_size(pb, bc_nodes, h):
    """Returns a copy of pb with the objective and the interior rows of
    A x = b divided by h^2. pb is not modified in place."""
    inv_h2 = 1.0 / (h * h)
    row_scale = np.full(pb.m, inv_h2)
    row_scale[np.asarray(bc_nodes, dtype=np.int64)] = 1.0
    return replace(pb, Q=(inv_h2 * pb.Q).tocsc(), c=inv_h2 * pb.c,
                   obj_const=inv_h2 * pb.obj_const,
                   A=(sp.diags(row_scale) @ pb.A).tocsc(), b=row_scale * pb.b)


# ===== QP generators =====
# All three are the L2-regularized PDE-constrained control problems of
# (Pearson & Gondzio, 2017), normalized by the mesh size (scale_by_mesh_size).


def make_poisson_l2_control(nc, beta, y_lower=-INF, y_upper=INF,
                            u_lower=-INF, u_upper=INF,
                            lump_mass=False, disc=Discretization.FEM):
    """2D Poisson control problem (control-constrained).

        Ω = [0,1]^2, y = 0 on boundary,
        D = -Δ,
        yhat = exp(-64((x1 - 0.5)^2 + (x2 - 0.5)^2)).

    (Pearson & Gondzio, 2017)
    """
    g = GridQ1(nc)
    x, y = g.node_coords()
    yhat = np.exp(-64.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))

    D, M = assemble_diff_by_discretization(g, disc, lump_mass)
    rhs = np.zeros(g.n_nodes)

    bc_nodes = fem_boundary_nodes(g)
    D, rhs = apply_dirichlet_bc(D, rhs, bc_nodes, np.zeros(bc_nodes.size))
    M = apply_dirichlet_bc_mass(M, bc_nodes)

    pb = make_problem_l2_from_mats(D, M, rhs, yhat, beta,
                                   y_lower, y_upper, u_lower, u_upper)
    return scale_by_mesh_size(pb, bc_nodes, g.x1d[1] - g.x1d[0])


def make_poisson_l2_state_control(nc, beta, y_lower=-INF, y_upper=INF,
                                  u_lower=-INF, u_upper=INF,
                                  lump_mass=False, disc=Discretization.FEM):
    """2D Poisson control problem (state-constrained).

        Ω = [0,1]^2, y = yhat on boundary,
        D = -Δ,
        yhat = sin(pi x1) sin(pi x2).

    (Pearson & Gondzio, 2017)
    """
    g = GridQ1(nc)
    x, y = g.node_coords()
    yhat = np.sin(np.pi * x) * np.sin(np.pi * y)

    D, M = assemble_diff_by_discretization(g, disc, lump_mass)
    rhs = np.zeros(g.n_nodes)

    bc_nodes = fem_boundary_nodes(g)
    D, rhs = apply_dirichlet_bc(D, rhs, bc_nodes, yhat[bc_nodes])
    M = apply_dirichlet_bc_mass(M, bc_nodes)

    pb = make_problem_l2_from_mats(D, M, rhs, yhat, beta,
                                   y_lower, y_upper, u_lower, u_upper)
    return scale_by_mesh_size(pb, bc_nodes, g.x1d[1] - g.x1d[0])


def make_convdiff_l2_control(nc, beta, y_lower=-INF, y_upper=INF,
                             u_lower=-INF, u_upper=INF, eps=0.01,
                             lump_mass=False, disc=Discretization.FEM):
    """2D convection-diffusion control problem.

        Ω = [0,1]^2, y = 0 on boundary,
        D = -eps * Δ + w ⋅ ∇ (constant wind w = [-1/sqrt(2), 1/sqrt(2)]^T),
        yhat = exp(-64((x1 - 0.5)^2 + (x2 - 0.5)^2)).

    (Pearson & Gondzio, 2017)
    """
    g = GridQ1(nc)
    x, y = g.node_coords()
    yhat = np.exp(-64.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))

    D, M = assemble_cd_by_discretization(g, disc, lump_mass, eps,
                                         fem.velocity_field_w_constant)
    rhs = np.zeros(g.n_nodes)

    bc_nodes = fem_boundary_nodes(g)
    D, rhs = apply_dirichlet_bc(D, rhs, bc_nodes, np.zeros(bc_nodes.size))
    M = apply_dirichlet_bc_mass(M, bc_nodes)

    pb = make_problem_l2_from_mats(D, M, rhs, yhat, beta,
                                   y_lower, y_upper, u_lower, u_upper)
    return scale_by_mesh_size(pb, bc_nodes, g.x1d[1] - g.x1d[0])


_L2_GENERATORS = {
    "poisson": make_poisson_l2_control,
    "poisson_state": make_poisson_l2_state_control,
    "convdiff": make_convdiff_l2_control,
}


def generate_pde_l2_qp(choice, nc, beta, y_lower=-INF, y_upper=INF,
                       u_lower=-INF, u_upper=INF, eps=0.01,
                       lumped_mass=False, discretization="fem"):
    """Generate an L2-regularized PDE-constrained QP (Pearson & Gondzio, 2017).

    choice = 'poisson'       - 2D Poisson control (control-constrained)
    choice = 'poisson_state' - 2D Poisson control (state-constrained)
    choice = 'convdiff'      - 2D convection-diffusion control

    nc     = grid exponent (grid size = 2^nc + 1 per direction)
    beta   = L2 regularisation weight
    y_lower/y_upper/u_lower/u_upper = box bounds on state/control (default ±inf)
    eps    = diffusion coefficient, 'convdiff' only.
    lumped_mass = if True, use the lumped (diagonal) mass matrix instead of the
                  consistent Q1 mass matrix. Ignored (always lumped) when
                  discretization='fd'.
    discretization = 'fem' (default, Q1 finite elements) or 'fd' (5-point
                  finite-difference stencil with first-order upwind convection).

    The QP is normalized by the mesh size (see scale_by_mesh_size): its
    objective and the interior rows of its state equation are divided by
    h^2 = 4^-nc, so that a fixed solver tol asks for the same accuracy on every
    mesh. The solution x = [y; u] is unchanged, but objective values come out
    divided by h^2.

    Returns the CSC/numpy dict consumed by ksp_qp_bind.solve_from_data() and
    benchmark_common.kspqp_to_qpalm().
    """
    if choice not in _L2_GENERATORS:
        raise ValueError("choice must be one of "
                         f"{sorted(_L2_GENERATORS)}, got: {choice!r}")
    disc = Discretization.parse(discretization)
    kwargs = dict(y_lower=y_lower, y_upper=y_upper, u_lower=u_lower, u_upper=u_upper,
                  lump_mass=lumped_mass, disc=disc)
    if choice == "convdiff":
        kwargs["eps"] = eps
    return _L2_GENERATORS[choice](nc, beta, **kwargs).to_dict()
