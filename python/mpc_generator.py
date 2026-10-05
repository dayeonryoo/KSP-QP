"""
Build linear Model Predictive Control (MPC) QPs in KSPQPdata dict form (same
format as ksp_qp_bind.parse_sif()), for benchmark_mpc.py.

MPC is a smooth QP: quadratic tracking cost, linear dynamics equalities, box
bounds on states/inputs. The constraint matrix is block-bidiagonal: each
dynamics row couples one timestep to the next.

Model
-----
Continuous-time dynamics x' = Ac x + Bc u, discretized via zero-order hold
at sample time Ts to x_{k+1} = A x_k + B u_k. At every control step, given
the current state xbar, MPC solves:

    min_{x_0..x_N, u_0..u_{N-1}}
        sum_{k=0}^{N-1} [(x_k-x_ref)'Q(x_k-x_ref) + (u_k-u_ref)'R(u_k-u_ref)]
        + (x_N-x_ref)'P(x_N-x_ref)
    s.t.  x_0 = xbar
          x_{k+1} = A x_k + B u_k,  k=0..N-1
          x_min <= x_k <= x_max,     k=1..N
          u_min <= u_k <= u_max,     k=0..N-1

P is the exact discrete-algebraic-Riccati-equation (DARE) solution for
(A,B,Q,R), a fixed point of the backward Riccati recursion, so the LQR
ground-truth check in validate_mpc_generator.py is exact for any N>=1.

Variable stacking:
    z = [x_0; u_0; x_1; u_1; ...; x_{N-1}; u_{N-1}; x_N],
    n_z = (N+1)*n_x + N*n_u.
Mapping to KSP-QP's min c'x + 0.5x'Qx s.t. Ax=b, lw<=Bx<=uw, lx<=x<=ux:
  - Q_ksp = blkdiag(2Q, 2R, ..., 2Q, 2R, 2P) (the factor of 2 reconciles the
    tracking cost's 1*(x-xref)'Q(x-xref) with KSP-QP's 0.5*x'Q_ksp*x)
  - c: per-block linear terms from expanding the quadratic around
    x_ref/u_ref; obj_const absorbs the constant terms left over.
  - A, b: block-bidiagonal equality matrix (x_0=xbar row-block, then N
    dynamics row-blocks each touching only x_k, u_k, x_{k+1}) -- assembled
    via sparse triplets, never densified.
  - lx, ux: box bounds (l = 0, no B/lw/uw). The x_0 block is +-inf, since
    the equality row pins it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import scipy.signal as sig

INF = np.inf


# ---------------------------------------------------------------------------
# KSPQPdata dict assembly
# ---------------------------------------------------------------------------

def _drop_zeros(M) -> sp.csc_matrix:
    """Dense or sparse block -> CSC with no explicitly stored zeros."""
    out = sp.csc_matrix(M)
    out.eliminate_zeros()
    return out


def _sparse_to_ksp(M: sp.spmatrix, key: str) -> dict:
    M = M.tocsc()
    # Safety net for all three matrices: a structurally stored zero is a real cost
    # to every solver downstream, and nothing here ever needs one.
    M.eliminate_zeros()
    M.sort_indices()
    return {
        f"{key}_data":    M.data.astype(np.float64),
        f"{key}_indices": M.indices.astype(np.int32),
        f"{key}_indptr":  M.indptr.astype(np.int32),
        f"{key}_shape":   M.shape,
    }


def _assemble(Q: sp.spmatrix, A: sp.spmatrix, B: sp.spmatrix,
              c: np.ndarray,  b: np.ndarray,
              lx: np.ndarray, ux: np.ndarray,
              lw: np.ndarray, uw: np.ndarray, obj_const: float = 0.0) -> dict:
    n_z = Q.shape[0]
    out: dict = {"n": n_z, "m": int(A.shape[0]), "l": int(B.shape[0]), "obj_const": float(obj_const)}
    out.update(_sparse_to_ksp(Q, "Q"))
    out.update(_sparse_to_ksp(A, "A"))
    out.update(_sparse_to_ksp(B, "B"))
    out["c"]  = c.astype(np.float64)
    out["b"]  = b.astype(np.float64)
    out["lx"] = lx.astype(np.float64)
    out["ux"] = ux.astype(np.float64)
    out["lw"] = lw.astype(np.float64)
    out["uw"] = uw.astype(np.float64)
    return out


# ---------------------------------------------------------------------------
# System definitions
# ---------------------------------------------------------------------------

@dataclass
class MpcSystem:
    name: str
    Ac: np.ndarray
    Bc: np.ndarray
    Ts: float
    Q: np.ndarray
    R: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray
    xbar0: np.ndarray
    n_x: int
    n_u: int


def discretize(Ac: np.ndarray, Bc: np.ndarray, Ts: float) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold discretization: x_{k+1} = Ad x_k + Bd u_k."""
    n_x = Ac.shape[0]
    Ad, Bd, _, _, _ = sig.cont2discrete((Ac, Bc, np.eye(n_x), np.zeros((n_x, Bc.shape[1]))),
                                        Ts, method="zoh")
    return Ad, Bd


def terminal_cost(Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Exact discrete-algebraic-Riccati-equation solution for (Ad,Bd,Q,R)."""
    return la.solve_discrete_are(Ad, Bd, Q, R)


def lqr_gain(Ad: np.ndarray, Bd: np.ndarray, P: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Infinite-horizon LQR gain K such that u* = -K @ xbar (validation only)."""
    return np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)


def platoon_system(M: int) -> MpcSystem:
    """Chain of M coupled double integrators (vehicle platoon), in velocity-
    deviation coordinates relative to a constant-speed lead vehicle (v_0=0
    identically -- standard simplifying assumption in the platooning-MPC
    literature, e.g. Ploeg et al. 2014, Zheng et al. 2016). State per
    vehicle i is (d_i, v_i) [spacing-error deviation, velocity deviation];
    d_i' = v_{i-1} - v_i, v_i' = u_i.
    """
    n_x, n_u = 2 * M, M
    Ac = np.zeros((n_x, n_x))
    Bc = np.zeros((n_x, n_u))
    for i in range(M):
        d_idx, v_idx = 2 * i, 2 * i + 1
        Ac[d_idx, v_idx] = -1.0
        if i > 0:
            Ac[d_idx, 2 * (i - 1) + 1] = 1.0
        Bc[v_idx, i] = 1.0

    Q = np.zeros((n_x, n_x))
    for i in range(M):
        Q[2 * i, 2 * i] = 10.
        Q[2 * i + 1, 2 * i + 1] = 1.
    R = 0.1 * np.eye(M)

    x_min = np.tile([-2., -5.], M)
    u_min = np.full(M, -2.)
    xbar0 = np.zeros(n_x)
    # Vehicle 1 perturbed (the rest at steady state). Since d_1' = -v_1 (v_0 = 0),
    # a large positive v_1 drives d_1 negative faster than the bounded actuator
    # can correct, so d_1 and v_1 can't both start near their bounds: (-1.8, 4.5)
    # is infeasible at every N, while (-1.0, 2.0) is feasible across
    # benchmark_mpc.py's default sweep.
    xbar0[0], xbar0[1] = -1.0, 2.0

    return MpcSystem(
        name=f"platoon_M{M}", Ac=Ac, Bc=Bc, Ts=0.1,
        Q=Q, R=R, x_min=x_min, x_max=-x_min, u_min=u_min, u_max=-u_min,
        xbar0=xbar0, n_x=n_x, n_u=n_u,
    )


SYSTEM_BUILDERS = {
    "platoon": lambda M=5, **kw: platoon_system(M),
}


# ---------------------------------------------------------------------------
# Sparse QP assembly
# ---------------------------------------------------------------------------

def _build_dynamics(Ad: np.ndarray, Bd: np.ndarray, xbar: np.ndarray,
                    N: int, n_x: int, n_u: int):
    """Block-bidiagonal equality matrix: x_0=xbar, then x_{k+1}=Ad x_k+Bd u_k."""
    stage = n_x + n_u
    n_z = (N + 1) * n_x + N * n_u
    col_x = lambda k: k * stage
    col_u = lambda k: k * stage + n_x
    m = (N + 1) * n_x

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(n_x):
        rows.append(i); cols.append(col_x(0) + i); vals.append(1.0)
    b = np.zeros(m)
    b[:n_x] = xbar

    Ad_coo, Bd_coo = sp.coo_matrix(Ad), sp.coo_matrix(Bd)
    for k in range(N):
        row0 = (k + 1) * n_x
        for r, c, v in zip(Ad_coo.row, Ad_coo.col, Ad_coo.data):
            rows.append(row0 + r); cols.append(col_x(k) + c); vals.append(-v)
        for r, c, v in zip(Bd_coo.row, Bd_coo.col, Bd_coo.data):
            rows.append(row0 + r); cols.append(col_u(k) + c); vals.append(-v)
        for i in range(n_x):
            rows.append(row0 + i); cols.append(col_x(k + 1) + i); vals.append(1.0)

    A = sp.coo_matrix((vals, (rows, cols)), shape=(m, n_z)).tocsc()
    return A, b, n_z, col_x, col_u


def _build_cost(Q: np.ndarray, R: np.ndarray, P: np.ndarray, N: int,
                n_x: int, n_u: int, n_z: int, col_x, col_u,
                x_ref: np.ndarray, u_ref: np.ndarray):
    # Convert each stage block to sparse before block_diag: sp.block_diag keeps a
    # dense input's explicit zeros, storing O(N*(n_x^2 + n_u^2)) entries instead of
    # O(N*(n_x + n_u)), and the bloat carries into KSP-QP's chol(Q) and the lifted
    # constraint matrix.
    Q_blk, R_blk, P_blk = (_drop_zeros(2 * B) for B in (Q, R, P))
    blocks = []
    for _ in range(N):
        blocks.append(Q_blk)
        blocks.append(R_blk)
    blocks.append(P_blk)
    Q_ksp = sp.block_diag(blocks, format="csc")

    c = np.zeros(n_z)
    for k in range(N):
        c[col_x(k):col_x(k) + n_x] = -2 * Q @ x_ref
        c[col_u(k):col_u(k) + n_u] = -2 * R @ u_ref
    c[col_x(N):col_x(N) + n_x] = -2 * P @ x_ref

    obj_const = N * (x_ref @ Q @ x_ref + u_ref @ R @ u_ref) + x_ref @ P @ x_ref
    return Q_ksp, c, float(obj_const)


def generate_mpc_qp(system: MpcSystem, N: int, xbar: np.ndarray,
                    x_ref: np.ndarray | None = None,
                    u_ref: np.ndarray | None = None) -> dict:
    """Build the horizon-N MPC QP for the given system at state xbar."""
    n_x, n_u = system.n_x, system.n_u
    if x_ref is None:
        x_ref = np.zeros(n_x)
    if u_ref is None:
        u_ref = np.zeros(n_u)

    Ad, Bd = discretize(system.Ac, system.Bc, system.Ts)
    P = terminal_cost(Ad, Bd, system.Q, system.R)

    A, b, n_z, col_x, col_u = _build_dynamics(Ad, Bd, xbar, N, n_x, n_u)
    Q_ksp, c, obj_const = _build_cost(system.Q, system.R, P, N, n_x, n_u, n_z,
                                       col_x, col_u, x_ref, u_ref)

    lx = np.full(n_z, -INF)
    ux = np.full(n_z, INF)
    for k in range(1, N + 1):
        lx[col_x(k):col_x(k) + n_x] = system.x_min
        ux[col_x(k):col_x(k) + n_x] = system.x_max
    for k in range(0, N):
        lx[col_u(k):col_u(k) + n_u] = system.u_min
        ux[col_u(k):col_u(k) + n_u] = system.u_max

    B = sp.csc_matrix((0, n_z))
    lw = np.zeros(0)
    uw = np.zeros(0)

    return _assemble(Q_ksp, A, B, c, b, lx, ux, lw, uw, obj_const)


def step_dynamics(system: MpcSystem, x: np.ndarray, u0: np.ndarray) -> np.ndarray:
    """Advance the true (discretized) system state by one control step."""
    Ad, Bd = discretize(system.Ac, system.Bc, system.Ts)
    return Ad @ x + Bd @ u0
