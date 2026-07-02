"""
AI Disclosure: This module was written by Claude Fable 5, via Claude Code

Native optimizers for coronagraph design problems.

Currently hosts a dense linear-program solver (:func:`linprog`) implementing
Mehrotra's predictor-corrector interior-point method on the prysm backend, so
the same code runs on numpy (CPU) or — after
``prysm.mathops.set_backend_to_cupy()`` — on cupy (GPU).

The solver targets the LP structure that arises in apodizer design (see the
ExactLP how-to): a **dense** constraint matrix with far fewer rows than
columns and a finite box on every variable,

    minimize    c^T x
    subject to  A x <= b,   l <= x <= u.

Each interior-point iteration reduces (via the normal equations) to forming
``A D^-1 A^T`` — one large GEMM — and one Cholesky factorization of an
``m x m`` matrix, where ``m`` is the *row* count.  For apodizer problems
``m = 4 * n_darkhole << n_pixels``, so the per-iteration cost is dominated by
the GEMM, which is exactly the operation GPUs accelerate best.  Convergence
is typically 20-40 iterations regardless of problem size.

Vertex caveat
-------------
Interior-point methods converge to the *analytic center of the optimal face*,
not to a vertex.  When the optimal face is not a single point, the solution
can carry more fractional (strictly-interior) entries than the vertex a
simplex solver such as HiGHS would return.  For binary-apodizer work this
only changes the size of the residue handed to the finishing step (e.g. the
AugmentedLagrangian pin ladder); the objective value is the same global
optimum.
"""

from collections import namedtuple

from prysm.mathops import linalg, np

LPResult = namedtuple(
    "LPResult",
    [
        "x",            # primal solution, length n
        "y",            # dual variables for A x <= b, length m
        "status",       # 0 converged, 1 iteration limit, 2 infeasible (heuristic), 3 numerical failure
        "message",
        "iterations",
        "fun",          # primal objective c^T x
        "duality_gap",  # complementarity gap, absolute
        "primal_residual",  # ||max(Ax - b, 0)||_inf
        "dual_residual",    # ||c + A^T y - z_l + z_u||_inf
    ],
)

_TERM = {
    0: "converged",
    1: "iteration limit reached",
    2: "primal infeasible (heuristic detection)",
    3: "numerical failure in Cholesky factorization",
}


def _chol_solve(L, rhs):
    """Solve ``(L L^T) sol = rhs`` given a lower-triangular Cholesky factor."""
    tmp = linalg.solve_triangular(L, rhs, lower=True)
    return linalg.solve_triangular(L.T, tmp, lower=False)


def linprog(
    c,
    A_ub,
    b_ub,
    lower_bounds=None,
    upper_bounds=None,
    x0=None,
    tol=1e-8,
    max_iter=100,
    verbose=False,
):
    """Solve ``min c^T x  s.t.  A_ub x <= b_ub,  l <= x <= u`` (dense IPM).

    Parameters
    ----------
    c : ndarray
        Objective vector, length n.  For maximum throughput pass ``-ones(n)``.
    A_ub : ndarray
        Dense inequality matrix, shape (m, n).  Row count m may be smaller or
        larger than n; the normal-equations system solved per iteration is
        m x m, so the solver is most efficient when m < n.
    b_ub : ndarray
        Inequality right-hand side, length m.
    lower_bounds, upper_bounds : ndarray or float, optional
        Finite box on every variable; scalars broadcast.  Default ``0`` / ``1``
        (the apodizer box).  Finiteness is required — it is what guarantees
        the LP is bounded, and the algorithm exploits it.
    x0 : ndarray, optional
        Initial point; clipped safely into the interior of the box.  A good
        warm start (e.g. a previous design) mostly helps the first few
        iterations; interior-point methods gain less from warm starts than
        simplex.
    tol : float, optional
        Relative tolerance applied to primal feasibility, dual feasibility,
        and the complementarity gap.
    max_iter : int, optional
        Iteration cap; each iteration is one GEMM + one Cholesky.
    verbose : bool, optional
        Print per-iteration residuals.

    Returns
    -------
    LPResult
        Named tuple; ``status == 0`` means all three optimality measures fell
        below ``tol``.  Arrays live on the active prysm backend (cupy arrays
        on GPU) — call ``dygdug.backend.asnumpy`` if you need them on the host.

    Notes
    -----
    Implements Mehrotra's predictor-corrector method (Mehrotra 1992; Nocedal
    & Wright ch. 14) on the perturbed KKT system of the slack form

    ``Ax + w = b``, ``w >= 0``, ``l <= x <= u`` with multipliers ``y`` (rows),
    ``z_l``/``z_u`` (bounds).  Eliminating ``w``, ``z_l``, ``z_u`` reduces each
    Newton step to the SPD normal equations

    ``[A D^-1 A^T + diag(w/y)] dy = rhs``,  ``D = diag(z_l/p + z_u/q)``,

    with ``p = x - l``, ``q = u - x``.
    """
    c = np.asarray(c, dtype=float).ravel()
    A = np.asarray(A_ub, dtype=float)
    b = np.asarray(b_ub, dtype=float).ravel()
    if A.ndim != 2:
        raise ValueError("A_ub must be 2-D")
    m, n = A.shape
    if c.size != n or b.size != m:
        raise ValueError(
            f"shape mismatch: A_ub is {m}x{n}, c has {c.size}, b_ub has {b.size}"
        )

    def _bound(value, default):
        if value is None:
            value = default
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = np.full(n, float(arr))
        if arr.size != n:
            raise ValueError(f"bound has length {arr.size}, expected {n}")
        return arr.ravel().copy()

    l = _bound(lower_bounds, 0.0)
    u = _bound(upper_bounds, 1.0)
    if not (bool(np.all(np.isfinite(l))) and bool(np.all(np.isfinite(u)))):
        raise ValueError("bounds must be finite (they bound the LP)")
    if bool(np.any(u <= l)):
        raise ValueError("upper_bounds must exceed lower_bounds elementwise")

    span = u - l
    # Strictly interior start.
    if x0 is None:
        x = l + 0.5 * span
    else:
        x = np.asarray(x0, dtype=float).ravel().copy()
        margin = 0.05 * span
        x = np.clip(x, l + margin, u - margin)

    # Positive initializations for slacks/multipliers.  The start may be
    # primal-infeasible (r_p != 0); the infeasible-start IPM drives it out.
    w = np.maximum(b - A @ x, 1.0)
    y = np.ones(m)
    z_scale = 1.0 + float(np.max(np.abs(c)))
    z_l = np.full(n, z_scale)
    z_u = np.full(n, z_scale)

    b_norm = 1.0 + float(np.max(np.abs(b)))
    c_norm = 1.0 + float(np.max(np.abs(c)))
    n_comp = m + 2 * n

    status, iteration = 1, 0
    for iteration in range(1, max_iter + 1):
        p = x - l
        q = u - x

        r_d = c + A.T @ y - z_l + z_u        # dual residual
        r_p = A @ x + w - b                  # primal residual
        gap = float(w @ y + p @ z_l + q @ z_u)
        mu = gap / n_comp
        obj = float(c @ x)

        rel_p = float(np.max(np.abs(r_p))) / b_norm
        rel_d = float(np.max(np.abs(r_d))) / c_norm
        rel_g = gap / (1.0 + abs(obj))
        if verbose:
            print(
                f"iter {iteration:3d}: obj {obj:+.6e} mu {mu:.2e} "
                f"rp {rel_p:.2e} rd {rel_d:.2e} gap {rel_g:.2e}"
            )
        if rel_p < tol and rel_d < tol and rel_g < tol:
            status = 0
            break

        # Heuristic infeasibility: dual variables blowing up while primal
        # infeasibility persists.  There is no Farkas certificate here.
        if float(np.max(y)) > 1e12 * c_norm and rel_p > 1e3 * tol:
            status = 2
            break

        # Normal equations M dy = rhs, M = A D^-1 A^T + diag(w/y).
        d = z_l / p + z_u / q
        dinv = 1.0 / d
        M = (A * dinv[None, :]) @ A.T
        wy = w / y
        idx = np.arange(m)
        M[idx, idx] += wy
        reg = 1e-14 * float(np.mean(M[idx, idx]))
        L = None
        for _ in range(4):
            M[idx, idx] += reg
            try:
                Lc = np.linalg.cholesky(M)
            except Exception:
                Lc = None
            # cupy's cholesky can emit NaNs instead of raising on a matrix
            # that is not numerically positive definite; check explicitly.
            if Lc is not None and bool(np.all(np.isfinite(Lc))):
                L = Lc
                break
            reg *= 1e4
        if L is None:
            status = 3
            break

        def newton(r3, r4, r5):
            """Newton step for complementarity residuals r3, r4, r5."""
            g = r_d + r4 / p - r5 / q
            rhs = r_p - r3 / y - A @ (g * dinv)
            dy = _chol_solve(L, rhs)
            dx = -(g + A.T @ dy) * dinv
            dw = -(r3 + w * dy) / y
            dzl = -(r4 + z_l * dx) / p
            dzu = (z_u * dx - r5) / q
            return dx, dw, dy, dzl, dzu

        def steplens(dx, dw, dy, dzl, dzu):
            """Max feasible primal/dual step lengths (nonnegativity)."""

            def ratio(v, dv):
                neg = dv < 0
                if not bool(np.any(neg)):
                    return 1.0
                return min(1.0, float(np.min(-v[neg] / dv[neg])))

            a_p = min(ratio(w, dw), ratio(p, dx), ratio(q, -dx))
            a_d = min(ratio(y, dy), ratio(z_l, dzl), ratio(z_u, dzu))
            return a_p, a_d

        # Predictor (affine scaling) step: sigma = 0.
        aff = newton(w * y, p * z_l, q * z_u)
        a_p, a_d = steplens(*aff)
        dx_a, dw_a, dy_a, dzl_a, dzu_a = aff
        mu_aff = (
            float(
                (w + a_p * dw_a) @ (y + a_d * dy_a)
                + (p + a_p * dx_a) @ (z_l + a_d * dzl_a)
                + (q - a_p * dx_a) @ (z_u + a_d * dzu_a)
            )
            / n_comp
        )
        sigma = min(1.0, max((mu_aff / mu) ** 3, 1e-8))

        # Corrector: recenter toward sigma*mu and cancel second-order terms.
        smu = sigma * mu
        step = newton(
            w * y + dw_a * dy_a - smu,
            p * z_l + dx_a * dzl_a - smu,
            q * z_u - dx_a * dzu_a - smu,
        )
        a_p, a_d = steplens(*step)
        eta = 0.99 if mu > 1e-6 else 0.999
        a_p = min(1.0, eta * a_p)
        a_d = min(1.0, eta * a_d)

        dx, dw, dy, dzl, dzu = step
        x = x + a_p * dx
        w = w + a_p * dw
        y = y + a_d * dy
        z_l = z_l + a_d * dzl
        z_u = z_u + a_d * dzu

    viol = float(np.max(np.maximum(A @ x - b, 0.0)))
    r_d_final = float(np.max(np.abs(c + A.T @ y - z_l + z_u)))
    gap = float(w @ y + (x - l) @ z_l + (u - x) @ z_u)
    return LPResult(
        x=x,
        y=y,
        status=status,
        message=_TERM[status],
        iterations=iteration,
        fun=float(c @ x),
        duality_gap=gap,
        primal_residual=viol,
        dual_residual=r_d_final,
    )
