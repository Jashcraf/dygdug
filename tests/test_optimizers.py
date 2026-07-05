"""Unit tests for dygdug.optimizers.linprog (dense Mehrotra IPM).

CPU-only and fast; each case cross-checks against scipy's HiGHS.
"""

import numpy as truenp
import pytest
import scipy.optimize

from dygdug.optimizers import linprog


def _random_feasible_lp(rng, m, n):
    """Random dense LP with box [0, 1] that is feasible by construction."""
    A = rng.standard_normal((m, n))
    x_feas = rng.uniform(0.2, 0.8, n)
    b = A @ x_feas + rng.uniform(0.1, 1.0, m)  # strict slack at x_feas
    c = rng.standard_normal(n)
    return c, A, b


def _reference(c, A, b):
    ref = scipy.optimize.linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), method="highs")
    assert ref.status == 0
    return ref


@pytest.mark.parametrize("seed,m,n", [(0, 30, 120), (1, 80, 50), (2, 5, 200)])
def test_matches_highs_on_random_lps(seed, m, n):
    rng = truenp.random.default_rng(seed)
    c, A, b = _random_feasible_lp(rng, m, n)
    ref = _reference(c, A, b)

    res = linprog(c, A, b, tol=1e-9)
    assert res.status == 0, res.message
    # Same optimum...
    assert res.fun == pytest.approx(ref.fun, rel=1e-6, abs=1e-8)
    # ...and a genuinely feasible point.
    assert res.primal_residual <= 1e-7
    assert float(truenp.min(res.x)) >= -1e-9
    assert float(truenp.max(res.x)) <= 1 + 1e-9


def test_throughput_style_objective_with_redundant_rows():
    """c = -1 (maximize sum) with duplicated rows, as in apodizer LPs."""
    rng = truenp.random.default_rng(3)
    c, A, b = _random_feasible_lp(rng, 40, 100)
    c = -truenp.ones(100)
    A = truenp.vstack([A, A[:10]])  # exact duplicates -> degenerate duals
    b = truenp.concatenate([b, b[:10]])
    ref = _reference(c, A, b)

    res = linprog(c, A, b, tol=1e-9)
    assert res.status == 0, res.message
    assert res.fun == pytest.approx(ref.fun, rel=1e-6)
    assert res.primal_residual <= 1e-7


def test_custom_bounds_and_warm_start():
    rng = truenp.random.default_rng(4)
    n = 60
    A = rng.standard_normal((20, n))
    l = rng.uniform(-2, -1, n)
    u = rng.uniform(1, 2, n)
    x_feas = 0.5 * (l + u)
    b = A @ x_feas + 0.5
    c = rng.standard_normal(n)
    ref = scipy.optimize.linprog(
        c, A_ub=A, b_ub=b, bounds=truenp.column_stack([l, u]), method="highs"
    )

    res = linprog(c, A, b, lower_bounds=l, upper_bounds=u, x0=x_feas, tol=1e-9)
    assert res.status == 0
    assert res.fun == pytest.approx(ref.fun, rel=1e-6)
    assert bool(truenp.all(res.x >= l - 1e-8))
    assert bool(truenp.all(res.x <= u + 1e-8))


def test_infeasible_problem_is_flagged():
    # x_0 <= -1 conflicts with x >= 0.
    A = truenp.zeros((1, 4))
    A[0, 0] = 1.0
    b = truenp.array([-1.0])
    res = linprog(truenp.ones(4), A, b, max_iter=60)
    assert res.status != 0
    assert res.primal_residual > 1e-3


def test_input_validation():
    with pytest.raises(ValueError):
        linprog(truenp.ones(3), truenp.ones((2, 4)), truenp.ones(2))
    with pytest.raises(ValueError):
        linprog(
            truenp.ones(3),
            truenp.ones((2, 3)),
            truenp.ones(2),
            lower_bounds=truenp.inf,
        )
    with pytest.raises(ValueError):
        linprog(
            truenp.ones(3),
            truenp.ones((2, 3)),
            truenp.ones(2),
            lower_bounds=1.0,
            upper_bounds=0.0,
        )


def test_field_box_apodizer_lp_matches_highs():
    """The real use case: the field-box LP from a small coronagraph."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_coropt_units import _tiny_model, WVL

    model = _tiny_model()
    coro, pupil = model.coro, model.coro.pupil
    dh_idx, pup_idx, n = model._dh_idx, model._pupil_idx, model.n_params
    m = dh_idx.size

    Nfoc = coro.fpm(WVL).shape[0]
    A = truenp.empty((m, n), dtype=complex)
    Ebar = truenp.zeros((Nfoc, Nfoc), dtype=complex)
    for k, i in enumerate(dh_idx):
        Ebar.ravel()[i] = 1.0
        coro.reverse(Ebar, WVL, include_fpm=True)
        A[k] = truenp.conj(coro.adjoint_at_entrance_pupil.ravel()[pup_idx])
        Ebar.ravel()[i] = 0.0
    pupil.update(truenp.zeros(n))
    b = coro.forward(WVL, include_fpm=True).ravel()[dh_idx].copy()
    pupil.update(truenp.ones(n))
    norm = float(truenp.max(truenp.abs(coro.forward(WVL, include_fpm=False)) ** 2))

    rn = 1 / truenp.sqrt(norm)
    M, boff = A * rn, b * rn
    A_ub = truenp.vstack([M.real, -M.real, M.imag, -M.imag])
    s = truenp.sqrt(1e-6 / 2)  # modest target keeps HiGHS fast at this scale
    b_ub = s + truenp.concatenate([-boff.real, boff.real, -boff.imag, boff.imag])
    c = -truenp.ones(n)

    ref = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method="highs")
    res = linprog(c, A_ub, b_ub, tol=1e-8)

    assert res.status == 0, res.message
    # Same throughput to 5 significant figures, and strictly feasible:
    assert res.fun == pytest.approx(ref.fun, rel=1e-5)
    assert res.primal_residual <= 1e-9  # amplitude units; s itself is ~7e-4
