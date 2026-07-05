"""Fast, CPU-only unit tests for dygdug.coropt / dygdug.cost_functions.

Unlike test_augmented_lagrangian.py (a long-running GPU benchmark script),
these are real unit tests: a tiny 64x64-pupil coronagraph, seconds to run,
no GPU, no plots.  Run with ``pytest tests/test_coropt_units.py``.
"""

import numpy as truenp
import pytest
from prysm.propagation import prepare_executor
from prysm.x.optym import PrysmLBFGSB

from dygdug.coropt import AugmentedLagrangian, VariablePupil
from dygdug.cost_functions import FieldBoxConstraint, ScaledFieldBoxConstraint
from dygdug.masks import FPM, Pupil
from dygdug.models import Coronagraph

WVL = 1.0


def _tiny_model(**overrides):
    """Build a small AugmentedLagrangian problem (64x64 pupil, 32x32 focal)."""
    Dpup = 1e3
    Npup = 64
    Nfoc = 32
    fno = 20.0
    efl = fno * Dpup
    lamD = WVL / Dpup * efl
    px_per_lamD = 4

    pupil = VariablePupil.circle(Dpup=Dpup, Npup=Npup, mode="amplitude")
    executor = prepare_executor(
        pupil_dx=Dpup / Npup,
        pupil_samples=Npup,
        focal_dx=lamD / px_per_lamD,
        focal_samples=Nfoc,
        wavelength=WVL,
        efl=efl,
        focal_shift=(0, 0),
        kind="mdft",
    )
    fpm = FPM.annular(Nfoc, lamD, px_per_lamD, inner_radius=2, outer_radius=6)
    lyot = Pupil.annular(Dpup, Npup, inner_radius=0.1 * Dpup / 2, outer_radius=0.8 * Dpup / 2)
    coro = Coronagraph(pupil=pupil, fpm=fpm, lyot_stop=lyot, executor=executor)

    n = pupil.n_params
    kwargs = dict(
        coro=coro,
        optimizer=PrysmLBFGSB,
        dark_hole=fpm(WVL),
        wvl=[WVL],
        contrast=1e-8,
        x0=truenp.ones(n),
        lower_bounds=truenp.zeros(n),
        upper_bounds=truenp.ones(n),
        constraint_kind="scaled-field",
        normalize_throughput=True,
        penalty=1.0,
        penalty_growth=2.0,
    )
    kwargs.update(overrides)
    return AugmentedLagrangian(**kwargs)


# ----------------------------------------------------------------------------
# ScaledFieldBoxConstraint
# ----------------------------------------------------------------------------
def test_scaled_field_constraint_is_rescaled_field_constraint():
    """Scaled residuals are the raw residuals / s; the feasible set matches."""
    rng = truenp.random.default_rng(7)
    n = 50
    E_dh = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    norm = 3.7e4
    contrast = truenp.full(n, 1e-9)
    s = truenp.sqrt(contrast / 2)

    raw = FieldBoxConstraint()
    scaled = ScaledFieldBoxConstraint()

    c_raw = raw.residual(E_dh, norm, contrast)
    c_scaled = scaled.residual(E_dh, norm, contrast)
    truenp.testing.assert_allclose(c_scaled, c_raw / s, rtol=1e-13)
    # Same sign everywhere -> identical feasible set.
    assert truenp.all((c_scaled <= 0) == (c_raw <= 0))

    # grad_seed: c_scaled = c_raw / s, so the seed with multipliers `pos` must
    # equal the raw seed with multipliers `pos / s`.
    pos = rng.uniform(0, 2, size=(2, n))
    g_scaled = scaled.grad_seed(E_dh, norm, contrast, pos)
    g_ref = raw.grad_seed(E_dh, norm, contrast, pos / s)
    truenp.testing.assert_allclose(g_scaled, g_ref, rtol=1e-13)

    # Feasible residuals are O(1): lower bound is exactly -1 (at E = 0).
    c_zero = scaled.residual(truenp.zeros(n, dtype=complex), norm, contrast)
    truenp.testing.assert_allclose(c_zero, -1.0, rtol=1e-13)


def test_constraint_kind_scaled_field_selects_scaled_constraint():
    model = _tiny_model()
    assert isinstance(model.constraint, ScaledFieldBoxConstraint)
    with pytest.raises(ValueError):
        _tiny_model(constraint_kind="bogus")


# ----------------------------------------------------------------------------
# fg: analytic gradient vs finite differences (constraint + huberized tent)
# ----------------------------------------------------------------------------
def test_fg_gradient_matches_finite_differences():
    """End-to-end check of dJ/dx with the scaled-field constraint and the
    smoothed (Huber) tent binarization penalty both active."""
    model = _tiny_model(
        binarize_kind="tent",
        binarize_weight=3.0,
        binarize_smoothing=0.05,
        throughput_weight=2.0,
    )
    rng = truenp.random.default_rng(3)
    x = rng.uniform(0.2, 0.8, size=model.n_params)
    # Activate the constraint penalty: nonzero multipliers everywhere.
    model.multipliers[:] = 0.5
    model.penalty = 2.0

    _, g = model.fg(x.copy())

    eps = 1e-6
    idx = rng.choice(model.n_params, size=12, replace=False)
    for i in idx:
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        fp, _ = model.fg(xp)
        fm, _ = model.fg(xm)
        g_fd = (fp - fm) / (2 * eps)
        truenp.testing.assert_allclose(g[i], g_fd, rtol=2e-4, atol=1e-9)


def test_smoothed_tent_matches_exact_tent_outside_kink():
    """Outside the Huber region the smoothed tent is the exact tent."""
    smooth = _tiny_model(binarize_kind="tent", binarize_weight=5.0,
                         binarize_smoothing=0.05)
    exact = _tiny_model(binarize_kind="tent", binarize_weight=5.0,
                        binarize_smoothing=0.0)
    rng = truenp.random.default_rng(11)
    # All pixels well away from mid-gray (|x - 0.5| > delta).
    x = truenp.where(rng.random(smooth.n_params) < 0.5,
                     rng.uniform(0.05, 0.3, smooth.n_params),
                     rng.uniform(0.7, 0.95, smooth.n_params))
    f_s, g_s = smooth.fg(x.copy())
    f_e, g_e = exact.fg(x.copy())
    truenp.testing.assert_allclose(f_s, f_e, rtol=1e-12)
    truenp.testing.assert_allclose(g_s, g_e, rtol=1e-12)


# ----------------------------------------------------------------------------
# Penalty growth guard
# ----------------------------------------------------------------------------
def test_penalty_grows_only_when_violation_stalls():
    # Impossible contrast so the design is always violated; inner_steps=0 keeps
    # x fixed, so the violation is exactly stagnant across steps.
    model = _tiny_model(contrast=1e-16, penalty=1.0, penalty_growth=2.0)

    _, info0 = model.step(inner_steps=0, apply_relaxation=False)
    # First outer iteration: no previous violation to compare -> no growth.
    assert info0["max_violation"] > 0
    assert info0["penalty_after"] == info0["penalty_before"] == 1.0

    _, info1 = model.step(inner_steps=0, apply_relaxation=False)
    # Stagnant violation -> growth fires.
    assert info1["penalty_after"] == pytest.approx(2.0 * info1["penalty_before"])

    # Feasible-within-tolerance -> no growth even though violation is stagnant.
    model.constraint_tolerance = 2 * model.last_violation
    _, info2 = model.step(inner_steps=0, apply_relaxation=False)
    assert info2["penalty_after"] == info2["penalty_before"]


# ----------------------------------------------------------------------------
# pin(): fix-and-release rounding
# ----------------------------------------------------------------------------
def test_pin_pins_decided_pixels_and_is_idempotent():
    model = _tiny_model()
    n = model.n_params
    rng = truenp.random.default_rng(5)
    x = rng.uniform(0.3, 0.7, size=n)
    x[:10] = 0.01   # decided low
    x[10:20] = 0.99  # decided high
    model.x = x.copy()
    model._x_start = x.copy()

    info = model.pin(threshold=0.05)
    assert info["n_pinned"] == 20
    assert info["n_pinned_total"] == 20
    assert info["n_free"] == n - 20
    truenp.testing.assert_array_equal(model.x[:10], 0.0)
    truenp.testing.assert_array_equal(model.x[10:20], 1.0)
    truenp.testing.assert_array_equal(model.lower_bounds[:20], model.upper_bounds[:20])
    # Undecided pixels untouched, bounds still open.
    truenp.testing.assert_array_equal(model.x[20:], x[20:])
    assert truenp.all(model.upper_bounds[20:] > model.lower_bounds[20:])

    # Idempotent: nothing new to pin at the same threshold.
    info = model.pin(threshold=0.05)
    assert info["n_pinned"] == 0

    # threshold=0.5 pins every remaining pixel to its nearest bound -> binary.
    info = model.pin(threshold=0.5)
    assert info["n_free"] == 0
    assert truenp.all((model.x == 0.0) | (model.x == 1.0))
    truenp.testing.assert_array_equal(
        model.x[20:], (x[20:] > 0.5).astype(float)
    )


def test_pinned_pixels_survive_a_step():
    """After pinning, an inner solve cannot move the pinned pixels."""
    model = _tiny_model(contrast=1e-4)
    n = model.n_params
    x = truenp.full(n, 0.6)
    x[:5] = 0.02
    model.x = x.copy()
    model._x_start = x.copy()

    model.pin(threshold=0.05)
    model.step(inner_steps=5, apply_relaxation=False)
    truenp.testing.assert_array_equal(model.x[:5], 0.0)
