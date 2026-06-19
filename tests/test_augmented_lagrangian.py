import os

import matplotlib.pyplot as plt
import numpy as truenp
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from prysm.mathops import ndimage, np, set_backend_to_cupy
from prysm.propagation import prepare_executor
from prysm.x.optym import PrysmLBFGSB, run_until
from tqdm import tqdm
from time import perf_counter

from dygdug.backend import asnumpy
from dygdug.coropt import (
    AugmentedLagrangian,
    CoronagraphOptimizer,
    JointOptimizer,
    ThroughputOptimizer,
    VariablePupil,
)
from dygdug.cost_functions import MeanSquaredErrorQuadratic
from dygdug.masks import FPM, Pupil
from dygdug.models import Coronagraph

# Animated GIF written at the end of the run, one frame per inner solve.
GIF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "al_progress.gif")

# Fixed log-intensity display range so frames are comparable across iterations.
# Brackets the 1e-8 contrast target with a couple decades of headroom.
_DISPLAY_VMIN = 1e-11
_DISPLAY_VMAX = 1e-3


def _frame_data(coro, pupil, wvl, direct_peak):
    """Snapshot the apodizer and normalized coronagraphic intensity.

    Returns owned copies (the underlying arrays are mutated by later
    iterations) of the apodizer amplitude and the coronagraphic focal-plane
    intensity normalized to the direct (no-FPM) peak.
    """
    apodizer = pupil.data
    if np.iscomplexobj(apodizer):
        apodizer = np.abs(apodizer)

    E_focal = coro.forward(wvl, include_fpm=True)
    intensity = (np.abs(E_focal) ** 2) / direct_peak

    # Pull to host numpy (a no-op copy on CPU, a device->host transfer under
    # cupy) so matplotlib can render it, and so later iterations mutating the
    # device arrays do not alias these frames.
    return asnumpy(apodizer).copy(), asnumpy(intensity).copy()


def _apply_relaxation(x, pupil_idx, shape, sigma):
    """Energy-preserving Gaussian-blur relaxation of an active-pixel vector.

    Scatters *x* into a full 2-D apodizer map, blurs it with an isotropic
    Gaussian of std *sigma* (px), re-gathers the active pixels, rescales to
    preserve the total active-pixel transmission, and clips to ``[0, 1]``.  This
    is the standalone analog of ``AugmentedLagrangian._apply_relaxation`` for the
    weighted-sum driver below; ``reshape(-1)`` (not ``.flat``) keeps it
    cupy-compatible, and ``ndimage`` is the prysm backend shim (scipy on CPU,
    cupyx.scipy.ndimage on GPU).
    """
    img = np.zeros(shape, dtype=float)
    img.reshape(-1)[pupil_idx] = x
    blurred = ndimage.gaussian_filter(img, sigma=sigma)
    x_smooth = blurred.reshape(-1)[pupil_idx].copy()

    s_orig = float(np.sum(x))
    s_smooth = float(np.sum(x_smooth))
    if s_smooth > 0:
        x_smooth = x_smooth * (s_orig / s_smooth)
    return np.clip(x_smooth, 0.0, 1.0)


def _save_animation(frames, gif_path, fps=4):
    """Animate the collected frames vs. outer iteration and write a GIF.

    Left panel: the apodizer amplitude solution (gray, linear, with colorbar).
    Right panel: the coronagraphic focal-plane intensity, normalized to the
    direct peak, on a fixed log scale (inferno, with colorbar).
    """
    apod0, int0 = frames[0]

    fig, (ax_apod, ax_int) = plt.subplots(1, 2, figsize=(11, 5))

    im_apod = ax_apod.imshow(apod0, cmap="gray", vmin=0.0, vmax=1.0)
    ax_apod.set_title("apodizer")
    ax_apod.set_xticks([])
    ax_apod.set_yticks([])
    fig.colorbar(im_apod, ax=ax_apod, fraction=0.046, pad=0.04)

    im_int = ax_int.imshow(
        int0,
        cmap="inferno",
        norm=LogNorm(vmin=_DISPLAY_VMIN, vmax=_DISPLAY_VMAX),
    )
    ax_int.set_title("coronagraphic intensity\n(norm. to direct peak)")
    ax_int.set_xticks([])
    ax_int.set_yticks([])
    fig.colorbar(im_int, ax=ax_int, fraction=0.046, pad=0.04)

    title = fig.suptitle("outer iteration 0")
    fig.tight_layout()

    def update(k):
        apod, intensity = frames[k]
        im_apod.set_data(apod)
        im_int.set_data(intensity)
        title.set_text(f"outer iteration {k}")
        return im_apod, im_int, title

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def test_augmented_lagrangian_runs_multiple_outer_iterations_on_segmented_pupil():
    """Smoke-test AugmentedLagrangian on the notebook's VariablePupil setup.

    This follows ``notebooks/coronagraph_optimization_multiple_cost.py`` but
    uses smaller pupil/focal arrays so the test exercises the outer AL loop
    without becoming a long optimization benchmark.  An animated GIF tracking
    the design vs. iteration is written to ``tests/al_progress.gif``.
    """
    circumscribed_diameter = 7.994 * 1e3
    efl = 10 * circumscribed_diameter
    flat_to_flat = 0.955 * 1e3
    gap_size = 0.006 * 1e3
    n_rings = 4
    npup = 512
    nfoc = 128
    wvl = 0.550
    lamD = wvl / circumscribed_diameter * efl
    px_per_lamD = 4
    exclude = [37, 41, 45, 49, 53, 57]

    # Outer/inner iteration budget.  The outer count sets the *resolution* of
    # the graduated-optimization schedules below, so it is deliberately large;
    # the inner solves exit early via the governor (ftol/gtol on the model), so
    # a modest inner cap is fine rather than grinding to deep convergence
    # against a stale multiplier/penalty estimate.
    n_outer = 80
    n_inner = 1000

    # Graduated-optimization schedules (Por, Proc. SPIE 12180, 121805J, 2022).
    # 1) Cool the progressive-relaxation Gaussian kernel from a broad blur
    #    (convexifies the landscape, escapes local minima) down to ~0 px
    #    (recovers segment-scale detail and lets the final solves land sharply).
    # 2) Tighten the dark-hole contrast target by decades (constraint homotopy)
    #    instead of demanding the final hard target from iteration 0.
    # Both are swept geometrically across the outer loop.  Multipliers carry
    # over between outer steps, so each tightened target is warm-started.
    sigma0, sigma_min = 4.0, 0.5      # relaxation kernel std, pixels
    c_start, c_final = 1e-10, 1e-10     # dark-hole contrast target (true contrast)
    # 3) Binarity is NOT enforced with a penalty.  As in standard APLC/shaped-
    #    pupil design, a binary {0, 1} apodizer emerges from *throughput
    #    maximization* driving the design to the vertices of the (linear, field)
    #    contrast-feasible set.  The AL's quadratic penalty term is smooth, so it
    #    smears that vertex into a gray interior point; the *pure* Lagrangian
    #    (-throughput + lambda^T c, c linear) is piecewise-linear and is minimized
    #    *at* a vertex.  So over the final iterations we (a) finish the contrast
    #    homotopy, (b) stop the relaxation blur, (c) disable the inner governor,
    #    and (d) anneal the penalty DOWN toward ~0 so the inner objective
    #    approaches that pure, vertex-seeking Lagrangian using the multipliers
    #    the AL converged earlier.  penalty_growth=1.0 (below) hands control of
    #    the penalty to this manual schedule.
    converge_frac = 0.8               # fraction of the loop after which to converge
    penalty_explore, penalty_min = 10.0, 10   # AL penalty: phase-1 value, then annealed down


    executor = prepare_executor(
        pupil_dx=circumscribed_diameter / npup,
        pupil_samples=npup,
        focal_dx=lamD / px_per_lamD,
        focal_samples=nfoc,
        wavelength=wvl,
        efl=efl,
        focal_shift=(0, 0),
        kind="mdft",
    )

    pupil = VariablePupil.hexagonal_segmented(
        Dpup=circumscribed_diameter,
        Npup=npup,
        rings=n_rings,
        segment_diameter=flat_to_flat,
        segment_separation=gap_size,
        exclude=exclude,
        mode="amplitude",
    )

    fpm = FPM.annular(nfoc, lamD, px_per_lamD, inner_radius=3, outer_radius=12)
    lyot_stop = Pupil.annular(
        Dpup=circumscribed_diameter,
        Npup=npup,
        inner_radius=circumscribed_diameter * 0.10 / 2,
        outer_radius=circumscribed_diameter * 0.95 / 2,
    )
    
    set_backend_to_cupy()
    coro = Coronagraph(pupil=pupil, fpm=fpm, lyot_stop=lyot_stop, executor=executor)

    x0 = pupil.data[pupil.mask].astype(float).copy()
    x0 -= truenp.random.random(x0.shape) / 10
    model = AugmentedLagrangian(
        coro=coro,
        optimizer=PrysmLBFGSB,
        dark_hole=fpm(wvl),
        wvl=[0.95 * wvl, wvl, 1.05 * wvl],
        # Start of the contrast homotopy; the outer loop overwrites
        # model.contrast each step down to c_final.
        contrast=c_start,
        x0=x0,
        lower_bounds=np.zeros(x0.size),
        upper_bounds=np.ones(x0.size),
        penalty=1,
        # Disable automatic penalty growth so the manual anneal-down schedule in
        # the convergence phase is in sole control of the penalty.
        penalty_growth=2.0,
        # Strong throughput pressure (normalized to O(1) mean transmission) is
        # what drives the design to the binary vertices once contrast is met.
        # At weight ~1 there was almost no force left to push transmission to
        # the bounds after feasibility, leaving the apodizer gray; ~50 keeps the
        # objective on the O(1) scale we tuned the penalty against while making
        # throughput maximization the dominant driver in the feasible region.
        throughput_weight=1.0,
        # Normalize the throughput objective by the active-pixel count so it is
        # O(1) (mean transmission) rather than O(n_params), keeping it on the
        # same scale as the (true-contrast) penalty term.
        normalize_throughput=True,
        # Linear field constraint (|Re a| <= s, |Im a| <= s): an LP whose feasible
        # polytope has binary vertices, so throughput maximization yields a binary
        # apodizer -- unlike the smooth-boundary intensity constraint, which is
        # minimized at gray interior points.
        constraint_kind="field",
        optimizer_kwargs={"maxls": 50},
        # Initial relaxation kernel; the outer loop cools it toward sigma_min.
        relaxation_sigma=sigma0,
        # Governor: end an inner solve early once the objective plateaus
        # (ftol) or its gradient is small (gtol), so the modest n_inner cap
        # is rarely the binding stop and multipliers update more often.
        ftol=1e-4,
        gtol=1e-6,
    )

    direct_peak = float(model._normalization[1])

    # Drive the outer loop manually so a frame is captured after each inner solve.
    # Relaxation perturbs only the next inner solve's starting point, so it is
    # disabled on the final outer iteration (matching AugmentedLagrangian.solve).
    # tqdm bars: the outer loop here, the inner loop inside model.step(progress=True).
    frames = []
    outer_bar = tqdm(range(n_outer), desc="outer")
    for i in outer_bar:
        is_last = i == n_outer - 1
        frac = i / max(n_outer - 1, 1)

        # Two-phase graduated optimization, applied by mutating the model in
        # place between inner solves:
        #   phase 1 (frac < converge_frac) -- explore: cool the relaxation
        #     kernel, tighten contrast to c_final, governor + blur on.
        #   phase 2 (frac >= converge_frac) -- converge: hold contrast at
        #     c_final, stop the blur, and disable the governor so the solves run
        #     to convergence and throughput maximization drives the design to
        #     the (binary) vertices of the contrast-feasible set.
        sigma_k = sigma0 * (sigma_min / sigma0) ** frac
        # contrast homotopy completes by the convergence phase, then holds
        c_frac = min(1.0, frac / converge_frac)
        c_k = c_start * (c_final / c_start) ** c_frac
        converging = frac >= converge_frac

        model.relaxation_sigma = sigma_k
        model.contrast[:] = c_k  # in-place: preserves (n_wvl, n_constraints) shape
        # Run the inner solve to convergence in phase 2 (vertices only emerge if
        # L-BFGS-B is not early-exited on a smooth interior point).
        model.ftol = None if converging else 1e-4
        model.gtol = None if converging else 1e-6

        # Anneal the AL penalty DOWN in the convergence phase so the inner
        # objective approaches the pure (piecewise-linear) Lagrangian, whose box
        # minimizer is a binary vertex.  In phase 1 the penalty stays at
        # penalty_explore to converge the multipliers / enforce feasibility.
        if converging:
            b = (frac - converge_frac) / (1.0 - converge_frac)  # 0 -> 1
            # model.penalty = penalty_explore * (penalty_min / penalty_explore) ** b
        # else:
        #     model.penalty = penalty_explore

        # Blur re-grays the apodizer each restart, so disable it while converging
        # (also skipped on the last step).
        apply_relax = (not is_last) and not converging

        _, info = model.step(
            inner_steps=n_inner, apply_relaxation=apply_relax, progress=True
        )
        # Watch feasibility descend and -- the quantity that actually drives
        # binarity in APLC designs -- mean transmission (throughput) climb.
        outer_bar.set_postfix(
            sigma=f"{sigma_k:.2f}",
            c=f"{c_k:.1e}",
            rho=f"{model.penalty:.2g}",
            thpt=f"{info['throughput'] / model.n_params:.3f}",
            viol=f"{info['max_violation']:.1e}",
        )
        frames.append(_frame_data(coro, pupil, wvl, direct_peak))

    x = model.x
    pupil.update(x)

    # Quantify how binary the final apodizer is: fraction of active pixels
    # within 5% of a bound.  Near 1.0 means the binarization schedule succeeded.
    near_binary = float(np.mean((x < 0.05) | (x > 0.95)))
    mean_transmission = float(np.mean(x))
    print(f"near-binary fraction: {near_binary:.1%}")
    print(f"mean transmission (throughput): {mean_transmission:.3f}")

    _save_animation(frames, GIF_PATH)
    print(f"saved {n_outer}-frame animation to {GIF_PATH}")
    return


# GIF for the weighted-sum (MSE + throughput) driver below.
JOINT_GIF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "joint_progress.gif"
)


def test_joint_mse_throughput_binary():
    """Reproduce the proven shaped-pupil recipe: minimize dark-hole intensity
    MSE minus mu * throughput as a single unconstrained weighted sum.

    Unlike :func:`...augmented_lagrangian...`, there is no hard contrast
    constraint, no Lagrange multipliers, and no penalty schedule -- just one
    ``PrysmLBFGSB`` minimization of
        J(x) = alpha_mse * sum_dh |E|^4  -  mu * sum(apodizer).
    ``mu`` (the throughput weight) is the scalar to grid-search: too small and
    the dark hole wins (low throughput, gray); too large and throughput wins
    (the dark hole is destroyed).  The contrast weight ``alpha_mse`` is held
    fixed.  Binarity is expected to emerge from the throughput term, as in the
    grid-search that reached 1e-12 binary solutions.
    """
    circumscribed_diameter = 7.994 * 1e3
    efl = 10 * circumscribed_diameter
    flat_to_flat = 0.955 * 1e3
    gap_size = 0.006 * 1e3
    n_rings = 4
    npup = 512
    nfoc = 128
    wvl = 0.550
    lamD = wvl / circumscribed_diameter * efl
    px_per_lamD = 4
    exclude = [37, 41, 45, 49, 53, 57]

    # The two objective weights, from the working notebook
    # (coronagraph_optimization_multiple_cost.py).  alpha_mse is fixed; mu is the
    # tiny throughput tie-breaker that selects the binary design among the
    # dark-hole-feasible solutions -- sweep mu around this value to trade
    # contrast vs throughput/binarity.
    alpha_mse = 1.0
    mu = 9e-11

    # Outer/inner budget.  Each outer iteration runs an inner PrysmLBFGSB solve
    # of the weighted-sum objective; between solves the converged apodizer is
    # blurred (progressive relaxation, Por 2022) to perturb the next solve's
    # start out of local minima.  The blur kernel is cooled over the outer loop.
    n_outer = 1
    n_inner = 500
    sigma0, sigma_min = 4.0, 0.5      # relaxation kernel std (px), cooled

    executor = prepare_executor(
        pupil_dx=circumscribed_diameter / npup,
        pupil_samples=npup,
        focal_dx=lamD / px_per_lamD,
        focal_samples=nfoc,
        wavelength=wvl,
        efl=efl,
        focal_shift=(0, 0),
        kind="mdft",
    )

    pupil = VariablePupil.hexagonal_segmented(
        Dpup=circumscribed_diameter,
        Npup=npup,
        rings=n_rings,
        segment_diameter=flat_to_flat,
        segment_separation=gap_size,
        exclude=exclude,
        mode="amplitude",
    )

    fpm = FPM.annular(nfoc, lamD, px_per_lamD, inner_radius=3, outer_radius=12)
    lyot_stop = Pupil.annular(
        Dpup=circumscribed_diameter,
        Npup=npup,
        inner_radius=circumscribed_diameter * 0.00 / 2,
        outer_radius=circumscribed_diameter * 0.95 / 2,
    )

    set_backend_to_cupy()
    coro = Coronagraph(pupil=pupil, fpm=fpm, lyot_stop=lyot_stop, executor=executor)

    wvls = [0.95 * wvl, wvl, 1.05 * wvl]

    # The proven weighted-sum objective: dark-hole intensity MSE minus
    # mu * throughput, assembled from existing dygdug pieces.  Constructing these
    # runs sync_coronagraph, moving the coronagraph's arrays (pupil data/mask,
    # FPM cache, executor matrices) onto the active backend (cupy).
    joint = JointOptimizer([
        CoronagraphOptimizer(
            dark_hole=fpm(wvl).astype(bool),
            coro=coro,
            wvl=wvls,
            cost=MeanSquaredErrorQuadratic(target=0, alpha=alpha_mse),
        ),
        ThroughputOptimizer(coro, alpha=mu),
    ])

    # Everything below must live on the active backend now that the model does;
    # build x0 on the host (so the numpy perturbation is valid), then move it on.
    dark_hole = np.asarray(fpm(wvl)).astype(bool)
    x0 = asnumpy(pupil.data[pupil.mask]).astype(float)
    x0 -= truenp.random.random(x0.shape) / 10
    x0 = np.asarray(x0, dtype=float)

    # Direct (no-FPM) peak for normalizing the displayed/printed contrast.
    pupil.update(x0)
    E_direct = coro.forward(wvl, include_fpm=False)
    direct_peak = float(np.max(np.abs(E_direct) ** 2))

    # Pupil scatter index + map shape for the relaxation helper (on-backend
    # after sync_coronagraph ran during the JointOptimizer construction above).
    pupil_idx = pupil._mask_idx
    shape = pupil.data.shape

    # Outer loop: an inner weighted-sum solve, then progressive relaxation of the
    # converged apodizer to seed the next solve out of local minima.  A fresh
    # PrysmLBFGSB is built each outer iteration (the relaxed start makes the old
    # L-BFGS history stale).  Relaxation is skipped on the final iteration, whose
    # unrelaxed inner-solve output is the returned design; the blur kernel is
    # cooled geometrically across the loop.
    x_start = x0.copy()
    xf = x0
    frames = []
    t1 = perf_counter()
    inner_bar = tqdm(range(n_inner), desc="inner L-BFGS-B")
    for i in range(n_outer):
        is_last = i == n_outer - 1

        opt = PrysmLBFGSB(
            joint.fg,
            x_start.copy(),
            lower_bounds=np.zeros(x0.size),
            upper_bounds=np.ones(x0.size),
            maxls=1,
        )
        for _ in inner_bar:
            try:
                opt.step()
            except StopIteration:
                break
        xf = opt.x.copy()

        pupil.update(xf)
        frames.append(_frame_data(coro, pupil, wvl, direct_peak))

        if is_last:
            x_start = xf
        else:
            sigma_k = sigma0 * (sigma_min / sigma0) ** (i / max(n_outer - 1, 1))
            x_start = _apply_relaxation(xf, pupil_idx, shape, sigma_k)

        inner_bar.set_postfix(
            thpt=f"{float(np.mean(xf)):.3f}",
            nbin=f"{float(np.mean((xf < 0.05) | (xf > 0.95))):.0%}",
        )
    print(f"Time to complete optimization = {perf_counter() - t1}")
    pupil.update(xf)

    near_binary = float(np.mean((xf < 0.05) | (xf > 0.95)))
    mean_transmission = float(np.mean(xf))
    I_dh = np.abs(coro.forward(wvl, include_fpm=True)[dark_hole]) ** 2
    mean_contrast = float(np.mean(I_dh)) / direct_peak
    print(f"mu = {mu:.3g}, alpha_mse = {alpha_mse:.3g}")
    print(f"near-binary fraction: {near_binary:.1%}")
    print(f"mean transmission (throughput): {mean_transmission:.3f}")
    print(f"mean dark-hole contrast: {mean_contrast:.2e}")

    if frames:
        _save_animation(frames, JOINT_GIF_PATH)
        print(f"saved {len(frames)}-frame animation to {JOINT_GIF_PATH}")
    return


if __name__ == "__main__":
    # AL route (constraint-based); kept for comparison:
    #test_augmented_lagrangian_runs_multiple_outer_iterations_on_segmented_pupil()
    test_joint_mse_throughput_binary()
