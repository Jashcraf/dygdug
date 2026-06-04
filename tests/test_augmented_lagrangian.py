import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from prysm.mathops import np
from prysm.propagation import prepare_executor
from prysm.x.optym import PrysmLBFGSB
from tqdm import tqdm

from dygdug.coropt import AugmentedLagrangian, VariablePupil
from dygdug.masks import FPM, Pupil
from dygdug.models import Coronagraph

# Directory where per-inner-iteration progress figures are written.
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "al_progress")

# Fixed log-intensity display range so frames are comparable across iterations.
# Brackets the 1e-8 contrast target with a couple decades of headroom.
_DISPLAY_VMIN = 1e-11
_DISPLAY_VMAX = 1e-3


def _save_progress_figure(coro, pupil, wvl, direct_peak, outer_iter, outdir):
    """Save the apodizer and log-scaled coronagraphic intensity for one outer iter.

    Left panel: the apodizer amplitude solution (gray, linear, with colorbar).
    Right panel: the coronagraphic focal-plane intensity, normalized to the
    direct (no-FPM) peak and shown on a fixed log scale (inferno, with colorbar).

    Called at the conclusion of each inner optimization so the saved frames
    track convergence of the design.
    """
    apodizer = np.asarray(pupil.data)
    if np.iscomplexobj(apodizer):
        apodizer = np.abs(apodizer)

    E_focal = coro.forward(wvl, include_fpm=True)
    intensity = (np.abs(E_focal) ** 2) / direct_peak

    fig, (ax_apod, ax_int) = plt.subplots(1, 2, figsize=(11, 5))

    im_apod = ax_apod.imshow(apodizer, cmap="gray")
    ax_apod.set_title("apodizer")
    ax_apod.set_xticks([])
    ax_apod.set_yticks([])
    fig.colorbar(im_apod, ax=ax_apod, fraction=0.046, pad=0.04)

    im_int = ax_int.imshow(
        intensity,
        cmap="inferno",
        norm=LogNorm(vmin=_DISPLAY_VMIN, vmax=_DISPLAY_VMAX),
    )
    ax_int.set_title("coronagraphic intensity\n(norm. to direct peak)")
    ax_int.set_xticks([])
    ax_int.set_yticks([])
    fig.colorbar(im_int, ax=ax_int, fraction=0.046, pad=0.04)

    fig.suptitle(f"outer iteration {outer_iter}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"outer_{outer_iter:03d}.png"), dpi=120)
    plt.close(fig)


def test_augmented_lagrangian_runs_multiple_outer_iterations_on_segmented_pupil():
    """Smoke-test AugmentedLagrangian on the notebook's VariablePupil setup.

    This follows ``notebooks/coronagraph_optimization_multiple_cost.py`` but
    uses smaller pupil/focal arrays so the test exercises the outer AL loop
    without becoming a long optimization benchmark.  A progress figure is
    written to ``tests/al_progress/`` at the conclusion of each inner solve.
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

    # Outer/inner iteration budget.  Modest by default so the script stays
    # runnable and produces a manageable number of progress frames; bump these
    # up for a real design run.
    n_outer = 20
    n_inner = 500

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
    coro = Coronagraph(pupil=pupil, fpm=fpm, lyot_stop=lyot_stop, executor=executor)

    x0 = pupil.data[pupil.mask].astype(float).copy()
    model = AugmentedLagrangian(
        coro=coro,
        optimizer=PrysmLBFGSB,
        dark_hole=fpm(wvl),
        wvl=[0.95 * wvl, wvl, 1.05 * wvl],
        # Contrast is now a true contrast relative to the direct (no-FPM) peak,
        # so 1e-8 means 1e-8 of the on-axis stellar peak in the dark hole.
        contrast=1e-8,
        x0=x0,
        lower_bounds=np.zeros(x0.size),
        upper_bounds=np.ones(x0.size),
        penalty=1.0,
        # Throughput weight of order unity now that the penalty term is scaled
        # in true-contrast units; this acts as the initial objective weighting.
        throughput_weight=1.0,
        optimizer_kwargs={"maxls": 10},
        relaxation_sigma=5,
    )

    os.makedirs(FIGURE_DIR, exist_ok=True)

    # Display normalization: central wavelength's direct-PSF peak, captured by
    # the optimizer at construction time (does not change appreciably during
    # the solve).  wvl list is [0.95 w, w, 1.05 w] -> central index is 1.
    direct_peak = float(model._normalization[1])

    # Drive the outer loop manually so a figure is saved after each inner solve.
    # Relaxation perturbs only the next inner solve's starting point, so it is
    # disabled on the final outer iteration (matching AugmentedLagrangian.solve).
    # tqdm bars: the outer loop here, the inner loop inside model.step(progress=True).
    for i in tqdm(range(n_outer), desc="outer"):
        is_last = i == n_outer - 1
        model.step(inner_steps=n_inner, apply_relaxation=not is_last, progress=True)
        _save_progress_figure(coro, pupil, wvl, direct_peak, i, FIGURE_DIR)

    x = model.x
    pupil.update(x)
    print(f"saved {n_outer} progress figures to {FIGURE_DIR}")
    return


if __name__ == "__main__":
    test_augmented_lagrangian_runs_multiple_outer_iterations_on_segmented_pupil()
