import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from prysm.mathops import np
from prysm.propagation import prepare_executor
from prysm.x.optym import PrysmLBFGSB
from tqdm import tqdm

from dygdug.backend import asnumpy
from dygdug.coropt import AugmentedLagrangian, VariablePupil
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

    # Outer/inner iteration budget.  Modest by default so the script stays
    # runnable and produces a manageable animation; bump these up for a real
    # design run.
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

    # Display normalization: central wavelength's direct-PSF peak, captured by
    # the optimizer at construction time (does not change appreciably during
    # the solve).  wvl list is [0.95 w, w, 1.05 w] -> central index is 1.
    direct_peak = float(model._normalization[1])

    # Drive the outer loop manually so a frame is captured after each inner solve.
    # Relaxation perturbs only the next inner solve's starting point, so it is
    # disabled on the final outer iteration (matching AugmentedLagrangian.solve).
    # tqdm bars: the outer loop here, the inner loop inside model.step(progress=True).
    frames = []
    for i in tqdm(range(n_outer), desc="outer"):
        is_last = i == n_outer - 1
        model.step(inner_steps=n_inner, apply_relaxation=not is_last, progress=True)
        frames.append(_frame_data(coro, pupil, wvl, direct_peak))

    x = model.x
    pupil.update(x)

    _save_animation(frames, GIF_PATH)
    print(f"saved {n_outer}-frame animation to {GIF_PATH}")
    return


if __name__ == "__main__":
    test_augmented_lagrangian_runs_multiple_outer_iterations_on_segmented_pupil()
