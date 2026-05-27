import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from prysm.mathops import np
from prysm.propagation import prepare_executor
from prysm.x.optym import PrysmLBFGSB

from dygdug.coropt import AugmentedLagrangian, VariablePupil
from dygdug.masks import FPM, Pupil
from dygdug.models import Coronagraph


def test_augmented_lagrangian_runs_multiple_outer_iterations_on_segmented_pupil():
    """Smoke-test AugmentedLagrangian on the notebook's VariablePupil setup.

    This follows ``notebooks/coronagraph_optimization_multiple_cost.py`` but
    uses smaller pupil/focal arrays so the test exercises the outer AL loop
    without becoming a long optimization benchmark.
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
        contrast=1e-10,
        x0=x0,
        lower_bounds=np.zeros(x0.size),
        upper_bounds=np.ones(x0.size),
        penalty=1.0,
        throughput_weight=1e-10,
        optimizer_kwargs={"maxls": 10},
    )

    x, history = model.solve(outer_steps=10, inner_steps=1000)

    pupil.update(x)

    coro_img = coro.forward(wvl)
    direct_img = coro.forward(wvl, include_fpm=False)

    coro_img = np.abs(coro_img) ** 2
    direct_img = np.abs(direct_img) ** 2

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(pupil.data, cmap="gray")
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(coro_img / direct_img.max(), norm=LogNorm(vmin=1e-10, vmax=1e-5))
    plt.colorbar()
    plt.show()

    assert model.outer_iter == 4
    assert len(history) == 4
    assert [entry["outer_iter"] for entry in history] == [0, 1, 2, 3]
    assert x.shape == x0.shape
    assert np.all(np.isfinite(x))
    assert np.all(x >= -1e-12)
    assert np.all(x <= 1 + 1e-12)
    assert np.all(np.isfinite(model.multipliers))
    assert history[-1]["max_violation"] >= 0
