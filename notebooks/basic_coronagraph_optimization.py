# %%
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from prysm.mathops import np
from prysm.propagation import focus_dft, prepare_executor, to_fpm_and_back
from prysm.x.optym import PrysmLBFGSB
from tqdm import tqdm

from dygdug.coropt import CoronagraphOptimizer, VariablePupil
from dygdug.masks import FPM, Pupil
from dygdug.models import Coronagraph

# %%
# Begin by preparing the propagation executor
# Drawing the LUVOIR-B Pupil
CIRCUMSCRIBED_DIAMETER = 7.994 * 1e3
EFL = 10 * CIRCUMSCRIBED_DIAMETER
FLAT_TO_FLAT = 0.955 * 1e3
GAP_SIZE = 0.006 * 1e3
N_RINGS = 4
Npup = 512
exclude = [37, 41, 45, 49, 53, 57]
inner_radius = 3
outer_radius = 12
Nfoc = 256
WVL = 0.550
lamD = WVL / (CIRCUMSCRIBED_DIAMETER) * EFL
px_per_lamD = 8

mdft = prepare_executor(
    pupil_dx=CIRCUMSCRIBED_DIAMETER / Npup,  # prefers mm
    pupil_samples=Npup,
    focal_dx=lamD / px_per_lamD,  # prefers microns
    focal_samples=Nfoc,
    wavelength=WVL,  # prefers microns
    efl=EFL,  # prefers mm
    focal_shift=(0, 0),
    kind="mdft",
)

# %%
# Drawing the LUVOIR-B Pupil - Transmissive areas are optimized
pupil = VariablePupil.hexagonal_segmented(
    Dpup=CIRCUMSCRIBED_DIAMETER,
    Npup=Npup,
    rings=N_RINGS,
    segment_diameter=FLAT_TO_FLAT,
    segment_separation=GAP_SIZE,
    exclude=exclude,
    mode="amplitude",
)

# Now draw an annular focal plane mask
fpm_func = FPM.annular(Nfoc, lamD, px_per_lamD, inner_radius, outer_radius)

# Draw an Annular Lyot Stop with a 95% Circumscribed diameter
inner_radius = CIRCUMSCRIBED_DIAMETER * 0.10 / 2
outer_radius = CIRCUMSCRIBED_DIAMETER * 0.95 / 2
lyot_stop = Pupil.annular(
    Dpup=CIRCUMSCRIBED_DIAMETER,
    Npup=Npup,
    inner_radius=inner_radius,
    outer_radius=outer_radius,
)

# %%
# Perform propagation
coro = Coronagraph(
    pupil=pupil,
    fpm=fpm_func,
    lyot_stop=lyot_stop,
    executor=mdft,
)

model = CoronagraphOptimizer(
    dark_hole=fpm_func(WVL),
    coro=coro,
    wvl=WVL,
)

x0 = pupil.data[pupil.mask].astype(float).copy()
lb = np.zeros(model.n_params)
ub = np.ones(model.n_params)

opt = PrysmLBFGSB(model.fg, x0, lower_bounds=lb, upper_bounds=ub)

pbar = tqdm(range(1000))
for i in pbar:
    x, f, g = opt.step()
    pbar.set_postfix(f=f)

print(f"final f: {f}")


# %%
pupil.update(opt.x)

plt.figure()
plt.imshow(pupil.data, cmap="gray")
plt.colorbar()
plt.show()
# %%
coro = Coronagraph(
    pupil=pupil,
    fpm=fpm_func,
    lyot_stop=lyot_stop,
    executor=mdft,
)
from matplotlib.colors import LogNorm

ref = np.abs(coro.forward(WVL, include_fpm=False)) ** 2
img = np.abs(coro.forward(WVL)) ** 2
plt.figure()
plt.title("Coronagraph Image")
plt.imshow(img / ref.max(), cmap="gray", norm=LogNorm(vmin=1e-10, vmax=1e-5))
plt.colorbar()
plt.show()

plt.figure()
plt.title("Direct Image")
plt.imshow(ref / ref.max(), cmap="gray", norm=LogNorm(vmin=1e-10, vmax=1))
plt.colorbar()
plt.show()
