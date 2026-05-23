# %%
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from prysm.mathops import np
from prysm.propagation import focus_dft, prepare_executor, to_fpm_and_back

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
# Drawing the LUVOIR-B Pupil
pupil = Pupil.hexagonal_segmented(
    Dpup=CIRCUMSCRIBED_DIAMETER,
    Npup=Npup,
    rings=N_RINGS,
    segment_diameter=FLAT_TO_FLAT,
    segment_separation=GAP_SIZE,
    exclude=exclude,
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

psf = coro.forward(WVL)

plt.figure()
plt.imshow(np.abs(psf) ** 2, cmap="gray", norm=LogNorm())
plt.colorbar()
plt.show()
