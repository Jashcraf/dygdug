# %%
import matplotlib.pyplot as plt
from prysm.mathops import np

from dygdug.masks import FPM, Pupil

# %%
# Drawing the LUVOIR-B Pupil
CIRCUMSCRIBED_DIAMETER = 7.994
FLAT_TO_FLAT = 0.955
GAP_SIZE = 0.006
N_RINGS = 4
Npup = 512
exclude = [37, 41, 45, 49, 53, 57]

pupil = Pupil.hexagonal_segmented(
    Dpup=CIRCUMSCRIBED_DIAMETER,
    Npup=Npup,
    rings=N_RINGS,
    segment_diameter=FLAT_TO_FLAT,
    segment_separation=GAP_SIZE,
    exclude=exclude,
)

plt.figure()
plt.imshow(pupil.data)
plt.xticks([], [])
plt.yticks([], [])
plt.show()
# %%
# Now draw an annular focal plane mask
inner_radius = 3
outer_radius = 12
Nfoc = 256
EFL = 10 * CIRCUMSCRIBED_DIAMETER
WVL = 550e-9
lamD = WVL / (CIRCUMSCRIBED_DIAMETER) * EFL
px_per_lamD = 8
fpm_func = FPM.annular(Nfoc, lamD, px_per_lamD, inner_radius, outer_radius)

plt.figure()
plt.imshow(fpm_func(WVL))
plt.xticks([], [])
plt.yticks([], [])
plt.show()
# %%
# Draw an Annular Lyot Stop with a 95% Circumscribed diameter
inner_radius = CIRCUMSCRIBED_DIAMETER * 0.10 / 2
outer_radius = CIRCUMSCRIBED_DIAMETER * 0.95 / 2
lyot_stop = Pupil.annular(
    Dpup=CIRCUMSCRIBED_DIAMETER,
    Npup=Npup,
    inner_radius=inner_radius,
    outer_radius=outer_radius,
)

plt.figure()
plt.imshow(lyot_stop.data)
plt.xticks([], [])
plt.yticks([], [])
plt.show()
