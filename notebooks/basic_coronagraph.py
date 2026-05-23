# %%
import matplotlib.pyplot as plt
from prysm.mathops import np

from dygdug.masks import Pupil

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
