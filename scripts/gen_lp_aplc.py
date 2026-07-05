
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from time import perf_counter

from prysm.mathops import np, set_backend_to_cupy
from prysm.propagation import prepare_executor

from dygdug.masks import Pupil, FPM, ImgSamplingSpec, annular_mask, knife_edge_mask
from dygdug.coropt import VariablePupil
from dygdug.models import Coronagraph

# Define instrument parameters
circumscribed_diameter = 10.95e3
Npup = 1024
Nfoc = 256
wvl = 1
fno = 13.66
efl = fno * circumscribed_diameter
lamD = wvl / circumscribed_diameter * efl
px_per_lamD = 8  # the "oversampling"

# Variable (amplitude-apodized) entrance pupil
pupil = VariablePupil.circle(Dpup=circumscribed_diameter, Npup=Npup, mode="amplitude")

executor = prepare_executor(
    pupil_dx=circumscribed_diameter / Npup,
    pupil_samples=Npup,
    focal_dx=lamD / px_per_lamD,
    focal_samples=Nfoc,
    wavelength=wvl,
    efl=efl,
    focal_shift=(0, 0),
    kind="mdft",
)

fpm = FPM.annular(
    N=Nfoc,
    lamD=lamD,
    px_per_lamD=px_per_lamD,
    inner_radius=3,
    outer_radius=8,
)

lyot = Pupil.annular(
    Dpup=circumscribed_diameter,
    Npup=Npup,
    inner_radius=0.1 * circumscribed_diameter / 2,
    outer_radius=0.8 * circumscribed_diameter / 2,
)

coro = Coronagraph(pupil=pupil, fpm=fpm, lyot_stop=lyot, executor=executor)

iss = ImgSamplingSpec(Nfoc, px_per_lamD, lamD=lamD)
dark_hole = annular_mask(iss, iwa=3, owa=8, theta_min=-90, theta_max=90)
dark_hole *= knife_edge_mask(iss, iwa=3)

dh_idx = np.flatnonzero(np.asarray(dark_hole).astype(bool).ravel())
pup_idx = pupil._mask_idx
n = pupil.n_params  # optimizable pupil pixels
m = dh_idx.size     # constrained dark-hole pixels

contrast_req = 1e-10

print(f"n = {n} pupil pixels, m = {m} dark-hole pixels -> 4m = {4*m} constraint rows")
print(f"n / 4m = {n / (4*m):.1f}  (want >> 1)")

t0 = perf_counter()
A = np.empty((m, n), dtype=complex)
Ebar = np.zeros((Nfoc, Nfoc), dtype=complex)
for k, i in enumerate(dh_idx):
    Ebar.ravel()[i] = 1.0
    coro.reverse(Ebar, wvl, include_fpm=True)
    A[k] = np.conj(coro.adjoint_at_entrance_pupil.ravel()[pup_idx])
    Ebar.ravel()[i] = 0.0
print(f"Jacobian ({m} adjoint propagations): {perf_counter()-t0:.1f} s, "
      f"{A.nbytes/1e6:.0f} MB")

# Offset from the frozen (anti-aliased rim) pixels: the field with x = 0.
pupil.update(np.zeros(n))
b = coro.forward(wvl, include_fpm=True).ravel()[dh_idx].copy()

# Contrast normalization: direct (no-FPM) peak of the open pupil.
pupil.update(np.ones(n))
norm = float(np.max(np.abs(coro.forward(wvl, include_fpm=False)) ** 2))

# Verify E_dh = A x + b against the real propagator.
rng = np.random.default_rng(0)
x_test = rng.uniform(0, 1, n)
pupil.update(x_test)
E_ref = coro.forward(wvl, include_fpm=True).ravel()[dh_idx]
err = np.max(np.abs(A @ x_test + b - E_ref)) / np.max(np.abs(E_ref))
print(f"affine model max relative error: {err:.1e}")
assert err < 1e-12, "affine model does not match the propagator"

s_req = np.sqrt(contrast_req / 2)
quantum = np.abs(A) / np.sqrt(norm)  # per-pixel amplitude quantum, normalized

print(f"amplitude budget s: {s_req:.1e}")
print(f"pixel field quantum: max {quantum.max():.1e}, median {np.median(quantum):.1e}")
print(f"s / max quantum: {s_req / quantum.max():.1f}  (want >~ 3)")


from dygdug.optimizers import linprog

rn = 1 / np.sqrt(norm)
M, boff = A * rn, b * rn
A_ub = np.vstack([M.real, -M.real, M.imag, -M.imag])
b_ub = s_req + np.concatenate([-boff.real, boff.real, -boff.imag, boff.imag])
print(f"LP: {A_ub.shape[0]} rows x {A_ub.shape[1]} cols ({A_ub.nbytes/1e6:.0f} MB)")

t0 = perf_counter()
res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub)
print(f"solve: {perf_counter()-t0:.0f} s, status {res.status}, {res.message}")
assert res.status == 0, "infeasible: this contrast is unreachable for this geometry"

x_lp = res.x.copy()
tol = 1e-9
x_lp[x_lp <= tol] = 0.0
x_lp[x_lp >= 1 - tol] = 1.0
frac = np.flatnonzero((x_lp > 0) & (x_lp < 1))

print(f"throughput (mean transmission): {x_lp.mean():.4f}")
print(f"fractional pixels: {frac.size}/{n} ({frac.size/n:.2%}); LP bound was {4*m}")

pupil.update(x_lp)
I_lp = np.abs(coro.forward(wvl, include_fpm=True)) ** 2 / norm

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].set_title("LP-optimal apodizer (~99% binary by construction)")
im0 = ax[0].imshow(pupil.data, cmap="gray", vmin=0, vmax=1)
plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
ax[1].set_title("Coronagraphic intensity")
im1 = ax[1].imshow(I_lp, cmap="inferno", norm=LogNorm(vmin=1e-10, vmax=1e-3))
plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
plt.savefig("lp_aplc.png", dpi=150, bbox_inches="tight")

print(f"LP design dark-hole max contrast: {I_lp.ravel()[dh_idx].max():.2e} "
      f"(requirement {contrast_req:.0e})")

