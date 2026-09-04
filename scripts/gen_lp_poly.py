"""Polychromatic LP apodizer design.

The broadband sibling of ``gen_lp_aplc.py``.  An amplitude apodizer enters the
propagation *linearly*, so at each wavelength the dark-hole field is exactly
affine in the parameter vector,

    E_dh(lambda) = A(lambda) x + b(lambda),

where ``A`` is the adjoint-derived Jacobian and ``b`` is the contribution of
the frozen (anti-aliased rim) pixels.  Both are wavelength-dependent -- the
executor bakes lambda into its frequency grid and norm -- but ``x`` is shared
across the band, and so is the throughput objective.  A broadband design is
therefore the *same* LP with the per-wavelength constraint blocks stacked
vertically:

    minimize    -sum(x)
    subject to  |Re E_dh(lambda_i)| <= s_i,  |Im E_dh(lambda_i)| <= s_i   for all i
                0 <= x <= 1

which is 4 * n_darkhole * n_wavelengths rows.  ``dygdug.optimizers.linprog``
needs no changes; only the assembly below does.

Cost scaling
------------
Row count grows linearly in the number of wavelengths, but the interior-point
work does not.  Each iteration forms ``A D^-1 A^T`` (a ``m_rows x m_rows x n``
GEMM) and factors an ``m_rows x m_rows`` matrix, so wall time grows roughly
quadratically in ``n_wvl`` for the GEMM and cubically for the Cholesky.  Three
to five wavelengths is usually enough to control an APLC across a 10-20%
band; beyond that, subsampling the dark hole (``DH_STRIDE`` below) trades
contrast for tractability.

Be careful with that trade: an unconstrained pixel is *not* free to assume it
inherits its neighbors' contrast.  At the reduced scale used to check this
script, ``DH_STRIDE = 8`` produced a design that satisfied every constrained
pixel yet ran ~30x over requirement across the full dark hole.  The
verification propagation at the end deliberately reports contrast over every
dark-hole pixel, not just the constrained ones, so the penalty is visible.
Keep the stride at 1 for a design you intend to trust, and treat larger values
as a way to size the problem, not to solve it.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from time import perf_counter

import numpy as onp
from prysm.mathops import np, set_backend_to_cupy

from dygdug.masks import Pupil, FPM, ImgSamplingSpec, annular_mask, knife_edge_mask
from dygdug.coropt import VariablePupil
from dygdug.models import Coronagraph, PolychromaticExecutor

# Define instrument parameters
circumscribed_diameter = 10.95e3
Npup = 512 
Nfoc = 192
fno = 13.66
efl = fno * circumscribed_diameter
px_per_lamD = 3  # the "oversampling"

# Band parameters.  The focal grid is fixed in physical units across the band
# (one detector, one fabricated mask), so lam/D -- and therefore the working
# angles measured in lam/D -- vary with wavelength.  All geometry below is
# specified at the band center.
wvl0 = 1
bandwidth = 0.10
n_wvl = 3
wvls = onp.linspace(wvl0 * (1 - bandwidth / 2), wvl0 * (1 + bandwidth / 2), n_wvl)

lamD = wvl0 / circumscribed_diameter * efl  # band-center lam/D, physical units
focal_dx = lamD / px_per_lamD

iwa, owa = 6, 10  # lam/D at band center
contrast_req = 1e-10
DH_STRIDE = 1  # keep every Nth dark-hole pixel as an LP constraint

# Variable (amplitude-apodized) entrance pupil
pupil = VariablePupil.circle(Dpup=circumscribed_diameter, Npup=Npup, mode="amplitude")

# One executor per wavelength, all sharing the pupil and focal grids so that a
# single FPM array, Lyot stop, and dark-hole index set serve the whole band.
executor = PolychromaticExecutor.from_wavelengths(
    wavelengths=wvls,
    pupil_dx=circumscribed_diameter / Npup,
    pupil_samples=Npup,
    focal_dx=focal_dx,
    focal_samples=Nfoc,
    efl=efl,
    focal_shift=(0, 0),
    kind="mdft",
)

# The mask is a fixed physical occulter; it is sized in band-center lam/D but
# does not change with wavelength.
fpm = FPM.annular(
    N=Nfoc,
    lamD=lamD,
    px_per_lamD=px_per_lamD,
    inner_radius=iwa,
    outer_radius=owa,
)

lyot = Pupil.annular(
    Dpup=circumscribed_diameter,
    Npup=Npup,
    inner_radius=0.1 * circumscribed_diameter / 2,
    outer_radius=0.8 * circumscribed_diameter / 2,
)

coro = Coronagraph(pupil=pupil, fpm=fpm, lyot_stop=lyot, executor=executor)

# Dark hole on the same fixed physical grid as the executors and the FPM.
iss = ImgSamplingSpec.from_N_lamD_px_per_lamD(Nfoc, lamD, px_per_lamD)
dark_hole = annular_mask(iss, iwa=iwa, owa=owa, theta_min=-90, theta_max=90)
dark_hole *= knife_edge_mask(iss, iwa=iwa)

dh_idx_full = np.flatnonzero(np.asarray(dark_hole).astype(bool).ravel())
dh_idx = dh_idx_full[::DH_STRIDE]
pup_idx = pupil._mask_idx
n = pupil.n_params  # optimizable pupil pixels
m = dh_idx.size     # constrained dark-hole pixels
rows = 4 * m * n_wvl

print(f"band: {wvls[0]:.4f} - {wvls[-1]:.4f} um ({bandwidth:.0%}), {n_wvl} samples")
print(f"working angles at band center: {iwa}-{owa} lam/D "
      f"({iwa * wvl0 / wvls[-1]:.2f}-{owa * wvl0 / wvls[-1]:.2f} lam/D at the red end, "
      f"{iwa * wvl0 / wvls[0]:.2f}-{owa * wvl0 / wvls[0]:.2f} at the blue end)")
print(f"n = {n} pupil pixels, m = {m} of {dh_idx_full.size} dark-hole pixels "
      f"(stride {DH_STRIDE}) -> {rows} constraint rows")
print(f"n / rows = {n / rows:.1f}  (want >> 1)")
print(f"A_ub {rows * n * 8 / 1e9:.1f} GB, normal equations {rows * rows * 8 / 1e9:.1f} GB")

# Fail fast with something actionable rather than dying in np.vstack an hour
# from now.  A_ub alone is rows x n; at the full design scale (Npup = 1024, a
# 3-8 lam/D dark hole at px_per_lamD = 8) that is hundreds of GB before any
# wavelength stacking, because n is ~800k pupil pixels.
MEMORY_BUDGET_GB = 16.0
est_gb = (rows * n + rows * rows) * 8 / 1e9
if est_gb > MEMORY_BUDGET_GB:
    raise SystemExit(
        f"estimated {est_gb:.0f} GB > MEMORY_BUDGET_GB = {MEMORY_BUDGET_GB:.0f} GB.\n"
        "The column count n (pupil pixels) drives A_ub and the row count drives\n"
        "the normal equations quadratically.  In order of preference: reduce Npup,\n"
        "reduce px_per_lamD (dark-hole pixels scale as its square), narrow the\n"
        "working angles, drop to fewer wavelengths, or -- last, and read the\n"
        "DH_STRIDE note in the module docstring first -- subsample the dark hole."
    )

# ---------------------------------------------------------------------------
# Per-wavelength constraint blocks
# ---------------------------------------------------------------------------
t0 = perf_counter()
rows_A, rows_b = [], []
normalization = onp.empty(n_wvl)
rng = np.random.default_rng(0)
Ebar = np.zeros((Nfoc, Nfoc), dtype=complex)

for iw, wvl in enumerate(wvls):
    tw = perf_counter()

    # Jacobian: one adjoint propagation per constrained dark-hole pixel.  This
    # is where the per-wavelength executor matters -- if reverse resolved the
    # wrong one, every block would be an identical copy and the "broadband"
    # design would only be valid at a single wavelength.
    A = np.empty((m, n), dtype=complex)
    for k, i in enumerate(dh_idx):
        Ebar.ravel()[i] = 1.0
        coro.reverse(Ebar, wvl, include_fpm=True)
        A[k] = np.conj(coro.adjoint_at_entrance_pupil.ravel()[pup_idx])
        Ebar.ravel()[i] = 0.0

    # Offset from the frozen (anti-aliased rim) pixels: the field with x = 0.
    pupil.update(np.zeros(n))
    b = coro.forward(wvl, include_fpm=True).ravel()[dh_idx].copy()

    # Contrast normalization: direct (no-FPM) peak of the open pupil.  This is
    # wavelength-dependent: the executor norm carries a 1/lambda, so the direct
    # peak intensity scales as 1/lambda^2.
    pupil.update(np.ones(n))
    norm = float(np.max(np.abs(coro.forward(wvl, include_fpm=False)) ** 2))
    normalization[iw] = norm

    # Verify E_dh = A x + b against the real propagator.  Kept inside the loop:
    # it is a decisive check that this wavelength's executor is wired through
    # both forward and reverse.
    x_test = rng.uniform(0, 1, n)
    pupil.update(x_test)
    E_ref = coro.forward(wvl, include_fpm=True).ravel()[dh_idx]
    err = float(np.max(np.abs(A @ x_test + b - E_ref)) / np.max(np.abs(E_ref)))
    assert err < 1e-12, f"affine model does not match the propagator at {wvl:.4f} um"

    # Split the per-wavelength amplitude budget between the real and imaginary
    # parts: |Re| <= s and |Im| <= s imply |E|^2 <= 2 s^2 = contrast.
    s_req = onp.sqrt(contrast_req / 2)
    rn = 1 / onp.sqrt(norm)
    M, boff = A * rn, b * rn
    rows_A += [M.real, -M.real, M.imag, -M.imag]
    rows_b += [s_req - boff.real, s_req + boff.real,
               s_req - boff.imag, s_req + boff.imag]

    quantum = np.abs(M)
    print(f"  {wvl:.4f} um: {perf_counter()-tw:.1f} s, affine error {err:.1e}, "
          f"s / max quantum {s_req / float(quantum.max()):.1f}  (want >~ 3)")

A_ub = np.vstack(rows_A)
b_ub = np.concatenate(rows_b)
del rows_A, rows_b
print(f"Jacobians ({m * n_wvl} adjoint propagations): {perf_counter()-t0:.1f} s")
print(f"LP: {A_ub.shape[0]} rows x {A_ub.shape[1]} cols ({A_ub.nbytes/1e6:.0f} MB)")

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
from dygdug.optimizers import linprog

t0 = perf_counter()
res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub)
print(f"solve: {perf_counter()-t0:.0f} s, status {res.status}, {res.message}")
assert res.status == 0, "infeasible: this contrast is unreachable over this band"

x_lp = res.x.copy()
tol = 1e-9
x_lp[x_lp <= tol] = 0.0
x_lp[x_lp >= 1 - tol] = 1.0
frac = np.flatnonzero((x_lp > 0) & (x_lp < 1))

print(f"throughput (mean transmission): {x_lp.mean():.4f}")
print(f"fractional pixels: {frac.size}/{n} ({frac.size/n:.2%}); LP bound was {rows}")

# ---------------------------------------------------------------------------
# Verify against the propagator at every wavelength, over the *full* dark hole
# ---------------------------------------------------------------------------
pupil.update(x_lp)
weights = executor.weights / executor.weights.sum()
intensities = []
I_band = np.zeros((Nfoc, Nfoc))
for iw, wvl in enumerate(wvls):
    I = np.abs(coro.forward(wvl, include_fpm=True)) ** 2 / normalization[iw]
    intensities.append(I)
    I_band = I_band + float(weights[iw]) * I
    print(f"  {wvl:.4f} um dark-hole max contrast: {I.ravel()[dh_idx_full].max():.2e} "
          f"(mean {I.ravel()[dh_idx_full].mean():.2e})")

print(f"band-averaged dark-hole max contrast: {I_band.ravel()[dh_idx_full].max():.2e} "
      f"(requirement {contrast_req:.0e})")

fig, ax = plt.subplots(1, n_wvl + 1, figsize=(5 * (n_wvl + 1), 5))
ax[0].set_title("Polychromatic LP apodizer")
im0 = ax[0].imshow(pupil.data, cmap="gray", vmin=0, vmax=1)
plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
for iw, wvl in enumerate(wvls):
    ax[iw + 1].set_title(f"{wvl:.4f} um")
    im = ax[iw + 1].imshow(intensities[iw], cmap="inferno",
                           norm=LogNorm(vmin=1e-11, vmax=1e-3))
    plt.colorbar(im, ax=ax[iw + 1], fraction=0.046, pad=0.04)
plt.savefig("lp_poly.png", dpi=150, bbox_inches="tight")
