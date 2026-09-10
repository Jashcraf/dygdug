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
import sys

import numpy as tnp
from astropy.io import fits
from tqdm import tqdm

from prysm import coordinates
from prysm.mathops import np, set_backend_to_cupy
from dygdug.masks import Pupil, FPM, ImgSamplingSpec, annular_mask, knife_edge_mask
from dygdug.coropt import VariablePupil
from dygdug.models import Coronagraph, PolychromaticExecutor


def to_host(arr):
    """Return *arr* as a host numpy array, whatever backend it lives on."""
    return arr.get() if hasattr(arr, "get") else tnp.asarray(arr)

# Handle incoming arguments
if len(sys.argv) > 1:
    BANDWIDTH = float(sys.argv[1])
else:
    BANDWIDTH = 10

# LUVOIR-B
FLAT_TO_FLAT = 0.955
GAP_SIZE = 0.006
N_RINGS = 4
CIRCUMSCRIBED_DIAMETER = 7.994

# Define instrument parameters
Npup = 512
Nfoc = 192
fno = 13.66
efl = fno * CIRCUMSCRIBED_DIAMETER
px_per_lamD = 4  # the "oversampling"
EXCLUDE = [
    37, 41, 45,
    49, 53, 57
]
iwa, owa = 6, 20  # lam/D at band center
AZMAG = 60  # full angular extent of the dark-hole wedge, degrees
LS_FRAC = 0.95  # Lyot stop outer radius, as a fraction of the pupil radius
LS_OBSCURATION_RATIO = 0.0  # Lyot stop inner radius, same convention
contrast_req = 1e-10

# Core-throughput curve: photometric aperture radius and off-axis sampling,
# both in band-center lam/D.
CORE_RADIUS = 0.7
CT_STEP = 0.05

# compute NWVLS so that there's at least one wavelength per 4%, also add a
# catch to make the minimum number of wavelengths 5. This means that it will
# only start adding wavelengths after 20%. Using np.ceil here to always use
# more wavelengths than we strictly need.
NWVLS = int(np.ceil(BANDWIDTH / 4))
if NWVLS < 5:
    NWVLS = 5


# Configure save directory, makes directory if it doesn't exist
from datetime import datetime
from pathlib import Path
import os
now = datetime.now()
SAVE_DIR = Path.home() / "dygdug/Data"
filename = f"APLC_{BANDWIDTH}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
SAVE_DIR = SAVE_DIR / filename
os.makedirs(SAVE_DIR, exist_ok=True)


# Band parameters
wvl0 = 0.350
bandwidth = BANDWIDTH / 100
n_wvl = NWVLS
wvls = np.linspace(wvl0 * (1 - bandwidth / 2), wvl0 * (1 + bandwidth / 2), n_wvl)

lamD = wvl0 / CIRCUMSCRIBED_DIAMETER * efl  # band-center lam/D, physical units
focal_dx = lamD / px_per_lamD


# Experimental, keep every Nth dark-hole pixel as an LP constraint
DH_STRIDE = 30
MEMORY_BUDGET_GB = 20.0


# Variable (amplitude-apodized) entrance pupil
pupil = VariablePupil.hexagonal_segmented(
    Dpup=CIRCUMSCRIBED_DIAMETER,
    Npup=Npup,
    rings=4,
    segment_diameter=FLAT_TO_FLAT,
    segment_separation=GAP_SIZE,
    exclude=EXCLUDE,
    mode="amplitude"
)

set_backend_to_cupy()
pupil.data = np.asarray(pupil.data)

# The optimizer writes transmission into pupil.data in place, so snapshot the
# unapodized aperture now -- it is both the design file and the denominator of
# the throughput curve below.
aperture = np.copy(pupil.data)

# Dygdug's polychromatic executor
executor = PolychromaticExecutor.from_wavelengths(
    wavelengths=wvls,
    pupil_dx=CIRCUMSCRIBED_DIAMETER / Npup,
    pupil_samples=Npup,
    focal_dx=focal_dx,
    focal_samples=Nfoc,
    efl=efl,
    focal_shift=(0, 0),
    kind="mdft",
)

# Remainder is a typical lyot coronagraph
fpm = FPM.annular(
    N=Nfoc,
    lamD=lamD,
    px_per_lamD=px_per_lamD,
    inner_radius=iwa,
    outer_radius=owa,
)

lyot = Pupil.annular(
    Dpup=CIRCUMSCRIBED_DIAMETER,
    Npup=Npup,
    inner_radius=LS_OBSCURATION_RATIO * CIRCUMSCRIBED_DIAMETER / 2,
    outer_radius=LS_FRAC * CIRCUMSCRIBED_DIAMETER / 2,
)

coro = Coronagraph(pupil=pupil, fpm=fpm, lyot_stop=lyot, executor=executor)

# Dark hole on the same fixed physical grid as the executors and the FPM.
iss = ImgSamplingSpec.from_N_lamD_px_per_lamD(Nfoc, lamD, px_per_lamD)
dark_hole = annular_mask(iss, iwa=iwa, owa=owa, theta_min=-AZMAG / 2, theta_max=AZMAG / 2)

dh_idx_full = np.flatnonzero(np.asarray(dark_hole).astype(bool).ravel())
dh_idx = dh_idx_full[::DH_STRIDE] # indexes every DH_STRIDE pixels in the array
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
normalization = np.empty(n_wvl)
rng = np.random.default_rng(0)
Ebar = np.zeros((Nfoc, Nfoc), dtype=complex)

for iw, wvl in enumerate(wvls):
    tw = perf_counter()

    # Jacobian: one adjoint propagation per constrained dark-hole pixel. 
    A = np.empty((m, n), dtype=complex)
    for k, i in enumerate(dh_idx):
        Ebar.ravel()[i] = 1.0
        coro.reverse(Ebar, wvl, include_fpm=True)
        A[k] = np.conj(coro.adjoint_at_entrance_pupil.ravel()[pup_idx])
        Ebar.ravel()[i] = 0.0

    # Offset from the frozen (anti-aliased rim) pixels: the field with x = 0.
    pupil.update(np.zeros(n))
    b = coro.forward(wvl, include_fpm=True).ravel()[dh_idx].copy()

    # Contrast normalization: direct (no-FPM) peak of the open pupil. 
    pupil.update(np.ones(n))
    norm = float(np.max(np.abs(coro.forward(wvl, include_fpm=False)) ** 2))
    normalization[iw] = norm

    # Verify E_dh = A x + b affine transformation against the real propagator
    x_test = rng.uniform(0, 1, n)
    pupil.update(x_test)
    E_ref = coro.forward(wvl, include_fpm=True).ravel()[dh_idx]
    err = float(np.max(np.abs(A @ x_test + b - E_ref)) / np.max(np.abs(E_ref)))
    assert err < 1e-12, f"affine model does not match the propagator at {wvl:.4f} um"

    # Split the per-wavelength amplitude budget between the real and imaginary
    # parts: |Re| <= s and |Im| <= s imply |E|^2 <= 2 s^2 = contrast.
    s_req = np.sqrt(contrast_req / 2)
    rn = 1 / np.sqrt(norm)
    M, boff = A * rn, b * rn
    rows_A += [M.real, -M.real, M.imag, -M.imag]
    rows_b += [s_req - boff.real, s_req + boff.real,
               s_req - boff.imag, s_req + boff.imag]

    quantum = np.abs(M)
    print(f"  {wvl:.4f} um: {perf_counter()-tw:.1f} s, affine error {err:.1e}, "
          f"s / max quantum {s_req / float(quantum.max()):.1f}  (want >~ 3)")

A_ub = np.vstack(rows_A)
b_ub = np.concatenate(rows_b)
del rows_A, rows_b # clear up memory
print(f"Jacobians ({m * n_wvl} adjoint propagations): {perf_counter()-t0:.1f} s")
print(f"LP: {A_ub.shape[0]} rows x {A_ub.shape[1]} cols ({A_ub.nbytes/1e6:.0f} MB)")

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------
from dygdug.optimizers import linprog

t0 = perf_counter()
res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, verbose=True, max_iter=500)
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

# The apodizer has to stay on the device for the off-axis propagations below,
# so take host copies for plotting and for the design files instead of moving
# pupil.data itself.
apodizer = to_host(pupil.data)
aperture_host = to_host(aperture)
intensities = [to_host(I) for I in intensities]
I_band_host = to_host(I_band)
dark_hole_host = to_host(dark_hole).astype(bool)

fig, ax = plt.subplots(1, n_wvl + 1, figsize=(5 * (n_wvl + 1), 5))
ax[0].set_title("Polychromatic LP apodizer")
im0 = ax[0].imshow(apodizer, cmap="gray", vmin=0, vmax=1)
plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
for iw, wvl in enumerate(wvls):
    ax[iw + 1].set_title(f"{wvl:.4f} um")
    im = ax[iw + 1].imshow(intensities[iw], cmap="inferno",
                           norm=LogNorm(vmin=1e-11, vmax=1e-3))
    plt.colorbar(im, ax=ax[iw + 1], fraction=0.046, pad=0.04)
plt.savefig(SAVE_DIR / "lp_poly.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Azimuthally averaged broadband contrast
# ---------------------------------------------------------------------------
# Focal-plane coordinates in band-center lam/D, on the same grid the FPM and
# the dark hole were drawn on.
xi, eta = coordinates.make_xy_grid(Nfoc, dx=focal_dx)
xi_ld, eta_ld = xi / lamD, eta / lamD
r_ld_host = to_host(np.hypot(xi_ld, eta_ld))


def azimuthal_average(image, r, mask, bin_width):
    """Mean of *image* over the pixels of *mask*, binned by radius *r*.

    Averaging over the mask rather than over full annuli keeps the profile
    inside the dark-hole wedge; the bright field outside it would otherwise
    dominate every bin.  All arguments are host arrays.
    """
    rr, vv = r[mask], image[mask]
    edges = tnp.arange(rr.min(), rr.max() + bin_width, bin_width)
    counts, _ = tnp.histogram(rr, bins=edges)
    sums, _ = tnp.histogram(rr, bins=edges, weights=vv)
    ok = counts > 0
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers[ok], sums[ok] / counts[ok]


sep_ld, contrast_profile = azimuthal_average(
    I_band_host, r_ld_host, dark_hole_host, bin_width=1 / px_per_lamD
)

# ---------------------------------------------------------------------------
# Core throughput vs. angular separation
# ---------------------------------------------------------------------------
# An off-axis source is a linear phase ramp across the pupil.  The ramp is
# scaled by wvl0 / wvl so that every wavelength lands at the same band-center
# lam/D, which is what the curve is plotted against.
x_pup, _ = coordinates.make_xy_grid(Npup, dx=pupil.dx, grid=False)


def propagate_field(field, wvl, include_fpm=True):
    """Push *field* through the coronagraph without disturbing coro.pupil."""
    saved = coro.pupil
    coro.pupil = Pupil(data=field, dx=pupil.dx)
    try:
        return coro.forward(wvl, include_fpm=include_fpm)
    finally:
        coro.pupil = saved


# The MDFT executors are normalized so that a round trip conserves energy, so
# the summed pupil intensity of the *unapodized* aperture is the throughput
# denominator: these curves are relative to the full telescope collecting area.
aperture_energy = float(np.sum(np.abs(aperture) ** 2))
apod = pupil.data  # the LP solution, still on the device

tilt_lds = tnp.arange(0.0, owa + CT_STEP, CT_STEP)
core_throughput = tnp.empty(tilt_lds.size)
total_throughput = tnp.empty(tilt_lds.size)

t0 = perf_counter()
for i, ld in enumerate(tqdm(tilt_lds, desc="core throughput")):
    # Photometric aperture, recentered on the off-axis source at each step.
    core = np.hypot(xi_ld - ld, eta_ld) < CORE_RADIUS
    core_sum = total_sum = 0.0
    for iw, wvl in enumerate(wvls):
        tilt = np.exp(2j * np.pi * x_pup * ld * wvl0 / (CIRCUMSCRIBED_DIAMETER * wvl))
        I = np.abs(propagate_field(apod * tilt, wvl, include_fpm=True)) ** 2
        w = float(weights[iw])
        total_sum += w * float(np.sum(I))
        core_sum += w * float(np.sum(I[core]))
    total_throughput[i] = total_sum / aperture_energy
    core_throughput[i] = core_sum / aperture_energy

print(f"core throughput ({tilt_lds.size} separations x {n_wvl} wavelengths): "
      f"{perf_counter()-t0:.0f} s")
in_dh = (tilt_lds >= iwa) & (tilt_lds <= owa)
print(f"core throughput over {iwa}-{owa} lam/D: "
      f"max {core_throughput[in_dh].max():.4f}, mean {core_throughput[in_dh].mean():.4f}")

# ---------------------------------------------------------------------------
# Contrast curve + core throughput figure
# ---------------------------------------------------------------------------
fig, (axc, axt) = plt.subplots(1, 2, figsize=(13, 5))

axc.semilogy(sep_ld, contrast_profile, color="C0",
             label=f"{BANDWIDTH:.0f}% band average")
axc.axhline(contrast_req, color="k", linestyle=":", label=f"{contrast_req:.0e} requirement")
axc.axvline(iwa, color="0.6", linestyle="--", linewidth=1)
axc.axvline(owa, color="0.6", linestyle="--", linewidth=1)
axc.set_title("Azimuthally averaged contrast")
axc.set_xlabel(r"Angular separation [$\lambda_0 / D$]")
axc.set_ylabel("Normalized intensity")
axc.set_xlim(0, owa * 1.05)
axc.legend(loc="upper right")

axt.plot(tilt_lds, core_throughput, color="C0",
         label=rf"$r = {CORE_RADIUS}\,\lambda_0 / D$")
axt.plot(tilt_lds, total_throughput, color="C0", linestyle="--", label=r"$r = \infty$")
axt.axvline(iwa, color="0.6", linestyle="--", linewidth=1)
axt.axvline(owa, color="0.6", linestyle="--", linewidth=1)
axt.set_title("Core throughput")
axt.set_xlabel(r"Angular separation [$\lambda_0 / D$]")
axt.set_ylabel("Throughput")
axt.set_xlim(0, owa * 1.05)
axt.set_ylim(0, None)
axt.legend(loc="lower right")

fig.tight_layout()
stem = (f"Polychromatic_{Npup}Npup_{Nfoc}Nimg_{iwa}IWA_{owa}OWA_{AZMAG}AZ_"
        f"{LS_OBSCURATION_RATIO}LSinner_{LS_FRAC}LSouter_{BANDWIDTH}bw_{n_wvl}wls")
fig.savefig(SAVE_DIR / f"{stem}.pdf", bbox_inches="tight")
fig.savefig(SAVE_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Design files
# ---------------------------------------------------------------------------
def stamp_design(header):
    """Common design metadata, so every file stands on its own."""
    header["NPIX"] = pupil.data.shape[0]
    header["DPUP"] = (CIRCUMSCRIBED_DIAMETER, "circumscribed pupil diameter")
    header["F/#"] = fno
    header["EFL"] = (efl, "same units as DPUP")
    header["CEN WVL"] = (wvl0, "[microns]")
    header["IWA [lam/D]"] = (iwa, "[lam/D]")
    header["OWA [lam/D]"] = (owa, "[lam/D]")
    header["AZMIN"] = (-AZMAG / 2, "[deg]")
    header["AZMAX"] = (AZMAG / 2, "[deg]")
    header["BW"] = (BANDWIDTH, "[%]")
    header["NWVLS"] = n_wvl
    header["OVERSAMPLE"] = (px_per_lamD, "[pix/lam/D]")
    header["LS FRAC RADIUS"] = LS_FRAC
    header["LS OBST RADIUS"] = LS_OBSCURATION_RATIO
    header["CONTRAST REQ"] = contrast_req
    return header


# Aperture, apodizer, FPM, Lyot stop
hdu_aper = fits.PrimaryHDU(aperture_host.astype(tnp.float64))
hdu_aper.header["NPIX"] = aperture_host.shape[0]
hdu_aper.header["DPUP"] = (CIRCUMSCRIBED_DIAMETER, "circumscribed pupil diameter")
hdu_aper.writeto(SAVE_DIR / "aperture.fits", overwrite=True)

hdu_apod = fits.PrimaryHDU(apodizer.astype(tnp.float64))
stamp_design(hdu_apod.header)
hdu_apod.writeto(
    SAVE_DIR / f"apodizer_{wvl0}cenwvl_{n_wvl}wvls_{BANDWIDTH}%.fits", overwrite=True
)

fpm_host = to_host(fpm(wvl0))
hdu_fpm = fits.PrimaryHDU(fpm_host.astype(tnp.float64))
hdu_fpm.header["FPM IWA"] = (iwa, "[lam/D]")
hdu_fpm.header["FPM OWA"] = (owa, "[lam/D]")
hdu_fpm.header["DX"] = (1 / px_per_lamD, "[lam/D/pix]")
hdu_fpm.header["NPIX"] = fpm_host.shape[0]
hdu_fpm.writeto(
    SAVE_DIR / f"fpm_{iwa}IWA_{owa}OWA_{px_per_lamD}OS.fits", overwrite=True
)

lyot_host = to_host(lyot.data)
hdu_lyot = fits.PrimaryHDU(lyot_host.astype(tnp.float64))
hdu_lyot.header["NPIX"] = lyot_host.shape[0]
hdu_lyot.header["LS FRAC RADIUS"] = LS_FRAC
hdu_lyot.header["LS OBST RADIUS"] = LS_OBSCURATION_RATIO
hdu_lyot.writeto(SAVE_DIR / "lyot_stop.fits", overwrite=True)

# Raw data behind the left panel: the profile, the band-averaged normalized
# intensity it was reduced from, the per-wavelength stack, and the dark hole.
primary = fits.PrimaryHDU(I_band_host.astype(tnp.float64))
stamp_design(primary.header)
primary.header["BUNIT"] = "normalized intensity"
primary.header["COMMENT"] = "band-averaged normalized intensity of the on-axis star"
profile_hdu = fits.BinTableHDU.from_columns(
    [
        fits.Column(name="separation", format="D", unit="lam/D", array=sep_ld),
        fits.Column(name="contrast", format="D", array=contrast_profile),
    ],
    name="PROFILE",
)
profile_hdu.header["COMMENT"] = "azimuthal average over the dark-hole wedge"
fits.HDUList([
    primary,
    profile_hdu,
    fits.ImageHDU(tnp.stack(intensities).astype(tnp.float64), name="PERWVL"),
    fits.ImageHDU(to_host(wvls).astype(tnp.float64), name="WAVELENGTHS"),
    fits.ImageHDU(dark_hole_host.astype(tnp.uint8), name="DARKHOLE"),
]).writeto(SAVE_DIR / "contrast_curve.fits", overwrite=True)

# Raw data behind the right panel.
ct_primary = fits.PrimaryHDU()
stamp_design(ct_primary.header)
ct_primary.header["CORERAD"] = (CORE_RADIUS, "[lam/D] photometric aperture radius")
ct_primary.header["CTSTEP"] = (CT_STEP, "[lam/D] off-axis sampling")
ct_primary.header["COMMENT"] = "throughput is relative to the unapodized aperture energy"
ct_table = fits.BinTableHDU.from_columns(
    [
        fits.Column(name="separation", format="D", unit="lam/D", array=tilt_lds),
        fits.Column(name="core_throughput", format="D", array=core_throughput),
        fits.Column(name="total_throughput", format="D", array=total_throughput),
    ],
    name="THROUGHPUT",
)
fits.HDUList([ct_primary, ct_table]).writeto(
    SAVE_DIR / "core_throughput.fits", overwrite=True
)

print(f"design files written to {SAVE_DIR}")
