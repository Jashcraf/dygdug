import inspect
from collections import namedtuple

from prysm import (
    coordinates,
    geometry,
    polynomials,
    propagation,
)
from prysm._richdata import RichData
from prysm.mathops import np
from prysm.segmented import CompositeHexagonalAperture
from prysm.x.dm import DM

WF = propagation.Wavefront

DEFAULT_IFN_FN = "influence_dm5v2.fits"

ImgSamplingSpec = namedtuple("ImageSamplingSpec", ["N", "lamD", "px_per_lamD"])


class ImgSamplingSpec:
    """Specification for image plane sampling."""

    def __init__(self, N, dx, lamD):
        self.N = N
        self.dx = dx
        self.lamD = lamD

    @classmethod
    def from_N_lamD_px_per_lamD(cls, N, lamD, px_per_lamD):
        dx = lamD / px_per_lamD
        return cls(N=N, dx=dx, lamD=lamD)


class WavelengthDependentFunctionCache:
    def __init__(self, f):
        self.f = f
        self.storage = {}

    def clear(self):
        self.storage = {}

    def __call__(self, wvl):
        data = self.storage.get(wvl, None)
        if data is None:
            data = self.storage[wvl] = self.f(wvl)

        return data

    def nbytes(self):
        total = 0
        for v in self.storage.values():
            total += v.nbytes

        return total


def _geometry_factory(func, N, dx, **kwargs):
    sig = inspect.signature(func)
    params = set(sig.parameters.keys())
    xyrt = {"x", "y", "r", "t"}
    # intersection = filter function parameters to just x,y,r,t so we know what to pass
    need_args = xyrt.intersection(params)

    x, y = coordinates.make_xy_grid(N, dx=dx)
    xyrt = dict(x=x, y=y)
    if "r" in need_args or "t" in need_args:
        r, t = coordinates.cart_to_polar(x, y)
        xyrt.update(dict(r=r, t=t))
    kwarg = {k: xyrt[k] for k in need_args}
    kwarg.update(kwargs)
    return func(**kwarg)


class Pupil:
    """Representation of a pupil."""

    def __init__(self, data, dx):
        """Create a new Pupil.

        Parameters
        ----------
        data : numpy.ndarray
            2D array, real or complex, containing the pupil
        dx : float
            intersample spacing, mm

        """
        self.data = data
        self.dx = dx
        # TODO: if we store this as an int instead of shape, all the prysm
        # routines will just go back to shape, but if user touches N, maybe
        # they want an int?  But int forces square, and prysm don't care...
        self.N = data.shape

    @classmethod
    def circle(cls, Dpup, Npup):
        dx = Dpup / Npup
        data = _geometry_factory(geometry.truecircle, N=Npup, dx=dx, radius=Dpup / 2)
        return cls(data=data, dx=dx)

    @classmethod
    def annular(cls, Dpup, Npup, inner_radius, outer_radius):
        """Create an annular pupil aperture.

        AI Disclosure: written by Claude Haiku 4.5

        Parameters
        ----------
        Dpup : float
            Diameter of the pupil in physical distance units
        Npup : int
            Number of pixels per side of the square pupil
        inner_radius : float
            Inner radius of annulus in physical distance units
        outer_radius : float
            Outer radius of annulus in physical distance units

        Returns
        -------
        Pupil
            Pupil with annular transmissive region
        """
        dx = Dpup / Npup
        x, y = coordinates.make_xy_grid(Npup, dx=dx)
        r = np.hypot(x, y)

        inner_circle = geometry.truecircle(inner_radius, r)
        outer_circle = geometry.truecircle(outer_radius, r)
        data = outer_circle * (1 - inner_circle)

        return cls(data=data, dx=dx)

    @classmethod
    def hexagonal_segmented(
        cls,
        Dpup,
        Npup,
        rings,
        segment_diameter,
        segment_separation,
        segment_angle=90,
        exclude=(),
    ):

        dx = Dpup / Npup
        x, y = coordinates.make_xy_grid(Npup, dx=dx)
        data = CompositeHexagonalAperture(
            x,
            y,
            rings=rings,
            segment_diameter=segment_diameter,
            segment_separation=segment_separation,
            segment_angle=segment_angle,
            exclude=exclude,
        ).amp
        return cls(data=data, dx=dx)


class FPM:
    """Representation of a focal plane mask."""

    def __init__(self, func_lam, dx):
        self.f = func_lam
        self.dx = dx

    def __call__(self, wvl):
        return self.f(wvl)

    @classmethod
    def lyot(cls, N, lamD, px_per_lamD, radius):
        # lamD has some physical unit (um, say)
        # px per lamD describes our resolution
        dx = lamD / px_per_lamD
        x, y = coordinates.make_xy_grid(N, dx=dx)
        r = np.hypot(x, y)

        def fpmfunc(wvl):
            return 1 - geometry.circle(radius * lamD, r)

        # TODO: need to del x, y here?  They should be collected because
        # they went out of scope, but maybe keeping r around through a closure
        # keeps them too?

        # humor
        caller_number_1 = WavelengthDependentFunctionCache(fpmfunc)
        return cls(caller_number_1, dx)

    @classmethod
    def annular(cls, N, lamD, px_per_lamD, inner_radius, outer_radius, theta_min=-90, theta_max=90):
        """Create an annular focal plane mask.

        AI Disclosure: written by Claude Haiku 4.5

        Parameters
        ----------
        N : int
            Number of pixels per side of the square mask
        lamD : float
            Lambda/D scale (physical units)
        px_per_lamD : float
            Pixels per lambda/D (resolution)
        inner_radius : float
            Inner radius of annulus in lambda/D
        outer_radius : float
            Outer radius of annulus in lambda/D
        theta_min : float, optional
            Minimum angle of the wedge in degrees (default is -90)
        theta_max : float, optional
            Maximum angle of the wedge in degrees (default is 90)

        Returns
        -------
        FPM
            Focal plane mask with annular transmissive region

        """
        dx = lamD / px_per_lamD
        x, y = coordinates.make_xy_grid(N, dx=dx)
        r = np.hypot(x, y)

        # use ImgSamplingSpec to store the parameters for the annular mask
        iss = ImgSamplingSpec(N, lamD, px_per_lamD)

        def fpmfunc(wvl):
            mask = annular_mask(iss, inner_radius, outer_radius, theta_min=theta_min, theta_max=theta_max)
            return mask.astype(float)

        caller_number_2 = WavelengthDependentFunctionCache(fpmfunc)
        return cls(caller_number_2, dx)

    @classmethod
    def unity(cls, N, lamD, px_per_lamD):
        dx = lamD / px_per_lamD
        ones = np.ones((N, N))

        def fpmfunc(wvl):
            return ones

        return cls(fpmfunc, dx)


def _make_symmetric_xy(N, dx, shift=False):
    """
    Controling the precise centering of prysm's xy grids
    """
    x, y = coordinates.make_xy_grid(N, dx=dx)

    if shift:
        shift = dx / 2
        x += shift
        y += shift

    return x, y


class ImgSamplingSpec:
    """Specification for image plane sampling.
    Taken from the equivalent class at github.com/brandondube/dygdug
    """
    def __init__(self, N, dx, lamD):
        self.N = N
        self.dx = dx
        self.lamD = lamD

    @classmethod
    def from_N_lamD_px_per_lamD(cls, N, lamD, px_per_lamD):
        dx = lamD/px_per_lamD
        return cls(N=N, dx=dx, lamD=lamD)


def annular_mask(iss, iwa, owa, theta_min=None, theta_max=None, shift=False):
    x, y = _make_symmetric_xy(iss.N, dx=iss.dx, shift=shift)
    r, t = coordinates.cart_to_polar(x, y)
    iwa = iwa * iss.lamD
    owa = owa * iss.lamD

    mask = (r > iwa) & (r <= owa)

    t_norm = t % (2 * np.pi)

    if theta_min is not None and theta_max is not None:

        # Normalize angles to [0, 2π]
        t_norm = t % (2 * np.pi)
        
        theta_min_rad = np.radians(theta_min) % (2 * np.pi)
        theta_max_rad = np.radians(theta_max) % (2 * np.pi)
        
        # Create mask for primary wedge
        if theta_max_rad < theta_min_rad:
            # Range wraps around
            angular_mask = (t_norm >= theta_min_rad) | (t_norm <= theta_max_rad)
        else:
            # Normal case
            angular_mask = (t_norm >= theta_min_rad) & (t_norm <= theta_max_rad)
        
        # Create mask for opposite wedge (180° rotated)
        t_opposite = (t_norm + np.pi) % (2 * np.pi)
        if theta_max_rad < theta_min_rad:
            angular_mask_opposite = (t_opposite >= theta_min_rad) | (t_opposite <= theta_max_rad)
        else:
            angular_mask_opposite = (t_opposite >= theta_min_rad) & (t_opposite <= theta_max_rad)
        
        # Combine both wedges
        angular_mask = angular_mask | angular_mask_opposite
        
        # Combine radial and angular masks
        mask = mask & angular_mask
    
    return mask


def knife_edge_mask(iss, iwa):
    x, y = _make_symmetric_xy(iss.N, dx=iss.dx)
    iwa = iwa * iss.lamD
    mask = x > iwa

    return mask