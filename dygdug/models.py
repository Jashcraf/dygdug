import numpy as onp

from prysm import propagation
from prysm.mathops import np
from prysm.propagation import prepare_executor
from prysm.x.dm import DM
from prysm.propagation import prepare_executor

# TODO: Add full support for object-oriented image chain modeling
# Would be good to add some abstraction that treats an arbitrary list
# of elements. Maybe definitions like a sequential raytracer would be
# interesting? e.g.
#
# from dygdug.models import Surface
# 
# # Dataclass for a surface, which can be a free-space prop or focal plane prop
# # distance kwarg is propagation distance for free-space props, or focal length for focal plane props
# # _after_ the surface.
# s1 = Surface(type="plane", distance=1.0)

class Imager:
    def __init__(self, pupil, executor):
        self.pupil = pupil
        self.executor = executor

    def forward(self):
        """
        Propagate the field forward through the imager.

        Returns
        -------
        field_at_focal : ndarray
            The complex field at the focal plane.
        """

        self.field_at_focal = propagation.focus_dft(
            self.pupil.data,
            executor=self.executor,
        )

        return self.field_at_focal
    
    def reverse(self):
        """
        Propagate the field backward through the imager.

        Returns
        -------
        adjoint_field : ndarray
            The complex field at the entrance pupil
        """

        self.adjoint_at_entrance_pupil = propagation.focus_dft_adjoint(
            self.field_at_focal,
            executor=self.executor,
        )

        return self.adjoint_at_entrance_pupil


class PolychromaticExecutor:
    """Wavelength-keyed bank of prysm propagation executors.

    This class behaves like a dictionary where the executors are keyed by wavelength.
    It lets the user construct the polychromatic executor directly, rather than calling
    `prepare_executor` for each wavelength. It also allows the user to specify weights 
    for each wavelength.

    Parameters
    ----------
    wavelengths : array-like
        The wavelengths for which the executors are defined.
    executors : list of prysm.Executor
        The list of executors corresponding to the wavelengths.
    weights : array-like, optional
        The weights for each wavelength. If not provided, equal weights are assumed.

    """

    def __init__(self, wavelengths, executors, weights=None):
        # Wavelengths and weights are scalar bookkeeping, so they stay on the
        # host even when the propagation backend is cupy.
        self.wavelengths = onp.asarray(wavelengths, dtype=float)
        self.executors = list(executors)
        self.weights = (onp.ones(self.wavelengths.size) if weights is None
                        else onp.asarray(weights, dtype=float))

        if len(self.executors) != self.wavelengths.size:
            raise ValueError(
                f'need one executor per wavelength; got {len(self.executors)} '
                f'executors for {self.wavelengths.size} wavelengths'
            )

        # A single FPM array, Lyot stop, and dark hole are only shared across
        # wavelengths if every executor was built on the same grid.
        grids = {(e.pupil_dx, e.focal_dx) for e in self.executors}
        if len(grids) != 1:
            raise ValueError('all executors must share pupil_dx and focal_dx')

    @classmethod
    def from_wavelengths(cls, wavelengths, pupil_dx, pupil_samples, focal_dx,
                         focal_samples, efl, weights=None, focal_shift=(0, 0),
                         kind='mdft'):
        """optional constructor to build the executor in dygdug, see the 
        `prepare_executor` function for details on the parameters.
        """
        execs = [prepare_executor(pupil_dx=pupil_dx, pupil_samples=pupil_samples,
                                  focal_dx=focal_dx, focal_samples=focal_samples,
                                  wavelength=w, efl=efl, focal_shift=focal_shift,
                                  kind=kind)
                 for w in wavelengths]
        return cls(wavelengths, execs, weights)

    def __getitem__(self, wvl):
        """dunder method that lets the PolychromaticExecutor behave like a dictionary keyed by wavelength.
        """
        i = int(onp.argmin(onp.abs(self.wavelengths - wvl)))
        if not onp.isclose(self.wavelengths[i], wvl, rtol=1e-9):
            raise KeyError(f'no executor for {wvl}; bank holds {self.wavelengths}')
        return self.executors[i]


class Coronagraph:
    def __init__(self, pupil, fpm, lyot_stop, executor):
        self.pupil = pupil
        self.fpm = fpm
        self.lyot_stop = lyot_stop

        # Executor can be a single executor or a PolychromaticExecutor for polychromatic propagation
        self.executor = executor
        self.is_polychromatic = isinstance(executor, PolychromaticExecutor)

        # Support generally complex-valued coro masks
        if np.iscomplexobj(self.pupil.data):
            self.PUPIL_IS_COMPLEX = True
        else:
            self.PUPIL_IS_COMPLEX = False

        if np.iscomplexobj(self.fpm(1)):
            self.FPM_IS_COMPLEX = True
        else:
            self.FPM_IS_COMPLEX = False

        if np.iscomplexobj(self.lyot_stop.data):
            self.LYOT_STOP_IS_COMPLEX = True
        else:
            self.LYOT_STOP_IS_COMPLEX = False

    def _executor_for(self, wvl):
        """Resolve the executor for *wvl*.

        The wavelength is baked into an executor's frequency grid and norm at
        construction, so a polychromatic model holds one executor per
        wavelength.  ``forward`` and ``reverse`` must resolve the same one, or
        the adjoint does not pair with the forward propagation.
        """
        if self.is_polychromatic:
            return self.executor[wvl]
        return self.executor

    def forward(self, wvl, include_fpm=True):
        """
        Propagate the field forward through the coronagraph.

        Parameters
        ----------
        wvl : float
            Wavelength of the field to propagate.
        include_fpm : bool, optional
            Whether to include the focal plane mask in the propagation.

        Returns
        -------
        field_at_focal : ndarray
            The complex field at the focal plane.
        """

        executor = self._executor_for(wvl)

        if include_fpm:
            self.field_at_lyot = propagation.to_fpm_and_back(
                self.pupil.data,
                self.fpm(wvl),
                executor=executor,
            )
        else:
            self.field_at_lyot = propagation.to_fpm_and_back(
                self.pupil.data,
                np.ones_like(self.fpm(wvl)),
                executor=executor,
            )

        self.field_at_focal = propagation.focus_dft(
            self.field_at_lyot * self.lyot_stop.data,
            executor=executor,
        )

        return self.field_at_focal

    def reverse(self, Ebar, wvl, include_fpm=True):
        """
        Propagate the field backward through the coronagraph.

        Parameters
        ----------
        Ebar : ndarray
            Gradient of cost function at coronagraphic focal plane.
            Should have shape equal to the output of `forward`.
        wvl : float
            Wavelength of the field to propagate.
        include_fpm : bool, optional
            Whether to include the focal plane mask in the propagation.

        Returns
        -------
        adjoint_field : ndarray
            The complex field at the entrance pupil
        """

        executor = self._executor_for(wvl)

        self.adjoint_at_lyot = propagation.focus_dft_adjoint(
            Ebar,
            executor=executor,
        )

        # Support generally complex-valued lyot stop masks, but
        # doesn't conjugate the lyot stop mask fi it's real valued
        if self.LYOT_STOP_IS_COMPLEX:
            self.adjoint_at_lyot *= self.lyot_stop.data.conj()
        else:
            self.adjoint_at_lyot *= self.lyot_stop.data

        fpm = self.fpm(wvl) if include_fpm else np.ones_like(self.fpm(wvl))
        
        # Recall that this conjugates internally
        self.adjoint_at_entrance_pupil = propagation.to_fpm_and_back_adjoint(
            self.adjoint_at_lyot,
            fpm,
            executor=executor,
        )

        return self.adjoint_at_entrance_pupil
