from prysm import propagation
from prysm.mathops import np
from prysm.x.dm import DM


class Coronagraph:
    def __init__(self, pupil, fpm, lyot_stop, executor):
        self.pupil = pupil
        self.fpm = fpm
        self.lyot_stop = lyot_stop
        self.executor = executor

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

    def forward(self, wvl, include_fpm=False):
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

        if include_fpm:
            self.field_at_lyot = propagation.to_fpm_and_back(
                self.pupil.data,
                self.fpm(wvl),
                executor=self.executor,
            )
        else:
            self.field_at_lyot = propagation.to_fpm_and_back(
                self.pupil.data,
                np.ones_like(self.fpm(wvl)),
                executor=self.executor,
            )

        self.field_at_focal = propagation.focus_dft(
            self.field_at_lyot * self.lyot_stop.data,
            executor=self.executor,
        )

        return self.field_at_focal

    def reverse(self, Ebar, wvl, include_fpm=False):
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

        self.adjoint_at_lyot = propagation.focus_dft_backprop(
            Ebar,
            executor=self.executor,
        )

        # Support generally complex-valued lyot stop masks, but
        # doesn't conjugate the lyot stop mask fi it's real valued
        if self.LYOT_STOP_IS_COMPLEX:
            self.adjoint_at_lyot *= self.lyot_stop.data.conj()
        else:
            self.adjoint_at_lyot *= self.lyot_stop.data

        # TODO: add wavelength-dependent function cached to fpm adjoint
        # to avoid adjointing the same function multiple times
        self.adjoint_at_entrance_pupil = propagation.to_fpm_and_back_backprop(
            self.adjoint_at_lyot,
            self.fpm(wvl).conj() if self.FPM_IS_COMPLEX else self.fpm(wvl),
            executor=self.executor,
        )

        if self.PUPIL_IS_COMPLEX:
            self.adjoint_at_entrance_pupil *= self.pupil.data.conj()
        else:
            self.adjoint_at_entrance_pupil *= self.pupil.data

        return self.adjoint_at_entrance_pupil
