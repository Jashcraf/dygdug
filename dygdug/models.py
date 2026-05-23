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
from prysm.x.dm import DM


class Coronagraph:
    def __init__(self, pupil, fpm, lyot_stop, executor):
        self.pupil = pupil
        self.fpm = fpm
        self.lyot_stop = lyot_stop
        self.executor = executor

    def forward(self, wvl, include_fpm=False):

        if include_fpm:
            field_at_lyot = propagation.to_fpm_and_back(
                self.pupil.data,
                self.fpm(wvl),
                executor=self.executor,
            )
        else:
            field_at_lyot = propagation.to_fpm_and_back(
                self.pupil.data,
                np.ones_like(self.fpm(wvl)),
                executor=self.executor,
            )

        field_at_focal = propagation.focus_dft(
            field_at_lyot * self.lyot_stop.data,
            executor=self.executor,
        )

        return field_at_focal
