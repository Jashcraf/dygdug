"""Helpers for interoperating with prysm's swappable array backend.

``prysm.mathops.np`` is a ``BackendShim`` whose underlying module is swapped in
place by ``prysm.mathops.set_backend_to_cupy`` / ``set_backend_to_defaults``.
Any module that did ``from prysm.mathops import np`` therefore transparently
follows the switch.  The remaining hazards for GPU execution are:

* a *host* numpy array leaking into a pipeline whose other operands live on the
  GPU (cupy raises rather than implicitly transferring), and
* a precomputed operator (e.g. a prysm ``MDFT`` built by ``prepare_executor``)
  whose basis matrices were created on the previous backend.

These helpers move arrays onto the currently active backend (uploading host
arrays to the GPU when cupy is active), pull arrays back to host numpy for
plotting / scipy / any host-only consumer, and re-home a prysm transform
operator's matrices onto the active backend.

cupy is imported lazily (only ever via prysm's own switch), so importing this
module is safe on machines without cupy.
"""
from prysm.mathops import np as _np

import numpy as _onp


def active_backend():
    """Return the module currently backing ``prysm.mathops.np`` (numpy or cupy)."""
    return _np._srcmodule


def is_cupy_active():
    """True when prysm's array backend is currently cupy."""
    return active_backend().__name__ == "cupy"


def _module_name(array):
    """Top-level module name of ``array``'s type (e.g. 'numpy' or 'cupy')."""
    return type(array).__module__.split(".")[0]


def asbackend(array):
    """Return *array* as an array of the currently active prysm backend.

    Host numpy arrays are uploaded to the GPU when cupy is active; cupy arrays
    are downloaded to host when the backend is on numpy.  An array already on
    the active backend is returned unchanged (no copy), so this is cheap to call
    on the hot path.  ``None`` passes through.
    """
    if array is None:
        return None
    backend = active_backend()
    name = backend.__name__
    if _module_name(array) == name:
        return array
    if name == "cupy":
        return backend.asarray(array)  # host -> device
    return asnumpy(array)  # device -> host


def asnumpy(array):
    """Return a host numpy copy of *array* (accepts numpy or cupy input).

    Useful at host-only boundaries: matplotlib, FITS I/O, scipy routines, etc.
    ``None`` passes through.
    """
    if array is None:
        return None
    get = getattr(array, "get", None)
    if callable(get):  # cupy ndarray exposes .get()
        return get()
    return _onp.asarray(array)


def sync_executor(executor):
    """Move a prysm transform operator's basis matrices onto the active backend.

    ``prepare_executor`` builds an ``MDFT``/``CZT`` whose ``Ex``/``Ey`` matrices
    live on whatever backend was active at construction time.  If the backend is
    switched afterward, the matrices must be re-homed or they will be multiplied
    against arrays on the other device.  Re-homing is a no-op when they already
    match the active backend.  Returns *executor* for chaining.
    """
    for attr in ("Ex", "Ey"):
        mat = getattr(executor, attr, None)
        if mat is not None:
            setattr(executor, attr, asbackend(mat))
    return executor


def _sync_attrs(obj, attrs):
    """In-place coerce each present array attribute of *obj* onto the backend."""
    if obj is None:
        return
    for attr in attrs:
        val = getattr(obj, attr, None)
        if val is not None and hasattr(val, "dtype"):
            setattr(obj, attr, asbackend(val))


def sync_coronagraph(coro, wavelengths=()):
    """Move a :class:`~dygdug.models.Coronagraph`'s persistent arrays on-backend.

    This is meant to be called *once*, at optimizer construction time, rather
    than on the ``forward``/``reverse`` hot path.  It re-homes every array the
    model touches repeatedly so the propagation stays entirely on one device:

    * the executor's transform matrices (:func:`sync_executor`);
    * the pupil's ``data`` plus its ``mask`` / ``_mask_idx`` (so the scatter in
      :meth:`VariablePupil.update` and the adjoint gather stay on-device);
    * the Lyot stop's ``data`` and, for :class:`VariableLyotStop`, its ``_r``
      radial grid (recomputed on every ``update``);
    * the focal-plane mask: a :class:`VariableFPM`'s ``data``/``mask``, and the
      cached array for each wavelength of a plain wavelength-keyed :class:`FPM`.

    Every conversion is a no-op when the array already lives on the active
    backend, so this is safe (and cheap) to call unconditionally.  Returns
    *coro* for chaining.
    """
    sync_executor(coro.executor)

    _sync_attrs(getattr(coro, "pupil", None), ("data", "mask", "_mask_idx"))
    _sync_attrs(getattr(coro, "lyot_stop", None), ("data", "_r"))

    fpm = getattr(coro, "fpm", None)
    if fpm is not None:
        # VariableFPM stores a mutable spatial array directly.
        _sync_attrs(fpm, ("data", "mask"))
        # Plain FPM defers to a wavelength-keyed cache; coerce each entry the
        # optimizer will request so fpm(wvl) returns an on-backend array.
        storage = getattr(getattr(fpm, "f", None), "storage", None)
        if storage is not None:
            for wvl in wavelengths:
                storage[wvl] = asbackend(fpm(wvl))

    return coro
