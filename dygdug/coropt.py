"""Coronagraph optimizer."""

from prysm import coordinates, geometry
from prysm.mathops import np

from dygdug.cost_functions import MeanSquaredErrorQuadratic
from dygdug.masks import FPM, Pupil


class VariablePupil(Pupil):
    """A Pupil whose active pixels can be updated from a flat parameter vector.

    Constructed identically to :class:`Pupil` (same classmethods: ``circle``,
    ``annular``, ``hexagonal_segmented``), with an additional *mode* argument
    that selects between amplitude and phase apodization:

    ``'amplitude'`` (default)
        ``x`` contains real transmission values.  ``pupil.data`` is stored as
        floating-point data so continuous amplitude updates are not coerced
        back to a binary mask.

    ``'phase'``
        ``x`` contains real phase values φ.  ``pupil.data`` is complex
        (``exp(1j·φ)``), which sets ``Coronagraph.PUPIL_IS_COMPLEX = True``.
    """

    def __init__(self, data, dx, mode="amplitude"):
        if mode not in ("amplitude", "phase"):
            raise ValueError(f"mode must be 'amplitude' or 'phase', got {mode!r}")

        # Capture the active region before any optimization updates.  Some
        # geometries (e.g. CompositeHexagonalAperture.amp) are boolean; keep
        # the mask binary, but cast the stored pupil to a continuous dtype.
        self.mask = data == 1
        self._mask_idx = np.flatnonzero(self.mask.ravel())
        self._n_params = int(self._mask_idx.size)
        if mode == "phase":
            # Cast to complex so Coronagraph.PUPIL_IS_COMPLEX is set correctly.
            data = data.astype(complex)
        else:
            data = data.astype(float)

        super().__init__(data, dx)
        self.mode = mode

    @property
    def n_params(self):
        """Number of optimizable parameters (active pixel count)."""
        return int(self.mask.sum())

    def update(self, x):
        """Scatter parameter vector *x* into the active (mask == True) pixels.

        Parameters
        ----------
        x : array_like
            1-D array of length ``self.mask.sum()``.
            *Amplitude mode*: real transmission values written directly.
            *Phase mode*: real phase values φ; stored as ``exp(1j·φ)``.
        """
        if self.mode == "phase":
            self.data[self.mask] = np.exp(1j * x)
        else:
            self.data[self.mask] = x

    # ------------------------------------------------------------------
    # Classmethods — thin wrappers that add the *mode* parameter
    # ------------------------------------------------------------------
    # The parent classmethods call cls(data=data, dx=dx), which hits
    # VariablePupil.__init__ with mode defaulting to 'amplitude'.  These
    # overrides let callers explicitly request phase mode without needing
    # to construct the geometry themselves.

    @classmethod
    def circle(cls, Dpup, Npup, mode="amplitude"):
        if mode not in ("amplitude", "phase"):
            raise ValueError(f"mode must be 'amplitude' or 'phase', got {mode!r}")
        inst = super().circle(Dpup, Npup)
        if mode == "phase":
            inst.data = inst.data.astype(complex)
            inst.mode = "phase"
        return inst

    @classmethod
    def annular(cls, Dpup, Npup, inner_radius, outer_radius, mode="amplitude"):
        if mode not in ("amplitude", "phase"):
            raise ValueError(f"mode must be 'amplitude' or 'phase', got {mode!r}")
        inst = super().annular(Dpup, Npup, inner_radius, outer_radius)
        if mode == "phase":
            inst.data = inst.data.astype(complex)
            inst.mode = "phase"
        return inst

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
        mode="amplitude",
    ):
        if mode not in ("amplitude", "phase"):
            raise ValueError(f"mode must be 'amplitude' or 'phase', got {mode!r}")
        inst = super().hexagonal_segmented(
            Dpup,
            Npup,
            rings,
            segment_diameter,
            segment_separation,
            segment_angle=segment_angle,
            exclude=exclude,
        )
        if mode == "phase":
            inst.data = inst.data.astype(complex)
            inst.mode = "phase"
        return inst


class VariableFPM(FPM):
    """A FPM whose active pixels can be updated from a flat parameter vector.

    This class is in progress, and doesnt support wavelength-dependent optimization
    just yet. This is mostly out of a lack of need, but could be relevant for
    designing Hybrid-Lyot Coronagraphs in the future.

    Constructed identically to :class:`FPM` (same classmethods: ``lyot``,
    ``annular``, ``unity``).  On construction the underlying function is
    evaluated once (these masks are all wavelength-independent in practice)
    to snapshot both the mutable :attr:`data` array and the boolean
    :attr:`mask` of active pixels.  :meth:`__call__` then returns
    :attr:`data` directly, bypassing the original function.
    """

    def __init__(self, func_lam, dx):
        super().__init__(func_lam, dx)
        # Evaluate once to get the initial spatial array.
        # None is a safe sentinel: all built-in fpmfunc closures ignore wvl.
        initial = func_lam(None)
        self.data = initial.copy()  # ensure a mutable, owned array
        self.mask = initial == 1

    def __call__(self, wvl):
        # Return the mutable data array; wvl is accepted for API compatibility.
        return self.data

    @property
    def n_params(self):
        """Number of optimizable parameters (active pixel count)."""
        return int(self.mask.sum())

    def update(self, x):
        """Scatter parameter vector *x* into the active (mask == True) pixels.

        Parameters
        ----------
        x : array_like
            1-D array of length ``self.mask.sum()``.  Values are written
            in row-major order into every pixel where the original FPM
            data equalled 1.
        """
        self.data[self.mask] = x


class VariableLyotStop(Pupil):
    """A Lyot stop whose inner/outer radii are the optimizable parameters.

    Unlike :class:`VariablePupil`, the parameter vector passed to
    :meth:`update` does not address individual pixels — it addresses the
    geometric radii of the stop, and :attr:`data` is recomputed from the
    stored radial grid on every call.

    Use :meth:`circular` for a solid disk stop (1 DOF: outer radius) or
    :meth:`annular` for a ring-shaped stop (2 DOFs: inner and outer radius).
    """

    def __init__(self, data, dx, r, annular):
        super().__init__(data, dx)
        self._r = r  # radial coordinate grid, computed once
        self._annular = annular

    @classmethod
    def circular(cls, Dpup, Npup, outer_radius):
        """Circular (solid disk) Lyot stop.

        Parameters
        ----------
        Dpup : float
            Pupil diameter in physical units.
        Npup : int
            Array side length in pixels.
        outer_radius : float
            Initial radius of the stop in physical units.
        """
        dx = Dpup / Npup
        x, y = coordinates.make_xy_grid(Npup, dx=dx)
        r = np.hypot(x, y)
        data = geometry.truecircle(outer_radius, r)
        return cls(data=data, dx=dx, r=r, annular=False)

    @classmethod
    def annular(cls, Dpup, Npup, inner_radius, outer_radius):
        """Annular (ring-shaped) Lyot stop.

        Parameters
        ----------
        Dpup : float
            Pupil diameter in physical units.
        Npup : int
            Array side length in pixels.
        inner_radius : float
            Initial inner radius of the annulus in physical units.
        outer_radius : float
            Initial outer radius of the annulus in physical units.
        """
        dx = Dpup / Npup
        x, y = coordinates.make_xy_grid(Npup, dx=dx)
        r = np.hypot(x, y)
        data = geometry.truecircle(outer_radius, r) * (
            1 - geometry.truecircle(inner_radius, r)
        )
        return cls(data=data, dx=dx, r=r, annular=True)

    @property
    def n_params(self):
        """Number of optimizable parameters (1 for circular, 2 for annular)."""
        return 2 if self._annular else 1

    def update(self, x):
        """Recompute the Lyot stop from new radius parameters.

        Parameters
        ----------
        x : array_like
            For a circular stop: ``[outer_radius]``.
            For an annular stop: ``[inner_radius, outer_radius]``.
        """
        if self._annular:
            inner_r, outer_r = x
            self.data = geometry.truecircle(outer_r, self._r) * (
                1 - geometry.truecircle(inner_r, self._r)
            )
        else:
            self.data = geometry.truecircle(x[0], self._r)


class ThroughputOptimizer:
    """Coronagraph optimizer that sums the values in the apodizer
    only meaningful for Amplitude-apodized coronagraphs as a proxy
    for core throughput, which is an application of CoronagraphOptimizer
    """

    def __init__(self, coro):

        # --- Discover variable elements in pupil → fpm → lyot_stop order ---
        self._variable_elems = []  # [(attr_name, element), ...]
        self._slices = []  # [slice into flat x, ...]
        offset = 0

        # Scan for pupil
        for attr in ("pupil", "fpm", "lyot_stop"):
            elem = getattr(coro, attr)
            if hasattr(elem, "n_params"):
                n = elem.n_params
                self._variable_elems.append((attr, elem))
                self._slices.append(slice(offset, offset + n))
                offset += n
        self.n_params = offset

    def _push(self, x):
        """Distribute *x* slices to their respective variable elements."""
        for sl, (_, elem) in zip(self._slices, self._variable_elems):
            elem.update(x[sl])

    def fg(self, x):
        """Evaluate cost *J* and gradient *dJ/dx* for parameter vector *x*."""
        self._push(x)


class CoronagraphOptimizer:
    """Coronagraph optimizer with automatic variable-element discovery.

    On construction, ``coro.pupil``, ``coro.fpm``, and ``coro.lyot_stop``
    are inspected for objects that expose an ``n_params`` attribute
    (:class:`VariablePupil`, :class:`VariableFPM`, :class:`VariableLyotStop`).
    A contiguous flat parameter layout is built from those elements in
    ``pupil → fpm → lyot_stop`` order so that :meth:`fg` can distribute
    a single vector *x* to the right place.

    Parameters
    ----------
    dark_hole : ndarray of bool
        2-D boolean mask selecting the dark-hole pixels in the final image
        plane.
    coro : Coronagraph
        Coronagraph model to optimise.
    wvl : float or sequence of float
        Wavelength(s) to include.  Cost and gradient are summed over all
        wavelengths.
    cost : cost-function class or instance, optional
        Must expose ``.forward(I)`` → scalar and ``.reverse(I)`` → array.
        Defaults to :class:`MeanSquaredErrorQuadratic`.
    """

    def __init__(self, dark_hole, coro, wvl, cost=MeanSquaredErrorQuadratic):
        # Accept 0/1 mask arrays, but always index with a boolean mask.
        # Integer masks would otherwise trigger NumPy advanced indexing.
        self.dh = np.asarray(dark_hole).astype(bool)
        self.coro = coro

        # Accept a scalar or a sequence of wavelengths.
        self.wvl = [wvl] if np.ndim(wvl) == 0 else list(wvl)

        # Accept the cost-function class or an already-constructed instance.
        self.cost_fn = cost() if isinstance(cost, type) else cost

        # --- Discover variable elements in pupil → fpm → lyot_stop order ---
        self._variable_elems = []  # [(attr_name, element), ...]
        self._slices = []  # [slice into flat x, ...]
        offset = 0

        # coro is our forward mode
        for attr in ("pupil", "fpm", "lyot_stop"):
            elem = getattr(coro, attr)
            if hasattr(elem, "n_params"):
                n = elem.n_params
                self._variable_elems.append((attr, elem))
                self._slices.append(slice(offset, offset + n))
                offset += n
        self.n_params = offset

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fg(self, x):
        """Evaluate cost *J* and gradient *dJ/dx* for parameter vector *x*.

        Variable elements are updated first, then the coronagraph is
        propagated forward.  Gradients are computed analytically for
        :class:`VariablePupil` via the adjoint model, and with central
        finite differences for :class:`VariableLyotStop` (1–2 DOFs, cheap).
        :class:`VariableFPM` gradient is not yet implemented.

        Parameters
        ----------
        x : array_like, shape ``(n_params,)``
            Flat parameter vector.  Layout is ``pupil | fpm | lyot_stop``
            (only present variable elements are included).

        Returns
        -------
        J : float
            Scalar cost summed over all wavelengths.
        grad : ndarray, shape ``(n_params,)``
            Gradient of *J* w.r.t. *x*.
        """

        # Updates all optimizeable optics
        self._push(x)

        # Accumulates cost and gradient over all wavelengths
        J = 0.0
        grad = np.zeros(self.n_params)

        for wvl in self.wvl:
            # Evaluate cost function on dh intensity
            E_focal = self.coro.forward(wvl, include_fpm=True)
            I_dh = np.abs(E_focal[self.dh]) ** 2
            J += float(self.cost_fn.forward(I_dh))

            # Wirtinger gradient at focal plane: dJ/dE* = (dJ/dI) · E.
            Ebar = np.zeros_like(E_focal)
            Ebar[self.dh] = self.cost_fn.reverse(I_dh) * E_focal[self.dh]

            # Back-propagate gradient to pupil plane (return value is 2D;
            # relevant slices are extracted below by the per-element helpers).
            self.coro.reverse(Ebar, wvl, include_fpm=True)

            for (attr, elem), sl in zip(self._variable_elems, self._slices):
                if isinstance(elem, VariablePupil):
                    grad[sl] += self._pupil_grad(elem)
                # VariableFPM: TODO (needs intermediate FPM-plane field)
                # VariableLyotStop: handled below via finite differences until
                # analytical gradient is available

        # Finite-difference gradient for VariableLyotStop.
        # Done outside the wavelength loop so it samples the full summed cost.
        for (attr, elem), sl in zip(self._variable_elems, self._slices):
            if isinstance(elem, VariableLyotStop):
                grad[sl] = self._fd_grad(x, sl)

        return J, grad

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def _push(self, x):
        """Distribute *x* slices to their respective variable elements."""
        for (attr, elem), sl in zip(self._variable_elems, self._slices):
            elem.update(x[sl])

    def _pupil_grad(self, elem):
        """Analytical gradient w.r.t. the real optimisation variables of a
        :class:`VariablePupil`, dispatched on ``elem.mode``.

        ``coro.reverse`` stores the Wirtinger adjoint
        ``adj = dJ/d(pupil.data)*``.  Because the optimisation variables are
        real, the directional derivative is
        ``dJ = 2 Re(conj(adj) · d(pupil.data))``.

        *Amplitude mode* — ``x`` are real transmissions, so
            ``d(pupil.data)/dx = 1`` and ``dJ/dx = 2 Re(adj)``.

        *Phase mode* — ``x`` are real phases φ, ``pupil.data = exp(iφ)``, so
            ``d(pupil.data)/dφ = i exp(iφ)`` and
            ``dJ/dφ = 2 Im(adj · conj(pupil.data))``.
        """
        adj = self.coro.adjoint_at_entrance_pupil[elem.mask]
        if elem.mode == "phase":
            field = elem.data[elem.mask]
            return 2 * np.imag(adj * field.conj())
        else:
            return 2 * np.real(adj)

    def _total_cost(self, x):
        """Forward-only cost evaluation used by the finite-difference helper."""
        self._push(x)
        J = 0.0
        for wvl in self.wvl:
            I_dh = np.abs(self.coro.forward(wvl, include_fpm=True)[self.dh]) ** 2
            J += float(self.cost_fn.forward(I_dh))
        return J

    def _fd_grad(self, x, sl, eps=1e-7):
        """Central-difference gradient for the parameters in slice *sl*.

        Only the elements in *sl* are perturbed; all variable elements
        are restored to *x* before returning.
        """
        g = np.empty(sl.stop - sl.start)
        x_work = x.copy()
        for i, idx in enumerate(range(sl.start, sl.stop)):
            x_work[idx] = x[idx] + eps
            J_plus = self._total_cost(x_work)
            x_work[idx] = x[idx] - eps
            J_minus = self._total_cost(x_work)
            x_work[idx] = x[idx]  # restore before next iteration
            g[i] = (J_plus - J_minus) / (2.0 * eps)
        self._push(x)  # restore all elements to original x
        return g


class JointOptimizer:
    def __init__(self, optlist):
        """
        For performing joint optimization of multiple parameters
        """
        self.optlist = optlist

    def fg(self, x):
        """
        Sums all of the functions and gradients
        """
        f = 0
        g = np.zeros(x.shape)

        for opt in self.optlist:
            _f, _g = opt.fg(x)
            f += _f
            g += _g

        return f, g
