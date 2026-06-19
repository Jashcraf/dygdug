from prysm.mathops import fft, np

# Cost-function domain convention
# --------------------------------
# Each cost class advertises the quantity it consumes via a ``domain`` class
# attribute, read by the optimizers in ``dygdug.coropt``:
#
# ``"intensity"`` (default)
#     ``forward`` / ``reverse`` operate on the real dark-hole intensity
#     ``I = |E|^2``.  ``reverse(I)`` returns ``dJ/dI``; the optimizer forms the
#     Wirtinger back-propagation seed as ``dJ/dE* = (dJ/dI) * E``.
#
# ``"field"``
#     ``forward`` / ``reverse`` operate on the complex dark-hole field ``E``
#     directly, so the cost may constrain the real and imaginary parts
#     independently.  ``reverse(E)`` returns the Wirtinger seed ``dJ/dE*``,
#     which the optimizer feeds straight into ``coro.reverse``.
#
# Classes without an explicit ``domain`` are treated as ``"intensity"``.


def log_sum_exp(x, alpha=1):
    """
    LogSumExp function, a smooth approximation to max()

    Parameters
    ----------
    x: ndarray
        data to find maximum value of
    alpha: float, optional
        "steepness" parameter for the function. Larger values
        tend to have higher accuracy for small x, but can
        result in underflow errors
    """
    return 1 / alpha * np.log(np.sum(np.exp(alpha * x)))


def softmax(x, alpha=1):
    """
    Softmax function, returns a probability distribution of the "likelihood"
    of every value to be the maximum value. Happens to be gradient of
    log_sum_exp

    Parameters
    ----------
    x: ndarray
        data to find maximum value of
    alpha: float, optional
        "steepness" parameter for the function. Larger values
        tend to have higher accuracy for small x, but can
        result in underflow errors
    """
    return np.exp(alpha * x) / np.sum(np.exp(alpha * x))


class LogSumExp:
    domain = "intensity"

    def __init__(self, target=0, alpha=1):
        """Object interface for the LogSumExp cost function, with
        'forward' and 'reverse' method for use in models with
        analytic gradients.
        """
        self.target = target
        self.alpha = alpha

    def forward(self, x):
        return log_sum_exp(self.target - x, alpha=self.alpha)

    def reverse(self, x):
        return softmax(self.target - x, alpha=self.alpha)


class MaxContrast:
    domain = "intensity"

    def __init__(self, target=0, alpha=1):
        """
        Targets maximum value, uses softmax to approximate gradient
        """
        self.target = target
        self.alpha = alpha

    def forward(self, x):
        return np.max(self.target - x)

    def reverse(self, x):
        return softmax(self.target - x, alpha=self.alpha)


class MeanSquaredErrorLinear:
    domain = "intensity"

    def __init__(self, target=0, alpha=1.0):
        """Mean squared error cost function with a linear
        penalty. Sign changes when constraint target is satisfied.
        In english, this means you are asking the optimizer:
        "Hey please get to 'target', but if you can do better, that's great"

        Parameters
        ----------
        target: float
            Target contrast. These are implicitly converted to squared unites.
            If you give it 1e-2, it will target 1e-4 so that 1e-2 is achieved.
        alpha: float
            Weight to multiply constraint by. Can be negative
            to flip the sign convention
        """
        self.target = target**2
        self.alpha = alpha

    def forward(self, x):
        err = x
        mse_mag = np.mean(err**2)
        return (mse_mag - self.target) * self.alpha

    def reverse(self, x):
        err = x
        mse_grad = 2 * err / x.size
        return mse_grad * self.alpha


class MeanSquaredErrorQuadratic:
    domain = "intensity"

    def __init__(self, target=0, alpha=1.0):
        """Mean squared error cost function with a linear
        penalty. Sign changes when constraint target is satisfied
        In english, this means you are asking the optimizer:
        "Hey please get to 'target', but if you can do better, don't"

        Parameters
        ----------
        target: float
            Target contrast, not MSE units.
        alpha: float
            Weight to multiply constraint by. Can be negative
            to flip the sign convention
        """
        self.target = target
        self.alpha = alpha

    def forward(self, x):
        err = x - self.target
        mse_mag = np.sum(err**2)
        return mse_mag * self.alpha

    def reverse(self, x):
        err = x - self.target
        mse_grad = 2 * err  #  / x.size
        return mse_grad * self.alpha


class PNorm:
    domain = "intensity"

    def __init__(self, target=0, alpha=10):
        self.target = target
        self.alpha = alpha

    def forward(self, x):
        exponent = 1 / self.alpha
        self.p_norm = (np.sum(x) ** self.alpha) ** exponent
        return self.target - self.p_norm

    def reverse(self, x):
        return x ** (self.alpha - 1) / (self.p_norm ** (self.alpha - 1))


class CoreThroughput:
    domain = "intensity"

    def __init__(self, target=0, alpha=1):
        """Core Throughput Maximization, negative sign applied to both
        forward and reverse to maximize core throughput instead of
        minimize

        Must be used with a dark_hole equal to the desired core window, with
        no FPM in place

        Parameters
        ----------
        target: ndarray
            Core window on-axis to evaluate the core throughput
            of a coronagraph.
        alpha: float
            weight to place on this cost function
        """
        self.target = target
        self.alpha = alpha

    def forward(self, x):
        """
        Sum in core window

        Parameters
        ----------
        x: ndarray
            Image plane intensity
        """

        return -1 * np.sum(x - self.target) * self.alpha

    def reverse(self, x):
        """
        Parameters
        ----------
        x: ndarray
            Image plane intensity
        """
        return -1 * (x - self.target) * self.alpha


class FieldMeanSquaredError:
    """Quadratic penalty on the real and imaginary parts of the field.

    A ``domain == "field"`` cost: ``forward`` / ``reverse`` operate on the
    complex dark-hole field ``E`` directly rather than on the intensity
    ``|E|^2``, so the real and imaginary parts can be driven to (and held at) a
    target independently.  This lets you, e.g., constrain ``Re(E)`` and
    ``Im(E)`` symmetrically instead of only their magnitude.

    The cost is

    ``J = alpha * sum( w_re (Re E - Re t)^2 + w_im (Im E - Im t)^2 )``

    where ``t`` is the (complex) ``target``.  With the defaults
    (``target = 0``, ``w_re = w_im = 1``) this reduces to ``alpha * sum |E|^2``.

    Parameters
    ----------
    target : complex or ndarray, optional
        Desired field value ``t``.  Scalar (broadcast over the dark hole) or an
        array matching the dark-hole field.  Default 0.
    alpha : float, optional
        Overall weight.  May be negative to flip the sign convention.
    real_weight, imag_weight : float, optional
        Independent weights on the real and imaginary squared errors.  Set one
        to 0 to constrain only the other component.
    """

    domain = "field"

    def __init__(self, target=0.0, alpha=1.0, real_weight=1.0, imag_weight=1.0):
        self.target = target
        self.alpha = alpha
        self.real_weight = real_weight
        self.imag_weight = imag_weight

    def forward(self, E):
        """Scalar cost from the complex dark-hole field ``E``."""
        err = E - self.target
        cost = self.real_weight * err.real**2 + self.imag_weight * err.imag**2
        return np.sum(cost) * self.alpha

    def reverse(self, E):
        """Wirtinger seed ``dJ/dE*`` for the complex dark-hole field ``E``.

        For a real cost ``J(Re E, Im E)`` the back-propagation seed is
        ``dJ/dE* = (1/2)(dJ/d(Re E) + i dJ/d(Im E))``.  Here
        ``dJ/d(Re E) = 2 alpha w_re (Re E - Re t)`` and likewise for the
        imaginary part, so the factor of 1/2 cancels the 2 to give the
        expression below.
        """
        err = E - self.target
        return self.alpha * (
            self.real_weight * err.real + 1j * self.imag_weight * err.imag
        )


# Constraint-function protocol (AugmentedLagrangian)
# --------------------------------------------------
# AugmentedLagrangian drives its dark-hole inequality constraints through a
# pluggable *constraint object* (passed as ``constraint=``).  Unlike the scalar
# forward/reverse cost protocol above, a constraint object exposes the per-pixel
# residual vector ``c(x)`` (``<= 0`` is feasible) and the Wirtinger seed of the
# active constraints, so it can bound the intensity or the real/imaginary field
# components independently.  A constraint object must implement:
#
#   multiplier_shape(n_wvl, n) -> tuple
#       Shape of the multiplier/constraint array.  The trailing axes are the
#       per-wavelength constraint shape returned by ``residual``.
#   residual(E_dh, norm, contrast) -> ndarray
#       Constraint values for one wavelength's dark-hole field ``E_dh`` (1-D,
#       length n), peak normalization ``norm`` (scalar), and per-pixel
#       ``contrast`` (length n).  ``<= 0`` is feasible.  Shape equals
#       ``multiplier_shape(n_wvl, n)[1:]``.
#   grad_seed(E_dh, norm, contrast, positive) -> complex ndarray, shape (n,)
#       Wirtinger seed ``dJ/dE*`` of the AL penalty w.r.t. the dark-hole field,
#       given the activated multipliers ``positive`` (same shape as residual).
#       The AL back-propagates this through the coronagraph for the gradient.


class IntensityConstraint:
    """Dark-hole intensity constraint ``|E|^2 / norm - contrast <= 0``.

    A convex quadratic per pixel whose optimum lies on a smooth (gray)
    boundary.  One inequality per dark-hole pixel per wavelength, so the
    multiplier array has shape ``(n_wvl, n)``.
    """

    domain = "intensity"

    def multiplier_shape(self, n_wvl, n):
        return (n_wvl, n)

    def residual(self, E_dh, norm, contrast):
        intensity = (E_dh.real * E_dh.real + E_dh.imag * E_dh.imag) / norm
        return intensity - contrast

    def grad_seed(self, E_dh, norm, contrast, positive):
        # c = |E|^2/norm - contrast, so dc/dE* = E/norm and the penalty seed is
        # dJ/dE* = positive * dc/dE* = (positive / norm) * E.
        return (positive / norm) * E_dh


class FieldBoxConstraint:
    """Linear box constraint on the normalized field components.

    With ``a = E / sqrt(norm)`` and ``s = sqrt(contrast / 2)`` this bounds
    ``|Re a| <= s`` and ``|Im a| <= s`` -- the axis-aligned box inscribed in the
    ``|a|^2 <= contrast`` intensity disk, so satisfying the box guarantees the
    intensity contrast.  Two inequalities per pixel (axis 1: ``0 -> real``,
    ``1 -> imag``), so the multiplier array has shape ``(n_wvl, 2, n)``.  A
    linear program whose feasible polytope has binary vertices, so throughput
    maximization drives a binary apodizer.
    """

    domain = "field"

    def multiplier_shape(self, n_wvl, n):
        return (n_wvl, 2, n)

    def residual(self, E_dh, norm, contrast):
        rn = 1.0 / np.sqrt(norm)
        s = np.sqrt(contrast / 2)
        re = np.abs(E_dh.real * rn) - s
        im = np.abs(E_dh.imag * rn) - s
        return np.stack([re, im], axis=0)

    def grad_seed(self, E_dh, norm, contrast, positive):
        # dc/dRe(E) = sign(Re a)/sqrt(norm) and dc/dIm(E) likewise, so
        # dJ/dE* = (1/2)(dJ/dRe + i dJ/dIm)
        #        = 0.5 rn (pos_re sign(Re a) + i pos_im sign(Im a)).
        rn = 1.0 / np.sqrt(norm)
        re = E_dh.real * rn
        im = E_dh.imag * rn
        pos_re, pos_im = positive[0], positive[1]
        return 0.5 * rn * (pos_re * np.sign(re) + 1j * pos_im * np.sign(im))
