"""Analytic gradient (VJP) for prysm.geometry.truecircle.

Forward function (from prysm/prysm/geometry.py)
------------------------------------------------
    def truecircle(radius, r):
        if radius == 0:
            return zeros_like(r)
        samples     = r.shape[0]
        one_pixel   = 2 / samples
        radius_plus = radius + one_pixel / 2        # = radius + 1/samples
        intermediate = (radius_plus - r) * (samples / 2)
        return clip(intermediate, 0, 1)

Derivative derivation
---------------------
All intermediate quantities are affine in `radius` and `r`, so the chain rule
is exact with no approximation.

Let  s  = samples / 2   (a constant given the grid)
     rp = radius + 1/samples

Then:
    intermediate = (rp - r) * s
    out          = clip(intermediate, 0, 1)

  d(out) / d(intermediate) = 1   if  0 < intermediate < 1   (transition band)
                            = 0   otherwise  (including at the kinks themselves)

  d(intermediate) / d(radius) = s   (elementwise)
  d(intermediate) / d(r[i,j]) = -s  (elementwise)

Combined (chain rule):
  d(out[i,j]) / d(radius)   =  s  * in_transition[i,j]
  d(out[i,j]) / d(r[i,j])   = -s  * in_transition[i,j]

where in_transition = (intermediate > 0) & (intermediate < 1).

The transition band spans  radius - 1/samples < r < radius + 1/samples,
i.e., one pixel of width centred on the aperture edge.

VJP (vector-Jacobian product)
------------------------------
Given an upstream gradient  g  (same shape as `out`, representing
d(loss)/d(out) for some scalar loss):

    g_radius = sum(g * s * in_transition)        [scalar]
    g_r      = g * (-s) * in_transition          [array, same shape as r]
"""

from prysm.mathops import np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _truecircle_intermediates(radius, r):
    """Compute the transition mask and half-samples constant.

    These are the only quantities needed for the backward pass.

    Parameters
    ----------
    radius : float
        Aperture radius, same units as ``r``.
    r : ndarray
        2-D array of radial coordinates.

    Returns
    -------
    in_transition : ndarray of bool
        True wherever the output is in the (0, 1) ramp region.
    half_samples : float
        samples / 2, the local slope of the ramp inside the transition band.
    """
    if radius == 0:
        return np.zeros(r.shape, dtype=bool), r.shape[0] / 2.0

    samples = r.shape[0]
    half_samples = samples / 2.0
    # one_pixel = 2 / samples  =>  one_pixel/2 = 1/samples
    radius_plus = radius + 1.0 / samples
    intermediate = (radius_plus - r) * half_samples
    in_transition = (intermediate > 0) & (intermediate < 1)
    return in_transition, half_samples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def truecircle_fwd(radius, r):
    """Forward pass of truecircle that also returns residuals for the backward.

    Identical in output to ``prysm.geometry.truecircle(radius, r)`` but
    additionally returns the information needed by :func:`truecircle_bwd`.

    Parameters
    ----------
    radius : float
        Aperture radius, same units as ``r``.
    r : ndarray
        2-D array of radial coordinates.

    Returns
    -------
    out : ndarray
        Anti-aliased circular mask in [0, 1].
    residuals : tuple
        ``(in_transition, half_samples)`` — pass unchanged to
        :func:`truecircle_bwd`.
    """
    if radius == 0:
        out = np.zeros_like(r)
        residuals = (np.zeros(r.shape, dtype=bool), r.shape[0] / 2.0)
        return out, residuals

    samples = r.shape[0]
    half_samples = samples / 2.0
    radius_plus = radius + 1.0 / samples
    intermediate = (radius_plus - r) * half_samples
    out = np.minimum(np.maximum(intermediate, 0), 1)
    in_transition = (intermediate > 0) & (intermediate < 1)
    residuals = (in_transition, half_samples)
    return out, residuals


def truecircle_bwd(g, residuals):
    """Backward pass (VJP) for truecircle.

    Given the upstream gradient ``g = d(loss)/d(out)``, returns the
    downstream gradients w.r.t. ``radius`` and ``r``.

    Parameters
    ----------
    g : ndarray
        Upstream gradient, same shape as the output of :func:`truecircle_fwd`.
    residuals : tuple
        The ``(in_transition, half_samples)`` tuple returned by
        :func:`truecircle_fwd`.

    Returns
    -------
    g_radius : float
        ``d(loss) / d(radius)``.  Scalar — the sum of all pixel contributions
        through the transition band.
    g_r : ndarray
        ``d(loss) / d(r)``.  Same shape as ``r``.  Non-zero only inside the
        one-pixel-wide transition band at the aperture edge.
    """
    in_transition, half_samples = residuals
    # Local derivative of out w.r.t. intermediate (1 in band, 0 outside)
    # times the slope of intermediate w.r.t. radius (+half_samples)
    # or r (-half_samples).
    local = in_transition * half_samples  # scalar mult, result is float array
    g_radius = np.sum(g * local)  # scalar (or 0-d array)
    g_r = -g * local  # same shape as r
    return g_radius, g_r


def truecircle_grad(g, radius, r):
    """Convenience wrapper: run forward + backward in one call.

    Equivalent to calling :func:`truecircle_fwd` and then
    :func:`truecircle_bwd`.  Useful when you only need the gradients and do
    not need to cache residuals across multiple backward calls.

    Parameters
    ----------
    g : ndarray
        Upstream gradient, same shape as the output of truecircle.  Pass
        ``np.ones_like(r)`` to obtain the gradient of the pixel sum.
    radius : float
        Aperture radius, same units as ``r``.
    r : ndarray
        2-D array of radial coordinates.

    Returns
    -------
    g_radius : float
        ``d(loss) / d(radius)``.
    g_r : ndarray
        ``d(loss) / d(r)``, same shape as ``r``.

    Examples
    --------
    Gradient of the total aperture area w.r.t. radius::

        import numpy as np
        from prysm.coordinates import make_xy_grid, cart_to_polar
        from dygdug.truecircle_grad import truecircle_grad

        N = 256
        x, y = make_xy_grid(N, dx=1/N)
        r, _ = cart_to_polar(x, y)

        g_radius, g_r = truecircle_grad(np.ones_like(r), radius=0.4, r=r)
        # g_radius ≈ circumference in pixel units
    """
    _, residuals = truecircle_fwd(radius, r)
    return truecircle_bwd(g, residuals)
