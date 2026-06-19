"""
Example: Creating sinusoidal wavefront error in the pupil plane
for a coronagraph with deformable mirror.
"""

import numpy as np
from prysm import coordinates

from dygdug.models import ImgSamplingSpec


def pixel_to_pupil_tilt(pixel_i, pixel_j, imgspec, N_pup, dx_pup, wavelength):
    """
    Convert an image plane pixel location to the required tilt (gradient) in the pupil plane
    to direct light to that pixel.

    Parameters
    ----------
    pixel_i : int
        Row index in image plane (0 to imgspec.N-1)
    pixel_j : int
        Column index in image plane (0 to imgspec.N-1)
    imgspec : ImgSamplingSpec
        Image sampling specification
    N_pup : int
        Size of pupil plane array
    dx_pup : float
        Pupil plane pixel size in meters
    wavelength : float
        Wavelength in meters

    Returns
    -------
    tuple of float
        (tilt_x, tilt_y) - phase gradient in radians per meter
    float
        Gradient magnitude (radians per meter) in pupil plane
    """
    # Convert pixel to image plane coordinates in meters
    center = imgspec.N / 2.0
    x_img = (pixel_j - center) * imgspec.dx * 1e-6  # convert microns to meters
    y_img = (pixel_i - center) * imgspec.dx * 1e-6  # convert microns to meters

    # Phase gradient in pupil plane needed to steer light to image plane position:
    # phase_gradient = 2*pi * image_coordinate / wavelength
    tilt_x = 2 * np.pi * x_img / wavelength
    tilt_y = 2 * np.pi * y_img / wavelength

    gradient_magnitude = np.sqrt(tilt_x**2 + tilt_y**2)

    return (tilt_x, tilt_y), gradient_magnitude


def create_sinusoidal_wfe_pupil_plane(
    N_pup, dx_pup, amplitude, frequency_x, frequency_y, phase=0
):
    """
    Create a 2D sinusoidal wavefront error map in the pupil plane.

    Parameters
    ----------
    N_pup : int
        Number of samples across the pupil array
    dx_pup : float
        Pupil plane pixel size in meters
    amplitude : float
        Amplitude of the sinusoid in wavefront units (meters, typically)
    frequency_x : float
        Spatial frequency in x direction (cycles per meter)
    frequency_y : float
        Spatial frequency in y direction (cycles per meter)
    phase : float, optional
        Phase offset (radians)

    Returns
    -------
    numpy.ndarray
        2D array of shape (N_pup, N_pup) containing wavefront error values in meters
    """
    # Create coordinate grids in the pupil plane
    x, y = coordinates.make_xy_grid(N_pup, dx=dx_pup)

    # Compute sinusoidal wavefront error
    wfe = amplitude * np.sin(2 * np.pi * (frequency_x * x + frequency_y * y) + phase)

    return wfe


def create_wfe_from_image_pixel(
    pixel_i,
    pixel_j,
    imgspec,
    N_pup,
    dx_pup,
    wavelength,
    amplitude,
    add_defocus=False,
    add_tilt=True,
):
    """
    Create a wavefront error in the pupil plane that produces effects at a specific
    image plane pixel location.

    This function creates a sinusoidal pattern in the pupil plane. The pattern
    characteristics are determined by the relationship between the image plane
    (defined by imgspec) and pupil plane (N_pup, dx_pup).

    Parameters
    ----------
    pixel_i : int
        Row index in image plane (0 to imgspec.N-1)
    pixel_j : int
        Column index in image plane (0 to imgspec.N-1)
    imgspec : ImgSamplingSpec
        Image sampling specification
    N_pup : int
        Size of pupil plane array
    dx_pup : float
        Pupil plane pixel size in meters
    wavelength : float
        Wavelength in meters
    amplitude : float
        Amplitude of sinusoidal pattern in meters
    add_defocus : bool, optional
        If True, add a defocus component (default: False)
    add_tilt : bool, optional
        If True, add a tilt component pointing to the image pixel (default: True)

    Returns
    -------
    numpy.ndarray
        2D array of shape (N_pup, N_pup) containing wavefront error in meters
    dict
        Diagnostic information
    """
    # Create base sinusoidal pattern
    # Spatial frequency based on wavelength scale
    freq_scale = 1.0 / (wavelength * 4)  # cycles per meter

    wfe = create_sinusoidal_wfe_pupil_plane(
        N_pup,
        dx_pup,
        amplitude=amplitude,
        frequency_x=freq_scale,
        frequency_y=freq_scale,
        phase=0,
    )

    # Optionally add tilt to steer toward the image pixel
    if add_tilt:
        (tilt_x, tilt_y), gradient_mag = pixel_to_pupil_tilt(
            pixel_i, pixel_j, imgspec, N_pup, dx_pup, wavelength
        )
        x, y = coordinates.make_xy_grid(N_pup, dx=dx_pup)
        wfe += tilt_x * x + tilt_y * y
    else:
        gradient_mag = 0.0

    # Optionally add defocus
    if add_defocus:
        x, y = coordinates.make_xy_grid(N_pup, dx=dx_pup)
        r_sq = x**2 + y**2
        wfe += 0.1 * r_sq  # arbitrary defocus strength

    diagnostics = {
        "image_pixel": (pixel_i, pixel_j),
        "pupil_shape": (N_pup, N_pup),
        "pupil_dx_m": dx_pup,
        "wavelength_m": wavelength,
        "amplitude_m": amplitude,
        "has_tilt": add_tilt,
        "has_defocus": add_defocus,
        "max_wfe_m": np.max(np.abs(wfe)),
        "rms_wfe_m": np.sqrt(np.mean(wfe**2)),
        "gradient_magnitude_rad_per_m": gradient_mag if add_tilt else 0.0,
    }

    return wfe, diagnostics


# Example usage
if __name__ == "__main__":
    # Define image plane sampling
    N_img = 256
    px_per_lamD = 4
    lamD_um = 5.0
    imgspec = ImgSamplingSpec.from_N_lamD_px_per_lamD(N_img, lamD_um, px_per_lamD)

    print("Image plane sampling specification:")
    print(f"  Array size: {imgspec.N}x{imgspec.N}")
    print(f"  Pixel size: {imgspec.dx} microns")
    print(f"  lambda/D: {imgspec.lamD} microns")
    print()

    # Define pupil plane
    N_pup = 128
    wavelength_m = 500e-9  # 500 nm
    pupil_diameter_m = 1.0
    dx_pup = pupil_diameter_m / N_pup

    print("Pupil plane parameters:")
    print(f"  Array size: {N_pup}x{N_pup}")
    print(f"  Pixel size: {dx_pup:.6f} meters")
    print(f"  Wavelength: {wavelength_m * 1e9:.1f} nm")
    print()

    # Example 1: Simple sinusoidal WFE in pupil plane
    print("Example 1: Simple sinusoidal WFE in pupil plane")
    freq_pupil = 1.0 / wavelength_m  # cycles per meter
    wfe_simple = create_sinusoidal_wfe_pupil_plane(
        N_pup,
        dx_pup,
        amplitude=10e-9,  # 10 nm
        frequency_x=freq_pupil * 0.1,
        frequency_y=freq_pupil * 0.1,
        phase=0,
    )
    print(f"  WFE shape: {wfe_simple.shape}")
    print(f"  RMS WFE: {np.sqrt(np.mean(wfe_simple**2)) * 1e9:.3f} nm")
    print()

    # Example 2: WFE tailored to center image plane pixel
    print("Example 2: Sinusoidal WFE from center image plane pixel")
    pixel_i, pixel_j = 128, 128  # center

    wfe_center, diag = create_wfe_from_image_pixel(
        pixel_i,
        pixel_j,
        imgspec,
        N_pup,
        dx_pup,
        wavelength=wavelength_m,
        amplitude=10e-9,
        add_tilt=True,
        add_defocus=False,
    )

    print(f"  Image pixel: ({pixel_i}, {pixel_j})")
    print(f"  Max WFE: {diag['max_wfe_m'] * 1e9:.3f} nm")
    print(f"  RMS WFE: {diag['rms_wfe_m'] * 1e9:.3f} nm")
    print(f"  Gradient magnitude: {diag['gradient_magnitude_rad_per_m']:.3e} rad/m")
    print()

    # Example 3: WFE at off-axis pixel with tilt and defocus
    print("Example 3: Off-axis pixel with tilt and defocus")
    pixel_i, pixel_j = 100, 150

    wfe_offaxis, diag = create_wfe_from_image_pixel(
        pixel_i,
        pixel_j,
        imgspec,
        N_pup,
        dx_pup,
        wavelength=wavelength_m,
        amplitude=15e-9,
        add_tilt=True,
        add_defocus=True,
    )

    print(f"  Image pixel: ({pixel_i}, {pixel_j})")
    print(f"  Max WFE: {diag['max_wfe_m'] * 1e9:.3f} nm")
    print(f"  RMS WFE: {diag['rms_wfe_m'] * 1e9:.3f} nm")
    print(f"  Components: tilt={diag['has_tilt']}, defocus={diag['has_defocus']}")
    print(f"  Gradient magnitude: {diag['gradient_magnitude_rad_per_m']:.3e} rad/m")
