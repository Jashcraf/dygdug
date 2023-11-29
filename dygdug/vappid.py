"""vAPP optimizer.

Apodized Amplitude Plate == AAP
Apodized Phase Plate == APP

Naming convention is XXXXOptimizer uses FFTs, XXXX2Optimizer uses MFTs
"""

from prysm.mathops import np, fft

from prysm.propagation import focus_fixed_sampling, focus_fixed_sampling_backprop


def ft_fwd(x):
    return fft.ifftshift(fft.fft2(fft.fftshift(x), norm='ortho'))

def ft_rev(x):
    return fft.ifftshift(fft.ifft2(fft.fftshift(x), norm='ortho'))


def normalize(vapp):
    phs = np.zeros_like(vapp.amp)
    old_phs = vapp.phs.copy()
    vapp.phs = phs
    vapp.fwd(vapp.phs[vapp.amp_select])
    Imax = vapp.I.max()
    norm = np.sqrt(Imax)
    vapp.amp = vapp.amp * (1/norm)  # cheaper to divide scalar, mul array
    vapp.fwd(old_phs[vapp.amp_select])
    return

class VAPPOptimizer:
    def __init__(self, amp, wvl, basis, dark_hole, dh_target=1e-10, initial_phase=None):
        if initial_phase is None:
            phs = np.zeros(amp.shape, dtype=float)

        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.wvl = wvl
        self.basis = basis
        self.dh = dark_hole
        self.D = dh_target
        self.phs = phs
        self.zonal = False

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        if not self.zonal:
            phs = np.tensordot(self.basis, x, axes=(0,0))
        else:
            phs = np.zeros(self.amp.shape, dtype=float)
            phs[self.amp_select] = x

        W = (2 * np.pi / self.wvl) * phs
        g = self.amp * np.exp(1j * W)
        G = ft_fwd(g)
        I = np.abs(G)**2
        E = np.sum((I[self.dh] - self.D)**2)
        self.phs = phs
        self.W = W
        self.g = g
        self.G = G
        self.I = I
        self.E = E
        return

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        self.update(x)
        Ibar = np.zeros(self.dh.shape, dtype=float)
        Ibar[self.dh] = 2*(self.I[self.dh] - self.D)
        Gbar = 2 * Ibar * self.G
        gbar = ft_rev(Gbar)
        Wbar = 2 * np.pi / self.wvl * np.imag(gbar * np.conj(self.g))
        if not self.zonal:
            abar = np.tensordot(self.basis, Wbar)

        self.Ibar = Ibar
        self.Gbar = Gbar
        self.gbar = gbar
        self.Wbar = Wbar

        if not self.zonal:
            self.abar = abar
            return self.abar
        else:
            return self.Wbar[self.amp_select]

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        return f, g


class VAPPOptimizer2:
    def __init__(self, amp, amp_dx, efl, wvl, basis, dark_hole, dh_dx, dh_target=1e-10, initial_phase=None):
        if initial_phase is None:
            phs = np.zeros(amp.shape, dtype=float)

        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.amp_dx = amp_dx
        self.efl = efl
        self.wvl = wvl
        self.basis = basis
        self.dh = dark_hole
        self.dh_dx = dh_dx
        self.D = dh_target
        self.phs = phs
        self.zonal = False

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        if not self.zonal:
            phs = np.tensordot(self.basis, x, axes=(0,0))
        else:
            phs = np.zeros(self.amp.shape, dtype=float)
            phs[self.amp_select] = x

        W = (2 * np.pi / self.wvl) * phs
        g = self.amp * np.exp(1j * W)
        # G = ft_fwd(g)
        G = focus_fixed_sampling(
            wavefunction=g,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.dh.shape,
            shift=(0, 0),
            method='mdft')
        I = np.abs(G)**2
        E = np.sum((I[self.dh] - self.D)**2)
        self.phs = phs
        self.W = W
        self.g = g
        self.G = G
        self.I = I
        self.E = E
        return

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        self.update(x)
        Ibar = np.zeros(self.dh.shape, dtype=float)
        Ibar[self.dh] = 2*(self.I[self.dh] - self.D)
        Gbar = 2 * Ibar * self.G
        # gbar = ft_rev(Gbar)
        gbar = focus_fixed_sampling_backprop(
            wavefunction=Gbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.phs.shape,
            shift=(0, 0),
            method='mdft')

        Wbar = 2 * np.pi / self.wvl * np.imag(gbar * np.conj(self.g))
        if not self.zonal:
            abar = np.tensordot(self.basis, Wbar)

        self.Ibar = Ibar
        self.Gbar = Gbar
        self.gbar = gbar
        self.Wbar = Wbar

        if not self.zonal:
            self.abar = abar
            return self.abar
        else:
            return self.Wbar[self.amp_select]

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        return f, g
    
class AAPOptimizer2:
    """An apodized pupil optimizer, mask is real-valued and gray-scale

    Note that this is *not* an APLC optimizer, it just does pupil-plane amplitude
    """
    def __init__(self, amp, amp_dx, efl, wvl, basis, dark_hole, dh_dx, dh_target=1e-10, initial_amplitude=None):
        if initial_amplitude is None:
            aap = np.zeros(amp.shape, dtype=float)

        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.amp_dx = amp_dx
        self.efl = efl
        self.wvl = wvl
        self.basis = basis
        self.dh = dark_hole
        self.dh_dx = dh_dx
        self.D = dh_target
        self.aap = aap
        self.zonal = False

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        if not self.zonal:
            aap = np.tensordot(self.basis, x, axes=(0,0))
        else:
            aap = np.zeros(self.amp.shape, dtype=float)
            aap[self.amp_select] = x

        # impose constraints
        aap = np.real(aap)
        g = self.amp * aap
        G = focus_fixed_sampling(
            wavefunction=g,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.dh.shape,
            shift=(0, 0),
            method='mdft')
        
        I = np.abs(G)**2
        E = np.sum((I[self.dh] - self.D)**2)

        self.aap = aap
        self.g = g
        self.G = G
        self.I = I
        self.E = E
        return

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        self.update(x)
        Ibar = np.zeros(self.dh.shape, dtype=float)
        Ibar[self.dh] = 2*(self.I[self.dh] - self.D)

        Gbar = 2 * Ibar * self.G
        gbar = focus_fixed_sampling_backprop(
            wavefunction=Gbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.aap.shape,
            shift=(0, 0),
            method='mdft')
        
        aapbar = np.real(gbar)

        if not self.zonal:
            abar = np.tensordot(self.basis, aapbar)

        self.Ibar = Ibar
        self.Gbar = Gbar
        self.gbar = gbar
        self.aapbar = aapbar

        if not self.zonal:
            self.abar = abar
            return self.abar
        else:
            return self.aapbar[self.amp_select]

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        return f, g
    

class APLCOptimizer2:
    """An apodized pupil coronagraph optimizer, pupil is real-valued and gray-scale,
    FPM and LS are fixed

    """
    def __init__(self, amp, amp_dx, efl, wvl, basis, dark_hole, dh_dx, fpm, ls, dh_target=1e-10, initial_amplitude=None):
        if initial_amplitude is None:
            aplc = np.zeros(amp.shape, dtype=float)

        self.amp = amp
        self.amp_select = self.amp > 1e-9
        self.amp_dx = amp_dx
        self.efl = efl
        self.wvl = wvl
        self.basis = basis
        self.dh = dark_hole
        self.dh_dx = dh_dx
        self.dh_target = dh_target
        self.aplc = aplc
        self.zonal = False
        self.fpm = fpm
        self.ls = ls
        self.cost = []

    def set_optimization_method(self, zonal=False):
        self.zonal = zonal

    def update(self, x):
        if not self.zonal:
            aplc = np.tensordot(self.basis, x, axes=(0,0))
        else:
            aplc = np.zeros(self.amp.shape, dtype=float)
            aplc[self.amp_select] = x

        # impose constraints
        aplc = np.real(aplc)
        b = self.amp * aplc

        # prop to focal plane mask
        B = focus_fixed_sampling(
            wavefunction=b,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.dh.shape,
            shift=(0, 0),
            method='mdft')
        
        # apply focal plane mask
        C = B * self.fpm

        # prop to lyot stop
        c = focus_fixed_sampling(
            wavefunction=C,
            input_dx=self.dh_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.amp_dx,
            output_samples=self.amp.shape,
            shift=(0, 0),
            method='mdft')

        # apply lyot stop
        d = c * self.ls

        # prop to image
        D = focus_fixed_sampling(
            wavefunction=d,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.dh.shape,
            shift=(0, 0),
            method='mdft')

        I = np.abs(D)**2
        E = np.sum((I[self.dh] - self.dh_target)**2)

        self.aplc = aplc
        self.I = I
        self.E = E
        self.cost.append(np.mean(I[self.dh]))

        # the fields
        self.b = b
        self.B = B
        self.c = c
        self.C = C
        self.d = d
        self.D = D

        return

    def fwd(self, x):
        self.update(x)
        return self.E

    def rev(self, x):
        self.update(x)
        Ibar = np.zeros(self.dh.shape, dtype=float)
        Ibar[self.dh] = 2*(self.I[self.dh] - self.dh_target)

        Dbar = 2 * Ibar * self.D

        # backprop from image to lyot stop
        dbar = focus_fixed_sampling_backprop(
            wavefunction=Dbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.aplc.shape,
            shift=(0, 0),
            method='mdft')
        
        # backprop lyot stop application
        cbar = self.ls.conj() * dbar

        # backprop from before stop to focal plane mask
        Cbar = focus_fixed_sampling_backprop(
            wavefunction=cbar,
            input_dx=self.dh_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.amp_dx,
            output_samples=self.amp.shape,
            shift=(0, 0),
            method='mdft')
        
        # backprop fpm application
        Bbar = self.fpm.conj() * Cbar

        # backprop from before fpm to pupil apodizer
        bbar = focus_fixed_sampling_backprop(
            wavefunction=Bbar,
            input_dx=self.amp_dx,
            prop_dist = self.efl,
            wavelength=self.wvl,
            output_dx=self.dh_dx,
            output_samples=self.aplc.shape,
            shift=(0, 0),
            method='mdft')
        
        aplcbar = np.real(bbar)

        if not self.zonal:
            abar = np.tensordot(self.basis, aplcbar)

        self.Ibar = Ibar
        self.bbar = bbar
        self.Bbar = Bbar
        self.cbar = cbar
        self.Cbar = Cbar
        self.dbar = dbar
        self.Dbar = Dbar
        self.aplcbar = aplcbar

        if not self.zonal:
            self.abar = abar
            return self.abar
        else:
            return self.aplcbar[self.amp_select]

    def fg(self, x):
        g = self.rev(x)
        f = self.E
        return f, g

