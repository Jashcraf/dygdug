def cdi_san(I0, Ip1, Ip2, Im1, Im2):
    """perform coherent differential imaging on speckle area nulling,
    using the algorithm outlined in Nishikawa 2022

    Parameters
    ----------
    I0 : ndarray
        reference image, before probe 
    Ip1 : ndarray
        probe image with positive sine probe
    Ip2 : ndarray
        probe image with positive cosine probe 
    Im1 : ndarray
        probe image with negative sine probe 
    Im2 : ndarray
        probe image with negative cosine probe
    """

    dI1 = Ip1 - Im1
    dI2 = Ip2 - Im2
    denI1 = Ip1 + Im1 - 2 * I0
    denI2 = Ip2 + Im2 - 2 * I0

    # construct coherent intensity
    Icoh_sin = (dI1 ** 2) / (8 * denI1)
    Icoh_cos = (dI2 ** 2) / (8 * denI2)
    Icoh = Icoh_sin + Icoh_cos

    return Icoh
