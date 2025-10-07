import numpy as np

def compute_average_ke_over_time(v):

    """
    Compute kinetic energy averaged over space for each time step.

    Args:
        v: velocity field with shape (time, 2, x, y)
    
    Returns:
        energy: kinetic energy with shape (time,)
    """

    energy = 0.5 * np.mean(v[:,0]**2 + v[:,1]**2, axis=(1,2))

    return energy

def compute_energy_spectrum(v, time=None):

    """Compute energy spectrum from velocity field.

    Args:
        v: velocity field with shape (time, 2, x, y)
        time: time index to compute spectrum
    
    Returns:
        spectrum: energy spectrum with shape (k,)
    """

    vx = v[0]
    vy = v[1]
    size = v.shape[2]

    vx_ft = np.fft.rfftn(vx, axes=(1,2))
    vy_ft = np.fft.rfftn(vy, axes=(1,2))
    E_ft = 1/2 * (vx_ft * np.conj(vx_ft) + vy_ft * np.conj(vy_ft))

    kx, ky = np.meshgrid(np.fft.fftfreq(size), np.fft.rfftfreq(size), indexing='ij')
    k = np.sqrt(kx**2 + ky**2)

    spectrum = np.zeros(size)
    for i in range(size):
        mask = (k>=i/size) & (k<(i+1)/size)
        spectrum[i] = np.sum(E_ft[time][mask])

    return spectrum