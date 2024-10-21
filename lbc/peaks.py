import jax
import jax.numpy as jnp
import jax.scipy as jsp

def make_gaussian_filter(sigma=1.0, size=None, normalize_sum=False):
    """
    Parameters:
        sigma (float): variance of the Gaussian
        size (int | None): must be odd, defaults to None (estimated from sigma)
    """

    if size is not None and size % 2 != 1:
        raise ValueError("size must be odd")

    if size is None:
        size = 2 * int(jnp.ceil(2 * sigma)) + 1

    middle = size // 2

    a = jnp.arange(size)
    dsq = (a[:, None] - middle)**2 + (a[None, :] - middle)**2
    G = jnp.exp(-dsq / (2 * sigma**2))

    if normalize_sum:
        G /= G.sum()
    else:
        G /= (2 * jnp.pi * sigma**2)

    return G

def gauss_smooth(image, sigma=1, size=3):
    """
    Parameters:
        image (Array): image
        sigma (float): width of the Gaussian kernel, defaults to 1.0
        size (int | None): size of the kernel, defaults to 3.
            If None, the necessary size is estimated from the given sigma.
    """
    G = make_gaussian_filter(sigma=sigma, size=size)
    return jsp.signal.convolve2d(image, G, mode='same')

def estimate_noise(image, sp=5, gridsize=20):
    """
    Parameters:
        image (Array): image
        sp (int): spacing between points in each pair
        gridsize (int): size of grid for estimates
    """
    # JAXifying `dsigma`
    nr, nc = image.shape
    dr = max(min(gridsize, nr // 4), 1)
    dc = max(min(gridsize, nc // 4), 1)

    diff = jnp.abs(image[:-sp:dr,:-sp:dc] - image[sp::dr,sp::dc]).flatten()
    diff = jnp.sort(diff)
    ndiff = len(diff)

    s, Nsigma = 0.0, 0.7
    ROOT_2 = jnp.sqrt(2)
    while s == 0.0:
        k = int(jnp.floor(ndiff * jax.scipy.special.erf(Nsigma / ROOT_2)))

        if k >= ndiff:
            print('warning: bad')
            s = 1.0
            break

        s = diff[k] / (Nsigma * ROOT_2)
        Nsigma += 0.1

    return s

def identify_signifcant_regions(image, sigma=None, plim=6.0):
    """
    Parameters:
        image (Array): image
        sigma (float | None): noise level, defaults to None.
            If None, calls `estimate_noise` with default parameters.
        plim (float): number of sigma to count as significant
    """
    if sigma is None:
        sigma = estimate_noise(image)

    limit = (sigma / (2.0 * jnp.sqrt(jnp.pi) * 1.0)) * plim
    return image >= limit

def identify_local_maxima(image, size=3):
    """
    Parameters:
        image (Array): image
        size (int): size of neighborhood to check for maximum. Must be odd.
    """
    if size % 2 != 1:
        raise ValueError('size must be positive and odd')

    half = size // 2

    # pad image by half size on all sides
    x = jnp.pad(image, half, mode='minimum')

    def is_maximum(i, j):
        center = jax.lax.dynamic_slice(x, [i + half, j + half], [1, 1]).squeeze()
        neighborhood = jax.lax.dynamic_slice(x, [i, j], [size, size])
        return center == neighborhood.max()

    v_is_maximum = jax.vmap(is_maximum, in_axes=(None, 0), out_axes=0)
    vv_is_maximum = jax.vmap(v_is_maximum, in_axes=(0, None), out_axes=0)

    nr, nc = image.shape
    return vv_is_maximum(jnp.arange(nr), jnp.arange(nc))

_d = jnp.array([-1, 0, +1])
_rs = jnp.tile(_d[:, None], (1, 3)).flatten()
_cs = jnp.tile(_d[None, :], (3, 1)).flatten()
A = jnp.stack([jnp.ones_like(_rs), _rs, _cs, _rs ** 2, _rs * _cs, _cs ** 2], axis=1)

def parabola_trick(image, peak_mask, n_sigma=1e-4):
    """
    Parameters:
        image (Array): image
        peak_mask (Array<bool>): array of same shape as image. True for local maxima, False otherwise.
    """
    pad_image = jnp.pad(image, 1)

    def single_peak(r, c):
        def _lstsq(a, b):
            return jnp.linalg.lstsq(a, b, rcond=None)[0].squeeze()

        z = jax.lax.dynamic_slice(pad_image, [r, c], [3, 3]).reshape((-1, 1))
        reg = jnp.var(z) * jnp.eye(2)

        a, b, c, d, e, f = _lstsq(A.T @ A, A.T @ z)
        D = jnp.array([[2 * d, e], [e, 2 * f]]) + n_sigma * reg
        c_c, r_c = _lstsq(D, jnp.array([[-b, -c]]).T)
        return jnp.array([r_c, c_c])

    v_peak = jax.vmap(single_peak, in_axes=(0, 0), out_axes=0)
    peaks_0 = jnp.argwhere(peak_mask)
    rows, cols = peaks_0.T
    dpeaks = v_peak(rows, cols)
    peaks = peaks_0 + dpeaks

    # ensure finite, within 3x3 and within image
    finite = jnp.isfinite(peaks).all(axis=1)
    within_3x3 = (jnp.abs(dpeaks) <= 1).all(axis=1)
    within_img = (
        (peaks_0[:, 0] > 0) &
        (peaks_0[:, 0] < image.shape[0] - 1) &
        (peaks_0[:, 1] > 0) &
        (peaks_0[:, 1] < image.shape[1] - 1)
    )
    mask = finite & within_3x3 & within_img

    return peaks[mask]

def detect_peaks(image, plim=6.0, thresh=8):
    """
    Parameters:
        image (Array): image
        plim (float): number of sigma to count as significant, defaults to 6
        thresh (float): minimum allowed peak value
    """
    sigma = estimate_noise(image)
    smooth = gauss_smooth(image)
    return parabola_trick(smooth, (
        (smooth > thresh) &
        identify_local_maxima(smooth) &
        identify_signifcant_regions(smooth, sigma=sigma, plim=plim)
    ), n_sigma=1e-5)
