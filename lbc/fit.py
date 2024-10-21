import jax
import jax.numpy as jnp
import optax

from scipy.spatial import KDTree

from pathlib import Path
from os.path import join
from tqdm.auto import tqdm

from .peaks import detect_peaks


ROOT_DIR = Path(__file__).parent.parent

L = 4096
WN_MAX = 3

def get_control_points():
    filename = join(ROOT_DIR, "notebooks", "fiducials.npz")
    with jnp.load(filename) as fid:
        XI = fid['xi']
        ETA = fid['eta']

    xi, eta = jnp.meshgrid(XI, ETA)
    xi = xi.flatten()
    eta = eta.flatten()
    return xi, eta

def wavenumber(j):
    return ((j + 1) // 2) * jnp.pi / L

def basis_1d(j, x):
    wn = wavenumber(j)
    phase = (-jnp.pi / 2) * (j % 2)
    return jnp.cos(wn * x + phase)

def basis_2d(jx, jy, x, y):
    return basis_1d(jx, x) * basis_1d(jy, y)

def basis_indices(wn_max=WN_MAX):
    j_max = 2 * wn_max + 1
    idx = [(jx, jy) for jx in range(j_max) for jy in range(j_max)]
    return jnp.array(idx)

v_basis_2d = jax.vmap(basis_2d, in_axes=(0, 0, None, None), out_axes=0)

XI, ETA = get_control_points()
INDICES = basis_indices()
BASIS_EVAL = v_basis_2d(INDICES[:, 0], INDICES[:, 1], XI, ETA)

def distort_factory(basis_eval, xi, eta):
    @jax.jit
    def distort(a, b):
        dxi  = (a[:, None] * basis_eval).sum(0)
        deta = (b[:, None] * basis_eval).sum(0)
        return jnp.stack((xi + dxi, eta + deta), -1)
    return distort

distort = distort_factory(BASIS_EVAL, XI, ETA)

def find_neighbors(distort_fn, peaks, theta_init=jnp.zeros((2, len(INDICES)))):
    undistorted = distort_fn(*theta_init)
    tree = KDTree(undistorted)
    dists, idx = tree.query(peaks, k=200)
    return tree, idx

# LOSS
def deltas_factory(peaks, idx):
    @jax.jit
    def deltas(d):
        return jnp.square(peaks[:, None, :] - d[idx]).sum(-1).min(-1)
    return deltas

def capped_mse_factory(deltas, peaks, idx, sigma=1, Q=10):
    @jax.jit
    def capped_mse(distorted):
        ds = deltas(distorted)
        ell = 1. / (sigma**2/ds + 1./Q)
        return ell.mean()
    return capped_mse

@jax.jit
def l2_regularization(params):
    return jnp.square(params).mean()

def forward_factory(distort, capped_mse, alpha=1e-2):
    @jax.jit
    def forward(theta):
        return capped_mse(distort(*theta)) + alpha * l2_regularization(theta)
    return forward

def fit_image(image, peaks=None, alpha=1e-2, sigma=5, Q=10, n_epochs=100, quiet=False):
    if peaks is None:
        peaks = detect_peaks(image)
        peaks = peaks[:, [1, 0]]

    tree, idx = find_neighbors(distort, peaks)
    deltas = deltas_factory(peaks, idx)
    capped_mse = capped_mse_factory(deltas, peaks, idx, sigma=sigma, Q=Q)
    forward = forward_factory(distort, capped_mse, alpha=alpha)
    forward_grad = jax.value_and_grad(forward)

    def lr_schedule(epoch: int):
        lr_0, lr_1 = 5e-1, 5e-2
        ep_0, ep_1 = 25, 100

        if epoch < ep_0:
            return lr_0
        if epoch < ep_1:
            t = (epoch - ep_0) / (ep_1 - ep_0)
            return lr_0 * (1 - t) + lr_1 * t
        return lr_1

    theta = jnp.zeros((2, len(INDICES)))
    opt = optax.adam(learning_rate=lr_schedule)
    opt_state = opt.init(theta)

    losses = []

    if quiet:
        _iter = range(n_epochs)
    else:
        _iter = tqdm(range(n_epochs), desc='fitting', unit='epoch')

    for _ in _iter:
        loss_val, grads = forward_grad(theta)

        if jnp.isnan(loss_val).any() or jnp.isnan(grads).any():
            raise RuntimeError("NaN encountered, stopping early")

        updates, opt_state = opt.update(grads, opt_state)
        theta = optax.apply_updates(theta, updates)

        losses.append(loss_val)

    return theta, losses
