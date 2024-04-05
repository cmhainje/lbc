import jax
from jax import numpy as jnp
from jax.scipy.signal import convolve

def loss(model, x):
    x_hat = jax.vmap(model)(x)

    # compute mask and grow by 2
    mask = jnp.count_nonzero(x, axis=-1, keepdims=True) != 0
    filter = jnp.ones((5, 1), dtype=bool)
    mask = convolve(mask, filter, mode='same').astype(bool)

    mse = ((x_hat - x) ** 2).mean(where=mask)
    return mse
