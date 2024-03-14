import jax

def loss(model, x):
    x_hat = jax.vmap(model)(x)
    mask = jax.numpy.count_nonzero(x, axis=-1, keepdims=True) != 0
    mse = ((x_hat - x) ** 2).mean(where=mask)
    return mse
