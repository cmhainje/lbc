import jax

def invert(x):
    return 0.5 * jax.numpy.arctanh(x)


# MSE between the images _without_ tanh to squash large values near 1
def loss(model, x):
    x_hat = jax.vmap(model)(x)
    mse = ((invert(x_hat) - invert(x)) ** 2).mean()
    return mse
