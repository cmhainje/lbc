import numpy as np
import jax
import equinox as eqx
import optax

from lbc.models import AutoEncoder

from glob import glob
from tqdm.auto import tqdm


# Load images
filenames = sorted(glob("/Users/connor/data/lbc/58540-processed/*.npy"))
print(len(filenames), "files found")
images = np.stack([np.load(fname) for fname in filenames]).reshape(-1, 1, 128, 128)
images = jax.numpy.array(images)
print("Input data shape:", images.shape)

# Create an AutoEncoder
key = jax.random.PRNGKey(seed=120)
model = AutoEncoder(hidden_channels=2, latent_dim=64, key=key)

# Try running the images through the model
output = jax.vmap(model)(images)
print("Output data shape:", output.shape)


# Make a loss function (just MSE between the images)
def reco_loss(model, x):
    x_hat = jax.vmap(model)(x)
    mse = ((x_hat - x) ** 2).mean()
    return mse


# Instantiate an optimizer
eta = 1e-4
optim = optax.adamw(eta)
opt_state = optim.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def train_step(model, opt_state, x):
    loss_value, grads = eqx.filter_value_and_grad(reco_loss)(model, x)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


# Try a single train step
model, opt_state, train_loss = train_step(model, opt_state, images)
print("Loss after one step:", train_loss)


# Try a few train steps
for i in tqdm(range(10_000), desc="Training", unit="step"):
    model, opt_state, train_loss = train_step(model, opt_state, images)

print("Loss after more steps:", train_loss)
