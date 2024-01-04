import numpy as np
import jax
import equinox as eqx
import optax
import cloudpickle
import argparse

from lbc.models import AutoEncoder

from glob import glob
from tqdm.auto import tqdm


# Parse number of epochs to train
ap = argparse.ArgumentParser()
ap.add_argument('n_epochs', default=10_000, type=int)
args = ap.parse_args()

# Load images
filenames = sorted(glob("/scratch/ch4407/lbc/58540-processed/*.npy"))
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


history = { "loss": [] }

CHECKPOINT_DIR = '/scratch/ch4407/lbc/checkpoints'


def save():
    path = f'{CHECKPOINT_DIR}/checkpoint_{len(history["loss"])}.pkl'
    print(f'Saving checkpoint at {path}')
    train_state = ( model, optim, opt_state, history )
    with open(path, 'wb') as f:
        cloudpickle.dump(train_state, f)


def load(path=None):
    if path is None:
        ckpts = glob(f'{CHECKPOINT_DIR}/checkpoint_*.pkl')
        if len(ckpts) > 0:
            n_epochs = [ int(c.split('_')[1].split('.')[0]) for c in ckpts ]
            most_recent = np.argmax(n_epochs)
            path = ckpts[most_recent]

    if path is None:
        path = ''

    print(f'Loading checkpoint at {path}')
    with open(path, 'rb') as f:
        train_state = cloudpickle.load(f)
    ( model, optim, opt_state, history ) = train_state
    return model, optim, opt_state, history


try:
    model, optim, opt_state, history = load()
except OSError:
    print("Failed to restore from a recent checkpoint: starting training from scratch.")
    pass


init_loss = reco_loss(model, images)
print("Initial loss:", init_loss)

# Train
for i in tqdm(range(args.n_epochs), desc="Training", unit="step"):
    model, opt_state, train_loss = train_step(model, opt_state, images)
    history['loss'].append(train_loss)

print("Final loss:", history['loss'][-1])
save()

