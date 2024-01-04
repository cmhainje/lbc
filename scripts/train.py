import numpy as np
import jax
import equinox as eqx
import optax
import cloudpickle
import argparse

from lbc.models import AutoEncoder

from glob import glob
from tqdm.auto import tqdm
from os import makedirs


# Parse number of epochs to train
ap = argparse.ArgumentParser()
ap.add_argument('n_epochs', default=10_000, type=int)
ap.add_argument('-c', '--hidden_channels', default=4, type=int)
ap.add_argument('-l', '--latent_dim', default=256, type=int)
ap.add_argument('--save_freq', default=100, type=int)
args = ap.parse_args()

# Load images
filenames = sorted(glob("/scratch/ch4407/lbc/58540-processed/*.npy"))
print(len(filenames), "files found")
images = np.stack([np.load(fname) for fname in filenames]).reshape(-1, 1, 128, 128)
images = jax.numpy.array(images)
print("Input data shape:", images.shape)

# Create an AutoEncoder
key = jax.random.PRNGKey(seed=120)
model = AutoEncoder(hidden_channels=args.hidden_channels, latent_dim=args.latent_dim, key=key)

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


history = { "epoch": [], "loss": [] }

CHECKPOINT_DIR = f'/scratch/ch4407/lbc/checkpoints/h{args.hidden_channels}_l{args.latent_dim:03d}'
makedirs(CHECKPOINT_DIR, exist_ok=True)

def save():
    path = f'{CHECKPOINT_DIR}/checkpoint_{history["epoch"][-1]}.pkl'
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

epoch_0 = history['epoch'][-1] if len(history['epoch']) > 0 else 0

# Train
n_epochs = args.n_epochs
if (epoch_0 + n_epochs) % args.save_freq == 0:
    n_epochs += 1

for i in tqdm(range(n_epochs), desc="Training", unit="step"):
    model, opt_state, train_loss = train_step(model, opt_state, images)

    if i % args.save_freq == 0:
        history['epoch'].append(epoch_0 + i)
        history['loss'].append(train_loss)

print("Final loss:", history['loss'][-1])

save()

print("Done!")

