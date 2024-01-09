import numpy as np
import jax
import equinox as eqx
import optax
import cloudpickle
import argparse

from lbc.models import AutoEncoder
from lbc import data as datautils

from glob import glob
from tqdm.auto import tqdm
from os import makedirs


# Parse number of epochs to train
ap = argparse.ArgumentParser()
ap.add_argument('n_epochs', default=10_000, type=int)
ap.add_argument('-c', '--hidden_channels', default=4, type=int)
ap.add_argument('-l', '--latent_dim', default=256, type=int)
ap.add_argument('--save_freq', default=100, type=int)
ap.add_argument('--checkpoint_freq', default=10000, type=int)
args = ap.parse_args()

# Load data
# filenames = sorted(glob("/scratch/ch4407/lbc/58540-processed/*.npy"))
# print(len(filenames), "files found")
# images = np.stack([np.load(fname) for fname in filenames]).reshape(-1, 1, 128, 128)
# images = jax.numpy.array(images)
# print("Input data shape:", images.shape)
X_train, X_val, X_test = datautils.split(datautils.load())


# Make a loss function (just MSE between the images)
def reco_loss(model, x):
    x_hat = jax.vmap(model)(x)
    mse = ((x_hat - x) ** 2).mean()
    return mse


# Instantiate model and optimizer
key = jax.random.PRNGKey(seed=120)
model = AutoEncoder(hidden_channels=args.hidden_channels, latent_dim=args.latent_dim, key=key)
eta = 1e-4
optim = optax.adamw(eta)
opt_state = optim.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def train_step(model, opt_state, x):
    loss_value, grads = eqx.filter_value_and_grad(reco_loss)(model, x)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


history = { "epoch": [], "loss": [], "loss_val": [] }

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


# init_loss = reco_loss(model, images)
# print("Initial loss:", init_loss)

epoch_0 = history['epoch'][-1] if len(history['epoch']) > 0 else 0

# Train
n_epochs = args.n_epochs
if (epoch_0 + n_epochs) % args.save_freq == 0:
    n_epochs += 1

for i, batch in tqdm(
    zip(range(n_epochs), datautils.dataloader(X_train)),
    desc="Training", unit="step", total=n_epochs,
):
    batch = jax.numpy.array(batch)
    model, opt_state, train_loss = train_step(model, opt_state, batch)

    if i % args.save_freq == 0:
        # Compute validation loss
        loss_val = reco_loss(model, X_val)

        history['epoch'].append(epoch_0 + i)
        history['loss'].append(train_loss)
        history['loss_val'].append(loss_val)

    if i > 0 and i % args.checkpoint_freq == 0:
        save()

print("Final losses:")
print("  train:", history['loss'][-1])
print("    val:", history['loss_val'][-1])

save()

print("Done!")

