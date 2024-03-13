import numpy as np
import jax
import equinox as eqx
import optax
import cloudpickle
import argparse

from lbc.models.base_tanh import AutoEncoder
from lbc.losses.mse import loss
from lbc.data import load_images, dataloader
from lbc.experiments import Experiment

from glob import glob
from tqdm.auto import tqdm
from os import makedirs


# Parse some training parameters
ap = argparse.ArgumentParser()
ap.add_argument('n_epochs', default=100_000, type=int)
ap.add_argument('-c', '--hidden_channels', default=4, type=int)
ap.add_argument('-l', '--latent_dim', default=256, type=int)
ap.add_argument('--camera', default='r', type=str)
ap.add_argument('--save_freq', default=100, type=int)
ap.add_argument('--checkpoint_freq', default=10_000, type=int)
args = ap.parse_args()

# Load data
X_train = load_images(which='train', camera=args.camera, dset_name='prep0312')
X_val   = load_images(which='val',   camera=args.camera, dset_name='prep0312')


# Instantiate model and optimizer
seed = 1126 if args.camera == 'r' else 120
model_hyper = dict(
    hidden_channels=args.hidden_channels,
    latent_dim=args.latent_dim,
)
optim_hyper = dict(
    learning_rate=1e-4,
    weight_decay=0.005,
)
loss_hyper = dict()

key = jax.random.PRNGKey(seed=seed)
model = AutoEncoder(key=key, **model_hyper)
optim = optax.adamw(**optim_hyper)
opt_state = optim.init(eqx.filter(model, eqx.is_array))

# Set up experiment
exp = Experiment(
    model.__class__, optax.adamw, loss,
    model_hyperparams=model_hyper,
    optim_hyperparams=optim_hyper,
    loss_hyperparams=loss_hyper,
    camera=args.camera,
    dset_name='prep0312',
    seed=seed,
    load_existing_if_match=True
)
# if experiment already existed, load its most recent checkpoint
if len(exp.checkpoints) > 0:
    model, optim, opt_state = exp.load_checkpoint()
    epoch_0 = exp.num_steps()
else:
    epoch_0 = 0


@eqx.filter_jit
def train_step(model, opt_state, x):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model, x)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


# Train
n_epochs = args.n_epochs
for i, batch in tqdm(
    zip(range(n_epochs), dataloader(X_train)),
    desc="Training", unit="step", total=n_epochs,
):
    batch = jax.numpy.array(batch)
    model, opt_state, train_loss = train_step(model, opt_state, batch)

    if np.isnan(train_loss):
        raise RuntimeError("NaN encountered")

    step_num = i + 1
    if step_num % args.save_freq == 0 or step_num == n_epochs:
        # Compute validation loss
        loss_val = loss(model, X_val)
        exp.record_metrics(epoch_0 + step_num, {
            'loss': train_loss,
            'loss_val': loss_val,
        })

    if step_num % args.checkpoint_freq == 0 or step_num == n_epochs:
        exp.save_checkpoint((model, optim, opt_state))


print("Final losses:")
print("  train:", exp.history['loss'][-1])
print("    val:", exp.history['loss_val'][-1])

print("Done!")
