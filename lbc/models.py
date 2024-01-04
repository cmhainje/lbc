import jax
import equinox as eqx

from equinox.nn import Conv2d, ConvTranspose2d, Linear
from jax.random import PRNGKeyArray

from typing import Optional


# initial model inspired by
# https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html


class Encoder(eqx.Module):
    layers: list

    def __init__(
        self, hidden_channels=4, latent_dim=32, key: Optional[PRNGKeyArray] = None
    ):
        if key is None:
            raise ValueError("key cant actually be None")
        keys = jax.random.split(key, 6)
        chan = hidden_channels
        conv_kw = dict(kernel_size=3, stride=2, padding=1)
        self.layers = [
            Conv2d(1, chan, key=keys[0], **conv_kw),  # 128x128 -> 64x64
            jax.nn.relu,
            Conv2d(chan, chan, key=keys[1], **conv_kw),  # 64x64 -> 32x32
            jax.nn.relu,
            Conv2d(chan, 2 * chan, key=keys[2], **conv_kw),  # 32x32 -> 16x16
            jax.nn.relu,
            Conv2d(2 * chan, 2 * chan, key=keys[3], **conv_kw),  # 16x16 -> 8x8
            jax.nn.relu,
            Conv2d(2 * chan, 2 * chan, key=keys[4], **conv_kw),  # 8x8 -> 4x4
            jax.nn.relu,
            jax.numpy.ravel,
            Linear(2 * 16 * chan, latent_dim, key=keys[5]),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(eqx.Module):
    linear: list
    layers: list

    def __init__(
        self, hidden_channels=4, latent_dim=32, key: Optional[PRNGKeyArray] = None
    ):
        if key is None:
            raise ValueError("key cant actually be None")
        keys = jax.random.split(key, 10)
        chan = hidden_channels
        conv_kw = dict(kernel_size=3, stride=2, padding=1, output_padding=1)
        self.linear = [
            Linear(latent_dim, 16 * 2 * chan, key=keys[0]),
            jax.nn.relu,
        ]
        self.layers = [
            ConvTranspose2d(2 * chan, 2 * chan, key=keys[1], **conv_kw),  # 4x4 -> 8x8
            jax.nn.relu,
            Conv2d(2 * chan, 2 * chan, kernel_size=3, padding=1, key=keys[2]),
            jax.nn.relu,
            ConvTranspose2d(2 * chan, 2 * chan, key=keys[3], **conv_kw),  # 8x8 -> 16x16
            jax.nn.relu,
            Conv2d(2 * chan, 2 * chan, kernel_size=3, padding=1, key=keys[4]),
            jax.nn.relu,
            ConvTranspose2d(2 * chan, chan, key=keys[5], **conv_kw),  # 16x16 -> 32x32
            jax.nn.relu,
            Conv2d(chan, chan, kernel_size=3, padding=1, key=keys[6]),
            jax.nn.relu,
            ConvTranspose2d(chan, chan, key=keys[7], **conv_kw),  # 32x32 -> 64x64
            jax.nn.relu,
            Conv2d(chan, chan, kernel_size=3, padding=1, key=keys[8]),
            jax.nn.relu,
            ConvTranspose2d(chan, 1, key=keys[9], **conv_kw),  # 64x64 -> 128x128
            jax.nn.sigmoid,  # ensure image data between 0 and 1
        ]

    def __call__(self, x):
        for layer in self.linear:
            x = layer(x)
        x = x.reshape(-1, 4, 4)
        for layer in self.layers:
            x = layer(x)
        return x * 5_000


class AutoEncoder(eqx.Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(
        self, hidden_channels=4, latent_dim=32, key: Optional[PRNGKeyArray] = None
    ):
        if key is None:
            raise ValueError("key cant actually be None")
        keys = jax.random.split(key, 2)

        self.encoder = Encoder(hidden_channels, latent_dim, key=keys[0])
        self.decoder = Decoder(hidden_channels, latent_dim, key=keys[1])

    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
