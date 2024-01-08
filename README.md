# LBC: Learning BOSS Calibration

building a generative model for BOSS spectrograph calibration data


## Authors

- **Connor Hainje** (NYU)
- **David W Hogg** (NYU) (MPIA) (Flatiron)


## Overview

Right now, this repo contains only an autoencoder which learns a small chunk
of a few arc images from a single night. 

- `notebooks/preprocess_data.ipynb` shows how arc images are processed into
  training data 
- `lbc/models.py` shows the autoencoder architecture
- `scripts/train.py` is the training script
- `notebooks/plot_training.ipynb` shows the results of training one example
  autoencoder


## Installation

This project uses [Poetry](https://python-poetry.org/) to manage dependences,
but newer versions of Python and pip should be able to install just fine from a
`pyproject.toml` file.

```bash
git clone https://github.com/cmhainje/lbc.git
cd lbc

# if using poetry
poetry install

# if using pip
pip install -e .
```

Now, you should have access to the package `lbc` (for our models) and any
dependences needed to run any script or notebook. If using Poetry, note that
you must use `poetry run python scripts/...` to run a script or register your
Poetry venv as a Jupyter kernel with your local Jupyter installation.

Also, if you want to use Jax with a GPU, re-install it after installing `lbc`.
To serve the lowest common denominator (my MacBook), Poetry is configured to
install Jax with only CPU support.

