# LBC: Learning BOSS Calibration

This project is completed in fulfillment of the experimental physics
requirement for my doctoral program at NYU.

## Installation

This project uses [Poetry](https://python-poetry.org/) to manage dependences,
but newer versions of Python/Pip should be able to install just fine from a
`pyproject.toml` file.

```bash
git clone https://github.com/cmhainje/lbc.git
cd lbc

# if using poetry
poetry install

# if using pip
pip install -e .
```

Now, you should have access to `lbc` (for our models) and any dependences needed
to run any script or notebook. If using Poetry, note that you must use `poetry
run python scripts/...` to run a script or register your Poetry venv as a
Jupyter kernel with your local Jupyter installation.
