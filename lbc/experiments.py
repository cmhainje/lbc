import cloudpickle
import json

from datetime import datetime
from uuid import UUID, uuid4
from os import makedirs
from functools import reduce
from collections import defaultdict


EXPERIMENT_DIR = '/scratch/ch4407/lbc/experiments'
STORE_PATH = f'{EXPERIMENT_DIR}/store.json'


# *** Global experiment store ***
def _load_store():
    try:
        with open(STORE_PATH, 'r') as f:
            return json.load(f)
    except OSError:
        return {}


def _save_store(store):
    with open(STORE_PATH, 'w') as f:
        json.dump(store, f)


def _update_store(experiment):
    store = _load_store()
    store[experiment.id.hex] = {
        "name": experiment.name,
        "date": experiment.date,
        "architecture": experiment.architecture,
        "hyperparams": experiment.hyperparams,
        "metadata": experiment.metadata,
        "checkpoints": experiment.checkpoints,
    }
    _save_store(store)


def query(
    model=None, optim=None, loss=None,
    model_hyperparams={},
    optim_hyperparams={},
    loss_hyperparams={},
    metadata={}
):
    def _match(exp):
        return reduce(lambda a, b: a and b, (
            model is None or model == exp["architecture"]["model"],
            optim is None or optim == exp["architecture"]["optim"],
            loss is None or loss == exp["architecture"]["loss"],
            model_hyperparams.items() <= exp["hyperparams"]["model"].items(),
            optim_hyperparams.items() <= exp["hyperparams"]["optim"].items(),
            loss_hyperparams.items() <= exp["hyperparams"]["loss"].items(),
            metadata.items() <= exp["metadata"].items(),
        ))

    return { id: exp for id, exp in _load_store().items() if _match(exp) }


class Experiment():
    def __init__(self,
        modeldef, optimdef, lossdef,
        model_hyperparams={}, optim_hyperparams={}, loss_hyperparams={},
        name="", load_existing_if_match=True, **kwargs
    ):
        self.id = uuid4()
        self.name = name if name != "" else self.id.hex
        self.date = datetime.now().isoformat()
        self.metadata = kwargs

        self.path = f'{EXPERIMENT_DIR}/{self.id.hex}'
        makedirs(self.path)
        makedirs(f"{self.path}/checkpoints")

        self.architecture = {
            "model": modeldef.__module__ + '.' + modeldef.__name__,
            "optim": optimdef.__name__,
            "loss":  lossdef.__module__ + '.' + lossdef.__name__,
        }

        self.hyperparams = {
            "model": model_hyperparams,
            "optim": optim_hyperparams,
            "loss":  loss_hyperparams,
        }

        self.history = defaultdict(list)

        self.checkpoints = []

        # check if there is an existing experiment with the same setup
        if load_existing_if_match:
            for id, metadata in _load_store().items():
                if (
                    metadata["architecture"] == self.architecture
                    and metadata["hyperparams"] == self.hyperparams
                    and metadata["metadata"] == self.metadata
                ):
                    print("Matching experiment found, loading...")
                    self = Experiment.load(UUID(id))
                    return

        # serialize the experiment and write metadata to global store
        self.serialize()
        _update_store(self)

    @staticmethod
    def load(id: UUID | str):
        if not isinstance(id, UUID):
            id = UUID(id)
        with open(f'{EXPERIMENT_DIR}/{id.hex}/experiment.pkl', 'rb') as f:
            return cloudpickle.load(f)

    def serialize(self):
        with open(f'{self.path}/experiment.pkl', 'wb') as f:
            cloudpickle.dump(self, f)

    def num_steps(self):
        return (
            self.history['epoch'][-1]
            if len(self.history['epoch']) > 0
            else 0
        )

    def record_metrics(self, epoch: int, metrics):
        self.history['epoch'].append(epoch)
        for k, v in metrics.items():
            self.history[k].append(v)
        self.serialize()

    def save_checkpoint(self, train_state):
        step = self.num_steps()
        with open(f'{self.path}/checkpoints/{step}.pkl', 'wb') as f:
            cloudpickle.dump(train_state, f)
        self.checkpoints.append(step)
        _update_store(self)  # since we track checkpoints in the store

    def load_checkpoint(self, step: int = -1):
        if len(self.checkpoints) == 0:
            raise ValueError("No checkpoints to load")
        if step < 0:
            step = self.checkpoints[-1]
        with open(f'{self.path}/checkpoints/{step}.pkl', 'rb') as f:
            train_state = cloudpickle.load(f)
        return train_state
