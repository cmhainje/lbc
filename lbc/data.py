import numpy as np
from glob import glob
from tqdm import tqdm


DATA_DIR = '/scratch/ch4407/lbc/data-processed'


def load(num=None, which='arc', seed=0):
    """Load `num` images. If `num` is None, loads them all."""
    rng = np.random.default_rng(seed)

    if 'arc' in which.lower():
        file_list = sorted(glob(f'{DATA_DIR}/arcs/*.npy'))
    elif 'flat' in which.lower():
        file_list = sorted(glob(f'{DATA_DIR}/flats/*.npy'))
    else:
        raise ValueError(f"Value of `which` ({which}) unknown.")
    
    file_list = np.array(file_list)
    rng.shuffle(file_list)

    if num is not None and num < len(file_list):
        file_list = file_list[:num]

    # Now that we've specified which files we want to load, let's load them
    images = np.stack([
        np.load(f) for f in tqdm(file_list, desc="Loading images")
    ]).reshape(-1, 1, 128, 128)

    return images


def split(*args, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=0):
    """Split dataset(s) into train/val/test sets."""
    rng = np.random.default_rng(seed)

    lengths = set( x.shape[0] for x in args )
    if len(lengths) != 1:
        raise ValueError("arrays don't have the same size")
    N = lengths.pop()

    val  = int(np.floor(val_frac * N))
    test = int(np.floor(test_frac * N))
    train = N - val - test
    div1 = train
    div2 = train + val

    if len(args) == 1:
        return args[0][:div1], args[0][div1:div2], args[0][div2:]
    else:
        return ((x[:div1], x[div1:div2], x[div2:]) for x in args)


def dataloader(*args, batch_size=64, seed=0):
    """Load batches from dataset(s)
    following https://docs.kidger.site/equinox/examples/score_based_diffusion
    translated into numpy and made multi-dataset agnostic"""
    rng = np.random.default_rng(seed)

    lengths = set( x.shape[0] for x in args )
    if len(lengths) != 1:
        raise ValueError("arrays don't have the same size")
    N = lengths.pop()

    indices = np.arange(N)
    while True:
        perm = rng.permutation(indices)
        start = 0
        end = batch_size
        while end < N:
            batch_perm = perm[start:end]
            if len(args) == 1:
                yield args[0][batch_perm]
            else:
                yield (x[batch_perm] for x in args)
            start = end
            end = start + batch_size

