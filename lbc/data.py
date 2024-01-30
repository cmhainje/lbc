import numpy as np
from glob import glob
from tqdm import tqdm


DATA_DIR = '/scratch/ch4407/lbc/data-processed'


def _normalize_flavor(flavor):
    flavor = flavor.lower()
    if flavor not in ['arc', 'flat']:
        raise ValueError(f"Value of `flavor` ({flavor}) unknown.")
    return flavor


def _normalize_camera(camera):
    camera = camera.lower()[0]
    if camera not in ['r', 'b']:
        raise ValueError(f"Value of `camera` ({camera}) unknown.")
    return camera


def load_images(num: int | None=None, flavor='arc', camera='r', seed=0):
    """Load `num` images. If `num` is None, loads them all."""
    rng = np.random.default_rng(seed)

    flavor = _normalize_flavor(flavor)
    camera = _normalize_camera(camera)

    file_list = np.array(sorted(glob(f'{DATA_DIR}/{flavor}s/*-{camera}1-*.npy')))
    rng.shuffle(file_list)

    if num is not None:
       if num < len(file_list):
           file_list = file_list[:num]
       else:
           print('Warning: `num` is larger than the number of files available. Returning all.')

    # Now that we've specified which files we want to load, let's load them
    images = np.stack([
        np.load(f) for f in tqdm(file_list, desc="Loading images")
    ]).reshape(-1, 1, 128, 128)

    return images


def load(flavor='arc', camera='r'):
    """Loads cached train/val/test datasets."""
    flavor = _normalize_flavor(flavor)
    camera = _normalize_camera(camera)
    filename = f"{DATA_DIR}/train-{flavor}-{camera}.npz"

    try:
        with np.load(filename) as data:
            return data['train'], data['val'], data['test']
    except OSError:
        print(f"Couldn't find cached dataset at {filename}. Generating...")
        train, val, test = split(load_images(flavor=flavor, camera=camera))
        np.savez(filename, train=train, val=val, test=test)
        print(f"Saved dataset at {filename}!")
        return train, val, test


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


def split_by_amplifier(X, red=True):
    """from https://data.sdss.org/datamodel/files/BOSS_SPECTRO_DATA/MJD/sdR.html
    note that indices therein are transposed and endpoint-inclusive"""

    if red:
        rows = [ ( 48, 2111 + 1), (2112, 4175 + 1) ]
        cols = [ (119, 2175 + 1), (2176, 4232 + 1) ]
        bias = [ ( 10,  100 + 1), (4250, 4340 + 1) ]
    else:
        rows = [ ( 56, 2111 + 1), (2112, 4167 + 1) ]
        cols = [ (128, 2175 + 1), (2176, 4223 + 1) ]
        bias = [ ( 10,   67 + 1), (4284, 4340 + 1) ]

    quads  = [X[r_lo:r_hi, c_lo:c_hi] for (c_lo, c_hi) in cols for (r_lo, r_hi) in rows]
    biases = [X[r_lo:r_hi, b_lo:b_hi] for (b_lo, b_hi) in bias for (r_lo, r_hi) in rows]
    return ((q, b) for q, b in zip(quads, biases))
