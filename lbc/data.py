import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from os.path import join


RAW_DIR = '/scratch/ch4407/lbc/data'
DATA_DIR = '/scratch/ch4407/lbc/data-'


# *** UTILITIES FOR WORKING WITH PROCESSED DATA ***


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


def _normalize_which(which):
    which = which.lower()
    if which not in ['train', 'val', 'test']:
        raise ValueError(f"Value of `which` ({which}) unknown.")
    return which


def load(which='train', flavor='arc', camera='r', dset_name='processed'):
    """Loads cached train/val/test datasets."""
    flavor = _normalize_flavor(flavor)
    camera = _normalize_camera(camera)
    which  = _normalize_which(which)

    prefix = join(DATA_DIR + dset_name, flavor + 's', f'{which}_{camera}_')
    images = np.load(prefix + 'images.npy')
    metadata = pd.read_csv(prefix + 'metadata.csv', sep=';')
    return images, metadata


def load_images(which='train', flavor='arc', camera='r', dset_name='processed'):
    """Loads cached train/val/test datasets."""
    flavor = _normalize_flavor(flavor)
    camera = _normalize_camera(camera)
    which  = _normalize_which(which)

    prefix = join(DATA_DIR + dset_name, flavor + 's', f'{which}_{camera}_')
    images = np.load(prefix + 'images.npy')
    return images


def load_metadata(which='train', flavor='arc', camera='r', dset_name='processed'):
    """Loads cached train/val/test datasets."""
    flavor = _normalize_flavor(flavor)
    camera = _normalize_camera(camera)
    which  = _normalize_which(which)

    prefix = join(DATA_DIR + dset_name, flavor + 's', f'{which}_{camera}_')
    metadata = pd.read_csv(prefix + 'metadata.csv', sep=';')
    return metadata


def split(*args, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=0):
    """Split dataset(s) into train/val/test sets."""
    rng = np.random.default_rng(seed)

    lengths = set( len(x) for x in args )
    if len(lengths) != 1:
        raise ValueError("arrays don't have the same size")
    N = lengths.pop()

    val  = int(np.floor(val_frac * N))
    test = int(np.floor(test_frac * N))
    train = N - val - test
    div1 = train
    div2 = train + val

    idx = np.arange(N)
    rng.shuffle(idx)
    idx_tr = idx[:div1]
    idx_va = idx[div1:div2]
    idx_te = idx[div2:]

    if len(args) == 1:
        return args[0][idx_tr], args[0][idx_va], args[0][idx_te]
    else:
        return ((x[idx_tr], x[idx_va], x[idx_te]) for x in args)


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


# *** UTILITIES FOR WORKING WITH RAW DATA ***

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
    return [(q, b) for q, b in zip(quads, biases)]


def bias_subtract(d, b):
    """subtract the median bias value from the data only in rows where data exists
    (converts data from unsigned to signed ints in case data-bias is negative)"""
    row_mask = (np.count_nonzero(d, axis=1) != 0)
    bsub = np.copy(d).astype(np.int32)
    bsub[row_mask] -= np.median(b[row_mask]).astype(np.uint16)
    return bsub


def squash_old(image):
    """given a bias-subtracted image, normalize pixel values approximately into the interval (0, 1)
    (note: normalization is nonlinear, some values may be slightly negative)"""
    x = np.copy(image).astype(np.float64)
    mask = np.count_nonzero(x, axis=1) > 0
    x[mask] -= np.percentile(x[mask], 0.01)
    x[mask] /= np.percentile(x[mask], 99.9)
    x[mask] = np.tanh(2 * x[mask])
    return x


def squash(image):
    """2024-03-12"""
    x = np.copy(image).astype(np.float64)
    mask = np.count_nonzero(x, axis=1) > 0
    x[mask] -= np.median(x[mask])
    x[mask] /= np.percentile(x[mask], 99.9)
    x[mask] = np.tanh(x[mask])
    return x


def wrangle_image(filename):
    from astropy.io import fits

    # load and process the image
    image = fits.getdata(filename)

    # in the future if we want to process all the quadrants, here's what it will look like
    # quads = split_by_amplifier(image, red=('-r-' in filename))
    # quads = [bias_subtract(d, b) for d, b in quads]
    # quads = [squash(q) for q in quads]

    # for now, we only care about the bottom left quadrant

    # old
    # x = squash(
    #     bias_subtract(
    #         *split_by_amplifier(image, red=('-r-' in filename))[2]
    #     )
    # )
    # return x[800:928, 800:928]

    # updated 2024-03-12
    x = bias_subtract(
        *split_by_amplifier(image, red=('-r1-' in filename))[2]
    )[800:928, 800:928]
    return squash(x)
