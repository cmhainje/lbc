import argparse
import numpy as np
import pandas as pd

from astropy.io import fits

from glob import glob
from tqdm import tqdm
from os import makedirs
from os.path import join

from lbc import data as dutil


ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('dest')
args = ap.parse_args()

print(f"LBC Data Preprocessing")
print(f"Arguments read:")
print(f"  src:    {args.src}")
print(f"  dest:   {args.dest}")
print(f"")

makedirs(args.dest, exist_ok=True)

# Make train/val/test file lists
files_r = sorted(glob(join(args.src, '*', f'*-r1-*')))
files_b = sorted(glob(join(args.src, '*', f'*-b1-*')))
print(len(files_r), 'red files found.')
print(len(files_b), 'blue files found.')
assert len(files_r) == len(files_b)

idx_split = dutil.split(np.arange(len(files_r)), seed=120)


def process_file_list(files, idx, outname='train'):
    N = len(idx)
    images = np.zeros((N, 1, 128, 128))
    metadata = [None] * N
    for i, index in enumerate(tqdm(idx)):
        fname = files[index]
        metadata[i] = dict(fits.getheader(fname))
        images[i,0] = dutil.wrangle_image(fname)

    np.save(
        join(args.dest, f'{outname}_images.npy'),
        images
    )
    pd.DataFrame(metadata).to_csv(
        join(args.dest, f'{outname}_metadata.csv'),
        sep=';',
        index=False,
    )


for idx, label in zip(idx_split, ['train', 'val', 'test']):
    print(f'Processing {label} set ({len(idx)} frames):')
    process_file_list(files_r, idx, f'{label}_r')
    process_file_list(files_b, idx, f'{label}_b')

print("Done!")
