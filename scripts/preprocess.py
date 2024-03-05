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
ap.add_argument('-c', '--camera', default='r', choices=['r', 'b'])
args = ap.parse_args()

print(f"LBC Data Preprocessing")
print(f"Arguments read:")
print(f"  src:    {args.src}")
print(f"  dest:   {args.dest}")
print(f"  camera: {args.camera}")
print(f"")

makedirs(args.dest, exist_ok=True)

# Make train/val/test file lists
file_list = sorted(glob(join(args.src, '*', f'*-{args.camera}1-*')))
print(len(file_list), 'files found.')
files_split = dutil.split(np.array(file_list), seed=120)


def process_file_list(files, outname='train'):
    N = len(files)
    images = np.zeros((N, 1, 128, 128))
    metadata = [None] * N
    for i, fname in enumerate(tqdm(files)):
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


for files, label in zip(files_split, ['train', 'val', 'test']):
    print(f'Processing {label} set ({len(files)} images):')
    process_file_list(files, f'{label}_{args.camera}')

print("Done!")

