import argparse
import numpy as np
import pandas as pd

from astropy.io import fits

from glob import glob
from tqdm import tqdm
from os import makedirs
from os.path import join, basename, exists


ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('dest')
args = ap.parse_args()

file_list = sorted(glob(join(args.src, '*', '*')))
print(len(file_list), 'files found.')

makedirs(args.dest, exist_ok=True)


def split_by_amplifier(image_data, red=True):
    if red:
        return (
            (
                image_data[48 : 2111 + 1, 119 : 2175 + 1],
                image_data[48 : 2111 + 1, 10 : 100 + 1],
            ),
            (
                image_data[48 : 2111 + 1, 2176 : 4232 + 1],
                image_data[48 : 2111 + 1, 4250 : 4340 + 1],
            ),
            (
                image_data[2112 : 4175 + 1, 119 : 2175 + 1],
                image_data[2112 : 4175 + 1, 10 : 100 + 1],
            ),
            (
                image_data[2112 : 4175 + 1, 2176 : 4232 + 1],
                image_data[2112 : 4175 + 1, 4250 : 4340 + 1],
            ),
        )

    else:
        return (
            (
                image_data[56 : 2111 + 1, 128 : 2175 + 1],
                image_data[56 : 2111 + 1, 10 : 67 + 1],
            ),
            (
                image_data[56 : 2111 + 1, 2176 : 4223 + 1],
                image_data[56 : 2111 + 1, 4284 : 4340 + 1],
            ),
            (
                image_data[2112 : 4167 + 1, 128 : 2175 + 1],
                image_data[2112 : 4167 + 1, 10 : 67 + 1],
            ),
            (
                image_data[2112 : 4167 + 1, 2176 : 4223 + 1],
                image_data[2112 : 4167 + 1, 4284 : 4340 + 1],
            ),
        )


def squash(X):
    median = np.median(X[X != 0])
    width = np.median(np.abs(X[X != 0] - median))

    vmin = max(median - 5 * width, 0)
    vmax = median + 5 * width

    # clamp all data to between these values
    X[X < vmin] = vmin
    X[X > vmax] = vmax

    # scale to [0, 1]
    X -= vmin
    X = X.astype(np.float64)
    X /= (vmax - vmin)
    return X



def process(filename):
    with fits.open(filename) as f:
        meta = dict(f[0].header)
        img  = f[0].data

#     splits = split_by_amplifier(img, red=meta['CAMERAS'] == 'r1')

#     quads = []
#     for data, bias in splits:
#         quad = np.copy(data).astype(np.int32)  # unsigned -> signed int

#         # look for rows that are completely blanked out
#         rows = quad.sum(1)
#         mask = rows != 0

#         # do bias subtraction only on rows with data
#         quad[mask] -= np.median(bias[mask]).astype(np.uint16)
#         quads.append(quad)

#     s_quads = [squash(q) for q in quads]
#     final = s_quads[2][800:928, 800:928]

    # no need to process the other quadrants right now
    d, b = split_by_amplifier(img, red=meta['CAMERAS'] == 'r1')[2]
    d = d.astype(np.int32)
    mask = d.sum(1) != 0
    d[mask] -= np.median(b[mask]).astype(np.uint16)
    final = squash(d)[800:928, 800:928]

    return meta, final


all_metadata = []
meta_path = join(args.dest, 'metadata.csv')

for fname in tqdm(file_list):
    meta, img = process(fname)
    all_metadata.append(meta)

    # new filename
    mjd    = meta['MJD']
    camera = meta['CAMERAS']
    frame  = f"{int(meta['EXPOSURE']):08d}"

    out_name = f"{mjd}-{camera}-{frame}.npy"
    np.save(join(args.dest, out_name), img)

df = pd.DataFrame(all_metadata)
df.to_csv(meta_path, mode='a', index=False, header=(not exists(meta_path)))
