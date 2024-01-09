"""
download_data.py
Connor Hainje (connor.hainje@nyu.edu)

Given a JSON file with MJDs and frame numbers (made with `scrape_logs.py`),
downloads the relevant datafiles from SDSS-V. Note that this will download
proprietary data, so you must be an SDSS-V collaborator.
"""

import argparse
import json
import subprocess

from os import makedirs
from os.path import join


BASE_URL = 'rsync://sdss5@dtn.sdss.org/sdsswork/data/boss/spectro/apo'


ap = argparse.ArgumentParser()
ap.add_argument('infile', default='2023_12.json')
ap.add_argument('-a', '--arcs', action='store_true')
ap.add_argument('-f', '--flats', action='store_true')
ap.add_argument('-t', '--tempfile', default='_file_list.txt')
ap.add_argument('-d', '--dest', default='2023_12')
ap.add_argument('--test', action='store_true')
args = ap.parse_args()


def make_filenames(d):
    # expects a dict with shape: mjd -> list(frame nums)
    fnames = []
    for mjd, frames in d.items():
        for frame in frames:
            fnames.append(f'{mjd}/sdR-b1-{frame}.fit.gz')
            fnames.append(f'{mjd}/sdR-r1-{frame}.fit.gz')
    return fnames


def download(file_list, dest):
    with open(args.tempfile, 'w') as f:
        if args.test:
            print('[TESTING MODE] Only going to download the first four files.')
            f.write('\n'.join(file_list[:4]))
        else:
            f.write('\n'.join(file_list))

    rsync_cmd = [
        "rsync",
        "-avz",
        f"--files-from={args.tempfile}",
        "--no-motd",
        BASE_URL,
        dest
    ]
    print("Running the following rsync command:")
    print(f"  {' '.join(rsync_cmd)}")

    process = subprocess.Popen(
        rsync_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    with process.stdout:
        for line in iter(process.stdout.readline, b''):
            if not line:
                break
            print(line.strip())

    # Wait for the process to complete and get the remaining output
    stdout, stderr = process.communicate()

    # Print the remaining output
    print(stdout)
    print(stderr)

    if process.returncode != 0:
        print("rsync failed with return code", process.returncode)


with open(args.infile, 'r') as f:
    data = json.load(f)

if args.arcs:
    arc_fnames = make_filenames(data['arcs'])
    print(len(arc_fnames), 'arcs to download.')
    dest = join(args.dest, 'arcs')
    makedirs(dest, exist_ok=True)
    download(arc_fnames, dest)

if args.flats:
    flat_fnames = make_filenames(data['flats'])
    print(len(flat_fnames), 'flats to download.')
    dest = join(args.dest, 'flats')
    makedirs(dest, exist_ok=True)
    download(flat_fnames, dest)

