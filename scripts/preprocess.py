import numpy as np
import pandas as pd

from astropy.io import fits

from glob import glob
from tqdm.auto import tqdm
from os.path import join, basename


def extract_headers(folder):
    try:
        df = pd.read_csv(join(folder, "headers.csv"))
    except FileNotFoundError:
        filenames = sorted(glob(f"{folder}/*.fit.gz"))
        headers = [None] * len(filenames)
        for i, filename in enumerate(tqdm(filenames, desc="Reading headers")):
            with fits.open(filename) as f:
                headers[i] = dict(f[0].header)
        df = pd.DataFrame(headers)
        df.to_csv(join(folder, "headers.csv"))

    return df


def split_amplifiers(image_data, red=True):
    # magic numbers are taken from
    # https://data.sdss.org/datamodel/files/BOSS_SPECTRO_DATA/MJD/sdR.html
    # note that the IDL slice syntax _includes_ the endpoint, so we add 1
    # also the rows and columns are transposed relative to the docs

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


def get_cropped_image(filename):
    img = fits.getdata(filename)

    # split by amplifiers and subtract bias
    camera = basename(filename).split("-")[1][1]
    split = split_amplifiers(img, red=camera == "r")
    bias_removed_quadrants = [d - np.median(b) for d, b in split]

    # for now, let's just take a 128x128 square from the bottom-left quadrant
    x = bias_removed_quadrants[2][800:928, 800:928]
    return x


def process_arcs(folder, output_folder):
    df = extract_headers(folder)
    arcs = df.loc[df["FLAVOR"] == "arc"]
    arc_exposures = np.unique(arcs["EXPOSURE"])

    # find the interesting columns
    cols_of_interest = [
        x
        for x in df.loc[:, df.nunique() > 1].columns
        if x not in ["FILENAME", "CAMERAS", "EXPOSURE"]
    ]

    all_metadata = []

    for exp_num in arc_exposures:
        # for now, we're only looking at r2 data
        exposure = df.loc[
            (df["CAMERAS"] == "r2") & (df["EXPOSURE"] == exp_num)
        ].squeeze()

        # extract the metadata
        metadata = exposure[cols_of_interest]

        # extract a cropped image
        filename = join(folder, exposure.FILENAME + ".gz")
        image = get_cropped_image(filename)

        # save
        all_metadata.append(metadata)
        out_filename = exposure.FILENAME.replace(".fit", ".npy")
        np.save(join(output_folder, out_filename), image)

    pd.DataFrame(all_metadata).to_csv(join(output_folder, "metadata.csv"))


if __name__ == "__main__":
    process_arcs(
        "/Users/connor/data/lbc/58540", "/Users/connor/data/lbc/58540-processed"
    )
