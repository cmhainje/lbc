def load_image(filename):
    import numpy as np
    from astropy.io import fits
    from lbc.data import split_by_amplifier

    image = fits.getdata(filename)
    camera = 'r' if '-r1-' in filename else 'b'
    quads = split_by_amplifier(image, red=(camera == 'r'))

    # bias subtraction
    pquads = []
    for i, (d, b) in enumerate(quads):
        bsub = np.copy(d).astype(np.int32)
        bsub -= np.median(b).astype(np.uint16)

        # crop quadrants to 2048x2048 squares
        if camera == 'r':
            bsub = bsub[16:] if i in [0, 2] else bsub[:-16]
            bsub = bsub[:, 9:] if i in [0, 1] else bsub[:, :-9]
        else:
            bsub = bsub[8:] if i in [0, 2] else bsub[:-8]
        pquads.append(bsub)

    return np.array([
        [pquads[0], pquads[2]],
        [pquads[1], pquads[3]],
    ])

def flatten_image(image):
    import jax.numpy as jnp
    image = jnp.concatenate([ image[:, 0], image[:, 1] ], axis=-1)
    image = jnp.concatenate([ image[0], image[1] ], axis=-2)
    return image
