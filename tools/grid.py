#!/usr/bin/env python2

"""Get grid-wise features."""

from __future__ import absolute_import, division, print_function
import argparse
from itertools import product

import numpy as np

import dwi.files
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
                   help='increase verbosity')
    p.add_argument('--image', required=True,
                   help='input image or pmap')
    p.add_argument('--prostate', metavar='MASKFILE', required=True,
                   help='prostate mask')
    p.add_argument('--lesions', metavar='MASKFILE', nargs='+', required=True,
                   help='lesion masks')
    p.add_argument('--winshape', metavar='I', type=int, nargs=3,
                   default=[1, 5, 5],
                   help='window shape')
    p.add_argument('--output', metavar='FILENAME', required=True,
                   help='output ASCII file')
    return p.parse_args()


def read_mask(path):
    """Read pmap as a mask."""
    return dwi.files.read_pmap(path)[0][..., 0].astype(np.bool)


def unify_masks(masks):
    """Unify a sequence of masks into one."""
    return np.sum(masks, axis=0, dtype=np.bool)


def grid_slices(imageshape, winshape, center):
    """Generate slice objects for a grid of windows around given center.

    Float center will be rounded.
    """
    center = tuple(int(round(x)) for x in center)
    starts = [i % w for i, w in zip(center, winshape)]
    stops = [i-w+1 for i, w in zip(imageshape, winshape)]
    its = (xrange(*x) for x in zip(starts, stops, winshape))
    for coords in product(*its):
        slices = tuple(slice(i, i+w) for i, w in zip(coords, winshape))
        yield slices


def main():
    args = parse_args()
    image, attrs = dwi.files.read_pmap(args.image)
    image = image[..., 0]
    prostate = read_mask(args.prostate)
    lesion = unify_masks([read_mask(x) for x in args.lesions])
    image[-prostate] = np.nan  # This ok?

    assert image.shape == prostate.shape == lesion.shape

    centroid = dwi.util.centroid(prostate)

    print(attrs)
    print('Image:', image.shape, image.dtype)
    print('Lesions:', len(args.lesions))
    print('Prostate centroid:', centroid)

    X = []
    for slices in grid_slices(image.shape, args.winshape, centroid):
        image_win = image[slices]
        prostate_win = prostate[slices]
        lesion_win = lesion[slices]
        assert image_win.shape == prostate_win.shape == lesion_win.shape
        # At least half of the window must be of prostate.
        if np.count_nonzero(prostate_win) / prostate_win.size >= 0.5:
            # wincorner = tuple(x.start for x in slices)
            wincenter = tuple(np.mean((x.start, x.stop)) for x in slices)
            x = [
                # wincorner,
                dwi.util.distance(centroid, wincenter),
                (np.count_nonzero(lesion_win) > 0),
                np.nanmean(image_win),
                np.nanmedian(image_win),
            ]
            X.append(x)

    params = 'distance lesion mean median'.split()
    attrs = dict(parameters=params, n_lesions=len(args.lesions))
    dwi.files.write_pmap(args.output, X, attrs)

    import scipy.stats
    X = np.array(X)
    print(X.shape, X.dtype)
    for i, j in product(range(X.shape[-1]), range(X.shape[-1])):
        if j > i:
            print(params[i], params[j],
                  scipy.stats.spearmanr(X[:, i], X[:, j]))


if __name__ == '__main__':
    main()