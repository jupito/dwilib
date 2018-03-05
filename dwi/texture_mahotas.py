"""Texture code relying on Mahotas library."""

from collections import OrderedDict

import numpy as np
from scipy import ndimage
import mahotas

import dwi.texture
import dwi.util


# Zernike moments.


def zernike(img, radius, degree=8, cm=None):
    """Zernike moments.

    This geometric moment derivate is based on alternative orthogonal
    polynomials, which makes it more optimal wrt. information redundancy. These
    are invariant to rotation.
    """
    img = np.asarray(img, dtype=np.float32)
    assert img.ndim == 2
    feats = mahotas.features.zernike_moments(img, radius, degree=degree, cm=cm)
    return feats


def zernike_map(img, winsize, mask=None, output=None):
    """Zernike moment map."""
    degree = dwi.rcParams.texture_zernike_degree
    radius = winsize // 2
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats = zernike(win, radius, degree=degree, cm=(radius, radius))
        if output is None:
            dtype = dwi.rcParams.texture_dtype
            output = np.zeros((len(feats),) + img.shape, dtype=dtype)
        for i, v in enumerate(feats):
            output[(i,) + pos] = v
    names = ['zernike({})'.format(i) for i in range(len(feats))]
    return output, names


# Haar transformation


def haar(img):
    """Haar wavelet transform."""
    # Cannot have nans here, they might have global influence.
    img = np.asarray_chkfinite(img)
    assert img.ndim == 2
    # assert img.shape[0] % 2 == img.shape[1] % 2 == 0
    # Prune possible odd borders.
    newshape = [x - x % 2 for x in img.shape]
    img = img[:newshape[0], :newshape[1]]
    a = mahotas.haar(img)
    h, w = [x//2 for x in a.shape]
    coeffs = [a[:h, :w], a[:h, w:],
              a[h:, :w], a[h:, w:]]
    coeffs = [ndimage.interpolation.zoom(l, 2.) for l in coeffs]
    return coeffs


def haar_levels(img, nlevels, drop_approx=False):
    """Multi-level Haar wavelet transform."""
    levels = []
    for _ in range(nlevels):
        coeffs = haar(img)
        levels.append(coeffs)
        img = coeffs[0]  # Set source for next iteration step.
    if drop_approx:
        levels = [l[1:] for l in levels]
    return levels


def haar_features(win):
    """Haar texture features of a single level."""
    d = OrderedDict()
    d['aav'] = np.mean(np.abs(win))
    d['std'] = np.std(win)
    return d


def haar_map(img, winsize, mask=None, output=None):
    """Haar texture feature map."""
    nlevels = dwi.rcParams.texture_haar_levels
    # Cannot have nans here, they might have global influence.
    nans = np.isnan(img)
    if np.count_nonzero(nans):
        img[nans] = 0  # XXX: Replace with minimum value instead?
    levels = haar_levels(img, nlevels, drop_approx=True)
    names = []
    for i, coeffs in enumerate(levels):
        for j, coeff in enumerate(coeffs):
            for pos, win in dwi.util.sliding_window(coeff, winsize, mask):
                feats = haar_features(win)
                if output is None:
                    dtype = dwi.rcParams.texture_dtype
                    output = np.zeros((len(levels), len(coeffs), len(feats),) +
                                      coeff.shape, dtype=dtype)
                for k, v in enumerate(feats.values()):
                    output[(i, j, k,) + pos] = v
            s = 'haar({level},{coeff},{feat})'
            names += [s.format(level=i+1, coeff=j+1, feat=k) for k in
                      feats.keys()]
    output.shape = (-1,) + levels[0][0].shape
    return output, names


# Haralick (never used this because of those weird eigenvalue errors)


def haralick(img, ignore_zeros=False):
    """Haralick texture features.

    14 features provided by mahotas. Averaged over 4 directions for orientation
    invariance.

    Note: This implementation has some issues, it fails with some input raising
    an exception about eigenvectors not converging.
    """
    assert img.ndim == 2, img.shape
    assert img.dtype == np.uint8, img.dtype
    a = mahotas.features.texture.haralick(img, ignore_zeros,
                                          compute_14th_feature=True)
    a = np.mean(a, axis=0)
    return a, mahotas.features.texture.haralick_labels


def haralick_map(img, winsize, mask=None, output=None, ignore_zeros=False):
    """Haralick texture feature map."""
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats, names = haralick(win, ignore_zeros=ignore_zeros)
        if output is None:
            dtype = dwi.rcParams.texture_dtype
            output = np.zeros((len(names),) + img.shape, dtype=dtype)
        for i, v in enumerate(feats):
            output[(i,) + pos] = v
    names = ['haralick({i}-{n})'.format(i=i+1, n=dwi.util.abbrev(n))
             for i, n in enumerate(names)]
    return output, names


def haralick_mbb(img, mask):
    """Haralick features for selected area inside minimum bounding box."""
    positions = dwi.util.bounding_box(mask)
    slices = [slice(*t) for t in positions]
    img = img[slices]
    mask = mask[slices]
    img[~mask] = 0
    feats, names = haralick(img, ignore_zeros=True)
    names = ['haralick({i}-{n})'.format(i=i+1, n=dwi.util.abbrev(n))
             for i, n in enumerate(names)]
    return feats, names
