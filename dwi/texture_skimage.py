"""Texture code relying on Scikit-image library."""

from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import logging

import numpy as np
import scipy as sp
import skimage.feature
import skimage.filters
import skimage.measure

import dwi.texture
import dwi.util


# Grey-Level Co-Occurrence Matrix (GLCM) features

PROPNAMES = 'contrast dissimilarity homogeneity energy correlation ASM'.split()


def glcm_props(img, names=PROPNAMES, distances=(1, 2, 3, 4),
               ignore_zeros=False):
    """Grey-level co-occurrence matrix (GLCM) texture features.

    Six features provided by scikit-image. Averaged over 4 directions for
    orientation invariance.
    """
    assert img.ndim == 2
    assert img.dtype == np.ubyte
    # Prune distances too long for the window.
    # Commented out: problems with mbb - sometimes includes, sometimes not.
    # distances = [x for x in distances if x <= min(img.shape)-1]
    max_distance = np.sqrt(img.shape[0]**2 + img.shape[1]**2) - 1
    distances = [x for x in distances if x <= max_distance]
    angles = (0, np.pi/4, np.pi/2, 3*np.pi/4)
    levels = img.max() + 1
    glcm = skimage.feature.greycomatrix(img, distances, angles, levels,
                                        symmetric=True, normed=True)
    if ignore_zeros and np.min(img) == 0:
        # Drop information on the first grey-level if it's zero (background).
        glcm = glcm[1:, 1:, ...]
    d = OrderedDict()
    for name in names:
        # Returns array of features indexed by (distance, angle).
        feats = skimage.feature.greycoprops(glcm, name)
        angular_means = np.mean(feats, axis=1)
        angular_ranges = np.ptp(feats, axis=1)
        for dist, am, ar in zip(distances, angular_means, angular_ranges):
            d[(name, dist, 'mean')] = am
            d[(name, dist, 'range')] = ar
    return d


def glcm_map(img, winsize, names=PROPNAMES, ignore_zeros=False, mask=None,
             output=None):
    """Grey-level co-occurrence matrix (GLCM) texture feature map."""
    img = dwi.texture.quantize(dwi.texture.normalize(img))
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats = glcm_props(win, names, ignore_zeros=ignore_zeros)
        if output is None:
            output = np.zeros((len(feats),) + img.shape,
                              dtype=dwi.texture.DTYPE)
        for i, value in enumerate(feats.values()):
            output[(i,) + pos] = value
    names = ['glcm{}'.format(t).translate(None, " '") for t in feats.keys()]
    return output, names


def glcm_mbb(img, mask):
    """Single GLCM features for selected area inside minimum bounding box."""
    img = dwi.texture.quantize(dwi.texture.normalize(img))
    positions = dwi.util.bounding_box(mask)
    slices = [slice(*t) for t in positions]
    img = img[slices]
    mask = mask[slices]
    img[-mask] = 0
    feats = glcm_props(img, ignore_zeros=True)
    output = feats.values()
    names = ['glcm{}'.format(t).translate(None, " '") for t in feats.keys()]
    return output, names


# Local Binary Pattern (LBP) features


def lbp_freq_map(img, winsize, neighbours=8, radius=None, mask=None):
    """Local Binary Pattern (LBP) frequency histogram map."""
    if radius is None:
        radius = winsize // 2
    n = neighbours + 2
    freqs = skimage.feature.local_binary_pattern(img, neighbours, radius,
                                                 method='uniform')
    assert freqs.max() == n - 1, freqs.max()
    output = np.zeros((n,) + img.shape, dtype=np.float16)
    # for i, a in enumerate(output):
    #     a[:, :] = (freqs == i)
    for origin, win in dwi.util.sliding_window(freqs, winsize, mask=mask):
        for i in range(n):
            output[i][origin] = np.count_nonzero(win == i) / win.size
    assert len(output) == n, output.shape
    names = ['lbp({r},{i})'.format(r=radius, i=i) for i in range(n)]
    # if mask is not None:
    #     output[:, -mask] = 0
    return output, names


# Gabor features


def gabor(img, sigmas=None, freqs=None):
    """Gabor features.

    Window size is wavelength 1. Frequency is 1/wavelength; the library uses
    frequency as input parameter. Averaged over 4 directions for orientation
    invariance.
    """
    img = np.asarray(img, dtype=np.double)
    if sigmas is None:
        sigmas = 1, 2, 3
    if freqs is None:
        freqs = 0.1, 0.2, 0.3, 0.4
    thetas = tuple(np.pi/4*i for i in range(4))
    names = 'mean', 'var', 'absmean', 'mag'
    shape = len(thetas), len(sigmas), len(freqs), len(names)
    feats = np.zeros(shape, dtype=dwi.texture.DTYPE)
    for i, j, k in np.ndindex(shape[:-1]):
        real, imag = skimage.filters.gabor(img, frequency=freqs[k],
                                           theta=thetas[i],
                                           sigma_x=sigmas[j],
                                           sigma_y=sigmas[j])
        feats[i, j, k, :] = (np.mean(real), np.var(real),
                             np.mean(np.abs(real)),
                             np.mean(np.sqrt(real**2+imag**2)))
    feats = np.mean(feats, axis=0)  # Average over directions.
    d = OrderedDict()
    for (i, j, k), value in np.ndenumerate(feats):
        key = sigmas[i], freqs[j], names[k]
        d[key] = value
    return d


def gabor_map(img, winsize, sigmas=None, freqs=None, mask=None, output=None):
    """Gabor texture feature map."""
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats = gabor(win, sigmas=sigmas, freqs=freqs)
        if output is None:
            output = np.zeros((len(feats),) + img.shape,
                              dtype=dwi.texture.DTYPE)
        for i, v in enumerate(feats.values()):
            output[(i,) + pos] = v
    names = ['gabor{}'.format(t).translate(None, " '") for t in feats.keys()]
    # fin = np.isfinite(output)
    # output[-fin] = 0  # Make non-finites zero.
    return output, names


def gabor_map_alt(img, winsize, wavelengths=None, mask=None, output=None):
    """Gabor texture feature map."""
    img = np.asarray(img, dtype=np.double)
    if wavelengths is None:
        wavelengths = [2**i for i in range(1, 6)]
    freqs = [1/i for i in wavelengths]
    thetas = [np.pi/4*i for i in range(4)]
    if output is None:
        output = np.zeros((len(thetas), len(freqs)) + img.shape,
                          dtype=dwi.texture.DTYPE)
    img = (img - img.mean()) / img.std()
    for i, theta in enumerate(thetas):
        for j, freq in enumerate(freqs):
            kernel = skimage.filters.gabor_kernel(freq, theta=theta)
            kernel = np.real(kernel)
            logging.info(' '.join([i, j, theta, freq, kernel.shape]))
            a = sp.ndimage.filters.convolve(img[:, :], kernel)
            output[i, j, :, :] = a
    output = np.mean(output, axis=0)  # Average over directions.
    names = ['gabor({})'.format(x) for x in wavelengths]
    return output, names


# Histogram of Oriented Gradients (HOG)


def hog(img):
    """Histogram of oriented gradients (HOG).

    Averaged over directions for orientation invariance.
    """
    # TODO: transform_sqrt=True is here to replicate old behaviour
    # normalise=True, which was removed as incorrect in skimage 0.12. Should
    # this also be removed here? Docs warn against using with negative values.
    kwargs = dict(orientations=8,
                  pixels_per_cell=img.shape,
                  cells_per_block=(1, 1),
                  transform_sqrt=True)
    feats = skimage.feature.hog(img, **kwargs)
    return np.mean(feats)


def hog_map(img, winsize, mask=None, output=None):
    """Histogram of oriented gradients (HOG) texture feature map."""
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats = [hog(win)]
        if output is None:
            output = np.zeros((len(feats),) + img.shape,
                              dtype=dwi.texture.DTYPE)
        for i, v in enumerate(feats):
            output[(i,) + pos] = v
    names = ['hog']
    return output, names


# Hu moments.


def hu(img, postproc=True):
    """The seven moments of Hu.

    If postproc is True, return the logarithms of absolute values.

    These are a derivative of the geometric moments, that are invariant under
    translation, scaling, and rotation (and reflection, if absolute taken).
    """
    img = np.asarray(img, dtype=np.double)
    assert img.ndim == 2
    m = skimage.measure.moments_central(img, img.shape[0]/2, img.shape[1]/2)
    m = skimage.measure.moments_normalized(m)
    m = skimage.measure.moments_hu(m)
    if postproc:
        m = abs(m)  # Last one changes sign on reflection.
        m[m == 0] = 1  # Required by log.
        m = np.log(m)  # They are small, usually logarithms are used.
    m = np.nan_to_num(m)  # Not sure why there are sometimes NaN values.
    assert m.shape == (7,)
    return m


def hu_map(img, winsize, mask=None, output=None):
    """Hu moment map."""
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats = hu(win)
        if output is None:
            output = np.zeros((len(feats),) + img.shape,
                              dtype=dwi.texture.DTYPE)
        for i, v in enumerate(feats):
            output[(i,) + pos] = v
    names = ['hu({})'.format(i) for i in range(len(feats))]
    return output, names


# Sobel.


def sobel(img, mask=None):
    """Sobel edge descriptor."""
    output = skimage.filters.sobel(img, mask=mask)
    return output


def sobel_map(img, winsize=None, mask=None):
    """Sobel edge descriptor map.

    Parameter winsize is not used, it is there for API compatibility.
    """
    output = np.array([sobel(img), sobel(img, mask=mask)])
    names = ['sobel', 'sobel_mask']
    return output, names
