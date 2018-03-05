"""Texture code relying on Scikit-image library."""

from collections import OrderedDict
from itertools import product

import numpy as np
import skimage.feature
import skimage.filters
import skimage.measure

import dwi.texture
import dwi.util


def translate_name(name):
    """Translate feature name."""
    return name.translate(str.maketrans('', '', " '"))


def get_angles(n):
    """Divide a half-circle into n angles."""
    return tuple(np.pi/n*i for i in range(n))


# Grey-Level Co-Occurrence Matrix (GLCM) features


def glcm_props(img, ignore_zeros=False):
    """Grey-level co-occurrence matrix (GLCM) texture features.

    Include the six features provided by scikit-image. Calculate mean and range
    over 4 directions for orientation invariance (features 'mean', 'range').

    Add an alternative approach for orientation invariance (feature 'alt', see
    Vignati et al. 2015: Texture features on T2-weighted magnetic resonance
    imaging: new potential biomarkers for prostate cancer aggressiveness).
    """
    names = dwi.rcParams.texture_glcm_names
    distances = dwi.rcParams.texture_glcm_distances
    assert img.ndim == 2, img.shape
    assert img.dtype == np.uint8, img.dtype
    # Prune distances too long for the window.
    # Commented out: problems with mbb - sometimes includes, sometimes not.
    # distances = [x for x in distances if x <= min(img.shape)-1]
    max_distance = np.sqrt(img.shape[0]**2 + img.shape[1]**2) - 1
    distances = [x for x in distances if x <= max_distance]
    angles = get_angles(4)
    levels = img.max() + 1
    glcm = skimage.feature.greycomatrix(img, distances, angles, levels,
                                        symmetric=True, normed=True)
    if ignore_zeros and np.min(img) == 0:
        # Drop information on the first grey-level if it's zero (background).
        glcm = glcm[1:, 1:, ...]
    glcm_mean = np.mean(glcm, axis=3, keepdims=True)
    d = OrderedDict()
    for name in names:
        # Returns array of features indexed by (distance, angle).
        feats = skimage.feature.greycoprops(glcm, name)
        feats_alt = skimage.feature.greycoprops(glcm_mean, name)
        angular_means = np.mean(feats, axis=1)
        angular_ranges = np.ptp(feats, axis=1)
        for dist, am, ar, alt in zip(distances, angular_means, angular_ranges,
                                     feats_alt):
            d[(name, dist, 'mean')] = am
            d[(name, dist, 'range')] = ar
            d[(name, dist, 'alt')] = alt
    return d


def glcm_map(img, winsize, mask=None, output=None, ignore_zeros=False):
    """Grey-level co-occurrence matrix (GLCM) texture feature map."""
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats = glcm_props(win, ignore_zeros=ignore_zeros)
        if output is None:
            dtype = dwi.rcParams.texture_dtype
            output = np.zeros((len(feats),) + img.shape, dtype=dtype)
        for i, value in enumerate(feats.values()):
            output[(i,) + pos] = value
    names = [translate_name('glcm{}'.format(t)) for t in feats.keys()]
    return output, names


def glcm_mbb(img, mask):
    """Single GLCM features for selected area inside minimum bounding box."""
    positions = dwi.util.bounding_box(mask)
    slices = [slice(*t) for t in positions]
    img = img[slices]
    mask = mask[slices]
    img[~mask] = 0
    feats = glcm_props(img, ignore_zeros=True)
    output = list(feats.values())
    names = [translate_name('glcm{}'.format(t)) for t in feats.keys()]
    return output, names


# Local Binary Pattern (LBP) features


def lbp_freq_map(img, winsize, mask=None):
    """Local Binary Pattern (LBP) frequency histogram map."""
    neighbours = dwi.rcParams.texture_lbp_neighbours
    radius = winsize // 2
    n = neighbours + 2
    freqs = skimage.feature.local_binary_pattern(img, neighbours, radius,
                                                 method='uniform')
    assert freqs.max() == n - 1, freqs.max()
    output = np.zeros((n,) + img.shape, dtype=np.float32)
    # for i, a in enumerate(output):
    #     a[:, :] = (freqs == i)
    for origin, win in dwi.util.sliding_window(freqs, winsize, mask=mask):
        for i in range(n):
            output[i][origin] = np.count_nonzero(win == i) / win.size
    assert len(output) == n, output.shape
    names = ['lbp({r},{i})'.format(r=radius, i=i) for i in range(n)]
    # if mask is not None:
    #     output[:, ~mask] = 0
    return output, names


# Gabor features


GABOR_FEAT_NAMES = ('mean', 'var', 'absmean', 'mag')


def gabor_feats(real, imag):
    return (np.mean(real), np.var(real), np.mean(np.abs(real)),
            np.mean(np.sqrt(real**2+imag**2)))


def gabor(img):
    """Gabor features.

    Window size is wavelength 1. Frequency is 1/wavelength; the library uses
    frequency as input parameter. Averaged over 4 directions for orientation
    invariance.
    """
    img = np.asarray(img, dtype=np.float32)
    sigmas = dwi.rcParams.texture_gabor_sigmas
    freqs = dwi.rcParams.texture_gabor_freqs
    thetas = get_angles(dwi.rcParams.texture_gabor_orientations)
    names = GABOR_FEAT_NAMES
    shape = len(thetas), len(sigmas), len(freqs), len(names)
    dtype = dwi.rcParams.texture_dtype
    feats = np.zeros(shape, dtype=dtype)
    for i, j, k in np.ndindex(shape[:-1]):
        real, imag = skimage.filters.gabor(img, frequency=freqs[k],
                                           theta=thetas[i],
                                           sigma_x=sigmas[j],
                                           sigma_y=sigmas[j])
        feats[i, j, k, :] = gabor_feats(real, imag)
    feats = np.mean(feats, axis=0)  # Average over directions.
    d = OrderedDict()
    for (i, j, k), value in np.ndenumerate(feats):
        key = sigmas[i], freqs[j], names[k]
        d[key] = value
    return d


def gabor_featmap(real, imag, winsize, mask):
    """Get Gabor feature map of shape (feats, height, width) from the filtered
    image.
    """
    assert real.shape == imag.shape, (real.shape, imag.shape)
    rit = iter(dwi.util.sliding_window(real, winsize, mask=mask))
    iit = iter(dwi.util.sliding_window(imag, winsize, mask=mask))
    shape = (len(GABOR_FEAT_NAMES),) + real.shape
    output = np.full(shape, np.nan, dtype=np.float32)
    for ((y, x), rwin), (_, iwin) in zip(rit, iit):
        output[:, y, x] = gabor_feats(rwin, iwin)
    return output


def get_sigma_x(frequency, bandwidth=1):
    """Get sigma x for Gabor filter by setting frequency cut-off to -6dB, with
    bandwidth in octaves.

    See Clausi &al. 2000: Designing Gabor filters for optimal texture
    separability. Bandwidth of one octave is suggested for default value.
    """
    return ((np.sqrt(np.log(2)) * (2**bandwidth + 1)) /
            (np.sqrt(2) * np.pi * frequency * (2**bandwidth - 1)))


def get_sigma_y(frequency, bandwidth=None):
    """Get sigma y for Gabor filter by setting frequency cut-off to -6dB, with
    bandwidth in octaves.

    See Clausi &al. 2000: Designing Gabor filters for optimal texture
    separability. Bandwidth of one octave is suggested for default value.
    """
    if bandwidth is None:
        bandwidth = np.pi / 6  # 30 degrees.
    return (np.sqrt(np.log(2)) /
            (np.sqrt(2) * np.pi * frequency * np.tan(bandwidth / 2)))


def gabor_map(img, winsize, mask=None, output=None):
    """Gabor texture feature map. This is the (more) correct way."""
    img = np.asarray(img, dtype=np.float32)
    sigmas = dwi.rcParams.texture_gabor_sigmas
    freqs = dwi.rcParams.texture_gabor_freqs
    thetas = get_angles(dwi.rcParams.texture_gabor_orientations)
    featnames = GABOR_FEAT_NAMES
    tmaps = []
    outnames = []
    reals = np.empty((len(thetas),) + img.shape, dtype=np.float32)
    imags = np.empty_like(reals)
    for sigma, freq in product(sigmas, freqs):
        if sigma is None:
            sigma_x = get_sigma_x(freq)
            sigma_y = get_sigma_y(freq)
        else:
            sigma_x = sigma_y = sigma
        for t, theta in enumerate(thetas):
            reals[t], imags[t] = skimage.filters.gabor(img, frequency=freq,
                                                       theta=theta,
                                                       # bandwidth=1)
                                                       sigma_x=sigma_x,
                                                       sigma_y=sigma_y)
            reals[t][np.isnan(reals[t])] = 0
            imags[t][np.isnan(imags[t])] = 0
            assert np.all(np.isfinite(reals[t])), ('r', freq, theta)
            assert np.all(np.isfinite(imags[t])), ('i', freq, theta)
        # Sum orientations for invariance.
        real = np.sum(reals, axis=0)
        imag = np.sum(imags, axis=0)
        featmaps = gabor_featmap(real, imag, winsize, mask)
        for featmap, name in zip(featmaps, featnames):
            tmaps.append(featmap)
            s = translate_name('gabor{}'.format((sigma, freq, name)))
            outnames.append(s)
    output = np.array(tmaps)
    return output, outnames


# Histogram of Oriented Gradients (HOG)


def hog(img):
    """Histogram of oriented gradients (HOG).

    Averaged over directions for orientation invariance.
    """
    # TODO: transform_sqrt=True is here to replicate old behaviour
    # normalise=True, which was removed as incorrect in skimage 0.12. Should
    # this also be removed here? Docs warn against using with negative values.
    kwargs = dict(orientations=dwi.rcParams.texture_hog_orientations,
                  pixels_per_cell=img.shape,
                  cells_per_block=(1, 1),
                  transform_sqrt=False)
    feats = skimage.feature.hog(img, **kwargs)
    return np.mean(feats)


def hog_map(img, winsize, mask=None, output=None):
    """Histogram of oriented gradients (HOG) texture feature map."""
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats = [hog(win)]
        if output is None:
            dtype = dwi.rcParams.texture_dtype
            output = np.zeros((len(feats),) + img.shape, dtype=dtype)
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
    img = np.asarray(img, dtype=np.double)  # Requires np.double.
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
            dtype = dwi.rcParams.texture_dtype
            output = np.zeros((len(feats),) + img.shape, dtype=dtype)
        for i, v in enumerate(feats):
            output[(i,) + pos] = v
    # TODO: Shift indices in feature names to be one-based.
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
