"""Texture analysis.

Texture methods are of three possible types: 'map' produces a texture map by
using a sliding window, 'mbb' uses selected voxels to produce a single value
for each slice, and 'all' produces a single value for selected voxels from all
slices.

Output type can be one of the following: 'map' just returns the map or spreads
the single values over all selected voxels, 'mean' and 'median' return just the
single values or reduces the map into a single average value.
"""

from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import product

import numpy as np
import scipy as sp
import skimage
import skimage.exposure
import skimage.feature
import skimage.filter

import dwi.util


def normalize(pmap, levels=128):
    """Normalize images within given range and convert to byte maps with given
    number of graylevels."""
    #in_range = (0, 0.03)
    in_range = (0, 0.005)
    #in_range = (0, 0.002)
    #in_range = (pmap.min(), pmap.max())
    #in_range = (0, sp.stats.scoreatpercentile(pmap, 99.8))
    in_range = (sp.stats.scoreatpercentile(pmap, 0.8),
                sp.stats.scoreatpercentile(pmap, 99.2))
    if pmap.dtype == np.int:
        # The rescaler cannot handle int32.
        pmap = np.asarray(pmap, dtype=np.int16)
    if pmap.dtype == np.int16:
        in_range = (0, 4095)
    pmap = skimage.exposure.rescale_intensity(pmap, in_range=in_range)
    pmap = skimage.img_as_ubyte(pmap)
    pmap /= (256/levels)
    return pmap


def abbrev(name):
    """Abbreviate multiword feature name."""
    if ' ' in name:
        name = ''.join(word[0] for word in name.split())
    return name


# Basic statistical features


def stats(img):
    """Statistical texture features that don't consider spatial relations."""
    d = OrderedDict()
    d['mean'] = np.mean(img)
    d['stddev'] = np.std(img)
    d['range'] = np.max(img) - np.min(img)
    d.update(dwi.util.fivenumd(img))
    for i in range(1, 10):
        d['decile%i' % i] = sp.stats.scoreatpercentile(img, i*10)
    d['kurtosis'] = sp.stats.kurtosis(img.ravel())
    d['skewness'] = sp.stats.skew(img.ravel())
    return d


def stats_map(img, winsize, names=None, mask=None, output=None):
    """Statistical texture feature map."""
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        d = stats(win)
        if names is None:
            names = d.keys()
        if output is None:
            output = np.zeros((len(names),) + img.shape)
        for i, name in enumerate(names):
            output[(i,) + pos] = d[name]
    names = ['stats({})'.format(n) for n in names]
    return output, names


def stats_mbb(img, mask):
    """Statistical texture features unified over a masked area."""
    voxels = img[mask]
    feats = stats(voxels)
    output = feats.values()
    names = ['stats({})'.format(k) for k in feats.keys()]
    return output, names


# Grey-Level Co-Occurrence Matrix (GLCM) features

PROPNAMES = 'contrast dissimilarity homogeneity energy correlation ASM'.split()


def glcm_props(img, names=PROPNAMES, distances=(1,2,3,4), ignore_zeros=False):
    """Grey-level co-occurrence matrix (GLCM) texture features.

    Six features provided by scikit-image. Averaged over 4 directions for
    orientation invariance.
    """
    from skimage.feature import greycomatrix, greycoprops
    assert img.ndim == 2
    assert img.dtype == np.ubyte
    #distances = [x for x in distances if x <= min(img.shape)-1]
    angles = (0, np.pi/4, np.pi/2, 3*np.pi/4)
    levels = img.max() + 1
    glcm = greycomatrix(img, distances, angles, levels, symmetric=True,
                        normed=True)
    if ignore_zeros and np.min(img) == 0:
        # Drop information on the first grey-level if it's zero (background).
        glcm = glcm[1:,1:,...]
    d = OrderedDict()
    for name in names:
        feats = greycoprops(glcm, name)  # Returns array (distance, angle).
        feats = np.mean(feats, axis=1)  # Average over angles.
        for dist, feat in zip(distances, feats):
            d[(name, dist)] = feat
        d[(name, 'avg')] = np.mean(feats)
    return d


def glcm_map(img, winsize, names=PROPNAMES, ignore_zeros=False, mask=None,
             output=None):
    """Grey-level co-occurrence matrix (GLCM) texture feature map."""
    img = normalize(img)
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats = glcm_props(win, names, ignore_zeros=ignore_zeros)
        if output is None:
            output = np.zeros((len(feats),) + img.shape)
        for i, value in enumerate(feats.values()):
            output[(i,) + pos] = value
    names = ['glcm{}'.format(t).translate(None, " '") for t in feats.keys()]
    return output, names


def glcm_mbb(img, mask):
    """Single GLCM features for selected area inside minimum bounding box."""
    img = normalize(img)
    positions = dwi.util.bounding_box(mask)
    slices = [slice(*t) for t in positions]
    img = img[slices]
    mask = mask[slices]
    img[-mask] = 0
    feats = glcm_props(img, ignore_zeros=True)
    output = feats.values()
    names = ['glcm{}'.format(t).translate(None, " '") for t in feats.keys()]
    return output, names


def haralick(img, ignore_zeros=False):
    """Haralick texture features.

    14 features provided by mahotas. Averaged over 4 directions for orientation
    invariance.

    Note: This implementation has some issues, it fails with some input raising
    an exception about eigenvectors not converging.
    """
    import mahotas
    a = mahotas.features.texture.haralick(img, ignore_zeros,
                                          compute_14th_feature=True)
    a = np.mean(a, axis=0)
    return a, mahotas.features.texture.haralick_labels


def haralick_map(img, winsize, ignore_zeros=False, mask=None, output=None):
    """Haralick texture feature map."""
    img = normalize(img)
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats, names = haralick(win, ignore_zeros=ignore_zeros)
        if output is None:
            output = np.zeros((len(names),) + img.shape)
        for i, v in enumerate(feats):
            output[(i,) + pos] = v
    names = ['haralick({i}-{n})'.format(i=i+1, n=abbrev(n)) for i, n in
             enumerate(names)]
    return output, names


def haralick_mbb(img, mask):
    """Haralick features for selected area inside minimum bounding box."""
    img = normalize(img)
    positions = dwi.util.bounding_box(mask)
    slices = [slice(*t) for t in positions]
    img = img[slices]
    mask = mask[slices]
    img[-mask] = 0
    feats, names = haralick(img, ignore_zeros=True)
    names = ['haralick({i}-{n})'.format(i=i+1, n=abbrev(n)) for i, n in
             enumerate(names)]
    return feats, names


# Local Binary Pattern (LBP) features


def lbp_freqs(img, winsize, neighbours=8, radius=1, roinv=1, uniform=1):
    """Local Binary Pattern (LBP) frequency histogram map.

    Invariant to global illumination change, local contrast magnitude, local
    rotation.
    """
    from dwi import lbp
    lbp_data = lbp.lbp(img, neighbours, radius, roinv, uniform)
    lbp_freq_data, n_patterns = lbp.get_freqs(lbp_data, winsize, neighbours,
                                              roinv, uniform)
    return lbp_data, lbp_freq_data, n_patterns


def lbp_freq_map(img, winsize, neighbours=8, radius=None, mask=None):
    """Local Binary Pattern (LBP) frequency histogram map."""
    if radius is None:
        radius = winsize // 2
    _, freqs, n = lbp_freqs(img, winsize, neighbours=neighbours, radius=radius)
    output = np.rollaxis(freqs, -1)
    names = ['lbp({r},{i})'.format(r=radius, i=i) for i in range(n)]
    if mask is not None:
        output[:,-mask] = 0
    return output, names


def lbpf_dist(hist1, hist2, method='chi-squared', eps=1e-6):
    """Measure the distance of two LBP frequency histograms.

    Method can be one of the following:
    intersection: histogram intersection
    log-likelihood: log-likelihood
    chi-squared: chi-squared
    """
    pairs = np.array([hist1, hist2]).T
    if method == 'intersection':
        r = sum(min(pair) for pair in pairs)
    elif method == 'log-likelihood':
        r = -sum(a*np.log(max(b, eps)) for a, b in pairs)
    elif method == 'chi-squared':
        r = sum((a-b)**2/(max(a+b, eps)) for a, b in pairs)
    else:
        raise Exception('Unknown distance measure: %s' % method)
    return r


# Gabor features


def gabor(img, sigmas=(1, 2, 3), freqs=(0.1, 0.2, 0.3, 0.4)):
    """Gabor features.

    Averaged over 4 directions for orientation invariance.
    """
    thetas = [np.pi/4*i for i in range(4)]
    shape = len(thetas), len(sigmas), len(freqs)
    feats = np.zeros(shape + (2,))
    for i, j, k in np.ndindex(shape):
        t, s, f = thetas[i], sigmas[j], freqs[k]
        kwargs = dict(frequency=f, theta=t, sigma_x=s, sigma_y=s)
        real, _ = skimage.filter.gabor_filter(img, **kwargs)
        feats[i,j,k,0] = real.mean()
        feats[i,j,k,1] = real.var()
    feats = np.mean(feats, axis=0)  # Average over directions.
    d = OrderedDict()
    for (i, j, k), value in np.ndenumerate(feats):
        key = sigmas[i], freqs[j], ('mean', 'var')[k]
        d[key] = value
    return d


def gabor_map(img, winsize, sigmas=(1, 2, 3), freqs=(0.1, 0.2, 0.3, 0.4),
              mask=None, output=None):
    """Gabor texture feature map."""
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats = gabor(win, sigmas, freqs)
        if output is None:
            output = np.zeros((len(feats),) + img.shape)
        for i, v in enumerate(feats.values()):
            output[(i,) + pos] = v
    names = ['gabor{}'.format(t).translate(None, " '") for t in feats.keys()]
    return output, names


# Histogram of Oriented Gradients (HOG)


def hog(img):
    """Histogram of oriented gradients (HOG).

    Averaged over directions for orientation invariance.
    """
    kwargs = dict(orientations=8,
                  pixels_per_cell=img.shape,
                  cells_per_block=(1, 1),
                  normalise=True)
    feats = skimage.feature.hog(img, **kwargs)
    return np.mean(feats)


def hog_map(img, winsize, mask=None, output=None):
    """Histogram of oriented gradients (HOG) texture feature map."""
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats = [hog(win)]
        if output is None:
            output = np.zeros((len(feats),) + img.shape)
        for i, v in enumerate(feats):
            output[(i,) + pos] = v
    names = ['hog']
    return output, names


# Image moments


def moment(img, p, q):
    """Image moment. See Tuceryan 1994: Moment Based Texture Segmentation."""
    img = np.asarray(img)
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise Exception('Image not square: {}'.format(img.shape))
    width = img.shape[0]
    center = width//2
    nc = lambda pos: (pos-center) / (width/2)  # Normalized coordinates [-1,1]
    f = lambda m, n: img[m,n] * nc(m)**p * nc(n)**q
    a = np.fromfunction(f, img.shape, dtype=int)
    return a.sum()


def moments(img, max_order=2):
    """Image moments of order up to p+q <= max_order."""
    r = range(max_order+1)
    tuples = (t for t in product(r, r) if sum(t) <= max_order)
    d = OrderedDict(((p, q), moment(img, p, q)) for p, q in tuples)
    return d


def moment_map(img, winsize, max_order=12, mask=None, output=None):
    """Image moment map."""
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats = moments(win, max_order=max_order)
        if output is None:
            output = np.zeros((len(feats),) + img.shape)
        for i, v in enumerate(feats.values()):
            output[(i,) + pos] = v
    names = ['moment{}'.format(t).translate(None, " '") for t in feats.keys()]
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
        m[m==0] = 1  # Required by log.
        m = np.log(m)  # They are small, usually logarithms are used.
    m = np.nan_to_num(m)  # Not sure why there are sometimes NaN values.
    assert m.shape == (7,)
    return m


def hu_map(img, winsize, mask=None, output=None):
    """Hu moment map."""
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats = hu(win)
        if output is None:
            output = np.zeros((len(feats),) + img.shape)
            for i, v in enumerate(feats):
                output[(i,) + pos] = v
    names = ['hu({})'.format(i) for i in range(len(feats))]
    return output, names


# Zernike moments.


def zernike(img, radius, degree=8, cm=None):
    """Zernike moments.

    This geometric moment derivate is based on alternative orthogonal
    polynomials, which makes it more optimal wrt. information redundancy. These
    are invariant to rotation.
    """
    import mahotas
    img = np.asarray(img)
    assert img.ndim == 2
    feats = mahotas.features.zernike_moments(img, radius, degree=degree, cm=cm)
    return feats


def zernike_map(img, winsize, radius=None, degree=8, mask=None, output=None):
    """Zernike moment map."""
    if radius is None:
        radius = winsize // 2
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats = zernike(win, radius, degree=degree, cm=(radius, radius))
        if output is None:
            output = np.zeros((len(feats),) + img.shape)
            for i, v in enumerate(feats):
                output[(i,) + pos] = v
    names = ['zernike({})'.format(i) for i in range(len(feats))]
    return output, names


# Haar transformation


def haar(img):
    """Haar wavelet transform."""
    import mahotas
    assert img.ndim == 2
    #assert img.shape[0] % 2 == img.shape[1] % 2 == 0
    # Prune possible odd borders.
    newshape = [x-x%2 for x in img.shape]
    img = img[:newshape[0], :newshape[1]]
    a = mahotas.haar(img)
    h, w = [x//2 for x in a.shape]
    coeffs = [a[:h,:w], a[:h,w:],
              a[h:,:w], a[h:,w:]]
    coeffs = [sp.ndimage.interpolation.zoom(l, 2.) for l in coeffs]
    return coeffs


def haar_levels(img, nlevels=4, drop_approx=False):
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


def haar_map(img, winsize, nlevels=4, mask=None, output=None):
    """Haar texture feature map."""
    levels = haar_levels(img, nlevels=nlevels, drop_approx=True)
    names = []
    for i, coeffs in enumerate(levels):
        for j, coeff in enumerate(coeffs):
            for pos, win in dwi.util.sliding_window(coeff, winsize, mask):
                feats = haar_features(win)
                if output is None:
                    output = np.zeros((len(levels), len(coeffs), len(feats),) +
                                      coeff.shape)
                for k, v in enumerate(feats.values()):
                    output[(i, j, k,) + pos] = v
            s = 'haar({level},{coeff},{feat})'
            names += [s.format(level=i+1, coeff=j+1, feat=k) for k in
                      feats.keys()]
    output.shape = (-1,) + levels[0][0].shape
    return output, names


# Sobel.


def sobel(img, mask=None):
    """Sobel edge descriptor."""
    output = skimage.filter.sobel(img, mask=mask)
    return output


def sobel_map(img, winsize=None, mask=None):
    """Sobel edge descriptor map.

    Parameter winsize is not used, it is there for API compatibility."""
    assert winsize is None
    output = np.array([sobel(img), sobel(img, mask=mask)])
    names = ['sobel', 'sobel_mask']
    return output, names


def sobel_mbb(img, mask):
    """Sobel edge descriptor map."""
    output = [np.mean(sobel(img, mask=mask))]
    names = ['sobel']
    return output, names


# General texture map.


# Methods that consider an n*n window.
METHODS = OrderedDict([
    ('stats', stats_map),
    ('glcm', glcm_map),
    ('haralick', haralick_map),
    ('lbp', lbp_freq_map),
    ('hog', hog_map),
    ('gabor', gabor_map),
    ('moment', moment_map),
    ('haar', haar_map),
    ('sobel', sobel_map),
    ('hu', hu_map),
    ('zernike', zernike_map),
    ])

# Methods that consider a minimum bounding box of selected voxels.
METHODS_MBB = OrderedDict([
    ('stats_mbb', stats_mbb),
    ('glcm_mbb', glcm_mbb),
    ('haralick_mbb', haralick_mbb),
    ('sobel_mbb', sobel_mbb),
    ])

# Methods that consider all selected voxels.
METHODS_ALL = OrderedDict([
    ('stats_all', stats_mbb),  # Use the same mbb function.
    ])

_METHODS = OrderedDict([
    ('stats', stats_map),
    ('glcm', glcm_map),
    ('haralick', haralick_map),
    ('lbp', lbp_freq_map),
    ('hog', hog_map),
    ('gabor', gabor_map),
    ('moment', moment_map),
    ('haar', haar_map),
    ('sobel', sobel_map),
    ('hu', hu_map),
    ('zernike', zernike_map),
    ('stats_mbb', stats_mbb),
    ('glcm_mbb', glcm_mbb),
    ('haralick_mbb', haralick_mbb),
    ('sobel_mbb', sobel_mbb),
    ('stats_all', stats_mbb),  # Use the same mbb function.
    ])


def get_texture_all(img, call, mask):
    feats, names = call(img, mask=mask)
    tmap = np.array(feats, ndmin=4)
    return tmap, names


def get_texture_mbb(img, call, mask):
    tmap = None
    for i, (img_slice, mask_slice) in enumerate(zip(img, mask)):
        if np.count_nonzero(mask_slice):
            feats, names = call(img_slice, mask=mask_slice)
            if tmap is None:
                tmap = np.zeros((len(img), 1, 1, len(names)))
            tmap[i, 0, 0, :] = feats
    return tmap, names


def get_texture_map(img, call, winsize, mask):
    tmap = None
    for i, (img_slice, mask_slice) in enumerate(zip(img, mask)):
        if np.count_nonzero(mask_slice):
            feats, names = call(img_slice, winsize, mask=mask_slice)
            if tmap is None:
                tmap = np.zeros(img.shape+(len(names),))
            feats = np.rollaxis(feats, 0, 3)
            tmap[i, :, :, :] = feats
    return tmap, names


def get_texture(img, method, winsize, mask, avg=False):
    """General texture map layer."""
    assert img.ndim == 3, img.ndim
    if mask is not None:
        assert mask.dtype == bool
        assert img.shape == mask.shape, (img.shape, mask.shape)
    call = _METHODS[method]
    if method.endswith('_all'):
        tmap, names = get_texture_all(img, call, mask)
        if not avg:
            raise NotImplementedError()
    elif method.endswith('_mbb'):
        tmap, names = get_texture_mbb(img, call, mask)
        if avg:
            tmap = np.mean(tmap, axis=0, keepdims=True)
        else:
            raise NotImplementedError()
    else:
        tmap, names = get_texture_map(img, call, winsize, mask)
        if avg:
            tmap = tmap[mask, :]
            tmap = np.mean(tmap, axis=0)
            tmap.shape = (1, 1, 1) + tmap.shape
        else:
            # Fill background with NaN.
            tmap[-mask] = np.nan
    names = ['{w}-{n}'.format(w=winsize, n=n) for n in names]
    return tmap, names
