"""Texture analysis."""

from __future__ import division
import collections
import itertools

import numpy as np
import scipy as sp
import skimage
import skimage.feature

import dwi.util

def normalize(pmap, levels=128):
    """Normalize images within given range and convert to byte maps with given
    number of graylevels."""
    import skimage
    import skimage.exposure
    #in_range = (0, 0.03)
    in_range = (0, 0.005)
    #in_range = (0, 0.002)
    #in_range = (pmap.min(), pmap.max())
    #from scipy.stats import scoreatpercentile
    #in_range = (0, scoreatpercentile(pmap, 99.8))
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
    d = collections.OrderedDict()
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

def glcm_props(img, names=PROPNAMES, distances=[1,2,3,4], ignore_zeros=False):
    """Grey-level co-occurrence matrix (GLCM) texture features averaged over 4
    directions (6 features provided by scikit-image)."""
    from skimage.feature import greycomatrix, greycoprops
    assert img.ndim == 2
    assert img.dtype == np.ubyte
    distances = [x for x in distances if x <= min(img.shape)-1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    levels = img.max() + 1
    glcm = greycomatrix(img, distances, angles, levels, symmetric=True,
            normed=True)
    if ignore_zeros and np.min(img) == 0:
        # Drop information on the first grey-level if it's zero (background).
        glcm = glcm[1:,1:,...]
    d = collections.OrderedDict()
    for name in names:
        feats = greycoprops(glcm, name) # Returns array (distance, angle).
        feats = np.mean(feats, axis=1) # Average over angles.
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
    """Haralick texture features averaged over 4 directions (14 features
    provided by mahotas)."""
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
    """Local Binary Pattern (LBP) frequency histogram map."""
    import lbp
    lbp_data = lbp.lbp(img, neighbours, radius, roinv, uniform)
    lbp_freq_data, n_patterns = lbp.get_freqs(lbp_data, winsize, neighbours,
            roinv, uniform)
    return lbp_data, lbp_freq_data, n_patterns

def lbp_freq_map(img, winsize, neighbours=8, radius=None, mask=None):
    """Local Binary Pattern (LBP) frequency histogram map.
    
    Note: mask parameter is not used: the feature map is calculated for whole
    image."""
    if radius is None:
        radius = winsize // 2
    _, freqs, n = lbp_freqs(img, winsize, neighbours=neighbours, radius=radius)
    output = np.rollaxis(freqs, -1)
    names = ['lbp({r},{i})'.format(r=radius, i=i) for i in range(n)]
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

def gabor(img, sigmas=[1, 2, 3], freqs=[0.1, 0.2, 0.3, 0.4]):
    """Gabor features averaged over 4 directions."""
    thetas = [np.pi/4*i for i in range(4)]
    shape = len(thetas), len(sigmas), len(freqs)
    feats = np.zeros(shape + (2,), dtype=np.double)
    for i, j, k in np.ndindex(shape):
        t, s, f = thetas[i], sigmas[j], freqs[k]
        kwargs = dict(frequency=f, theta=t, sigma_x=s, sigma_y=s)
        real, _ = skimage.filter.gabor_filter(img, **kwargs)
        feats[i,j,k,0] = real.mean()
        feats[i,j,k,1] = real.var()
    feats = np.mean(feats, axis=0) # Average over directions.
    d = collections.OrderedDict()
    for (i, j, k), value in np.ndenumerate(feats):
        key = sigmas[i], freqs[j], ('mean', 'var')[k]
        d[key] = value
    return d

def gabor_map(img, winsize, sigmas=[1, 2, 3], freqs=[0.1, 0.2, 0.3, 0.4],
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
    """Histogram of Gradients (HoG) averaged over directions."""
    kwargs = dict(
            orientations=8,
            pixels_per_cell=img.shape,
            cells_per_block=(1,1),
            normalise=True,
            )
    feats = skimage.feature.hog(img, **kwargs)
    return np.mean(feats)

def hog_map(img, winsize, mask=None, output=None):
    """Histogram of Gradients (HoG) texture feature map."""
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
    nc = lambda pos: (pos-center)/(width/2) # Normalized coordinates [-1,1]
    f = lambda m, n: img[m,n] * nc(m)**p * nc(n)**q
    a = np.fromfunction(f, img.shape, dtype=np.int)
    moment = a.sum()
    return moment

def moments(img, max_order=2):
    """Image moments of order up to p+q <= max_order."""
    r = range(max_order+1)
    tuples = (t for t in itertools.product(r, r) if sum(t) <= max_order)
    d = collections.OrderedDict(((p, q), moment(img, p, q)) for p, q in tuples)
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
    coeffs = [
            a[:h,:w], a[:h,w:],
            a[h:,:w], a[h:,w:],
            ]
    coeffs = [sp.ndimage.interpolation.zoom(l, 2.) for l in coeffs]
    return coeffs

def haar_levels(img, nlevels=4, drop_approx=False):
    """Multi-level Haar wavelet transform."""
    levels = []
    for _ in range(nlevels):
        coeffs = haar(img)
        levels.append(coeffs)
        img = coeffs[0] # Set source for next iteration step.
    if drop_approx:
        levels = [l[1:] for l in levels]
    return levels

def haar_features(win):
    """Haar texture features of a single level."""
    d = collections.OrderedDict()
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
            names += [s.format(level=i+1, coeff=j+1, feat=k) for k in feats.keys()]
    output.shape = (-1,) + levels[0][0].shape
    return output, names

# Sobel.

def sobel(img, mask=None):
    """Sobel edge descriptor."""
    import skimage
    import skimage.filter
    output = skimage.filter.sobel(img, mask=mask)
    return output

def sobel_map(img, winsize=None, mask=None):
    """Sobel edge descriptor map.
    
    Parameter winsize is not used, it is there for API compatibility."""
    output = np.array([sobel(img), sobel(img, mask=mask)])
    names = ['sobel', 'sobel_mask']
    return output, names

def sobel_mbb(img, mask):
    """Sobel edge descriptor map."""
    output = [np.mean(sobel(img, mask=mask))]
    names = ['sobel']
    return output, names
