"""Texture analysis."""

from __future__ import division
import collections
import itertools

import numpy as np
import scipy as sp
import skimage
import skimage.feature

import dwi.util

def normalize(pmap):
    """Normalize images within given range and convert to byte maps."""
    import skimage
    import skimage.exposure
    #in_range = (0, 0.03)
    in_range = (0, 0.005)
    #in_range = (0, 0.002)
    #in_range = (pmap.min(), pmap.max())
    pmap = skimage.exposure.rescale_intensity(pmap, in_range=in_range)
    pmap = skimage.img_as_ubyte(pmap)
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
    return output, names

# Grey-Level Co-Occurrence Matrix (GLCM) features

PROPNAMES = 'contrast dissimilarity homogeneity energy correlation ASM'.split()

def glcm_props(img, names=PROPNAMES):
    """Grey-level co-occurrence matrix (GLCM) texture features averaged over 4
    directions (6 features provided by scikit-image)."""
    from skimage.feature import greycomatrix, greycoprops
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    levels = 256
    glcm = greycomatrix(img, distances, angles, levels, symmetric=True,
            normed=True)
    keys = names
    values = [np.mean(greycoprops(glcm, p)) for p in names]
    d = collections.OrderedDict((k, v) for k, v in zip(keys, values))
    return d

def glcm_map(img, winsize, names=PROPNAMES, mask=None, output=None):
    """Grey-level co-occurrence matrix (GLCM) texture feature map."""
    img = normalize(img)
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        d = glcm_props(win, names)
        if output is None:
            output = np.zeros((len(names),) + img.shape)
        for i, name in enumerate(names):
            output[(i,) + pos] = d[name]
    return output, names

#def coprops(windows):
#    # XXX Obsolete
#    from skimage.feature import greycomatrix, greycoprops
#    props = np.zeros((len(windows), len(PROPNAMES)))
#    for i, win in enumerate(windows):
#        #win = skimage.img_as_ubyte(win)
#        #if win.min() == 0:
#        #    props[i].fill(0)
#        #    continue
#        distances = [1]
#        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
#        glcm = greycomatrix(win, distances, angles, 256, symmetric=True,
#                normed=True)
#        for j, propname in enumerate(PROPNAMES):
#            a = greycoprops(glcm, propname)
#            props[i,j] = np.mean(a)
#    return props
#
#def texture_pmap(img, win_step):
#    # XXX Obsolete
#    from skimage.util import view_as_windows
#    pmap = np.zeros((len(PROPNAMES)+1, img.shape[0]/win_step+1,
#        img.shape[1]/win_step+1))
#    windows = view_as_windows(img, (5,5), step=win_step)
#    #props = coprops(windows.reshape(-1,5,5))
#    #props.shape = (windows.shape[0], windows.shape[1], len(PROPNAMES))
#    for i, row in enumerate(windows):
#        for j, win in enumerate(row):
#            if win.min() > 0:
#                v = coprops([win])[0]
#            else:
#                v = 0
#            pmap[0,i,j] = np.median(win)
#            pmap[1:,i,j] = v
#    return pmap

def haralick(img):
    """Haralick texture features averaged over 4 directions (14 features
    provided by mahotas)."""
    import mahotas
    a = mahotas.features.texture.haralick(img, compute_14th_feature=True)
    a = np.mean(a, axis=0)
    return a, mahotas.features.texture.haralick_labels

def haralick_map(img, winsize, mask=None, output=None):
    """Haralick texture feature map."""
    img = normalize(img)
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats, names = haralick(win)
        if output is None:
            output = np.zeros((len(names),) + img.shape)
        for i, v in enumerate(feats):
            output[(i,) + pos] = v
    names = ['haralick{i}-{n}'.format(i=i+1, n=abbrev(n)) for i, n in
            enumerate(names)]
    return output, names

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
    
    Note: mask parameter is not used."""
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
    """Histogram of Gradients (HoG) texture features."""
    # TODO Average over directions
    kwargs = dict(
            orientations=2,
            pixels_per_cell=(2,2),
            cells_per_block=(2,2),
            normalise=True,
            )
    feats = skimage.feature.hog(img, **kwargs)
    return feats

def hog_map(img, winsize, mask=None, output=None):
    """Histogram of Gradients (HoG) texture feature map."""
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        feats = hog(win)
        if output is None:
            output = np.zeros((len(feats),) + img.shape)
        for i, v in enumerate(feats):
            output[(i,) + pos] = v
    names = ['hog({})'.format(i) for i in range(len(feats))]
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
    levels = np.array([
            a[:h,:w], a[:h,w:],
            a[h:,:w], a[h:,w:],
            ])
    return levels

def haar_level_features(win):
    """Haar texture features of a single level."""
    d = collections.OrderedDict()
    d['aav'] = np.mean(np.abs(win))
    d['std'] = np.std(win)
    # TODO: Uses only 4 corner pixels.
    a = win[0,:] - win[-1,:]
    d['vert'] = np.mean(np.abs(a))
    a = win[:,0] - win[:,-1]
    d['horz'] = np.mean(np.abs(a))
    a = [win[0,0] - win[-1,-1], win[0,-1] - win[-1,0]]
    d['diag'] = np.mean(np.abs(a))
    return d

def haar_features(img):
    """Haar texture features."""
    levels = haar(img)
    d = collections.OrderedDict()
    for i, level in enumerate(levels):
        feats = haar_level_features(level)
        for k, v in feats.iteritems():
            d[(i, k)] = v
    return d

def haar_map(img, winsize, mask=None, output=None):
    """Haar texture feature map."""
    levels = haar(img)
    levels = sp.ndimage.interpolation.zoom(levels, (1,2,2))
    names = []
    for i, level in enumerate(levels):
        for pos, win in dwi.util.sliding_window(level, winsize, mask):
            feats = haar_level_features(win)
            if output is None:
                output = np.zeros((len(levels)*len(feats),) + level.shape)
            for j, v in enumerate(feats.values()):
                output[(i*len(feats)+j,) + pos] = v
        names += ['haar(%i,%s)' % (i, k) for k in feats.keys()]
    return output, names
