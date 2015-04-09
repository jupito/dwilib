"""Texture analysis."""

from __future__ import division
import collections
import itertools

import numpy as np
import scipy as sp
import skimage

import dwi.util

# Basic statistical features

def stats(img):
    """Get basic statistical features."""
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

def stats_map(img, winsize, names=None, output=None):
    """Get basic statistical texture feature map."""
    for pos, win in dwi.util.sliding_window(img, winsize):
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
    """Get grey-level co-occurrence matrix texture properties over an image."""
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

def glcm_map(img, winsize, names=PROPNAMES, output=None):
    """Grey-level co-occurrence matrix texture feature map."""
    for pos, win in dwi.util.sliding_window(img, winsize):
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
    """Haralick texture features (14) averaged over directions."""
    import mahotas
    a = mahotas.features.texture.haralick(img, compute_14th_feature=True)
    a = np.mean(a, axis=0)
    return a, mahotas.features.texture.haralick_labels

def haralick_map(img, winsize, output=None):
    """Haralick texture feature map."""
    for pos, win in dwi.util.sliding_window(img, winsize):
        feats, names = haralick(win)
        if output is None:
            output = np.zeros((len(names),) + img.shape)
        for i, v in enumerate(feats):
            output[(i,) + pos] = v
    return output, names

# Local Binary Pattern (LBP) features

def lbp_freqs(img, winsize=3, neighbours=8, radius=1, roinv=1, uniform=1):
    """Calculate local binary pattern (LBP) frequency map."""
    import lbp
    lbp_data = lbp.lbp(img, neighbours, radius, roinv, uniform)
    lbp_freq_data, n_patterns = lbp.get_freqs(lbp_data, winsize, neighbours,
            roinv, uniform)
    return lbp_data, lbp_freq_data, n_patterns

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

def gabor_features(img, sigmas=[1, 3], freqs=[0.25, 0.4]):
    # XXX: Obsolete
    thetas = [np.pi/4*i for i in range(4)]
    shape = len(thetas), len(sigmas), len(freqs)
    feats = np.zeros(shape + (2,), dtype=np.double)
    for i, j, k in np.ndindex(shape):
        t, s, f = thetas[i], sigmas[j], freqs[k]
        kwargs = dict(frequency=f, theta=t, sigma_x=s, sigma_y=s)
        real, _ = skimage.filter.gabor_filter(img, **kwargs)
        feats[i,j,k,0] = real.mean()
        feats[i,j,k,1] = real.var()
    #feats = feats.reshape((-1,2))
    feats = np.mean(feats, axis=0) # Average directions.
    return feats

def gabor_features_d(img, sigmas=[1, 3], freqs=[0.1, 0.25, 0.4]):
    """Get Gabor features."""
    thetas = [np.pi/4*i for i in range(4)]
    shape = len(thetas), len(sigmas), len(freqs)
    feats = np.zeros(shape + (2,), dtype=np.double)
    d = collections.OrderedDict()
    for i, j, k in np.ndindex(shape):
        t, s, f = thetas[i], sigmas[j], freqs[k]
        kwargs = dict(frequency=f, theta=t, sigma_x=s, sigma_y=s)
        real, _ = skimage.filter.gabor_filter(img, **kwargs)
        feats[i,j,k,0] = real.mean()
        feats[i,j,k,1] = real.var()
        d[(t/np.pi,s,f,'mean')] = real.mean()
        d[(t/np.pi,s,f,'var')] = real.var()
    feats_distavg = np.mean(feats, axis=0) # Average directions.
    return d

# Histogram of Oriented Gradients (HOG)

def hog(img):
    return skimage.feature.hog(img, orientations=2, pixels_per_cell=(2,2),
            cells_per_block=(2,2), normalise=True)

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

def moments(img, max_sum=2):
    """Image moments up to p+q <= max_sum."""
    r = range(max_sum+1)
    tuples = (t for t in itertools.product(r, r) if sum(t) <= max_sum)
    d = collections.OrderedDict(((p, q), moment(img, p, q)) for p, q in tuples)
    return d

# Haar transformation

def haar(img):
    """Haar"""
    import mahotas
    assert img.ndim == 2
    #assert img.shape[0] % 2 == img.shape[1] % 2 == 0
    # Prune possible odd borders.
    newshape = [x-x%2 for x in img.shape]
    img = img[:newshape[0], :newshape[1]]
    a = mahotas.haar(img)
    h, w = [x//2 for x in a.shape]
    levels = [
            a[:h,:w], a[:h,w:],
            a[h:,:w], a[h:,w:],
        ]
    return levels

def haar_features(img):
    """Haar features."""
    levels = haar(img)
    d = collections.OrderedDict()
    for i, l in enumerate(levels):
        assert l.shape == (2, 2), 'TODO: Required shape for now'
        d[(i, 'aav')] = np.mean(np.abs(l))
        d[(i, 'std')] = np.std(l)
        a = l[0,:] - l[1,:]
        d[(i, 'vert')] = np.mean(np.abs(a))
        a = l[:,0] - l[:,1]
        d[(i, 'horz')] = np.mean(np.abs(a))
        a = [l[0,0]-l[1,1], l[0,1]-l[1,0]]
        d[(i, 'diag')] = np.mean(np.abs(a))
    return d
