"""Texture analysis."""

import collections

import numpy as np
import skimage
from skimage.feature import greycomatrix, greycoprops
from skimage.util import view_as_windows

import dwi.util

PROPNAMES = ['contrast', 'dissimilarity', 'homogeneity', 'energy',
        'correlation', 'ASM']
EPSILON = 1e-6

def firstorder(img):
    """Get first-order statistics."""
    d = collections.OrderedDict()
    d['mean'] = np.mean(img)
    d['stddev'] = np.std(img)
    d['range'] = np.max(img) - np.min(img)
    d.update(dwi.util.fivenumd(img))
    return d

def get_coprops_img(img, propnames=PROPNAMES):
    """Get co-occurrence matrix texture properties ower an image."""
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(img, distances, angles, 256, symmetric=True,
            normed=True)
    keys = propnames
    values = [np.mean(greycoprops(glcm, p)) for p in propnames]
    d = collections.OrderedDict((k, v) for k, v in zip(keys, values))
    return d

def get_coprops(windows):
    props = np.zeros((len(windows), len(PROPNAMES)))
    for i, win in enumerate(windows):
        #win = skimage.img_as_ubyte(win)
        #if win.min() == 0:
        #    props[i].fill(0)
        #    continue
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = greycomatrix(win, distances, angles, 256, symmetric=True,
                normed=True)
        for j, propname in enumerate(PROPNAMES):
            a = greycoprops(glcm, propname)
            props[i,j] = np.mean(a)
    return props

def get_texture_pmap(img, win_step):
    pmap = np.zeros((len(PROPNAMES)+1, img.shape[0]/win_step+1,
        img.shape[1]/win_step+1))
    windows = view_as_windows(img, (5,5), step=win_step)
    #props = get_coprops(windows.reshape(-1,5,5))
    #props.shape = (windows.shape[0], windows.shape[1], len(PROPNAMES))
    for i, row in enumerate(windows):
        for j, win in enumerate(row):
            if win.min() > 0:
                v = get_coprops([win])[0]
            else:
                v = 0
            pmap[0,i,j] = np.median(win)
            pmap[1:,i,j] = v
    return pmap

def get_lbp(img, winsize=3, neighbours=8, radius=1, roinv=1, uniform=1):
    """Calculate local binary patterns (LBP)."""
    import lbp
    return lbp.lbp(img, neighbours, radius, roinv, uniform)

def get_lbp_freqs(img, winsize=3, neighbours=8, radius=1, roinv=1, uniform=1):
    """Calculate local binary pattern (LBP) frequencies."""
    import lbp
    lbp_data = lbp.lbp(img, neighbours, radius, roinv, uniform)
    lbp_freq_data, n_patterns = lbp.get_freqs(lbp_data, winsize, neighbours,
            roinv, uniform)
    return lbp_data, lbp_freq_data, n_patterns

def lbpf_dist(hist1, hist2, method='chi-squared', eps=EPSILON):
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

def get_gabor_features(img, sigmas=[1, 3], freqs=[0.25, 0.4]):
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

def get_gabor_features_d(img, sigmas=[1, 3], freqs=[0.1, 0.25, 0.4]):
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
