"""Texture analysis.

Texture methods are of three possible types: 'map' produces a texture map by
using a sliding window, 'mbb' uses selected voxels to produce a single value
for each slice, and 'all' produces a single value for selected voxels from all
slices.

Output type can be one of the following: 'map' just returns the map or spreads
the single values over all selected voxels, 'mean' and 'median' return just the
single values or reduces the map into a single average value.

Scikit-image and Mahotas libraries are used for the calculations.
"""

from __future__ import absolute_import, division, print_function
from collections import OrderedDict

import numpy as np
import scipy as sp

import dwi.hdf5
import dwi.util
from dwi.texture_skimage import (glcm_map, glcm_mbb, lbp_freq_map, hog_map,
                                 gabor_map, sobel_map, hu_map)
from dwi.texture_mahotas import haar_map, zernike_map
# from dwi.texture_jp import lbp_freq_map


DTYPE = np.float32  # Type used for storing texture features.


def abbrev(name):
    """Abbreviate multiword feature name."""
    if ' ' in name:
        name = ''.join(word[0] for word in name.split())
    return name


# Basic statistical features


def stats(img):
    """Statistical texture features that don't consider spatial relations."""
    img = np.asanyarray(img)
    d = OrderedDict()
    d['mean'] = np.mean(img)
    d['stddev'] = np.std(img)
    d['range'] = np.max(img) - np.min(img)
    d.update(dwi.util.fivenumd(img))
    for i in range(1, 10):
        d['decile%i' % i] = np.percentile(img, i*10)
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
            output = np.zeros((len(names),) + img.shape, dtype=DTYPE)
        for i, name in enumerate(names):
            output[(i,) + pos] = d[name]
    names = ['stats({})'.format(n) for n in names]
    return output, names


def stats_mbb(img, mask):
    """Statistical texture features unified over a masked area."""
    # TODO: Add area size?
    voxels = img[mask]
    feats = stats(voxels)
    output = feats.values()
    names = ['stats({})'.format(k) for k in feats.keys()]
    return output, names


# General API for texture map.


METHODS = OrderedDict([
    # Methods that consider an n*n window.
    ('stats', stats_map),
    ('glcm', glcm_map),
    # ('haralick', haralick_map),
    ('lbp', lbp_freq_map),
    ('hog', hog_map),
    ('gabor', gabor_map),
    # ('gaboralt', gabor_map_alt),
    # ('moment', moment_map),
    ('haar', haar_map),
    ('hu', hu_map),
    ('zernike', zernike_map),
    ('sobel', sobel_map),
    # Methods that consider a minimum bounding box of selected voxels.
    ('stats_mbb', stats_mbb),
    ('glcm_mbb', glcm_mbb),
    # ('haralick_mbb', haralick_mbb),
    # Methods that consider all selected voxels.
    ('stats_all', stats_mbb),  # Use the same mbb function.
    ])


def get_texture_all(img, call, mask):
    feats, names = call(img, mask=mask)
    tmap = np.empty(img.shape + (len(names),), dtype=DTYPE)
    tmap.fill(np.nan)
    tmap[mask, :] = feats
    return tmap, names


def get_texture_mbb(img, call, mask):
    tmap = None
    for i, (img_slice, mask_slice) in enumerate(zip(img, mask)):
        if np.count_nonzero(mask_slice):
            feats, names = call(img_slice, mask=mask_slice)
            if tmap is None:
                tmap = np.empty(img.shape + (len(names),), dtype=DTYPE)
                tmap.fill(np.nan)
            tmap[i, mask_slice, :] = feats
    return tmap, names


def get_texture_map(img, call, winsize, mask, path=None):
    tmap = None
    for i, (img_slice, mask_slice) in enumerate(zip(img, mask)):
        if np.count_nonzero(mask_slice):
            feats, names = call(img_slice, winsize, mask=mask_slice)
            if tmap is None:
                shape = img.shape + (len(names),)
                if path is None:
                    # tmap = np.zeros(shape, dtype=DTYPE)
                    tmap = np.full(shape, np.nan, dtype=DTYPE)
                else:
                    tmap = dwi.hdf5.create_hdf5(path, shape, DTYPE,
                                                fillvalue=np.nan)
            feats = np.rollaxis(feats, 0, 3)
            feats[-mask_slice, :] = np.nan  # Fill background with NaN.
            tmap[i, :, :, :] = feats
    return tmap, names


def get_texture(img, method, winspec, mask, avg=False, path=None):
    """General texture map layer."""
    assert img.ndim == 3, img.ndim
    if mask is not None:
        assert mask.dtype == bool
        assert img.shape == mask.shape, (img.shape, mask.shape)
    call = METHODS[method]
    if winspec == 'all':
        assert method.endswith('_all')
        tmap, names = get_texture_all(img, call, mask)
        assert tmap.shape[-1] == len(names), (tmap.shape[-1], len(names))
        if avg:
            # It's all the same value.
            # Feeding np.nanmean whole image hogs too much memory, circumvent.
            tmap = np.array([np.nanmean(x) for x in np.rollaxis(tmap, -1)],
                            dtype=DTYPE)
            tmap.shape = 1, 1, 1, len(names)
    elif winspec == 'mbb':
        assert method.endswith('_mbb')
        tmap, names = get_texture_mbb(img, call, mask)
        assert tmap.shape[-1] == len(names), (tmap.shape[-1], len(names))
        if avg:
            # Take average of each slice; slice-wise they are the same value.
            # Feeding np.nanmean whole image hogs too much memory, circumvent.
            a = np.empty((len(tmap), len(names)), dtype=DTYPE)
            for s, p in np.ndindex(a.shape):
                a[s, p] = np.nanmean(tmap[s, :, :, p])
            tmap = a
            tmap = np.nanmean(tmap, axis=0)
            tmap.shape = 1, 1, 1, len(names)
    else:
        tmap, names = get_texture_map(img, call, int(winspec), mask, path=path)
        assert tmap.shape[-1] == len(names), (tmap.shape[-1], len(names))
        if avg:
            # Take average of all selected voxels.
            tmap = tmap[mask, :]
            assert tmap.ndim == 2
            tmap = np.mean(tmap, axis=0)
            tmap.shape = 1, 1, 1, len(names)
    names = ['{w}-{n}'.format(w=winspec, n=n) for n in names]
    return tmap, names
