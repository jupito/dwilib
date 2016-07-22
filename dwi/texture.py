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
import logging

import numpy as np
import scipy as sp

import dwi.hdf5
import dwi.util
import dwi.texture_mahotas
import dwi.texture_skimage


def abbrev(name):
    """Abbreviate multiword feature name."""
    if ' ' in name:
        name = ''.join(word[0] for word in name.split())
    return name


# Basic statistical features


def stats(img):
    """Statistical texture features that don't consider spatial relations."""
    # TODO: Consider renaming decileX to percentileX*10.
    # TODO: Consider IQR, MAD, interdecile range, midhinge, trimean, trimmed
    # mean, winsorized mean.
    img = np.asanyarray(img)
    d = OrderedDict()
    d['mean'] = np.mean(img)
    d['stddev'] = np.std(img)
    d['range'] = np.max(img) - np.min(img)
    d.update(dwi.util.fivenumd(img))
    del d['median']  # This will be 'decile5'.
    for i in range(1, 10):
        d['decile%i' % i] = np.percentile(img, i*10)
    d['kurtosis'] = sp.stats.kurtosis(img.ravel())
    d['skewness'] = sp.stats.skew(img.ravel())
    return d


def stats_map(img, winsize, mask=None, output=None):
    """Statistical texture feature map."""
    for pos, win in dwi.util.sliding_window(img, winsize, mask=mask):
        d = stats(win)
        names = d.keys()
        if output is None:
            dtype = dwi.rcParams['texture.dtype']
            output = np.zeros((len(names),) + img.shape, dtype=dtype)
        for i, value in enumerate(d.values()):
            output[(i,) + pos] = value
    names = ['stats({})'.format(x) for x in names]
    return output, names


def stats_mbb(img, mask):
    """Statistical texture features unified over a masked area."""
    # TODO: Add area size?
    voxels = img[mask]
    feats = stats(voxels)
    output = feats.values()
    names = ['stats({})'.format(x) for x in feats.keys()]
    return output, names


# General API for texture map.


METHODS = OrderedDict([
    # Methods that consider an n*n window.
    ('stats', stats_map),
    ('glcm', dwi.texture_skimage.glcm_map),
    # ('haralick', dwi.texture_mahotas.haralick_map),
    ('lbp', dwi.texture_skimage.lbp_freq_map),
    ('hog', dwi.texture_skimage.hog_map),
    ('gabor', dwi.texture_skimage.gabor_map),
    ('gaborold', dwi.texture_skimage.gabor_map_old),
    ('haar', dwi.texture_mahotas.haar_map),
    ('hu', dwi.texture_skimage.hu_map),
    ('zernike', dwi.texture_mahotas.zernike_map),
    ('sobel', dwi.texture_skimage.sobel_map),

    # Methods that consider a minimum bounding box of selected voxels.
    ('stats_mbb', stats_mbb),
    ('glcm_mbb', dwi.texture_skimage.glcm_mbb),
    # ('haralick_mbb', dwi.texture_mahotas.haralick_mbb),

    # Methods that consider all selected voxels.
    ('stats_all', stats_mbb),  # Use the same mbb function.
    ])


def get_texture_all(img, call, mask):
    feats, names = call(img, mask=mask)
    dtype = dwi.rcParams['texture.dtype']
    tmap = np.full(img.shape + (len(names),), np.nan, dtype=dtype)
    tmap[mask, :] = feats
    return tmap, names


def get_texture_mbb(img, call, mask):
    tmap = None
    for i, (img_slice, mask_slice) in enumerate(zip(img, mask)):
        if np.count_nonzero(mask_slice):
            feats, names = call(img_slice, mask=mask_slice)
            if tmap is None:
                dtype = dwi.rcParams['texture.dtype']
                tmap = np.full(img.shape + (len(names),), np.nan, dtype=dtype)
            tmap[i, mask_slice, :] = feats
    return tmap, names


def get_texture_map(img, call, winsize, mask):
    path = dwi.rcParams['texture.path']
    tmap = None
    for i, (img_slice, mask_slice) in enumerate(zip(img, mask)):
        if np.count_nonzero(mask_slice):
            feats, names = call(img_slice, winsize, mask=mask_slice)
            if tmap is None:
                shape = img.shape + (len(names),)
                dtype = dwi.rcParams['texture.dtype']
                if path is None:
                    tmap = np.full(shape, np.nan, dtype=dtype)
                else:
                    s = 'Array is manipulated on disk, it is slow: %s'
                    logging.warning(s, path)
                    tmap = dwi.hdf5.create_hdf5(path, shape, dtype,
                                                fillvalue=np.nan)
            feats = np.rollaxis(feats, 0, 3)
            feats[-mask_slice, :] = np.nan  # Fill background with NaN.
            tmap[i, :, :, :] = feats
    return tmap, names


def get_texture(img, method, winspec, mask):
    """General texture map layer."""
    avg = dwi.rcParams['texture.avg']
    dtype = dwi.rcParams['texture.dtype']
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
                            dtype=dtype)
            tmap.shape = 1, 1, 1, len(names)
    elif winspec == 'mbb':
        assert method.endswith('_mbb')
        tmap, names = get_texture_mbb(img, call, mask)
        assert tmap.shape[-1] == len(names), (tmap.shape[-1], len(names))
        if avg:
            # Take average of each slice; slice-wise they are the same value.
            # Feeding np.nanmean whole image hogs too much memory, circumvent.
            a = np.empty((len(tmap), len(names)), dtype=dtype)
            for s, p in np.ndindex(a.shape):
                a[s, p] = np.nanmean(tmap[s, :, :, p])
            tmap = a
            tmap = np.nanmean(tmap, axis=0)
            tmap.shape = 1, 1, 1, len(names)
    else:
        tmap, names = get_texture_map(img, call, int(winspec), mask)
        assert tmap.shape[-1] == len(names), (tmap.shape[-1], len(names))
        if avg:
            # Take average of all selected voxels.
            tmap = tmap[mask, :]
            assert tmap.ndim == 2
            tmap = np.mean(tmap, axis=0)
            tmap.shape = 1, 1, 1, len(names)
    names = ['{w}-{n}'.format(w=winspec, n=n) for n in names]
    return tmap, names
