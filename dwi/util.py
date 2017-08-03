"""Utility functionality."""

import json
import logging
import os
import platform
from collections import OrderedDict
from functools import reduce

import numpy as np

from scipy import spatial
from scipy.ndimage import interpolation

from skimage import exposure


def get_loglevel(name):
    """Return the numeric correspondent of a logging level name."""
    try:
        return getattr(logging, name.upper())
    except AttributeError:
        raise ValueError('Invalid log level: {}'.format(name))


def abbrev(name):
    """Abbreviate multiword feature name."""
    if ' ' in name:
        name = ''.join(word[0] for word in name.split())
    return name


def all_equal(a):
    """Tell whether all members of (multidimensional) array are equal, while
    ignoring nan values.
    """
    a = np.asarray(a)
    return np.nanmin(a) == np.nanmax(a)


def crop_image(image, subwindow, onebased=False):
    """Get a view of image subwindow defined as Python-like start:stop
    indices.
    """
    if onebased:
        subwindow = [i-1 for i in subwindow]
    z1, z2, y1, y2, x1, x2 = subwindow
    return image[z1:z2, y1:y2, x1:x2]


def select_subwindow(image, subwindow, onebased=False):
    """Get a copy of image with only a subwindow selected and everything else
    set to nan.
    """
    if onebased:
        subwindow = [i-1 for i in subwindow]
    z1, z2, y1, y2, x1, x2 = subwindow
    mask = np.zeros_like(image, dtype=np.bool)
    mask[z1:z2, y1:y2, x1:x2] = True
    copy = image.copy()
    copy[-mask] = np.nan
    return copy


def normalize_sequence(value, rank):
    """For scalar input, duplicate it into a sequence of rank length. For
    sequence input, check for correct length.
    """
    # Like e.g. scipy.ndimage.zoom() -> _ni_support._normalize_sequence().
    try:
        lst = list(value)
    except TypeError:
        lst = [value] * rank
    if len(lst) != rank:
        raise ValueError('Invalid sequence length.')
    return lst


def sliding_window(a, winshape, mask=None):
    """Multidimensional sliding window iterator with arbitrary window shape.

    Yields window origin (center) and view to window. Window won't overlap
    image border. If a mask array is provided, windows are skipped unless
    origin is selected in mask.
    """
    a = np.asanyarray(a)
    winshape = normalize_sequence(winshape, a.ndim)
    if not all(0 < w <= i for w, i in zip(winshape, a.shape)):
        raise ValueError('Invalid window shape: {}'.format(winshape))
    shape = tuple(i-w+1 for i, w in zip(a.shape, winshape))
    for indices in np.ndindex(shape):
        origin = tuple(i+w//2 for i, w in zip(indices, winshape))
        if mask is None or mask[origin]:
            slices = [slice(i, i+w) for i, w in zip(indices, winshape)]
            window = np.squeeze(a[slices])
            yield origin, window


def bounding_box(array, pad=0):
    """Return the minimum bounding box with optional padding.

    Parameter pad can be a single integer or a sequence of dimensions. It may
    contain infinity for maximum padding.

    The value to leave outside box is nan, if any, otherwise zero.

    Use example:
        mbb = dwi.util.bounding_box(mask)
        slices = [slice(*x) for x in mbb]
        img = img[slices]
    """
    array = np.asanyarray(array)
    pad = normalize_sequence(pad, array.ndim)
    nans = np.isnan(array)
    if np.any(nans):
        array = ~nans
    r = []
    for a, l, p in zip(array.nonzero(), array.shape, pad):
        x = max(min(a)-p, 0)
        y = min(max(a)+1+p, l)
        r.append((x, y))
    return tuple(r)


def bbox(array, pad=0):
    """Like bounding_box() but return slice objects."""
    return tuple(slice(a, b) for a, b in bounding_box(array, pad=pad))


def fivenum(a):
    """Return the Tukey five-number summary (minimum, quartile 1, median,
    quartile 3, maximum), while ignoring nan values.
    """
    return tuple(np.nanpercentile(a, range(0, 101, 25)))


def fivenums(a, fmt='.4g'):
    """Return the Tukey five-number summary as a formatted string."""
    s = '({})'.format(', '.join(['{:{f}}'] * 5))
    return s.format(*fivenum(a), f=fmt)


def distance(a, b, spacing=None):
    """Return the Euclidean distance of two vectors."""
    a = np.asarray(a)
    b = np.asarray(b)
    if spacing is None:
        return spatial.distance.euclidean(a, b)
    spacing = np.asarray(spacing)
    return distance(a * spacing, b * spacing)


def normalize_si_curve(si):
    """Normalize a signal intensity curve (divide all by the first value).

    Note that this function does not manage error cases where the first value
    is zero or the curve rises at some point. See normalize_si_curve_fix().
    """
    assert si.ndim == 1
    si[:] /= si[0]


def normalize_si_curve_fix(si):
    """Normalize a signal intensity curve (divide all by the first value).

    This version handles some error cases. If the first value is zero, all
    values are just set to zero. If any value is higher than the previous one,
    it is set to the same value (curves are never supposed to rise).
    """
    assert si.ndim == 1
    if si[0] == 0:
        si[:] = 0
    else:
        for i in range(1, len(si)):
            if si[i] > si[i-1]:
                si[i] = si[i-1]
        si[:] /= si[0]


def scale(a):
    """Apply feature scaling: bring all values to range [0, 1], while ignoring
    nan values.
    """
    # TODO: Allow in-place, this is a likely memory hog.
    a = np.asanyarray(a)
    mn, mx = np.nanmin(a), np.nanmax(a)
    return (a-mn) / (mx-mn)


def centroid(img):
    """Calculate image centroid, i.e. center of mass, as a tuple of floats.

    NaN values are considered massless.
    """
    img = np.asanyarray(img)
    all_axes = tuple(range(img.ndim))
    centers = []
    for i in all_axes:
        other_axes = tuple(x for x in all_axes if x != i)
        summed = np.nansum(img, axis=other_axes)
        assert summed.ndim == 1
        c = np.nansum([i*x for i, x in enumerate(summed)]) / np.nansum(summed)
        centers.append(c)
    return tuple(centers)


def atleast_nd(n, a):
    """Return a view to array with at least n dimensions. This is
    a generalization of numpy.atleast_{1,2,3}d() except all dimensions are
    added in front.
    """
    a = np.asanyarray(a)
    if a.ndim < n:
        a = a.reshape((1,) * (n-a.ndim) + a.shape)
    return a


def unify_masks(masks):
    """Unify a sequence of masks into one."""
    return reduce(np.maximum, masks)


def asbool(a):
    """Gracefully convert a numeric ndarray to boolean. Round it and clip to
    [0, 1]. (Simply using np.astype(np.bool) does not work.)
    """
    a = np.asanyarray(a)
    a = a.round()
    a.clip(0, 1, out=a)
    a = a.astype(np.bool, copy=False)
    return a


def zoom(image, factor, order=1, **kwargs):
    """Zoom by a factor (float or sequence).

    Note: Scipy's zoom() used here seems to sometimes jam on float16.
    """
    return interpolation.zoom(image, factor, order=order, **kwargs)


def zoom_as_float(image, factor, **kwargs):
    """Convert to float, zoom, convert back. Special boolean handling."""
    typ = image.dtype
    image = image.astype(np.float, copy=False)
    image = zoom(image, factor, **kwargs)
    if typ == np.bool:
        image = asbool(image)
    else:
        image = image.astype(typ, copy=False)
    return image


def dump_json(obj, separators=(', ', ': '), sort_keys=False):
    """Dump object into a JSON string."""
    if sort_keys is None:
        sort_keys = not isinstance(obj, OrderedDict)  # Let it sort itself.
    return json.dumps(obj, separators=separators, sort_keys=sort_keys)


def normalize(pmap, mode):
    """Normalize images within mode-specific range."""
    shortcuts = dict(ADCm='DWI-Mono-ADCm', ADCk='DWI-Kurt-ADCk',
                     K='DWI-Kurt-K')
    mode = shortcuts.get(str(mode), mode)
    if mode == 'DWI':
        in_range = (100, 2500)
    elif mode == 'DWI-b2000':
        in_range = (0, 400)
        # TODO: 5-1a has different scale, could handle in DICOM loader?
        if pmap.max() < 100:
            in_range = tuple(x/100 for x in in_range)
    elif mode in ('DWI-Mono-ADCm', 'DWI-Kurt-ADCk'):
        assert pmap.dtype in [np.float32, np.float64]
        # in_range = (0, 0.005)
        in_range = (0, 0.004)
        # in_range = (0, 0.003)
    elif mode == 'DWI-Kurt-K':
        # in_range = (pmap.min(), pmap.max())
        # in_range = (0, np.percentile(pmap, 99.8))
        # in_range = tuple(np.percentile(pmap, (0, 99.8)))
        # in_range = tuple(np.percentile(pmap, (0.8, 99.2)))
        in_range = (0, 2)
    elif mode in ('T2', 'T2-fitted'):
        in_range = (0, 300)
    elif mode == 'T2w-std':
        in_range = (1, 4095)
    elif mode == 'T2w':
        if pmap.dtype == np.int32:
            # The rescaler cannot handle int32.
            pmap = np.asarray(pmap, dtype=np.int16)
        assert pmap.dtype == np.int16
        in_range = (0, 2000)
    else:
        raise ValueError('Invalid mode: {}'.format(mode))
    logging.debug('Normalizing: %s, %s', mode, in_range)
    pmap = pmap.astype(np.float32, copy=False)
    pmap = np.nan_to_num(pmap)
    pmap = exposure.rescale_intensity(pmap, in_range=in_range)
    return pmap


def quantize(img, levels=32, dtype=np.uint8):
    """Uniform quantization from float [0, 1] to int [0, levels-1]."""
    img = np.asarray(img)
    assert np.issubsctype(img, np.floating), img.dtype
    assert np.all(img >= 0) and np.all(img <= 1), (img.min(), img.max())
    # img = skimage.img_as_ubyte(img)
    # img //= int(round(256 / levels))
    return (img * levels).clip(0, levels-1).astype(dtype)


def cpu_count():
    """Return CPU count, if possible."""
    n = os.cpu_count()
    if n is None:
        raise OSError(None, 'Could not determine CPU count')
    return n


def hostname():
    """Try to return system hostname in a portable fashion."""
    name = platform.uname().node
    if not name:
        raise OSError(None, 'Could not determine hostname')
    return name
