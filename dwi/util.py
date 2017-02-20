"""Utility functionality."""

from __future__ import absolute_import, division, print_function
from functools import reduce, total_ordering
import json
import logging

import numpy as np
from scipy import spatial
import skimage.exposure


@total_ordering
class ImageMode(object):
    """Image mode identifier."""
    def __init__(self, value, sep='-'):
        """Initialize with a string or a sequence."""
        if isstring(value):
            value = value.split(sep)
        self.value = tuple(value)
        self.sep = sep

    def __iter__(self):
        return iter(self.value)

    def __getitem__(self, key):
        return self.__class__(self.value[key])

    def __len__(self):
        return len(self.value)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join(self))

    def __str__(self):
        return self.sep.join(iter(self))

    def __lt__(self, other):
        return tuple(self) < tuple(ImageMode(other))

    def __eq__(self, other):
        return tuple(self) == tuple(ImageMode(other))

    def __hash__(self):
        return hash(tuple(self))

    # def __add__(self, other):
    #     """Append a component."""
    #     return self.__class__(self.value + (other,))

    # def __sub__(self, other):
    #     """Remove a tailing component."""
    #     v = self.value
    #     if v[-1] == other:
    #         v = v[:-1]
    #     return self.__class__(v)


def get_loglevel(name):
    """Return the numeric correspondent of a logging level name."""
    try:
        return getattr(logging, name.upper())
    except AttributeError:
        raise ValueError('Invalid log level: {}'.format(name))


def iterable(x):
    """Tell whether an object is iterable or not."""
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True


def isstring(x):
    """Check for string-ness."""
    try:
        return isinstance(x, basestring)
    except NameError:
        return isinstance(x, str)


def all_equal(a):
    """Tell whether all members of (multidimensional) array are equal, while
    ignoring nan values.
    """
    a = np.asarray(a)
    return np.nanmin(a) == np.nanmax(a)


def pairs(seq):
    """Return sequence split in two, each containing every second item."""
    if len(seq) % 2:
        raise Exception('Sequence length not even.')
    return seq[0::2], seq[1::2]


def get_indices(seq, val):
    """Return indices of elements containing given value in a sequence."""
    r = []
    for i, v in enumerate(seq):
        if v == val:
            r.append(i)
    return r


def crop_image(image, subwindow, onebased=False):
    """Get a view of image subwindow defined as Python-like start:stop
    indices.
    """
    if onebased:
        subwindow = [i-1 for i in subwindow]
    z1, z2, y1, y2, x1, x2 = subwindow
    view = image[z1:z2, y1:y2, x1:x2]
    return view


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
        raise Exception('Invalid window shape: {}'.format(winshape))
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
    return tuple(np.nanpercentile(a, (0, 25, 50, 75, 100)))


def fivenums(a):
    """Return the Tukey five-number summary as a formatted string."""
    return '({:{f}}, {:{f}}, {:{f}}, {:{f}}, {:{f}})'.format(*fivenum(a),
                                                             f='.4g')


def distance(a, b):
    """Return the Euclidean distance of two vectors."""
    return spatial.distance.euclidean(a, b)


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
    a = a.astype(np.bool)
    return a


def dump_json(obj, separators=(', ', ': '), sort_keys=False):
    """Dump object into a JSON string."""
    if sort_keys is None:
        sort_keys = not isinstance(obj, OrderedDict)  # Let it sort itself.
    return json.dumps(obj, separators=separators, sort_keys=sort_keys)


def normalize(pmap, mode):
    """Normalize images within mode-specific range."""
    if mode in ('DWI-Mono-ADCm', 'DWI-Kurt-ADCk'):
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
    # logging.info('Normalizing: %s, %s', mode, in_range)
    pmap = pmap.astype(np.float32, copy=False)
    pmap = np.nan_to_num(pmap)
    pmap = skimage.exposure.rescale_intensity(pmap, in_range=in_range)
    return pmap


def quantize(img, levels=32, dtype=np.uint8):
    """Uniform quantization from float [0, 1] to int [0, levels-1]."""
    img = np.asarray(img)
    assert np.issubsctype(img, np.floating), img.dtype
    assert np.all(img >= 0) and np.all(img <= 1), (img.min(), img.max())
    # img = skimage.img_as_ubyte(img)
    # img //= int(round(256 / levels))
    return (img * levels).clip(0, levels-1).astype(dtype)
