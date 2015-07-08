"""Utility functionality."""

from __future__ import division, print_function
import collections
import itertools
import random

import numpy as np
import scipy as sp

def all_equal(a):
    """Tell whether all members of (multidimensional) array are equal."""
    a = np.asarray(a)
    return a.min() == a.max()

def make2d(size, height=None):
    """Turn 1d size into 2d shape by growing the height until it fits."""
    if height:
        assert height <= size
        width = size // height
        if height * width == size:
            return height, width
        else:
            return make2d(size, height+1)
    else:
        return make2d(size, int(np.sqrt(size)))

def fabricate_subwindow(size, height=None):
    """Fabricate a subwindow specification."""
    height, width = make2d(size, height=height)
    return 0, height, 0, width

def combinations(l):
    """Return combinations of list elements."""
    return [x for x in itertools.product(*l)]

def chunks(seq, n):
    """Return sequence as chunks of n elements."""
    if len(seq) % n:
        raise Exception('Sequence length not divisible.')
    return (seq[i:i+n] for i in xrange(0, len(seq), n))

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
    indices."""
    if onebased:
        subwindow = [i-1 for i in subwindow]
    z1, z2, y1, y2, x1, x2 = subwindow
    view = image[z1:z2, y1:y2, x1:x2]
    return view

def subwindow_shape(subwindow):
    """Return subwindow shape."""
    return tuple(b-a for a, b in chunks(subwindow, 2))

def sliding_window(a, winshape, mask=None):
    """Multidimensional sliding window iterator with arbitrary window shape.

    Yields window origin (center) and view to window. Window won't overlap
    image border. If a mask array is provided, windows are skipped unless origin
    is selected in mask.
    """
    if isinstance(winshape, int):
        winshape = (winshape,) * a.ndim # Can give single value for window size.
    if len(winshape) != a.ndim:
        raise Exception('Invalid window dimensionality: {}'.format(winshape))
    if not all(0 < w <= i for w, i in zip(winshape, a.shape)):
        raise Exception('Invalid window shape: {}'.format(winshape))
    a = np.asarray(a)
    shape = tuple(i-w+1 for i, w in zip(a.shape, winshape))
    for indices in np.ndindex(shape):
        origin = tuple(i+w//2 for i, w in zip(indices, winshape))
        if mask is None or mask[origin]:
            slices = [slice(i, i+w) for i, w in zip(indices, winshape)]
            window = np.squeeze(a[slices])
            yield origin, window

def bounding_box(array, pad=0):
    """Return the minimum bounding box with optional padding.

    Parameter pad can be a tuple of each dimension or a single number. It can
    contain infinity for maximum padding.
    """
    if np.isscalar(pad):
        pad = (pad,) * array.ndim
    r = []
    for a, l, p in zip(array.nonzero(), array.shape, pad):
        x = max(min(a)-p, 0)
        y = min(max(a)+1+p, l)
        r.append((x, y))
    #return tuple(map(int, r))
    return tuple(r)

def median(a, axis=None, keepdims=False, dtype=None):
    """Added keepdims parameter for NumPy 1.8 median. See numpy.mean."""
    a = np.asanyarray(a)
    r = np.median(a, axis=axis)
    if keepdims:
        shape = list(a.shape)
        if axis is None:
            r = np.array(r, dtype=a.dtype)
            shape = [1 for _ in shape]
        else:
            shape[axis] = 1
        r.shape = shape
    if dtype:
        r = r.astype(dtype)
    return r

def resample_bootstrap_single(a):
    """Get a bootstrap resampled group for single array."""
    indices = [random.randint(0, len(a)-1) for _ in a]
    return a[indices]

def resample_bootstrap(Y, X):
    """Get a bootstrap resampled group without stratification."""
    indices = [random.randint(0, len(Y)-1) for _ in Y]
    return Y[indices], X[indices]

def resample_bootstrap_stratified(Y, X):
    """Get a bootstrap resampled group with stratification.

    Note that as a side-effect the resulting Y array will be sorted, but that
    doesn't matter because X will be randomized accordingly.
    """
    uniques = np.unique(Y)
    indices = []
    for u in uniques:
        l = get_indices(Y, u)
        l_rnd = [l[random.randint(0, len(l)-1)] for _ in l]
        for v in l_rnd:
            indices.append(v)
    return Y[indices], X[indices]

def fivenum(a):
    """Tukey five-number summary (min, q1, median, q3, max)."""
    from scipy.stats import scoreatpercentile
    q1 = scoreatpercentile(a, 25)
    q3 = scoreatpercentile(a, 75)
    return np.min(a), q1, np.median(a), q3, np.max(a)

def fivenumd(a):
    """Tukey five-number summary (min, q1, median, q3, max)."""
    keys = 'min q1 median q3 max'.split()
    values = fivenum(a)
    d = collections.OrderedDict(zip(keys, values))
    return d

def stem_and_leaf(values):
    """A quick and dirty text mode stem-and-leaf diagram that only uses first
    and second decimal places.
    """
    stems = {}
    for v in sorted(values):
        a = stems.setdefault(int(v*10), [])
        a.append(int((v*10 - int(v*10)) * 10))
    strings = []
    for i in range(11):
        strings.append('%i|%s' % (i, ''.join(map(str, stems.get(i, [])))))
    return strings

def tilde(a):
    """Logical 'not' operator for NumPy objects that behaves like MATLAB
    tilde."""
    typ = a.dtype
    return (~a.astype(bool)).astype(typ)

# Note: These are obsolete. use calculate_roc_auc() instead of roc(), roc_auc().
#
#def roc(truths, scores):
#    """Calculate ROC curve. Based on HM's matlab implementation."""
#    truths = np.array(truths)
#    scores = np.array(scores)
#    indices_sorted = scores.argsort()
#    scores = scores[indices_sorted]
#    truths = truths[indices_sorted]
#    values = np.unique(scores)
#    tp = np.array(values)
#    tn = np.array(values)
#    fp = np.array(values)
#    fn = np.array(values)
#    for i, value in enumerate(values):
#        c = np.ones_like(scores)
#        c[scores >= value] = 0 # Set classifications.
#        tp[i] = sum(c * truths)
#        tn[i] = sum(tilde(c) * tilde(truths))
#        fp[i] = sum(c * tilde(truths))
#        fn[i] = sum(tilde(c) * truths)
#    fpr = fp / (fp+tn)
#    tpr = tp / (tp+fn)
#    acc = np.mean((tp+tn) / (tp+fp+fn+tn))
#    return fpr, tpr, acc
#
#def roc_auc(fpr, tpr):
#    """Calculate ROC AUC from false and true positive rates."""
#    area = 0
#    for i in range(len(fpr))[1:]:
#        area += abs(fpr[i]-fpr[i-1]) * (tpr[i]+tpr[i-1]) / 2
#    return area

def calculate_roc_auc(y, x, autoflip=False):
    """Calculate ROC and AUC from data points and their classifications."""
    import sklearn.metrics
    y = np.asarray(y)
    x = np.asarray(x)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y, x)
    auc = sklearn.metrics.auc(fpr, tpr)
    if autoflip and auc < 0.5:
        fpr, tpr, auc = calculate_roc_auc(y, -x, autoflip=False)
    return fpr, tpr, auc

def bootstrap_aucs(y, x, n=2000):
    """Produce an array of bootstrapped ROC AUCs."""
    aucs = np.zeros(n)
    for i in range(n):
        yb, xb = resample_bootstrap_stratified(y, x)
        _, _, auc = calculate_roc_auc(yb, xb)
        aucs[i] = auc
    return aucs

def compare_aucs(aucs1, aucs2):
    """Compare two arrays of (bootstrapped) ROC AUC values, with the method
    described in pROC software."""
    import scipy.stats
    aucs1 = np.asarray(aucs1)
    aucs2 = np.asarray(aucs2)
    D = aucs1 - aucs2
    z = np.mean(D) / np.std(D)
    p = 1.0 - scipy.stats.norm.cdf(abs(z))
    return np.mean(D), z, p

def ci(x, p=0.05):
    """Confidence interval of a normally distributed array."""
    x = sorted(x)
    l = len(x)
    i1 = int(round((p/2) * l + 0.5))
    i2 = int(round((1-p/2) * l - 0.5))
    ci1 = x[i1]
    ci2 = x[i2]
    return ci1, ci2

def distance(a, b):
    """Return the Euclidean distance of two vectors."""
    return sp.spatial.distance.euclidean(a, b)


def normalize_si_curve(si):
    """Normalize a signal intensity curve (divide all by the first value).
    
    Note that this function does not manage error cases where the first value
    is zero or the curve rises at some point. See normalize_si_curve_fix().
    """
    for i in range(len(si)):
        si[i] /= si[0]

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
        for i in range(len(si)):
            si[i] /= si[0]

def scale(a):
    """Feature scaling. Bring all values to [0, 1] range."""
    a = np.asanyarray(a)
    min, max = a.min(), a.max()
    return (a-min) / (max-min)

def clip_pmap(img, params):
    """Clip pmap's parameter-specific intensity outliers in-place."""
    for i in range(img.shape[-1]):
        if params[i].startswith('ADC'):
            img[...,i].clip(0, 0.002, out=img[...,i])
        elif params[i].startswith('K'):
            img[...,i].clip(0, 2, out=img[...,i])

def clip_outliers(a, min_pc=0, max_pc=99.8, out=None):
    """Clip outliers based on percentiles."""
    from scipy.stats import scoreatpercentile
    a = np.asanyarray(a)
    min_score = scoreatpercentile(a, min_pc)
    max_score = scoreatpercentile(a, max_pc)
    return a.clip(min_score, max_score, out=out)

def add_dummy_feature(X):
    """Add an extra dummy feature to an array of samples."""
    r = np.ones((X.shape[0], X.shape[1]+1), dtype=X.dtype)
    r[:,:-1] = X
    return r

def split_roi(pmaps):
    """Split samples to ROI1 and 2."""
    l = pmaps.shape[1]
    if l > 1:
        pmaps1 = pmaps[:, :l//2, :]
        pmaps2 = pmaps[:, l//2:, :]
    else:
        pmaps1 = pmaps
        pmaps2 = []
    return pmaps1, pmaps2

def select_measurements(pmaps, numsscans, meas):
    """Select measurement baselines to use."""
    if meas == 'all':
        r = pmaps, numsscans
    elif meas == 'mean':
        r = baseline_mean(pmaps, numsscans)
    elif meas == 'a':
        r = pmaps[0::2], numsscans[0::2]
    elif meas == 'b':
        r = pmaps[1::2], numsscans[1::2]
    else:
        raise Exception('Invalid measurement identifier: %s' % meas)
    return r

def baseline_mean(pmaps, numsscans):
    """Take means of each pair of pmaps."""
    baselines = np.array(pairs(pmaps))
    pmaps = np.mean(baselines, axis=0)
    numsscans = pairs(numsscans)[0]
    return pmaps, numsscans

def get_group_id(groups, value):
    """Get group id of a single value."""
    for i, group in enumerate(groups):
        if value in group:
            return i
    return len(groups)

def group_labels(groups, values):
    """Replace labels with group id's.

    Parameter groups is a sequence of sequences that indicate labels belonging
    to each group. Default group will be len(groups)."""
    group_ids = []
    for value in values:
        group_ids.append(get_group_id(groups, value))
    return group_ids

def sole(iterable, desc=None):
    """Return the sole item of an iterable. Raise an exception if the number of
    items is something else than exactly one."""
    if desc is None:
        desc = str(iterable)
    it = iter(iterable)
    try:
        item = next(it)
    except StopIteration:
        raise IOError('No item: %s' % desc)
    try:
        next(it)
    except StopIteration:
        return item
    raise IOError('More than one item: %s' % desc)

def iglob(path, typ='any'):
    """Glob iterator that can filter paths by their type."""
    import glob
    import os.path
    it = glob.iglob(path)
    if typ == 'any':
        pass
    elif typ == 'file':
        it = itertools.ifilter(os.path.isfile, it)
    elif typ == 'dir':
        it = itertools.ifilter(os.path.isdir, it)
    else:
        raise Exception('Invalid path type: {}'.format(typ))
    return it

def sglob(path, typ='any'):
    """Single glob: glob exactly one file."""
    return sole(iglob(path, typ), path)

def parse_filename(filename):
    """Parse input filename formatted as 'num_name_hB_[12][ab]_*'."""
    import re
    #m = re.match(r'(\d+)_([\w_]+)_[^_]*_(\d\w)_', filename)
    m = re.search(r'(\d+)_(\w*)_?(\d\w)_', filename)
    if m:
        num, name, scan = m.groups()
        num = int(num)
        name = name.lower()
        scan = scan.lower()
        return num, name, scan
    raise Exception('Cannot parse filename: {}'.format(filename))

def parse_num_scan(filename):
    """Like parse_filename() but return only num, scan."""
    num, _, scan = parse_filename(filename)
    return num, scan

def scan_pairs(afs):
    """Check that the ascii files are correctly paired as scan baselines. Return
    list of (patient number, scan 1, scan 2) tuples."""
    baselines = pairs(afs)
    r = []
    for af1, af2 in zip(*baselines):
        num1, scan1 = parse_num_scan(af1.basename)
        num2, scan2 = parse_num_scan(af2.basename)
        if num1 != num2 or scan1[0] != scan2[0]:
            raise Exception('Not a pair: %s, %s' % (af1.basename, af2.basename))
        r.append((num1, scan1, scan2))
    return r
