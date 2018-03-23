"""Obsolete code, kept for compatibility."""

import logging

import numpy as np

from . import asciifile, dataset
from .types import Path, TextureSpec


def _pmap_path(directory, case, scan, roi=None):
    """Return pmap path."""
    directory = Path(directory)
    d = dict(c=case, s=scan, r=roi)
    if roi is None:
        s = '{c}_*{s}*.txt'
    else:
        d['r'] += 1
        s = '{c}_*{s}_{r}*.txt'
    pattern = s.format(**d)
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise FileNotFoundError(directory / pattern)
    return paths[0]


def _select_voxel(pmap, voxel):
    """Select voxel to use."""
    if voxel == 'all':
        return pmap  # Use all voxels.
    elif voxel == 'sole':
        # Use sole voxel (raise exception if more found).
        if len(pmap) != 1:
            raise ValueError('Too many voxels: {}'.format(len(pmap)))
        return pmap
    elif voxel == 'mean':
        return np.mean(pmap, axis=0, keepdims=True)  # Use mean voxel.
    elif voxel == 'median':
        return np.median(pmap, axis=0, keepdims=True)  # Use median.
    else:
        return pmap[[int(voxel)]]  # Use single voxel only.


def _read_pmap(directory, case, scan, roi=None, voxel='all'):
    """Read single pmap. XXX: Obsolete code."""
    af = asciifile.AsciiFile(_pmap_path(directory, case, scan, roi=roi))
    pmap = _select_voxel(af.a, voxel)
    return pmap, af.params(), af.filename


def _read_pmaps(patients, pmapdir, voxel='all', multiroi=False, dropok=False,
                location=None):
    """Read pmaps."""
    data = []
    for pat, scan, lesion in dataset.iterlesions(patients):
        if not multiroi and lesion.index != 0:
            continue
        if location is not None and lesion.location != location:
            continue
        case = pat.num
        roi = lesion.index if multiroi else None
        try:
            pmap, params, pathname = _read_pmap(pmapdir, case, scan, roi=roi,
                                                voxel=voxel)
        except IOError:
            if dropok:
                logging.warning('Cannot read pmap for %s, dropping...',
                                (case, scan, roi))
                continue
            else:
                raise
        d = dict(case=case, scan=scan, roi=lesion.index, score=lesion.score,
                 label=lesion.label, pmap=pmap, params=params,
                 pathname=pathname)
        data.append(d)
        if pmap.shape != data[0]['pmap'].shape:
            raise ValueError('Irregular shape: %s' % pathname)
        if params != data[0]['params']:
            raise ValueError('Irregular params: %s' % pathname)
    return data


def _grouping(data):
    """Return different scores sorted, grouped scores, and their sample sizes.

    See `collect_data()`."""
    scores = [d['score'] for d in data]
    labels = [d['label'] for d in data]
    n_labels = max(labels) + 1
    groups = [[] for _ in range(n_labels)]
    for s, l in zip(scores, labels):
        groups[l].append(s)
    different_scores = sorted(set(scores))
    group_scores = [sorted(set(g)) for g in groups]
    group_sizes = [len(g) for g in groups]
    return different_scores, group_scores, group_sizes


def collect_data(patients, pmapdirs, normalvoxel=None, voxel='all',
                 multiroi=False, dropok=False, location=None, verbose=False):
    """Collect all data (each directory, each pmap, each feature)."""
    X, Y = [], []
    params = []
    scores = None
    for i, pmapdir in enumerate(pmapdirs):
        data = _read_pmaps(patients, pmapdir, voxel=voxel, multiroi=multiroi,
                           dropok=dropok, location=location)
        if scores is None:
            scores, groups, group_sizes = _grouping(data)
        for j, param in enumerate(data[0]['params']):
            x = [v[j] for d in data for v in d['pmap']]
            if normalvoxel is None:
                y = [d['label'] for d in data for v in d['pmap']]
            else:
                y = [int(k != normalvoxel) for d in data for k in
                     range(len(d['pmap']))]
            X.append(np.asarray(x))
            Y.append(np.asarray(y))
            params.append('{}:{}'.format(i, param))

    # Print info.
    if verbose > 1:
        d = dict(n=len(X[0]),
                 ns=len(scores), s=scores,
                 ng=len(groups), g=' '.join(str(x) for x in groups),
                 gs=', '.join(str(x) for x in group_sizes))
        print('Samples: {n}'.format(**d))
        print('Scores: {ns}: {s}'.format(**d))
        print('Groups: {ng}: {g}'.format(**d))
        print('Group sizes: {gs}'.format(**d))

    return X, Y, params


def param_to_tspec(param):
    """Get partial TextureSpec from param string (only winsize and method!)."""
    winsize, name = param.split('-', 1)
    method = name.split('(', 1)[0]
    return TextureSpec(method, int(winsize), None)
