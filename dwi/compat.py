"""Obsolete code, kept for compatibility."""

import logging

import numpy as np

from . import asciifile, dataset, files, patient
from .types import Path, TextureSpec


def _pmap_path(directory, case, scan, roi=None):
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


def read_pmaps(patients_file, pmapdir, thresholds=('3+3',), voxel='all',
               multiroi=False, dropok=False, location=None):
    """Read pmaps labeled by their Gleason score.

    Label thresholds are maximum scores of each label group. Labels are ordinal
    of score if no thresholds provided.

    XXX: Obsolete code, used still by tools/roc_auc.py and
    tools/correlation.py.
    """
    # TODO: Support for selecting measurements over scan pairs
    patients = files.read_patients_file(patients_file)
    patient.label_lesions(patients, thresholds=thresholds)
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


def param_to_tspec(param):
    """Get partial TextureSpec from param string (only winsize and method!)."""
    winsize, name = param.split('-', 1)
    method = name.split('(', 1)[0]
    return TextureSpec(int(winsize), method, None)
