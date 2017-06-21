"""Dataset handling."""

from __future__ import absolute_import, division, print_function
try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache  # For Python2.
import logging

import numpy as np

import dwi.asciifile
import dwi.image
import dwi.files
from .types import Path
import dwi.paths
import dwi.patient
import dwi.util


class Dataset(object):
    def __init__(self, mode, samplelist, cases=None):
        self.mode = dwi.ImageMode(mode)
        self.samplelist = samplelist
        self.cases = cases

    @property
    def samplelist_path(self):
        return dwi.paths.samplelist_path(self.mode, self.samplelist)

    def each_patient(self):
        patients = dwi.files.read_patients_file(self.samplelist_path)
        for p in patients:
            if self.cases is None or p.num in self.cases:
                yield p

    def each_image_id(self):
        for p in self.each_patient():
            for s in p.scans:
                yield p.num, s, p.lesions

    def each_lesion(self):
        for case, scan, lesions in self.each_image_id():
            for lesion in lesions:
                yield case, scan, lesion


# @lru_cache(maxsize=16)
def read_prostate_mask(mode, case, scan):
    path = dwi.paths.mask_path(mode, 'prostate', case, scan)
    return dwi.image.Image.read_mask(path)


def read_lesion_mask(mode, case, scan, lesion):
    path = dwi.paths.mask_path(mode, 'lesion', case, scan,
                               lesion=lesion.index+1)
    return dwi.image.Image.read_mask(path)


# @lru_cache(maxsize=16)
def read_lesion_masks(mode, case, scan, lesions, only_largest=False):
    masks = (read_lesion_mask(mode, case, scan, x) for x in lesions)
    if only_largest:
        # Use only the biggest lesion.
        d = {np.count_nonzero(x): x for x in masks}
        masks = [d[max(d.keys())]]
        logging.warning([mode, case, scan, lesions, d.keys(),
                         np.count_nonzero(masks[0])])
    return dwi.util.unify_masks(masks)


def iterlesions(patients):
    """Generate all case, scan, lesion combinations."""
    if dwi.util.isstring(patients):
        patients = dwi.files.read_patients_file(patients)
    for p in patients:
        for s in p.scans:
            for l in p.lesions:
                yield p, s, l


def read_pmaps(patients_file, pmapdir, thresholds=('3+3',), voxel='all',
               multiroi=False, dropok=False, location=None):
    """Read pmaps labeled by their Gleason score.

    Label thresholds are maximum scores of each label group. Labels are ordinal
    of score if no thresholds provided.

    XXX: Obsolete code, used still by tools/roc_auc.py and
    tools/correlation.py.
    """
    # TODO: Support for selecting measurements over scan pairs
    patients = dwi.files.read_patients_file(patients_file)
    dwi.patient.label_lesions(patients, thresholds=thresholds)
    data = []
    for patient, scan, lesion in iterlesions(patients):
        if not multiroi and lesion.index != 0:
            continue
        if location is not None and lesion.location != location:
            continue
        case = patient.num
        roi = lesion.index if multiroi else None
        try:
            pmap, params, pathname = read_pmap(pmapdir, case, scan, roi=roi,
                                               voxel=voxel)
        except IOError:
            if dropok:
                print('Cannot read pmap for {}, dropping...'.format(
                    (case, scan, roi)))
                continue
            else:
                raise
        d = dict(case=case, scan=scan, roi=lesion.index, score=lesion.score,
                 label=lesion.label, pmap=pmap, params=params,
                 pathname=pathname)
        data.append(d)
        if pmap.shape != data[0]['pmap'].shape:
            raise Exception('Irregular shape: %s' % pathname)
        if params != data[0]['params']:
            raise Exception('Irregular params: %s' % pathname)
    return data


def read_pmap(dirname, case, scan, roi=None, voxel='all'):
    """Read single pmap. XXX: Obsolete code."""
    d = dict(c=case, s=scan, r=roi)
    if roi is None:
        s = '{c}_*{s}*.txt'
    else:
        d['r'] += 1
        s = '{c}_*{s}_{r}*.txt'
    path, = Path(dirname).glob(s.format(**d))
    af = dwi.asciifile.AsciiFile(path)
    pmap = af.a
    params = af.params()
    if pmap.shape[-1] != len(params):
        # TODO: Move to Asciifile initializer?
        raise Exception('Number of parameters mismatch: %s' % af.filename)
    # Select voxel to use.
    if voxel == 'all':
        pass  # Use all voxels.
    elif voxel == 'sole':
        # Use sole voxel (raise exception if more found).
        if len(pmap) != 1:
            raise Exception('Too many voxels: {}'.format(len(pmap)))
    elif voxel == 'mean':
        pmap = np.mean(pmap, axis=0, keepdims=True)  # Use mean voxel.
    elif voxel == 'median':
        pmap = np.median(pmap, axis=0, keepdims=True)  # Use median.
    else:
        pmap = pmap[[int(voxel)]]  # Use single voxel only.
    return pmap, params, af.filename


def read_tmap(mode, case, scan, tspec=None, masktype='prostate', **kwargs):
    """Read a texture map."""
    method, winsize = tspec or ('raw', 1)
    path = dwi.paths.texture_path(mode, case, scan, None, masktype, 'all', 0,
                                  method, winsize, voxel='all')
    return dwi.image.Image.read(path, **kwargs)
