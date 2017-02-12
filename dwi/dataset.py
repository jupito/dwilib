"""Dataset handling."""

from __future__ import absolute_import, division, print_function

import numpy as np

import dwi.asciifile
import dwi.files
import dwi.patient


def iterlesions(patients):
    """Generate all case, scan, lesion combinations."""
    if isinstance(patients, basestring):
        patients = dwi.files.read_patients_file(patients)
    for p in patients:
        for s in p.scans:
            for l in p.lesions:
                yield p, s, l


def read_pmaps(patients_file, pmapdir, thresholds=('3+3',), voxel='all',
               multiroi=False, dropok=False, location=None):
    """Read pmaps labeled by their Gleason score.

    Label thresholds are maximum scores of each label group. Labels are ordinal
    of score if no thresholds provided."""
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
    """Read single pmap."""
    d = dict(d=dirname, c=case, s=scan, r=roi)
    if roi is None:
        s = '{d}/{c}_*{s}*.txt'
    else:
        d['r'] += 1
        s = '{d}/{c}_*{s}_{r}*.txt'
    path, = dwi.files.iglob(s.format(**d))
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
