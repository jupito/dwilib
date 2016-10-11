"""Routines for handling patient lists."""

from __future__ import absolute_import, division, print_function
from functools import total_ordering
import glob
import os.path

import numpy as np

import dwi.asciifile
import dwi.files
import dwi.util

# Low group: 3 only; intermediate: 4 secondary or tertiary w/o 5; high: rest.
THRESHOLDS_STANDARD = ('3+3', '3+4')

@total_ordering
class GleasonScore(object):
    """Gleason score is a two or three-value measure of prostate cancer
    severity.
    """
    def __init__(self, score):
        """Intialize with a sequence or a string like '3+4+5' (third digit is
        optional).
        """
        if isinstance(score, basestring):
            s = score.split('+')
        elif isinstance(score, GleasonScore):
            s = score.score
        else:
            s = score
        s = tuple(int(x) for x in s)
        if len(s) == 2:
            s += (0,)  # Internal representation always has three digits.
        if len(s) != 3:
            raise ValueError('Invalid gleason score: {}'.format(score))
        self.score = s

    def __iter__(self):
        score = self.score
        if not score[-1]:
            score = score[0:-1]  # Drop trailing zero.
        return iter(score)

    def __repr__(self):
        return '+'.join(str(x) for x in iter(self))

    def __lt__(self, other):
        return self.score < GleasonScore(other).score

    def __eq__(self, other):
        return self.score == GleasonScore(other).score

    def __ne__(self, other):
        return self.score != GleasonScore(other).score

    def __hash__(self):
        return hash(self.score)


class Lesion(object):
    """Lesion is a lump of cancer tissue."""
    def __init__(self, index, score, location):
        self.index = int(index)  # No. in patient.
        self.score = GleasonScore(score)  # Gleason score.
        self.location = str(location).lower()  # PZ or CZ.

    def __repr__(self):
        return repr((self.index, self.score, self.location))

    def __eq__(self, other):
        return (self.score, self.location) == (other.score, other.location)


@total_ordering
class Patient(object):
    """Patient case."""
    def __init__(self, num, name, scans, lesions):
        self.num = int(num)
        self.name = str(name).lower()
        self.scans = scans
        self.lesions = lesions
        self.score = lesions[0].score  # For backwards compatibility.

    def __repr__(self):
        return repr(self.tuple())

    def __hash__(self):
        return hash(self.tuple())

    def __eq__(self, other):
        return self.tuple() == other.tuple()

    def __lt__(self, other):
        return self.tuple() < other.tuple()

    def tuple(self):
        return self.num, self.name, self.scans, self.lesions


def label_lesions(patients, thresholds=None):
    """Label lesions according to score groups."""
    # Alternative: np.searchsorted(thresholds, [x.score for x in l])
    if thresholds is None:
        thresholds = THRESHOLDS_STANDARD
    thresholds = [GleasonScore(x) for x in thresholds]
    lesions = (l for p in patients for l in p.lesions)
    for l in lesions:
        l.label = sum(l.score > x for x in thresholds)


def scan_in_patients(patients, num, scan):
    """Is this scan listed in the patients sequence?"""
    return any(num == p.num and scan in p.scans for p in patients)


def load_files(patients, filenames, pairs=False):
    """Load pmap files. If pairs=True, require scan pairs together."""
    pmapfiles = []
    if len(filenames) == 1:
        # Workaround for platforms without shell-level globbing.
        l = glob.glob(filenames[0])
        if len(l) > 0:
            filenames = l
    for f in filenames:
        num, scan = dwi.util.parse_num_scan(os.path.basename(f))
        if patients is None or scan_in_patients(patients, num, scan):
            pmapfiles.append(f)
    afs = [dwi.asciifile.AsciiFile(x) for x in pmapfiles]
    if pairs:
        dwi.util.scan_pairs(afs)
    ids = [dwi.util.parse_num_scan(af.basename) for af in afs]
    pmaps = [af.a for af in afs]
    pmaps = np.array(pmaps)
    params = afs[0].params()
    assert pmaps.shape[-1] == len(params), 'Parameter name mismatch.'
    return pmaps, ids, params


def cases_scans(patients, cases=None, scans=None):
    """Generate all case, scan combinations, with optional whitelists."""
    for p in patients:
        if cases is None or p.num in cases:
            for s in p.scans:
                if scans is None or s in scans:
                    yield p.num, s


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
    path = dwi.util.sglob(s.format(**d))
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


def grouping(data):
    """Return different scores sorted, grouped scores, and their sample sizes.

    See read_pmaps()."""
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
