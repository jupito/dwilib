import os.path
from functools import total_ordering
import numpy as np

import dwi.asciifile
import dwi.files
import dwi.util

@total_ordering
class GleasonScore(object):
    def __init__(self, score):
        """Intialize with a sequence or a string like '3+4+5' (third digit is
        optional)."""
        if isinstance(score, str) or isinstance(score, unicode):
            s = score.split('+')
        s = tuple(map(int, s))
        if len(s) == 2:
            s += (0,) # Internal representation always has three digits.
        if not len(s) == 3:
            raise Exception('Invalid gleason score: %s', score)
        self.score = s

    def __repr__(self):
        s = self.score
        if not s[-1]:
            s = s[0:-1] # Drop trailing zero.
        return '+'.join(map(str, s))

    def __hash__(self):
        return hash(self.score)

    def __iter__(self):
        return iter(self.score)

    def __eq__(self, other):
        return self.score == other.score

    def __lt__(self, other):
        return self.score < other.score

class Lesion(object):
    def __init__(self, score, location):
        self.score = score
        self.location = location

    def __repr__(self):
        return repr((self.score, self.location))

@total_ordering
class Patient(object):
    def __init__(self, num, name, scans, lesions):
        self.num = num
        self.name = name
        self.scans = scans
        self.lesions = lesions
        self.score = lesions[0].score # For backwards compatibility.

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

def scan_in_patients(patients, num, scan):
    """Is this scan listed in the patients sequence?"""
    for p in patients:
        if p.num == num and scan in p.scans:
            return True
    return False

def get_patient(patients, num):
    """Search a patient from sequence by patient number."""
    for p in patients:
        if p.num == num:
            return p
    raise Exception('Patient not found: {}'.format(num))

def get_gleason_scores(patients):
    """Get all separate Gleason scores, sorted."""
    #return sorted({p.score for p in patients})
    return sorted({l.score for l in p.lesions for p in patients})

def score_ord(scores, score):
    """Get Gleason score's ordinal number."""
    return sorted(scores).index(score)

def load_files(patients, filenames, pairs=False):
    """Load pmap files."""
    pmapfiles = []
    for f in filenames:
        num, scan = dwi.util.parse_num_scan(os.path.basename(f))
        if scan_in_patients(patients, num, scan):
            pmapfiles.append(f)
    afs = map(dwi.asciifile.AsciiFile, pmapfiles)
    if pairs:
        dwi.util.scan_pairs(afs)
    ids = [dwi.util.parse_num_scan(af.basename) for af in afs]
    pmaps = [af.a for af in afs]
    pmaps = np.array(pmaps)
    params = afs[0].params()
    assert pmaps.shape[-1] == len(params), 'Parameter name mismatch.'
    #print 'Filenames: %i, loaded: %i, lines: %i, columns: %i'\
    #        % (len(filenames), pmaps.shape[0], pmaps.shape[1], pmaps.shape[2])
    return pmaps, ids, params

def load_labels(patients, nums, labeltype='score'):
    """Load labels according to patient numbers."""
    gs = get_gleason_scores(patients)
    scores = [get_patient(patients, n).score for n in nums]
    if labeltype == 'score':
        # Use Gleason score.
        labels = scores
    elif labeltype == 'ord':
        # Use ordinal.
        labels = [score_ord(gs, s) for s in scores]
    elif labeltype == 'bin':
        # Is aggressive? (ROI1.)
        labels = [s > GleasonScore('3+4') for s in scores]
    elif labeltype == 'cancer':
        # Is cancer? (ROI1 vs 2, all true for ROI1.)
        labels = [1] * len(scores)
    else:
        raise Exception('Invalid parameter: %s' % labeltype)
    return labels

def lesions(patients):
    """Generate all case, scan, lesion combinations."""
    for p in patients:
        for s in p.scans:
            for i, l in enumerate(p.lesions):
                yield p, s, i, l

def read_pmaps(samplelist_file, patients_file, pmapdir, thresholds=['3+3'],
        voxel='all', multiroi=False):
    """Read pmaps labeled by their Gleason score.
    
    Label thresholds are maximum scores of each label group. Labels are ordinal
    of score if no thresholds provided."""
    # TODO Support for selecting measurements over scan pairs
    thresholds = map(GleasonScore, thresholds)
    samples = dwi.files.read_sample_list(samplelist_file)
    patientsinfo = dwi.files.read_patients_file(patients_file)
    data = []
    for patient, scan, i, lesion in lesions(patientsinfo):
        if not multiroi and i != 0:
            continue
        case = patient.num
        score = lesion.score
        if thresholds:
            label = sum(score > t for t in thresholds)
        else:
            label = score_ord(get_gleason_scores(patientsinfo), score)
        roi = i if multiroi else None
        pmap, params, pathname = read_pmap(pmapdir, case, scan, roi=roi,
                voxel=voxel)
        d = dict(case=case, scan=scan, roi=i, score=score, label=label,
                pmap=pmap, params=params, pathname=pathname)
        data.append(d)
        if pmap.shape != data[0]['pmap'].shape:
            raise Exception('Irregular shape: %s' % pathname)
        if params != data[0]['params']:
            raise Exception('Irregular params: %s' % pathname)
    return data

def grouping(data):
    """Return different scores, grouped scores, and their sample sizes.
    
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
        # TODO Move to Asciifile initializer?
        raise Exception('Number of parameters mismatch: %s' % af.filename)
    # Select voxel to use.
    if voxel == 'all':
        pass # Use all voxels.
    elif voxel == 'mean':
        pmap = np.mean(pmap, axis=0, keepdims=True) # Use mean voxel.
    elif voxel == 'median':
        pmap = dwi.util.median(pmap, axis=0, keepdims=True) # Use median voxel.
    else:
        pmap = pmap[[int(voxel)]] # Use single voxel only.
    return pmap, params, af.filename
