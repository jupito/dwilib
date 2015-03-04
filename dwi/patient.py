import os.path
import re
from functools import total_ordering
import numpy as np

import dwi.asciifile
import dwi.util

@total_ordering
class GleasonScore(object):
    def __init__(self, score):
        """Intialize with a sequence or a string like '3+4+5' (third digit is
        optional)."""
        s = score.split('+') if isinstance(score, str) else list(score)
        if not 2 <= len(s) <= 3:
            raise Exception('Invalid gleason score: %s', score)
        if len(s) == 2:
            s.append(0)
        self.score = tuple(map(int, s))

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

@total_ordering
class Patient(object):
    def __init__(self, num, name, scans, score):
        self.num = num
        self.name = name
        self.scans = scans
        self.score = score

    def __repr__(self):
        return repr(self.tuple())

    def __hash__(self):
        return hash(self.tuple())

    def __eq__(self, other):
        return self.tuple() == other.tuple()

    def __lt__(self, other):
        return self.tuple() < other.tuple()

    def tuple(self):
        return self.num, self.name, self.scans, self.score

def read_patients_file(filename):
    """Load a list of patients.

    Row format: num name scan,... score1+score2
    """
    patients = []
    #p = re.compile(r'(\d+)\s+(\w+)\s+([\w,]+)\s+(\d\+\d)')
    p = re.compile(r'(\d+)\s+(\w+)\s+([\w,]+)\s+(\d\+\d(\+\d)?)')
    with open(filename, 'rU') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] == '#':
                continue
            m = p.match(line)
            if m:
                num, name, scans, score = m.groups()[0:4]
                num = int(num)
                name = name.lower()
                scans = sorted(scans.lower().split(','))
                score = GleasonScore(score)
                patient = Patient(num, name, scans, score)
                patients.append(patient)
            else:
                raise Exception('Invalid line in patients file: %s', line)
    return sorted(patients)

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
    return None

def get_gleason_scores(patients):
    """Get all separate Gleason scores, sorted."""
    scores = {p.score for p in patients}
    return sorted(scores)

def score_ord(scores, score):
    """Get Gleason score's ordinal number."""
    return sorted(scores).index(score)

def read_exclude_file(filename):
    """Load a list scans to exclude."""
    exclusions = []
    p = re.compile(r'(\d+)\s+([*\w]+)')
    with open(filename, 'rU') as f:
        for line in f:
            m = p.match(line.strip())
            if m:
                num, scan = m.groups()
                num = int(num)
                scan = scan.lower()
                exclusions.append((num, scan))
    return sorted(list(set(exclusions)))

def scan_excluded(exclusions, num, scan):
    """Tell whether given scan should be excluded."""
    for n, s in exclusions:
        if n == num:
            if s == scan or s == '*':
                return True
    return False

def exclude_files(pmapfiles, patients, exclusions=[]):
    """Return filenames without those that are to be excluded."""
    r = []
    for f in pmapfiles:
        num, scan = dwi.util.parse_num_scan(os.path.basename(f))
        p = get_patient(patients, num)
        if not p:
            continue # Patient not mentioned in patients file: exclude.
        if not scan_in_patients(patients, num, scan):
            continue # Scan not mentioned in patients file: exclude.
        if scan_excluded(exclusions, num, scan):
            continue # Scan mentioned in exclude file: exclude.
        r.append(f)
    return r

def load_files(patients, filenames, pairs=False):
    """Load pmap files."""
    pmapfiles = exclude_files(filenames, patients)
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

def read_pmaps(samplelist_file, patients_file, pmapdir, thresholds=['3+3'],
        voxel='all'):
    """Read pmaps labeled by their Gleason score.
    
    Label thresholds are maximum scores of each label group. Labels are ordinal
    of score if no thresholds provided."""
    # TODO Support for selecting measurements over scan pairs
    thresholds = map(GleasonScore, thresholds)
    samples = dwi.util.read_sample_list(samplelist_file)
    patientsinfo = read_patients_file(patients_file)
    data = []
    for sample in samples:
        case = sample['case']
        score = get_patient(patientsinfo, case).score
        if thresholds:
            label = sum(score > t for t in thresholds)
        else:
            label = score_ord(get_gleason_scores(patientsinfo), score)
        for scan in sample['scans']:
            pmap, params, pathname = read_pmap(pmapdir, case, scan, voxel=voxel)
            d = dict(case=case, scan=scan, score=score, label=label, pmap=pmap,
                    params=params, pathname=pathname)
            data.append(d)
            if pmap.shape != data[0]['pmap'].shape:
                raise Exception('Irregular shape: %s' % pathname)
            if params != data[0]['params']:
                raise Exception('Irregular params: %s' % pathname)
    return data

def read_pmap(dirname, case, scan, voxel='all'):
    """Read single pmap."""
    d = dict(d=dirname, c=case, s=scan)
    path = dwi.util.sglob('{d}/{c}_*_{s}_*.txt'.format(**d))
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
