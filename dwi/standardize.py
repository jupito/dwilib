"""Image standardization.

See Nyul et al. 2000: New variants of a method of MRI scale standardization.
"""

from __future__ import division

import numpy as np

import dwi.files
import dwi.util

DEFAULT_CONFIGURATION = dict(
        pc=(0., 99.8),
        landmarks=[i*10 for i in range(1, 10)], # Deciles
        scale=(1, 4095),
        )

def landmark_scores(img, pc1, pc2, landmarks, thresholding=True):
    """Get scores at histogram landmarks."""
    from scipy.stats import scoreatpercentile
    if thresholding:
        threshold = np.mean(img)
        img = img[img > threshold]
    p1 = scoreatpercentile(img, pc1)
    p2 = scoreatpercentile(img, pc2)
    scores = [scoreatpercentile(img, i) for i in landmarks]
    return p1, p2, scores

def map_onto_scale(p1, p2, s1, s2, v):
    """Map value v from original scale [p1, p2] onto standard scale [s1, s2]."""
    assert p1 <= p2, (p1, p2)
    assert s1 <= s2, (s1, s2)
    if p1 == p2:
        assert s1 == s2, (s1, s2)
        return s1
    f = (v-p1) / (p2-p1)
    r = f * (s2-s1) + s1
    return r

def transform(img, p1, p2, scores, s1, s2, mapped_scores):
    """Transform image onto standard scale."""
    scores = [p1] + list(scores) + [p2]
    mapped_scores = [s1] + list(mapped_scores) + [s2]
    r = np.zeros_like(img, dtype=np.int)
    for pos, v in np.ndenumerate(img):
        # Select slot where to map.
        slot = sum(v > s for s in scores)
        slot = np.clip(slot, 1, len(scores)-1)
        r[pos] = map_onto_scale(scores[slot-1], scores[slot],
                mapped_scores[slot-1], mapped_scores[slot], v)
    return r

def standardize(img, cfg):
    """Transform an image based on a configuration (file)."""
    if isinstance(cfg, str) or isinstance(cfg, unicode):
        cfg = read_standardization_configuration(cfg)
    d = cfg
    p1, p2, scores = landmark_scores(img, d['pc1'], d['pc2'], d['landmarks'])
    img = transform(img, p1, p2, scores, d['s1'], d['s2'], d['mapped_scores'])
    return img

def write_standardization_configuration(filename, pc1, pc2, landmarks, s1, s2,
        mapped_scores):
    """Write image standardization configuration file."""
    with open(filename, 'w') as f:
        f.write(dwi.files.toline([pc1, pc2]))
        f.write(dwi.files.toline(landmarks))
        f.write(dwi.files.toline([s1, s2]))
        f.write(dwi.files.toline(mapped_scores))

def read_standardization_configuration(filename):
    """Read image standardization configuration file."""
    lines = list(dwi.files.valid_lines(filename))[:4]
    lines = [l.split() for l in lines]
    d = collections.OrderedDict()
    d['pc1'], d['pc2'] = map(float, lines[0])
    d['landmarks'] = map(float, lines[1])
    d['s1'], d['s2'] = map(int, lines[2])
    d['mapped_scores'] = map(int, lines[3])
    if len(d['landmarks']) != len(d['mapped_scores']):
        raise Exception('Invalid standardization file: {}'.format(filename))
    return d
