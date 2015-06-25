"""Image standardization.

This module implements the percentile landmark method described in [1]. Much of
the notation (variable names etc.) comes from this paper.

TODO: Creating the standard mapped landmarks is missing here at the moment, it
should be done by averaging the mapped landmarks from all images. See
tools/standardize.py.

[1] Nyul et al. 2000: New variants of a method of MRI scale standardization.
"""

from __future__ import division
import collections

import numpy as np

import dwi.files
import dwi.util

DEFAULT_CONFIGURATION = dict(
        pc=(0., 99.8), # Min, max percentiles.
        landmarks=[i*10 for i in range(1, 10)], # Landmark percentiles.
        scale=(1, 4095), # Min, max intensities on standard scale.
        )

def landmark_scores(img, pc1, pc2, landmarks, thresholding=True):
    """Get scores at histogram landmarks.

    Parameters
    ----------
    img : ndarray
        Model used for fitting.
    pc1, pc2 : number
        Minimum and maximum percentiles.
    landmarks : iterable
        Landmark percentiles.
    thresholding : bool, optional
        Whether to threshold by mean (default True). This includes only values
        higher than mean, which should help ignore the image background.

    Returns
    -------
    p1, p2 : float
        Minimum and maximum percentile scores.
    scores : array of floats
        Landmark percentile scores.
    """
    from scipy.stats import scoreatpercentile
    if thresholding:
        threshold = np.mean(img)
        img = img[img > threshold]
    p1 = scoreatpercentile(img, pc1)
    p2 = scoreatpercentile(img, pc2)
    scores = [scoreatpercentile(img, i) for i in landmarks]
    return p1, p2, scores

def map_onto_scale(p1, p2, s1, s2, v):
    """Map value v from original scale [p1, p2] onto standard scale [s1, s2].

    Parameters
    ----------
    p1, p2 : number
        Minimum and maximum percentile scores.
    s1, s2 : number
        Minimum and maximum intensities on the standard scale.
    v : number
        Value to map.

    Returns
    -------
    r : float
        Mapped value.
    """
    assert p1 <= p2, (p1, p2)
    assert s1 <= s2, (s1, s2)
    if p1 == p2:
        assert s1 == s2, (s1, s2)
        return s1
    f = (v-p1) / (p2-p1)
    r = f * (s2-s1) + s1
    return r

def transform(img, p1, p2, scores, s1, s2, mapped_scores):
    """Transform image onto standard scale.

    Parameters
    ----------
    img : ndarray
        Image to transform.
    p1, p2 : number
        Minimum and maximum percentile scores.
    scores : array of numbers
        Landmark percentile scores.
    s1, s2 : number
        Minimum and maximum intensities on the standard scale.
    mapped_scores : array of numbers
        Standard landmark percentile scores on the standard scale.

    Returns
    -------
    r : ndarray of integers
        Transformed image.
    """
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
    """Transform an image based on a configuration (file).

    Parameters
    ----------
    img : ndarray
        Image to transform.
    cfg : filename (or dict)
        Standardization configuration file (or configuration already read).

    Returns
    -------
    img : ndarray of integers
        Transformed image.
    """
    if isinstance(cfg, str) or isinstance(cfg, unicode):
        cfg = read_standardization_configuration(cfg)
    d = cfg
    p1, p2, scores = landmark_scores(img, d['pc1'], d['pc2'], d['landmarks'])
    img = transform(img, p1, p2, scores, d['s1'], d['s2'], d['mapped_scores'])
    return img

def write_standardization_configuration(filename, pc1, pc2, landmarks, s1, s2,
        mapped_scores):
    """Write image standardization configuration file.

    Parameters
    ----------
    filename : string
        Output filename.
    pc1, pc2 : number
        Minimum and maximum percentiles.
    landmarks : array of numbers
        Landmark percentiles.
    s1, s2 : number
        Minimum and maximum intensities on the standard scale.
    mapped_scores : array of numbers
        Standard landmark percentile scores on the standard scale.
    """
    with open(filename, 'w') as f:
        f.write(dwi.files.toline([pc1, pc2]))
        f.write(dwi.files.toline(landmarks))
        f.write(dwi.files.toline([s1, s2]))
        f.write(dwi.files.toline(mapped_scores))

def read_standardization_configuration(filename):
    """Read image standardization configuration file.

    Parameters
    ----------
    filename : string
        Input filename.

    Returns
    -------
    d : OrderedDict
        Standardization configuration.
    """
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
