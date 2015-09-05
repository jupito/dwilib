"""Image standardization.

This module implements the percentile landmark method described in [1]. Much of
the notation (variable names etc.) comes from this paper.

Learning:
    - Use landmark_scores() to get percentile scores for all images.
    - Use map_onto_scale() to map landmark scores to the standard scale.
    - Average mapped landmarks to get standard landmarks. TODO: This should be
      implemented here somehow.
    - Use write_std_cfg() to output configuration.

Standardizing:
    - Use read_std_cfg() to input configuration.
    - Use standardize() to standardize images according to configuration.
    - You can also use transform() to do this more free-form.

See also tools/standardize.py.

[1] Nyul et al. 2000: New variants of a method of MRI scale standardization.
"""

# TODO: Mention that the whole image must be included when standardizing.
# TODO: Replace scoreatpercentile() with np.percentile().

from __future__ import absolute_import, division, print_function
import collections

import numpy as np

import dwi.files
import dwi.util


def default_configuration():
    """Default standardization configuration."""
    return dict(
        pc=(0., 99.8),  # Min, max percentiles.
        landmarks=tuple(range(10, 100, 10)),  # Landmark percentiles.
        scale=(1, 4095),  # Min, max intensities on standard scale.
        )


def landmark_scores(img, pc, landmarks, thresholding=True):
    """Get scores at histogram landmarks.

    Parameters
    ----------
    img : ndarray
        Model used for fitting.
    pc : pair of numbers
        Minimum and maximum percentiles.
    landmarks : iterable of numbers
        Landmark percentiles.
    thresholding : bool, optional
        Whether to threshold by mean (default True). This includes only values
        higher than mean, which should help ignore the image background.

    Returns
    -------
    p : pair of floats
        Minimum and maximum percentile scores.
    scores : tuple of floats
        Landmark percentile scores.
    """
    from scipy.stats import scoreatpercentile
    if thresholding:
        # threshold = np.mean(img)
        threshold = np.median(img)
        img = img[img > threshold]
    p1 = scoreatpercentile(img, pc[0])
    p2 = scoreatpercentile(img, pc[1])
    scores = tuple(scoreatpercentile(img, i) for i in landmarks)
    return (p1, p2), scores


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
        assert s1 == s2, (p1, p2, s1, s2)
        return s1
    f = (v-p1) / (p2-p1)
    r = f * (s2-s1) + s1
    return r


def transform(img, p, scores, scale, mapped_scores):
    """Transform image onto standard scale.

    Parameters
    ----------
    img : ndarray
        Image to transform.
    p : pair of numbers
        Minimum and maximum percentile scores.
    scores : iterable of numbers
        Landmark percentile scores.
    scale : pair of numbers
        Minimum and maximum intensities on the standard scale.
    mapped_scores : iterable of numbers
        Standard landmark percentile scores on the standard scale.

    Returns
    -------
    r : ndarray of integers
        Transformed image.
    """
    p1, p2 = p
    s1, s2 = scale
    scores = [p1] + list(scores) + [p2]
    mapped_scores = [s1] + list(mapped_scores) + [s2]
    r = np.zeros_like(img, dtype=np.int16)
    for pos, v in np.ndenumerate(img):
        # Select slot where to map.
        slot = sum(v > s for s in scores)
        slot = np.clip(slot, 1, len(scores)-1)
        r[pos] = map_onto_scale(scores[slot-1], scores[slot],
                                mapped_scores[slot-1], mapped_scores[slot], v)
    r = np.clip(r, s1-1, s2+1, out=r)
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
    if isinstance(cfg, basestring):
        cfg = read_std_cfg(cfg)
    d = cfg
    p, scores = landmark_scores(img, d['pc'], d['landmarks'])
    img = transform(img, p, scores, d['scale'], d['mapped_scores'])
    return img


def write_std_cfg(filename, pc, landmarks, scale, mapped_scores):
    """Write image standardization configuration file.

    Parameters
    ----------
    filename : string
        Output filename.
    pc : pair of numbers
        Minimum and maximum percentiles.
    landmarks : iterable of numbers
        Landmark percentiles.
    scale : pair of numbers
        Minimum and maximum intensities on the standard scale.
    mapped_scores : iterable of numbers
        Standard landmark percentile scores on the standard scale.
    """
    with open(filename, 'w') as f:
        f.write(dwi.files.toline(pc))
        f.write(dwi.files.toline(landmarks))
        f.write(dwi.files.toline(scale))
        f.write(dwi.files.toline(mapped_scores))


def read_std_cfg(filename):
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
    d['pc'] = tuple(float(x) for x in lines[0])
    d['landmarks'] = tuple(float(x) for x in lines[1])
    d['scale'] = tuple(int(x) for x in lines[2])
    d['mapped_scores'] = tuple(int(x) for x in lines[3])
    if len(d['landmarks']) != len(d['mapped_scores']):
        raise Exception('Invalid standardization file: {}'.format(filename))
    return d
