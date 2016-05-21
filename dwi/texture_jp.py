"""Texture code relying on an in-house LBP implementation by JP."""

from __future__ import absolute_import, division, print_function

import numpy as np

import dwi.lbp


def lbp_freqs(img, winsize, neighbours=8, radius=1, roinv=1, uniform=1):
    """Local Binary Pattern (LBP) frequency histogram map.

    Invariant to global illumination change, local contrast magnitude, local
    rotation.
    """
    lbp_data = dwi.lbp.lbp(img, neighbours, radius, roinv, uniform)
    lbp_freq_data, n_patterns = dwi.lbp.get_freqs(lbp_data, winsize,
                                                  neighbours, roinv, uniform)
    return lbp_data, lbp_freq_data, n_patterns


def lbp_freq_map(img, winsize, neighbours=8, radius=None, mask=None):
    """Local Binary Pattern (LBP) frequency histogram map."""
    if radius is None:
        radius = winsize // 2
    _, freqs, n = lbp_freqs(img, winsize, neighbours=neighbours, radius=radius)
    output = np.rollaxis(freqs, -1)
    names = ['lbp({r},{i})'.format(r=radius, i=i) for i in range(n)]
    if mask is not None:
        output[:, -mask] = 0
    return output, names
