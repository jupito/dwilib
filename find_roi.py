#!/usr/bin/env python2

"""Find most interesting ROI's in a DWI image."""

import os.path
import sys
import argparse

import numpy as np

from dwi import asciifile
from dwi import fit
from dwi import util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description =
            'Find interesting ROI\'s in a DWI image.')
    p.add_argument('--input', '-i', required=True,
            help='input files')
    #p.add_argument('--roi', '-r', metavar='i', nargs=6, type=int, default=[],
    #        help='ROI (6 integers)')
    p.add_argument('--dim', '-d', metavar='i', nargs=3, type=int,
            default=[1,5,5], help='dimensions of wanted ROI (3 integers)')
    p.add_argument('--graphic', '-g', action='store_true',
            help='show graphic')
    p.add_argument('--verbose', '-v', action='count',
            help='be more verbose')
    args = p.parse_args()
    return args

def get_score_param(img, param):
    """Return parameter score of given ROI."""
    if param.startswith('ADC'):
        r = 1-np.mean(img)
        if np.min(img) < 0.0002:
            r =- 1000000
    elif param.startswith('K'):
        r = np.mean(img)/1000
    elif param.startswith('score'):
        r = np.mean(img)
    else:
        r = 0 # Unknown parameter
    return r

def get_score(img, params):
    """Return total score of given ROI."""
    scores = [get_score_param(i, p) for i, p in zip(img.T, params)]
    r = sum(scores)
    return r

def get_roi_scores(img, d, params):
    """Return array of all scores for each possible ROI of given dimension."""
    scores_shape = tuple((img.shape[i]-d[i]+1 for i in range(3)))
    scores = np.zeros(scores_shape)
    scores.fill(np.nan)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            for k in range(scores.shape[2]):
                z = (i, i+d[0])
                y = (j, j+d[1])
                x = (k, k+d[2])
                roi = img[z[0]:z[1], y[0]:y[1], x[0]:x[1], :]
                scores[i,j,k] = get_score(roi, params)
    return scores

def get_scoremap(img, d, params, n_rois):
    """Return array like original image, with scores of n_rois best ROI's."""
    scores = get_roi_scores(img, d, params)
    #print np.unravel_index(scores.argmax(), scores.shape)
    indices = scores.ravel().argsort()[::-1] # Sort ROI's by descending score.
    indices = indices[0:n_rois] # Select best ones.
    indices = [np.unravel_index(i, scores.shape) for i in indices]
    #scoremap = np.zeros_like(img[...,0])
    scoremap = np.zeros(img.shape[0:-1] + (1,))
    for z, y, x in indices:
        scoremap[z:z+d[0], y:y+d[1], x:x+d[2], 0] += scores[z,y,x]
    return scoremap


args = parse_args()

af = asciifile.AsciiFile(args.input)
img = af.a.view()
params = af.params()
img.shape = af.subwindow_shape() + (img.shape[-1],)
print 'Image shape: {}'.format(img.shape)

# Clip outliers.
for i in range(img.shape[-1]):
    if params[i].startswith('ADC'):
        img[...,i].clip(0, 0.002, out=img[...,i])
    elif params[i].startswith('K'):
        img[...,i].clip(0, 2, out=img[...,i])

dims = [(1,i,i) for i in range(5, 10)]
n_rois = 5000
scoremaps = [get_scoremap(img, d, params, n_rois) for d in dims]
sum_scoremaps = sum(scoremaps)

roimap = get_scoremap(sum_scoremaps, args.dim, ['score'], 1)
corner = [axis[0] for axis in roimap[...,0].nonzero()]
coords = [(x, x+d) for x, d in zip(corner, args.dim)]
print 'Optimal ROI: {}'.format(coords)

if args.verbose:
    for i, p in enumerate(params):
        z, y, x = coords
        a = img[z[0]:z[1], y[0]:y[1], x[0]:x[1], i]
        print p, a.min(), a.max(), np.median(a)

if args.graphic:
    import matplotlib
    import matplotlib.pyplot as plt
    for pmap in [sum_scoremaps, roimap]:
        iview = img[0,...,0]
        pview = pmap[0,...,0]
        view = np.zeros(iview.shape + (3,))
        view[...,2] = iview / iview.max()
        view[...,0] = pview / pview.max()
        plt.imshow(view, interpolation='nearest')
        plt.show()
