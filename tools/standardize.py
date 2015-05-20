#!/usr/bin/env python2

"""Standardize images."""

from __future__ import division
import argparse

import numpy as np
import scipy as sp

import dwi.dataset
import dwi.mask
import dwi.plot
import dwi.util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
            help='increase verbosity')
    p.add_argument('--patients',
            help='sample list file')
    p.add_argument('--subregiondir',
            help='subregion bounding box directory')
    p.add_argument('--pmapdir', default='dicoms_Mono_combinedDICOM',
            help='input parametric map directory')
    p.add_argument('--param', default='ADCm',
            help='image parameter to use')
    p.add_argument('--pc', metavar='I', nargs=2, type=float,
            default=[0, 99.8],
            help='minimum and maximum percentiles')
    p.add_argument('--scale', metavar='I', nargs=2, type=int,
            default=[1, 4095],
            help='standard scale minimum and maximum')
    p.add_argument('--cases', metavar='I', nargs='*', type=int, default=[],
            help='case numbers')
    p.add_argument('--scans', metavar='S', nargs='*', default=[],
            help='scan identifiers')
    args = p.parse_args()
    return args

def set_landmarks(data, pc1, pc2):
    from scipy.stats import scoreatpercentile
    for d in data:
        img = d['img']
        threshold = np.mean(img)
        img = img[img > threshold]
        d['p1'] = scoreatpercentile(img, pc1)
        d['p2'] = scoreatpercentile(img, pc2)
        #percentiles = [25, 50, 75]
        percentiles = [i*10 for i in range(1, 10)]
        d['landmarks'] = percentiles
        d['scores'] = [scoreatpercentile(img, i) for i in percentiles]

def map_landmarks(data, s1, s2):
    for d in data:
        p1, p2 = d['p1'], d['p2']
        d['mapped_scores'] = [int(map_onto_scale(p1, p2, s1, s2, v)) for v in
                d['scores']]

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

def transform_images(data, s1, s2, mapped_scores):
    for d in data:
        d['img_scaled'] = transform(d['img'], d['p1'], d['p2'], d['scores'], s1,
                s2, mapped_scores)
        print dwi.util.fivenum(d['img_scaled'])


def plot(data, s1, s2, outfile):
    import pylab as pl
    for d in data:
        img = d['img']
        hist, bin_edges = np.histogram(img, bins=50, density=True)
        pl.plot(bin_edges[:-1], hist)
    pl.show()
    pl.close()
    for d in data:
        img = d['img_scaled']
        hist, bin_edges = np.histogram(img, bins=50, density=True)
        pl.plot(bin_edges[:-1], hist)
    pl.show()
    pl.close()
    for d in data:
        y = d['scores']
        x = range(len(y))
        pl.plot(x, y)
    pl.show()
    pl.close()
    dwi.plot.show_images([[d['img'], d['img_scaled']] for d in data], vmin=s1,
            vmax=s2, outfile=outfile)


args = parse_args()
pc1, pc2 = args.pc
s1, s2 = args.scale
if args.verbose:
    print 'Reading data...'
data = dwi.dataset.dataset_read_samplelist(args.patients, args.cases,
        args.scans)
if args.subregiondir:
    dwi.dataset.dataset_read_subregions(data, args.subregiondir)
dwi.dataset.dataset_read_pmaps(data, args.pmapdir, [args.param])

if args.verbose:
    print 'Data:'
    for d in data:
        d['img'] = d['image'][15,...,0]
        print d['case'], d['scan'], d['img'].shape, dwi.util.fivenum(d['img'])

set_landmarks(data, pc1, pc2)
if args.verbose:
    print 'Landmark scores:'
    for d in data:
        print d['case'], d['scan'], (d['p1'], d['p2']), d['scores']

map_landmarks(data, s1, s2)
if args.verbose:
    print 'Mapped landmark scores:'
    for d in data:
        print d['case'], d['scan'], (s1, s2), d['mapped_scores']

mapped_scores = np.array([d['mapped_scores'] for d in data],
        dtype=np.int16)
print mapped_scores.shape
mapped_scores = np.mean(mapped_scores, axis=0, dtype=mapped_scores.dtype)
print mapped_scores

#transform_images(data, s1, s2, mapped_scores)

#plot(data, s1, s2, 'std.png')
