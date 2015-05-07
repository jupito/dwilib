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
    p.add_argument('--samplelist',
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
    p.add_argument('--scale', metavar='I', nargs=2, type=float,
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
        img = d['image'][...,0]
        d['p1'] = scoreatpercentile(img, pc1)
        d['p2'] = scoreatpercentile(img, pc2)
        #d['deciles'] = [scoreatpercentile(img, i*10) for i in range(1, 10)]
        percentiles = [0, 25, 50, 75, 99.8]
        d['landmarks'] = [scoreatpercentile(img, i) for i in percentiles]

def map_landmarks(data, s1, s2):
    for d in data:
        p1, p2 = d['p1'], d['p2']
        d['mapped_landmarks'] = [map_onto_scale(p1, p2, s1, s2, v) for v in
                d['landmarks']]

def map_onto_scale(p1, p2, s1, s2, v):
    """Map value v from original scale [p1, p2] onto standard scale [s1, s2]."""
    f = (v-p1) / (p2-p1)
    r = f * (s2-s1) + s1
    return r

#def select_ioi(data, pc1, pc2):
#    """Select intensity of interest (IOI) parts of images."""
#    for d in data:
#        img = d['image'][...,0]
#        s1 = np.percentile(img, pc1)
#        s2 = np.percentile(img, pc2)
#        img = img[img>=s1]
#        img = img[img<=s2]
#        d['ioi'] = img

def plot(data):
    import pylab as pl
    for d in data:
        img = d['image']
        hist, bin_edges = np.histogram(img, bins=1000, density=True)
        pl.plot(bin_edges[:-1], hist)
    pl.show()
    pl.close()
    for d in data:
        y = d['landmarks']
        x = range(len(y))
        pl.plot(x, y)
    pl.show()
    pl.close()


args = parse_args()
if args.verbose:
    print 'Reading data...'
data = dwi.dataset.dataset_read_samplelist(args.samplelist, args.cases,
        args.scans)
dwi.dataset.dataset_read_patientinfo(data, args.samplelist)
if args.subregiondir:
    dwi.dataset.dataset_read_subregions(data, args.subregiondir)
dwi.dataset.dataset_read_pmaps(data, args.pmapdir, [args.param])

if args.verbose:
    print 'Data:'
    for d in data:
        img = d['image'][...,0]
        print d['case'], d['scan'], img.shape, dwi.util.fivenum(img)

set_landmarks(data, *args.pc)
if args.verbose:
    print 'Landmarks:'
    for d in data:
        print d['case'], d['scan'], d['p1'], d['p2'], d['landmarks']

map_landmarks(data, *args.scale)
if args.verbose:
    print 'Mapped landmarks:'
    for d in data:
        print d['case'], d['scan'], args.scale, d['mapped_landmarks']

mapped_landmarks = np.array([d['mapped_landmarks'] for d in data])
print mapped_landmarks.shape
print np.mean(mapped_landmarks, axis=0)
print np.median(mapped_landmarks, axis=0)

plot(data)
