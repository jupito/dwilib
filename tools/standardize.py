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
    p.add_argument('--minmax', metavar='I', nargs=2, type=int,
            default=[0, 99.8],
            help='minimum and maximum percentiles')
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
        d['deciles'] = [scoreatpercentile(img, i*10) for i in range(1, 10)]

def select_ioi(data, pc1, pc2):
    """Select intensity of interest (IOI) parts of images."""
    for d in data:
        img = d['image'][...,0]
        s1 = np.percentile(img, pc1)
        s2 = np.percentile(img, pc2)
        img = img[img>=s1]
        img = img[img<=s2]
        d['ioi'] = img

def plot(data):
    import pylab as pl
    for d in data:
        img = d['ioi']
        hist, bin_edges = np.histogram(img, bins=1000, density=True)
        pl.plot(bin_edges[:-1], hist)
    pl.show()
    pl.close()
    for d in data:
        y = d['deciles']
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
print
for d in data:
    img = d['image'][...,0]
    if args.verbose:
        print d['case'], d['scan'], img.shape, dwi.util.fivenum(img)

set_landmarks(data, *args.minmax)
print
for d in data:
    if args.verbose:
        print d['case'], d['scan'], img.size, d['p1'], d['p2']

select_ioi(data, *args.minmax)
print
for d in data:
    img = d['ioi']
    if args.verbose:
        print d['case'], d['scan'], img.size, dwi.util.fivenum(img)

plot(data)
