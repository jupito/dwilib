#!/usr/bin/env python2

"""Standardize images."""

from __future__ import division
import argparse

import numpy as np

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
    p.add_argument('--cases', metavar='I', nargs='*', type=int, default=[],
            help='case numbers')
    p.add_argument('--scans', metavar='S', nargs='*', default=[],
            help='scan identifiers')
    args = p.parse_args()
    return args

def select_ioi(data, pc1, pc2):
    """Select intensity of interest (IOI) parts of images."""
    for d in data:
        img = d['image'][...,0]
        s1 = np.percentile(img, pc1)
        s2 = np.percentile(img, pc2)
        img = img[img>=s1]
        img = img[img<=s2]
        d['ioi'] = img

def plot_hist(hist, bin_edges):
    import pylab as pl
    pl.plot(bin_edges[1:], hist)
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

for d in data:
    img = d['image'][...,0]
    if args.verbose:
        print d['case'], d['scan'], img.shape, dwi.util.fivenum(img)

select_ioi(data, 0, 99.8)

for d in data:
    img = d['ioi']
    if args.verbose:
        print d['case'], d['scan'], img.shape, dwi.util.fivenum(img)

import pylab as pl
for d in data:
    img = d['ioi']
    hist, bin_edges = np.histogram(img, bins=1000, density=True)
    pl.plot(bin_edges[:-1], hist)
pl.show()
pl.close()
