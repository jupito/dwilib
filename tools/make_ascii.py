#!/usr/bin/env python2

import sys
import argparse
import numpy as np
from numpy import mean, std
import scipy as sp
import scipy.stats
import pylab as pl
from sklearn import preprocessing
from sklearn import metrics

import dwi.files
import dwi.patient
import dwi.util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description =
            'Make single ASCII file from several ones.')
    p.add_argument('--pmaps', '-m', nargs='+', required=True,
            help='pmap files')
    p.add_argument('--scans', '-s', default='scans.txt',
            help='scans file')
    p.add_argument('--labeltype', '-l',
            choices=['score', 'ord', 'bin', 'cancer'], default='score',
            help='label type')
    p.add_argument('--verbose', '-v', action='count',
            help='be more verbose')
    args = p.parse_args()
    return args

def load_data(pmaps, labels, group_ids):
    """Load data indicated by command arguments."""
    assert len(pmaps) == len(labels) == len(group_ids)
    X = []
    Y = []
    G = []
    for pmap, label, group_id in zip(pmaps, labels, group_ids):
        for x in pmap:
            X.append(x)
            Y.append(label)
            G.append(group_id)
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    return X, Y, G


# Handle arguments.
args = parse_args()
patients = dwi.files.read_patients_file(args.scans)
pmaps, numsscans, params = dwi.patient.load_files(patients, args.pmaps, pairs=True)
pmaps, numsscans = dwi.util.baseline_mean(pmaps, numsscans)

nums = [n for n, _ in numsscans]
labels = dwi.patient.load_labels(patients, nums, args.labeltype)
pmaps = pmaps[:,0,:] # Use ROI1 only.

# Print header
if args.verbose:
    print '# case scan score',
    for p in params:
        print p,
    print

# Print cases.
for numscan, label, pmap in zip(numsscans, labels, pmaps):
    d = dict(n=numscan[0], s=numscan[1][0], l=label)
    print '{n:02} {s} {l}'.format(**d),
    for p in pmap:
        print p,
    print
