#!/usr/bin/env python2

"""Inspect correlation of parameters and Gleason score."""

# NOTE: Obsolete code.

import argparse
import math

import numpy as np
import scipy.stats

import dwi.files
import dwi.patient
import dwi.util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
            help='be more verbose')
    p.add_argument('--scans', '-s', default='scans.txt',
            help='scans file')
    p.add_argument('--pmaps', '-m', nargs='+', required=True,
            help='pmap files')
    p.add_argument('--labeltype', '-l', choices=['score', 'bin', 'ord'],
            default='score', help='label type')
    p.add_argument('--groups', '-g', nargs='+', default=[],
            help='Gleason score grouping')
    p.add_argument('--measurements', choices=['all', 'mean', 'a', 'b'],
            default='all', help='measurement baselines')
    p.add_argument('--average', '-a', action='store_true',
            help='average input voxels into one (otherwise use only first)')
    args = p.parse_args()
    return args

def correlation(x, y):
    """Calculate correlation with p-value and confidence interval."""
    assert len(x) == len(y)
    #r, p = scipy.stats.pearsonr(x, y)
    #r, p = scipy.stats.kendalltau(x, y)
    r, p = scipy.stats.spearmanr(x, y)
    n = len(x)
    stderr = 1.0 / math.sqrt(n-3)
    delta = 1.96 * stderr
    lower = math.tanh(math.atanh(r) - delta)
    upper = math.tanh(math.atanh(r) + delta)
    return dict(r=r, p=p, lower=lower, upper=upper)


args = parse_args()
patients = dwi.files.read_patients_file(args.scans)
pmaps, numsscans, params = dwi.patient.load_files(patients, args.pmaps,
        pairs=True)
pmaps, numsscans = dwi.util.select_measurements(pmaps, numsscans,
        args.measurements)

if args.average:
    X = np.mean(pmaps, axis=1)
else:
    X = pmaps[:,0,:] # Use ROI1 only.
nums = [n for n, s in numsscans]
labels = dwi.patient.load_labels(patients, nums, args.labeltype)
different_labels = sorted(list(set(labels)))

if args.verbose > 1:
    print ('Samples: %i, features: %i, labels: %i, type: %s'
            % (X.shape[0], X.shape[1], len(set(labels)), args.labeltype))
    print 'Labels: %s' % different_labels

# Group samples.
if args.labeltype == 'score':
    group_labels = args.groups or different_labels
    if args.verbose > 1:
        print 'Groups: %s' % group_labels
    groups = [[dwi.patient.GleasonScore(x)] for x in group_labels]
    labels = dwi.util.group_labels(groups, labels)

# Add parameter ADCk/K if possible.
try:
    i, j = params.index('ADCk'), params.index('K')
    X = dwi.util.add_dummy_feature(X)
    X[:,-1] = X[:,i] / X[:,j]
    params += ('ADCk/K',)
except ValueError, e:
    pass # Parameters not found.

# Add parameter ADCkN/KN if possible.
try:
    i, j = params.index('ADCkN'), params.index('KN')
    X = dwi.util.add_dummy_feature(X)
    X[:,-1] = X[:,i] / X[:,j]
    params += ('ADCkN/KN',)
except ValueError, e:
    pass # Parameters not found.

# Print coefficients for each parameter.
if args.verbose > 1:
    print '# param   \tr\tp\tlower\tupper'
skipped_params = ['SI0N', 'C', 'RMSE']
for x, param in zip(X.T, params):
    if param in skipped_params:
        continue
    d = dict(param=param)
    d.update(correlation(x, labels))
    if args.verbose:
        s = '{param:10}\t{r:+.3f}\t{p:.3f}\t{lower:+.3f}\t{upper:+.3f}'
    else:
        s = '{r:+.3f}'
    print s.format(**d)
