#!/usr/bin/env python2

"""Calculate correlation for parametric maps vs. Gleason scores."""

import argparse
import math
import numpy as np
import scipy.stats

import dwi.patient
import dwi.util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
            help='be more verbose')
    p.add_argument('--patients', default='patients.txt',
            help='patients file')
    p.add_argument('--samplelist', default='samples_all.txt',
            help='sample list file')
    p.add_argument('--pmapdir', nargs='+', required=True,
            help='input pmap directory')
    p.add_argument('--thresholds', nargs='*', default=[],
            help='classification thresholds (group maximums)')
    p.add_argument('--voxel', default='all',
            help='index of voxel to use, or all, mean, median')
    p.add_argument('--average', action='store_true',
            help='average input voxels into one')
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
X, Y = [], []
Params = []
for i, pmapdir in enumerate(args.pmapdir):
    data = dwi.patient.read_pmaps(args.samplelist, args.patients, pmapdir,
            args.thresholds, args.average, voxel=args.voxel)
    params = data[0]['params']
    labels = set(d['score'] for d in data)
    for j, param in enumerate(params):
        x, y = [], []
        for d in data:
            for v in d['pmap']:
                x.append(v[j])
                y.append(d['label'])
        X.append(np.asarray(x))
        Y.append(np.asarray(y))
        Params.append('%i:%s' % (i, param))

for x, y, param in zip(X, Y, Params):
    if args.verbose > 1:
        d = dict(ns=len(x), nl=len(labels), l=sorted(labels))
        print param
        print 'Samples: {ns}'.format(**d)
        print 'Labels: {nl}: {l}'.format(**d)
    d = dict(param=param)
    d.update(correlation(x, y))
    if args.verbose > 1:
        print '# param\tr\tp\tlower\tupper'
    if args.verbose:
        s = '{param:10}\t{r:+.3f}\t{p:.3f}\t{lower:+.3f}\t{upper:+.3f}'
    else:
        s = '{r:+.3f}'
    print s.format(**d)
