#!/usr/bin/env python2

import sys
import argparse
import numpy as np
from numpy import mean, std
import scipy as sp
import scipy.stats

from dwi import patient
from dwi import util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description = 'Compare bootstrapped ROC AUCs.')
    p.add_argument('--pmaps', '-m', nargs='+', required=True,
            help='pmap files')
    p.add_argument('--scans', '-s', default='scans.txt',
            help='scans file')
    p.add_argument('--roi2', '-2', action='store_true',
            help='use ROI2')
    p.add_argument('--measurements',
            choices=['all', 'mean', 'a', 'b'], default='all',
            help='measurement baselines')
    p.add_argument('--labeltype', '-l',
            choices=['score', 'ord', 'bin', 'cancer'], default='bin',
            help='label type')
    p.add_argument('--negatives', '-g', nargs='+', default=['3+3'],
            help='group of Gleason scores classified as negative')
    p.add_argument('--threshold', '-t', type=int, default=1,
            help='classification threshold')
    p.add_argument('--nboot', '-b', type=int, default=2000,
            help='number of bootstraps')
    p.add_argument('--nmodels', '-n', type=int, default=2,
            help='number of models')
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
    X = np.array(X)
    Y = np.array(Y)
    return X, Y, G

def bootstrap_aucs(y, x, n=2000):
    """Produce an array of bootstrapped ROC AUCs."""
    aucs = np.zeros((n))
    for i in range(n):
        yb, xb = util.resample_bootstrap_stratified(y, x)
        #fpr, tpr, _ = util.roc(yb, xb)
        #auc = util.roc_auc(fpr, tpr)
        _, _, auc = util.calculate_roc_auc(yb, xb)
        aucs[i] = auc
    return aucs

def compare_aucs(aucs1, aucs2):
    """Compare two arrays of (bootstrapped) ROC AUC values, with the method
    described in pROC software."""
    D = aucs1 - aucs2
    z = mean(D) / std(D)
    p = 1 - sp.stats.norm.cdf(abs(z))
    return mean(D), z, p


# Handle arguments.
args = parse_args()
if args.labeltype == 'cancer':
    args.roi2 = True # Cancer vs. no cancer requires ROI2.
patients = patient.read_patients_file(args.scans)
l = len(args.pmaps)/args.nmodels
Pmapfiles = list(util.chunks(args.pmaps, l)) # Divide filenames by parameter.

# Read input files.
Pmaps = []
Numsscans = []
Params = []
for pmapfiles in Pmapfiles:
    pmaps, numsscans, params = patient.load_files(patients, pmapfiles, pairs=True)
    pmaps, numsscans = util.select_measurements(pmaps, numsscans, args.measurements)
    Pmaps.append(pmaps)
    Numsscans.append(numsscans)
    Params.append(params)

# Collect sample vectors.
X = []
Y = []
for pmaps, numsscans in zip(Pmaps, Numsscans):
    nums = [n for n, _ in numsscans]
    labels = patient.load_labels(patients, nums, args.labeltype)
    labels_nocancer = [0] * len(labels)
    pmaps1, pmaps2 = util.split_roi(pmaps)
    x1, y1, _ = load_data(pmaps1, labels, numsscans)
    x2, y2, _ = load_data(pmaps2, labels_nocancer, numsscans)
    X.append(np.concatenate((x1, x2)) if args.roi2 else x1)
    Y.append(np.concatenate((y1, y2)) if args.roi2 else y1)

# Group samples as negatives and positives.
for i in range(len(Y)):
    if args.labeltype == 'ord':
        groups = [range(args.threshold)]
        Y[i] = np.array(util.group_labels(groups, Y[i]))
    elif args.labeltype == 'score':
        groups = [map(patient.GleasonScore, args.negatives)]
        Y[i] = np.array(util.group_labels(groups, Y[i]))

if args.verbose:
    print 'Samples: %i, features: %i, labels: %i, type: %s'\
            % (X[0].shape[0], X[0].shape[1], len(set(labels)), args.labeltype)
    print 'Labels: %s' % sorted(list(set(labels)))
    print 'Positives: %d' % sum(Y[0])
    print args.threshold, args.negatives

# Group samples all by parameter.
X_all = []
Y_all = []
params_all = []
skipped_params = ['SI0N', 'C', 'RMSE']
for x, y, params in zip(X, Y, Params):
    for i, param in enumerate(params):
        if param in skipped_params:
            continue
        X_all.append(x[:,i])
        Y_all.append(y)
        params_all.append(param)

util.negate_for_roc(X_all, params_all)

if args.verbose:
    print 'Bootstrapping %i parameters %i times...' %\
            (len(params_all), args.nboot)

# Collect AUC's and bootstrapped AUC's.
aucs = []
aucs_bs = []
for x, y in zip(X_all, Y_all):
    #fpr, tpr, _ = util.roc(y, x)
    #auc = util.roc_auc(fpr, tpr)
    _, _, auc = util.calculate_roc_auc(y, x)
    aucs.append(auc)
    bs = bootstrap_aucs(y, x, args.nboot)
    aucs_bs.append(bs)

# Print AUC's and mean bootstrapped AUC's.
if args.verbose:
    print '# param\tAUC\tAUCbs\tlower\tupper'
for param, auc, auc_bs in zip(params_all, aucs, aucs_bs):
    avg = mean(auc_bs)
    ci1, ci2 = util.ci(auc_bs)
    print '%s\t%0.6f\t%0.6f\t%0.6f\t%0.6f' % (param, auc, avg, ci1, ci2)

# Print bootstrapped AUC comparisons.
if args.verbose:
    print '# param1\tparam2\tdiff\tZ\tp'
done = []
for i, param_i in enumerate(params_all):
    for j, param_j in enumerate(params_all):
        if i == j:
            continue
        if (i, j) in done or (j, i) in done:
            continue
        done.append((i,j))
        d, z, p = compare_aucs(aucs_bs[i], aucs_bs[j])
        print '%s\t%s\t%+0.6f\t%+0.6f\t%0.6f' % (param_i, param_j, d, z, p)
