#!/usr/bin/env python2

"""Calculate ROC AUC for parametric maps vs. Gleason scores. Optionally draw the
ROC curves into a file."""

import argparse
import numpy as np

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
    p.add_argument('--threshold', default='3+3',
            help='classification threshold (maximum negative)')
    p.add_argument('--nboot', type=int, default=2000,
            help='number of bootstraps')
    p.add_argument('--voxel', default='all',
            help='index of voxel to use, or all, mean, median')
    p.add_argument('--autoflip', action='store_true',
            help='flip data when AUC < .5')
    p.add_argument('--compare', action='store_true',
            help='do AUC comparison')
    p.add_argument('--figure',
            help='output figure file')
    args = p.parse_args()
    return args


args = parse_args()

# Collect all parameters.
X, Y = [], []
Params = []
Labels = set()
for i, pmapdir in enumerate(args.pmapdir):
    data = dwi.patient.read_pmaps(args.samplelist, args.patients, pmapdir,
            [args.threshold], voxel=args.voxel)
    Labels = Labels.union(set(d['score'] for d in data))
    groups = [set() for _ in range(max(d['label'] for d in data) + 1)]
    for d in data:
        groups[d['label']].add(d['score'])
    groups = [sorted(g) for g in groups]
    for j, param in enumerate(data[0]['params']):
        x, y = [], []
        for d in data:
            for v in d['pmap']:
                x.append(v[j])
                y.append(d['label'])
        X.append(np.asarray(x))
        Y.append(np.asarray(y))
        Params.append('%i:%s' % (i, param))

# Print info on each parameter.
if args.verbose > 1:
    n_samples = len(X[0])
    n_pos = sum(Y[0])
    n_neg = n_samples - n_pos
    d = dict(ns=n_samples, nn=n_neg, np=n_pos,
            nl=len(Labels), l=sorted(Labels),
            ng=len(groups), g=' '.join(map(str, groups)))
    print 'Samples: {ns}'.format(**d)
    print 'Labels: {nl}: {l}'.format(**d)
    print 'Groups: {ng}: {g}'.format(**d)
    print 'Negatives: {nn}, Positives: {np}'.format(**d)

# Print AUCs and bootstrapped AUCs.
if args.verbose:
    print '# param  AUC  AUC_BS_mean  lower  upper'
Auc_bs = []
params_maxlen = max(len(p) for p in Params)
for x, y, param in zip(X, Y, Params):
    fpr, tpr, auc = dwi.util.calculate_roc_auc(y, x)
    if args.autoflip and auc < 0.5:
        x = -x
        fpr, tpr, auc = dwi.util.calculate_roc_auc(y, x)
    auc_bs = dwi.util.bootstrap_aucs(y, x, args.nboot)
    avg = np.mean(auc_bs)
    ci1, ci2 = dwi.util.ci(auc_bs)
    d = dict(param=param, auc=auc, avg=avg, ci1=ci1, ci2=ci2)
    if args.verbose:
        s = '{param:%i}  {auc:.6f}  {avg:.6f}  {ci1:.6f}  {ci2:.6f}' % params_maxlen
    else:
        s = '{auc:f}'
    print s.format(**d)
    Auc_bs.append(auc_bs)

# Print bootstrapped AUC comparisons.
if args.compare:
    if args.verbose:
        print '# param1\tparam2\tdiff\tZ\tp'
    done = []
    for i, param_i in enumerate(Params):
        for j, param_j in enumerate(Params):
            if i == j or (i, j) in done or (j, i) in done:
                continue
            done.append((i,j))
            d, z, p = dwi.util.compare_aucs(Auc_bs[i], Auc_bs[j])
            print '%s\t%s\t%+0.6f\t%+0.6f\t%0.6f' % (param_i, param_j, d, z, p)

# Plot the ROCs.
if args.figure:
    plot(args.figure)


def plot(X, Y, params, filename):
    """Plot ROCs."""
    import pylab as pl
    n_rows, n_cols = len(params), 1
    pl.figure(figsize=(n_cols*6, n_rows*6))
    skipped_params = ['SI0N', 'C', 'RMSE']
    for x, param, row in zip(X.T, params, range(n_rows)):
        if param in skipped_params:
            continue
        fpr, tpr, auc = dwi.util.calculate_roc_auc(Y, x, autoflip=args.autoflip)
        if args.verbose:
            print '%s:\tAUC: %f' % (param, auc)
        else:
            print '%f' % auc
        pl.subplot2grid((n_rows, n_cols), (row, 0))
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive rate')
        pl.ylabel('True Positive rate')
        pl.title('%s' % param)
        pl.legend(loc='lower right')
    if filename:
        print 'Writing %s...' % filename
        pl.savefig(filename, bbox_inches='tight')
