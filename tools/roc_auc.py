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
    p.add_argument('--roi2', action='store_true',
            help='use ROI2')
    p.add_argument('--measurements',
            choices=['all', 'mean', 'a', 'b'], default='all',
            help='measurement baselines')
    p.add_argument('--threshold', default='3+3',
            help='classification threshold (maximum negative)')
    p.add_argument('--nboot', type=int, default=2000,
            help='number of bootstraps')
    p.add_argument('--average', action='store_true',
            help='average input voxels into one')
    p.add_argument('--autoflip', action='store_true',
            help='flip data when AUC < .5')
    p.add_argument('--figure',
            help='output figure file')
    args = p.parse_args()
    return args


args = parse_args()
X, Y = [], []
Params = []
for i, pmapdir in enumerate(args.pmapdir):
    data = dwi.patient.read_pmaps(args.samplelist, args.patients, pmapdir,
            [args.threshold], args.average)
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

# Print info on each parameter.
if args.verbose > 2:
    for x, y, param in zip(X, Y, Params):
        d = dict(ns=len(x), nl=len(labels), l=sorted(labels), npos=sum(y))
        print param
        print 'Samples: {ns}'.format(**d)
        print 'Labels: {nl}: {l}'.format(**d)
        print 'Positives: {npos}'.format(**d)

if args.verbose > 1:
    print '# param\tAUC\tAUC_BS_mean\tlower\tupper'
Auc_bs = []
for x, y, param in zip(X, Y, Params):
    fpr, tpr, auc = dwi.util.calculate_roc_auc(y, x)
    if args.autoflip and auc < 0.5:
        x = -x
        fpr, tpr, auc = dwi.util.calculate_roc_auc(y, x)
    auc_bs = dwi.util.bootstrap_aucs(y, x, args.nboot)
    avg = np.mean(auc_bs)
    ci1, ci2 = dwi.util.ci(auc_bs)
    if args.verbose:
        print '%s\t%0.6f\t%0.6f\t%0.6f\t%0.6f' % (param, auc, avg, ci1, ci2)
    else:
        print '%f' % auc
    Auc_bs.append(auc_bs)

# Print bootstrapped AUC comparisons.
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

if args.figure:
    plot(args.figure)


# Plot ROCs.
def plot(X, Y, params, filename):
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
