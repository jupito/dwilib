#!/usr/bin/env python2

"""Calculate ROC AUC for parametric maps vs. Gleason scores. Optionally draw the
ROC curves into a file."""

import argparse

import numpy as np

import dwi.files
import dwi.util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
            help='be more verbose')
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
    p.add_argument('--average', '-a', action='store_true',
            help='average input voxels into one')
    p.add_argument('--autoflip', action='store_true',
            help='flip data when AUC < .5')
    p.add_argument('--outfile', '-o',
            help='output file')
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
    X = np.array(X, dtype=np.float)
    Y = np.array(Y)
    return X, Y, G


# Handle arguments.
args = parse_args()
if args.labeltype == 'cancer':
    args.roi2 = True # Cancer vs. no cancer requires ROI2.
patients = dwi.files.read_patients_file(args.scans)
pmaps, numsscans, params = dwi.patient.load_files(patients, args.pmaps, pairs=True)
pmaps, numsscans = dwi.util.select_measurements(pmaps, numsscans, args.measurements)

nums = [n for n, s in numsscans]
if args.average:
    pmaps1, pmaps2 = np.mean(pmaps, axis=1, keepdims=True), []
else:
    pmaps1, pmaps2 = dwi.util.split_roi(pmaps)

labels = dwi.patient.load_labels(patients, nums, args.labeltype)
labels_nocancer = [0] * len(labels)
X1, Y1, G1 = load_data(pmaps1, labels, numsscans)
if len(pmaps2):
    X2, Y2, G2 = load_data(pmaps2, labels_nocancer, numsscans)

if args.roi2:
    X = np.concatenate((X1, X2))
    Y = np.concatenate((Y1, Y2))
    G = G1 + G2
else:
    X = X1
    Y = Y1
    G = G1

# Group samples as negatives and positives.
if args.labeltype == 'ord':
    groups = [range(args.threshold)]
    Y = np.array(dwi.util.group_labels(groups, Y))
elif args.labeltype == 'score':
    groups = [map(dwi.patient.GleasonScore, args.negatives)]
    Y = np.array(dwi.util.group_labels(groups, Y))

if args.verbose > 1:
    print ('Samples: %i, features: %i, labels: %i, type: %s'
            % (X.shape[0], X.shape[1], len(set(labels)), args.labeltype))
    print 'Labels: %s' % sorted(list(set(labels)))
    print 'Positives: %d' % sum(Y)

# Plot ROCs.
if args.outfile:
    import pylab as pl
    n_rows, n_cols = len(params), 1
    pl.figure(figsize=(n_cols*6, n_rows*6))
skipped_params = ['SI0N', 'C', 'RMSE']
for x, param, row in zip(X.T, params, range(len(params))):
    if param in skipped_params:
        continue
    fpr, tpr, auc = dwi.util.calculate_roc_auc(Y, x, autoflip=args.autoflip)
    if args.verbose:
        print '%s:\tAUC: %f' % (param, auc)
    else:
        print '%f' % auc
    if args.outfile:
        pl.subplot2grid((n_rows, n_cols), (row, 0))
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive rate')
        pl.ylabel('True Positive rate')
        pl.title('%s' % param)
        pl.legend(loc='lower right')

if args.outfile:
    print 'Writing %s...' % args.outfile
    pl.savefig(args.outfile, bbox_inches='tight')
