#!/usr/bin/env python2

"""Calculate ROC AUC for parametric maps vs. Gleason scores. Optionally draw the
ROC curves into a file."""

import argparse
import glob
import numpy as np

import dwi.asciifile
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
    p.add_argument('--pmapdir', required=True,
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

def read_data(samplelist_file, patients_file, pmapdir, thresholds=['3+3'],
        average=False):
    """Read data. Thresholds are maximum scores of each group."""
    # TODO Support for selecting measurements over scan pairs
    thresholds = map(dwi.patient.GleasonScore, thresholds)
    samples = dwi.util.read_sample_list(samplelist_file)
    patientsinfo = dwi.patient.read_patients_file(patients_file)
    data = []
    for sample in samples:
        case = sample['case']
        score = dwi.patient.get_patient(patientsinfo, case).score
        label = sum(score > t for t in thresholds)
        for scan in sample['scans']:
            pmap, params, pathname = read_pmapfile(pmapdir, case, scan, average)
            d = dict(case=case, scan=scan, score=score, label=label, pmap=pmap,
                    params=params, pathname=pathname)
            data.append(d)
            if pmap.shape != data[0]['pmap'].shape:
                raise Exception('Irregular shape: %s' % pathname)
            if params != data[0]['params']:
                raise Exception('Irregular params: %s' % pathname)
    return data

def read_pmapfile(dirname, case, scan, average):
    """Read single pmap."""
    d = dict(d=dirname, c=case, s=scan)
    s = '{d}/{c}_*_{s}_*.txt'.format(**d)
    paths = glob.glob(s)
    if len(paths) != 1:
        raise Exception('Ambiguous pmap: %s' % s)
    af = dwi.asciifile.AsciiFile(paths[0])
    pmap = af.a
    params = af.params()
    if pmap.shape[-1] != len(params):
        # TODO Move to Asciifile initializer?
        raise Exception('Number of parameters mismatch: %s' % af.filename)
    if average:
        pmap = np.mean(pmap, axis=0, keepdims=True)
    return pmap, params, af.filename


args = parse_args()
data = read_data(args.samplelist, args.patients, args.pmapdir, [args.threshold],
        args.average)
params = data[0]['params']
labels = set(d['score'] for d in data)
X, Y = [], []
Params = []
for i, param in enumerate(params):
    x, y = [], []
    for d in data:
        for v in d['pmap']:
            x.append(v[i])
            y.append(d['label'])
    x = np.asarray(x)
    y = np.asarray(y)
X.append(x)
Y.append(y)
Params.append(param)

for x, y, param in zip(X, Y, Params):
    if args.verbose > 1:
        d = dict(ns=len(x), nl=len(labels), l=sorted(labels), npos=sum(y))
        print param
        print 'Samples: {ns}'.format(**d)
        print 'Labels: {nl}: {l}'.format(**d)
        print 'Positives: {npos}'.format(**d)
    fpr, tpr, auc = dwi.util.calculate_roc_auc(y, x)
    if args.autoflip and auc < 0.5:
        x = -x
        fpr, tpr, auc = dwi.util.calculate_roc_auc(y, x)
    auc_bs = dwi.util.bootstrap_aucs(y, x, args.nboot)
    avg = np.mean(auc_bs)
    ci1, ci2 = dwi.util.ci(auc_bs)
    print '%s\t%0.6f\t%0.6f\t%0.6f\t%0.6f' % (param, auc, avg, ci1, ci2)

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
