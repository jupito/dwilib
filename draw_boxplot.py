#!/usr/bin/env python2

# Draw boxplots for parameters.

import os.path
import sys
import numpy as np
import pylab as pl

from dwi import patient
from dwi import util

def load_param(param, pmaps, labels, different_labels, label_nocancer, split=True):
    """Load data indicated by command arguments."""
    X = range(len(different_labels))
    for i in range(len(X)):
        X[i] = []
    for pmap, label in zip(pmaps, labels):
        roi1 = pmap
        roi2 = []
        l = len(pmap)
        if split and l > 1:
            roi1 = pmap[0:l/2]
            roi2 = pmap[l/2:]
        for x in roi1:
            i = different_labels.index(label)
            X[i].append(x[param])
        for x in roi2:
            i = different_labels.index(label_nocancer)
            X[i].append(x[param])
    return X

def limits(seq, margin=0.1):
    """Give limits with margin."""
    mn, mx = np.min(seq), np.max(seq)
    d = mx-mn
    return mn-margin*d, mx+margin*d


a, filenames = util.get_args(2)
outfile, patients_file = a
patients = patient.read_patients_file(patients_file)
pmaps, numsscans, params = patient.load_files(patients, filenames, pairs=True)
pmaps, numsscans = util.baseline_mean(pmaps, numsscans)
pmaps = pmaps[:,0:1,:] # Use ROI1 only.

gs = patient.get_gleason_scores(patients)
scores = [patient.get_patient(patients, n).score for n, _ in numsscans]
scores_ord = [patient.score_ord(gs, s) for s in scores] # Use ordinal.
scores_bin = [s.is_aggressive() for s in scores] # Is aggressive? (ROI 1.)
scores_cancer = [1] * len(scores) # Is cancer? (ROI 1 vs 2.)

#labels = scores_ord
labels = scores
#label_nocancer = 0
label_nocancer = patient.GleasonScore((0, 0))
#different_labels = sorted(list(set(labels + [label_nocancer])))
different_labels = sorted(list(set(labels)))

X = []
for i in range(len(params)):
    X.append(load_param(i, pmaps, labels, different_labels, label_nocancer))

# Print info on each score and parameter.
for i, label in enumerate(different_labels):
    nums = [n for n, _ in numsscans]
    cases = np.sort(np.array(nums)[np.array(labels) == label])
    d = dict(i=i+1, label=label, ncases=len(cases), cases=cases)
    print '{i:d}: score: {label:s}, cases: {ncases:d}: {cases:s}'.format(**d)
    a = [p[i] for p in X]
    for param, values in zip(params, a):
        d = dict(param=param)
        d.update(util.fivenumd(values))
        print '\t{param}\t{min:10.5f}\t{median:10.5f}\t{max:10.5f}'.format(**d)

n_rows, n_cols = len(params), 1
pl.figure(figsize=(n_cols*6, n_rows*6))
pl.rc('xtick', direction='out')
pl.rc('ytick', direction='out')
pl.set_cmap('gray')

for i, p in enumerate(params):
    pl.subplot2grid((n_rows, n_cols), (i, 0))
    pl.title('Parameter: %s' % p)
    pl.xlabel('Gleason score')
    pl.ylabel('Parameter value')
    #pl.xticks(range(len(X[i])), different_labels)
    #d = pl.boxplot(X[i], vert=True, notch=False)
    #print '%s %s' % (p, [list(line.get_ydata()) for line in d['fliers']])
    x = scores_ord
    y = [pmap[0,i] for pmap in pmaps]
    pl.xticks(range(len(different_labels)), different_labels)
    pl.ylim(limits(y))
    pl.scatter(x, y)
    # Draw lines for medians.
    x = range(len(different_labels))
    y = [np.median(X[i][j]) for j in x]
    pl.scatter(x, y, s=1000, c='r', marker='_')

pl.savefig(outfile, bbox_inches='tight')
