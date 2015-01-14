#!/usr/bin/env python2

"""Calculate ROC AUC for parametric maps vs. Gleason scores. Optionally draw the
ROC curves into a file."""

import argparse
import numpy as np

from dwi import patient
from dwi import util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
            help='increase verbosity')
    p.add_argument('--samplelist', default='samples_all.txt',
            help='sample list file')
    p.add_argument('--scans', '-s', default='scans.txt',
            help='scans file')
    p.add_argument('--threshold', '-t', type=int, default=1,
            help='classification threshold (largest negative)')
    p.add_argument('--average', '-a', action='store_true',
            help='average input voxels into one')
    p.add_argument('--autoflip', action='store_true',
            help='flip data when AUC < .5')
    p.add_argument('--outfile', '-o',
            help='figure output file')
    args = p.parse_args()
    return args

"""
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
    Y = np.array(Y)
    return X, Y, G


# Handle arguments.
args = parse_args()
if args.labeltype == 'cancer':
    args.roi2 = True # Cancer vs. no cancer requires ROI2.
patients = patient.read_patients_file(args.scans)
pmaps, numsscans, params = patient.load_files(patients, args.pmaps, pairs=True)
pmaps, numsscans = util.select_measurements(pmaps, numsscans, args.measurements)

nums = [n for n, s in numsscans]
if args.average:
    pmaps1, pmaps2 = np.mean(pmaps, axis=1, keepdims=True), []
else:
    pmaps1, pmaps2 = util.split_roi(pmaps)

labels = patient.load_labels(patients, nums, args.labeltype)
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
    Y = np.array(util.group_labels(groups, Y))
elif args.labeltype == 'score':
    groups = [map(patient.GleasonScore, args.negatives)]
    Y = np.array(util.group_labels(groups, Y))

if args.verbose:
    print 'Samples: %i, features: %i, labels: %i, type: %s'\
            % (X.shape[0], X.shape[1], len(set(labels)), args.labeltype)
    print 'Labels: %s' % sorted(list(set(labels)))
    print 'Positives: %d' % sum(Y)

util.negate_for_roc(X.T, params)

# Plot ROCs.
if args.outfile:
    import pylab as pl
    n_rows, n_cols = len(params), 1
    pl.figure(figsize=(n_cols*6, n_rows*6))
skipped_params = ['SI0N', 'C', 'RMSE']
for x, param, row in zip(X.T, params, range(len(params))):
    if param in skipped_params:
        continue
    fpr, tpr, auc = util.calculate_roc_auc(Y, x, autoflip=args.autoflip)
    print '%s:\tAUC: %f' % (param, auc)
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
"""

def read_data(samplelist_file, cases):
    samples = dwi.util.read_sample_list(samplelist_file)
    subwindows = dwi.util.read_subwindows(IN_SUBWINDOWS_FILE)
    patientsinfo = dwi.patient.read_patients_file(IN_PATIENTS_FILE)
    data = []
    for sample in samples:
        case = sample['case']
        if cases and not case in cases:
            continue
        score = dwi.patient.get_patient(patientsinfo, case).score
        for scan in sample['scans']:
            try:
                subwindow = subwindows[(case, scan)]
                slice_index = subwindow[0] # Make it zero-based.
            except KeyError:
                # No subwindow defined.
                subwindow = None
                slice_index = None
            subregion = read_subregion(case, scan)
            masks = read_roi_masks(case, scan)
            cancer_mask, normal_mask = masks['ca'], masks['n']
            prostate_mask = read_prostate_mask(case, scan)
            image = read_image(case, scan, PARAMS[0])
            cropped_cancer_mask = cancer_mask.crop(subregion)
            cropped_normal_mask = normal_mask.crop(subregion)
            cropped_prostate_mask = prostate_mask.crop(subregion)
            cropped_image = dwi.util.crop_image(image, subregion).copy()
            #cropped_image = cropped_image[[slice_index],...] # TODO: one slice
            clip_outliers(cropped_image)
            d = dict(case=case, scan=scan, score=score,
                    subwindow=subwindow,
                    slice_index=slice_index,
                    subregion=subregion,
                    cancer_mask=cropped_cancer_mask,
                    normal_mask=cropped_normal_mask,
                    prostate_mask=cropped_prostate_mask,
                    original_shape=image.shape,
                    image=cropped_image)
            data.append(d)
            assert d['cancer_mask'].array.shape ==\
                    d['normal_mask'].array.shape ==\
                    d['prostate_mask'].array.shape ==\
                    d['image'].shape[0:3]
    return data

args = parse_args()
data = read_data(args.samplelist, args.cases)
