#!/usr/bin/env python2

"""Calculate correlation between parameters in parametric maps."""

from __future__ import absolute_import, division, print_function
import argparse
from itertools import product

import numpy as np
import scipy.stats

import dwi.files
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
                   help='be more verbose')
    p.add_argument('--thresholds', nargs=2, type=float, default=[0.5, 0.1],
                   help='thresholds for labeling as prostate, lesion')
    p.add_argument('pmaps', nargs='+',
                   help='input pmaps')
    return p.parse_args()


# def print_correlations(data, params):
#     """Print correlations."""
#     data = np.asarray(data)
#     indices = range(data.shape[-1])
#     for i, j in product(indices, indices):
#         if i < j:
#             rho, pvalue = scipy.stats.spearmanr(data[:, i], data[:, j])
#             s = 'Spearman: {:8} {:8} {:+1.4f} {:+1.4f}'
#             print(s.format(params[i], params[j], rho, pvalue))


# def print_aucs(data, params):
#     """Print ROC AUCs."""
#     data = np.asarray(data)
#     y = data[:, 0]
#     for i in range(1, data.shape[-1]):
#         x = data[:, i]
#         param = params[i]
#         _, _, auc = dwi.util.calculate_roc_auc(y, x, autoflip=True)
#         s = 'AUC: {p:8} {a:+1.3f}'
#         print(s.format(p=param, a=auc))


# def print_aucs_(data, params):
#     """Print ROC AUCs."""
#     data = np.asarray(data)
#     indices = range(data.shape[-1])
#     for i, j in product(indices, indices):
#         if i != j:
#             y, x = data[:, i], data[:, j]
#             py, px = params[i], params[j]
#             try:
#                 _, _, auc = dwi.util.calculate_roc_auc(y, x, autoflip=True)
#             except ValueError:
#                 continue
#             s = 'AUC:  {py:8}  {px:8}  {a:+1.3f}'
#             print(s.format(py=py, px=px, a=auc))


# def floor_or_ceil(a, threshold=0.5, out=None):
#     """Return floor or ceiling based on a minimum threshold."""
#     if a < threshold:
#         return np.floor(a, out=out)
#     else:
#         return np.ceil(a, out=out)


def main():
    args = parse_args()
    tuples = [dwi.files.read_pmap(x) for x in args.pmaps]
    pmaps = [x[0] for x in tuples]
    attrs = [x[1] for x in tuples]
    params = attrs[0]['parameters']
    if args.verbose > 1:
        print('Parameters:', params)
    if (any(x['parameters'] != params for x in attrs) or
        any(x.shape[-1] != len(params) for x in pmaps)):
        raise ValueError('Nonuniform parameters')

    # Flatten and concatenate pmaps.
    pmap = np.concatenate([x.reshape((-1, len(params))) for x in pmaps])
    if args.verbose > 1:
        print('Means:', np.mean(pmap, axis=0))
        print('Total sample size:', len(pmap))

    # # Replace percentages with integer labels according to thresholds.
    # for a in pmap:
    #     a[0] = floor_or_ceil(a[0], args.thresholds[0])
    #     a[1] = floor_or_ceil(a[1], args.thresholds[1])
    # prostate = pmap[pmap[:, 0] > 0]  # Samples labeled as prostate.

    # ppt = np.mean(pmap[:, 0])  # Total prostate per image volume.
    # lpp = np.mean(prostate[:, 1])  # Total lesion per prostate volume.

    # index = -1  # Last value on row is the intensity.
    # param = params[index]
    # # ROC AUC of sample value vs. sample being lesion.
    # y = prostate[:, 1]
    # x = prostate[:, index]
    # _, _, auc = dwi.util.calculate_roc_auc(y, x, autoflip=True)

    # Sample values are in the last column.
    index = -1
    param = params[index]
    x = pmap[:, index]

    # Labels.
    y_prostate = pmap[:, 0] >= args.thresholds[0]
    y_lesion = pmap[:, 1] >= args.thresholds[1]

    # Average volume percentages.
    ppt = np.mean(y_prostate)  # Prostate per total.
    lpp = np.mean(y_lesion[y_prostate])  # Lesion per prostate.

    # ROC AUC of prostate sample value predicting sample being lesion.
    _, _, auc = dwi.util.calculate_roc_auc(y_lesion[y_prostate], x[y_prostate],
                                           autoflip=True)

    if args.verbose:
        print('# auc p/total l/p p-threshold l-threshold param')
    d = dict(p=param, pt=args.thresholds[0], lt=args.thresholds[1],
             ppt=ppt, lpp=lpp, auc=auc)
    print('{auc:.3f}  {ppt:.3f}  {lpp:.3f}  {pt}  {lt}  {p}'.format(**d))


if __name__ == '__main__':
    main()
