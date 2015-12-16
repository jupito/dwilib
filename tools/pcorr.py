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


def print_correlations(data, params):
    """Print correlations."""
    data = np.asarray(data)
    indices = range(data.shape[-1])
    for i, j in product(indices, indices):
        if i < j:
            rho, pvalue = scipy.stats.spearmanr(data[:, i], data[:, j])
            s = 'Spearman: {:8} {:8} {:+1.4f} {:+1.4f}'
            print(s.format(params[i], params[j], rho, pvalue))


def print_aucs(data, params):
    """Print ROC AUCs."""
    data = np.asarray(data)
    y = data[:, 0]
    for i in range(1, data.shape[-1]):
        x = data[:, i]
        param = params[i]
        _, _, auc = dwi.util.calculate_roc_auc(y, x, autoflip=True)
        s = 'AUC: {p:8} {a:+1.3f}'
        print(s.format(p=param, a=auc))


def print_aucs_(data, params):
    """Print ROC AUCs."""
    data = np.asarray(data)
    indices = range(data.shape[-1])
    for i, j in product(indices, indices):
        if i != j:
            y, x = data[:, i], data[:, j]
            py, px = params[i], params[j]
            try:
                _, _, auc = dwi.util.calculate_roc_auc(y, x, autoflip=True)
            except ValueError:
                continue
            s = 'AUC:  {py:8}  {px:8}  {a:+1.3f}'
            print(s.format(py=py, px=px, a=auc))


def floor_or_ceil(a, threshold=0.5, out=None):
    """Return floor or ceiling based on a minimum threshold."""
    if a < threshold:
        return np.floor(a, out=out)
    else:
        return np.ceil(a, out=out)


def main():
    args = parse_args()
    tuples = [dwi.files.read_pmap(x) for x in args.pmaps]
    pmaps = [x[0] for x in tuples]
    attrs = [x[1] for x in tuples]
    params = attrs[0]['parameters']
    # print(params)
    if not all(x['parameters'] == params for x in attrs):
        raise ValueError('Nonuniform parameters')
    for pmap in pmaps:
        assert pmap.shape[-1] == len(params)
        pmap.shape = (-1, len(params))
        # print(pmap.shape)
    pmap = np.concatenate(pmaps)
    # print(pmap.shape)
    # print(np.mean(pmap, axis=0))
    for a in pmap:
        a[0] = floor_or_ceil(a[0], args.thresholds[0])
        a[1] = floor_or_ceil(a[1], args.thresholds[1])
    d = dict(pt=args.thresholds[0], lt=args.thresholds[1], param=params[2])
    d['ppt'] = np.count_nonzero(pmap[:, 0]) / len(pmap)
    pmap = pmap[pmap[:, 0] > 0]  # Include only prostate cubes.
    d['lpp'] = np.count_nonzero(pmap[:, 1]) / len(pmap)
    _, _, d['auc'] = dwi.util.calculate_roc_auc(pmap[:, 1], pmap[:, 2],
                                                autoflip=True)
    if args.verbose:
        print('# auc p/total l/p p-threshold l-threshold param')
    print('{auc:.3f}  {ppt:.3f}  {lpp:.3f}  {pt}  {lt}  {param}'.format(**d))
    # print(np.mean(pmap, axis=0))
    # print_correlations(pmap, params)
    # print_aucs_(pmap, params)


if __name__ == '__main__':
    main()
