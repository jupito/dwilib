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
    # print_correlations(pmap, params)
    print_aucs(pmap, params)


if __name__ == '__main__':
    main()
