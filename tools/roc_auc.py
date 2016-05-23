#!/usr/bin/env python2

"""Calculate ROC AUC for parametric maps vs. Gleason scores. Optionally compare
AUCs and draw the ROC curves into a file."""

from __future__ import absolute_import, division, print_function
import argparse
import numpy as np

import dwi.patient
import dwi.plot
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
                   help='be more verbose')
    p.add_argument('--patients', default='patients.txt',
                   help='patients file')
    p.add_argument('--pmapdir', nargs='+', required=True,
                   help='input pmap directory')
    p.add_argument('--threshold', default='3+3',
                   help='classification threshold (maximum negative)')
    p.add_argument('--nboot', type=int,
                   help='number of bootstraps (try 2000)')
    p.add_argument('--voxel', default='all',
                   help='index of voxel to use, or all, sole, mean, median')
    p.add_argument('--normalvoxel', type=int,
                   help='index of voxel to use as Gleason score zero: '
                   'implies --voxel=all, ignores --threshold')
    p.add_argument('--multilesion', action='store_true',
                   help='use all lesions, not just first for each')
    p.add_argument('--autoflip', action='store_true',
                   help='flip data when AUC < 0.5')
    p.add_argument('--compare', action='store_true',
                   help='do AUC comparison')
    p.add_argument('--dropok', action='store_true',
                   help='allow dropping of files not found')
    p.add_argument('--figure',
                   help='output figure file')
    return p.parse_args()


def main():
    args = parse_args()
    if args.normalvoxel is not None and args.voxel != 'all':
        raise ValueError('Argument --normalvoxel implies --voxel=all')
    thresholds = [args.threshold]

    # Collect all parameters.
    X, Y = [], []
    Params = []
    scores = None
    for i, pmapdir in enumerate(args.pmapdir):
        data = dwi.patient.read_pmaps(args.patients, pmapdir, thresholds,
                                      voxel=args.voxel,
                                      multiroi=args.multilesion,
                                      dropok=args.dropok)
        if scores is None:
            scores, groups, group_sizes = dwi.patient.grouping(data)
        for j, param in enumerate(data[0]['params']):
            x, y = [], []
            for d in data:
                for k, v in enumerate(d['pmap']):
                    label = d['label']
                    if args.normalvoxel is not None:
                        label = int(k != args.normalvoxel)
                    x.append(v[j])
                    y.append(label)
            X.append(np.asarray(x))
            Y.append(np.asarray(y))
            Params.append('%i:%s' % (i, param))

    # Print info.
    if args.verbose > 1:
        d = dict(n=len(X[0]),
                 ns=len(scores), s=scores,
                 ng=len(groups), g=' '.join(str(x) for x in groups),
                 gs=', '.join(str(x) for x in group_sizes))
        print('Samples: {n}'.format(**d))
        print('Scores: {ns}: {s}'.format(**d))
        print('Groups: {ng}: {g}'.format(**d))
        print('Group sizes: {gs}'.format(**d))

    # Print AUCs and bootstrapped AUCs.
    if args.verbose > 1:
        print('# param  AUC  AUC_BS_mean  lower  upper')
    Auc_bs = []
    params_maxlen = max(len(p) for p in Params)
    d = dict(l=params_maxlen)
    for x, y, param in zip(X, Y, Params):
        d['param'] = param
        s = '{param:{l}}  {auc:.3f}'
        if np.any(np.isnan(x)):
            d['auc'] = np.nan
            print(s.format(**d))
            continue
        _, _, auc = dwi.util.calculate_roc_auc(y, x, autoflip=False)
        if args.autoflip and auc < 0.5:
            x = -x
            _, _, auc = dwi.util.calculate_roc_auc(y, x)
        d['auc'] = auc
        if args.nboot:
            # Note: x may now be negated (ROC flipped).
            auc_bs = dwi.util.bootstrap_aucs(y, x, args.nboot)
            avg = np.mean(auc_bs)
            ci1, ci2 = dwi.util.ci(auc_bs)
            d.update(avg=avg, ci1=ci1, ci2=ci2)
            Auc_bs.append(auc_bs)
            s += '  {avg:.3f}  {ci1:.3f}  {ci2:.3f}'
        print(s.format(**d))

    # Print bootstrapped AUC comparisons.
    if args.nboot and args.compare:
        if args.verbose > 1:
            print('# param1  param2  diff  Z  p')
        done = []
        for i, param_i in enumerate(Params):
            for j, param_j in enumerate(Params):
                if i == j or (i, j) in done or (j, i) in done:
                    continue
                done.append((i, j))
                d, z, p = dwi.util.compare_aucs(Auc_bs[i], Auc_bs[j])
                s = '{pi:{l}}  {pj:{l}}  {d:+.4f}  {z:+.4f}  {p:.4f}'
                print(s.format(pi=param_i, pj=param_j, d=d, z=z, p=p,
                               l=params_maxlen))

    # Plot the ROCs.
    if args.figure:
        if args.verbose > 1:
            print('Plotting to {}...'.format(args.figure))
        dwi.plot.plot_rocs(X, Y, params=Params, autoflip=args.autoflip,
                           outfile=args.figure)


if __name__ == '__main__':
    main()
