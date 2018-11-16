#!/usr/bin/python3

"""Calculate ROC AUC for parametric maps vs. Gleason scores. Optionally compare
AUCs and draw the ROC curves into a file.
"""

import argparse
import numpy as np

import dwi.files
import dwi.patient
import dwi.plot
import dwi.stats
import dwi.util
from dwi.compat import collect_data


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-v', '--verbose', action='count',
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
    p.add_argument('--scan', type=int, default=None,
                   help='index of scan to use, if not all')
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
    """Main."""
    args = parse_args()
    if args.normalvoxel is not None and args.voxel != 'all':
        raise ValueError('Argument --normalvoxel implies --voxel=all')

    patients = dwi.files.read_patients_file(args.patients)
    if args.scan is not None:
        dwi.patient.keep_scan(patients, args.scan)
    dwi.patient.label_lesions(patients, thresholds=[args.threshold])
    X, Y, params = collect_data(patients, args.pmapdir, voxel=args.voxel,
                                multiroi=args.multilesion, dropok=args.dropok,
                                normalvoxel=args.normalvoxel,
                                verbose=args.verbose)

    # Print AUCs and bootstrapped AUCs.
    if args.verbose > 1:
        print('# param  AUC  AUC_BS_mean  lower  upper')
    Auc_bs = []
    params_maxlen = max(len(x) for x in params)
    d = dict(l=params_maxlen)
    for x, y, param in zip(X, Y, params):
        d['param'] = param
        s = '{param:{l}}  {auc:.3f}'
        if np.any(np.isnan(x)):
            d['auc'] = np.nan
            print(s.format(**d))
            continue
        # Scaling may be required for correct results.
        x = dwi.stats.scale_standard(x)
        d.update(dwi.stats.roc_auc(y, x, autoflip=args.autoflip,
                                   nboot=args.nboot))
        if args.nboot:
            Auc_bs.append(d.pop('aucs'))
            s += '  {ci1:.3f}  {ci2:.3f}'
        print(s.format(**d))

    # Print bootstrapped AUC comparisons.
    if args.nboot and args.compare:
        if args.verbose > 1:
            print('# param1  param2  diff  Z  p')
        done = []
        for i, param_i in enumerate(params):
            for j, param_j in enumerate(params):
                if i == j or (i, j) in done or (j, i) in done:
                    continue
                done.append((i, j))
                d, z, p = dwi.stats.compare_aucs(Auc_bs[i], Auc_bs[j])
                s = '{pi:{l}}  {pj:{l}}  {d:+.4f}  {z:+.4f}  {p:.4f}'
                print(s.format(pi=param_i, pj=param_j, d=d, z=z, p=p,
                               l=params_maxlen))

    # Plot the ROCs.
    if args.figure:
        if args.verbose > 1:
            print('Plotting to {}...'.format(args.figure))
        dwi.plot.plot_rocs(X, Y, params=params, autoflip=args.autoflip,
                           outfile=args.figure)


if __name__ == '__main__':
    main()
