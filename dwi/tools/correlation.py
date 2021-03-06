#!/usr/bin/python3

"""Calculate correlation for parametric maps vs. Gleason scores."""

import argparse

import dwi.files
import dwi.patient
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
    p.add_argument('--thresholds', nargs='*', default=['3+3'],
                   help='classification thresholds (group maximums)')
    p.add_argument('--voxel', default='all',
                   help='index of voxel to use, or all, sole, mean, median')
    p.add_argument('--multilesion', action='store_true',
                   help='use all lesions, not just first for each')
    p.add_argument('--dropok', action='store_true',
                   help='allow dropping of files not found')
    return p.parse_args()


def main():
    """Main."""
    args = parse_args()

    patients = dwi.files.read_patients_file(args.patients)
    dwi.patient.label_lesions(patients, thresholds=args.thresholds)
    X, Y, params = collect_data(patients, args.pmapdir, voxel=args.voxel,
                                multiroi=args.multilesion, dropok=args.dropok,
                                verbose=args.verbose)

    # Print correlations.
    if args.verbose > 1:
        print('# param  r  p  lower  upper')
    params_maxlen = max(len(x) for x in params)
    for x, y, param in zip(X, Y, params):
        d = dict(param=param, len=params_maxlen, f='.3f')
        d.update(dwi.stats.correlation(x, y))
        if args.verbose:
            s = '{param:{len}}  {r:+{f}}  {p:{f}}  {lower:+{f}}  {upper:+{f}}'
        else:
            s = '{r:+.3f}'
        print(s.format(**d))


if __name__ == '__main__':
    main()
