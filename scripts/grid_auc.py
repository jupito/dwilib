#!/usr/bin/python3

"""Ad-hoc script to call pcorr.py. The grid data should be restructured."""

import argparse
import os
from itertools import product


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
                   help='be more verbose')
    p.add_argument('directory',
                   help='input directory')
    return p.parse_args()


def do_method(directory, lesion_thresholds, name, winsizes, nfeats):
    for l, w, f in product(lesion_thresholds, winsizes, range(nfeats)):
        do_feat(directory, l, name, w, f)


def do_feat(directory, lesion_threshold, name, winsize, feat):
    prostate_threshold = 0.5
    path = '{d}/{n}-{w}/*-*-{f}.txt'
    path = path.format(d=directory, n=name, w=winsize, f=feat)
    cmd = 'pcorr.py --thresholds {pt} {lt} {path}'
    cmd = cmd.format(pt=prostate_threshold, lt=lesion_threshold, path=path)
    exit_status = os.system(cmd)
    assert exit_status == 0, (cmd, exit_status)
    return exit_status


def main():
    args = parse_args()
    lesion_thresholds = [x/10 for x in range(1, 10)]
    if 'DWI' in args.directory:
        rng = range(3, 16, 2)
    elif 'T2' in args.directory:
        rng = range(3, 30, 4)
    else:
        raise Exception('Unknown input')
    methods = [
        ('raw', [1], 1),
        ('gabor', rng, 48),
        ('glcm', rng, 30),
        # ('glcm_mbb', ['mbb'], 30),
        ('haar', rng, 24),
        # ('hog', rng, 1),
        ('hu', rng, 7),
        ('lbp', rng, 10),
        # ('sobel', [3], 2),
        # ('stats_all', ['all'], 19),
        # ('zernike', rng, 25),
    ]
    for method in methods:
        do_method(args.directory, lesion_thresholds, *method)


if __name__ == '__main__':
    main()
