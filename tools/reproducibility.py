#!/usr/bin/python

"""Calculate reproducibility coefficients for parametric maps. Input consists
of pmap scan pairs grouped together.
"""

from __future__ import absolute_import, division, print_function
import argparse
import glob
import os.path
import re

import numpy as np

import dwi.files
import dwi.plot
import dwi.stats
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-v', '--verbose', action='count',
                   help='be more verbose')
    p.add_argument('-p', '--patients',
                   help='patients file')
    p.add_argument('-b', '--nboot', type=int, default=2000,
                   help='number of bootstraps')
    p.add_argument('--voxel', default='0',
                   help='index of voxel to use, or mean or median')
    p.add_argument('-m', '--pmaps', nargs='+', required=True,
                   help='pmap files, pairs grouped together')
    p.add_argument('--figdir',
                   help='figure output directory')
    return p.parse_args()


def as_pairs(seq):
    """Return sequence split in two, each containing every second item."""
    if len(seq) % 2:
        raise ValueError('Sequence length not even: {}'.format(len(seq)))
    return seq[0::2], seq[1::2]


def plot(values, param, figdir):
    """Plot a parameter; its two baselines and their differences.

    This function was originally made in order to find outliers.
    """
    baselines = np.asarray(as_pairs(values))
    n = len(baselines[0])
    it = dwi.plot.generate_plots(ncols=3, titles=(param,)*3,
                                 xlabels=('index',)*3,
                                 ylabels=('value', 'difference', 'value'),
                                 path='{}/{}.png'.format(figdir, param))
    for i, plt in enumerate(it):
        if i == 0:
            # Plot absolute values.
            x = range(2 * n)
            y = sorted(values)
            c = ('lightgray', 'white') * n
            plt.scatter(x, y, c=c)
            plt.axis((min(x), max(x), min(y), max(y)))
        elif i == 1:
            # Plot differences.
            x = range(n)
            y = sorted(np.abs(baselines[0] - baselines[1]))
            plt.scatter(x, y, c='lightgray')
            plt.axis((min(x), max(x), min(y), max(y)))
        elif i == 2:
            # Plot sample pairs as bars.
            def key(pair):
                a, b = pair
                return abs(a-b)
            pairs = baselines.T
            pairs = np.asarray(sorted(pairs, key=key))
            left = range(n)
            bottom = np.min(pairs, axis=1)
            height = np.max(pairs, axis=1) - bottom
            plt.bar(left, height, bottom=bottom, color='lightgray')
            # bottom = range(n)
            # left = np.min(pairs, axis=1)
            # width = np.max(pairs, axis=1) - left
            # plt.barh(bottom, width, left=left, color='lightgray')
            plt.axis('tight')


def glob_if_needed(filenames):
    """Workaround for platforms without shell-level globbing."""
    if len(filenames) == 1:
        lst = glob.glob(filenames[0])
        if len(lst) > 0:
            filenames = lst
    return filenames


def sort_pmapfiles(paths):
    """Kludge to sort input files for platforms where globbing leaves them
    unsorted. Requires certain format.
    """
    def sortkey(path):
        head, tail = os.path.split(path)
        root, ext = os.path.splitext(tail)
        c, s, l = root.split('_')
        return head, c, l, s
    return sorted(paths, key=sortkey)


def parse_filename(filename):
    """Parse input filename formatted as 'num_name_hB_[12][ab]_*'."""
    # m = re.match(r'(\d+)_([\w_]+)_[^_]*_(\d\w)_', filename)
    m = re.search(r'(\d+)_(\w*)_?(\d\w)_', filename)
    if m is None:
        raise Exception('Cannot parse filename: {}'.format(filename))
    num, name, scan = m.groups()
    return int(num), name.lower(), scan.lower()


def scan_pairs(afs):
    """Check that the ascii files are correctly paired as scan baselines.
    Return list of (patient number, scan 1, scan 2) tuples.
    """
    baselines = as_pairs(afs)
    r = []
    for af1, af2 in zip(*baselines):
        num1, _, scan1 = parse_filename(af1.basename)
        num2, _, scan2 = parse_filename(af2.basename)
        if num1 != num2 or scan1[0] != scan2[0]:
            raise Exception('Not a pair: {}, {}'.format(af1.basename,
                                                        af2.basename))
        r.append((num1, scan1, scan2))
    return r


def scan_in_patients(patients, num, scan):
    """Is this scan listed in the patients sequence?"""
    return any(num == p.num and scan in p.scans for p in patients)


def load_files(patients, filenames, pairs=False):
    """Load pmap files. If pairs=True, require scan pairs together."""
    pmapfiles = []
    for f in filenames:
        num, _, scan = parse_filename(os.path.basename(f))
        if patients is None or scan_in_patients(patients, num, scan):
            pmapfiles.append(f)
    afs = [dwi.asciifile.AsciiFile(x) for x in pmapfiles]
    if pairs:
        scan_pairs(afs)
    pmaps = [af.a for af in afs]
    pmaps = np.array(pmaps)
    params = afs[0].params()
    assert pmaps.shape[-1] == len(params), 'Parameter name mismatch.'
    return pmaps, params


def main():
    args = parse_args()
    if args.patients:
        patients = dwi.files.read_patients_file(args.patients)
    else:
        patients = None

    paths = glob_if_needed(args.pmaps)
    paths = sort_pmapfiles(paths)  # XXX: Temporary kludge.
    pmaps, params = load_files(patients, paths, pairs=True)

    # Select voxel to use.
    if args.voxel == 'mean':
        X = np.mean(pmaps, axis=1)  # Use mean voxel.
    elif args.voxel == 'median':
        X = np.median(pmaps, axis=1)  # Use median voxel.
    else:
        i = int(args.voxel)
        X = pmaps[:, i, :]  # Use single voxel only.

    if args.verbose > 1:
        s = 'Samples: {}, features: {}, voxel: {}, bootstraps: {}'
        print(s.format(X.shape[0], X.shape[1], args.voxel, args.nboot))

    # Print results for each parameter.
    if args.verbose:
        print('# avg[lower-upper] '
              'msd/avg CI/avg wCV CoR/avg '
              'ICC bsICC[lower-upper] '
              'param')
    output = (
        '{avg:.8f}[{avg_ci1:.8f}-{avg_ci2:.8f}] '
        '{msdr:.4f} {cir:.4f} {wcv:.4f} {corr:.4f} '
        '{icc:5.2f} {icc_bs:5.2f}[{icc_ci1:5.2f}-{icc_ci2:5.2f}] '
        '{param}'
        )
    # output = (
    #     '{corr:7.2f} '
    #     '{icc:5.2f} {icc_bs:5.2f}[{icc_ci1:5.2f}-{icc_ci2:5.2f}] '
    #     '{param}'
    #     )
    skipped_params = 'SI0N C RMSE'.split()
    for values, param in zip(X.T, params):
        if param in skipped_params:
            continue
        if dwi.util.all_equal(values):
            continue
        if args.figdir:
            plot(values, param, args.figdir)
        baselines = as_pairs(values)
        d = dict(param=param)
        d.update(dwi.stats.repeatability_coeff(*baselines, avgfun=np.median))
        d['msdr'] = d['msd']/d['avg']
        d['cir'] = d['ci']/d['avg']
        d['corr'] = d['cor']/d['avg']
        d['icc'] = dwi.stats.icc(baselines)
        (d['icc_bs'], d['icc_ci1'],
         d['icc_ci2']) = dwi.stats.bootstrap_icc(baselines, nboot=args.nboot)
        print(output.format(**d))


if __name__ == '__main__':
    main()
