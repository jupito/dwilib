#!/usr/bin/env python2

"""Calculate reproducibility coefficients for parametric maps by Bland-Altman
analysis. Input consists of pmap scan pairs grouped together.
"""

from __future__ import absolute_import, division, print_function
import argparse
import glob
import os.path

import numpy as np

import dwi.files
import dwi.patient
import dwi.plot
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


def mean_squared_difference(a1, a2):
    """Return mean squared difference of two arrays."""
    a1 = np.asanyarray(a1)
    a2 = np.asanyarray(a2)
    assert len(a1) == len(a2), 'Array length mismatch'
    n = len(a1)
    ds = a1-a2
    sds = ds**2
    msd = np.sqrt(sum(sds) / (n-1))
    return msd


def coefficients(a1, a2, avgfun=np.mean):
    """Return average, average squared difference, confidence interval,
    within-patient coefficient of variance, coefficient of repeatability."""
    a1 = np.asanyarray(a1)
    a2 = np.asanyarray(a2)
    assert len(a1) == len(a2), 'Array length mismatch'
    n = len(a1)
    a = np.concatenate((a1, a2))
    avg = avgfun(a)
    avg_ci1, avg_ci2 = dwi.util.ci(a)
    msd = mean_squared_difference(a1, a2)
    ci = 1.96*msd / np.sqrt(n)
    wcv = (msd/np.sqrt(2)) / avg
    cor = 1.96*msd
    d = dict(avg=avg, avg_ci1=avg_ci1, avg_ci2=avg_ci2, msd=msd, ci=ci,
             wcv=wcv, cor=cor)
    return d


def icc(baselines):
    """Calculate ICC(3,1) intraclass correlation.

    See Shrout, Fleiss 1979: Intraclass Correlations: Uses in Assessing Rater
    Reliability.
    """
    data = np.array(baselines)
    k, n = data.shape  # Number of raters, targets.
    mpt = np.mean(data, axis=0)  # Mean per target.
    mpr = np.mean(data, axis=1)  # Mean per rater.
    tm = np.mean(data)  # Total mean.
    wss = sum(sum((data-mpt)**2))  # Within-target sum of squares.
    # wms = wss / (n * (k-1))  # Within-target mean of squares.
    rss = sum((mpr-tm)**2) * n  # Between-rater sum of squares.
    # rms = rss / (k-1)  # Between-rater mean of squares.
    bss = sum((mpt-tm)**2) * k  # Between-target sum of squares.
    bms = bss / (n-1)  # Between-target mean of squares.
    ess = wss - rss  # Residual sum of squares.
    ems = ess / ((k-1) * (n-1))  # Residual mean of squares.
    icc31 = (bms - ems) / (bms + (k-1)*ems)
    return icc31


def bootstrap_icc(baselines, nboot=2000):
    """Calculate ICC bootstrapped target-wise. Return mean and confidence
    intervals.
    """
    data = np.array(baselines)
    values = np.zeros((nboot,))
    for i in xrange(nboot):
        sample = dwi.util.resample_bootstrap_single(data.T).T
        values[i] = icc(sample)
    mean = np.mean(values)
    ci1, ci2 = dwi.util.ci(values)
    return mean, ci1, ci2


def plot(values, param, figdir):
    """Plot a parameter; its two baselines and their differences.

    This function was originally made in order to find outliers.
    """
    baselines = np.asarray(dwi.util.pairs(values))
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


def sort_pmapfiles(paths):
    """Kludge to sort input files for Windows without shell doing it. Requires
    certain format.
    """
    def sortkey(path):
        head, tail = os.path.split(path)
        root, ext = os.path.splitext(tail)
        c, s, l = root.split('_')
        return head, c, l, s
    return sorted(paths, key=sortkey)


def scan_in_patients(patients, num, scan):
    """Is this scan listed in the patients sequence?"""
    return any(num == p.num and scan in p.scans for p in patients)


def load_files(patients, filenames, pairs=False):
    """Load pmap files. If pairs=True, require scan pairs together."""
    pmapfiles = []
    if len(filenames) == 1:
        # Workaround for platforms without shell-level globbing.
        l = glob.glob(filenames[0])
        if len(l) > 0:
            filenames = l
    for f in filenames:
        num, scan = dwi.util.parse_num_scan(os.path.basename(f))
        if patients is None or scan_in_patients(patients, num, scan):
            pmapfiles.append(f)
    afs = [dwi.asciifile.AsciiFile(x) for x in pmapfiles]
    if pairs:
        dwi.util.scan_pairs(afs)
    ids = [dwi.util.parse_num_scan(af.basename) for af in afs]
    pmaps = [af.a for af in afs]
    pmaps = np.array(pmaps)
    params = afs[0].params()
    assert pmaps.shape[-1] == len(params), 'Parameter name mismatch.'
    return pmaps, ids, params


def main():
    args = parse_args()
    if args.patients:
        patients = dwi.files.read_patients_file(args.patients)
    else:
        patients = None

    paths = sort_pmapfiles(args.pmaps)  # XXX: Temporary kludge.
    pmaps, _, params = load_files(patients, paths, pairs=True)

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

    # Print coefficients for each parameter.
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
    output = (
        '{corr:7.2f} '
        '{icc:5.2f} {icc_bs:5.2f}[{icc_ci1:5.2f}-{icc_ci2:5.2f}] '
        '{param}'
        )
    skipped_params = 'SI0N C RMSE'.split()
    for values, param in zip(X.T, params):
        if param in skipped_params:
            continue
        if dwi.util.all_equal(values):
            continue
        if args.figdir:
            plot(values, param, args.figdir)
        baselines = dwi.util.pairs(values)
        d = dict(param=param)
        d.update(coefficients(*baselines, avgfun=np.median))
        d['msdr'] = d['msd']/d['avg']
        d['cir'] = d['ci']/d['avg']
        d['corr'] = d['cor']/d['avg']
        d['icc'] = icc(baselines)
        (d['icc_bs'], d['icc_ci1'],
         d['icc_ci2']) = bootstrap_icc(baselines, nboot=args.nboot)
        print(output.format(**d))


if __name__ == '__main__':
    main()
