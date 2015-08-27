#!/usr/bin/env python2

"""Plot histograms of images. They may not contain NaN values."""

from __future__ import absolute_import, division, print_function
import argparse

import numpy as np

import dwi.files
import dwi.plot
import dwi.standardize
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
                   help='increase verbosity')
    p.add_argument('--input', nargs='+',
                   help='input files')
    p.add_argument('--param', type=int, default=0,
                   help='image parameter index to use')
    p.add_argument('--std',
                   help='standardization configuration file')
    p.add_argument('--fig', required=True,
                   help='output figure file')
    return p.parse_args()


def histogram(a, m1=None, m2=None, bins=20):
    """Create histogram from data between [m1, m2], with bin centers."""
    a = np.asarray(a)
    mn, mx = a.min(), a.max()
    if m1 is not None:
        a = a[a >= m1]
    if m2 is not None:
        a = a[a <= m2]
    # bins = a.size / 100000
    hist, bin_edges = np.histogram(a, bins=bins, density=True)
    bin_centers = [np.mean(t) for t in zip(bin_edges, bin_edges[1:])]
    return hist, bin_centers, mn, mx


def plot_histograms(Histograms, outfile):
    import pylab as pl
    ncols, nrows = len(Histograms), 1
    fig = pl.figure(figsize=(ncols*6, nrows*6))
    # pl.yscale('log')
    for i, histograms in enumerate(Histograms):
        if histograms:
            fig.add_subplot(1, len(Histograms), i+1)
            minmin, maxmax = 0, 0
            for hist, bins, mn, mx in histograms:
                pl.plot(bins, hist)
                minmin = min(minmin, mn)
                maxmax = max(maxmax, mx)
            pl.title('[{}, {}]'.format(minmin, maxmax))
    # if histograms2:
    #     fig.add_subplot(1, 2, 2)
    #     for hist, bins in histograms2:
    #         pl.plot(bins, hist)
    pl.tight_layout()
    print('Plotting to {}...'.format(outfile))
    pl.savefig(outfile, bbox_inches='tight')
    pl.close()


def main():
    args = parse_args()

    std_cfg = None
    if args.std:
        std_cfg = dwi.standardize.read_std_cfg(args.std)
        for k, v in std_cfg.items():
            print(k, v)

    histograms = []
    histograms_std = []
    for path in args.input:
        img, _ = dwi.files.read_pmap(path)
        img = img[..., args.param]
        if args.verbose:
            print('Read {s} from {p}...'.format(s=img.shape, p=path))
        if std_cfg:
            img_std = dwi.standardize.standardize(img, std_cfg)
        histograms.append(histogram(img, None, None))
        if std_cfg:
            s1, s2 = std_cfg['s1'], std_cfg['s2']
            histograms_std.append(histogram(img_std, s1, s2))
    plot_histograms([histograms, histograms_std], args.fig)


if __name__ == '__main__':
    main()
