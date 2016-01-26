#!/usr/bin/env python2

"""Plot histograms of images. Possible nans and infinities are ignored."""

from __future__ import absolute_import, division, print_function
import argparse

import numpy as np

import dwi.files
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
    p.add_argument('--fig', required=True,
                   help='output figure file')
    p.add_argument('--smooth', action='store_true',
                   help='smoothen the histogram by spline interpolation')
    return p.parse_args()


def histogram(a, m1=None, m2=None, bins=20):
    """Create histogram from data between (m1, m2), with bin centers."""
    a = np.asarray(a)
    if m1 is not None:
        a = a[a > m1]
    if m2 is not None:
        a = a[a < m2]
    mn, mx = a.min(), a.max()
    # bins = a.size / 1000000
    hist, bin_edges = np.histogram(a, bins=bins, density=True)
    bin_centers = [np.mean(t) for t in zip(bin_edges, bin_edges[1:])]
    return hist, bin_centers, mn, mx


def plot_histograms(Histograms, outfile, smooth=False):
    import pylab as pl
    ncols, nrows = len(Histograms), 1
    fig = pl.figure(figsize=(ncols*6, nrows*6))
    # pl.yscale('log')
    for i, histograms in enumerate(Histograms):
        if histograms:
            fig.add_subplot(1, len(Histograms), i+1)
            minmin, maxmax = None, None
            for hist, bins, mn, mx in histograms:
                x, y = bins, hist
                if smooth:
                    import scipy.interpolate
                    x_smooth = np.linspace(min(x), max(x), 300)
                    y_smooth = scipy.interpolate.spline(x, y, x_smooth)
                    y_smooth[y_smooth < 0] = 0
                    x, y = x_smooth, y_smooth
                pl.plot(x, y)
                if minmin is None:
                    minmin = mn
                if maxmax is None:
                    maxmax = mx
                minmin = min(minmin, mn)
                maxmax = max(maxmax, mx)
            pl.title('[{:.5g}, {:.5g}]'.format(minmin, maxmax))
    pl.tight_layout()
    print('Plotting to {}...'.format(outfile))
    pl.savefig(outfile, bbox_inches='tight')
    pl.close()


def main():
    args = parse_args()

    histograms = []
    histograms_std = []
    for path in args.input:
        img, _ = dwi.files.read_pmap(path)
        img = img[..., args.param]
        original_shape, original_size = img.shape, img.size
        img = img[np.isfinite(img)]
        if args.verbose:
            print('Read {s}, {t}, {fp:.1%}, {m:.4g}, {fn}, {p}'.format(
                s=original_shape, t=img.dtype, fp=img.size/original_size,
                m=np.mean(img), fn=dwi.util.fivenums(img), p=path))
        histograms.append(histogram(img, None, None))
        # cutoffs = img.min(), img.max()
        cutoffs = np.percentile(img, (0, 99))
        histograms_std.append(histogram(img, *cutoffs))
    plot_histograms([histograms, histograms_std], args.fig, smooth=args.smooth)

    # All together.
    # histograms = []
    # histograms_std = []
    # images = (dwi.files.read_pmap(x)[0].squeeze() for x in args.input)
    # img = np.zeros((0,))
    # for image in images:
    #     img = np.concatenate([img, image])
    # print(img.shape)
    # histograms.append(histogram(img, None, None))
    # histograms_std.append(histogram(img, img.min(), img.max()))
    # plot_histograms([histograms, histograms_std], args.fig+'-together.png')


if __name__ == '__main__':
    main()
