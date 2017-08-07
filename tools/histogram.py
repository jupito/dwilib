#!/usr/bin/python3

"""Plot histograms of images. Possible nans and infinities are ignored."""

import argparse
from collections import OrderedDict
import logging

import numpy as np
import pylab as pl
from scipy import interpolate

import dwi.files
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
                   help='increase verbosity')
    p.add_argument('--input', nargs='+',
                   help='input files')
    p.add_argument('--param', type=int, nargs='*',
                   help='image parameter index to use')
    p.add_argument('--fig', required=True,
                   help='output figure file')
    p.add_argument('--smooth', action='store_true',
                   help='smoothen the histogram by spline interpolation')
    return p.parse_args()


def histogram(a, m1=None, m2=None, inclusive=True, bins='doane'):
    """Create histogram from data between (m1, m2), with bin centers."""
    a = np.asarray(a)
    if m1 is not None:
        if inclusive:
            a = a[a >= m1]
        else:
            a = a[a > m1]
    if m2 is not None:
        if inclusive:
            a = a[a <= m2]
        else:
            a = a[a < m2]
    mn, mx = a.min(), a.max()
    hist, bin_edges = np.histogram(a, bins=bins, density=False)
    bin_centers = [np.mean(t) for t in zip(bin_edges, bin_edges[1:])]
    return hist, bin_centers, mn, mx


def smoothen(x, y):
    """Smoothen histogram."""
    x_smooth = np.linspace(min(x), max(x), 300)
    y_smooth = interpolate.spline(x, y, x_smooth)
    y_smooth[y_smooth < 0] = 0  # Don't let it dive negative.
    return x_smooth, y_smooth


def plot_histograms(Histograms, outfile, smooth=False):
    """Plot subfigures, each having several histograms bundled together."""
    nrows = len({x[0] for x in Histograms})
    ncols = len({x[1] for x in Histograms})
    # logging.warning('## %s ', [nrows, ncols])
    fig = pl.figure(figsize=(ncols*6, nrows*6))
    # pl.yscale('log')
    for i, ((param, rng), histograms) in enumerate(Histograms.items(), 1):
        # logging.warning('#### %s ', [i, param, rng, len(histograms)])
        if histograms:
            fig.add_subplot(nrows, ncols, i)
            minmin, maxmax = None, None
            for hist, bins, mn, mx in histograms:
                x, y = bins, hist
                if smooth:
                    x, y = smoothen(x, y)
                pl.plot(x, y)
                # pl.bar(x, y, width=x[1]-x[0])
                if minmin is None:
                    minmin = mn
                if maxmax is None:
                    maxmax = mx
                minmin = min(minmin, mn)
                maxmax = max(maxmax, mx)
            # s = '{}; {}; [{:.5g}, {:.5g}]'.format(len(histograms), rng,
            #                                       minmin, maxmax)
            # s = param + '; ' + s
            s = '{p}; {l}; {r}; [{min:.5g}, {max:.5g}]'
            d = dict(p=param, l=len(histograms), r=rng, min=minmin, max=maxmax)
            pl.title(s.format(**d))
    # pl.tight_layout()
    logging.info('Plotting to %s...', outfile)
    pl.savefig(outfile, bbox_inches='tight')
    pl.close()


def add_histograms(hists, path, img, param, ranges, verbose):
    """Add histograms for a file."""
    original_shape, original_size = img.shape, img.size
    img = img[dwi.util.bbox(img)]
    img = img[np.isfinite(img)]
    if np.any(img < 0):
        # negatives = img[img < 0]
        logging.warning('Image contains negatives: %s', path)
    if verbose:
        s = 'Read {s}, {t}, {fp:.1%}, {m:.4g}, {fn}, {param}, {p}'
        d = dict(s=original_shape, t=img.dtype, fp=img.size/original_size,
                 m=np.mean(img), fn=dwi.util.fivenums(img), param=param,
                 p=path))
        print(s.format(**d))
    for rng in ranges:
        if isinstance(rng, list):
            incl = True
        if isinstance(rng, tuple):
            incl = False
        m1, m2 = np.percentile(img, rng)
        key = (param, str(rng))
        hists.setdefault(key, []).append(histogram(img, m1, m2, incl))
    # hists[0].append(histogram(img, None, None))
    # hists[1].append(histogram(img, 0, 100))
    # hists[2].append(histogram(img, 0.1, 99.9))
    # hists[3].append(histogram(img, 1, 99))
    # hists[4].append(histogram(img, 2, 98))


def main():
    """Main."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    ranges = [[0, 100], (0, 100), [0, 99], (1, 95)]
    hists = OrderedDict()
    for path in args.input:
        img, attrs = dwi.files.read_pmap(path, params=args.param,
                                         dtype=np.float32)
        for i, param in enumerate(attrs['parameters']):
            add_histograms(hists, path, img[..., i], param, ranges,
                           args.verbose)
    plot_histograms(hists, args.fig, smooth=args.smooth)


if __name__ == '__main__':
    main()
