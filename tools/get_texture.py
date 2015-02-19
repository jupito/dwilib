#!/usr/bin/env python2

"""Calculate texture properties for a ROI."""

import argparse
import glob
import re
import numpy as np
import skimage

import dwi.asciifile
import dwi.texture
import dwi.util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument('--verbose', '-v', action='count',
            help='increase verbosity')
    p.add_argument('--input', '-i', metavar='FILENAME', required=True,
            nargs='+', default=[], help='input ASCII file')
    p.add_argument('--basic', action='store_true',
            help='get basic texture properties')
    p.add_argument('--lbp', action='store_true',
            help='get local binary patterns')
    args = p.parse_args()
    return args

def normalize(pmap):
    """Normalize images within given range and convert to byte maps."""
    in_range = (0, 0.03)
    pmap = skimage.exposure.rescale_intensity(pmap, in_range=in_range)
    pmap = skimage.img_as_ubyte(pmap)
    return pmap

def plot(img):
    import pylab as pl
    pl.rcParams['image.cmap'] = 'gray'
    pl.rcParams['image.aspect'] = 'equal'
    pl.rcParams['image.interpolation'] = 'none'
    pl.imshow(img)
    pl.show()


args = parse_args()
for infile in args.input:
    af = dwi.asciifile.AsciiFile(infile)
    basename = af.basename
    img = af.a.reshape((5,5))
    if args.verbose > 1:
        print 'Image shape: %s' % (img.shape,)

    # Write basic properties.
    if args.basic:
        img_normalized = normalize(img)
        propnames = ['median', 'mean', 'stddev']
        props = [np.median(img), np.mean(img), np.std(img)]
        propnames += dwi.texture.PROPNAMES
        props += dwi.texture.get_coprops_img(img_normalized)

        outfile = 'props_%s' % basename
        if args.verbose:
            print 'Writing (%s) to %s' % (', '.join(propnames), outfile)
        with open(outfile, 'w') as f:
            f.write(' '.join(map(str, props)) + '\n')

    # Write LBP properties.
    if args.lbp:
        lbp_freq_data, n_patterns = dwi.texture.get_lbp_freqs(img)
        lbp_freq_data.shape = (-1, n_patterns)

        outfile = 'lbpf_%s' % basename
        if args.verbose:
            print 'Writing LBP frequencies to %s' % outfile
        with open(outfile, 'w') as f:
            for patterns in lbp_freq_data:
                f.write(' '.join(map(str, patterns)) + '\n')
