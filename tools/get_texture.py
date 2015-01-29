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
    p.add_argument('--output', '-o', metavar='FILENAME',
            help='output ASCII file')
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
af = dwi.asciifile.AsciiFile(args.input)
img = af.a.reshape((5,5))
if args.verbose:
    print img.shape

img = normalize(img)
props = dwi.texture.get_coprops_img(img)

if args.verbose:
    print 'Writing (%s) to %s' % (', '.join(dwi.texture.PROPNAMES), args.output)
with open(args.output, 'w') as f:
    f.write(' '.join(map(str, props)))

#if args.total:
#    data['cancer_coprops'] = dwi.texture.get_coprops(data['cancer_rois'])
#    data['normal_coprops'] = dwi.texture.get_coprops(data['normal_rois'])
#    data['other_coprops'] = dwi.texture.get_coprops(data['other_rois'])
#    import scipy
#    import scipy.stats
#    for i in range(len(dwi.texture.PROPNAMES)):
#        print scipy.stats.spearmanr(data['cancer_coprops'][i], data['normal_coprops'][i])
#    aucs = get_texture_aucs(data)
#    for propname, auc in zip(dwi.texture.PROPNAMES, aucs):
#        print propname, auc
#
#tuples = zip(data['images'], data['cases'], data['scans'])
#for img, case, scan in tuples:
#    if case in args.cases:
#        title = '%s %s' % (case, scan)
#        draw_props(img, title, args.step)
