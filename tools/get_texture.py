#!/usr/bin/env python2

"""Calculate texture properties for a masked area."""

#TODO pmap normalization for GLCM
#TODO Gabor clips pmap, only suitable for ADCm
#TODO GLCM uses only length 1

import argparse
import collections
import glob
import re
import numpy as np

import dwi.asciifile
import dwi.dataset
import dwi.mask
import dwi.plot
import dwi.texture
import dwi.util

METHODS = collections.OrderedDict([
        ('stats', dwi.texture.stats_map),
        ('glcm', dwi.texture.glcm_map),
        ('haralick', dwi.texture.haralick_map),
        ('lbp', dwi.texture.lbp_freq_map),
        ('hog', dwi.texture.hog_map),
        ('gabor', dwi.texture.gabor_map),
        ('moment', dwi.texture.moment_map),
        ('haar', dwi.texture.haar_map),
        ])

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
            help='increase verbosity')
    p.add_argument('--pmapdir', default='results_Mono_combinedDICOM',
            help='input parametric map directory')
    p.add_argument('--param', default='ADCm',
            help='image parameter to use')
    p.add_argument('--case', type=int,
            help='case number')
    p.add_argument('--scan',
            help='scan identifier')
    p.add_argument('--mask',
            help='mask file to use')
    p.add_argument('--methods', metavar='METHOD', nargs='*',
            help='methods ({})'.format(', '.join(METHODS.keys())))
    p.add_argument('--winsize', type=int, default=5,
            help='window size length')
    p.add_argument('--output', metavar='FILENAME',
            help='output ASCII file')
    args = p.parse_args()
    return args


args = parse_args()
if args.verbose:
    print 'Reading data...'
data = dwi.dataset.dataset_read_samples([(args.case, args.scan)])
dwi.dataset.dataset_read_pmaps(data, args.pmapdir, [args.param])
mask = dwi.mask.read_mask(args.mask)

img = data[0]['image']
if isinstance(mask, dwi.mask.Mask):
    mask = mask.convert_to_3d(img.shape[0])

img_slice = mask.selected_slice(img)[:,:,0]
mask_slice = mask.array[mask.selected_slices()[0]]
winsize = args.winsize
if args.verbose > 1:
    print 'Image: %s, winsize: %s' % (img.shape, winsize)

if args.verbose:
    print 'Calculating texture features...'
feats = []
featnames = []
for method, call in METHODS.items():
    if args.methods is None or method in args.methods:
        if args.verbose > 1:
            print method
        tmap, names = call(img_slice, winsize, mask=mask_slice)
        feats += map(np.mean, tmap[:,mask_slice])
        featnames += names

if args.verbose:
    print 'Writing %s features to %s' % (len(feats), args.output)
dwi.asciifile.write_ascii_file(args.output, [feats], featnames)
