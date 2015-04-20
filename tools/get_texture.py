#!/usr/bin/env python2

"""Calculate texture properties for a masked area."""

#TODO pmap normalization for GLCM
#TODO Gabor clips pmap, only suitable for ADCm
#TODO GLCM uses only length 1

from __future__ import division
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
        #('haralick', dwi.texture.haralick_map),
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
    p.add_argument('--slices', default='maxfirst',
            help='slice selection (maxfirst, max, all)')
    p.add_argument('--winsizes', metavar='I', nargs='*', type=int, default=[5],
            help='window side lengths')
    p.add_argument('--portion', type=float, required=False, default=0,
            help='portion of selected voxels required for each window')
    p.add_argument('--output', metavar='FILENAME',
            help='output ASCII file')
    args = p.parse_args()
    return args

def max_mask(mask, winsize):
    """Return a mask that has the voxels selected that have the maximum number
    of surrounding voxels selected in the original mask.
    """
    d = collections.defaultdict(list)
    for pos, win in dwi.util.sliding_window(mask, winsize, mask=mask):
        d[np.count_nonzero(win)].append(pos)
    r = np.zeros_like(mask)
    for pos in d[max(d)]:
        r[pos] = True
    return r

def portion_mask(mask, winsize, portion=1., resort_to_max=True):
    """Return a mask that selects (only) voxels that have the window at each
    selected voxel origin up to a minimum portion in the original mask selected
    (1 means the whole window must be selected, 0 gives the original mask).

    If resort_to_max is true, the window with maximum number of selected voxels
    is used in case the resulting mask would otherwise be empty.
    """
    r = np.zeros_like(mask)
    for pos, win in dwi.util.sliding_window(mask, winsize, mask=mask):
        if np.count_nonzero(win) / win.size >= portion:
            r[pos] = True
    if resort_to_max and np.count_nonzero(r) == 0:
        r = max_mask(mask, winsize)
    return r


args = parse_args()
if args.verbose:
    print 'Reading data...'
data = dwi.dataset.dataset_read_samples([(args.case, args.scan)])
dwi.dataset.dataset_read_pmaps(data, args.pmapdir, [args.param])
mask = dwi.mask.read_mask(args.mask)

img = data[0]['image']
if isinstance(mask, dwi.mask.Mask):
    mask = mask.convert_to_3d(img.shape[0])

if args.slices == 'maxfirst':
    slice_indices = [mask.max_slices()[0]]
elif args.slices == 'max':
    slice_indices = mask.max_slices()
elif args.slices == 'all':
    slice_indices = mask.selected_slices()
else:
    raise Exception('Invalid slice set specification', args.slices)

img_slices = img[slice_indices,:,:,0]
mask_slices = mask.array[slice_indices]
winshapes = [(1,w,w) for w in args.winsizes]
pmasks = [portion_mask(mask_slices, w, args.portion) for w in winshapes]

if args.verbose > 1:
    d = dict(s=img.shape, i=slice_indices, n=np.count_nonzero(mask_slices),
            w=args.winsizes)
    print 'Image: {s}, slice: {i}, voxels: {n}, windows: {w}'.format(**d)

if args.verbose:
    print 'Calculating texture features...'
feats = []
featnames = []
for method, call in METHODS.items():
    if args.methods is None or method in args.methods:
        if args.verbose > 1:
            print method
        for winsize, pmask_slices in zip(args.winsizes, pmasks):
            tmaps_all = None
            for img_slice, pmask_slice in zip(img_slices, pmask_slices):
                tmaps, names = call(img_slice, winsize, mask=pmask_slice)
                tmaps = tmaps[:,pmask_slice]
                if tmaps_all is None:
                    tmaps_all = tmaps
                else:
                    np.concatenate((tmaps_all, tmaps), axis=-1)
            feats += map(np.mean, tmaps_all)
            names = ['{w}-{n}'.format(w=winsize, n=n) for n in names]
            featnames += names

if args.verbose:
    print 'Writing %s features to %s' % (len(feats), args.output)
dwi.asciifile.write_ascii_file(args.output, [feats], featnames)
