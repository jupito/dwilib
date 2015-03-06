#!/usr/bin/env python2

"""Mask tool."""

import argparse
import os.path

import numpy as np

import dwi.mask
import dwi.util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument('-v', '--verbose', action='count',
            help='increase verbosity')
    p.add_argument('--input', '-i', metavar='INFILE', required=True,
            help='input mask file')
    p.add_argument('--subregion', '-s', metavar='OUTFILE',
            help='output subregion file')
    p.add_argument('--pad', metavar='I', type=int, default=10,
            help='subregion padding size')
    args = p.parse_args()
    return args

def write_subregion(mask, pad, infile, filename):
    """Write a subregion file from mask."""
    bb = mask.bounding_box(pad=(np.inf, pad, pad))
    comment = '%s, %i' % (os.path.basename(infile), pad)
    print 'Writing subregion to %s with padding of %s...' % (filename, pad)
    dwi.util.write_subregion_file(filename, bb, comment=comment)

args = parse_args()
mask = dwi.mask.read_mask(args.input)

selected_slices = list(mask.selected_slices())
mbb = mask.bounding_box()
d = dict(infile=args.input, shape=mask.shape, mbb=mbb,
        mbb_shape=dwi.util.subwindow_shape(mbb), nsel=mask.n_selected(),
        nsl=len(selected_slices), sl=selected_slices)
print 'mask: {infile}\n'\
        'minimum bounding box: {mbb_shape}: {mbb}\n'\
        'selected voxels: {nsel}\n'\
        'selected slices: {nsl}: {sl}'.format(**d)

if args.subregion:
    write_subregion(mask, args.pad, args.input, args.subregion)