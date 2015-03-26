#!/usr/bin/env python2

"""Zero out all voxels in multi-file DICOM image according to a mask."""

import argparse
import collections
import os
import numpy as np

import dicom

import dwi.dwimage
import dwi.mask
import dwi.util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-v', '--verbose', action='count',
            help='increase verbosity')
    p.add_argument('--image', '-i', metavar='IMAGE', required=True,
            help='input DICOM image directory')
    p.add_argument('--mask', '-m', metavar='MASK', required=True,
            help='mask path')
    args = p.parse_args()
    return args

def get_slices(dirname):
    """Return filename lists indexed by slice position."""
    filenames = os.listdir(dirname)
    pathnames = [os.path.join(dirname, f) for f in filenames]
    orientation = None
    shape = None
    positions = collections.defaultdict(list)
    for pathname in pathnames:
        ds = dicom.read_file(pathname)
        if not 'PixelData' in ds:
            continue
        orientation = orientation or ds.ImageOrientationPatient
        if ds.ImageOrientationPatient != orientation:
            raise Exception("Orientation mismatch.")
        shape = shape or ds.pixel_array.shape
        if ds.pixel_array.shape != shape:
            raise Exception("Shape mismatch.")
        position = tuple(map(float, ds.ImagePositionPatient))
        positions[position].append(pathname)
    slices = [positions[k] for k in sorted(positions.keys())]
    return slices

def mask_out_slice(mask_slice, pathname):
    """Mask out a slice (set all unselected voxels to zero).
    
    See https://code.google.com/p/pydicom/wiki/WorkingWithPixelData
    """
    ds = dicom.read_file(pathname)
    if mask_slice.shape != ds.pixel_array.shape:
        raise Exception('Slice shape mismatch')
    ds.pixel_array *= mask_slice
    ds.PixelData = ds.pixel_array.tostring() # Must be written to PixelData.
    ds.save_as(pathname)


args = parse_args()
mask = dwi.mask.read_mask(args.mask)
slices = get_slices(args.image)

if mask.shape()[0] != len(slices):
    raise Exception('Number of slices mismatch.')

for mask_slice, pathnames, i in zip(mask.array, slices, range(len(slices))):
    for p in pathnames:
        if args.verbose:
            d = dict(i=i, n_selected=np.sum(mask_slice != False), p=p)
            print 'Slice {i:d} ({n_selected:d}): {p:s}'.format(**d)
        mask_out_slice(mask_slice, p)
