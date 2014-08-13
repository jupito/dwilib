#!/usr/bin/env python2

"""Convert ROI mask files from MATLAB format to ASCII."""

import sys

import scipy.io

def line_to_text(line):
    return ''.join(map(str, line))

def mask_to_text(mask):
    return '\n'.join(map(line_to_text, mask))


filename = sys.argv[1]
mat = scipy.io.loadmat(filename, struct_as_record=False)

roislices = map(str, mat['ROIslices'][0])
print 'ROIslices: {}'.format(' '.join(roislices))

for roi in mat['ROIs'][0]:
    struct = roi[0,0]
    print 'name: {}'.format(struct.name[0])
    print 'vol: {}'.format(struct.vol[0,0])
    print 'shape: {}'.format(struct.mask.shape)
    print mask_to_text(struct.mask)
