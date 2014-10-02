#!/usr/bin/env python2

"""Convert ROI mask files from MATLAB format to ASCII."""

import os
import sys

import scipy.io

def line_to_text(line):
    return ''.join(map(str, line))

def mask_to_text(mask):
    return '\n'.join(map(line_to_text, mask))


infiles = sys.argv[1:]
for infile in infiles:
    mat = scipy.io.loadmat(infile, struct_as_record=False)
    roislices = map(int, mat['ROIslices'][0])
    rois = mat['ROIs'][0]
    for i, roi in enumerate(rois):
        try:
            roislice = roislices[i]
        except IndexError:
            roislice = roislices[0]
        struct = roi[0,0]
        name = struct.name[0]
        #vol = struct.vol[0,0]
        outfile = '%s_%i_%s.mask' % (os.path.basename(infile), i+1, name)
        print 'Writing %s' % outfile
        with open(outfile, 'wb') as f:
            f.write('slice: %i\n' % roislice)
            f.write(mask_to_text(struct.mask))
