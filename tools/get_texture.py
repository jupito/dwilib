#!/usr/bin/env python2

"""Calculate texture properties for a ROI."""

import argparse
import glob
import re
import numpy as np
import skimage

import dwi.plot
import dwi.texture
import dwi.util
import dwi.dwimage

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument('--verbose', '-v', action='count',
            help='increase verbosity')
    p.add_argument('--input', '-i', metavar='FILENAME', required=True,
            nargs='+', default=[],
            help='input ASCII file')
    p.add_argument('--methods', '-m', metavar='METHOD', required=True,
            nargs='+', default=[],
            help='methods separated by comma: glcm, lbp, gabor, all')
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


args = parse_args()
for infile in args.input:
    dwimage = dwi.dwimage.load(infile)[0]
    img = dwimage.image[0,:,:,0]
    basename = dwimage.basename
    if 1 in img.shape:
        img.shape = dwi.util.make2d(img.size)
    if args.verbose > 1:
        print 'Image shape: %s' % (img.shape,)

    # Write GLCM properties.
    if 'glcm' in args.methods or 'all' in args.methods:
        img_normalized = normalize(img)
        propnames = ['median', 'mean', 'stddev']
        props = [np.median(img), np.mean(img), np.std(img)]
        propnames += dwi.texture.PROPNAMES
        props += dwi.texture.get_coprops_img(img_normalized)

        outfile = args.output or 'props_%s' % basename
        if args.verbose:
            print 'Writing (%s) to %s' % (', '.join(propnames), outfile)
        with open(outfile, 'w') as f:
            f.write(' '.join(map(str, props)) + '\n')

    # Write LBP properties.
    if 'lbp' in args.methods or 'all' in args.methods:
        _, lbp_freq_data, n_patterns = dwi.texture.get_lbp_freqs(img)
        lbp_freq_data.shape = (-1, n_patterns)

        outfile = args.output or 'lbpf_%s' % basename
        if args.verbose:
            print 'Writing LBP frequencies to %s' % outfile
        with open(outfile, 'w') as f:
            for patterns in lbp_freq_data:
                f.write(' '.join(map(str, patterns)) + '\n')

    # Write Gabor properties.
    if 'gabor' in args.methods or 'all' in args.methods:
        # TODO only for ADCm, clips them
        img = img.copy()
        img.shape += (1,)
        dwi.util.clip_pmap(img, ['ADCm'])
        #img = (img - img.mean()) / img.std()
        props = dwi.texture.get_gabor_features(img[...,0]).ravel()

        outfile = args.output or 'gabor_%s' % basename
        if args.verbose:
            print 'Writing Gabor properties to %s' % outfile
        with open(outfile, 'w') as f:
            f.write(' '.join(map(str, props)) + '\n')

    #img = img[50:150, 50:150]
    #lbp_data, lbp_freq_data, patterns = dwi.texture.get_lbp_freqs(img)
    #freqs = np.rollaxis(lbp_freq_data, 2)
    #dwi.plot.show_images([[img, lbp_data], freqs[:5], freqs[5:]])
