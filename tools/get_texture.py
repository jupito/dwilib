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
            help='input ASCII file')
    p.add_argument('--methods', '-m', metavar='METHOD', nargs='+',
            default=['all'],
            help='methods separated by comma: basic, glcm, lbp, gabor, all')
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
dwimage = dwi.dwimage.load(args.input)[0]
img = dwimage.image[0,:,:,0]
if 1 in img.shape:
    img.shape = dwi.util.make2d(img.size)
if args.verbose > 1:
    print 'Image shape: %s' % (img.shape,)

propnames = []
props = []

# Write basic properties.
if 'basic' in args.methods or 'all' in args.methods:
    propnames += ['median', 'mean', 'stddev']
    props += [np.median(img), np.mean(img), np.std(img)]

# Write GLCM properties.
if 'glcm' in args.methods or 'all' in args.methods:
    img_normalized = normalize(img)
    propnames += dwi.texture.PROPNAMES
    props += dwi.texture.get_coprops_img(img_normalized)

# Write LBP properties.
if 'lbp' in args.methods or 'all' in args.methods:
    _, lbp_freq_data, n_patterns = dwi.texture.get_lbp_freqs(img)
    lbp_freq_data = lbp_freq_data.reshape((-1, n_patterns))
    propnames += ['lbpf{:d}'.format(i) for i in range(n_patterns)]
    props += list(lbp_freq_data.mean(axis=0))

# Write Gabor properties.
if 'gabor' in args.methods or 'all' in args.methods:
    # TODO only for ADCm, clips them
    img = img.copy()
    img.shape += (1,)
    dwi.util.clip_pmap(img, ['ADCm'])
    #img = (img - img.mean()) / img.std()
    gabor = dwi.texture.get_gabor_features(img[...,0]).ravel()
    propnames += ['gabor{:d}'.format(i) for i in range(len(gabor))]
    props += list(gabor)

if args.verbose:
    print 'Writing %s features to %s' % (len(props), args.output)
dwi.asciifile.write_ascii_file(args.output, [props], propnames)

#img = img[50:150, 50:150]
#lbp_data, lbp_freq_data, patterns = dwi.texture.get_lbp_freqs(img)
#freqs = np.rollaxis(lbp_freq_data, 2)
#dwi.plot.show_images([[img, lbp_data], freqs[:5], freqs[5:]])
