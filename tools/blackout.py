#!/usr/bin/python3

"""Black out patient id stuff in histology images."""

import gc
import logging

import numpy as np
from skimage import io

import dwi.conf
import dwi.util
from dwi.files import Path


def parse_args():
    """Parse command-line arguments."""
    p = dwi.conf.get_parser(description=__doc__)
    p.add('infiles', nargs='+', type=Path,
          help='input image files')
    p.add('-o', '--outdir', type=Path, default=Path('out'),
          help='output directory')
    return p.parse_args()


def process_image(img):
    print(img.size, img.shape, img.dtype)
    while max(img.shape) > 4096:
        img = img[::2, ::2, :].copy()
    print(img.shape, img.dtype)
    # print(dwi.util.fivenum(img))
    # print(dwi.util.fivenum(img[..., 0]))
    # print(dwi.util.fivenum(img[..., 1]))
    # print(dwi.util.fivenum(img[..., 2]))
    # img[0:300, :, 0] = 0
    # img[1000:1300, :, 1] = 0
    # img[2000:2300, :, 2] = 0
    head = int(img.shape[1] / 4)
    for i in range(len(img)):
        # if np.median(img[i]) < 200:
        #     img[i, img[i] < 200] = 0
        if all(np.mean(img[i, :head, j]) < 200 for j in range(3)):
            img[i, :, :] = 255
    return img


def process_file(infile, outfile):
    """Process a file."""
    img = io.imread(infile)
    try:
        img = process_image(img)
    except Exception as e:
        logging.exception(infile)
        return
    io.imshow(img)
    print('Saving to {}'.format(outfile))
    io.imsave(outfile, img)


def main():
    """Main."""
    args = parse_args()
    for infile in args.infiles:
        outfile = args.outdir / infile.name
        process_file(infile, outfile)
        gc.collect()


if __name__ == '__main__':
    main()
