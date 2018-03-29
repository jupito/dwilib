#!/usr/bin/python3

"""Black out patient id stuff in histology images."""

import gc
import logging

import numpy as np
from skimage import io
import PIL.Image

import dwi.conf
import dwi.util
from dwi.files import Path

PIL.Image.MAX_IMAGE_PIXELS *= 4  # Elude DecompressionBombError.


def parse_args():
    """Parse command-line arguments."""
    p = dwi.conf.get_parser(description=__doc__)
    p.add('-c', '--color', type=int, default=255,
          help='color')
    p.add('-m', '--maxshape', type=int, nargs=2,
          help='maximum width')
    p.add('-o', '--outdir', type=Path, default=Path('out'),
          help='output directory')
    p.add('infiles', nargs='+', type=Path,
          help='input image files')
    return p.parse_args()


def resize_image(img, maxshape):
    """Halve image axes until within maximum shape."""
    # while max(img.shape) > 4096:
    while img.shape[0] > maxshape[0] or img.shape[1] > maxshape[1]:
        img = img[::2, ::2, :].copy()
    return img


def process_image(img, color):
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
            img[i, :, :] = color
    return img


def process_file(infile, outfile, color, maxshape, verbose):
    """Process a file."""
    img = io.imread(infile)
    if verbose > 0:
        print(img.shape, img.dtype, img.size)
    try:
        if maxshape is not None:
            img = resize_image(img, maxshape)
            if verbose > 1:
                print(img.shape, img.dtype, img.size)
        img = process_image(img, color)
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
        process_file(infile, outfile, args.color, args.maxshape, args.verbose)
        gc.collect()


if __name__ == '__main__':
    main()
