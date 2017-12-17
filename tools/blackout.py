#!/usr/bin/python3

"""Black out patient id stuff in histology images."""

# import numpy as np
from skimage import io

import dwi.conf
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = dwi.conf.get_parser(description=__doc__)
    p.add('infiles', nargs='+',
          help='input image files')
    return p.parse_args()


def process_image(img):
    print(img.shape, img.dtype)
    # print(dwi.util.fivenum(img))
    # print(dwi.util.fivenum(img[..., 0]))
    # print(dwi.util.fivenum(img[..., 1]))
    # print(dwi.util.fivenum(img[..., 2]))
    # img[0:300, :, 0] = 0
    # img[1000:1300, :, 1] = 0
    # img[2000:2300, :, 2] = 0
    for i in range(len(img)):
        img[i, img[i] < 200] = 0


def process_file(infile, outfile):
    """Process a file."""
    img = io.imread(infile)
    process_image(img)
    io.imshow(img)
    print('Saving to {}'.format(outfile))
    io.imsave(outfile, img)


def main():
    """Main."""
    args = parse_args()
    for infile in args.infiles:
        outfile = infile + '.out.jpg'
        process_file(infile, outfile)


if __name__ == '__main__':
    main()
