#!/usr/bin/python3

"""Read CPR files."""

import logging
import re
import sys

import numpy as np

import dwi.files
from dwi.types import Path


def read_cpr(path):
    """Read and parse a CPR file. Return masks, which are (id, RLE data).

    Note: Python 3.6.4 docs say its XML module is not secure, se we use regexp.
    """
    # Example: <Mask id="123">[12 34 56]</Mask>
    mask_pattern = r'<Mask\s.*?id="(.*?)".*?>.*?\[(.*?)\].*?</Mask>'
    text = path.read_text()
    matches = re.finditer(mask_pattern, text, flags=re.DOTALL)

    def parse_match(m):
        number, mask = m.groups()
        number = int(number)
        mask = [int(x) for x in mask.split()]
        return number, mask

    masks = [parse_match(x) for x in matches]
    return masks


def parse_mask(mask):
    """Decode a run length encoded mask into an array."""
    lst = []
    for length in mask:
        n = int(length > 0)  # Run is 1 if positive, 0 if negative.
        lst.extend([n] * abs(length))
    return np.array(lst, dtype=np.bool)


def main(path, shape, outdir, fmt='h5'):
    logging.info(path)
    masks = read_cpr(path)
    logging.info(len(masks))
    # logging.info(masks)
    # logging.info(masks[-1][1])
    # logging.info(len(parse_mask(masks[-1][1])))
    for i, m in enumerate(masks, 1):
        number, mask = m
        logging.info('Mask: %i, $i', number, len(mask))
        mask = parse_mask(mask)
        try:
            mask.shape = shape + (1,)
        except ValueError as e:
            logging.error('%s: %s', e, path)
            continue
        assert mask.ndim == 4, mask.shape
        outname = '{p}.{i:02d}-{n}.{f}'.format(p=path.name, i=i, n=number,
                                               f=fmt)
        outpath = outdir / outname
        attrs = {}
        print('Writing mask shape {}: {}'.format(mask.shape, outpath))
        dwi.files.ensure_dir(outpath)
        dwi.files.write_pmap(outpath, mask, attrs)


if __name__ == '__main__':
    # Arguments: input file; output directory; shape (eg 20,224,224).
    logging.basicConfig(level=logging.INFO)
    path, outdir, shape = sys.argv[1:]
    logging.info(path, outdir, shape, sys.argv[1:])
    path = Path(path)
    shape = tuple(int(x) for x in shape.split(','))
    outdir = Path(outdir)
    main(path, shape, outdir)
