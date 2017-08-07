#!/usr/bin/python3

"""Get pmaps as (z,y,x,label,ADCm,ADCk,K)
where label = {-1: background, 0: prostate, 1...n: lesion #1...n}.
"""

import logging

import numpy as np

import dwi.files
import dwi.paths
import dwi.patient
from dwi.types import ImageMode

LOGLEVEL = logging.INFO
# LOGLEVEL = logging.DEBUG


def main():
    modes = ['DWI-Mono-ADCm', 'DWI-Kurt-ADCk', 'DWI-Kurt-K']
    modes = [ImageMode(x) for x in modes]
    samplelist = 'bothscansonly'
    logging.basicConfig(level=LOGLEVEL)
    process_all(modes, samplelist)


def process_all(modes, samplelist):
    print(modes)
    print(samplelist)
    path = dwi.paths.samplelist_path(modes[0], samplelist)
    patients = dwi.files.read_patients_file(path)
    for p in patients:
        for s in p.scans:
            process_case(modes, p.num, s, p.lesions)


def process_case(modes, case, scan, lesions):
    print(modes, case, scan, lesions)

    # read prostate mask.
    path = dwi.paths.mask_path(modes[0], 'prostate', case, scan)
    mask = dwi.files.read_mask(path)
    logging.info('Read prostate mask: %s, %s, %s', path, mask.shape,
                 mask.dtype)
    shape = mask.shape + (1 + len(modes),)  # Create output array.
    output = np.zeros(shape, dtype=np.float32)  # Mark non-prostate with 0.
    output[mask, 0] = -1  # Mark non-lesion prostate with -1.

    # read lesion masks.
    for i, lesion in enumerate(lesions, 1):
        path = dwi.paths.mask_path(modes[0], 'lesion', case, scan, lesion=i)
        mask = dwi.files.read_mask(path)
        logging.info('Read lesion mask: %s, %s, %s, %s', lesion, path,
                     mask.shape, mask.dtype)
        output[mask, 0] = i  # Mark lesion #n with n.

    # read pmaps.
    for i, mode in enumerate(modes, 1):
        path = dwi.paths.pmap_path(mode, case, scan)
        # pmap, attrs = dwi.files.read_pmap(path, dtype=np.float32)
        pmap, attrs = dwi.files.read_pmap(path)
        logging.info('Read pmap %s, %s, %s', path, pmap.shape, pmap.dtype)
        logging.debug(attrs)
        pmap = pmap[:, :, :, 0]
        output[..., i] = pmap

    # write array
    path = 'labeled_pmaps/{m[0]}/{c}-{s}.h5'.format(m=modes[0], c=case, s=scan)
    print('Writing: {}, {}, {}'.format(path, output.shape, output.dtype))
    attrs = {}
    attrs['parameters'] = ['masks'] + [str(x) for x in modes]
    dwi.files.ensure_dir(path)
    dwi.files.write_pmap(path, output, attrs)


if __name__ == '__main__':
    main()
