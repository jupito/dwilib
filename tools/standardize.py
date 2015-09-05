#!/usr/bin/env python2

"""Standardize images.

See Nyul et al. 2000: New variants of a method of MRI scale standardization.
"""

from __future__ import absolute_import, division, print_function
import argparse

import numpy as np

import dwi.dataset
import dwi.files
import dwi.patient
import dwi.util
import dwi.standardize


DEF_CFG = dwi.standardize.default_configuration()


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
                   help='increase verbosity')
    p.add_argument('--pc', metavar='I', nargs=2, type=float,
                   default=DEF_CFG['pc'],
                   help='minimum and maximum percentiles')
    p.add_argument('--scale', metavar='I', nargs=2, type=int,
                   default=DEF_CFG['scale'],
                   help='standard scale minimum and maximum')
    p.add_argument('--train', metavar='PATH', nargs='+',
                   help='train: output configuration, input, input, ...')
    p.add_argument('--transform', metavar='PATH', nargs=3,
                   help='transform: input configuration, input, output')
    return p.parse_args()


def main():
    args = parse_args()

    # Generate and write configuration.
    if args.train:
        cfgpath, inpaths = args.train[0], args.train[1:]
        pc1, pc2 = args.pc
        s1, s2 = args.scale
        landmarks = DEF_CFG['landmarks']
        data = []
        for inpath in inpaths:
            img, attrs = dwi.files.read_pmap(inpath)
            p1, p2, scores = dwi.standardize.landmark_scores(img, pc1, pc2,
                                                             landmarks)
            mapped_scores = [int(dwi.standardize.map_onto_scale(p1, p2, s1, s2,
                                                                x)) for x in
                             scores]
            # print(img.shape, dwi.util.fivenum(img), inpath)
            # print((p1, p2), scores, inpath)
            print(img.shape, mapped_scores, inpath)
            data.append(dict(p1=p1, p2=p2, scores=scores,
                             mapped_scores=mapped_scores))
        mapped_scores = np.array([d['mapped_scores'] for d in data],
                                 dtype=np.int)
        mapped_scores = np.mean(mapped_scores, axis=0,
                                dtype=mapped_scores.dtype)
        mapped_scores = list(mapped_scores)
        print(mapped_scores)
        dwi.standardize.write_std_cfg(cfgpath, pc1, pc2, landmarks, s1, s2,
                                      mapped_scores)

    # Read configuration, read and standardize image, write it.
    if args.transform:
        cfgpath, inpath, outpath = args.transform

        img, attrs = dwi.files.read_pmap(inpath)
        if img.shape[-1] != 1:
            raise Exception('Incorrect shape: {}'.format(inpath))
        if args.verbose:
            print('in {s}, {t}, {f}, {p}'.format(s=img.shape, t=img.dtype,
                                                 f=dwi.util.fivenum(img),
                                                 p=inpath))

        img = dwi.standardize.standardize(img, cfgpath)
        attrs['standardization'] = 'L4'

        if args.verbose:
            print('out {s}, {t}, {f}, {p}'.format(s=img.shape, t=img.dtype,
                                                  f=dwi.util.fivenum(img),
                                                  p=outpath))
        dwi.files.write_pmap(outpath, img, attrs)


if __name__ == '__main__':
    main()
