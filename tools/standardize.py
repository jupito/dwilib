#!/usr/bin/python3

"""Standardize images.

See Nyul et al. 2000: New variants of a method of MRI scale standardization.
"""

import argparse

import numpy as np

import dwi.files
import dwi.mask
import dwi.patient
import dwi.util
import dwi.standardize


DEF_CFG = dwi.standardize.default_configuration()


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
                   help='increase verbosity')
    p.add_argument('--train', nargs='+', metavar=('CFG', 'INPUT'),
                   help='train: output configuration, input, input, ...')
    p.add_argument('--transform', nargs=3, metavar=('CFG', 'INPUT', 'OUTPUT'),
                   help='transform: input configuration, input, output')
    p.add_argument('--pc', nargs=2, metavar=('PC1', 'PC2'), type=float,
                   default=DEF_CFG['pc'],
                   help='minimum and maximum percentiles')
    p.add_argument('--scale', nargs=2, metavar=('S1', 'S2'), type=int,
                   default=DEF_CFG['scale'],
                   help='standard scale minimum and maximum')
    p.add_argument('--thresholding', choices=('none', 'mean', 'median'),
                   default='median',
                   help='thresholding strategy (none, mean, median)')
    p.add_argument('--mask',
                   help='mask file for selecting foreground during transform')
    return p.parse_args()


def get_stats(pc, scale, landmarks, img, thresholding):
    """Gather info from single image."""
    p, scores = dwi.standardize.landmark_scores(img, pc, landmarks,
                                                thresholding)
    p1, p2 = p
    s1, s2 = scale
    mapped_scores = [dwi.standardize.map_onto_scale(p1, p2, s1, s2, x) for x in
                     scores]
    mapped_scores = [int(x) for x in mapped_scores]
    return dict(p=p, scores=scores, mapped_scores=mapped_scores)


def train(pc, scale, landmarks, inpaths, cfgpath, thresholding, verbose):
    """Training phase."""
    data = []
    for inpath in inpaths:
        img, _ = dwi.files.read_pmap(inpath)
        if img.shape[-1] != 1:
            raise Exception('Incorrect shape: {}'.format(inpath))
        d = get_stats(pc, scale, landmarks, img, thresholding)
        if verbose:
            # print(img.shape, dwi.util.fivenum(img), inpath)
            # print(d['p'], d['scores'], inpath)
            print(img.shape, d['mapped_scores'], inpath)
        data.append(d)
    mapped_scores = np.array([x['mapped_scores'] for x in data], dtype=np.int)
    mapped_scores = np.mean(mapped_scores, axis=0, dtype=mapped_scores.dtype)
    mapped_scores = list(mapped_scores)
    if verbose:
        print(mapped_scores)
    dwi.standardize.write_std_cfg(cfgpath, pc, landmarks, scale, mapped_scores,
                                  thresholding)


def transform(cfgpath, inpath, outpath, maskpath, verbose):
    """Transform phase."""
    img, attrs = dwi.files.read_pmap(inpath)
    if maskpath is not None:
        mask = dwi.mask.read_mask(maskpath)
        if verbose:
            print('Using mask {}'.format(mask))

    if img.shape[-1] != 1:
        raise Exception('Incorrect shape: {}'.format(inpath))
    if verbose:
        print('in {s}, {t}, {f}, {p}'.format(s=img.shape, t=img.dtype,
                                             f=dwi.util.fivenums(img),
                                             p=inpath))
    img = dwi.standardize.standardize(img, cfgpath, mask=mask.array)
    attrs['standardization'] = 'L4'
    if verbose:
        print('out {s}, {t}, {f}, {p}'.format(s=img.shape, t=img.dtype,
                                              f=dwi.util.fivenum(img),
                                              p=outpath))
    dwi.files.write_pmap(outpath, img, attrs)


def main():
    args = parse_args()

    if args.train:
        cfgpath, inpaths = args.train[0], args.train[1:]
        train(args.pc, args.scale, DEF_CFG['landmarks'], inpaths, cfgpath,
              args.thresholding, args.verbose)

    if args.transform:
        cfgpath, inpath, outpath = args.transform
        transform(cfgpath, inpath, outpath, args.mask, args.verbose)


if __name__ == '__main__':
    main()
