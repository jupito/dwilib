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
    p.add_argument('--patients',
                   help='sample list file')
    p.add_argument('--pmapdir',
                   help='input parametric map directory')
    p.add_argument('--param', default='raw',
                   help='image parameter to use')
    p.add_argument('--pc', metavar='I', nargs=2, type=float,
                   default=DEF_CFG['pc'],
                   help='minimum and maximum percentiles')
    p.add_argument('--scale', metavar='I', nargs=2, type=int,
                   default=DEF_CFG['scale'],
                   help='standard scale minimum and maximum')
    p.add_argument('--outconf',
                   help='output file for standardization configuration')
    p.add_argument('--transform', metavar='PATH', nargs=3,
                   help='transform: configuration, input, output')
    return p.parse_args()


# def histogram(a, m1=None, m2=None, bins=20):
#     """Create histogram from data between [m1, m2], with bin centers."""
#     a = np.asarray(a)
#     if m1 is not None:
#         a = a[a >= m1]
#     if m2 is not None:
#         a = a[a <= m2]
#     hist, bin_edges = np.histogram(a, bins=bins, density=True)
#     bin_centers = [np.mean(t) for t in zip(bin_edges, bin_edges[1:])]
#     return hist, bin_centers


# def plot(data, s1, s2, outfile):
#     import pylab as pl
#     for d in data:
#         img = d['img']
#         hist, bin_edges = np.histogram(img, bins=50, density=True)
#         pl.plot(bin_edges[:-1], hist)
#     pl.show()
#     pl.close()
#     for d in data:
#         img = d['img_scaled']
#         hist, bin_edges = np.histogram(img, bins=50, density=True)
#         pl.plot(bin_edges[:-1], hist)
#     pl.show()
#     pl.close()
#     #for d in data:
#     #    y = d['scores']
#     #    x = range(len(y))
#     #    pl.plot(x, y)
#     #pl.show()
#     #pl.close()
#     print('Plotting to {}...'.format(outfile))
#     images = [[d['img'][15,:,:,0], d['img_scaled'][15,:,:,0]] for d in data]
#     dwi.plot.show_images(images, vmin=s1, vmax=s2, outfile=outfile)


# def plot_histograms(histograms1, histograms2, outfile):
#     import pylab as pl
#     ncols, nrows = 2, 1
#     fig = pl.figure(figsize=(ncols*6, nrows*6))
#     # pl.yscale('log')
#     fig.add_subplot(1, 2, 1)
#     for hist, bins in histograms1:
#         pl.plot(bins, hist)
#     fig.add_subplot(1, 2, 2)
#     for hist, bins in histograms2:
#         pl.plot(bins, hist)
#     pl.tight_layout()
#     print('Plotting to {}...'.format(outfile))
#     pl.savefig(outfile, bbox_inches='tight')
#     pl.close()


def main():
    args = parse_args()

    # Generate and write configuration.
    if args.outconf:
        patients = dwi.files.read_patients_file(args.patients)
        pc1, pc2 = args.pc
        s1, s2 = args.scale
        landmarks = DEF_CFG['landmarks']
        data = []
        for case, scan in dwi.patient.cases_scans(patients):
            img = dwi.dataset.read_dicom_pmap(args.pmapdir, case, scan,
                                              args.param)
            p1, p2, scores = dwi.standardize.landmark_scores(img, pc1, pc2,
                                                             landmarks)
            mapped_scores = [int(dwi.standardize.map_onto_scale(p1, p2, s1, s2,
                                                                x)) for x in
                             scores]
            # print(case, scan, img.shape, dwi.util.fivenum(img))
            # print(case, scan, (p1, p2), scores)
            print(case, scan, img.shape, mapped_scores)
            data.append(dict(p1=p1, p2=p2, scores=scores,
                             mapped_scores=mapped_scores))
        mapped_scores = np.array([d['mapped_scores'] for d in data],
                                 dtype=np.int)
        mapped_scores = np.mean(mapped_scores, axis=0,
                                dtype=mapped_scores.dtype)
        mapped_scores = list(mapped_scores)
        print(mapped_scores)
        dwi.standardize.write_std_cfg(args.outconf, pc1, pc2, landmarks, s1,
                                      s2, mapped_scores)

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
