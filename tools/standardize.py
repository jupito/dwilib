#!/usr/bin/env python2

"""Standardize images."""

from __future__ import division
import argparse

import numpy as np
import scipy as sp

import dwi.dataset
import dwi.patient
import dwi.plot
import dwi.util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
            help='increase verbosity')
    p.add_argument('--patients',
            help='sample list file')
    p.add_argument('--subregiondir',
            help='subregion bounding box directory')
    p.add_argument('--pmapdir', default='dicoms_Mono_combinedDICOM',
            help='input parametric map directory')
    p.add_argument('--param', default='ADCm',
            help='image parameter to use')
    p.add_argument('--pc', metavar='I', nargs=2, type=float,
            default=[0, 99.8],
            help='minimum and maximum percentiles')
    p.add_argument('--scale', metavar='I', nargs=2, type=int,
            default=[1, 4095],
            help='standard scale minimum and maximum')
    p.add_argument('--cases', metavar='I', nargs='*', type=int, default=[],
            help='case numbers')
    p.add_argument('--scans', metavar='S', nargs='*', default=[],
            help='scan identifiers')
    p.add_argument('--outconf',
            help='output file for standardization configuration')
    p.add_argument('--inconf',
            help='output file for standardization configuration')
    args = p.parse_args()
    return args

def landmark_scores(img, pc1, pc2, landmarks, thresholding=True):
    from scipy.stats import scoreatpercentile
    if thresholding:
        threshold = np.mean(img)
        img = img[img > threshold]
    p1 = scoreatpercentile(img, pc1)
    p2 = scoreatpercentile(img, pc2)
    scores = [scoreatpercentile(img, i) for i in landmarks]
    return p1, p2, scores

def map_onto_scale(p1, p2, s1, s2, v):
    """Map value v from original scale [p1, p2] onto standard scale [s1, s2]."""
    assert p1 <= p2, (p1, p2)
    assert s1 <= s2, (s1, s2)
    if p1 == p2:
        assert s1 == s2, (s1, s2)
        return s1
    f = (v-p1) / (p2-p1)
    r = f * (s2-s1) + s1
    return r

def transform(img, p1, p2, scores, s1, s2, mapped_scores):
    """Transform image onto standard scale."""
    scores = [p1] + list(scores) + [p2]
    mapped_scores = [s1] + list(mapped_scores) + [s2]
    r = np.zeros_like(img, dtype=np.int)
    for pos, v in np.ndenumerate(img):
        # Select slot where to map.
        slot = sum(v > s for s in scores)
        slot = np.clip(slot, 1, len(scores)-1)
        r[pos] = map_onto_scale(scores[slot-1], scores[slot],
                mapped_scores[slot-1], mapped_scores[slot], v)
    return r

#def transform_images(data, pc1, pc2, landmarks, s1, s2, mapped_scores):
#    for d in data:
#        p1, p2, scores = landmark_scores(d['img'], pc1, pc2, landmarks)
#        d['img_scaled'] = transform(d['img'], p1, p2, scores, s1, s2,
#                mapped_scores)
#        print dwi.util.fivenum(d['img_scaled'])

def plot(data, s1, s2, outfile):
    import pylab as pl
    for d in data:
        img = d['img']
        hist, bin_edges = np.histogram(img, bins=50, density=True)
        pl.plot(bin_edges[:-1], hist)
    pl.show()
    pl.close()
    for d in data:
        img = d['img_scaled']
        hist, bin_edges = np.histogram(img, bins=50, density=True)
        pl.plot(bin_edges[:-1], hist)
    pl.show()
    pl.close()
    #for d in data:
    #    y = d['scores']
    #    x = range(len(y))
    #    pl.plot(x, y)
    #pl.show()
    #pl.close()
    print 'Plotting to {}...'.format(outfile)
    images = [[d['img'][15,:,:,0], d['img_scaled'][15,:,:,0]] for d in data]
    dwi.plot.show_images(images, vmin=s1, vmax=s2, outfile=outfile)


args = parse_args()
pc1, pc2 = args.pc
s1, s2 = args.scale
landmarks = [i*10 for i in range(1, 10)] # Deciles
patients = dwi.files.read_patients_file(args.patients)

if args.outconf:
    data = []
    for case, scan in dwi.patient.cases_scans(patients, args.cases, args.scans):
        img = dwi.dataset.read_dicom_pmap(args.pmapdir, case, scan, args.param)
        #img = img[15]
        p1, p2, scores = landmark_scores(img, pc1, pc2, landmarks)
        mapped_scores = [int(map_onto_scale(p1, p2, s1, s2, x)) for x in scores]
        #print case, scan, img.shape, dwi.util.fivenum(img)
        #print case, scan, (p1, p2), scores
        print case, scan, img.shape, mapped_scores
        data.append(dict(p1=p1, p2=p2, scores=scores,
                mapped_scores=mapped_scores))
    mapped_scores = np.array([d['mapped_scores'] for d in data], dtype=np.int)
    mapped_scores = np.mean(mapped_scores, axis=0, dtype=mapped_scores.dtype)
    mapped_scores = list(mapped_scores)
    print mapped_scores
    dwi.files.write_standardization_configuration(args.outconf, pc1, pc2,
            landmarks, s1, s2, mapped_scores)

if args.inconf:
    d = dwi.files.read_standardization_configuration(args.inconf)
    pc1 = d['pc1']
    pc2 = d['pc2']
    landmarks = d['landmarks']
    s1 = d['s1']
    s2 = d['s2']
    mapped_scores = d['mapped_scores']
    for k, v in d.items():
        print k, v
    #data = dwi.dataset.dataset_read_samplelist(args.patients, args.cases,
    #        args.scans)
    #dwi.dataset.dataset_read_pmaps(data, args.pmapdir, [args.param])
    #transform_images(data, pc1, pc2, landmarks, s1, s2, mapped_scores)
    #plot(data, s1, s2, 'std.png')

    for case, scan in dwi.patient.cases_scans(patients, args.cases, args.scans):
        img = dwi.dataset.read_dicom_pmap(args.pmapdir, case, scan, args.param)
        p1, p2, scores = landmark_scores(img, pc1, pc2, landmarks)
        print case, scan, img.shape, (p1, p2)
        img = img[15,:,:,0] # Scale and visualize slice 15 only.
        img_scaled = transform(img, p1, p2, scores, s1, s2, mapped_scores)
        s = 'std/{c}_{s}.png'.format(c=case, s=scan)
        dwi.plot.show_images([[img, img_scaled]], vmin=s1, vmax=s2, outfile=s)
