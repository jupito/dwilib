#!/usr/bin/env python2

"""Test texture properties for ROI search."""

import argparse
import glob
import re
import numpy as np
import skimage

import dwi.dataset
import dwi.util
import dwi.patient
import dwi.dwimage
import dwi.mask
import dwi.texture

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument('--verbose', '-v', action='count',
            help='increase verbosity')
    p.add_argument('--dir', '-d', required=True,
            help='image directory')
    p.add_argument('--cases', '-c', nargs='+', type=int, default=[],
            help='case numbers to draw')
    p.add_argument('--total', '-t', action='store_true',
            help='show total AUC')
    p.add_argument('--step', '-s', type=int, default=5,
            help='window step size')
    args = p.parse_args()
    return args

def draw_props(img, title, win_step):
    pmap = dwi.texture.get_texture_pmap(img, win_step)

    import matplotlib
    import matplotlib.pyplot as plt
    import pylab as pl
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.aspect'] = 'equal'
    plt.rcParams['image.interpolation'] = 'none'
    n_cols, n_rows = len(dwi.texture.PROPNAMES)+1, 1
    fig = plt.figure(figsize=(n_cols*6, n_rows*6))
    plt.title(title)
    for i, name in enumerate(['original']+dwi.texture.PROPNAMES):
        ax = fig.add_subplot(1, n_cols, i+1)
        ax.set_title(name)
        plt.imshow(pmap[i])
    #plt.tight_layout()
    outfile = title+'.png'
    print 'Writing %s' % outfile
    plt.savefig(outfile, bbox_inches='tight')


data = dwi.dataset.dataset_read_samplelist('samples_train.txt', cases=args.cases)
dwi.dataset.dataset_read_patientinfo(data, 'patients.txt')
dwi.dataset.dataset_read_subregions(data, 'bounding_box_100_10pad')
dwi.dataset.dataset_read_pmaps(data, 'results_Mono_combinedDICOM', 'ADCm')
#dwi.dataset.dataset_read_prostate_masks(data, 'masks_prostate')
dwi.dataset.dataset_read_roi_masks(data, 'masks_rois', shape=(5,5))

for d in data:
    print d['case'], d['scan'], d['score'], d['image'].shape
    #print d['subregion'], dwi.util.subwindow_shape(d['subregion'])

