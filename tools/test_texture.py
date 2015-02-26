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
import dwi.plot
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

def get_roi_slices(data):
    for d in data:
        img = d['image']
        d['cancer_slice'] = d['cancer_mask'].get_masked_slice(img)
        d['normal_slice'] = d['normal_mask'].get_masked_slice(img)

def get_lbpf(img):
    lbp, lbp_freq, n_patterns = dwi.texture.get_lbp_freqs(img)
    return lbp_freq
    #return lbp_freq[...,1:] # Drop non-uniform patterns

def avg_lbpf_map(histogram_map):
    """Return the average of multidimensional LPB frequency histogram map."""
    histogram_map = np.asanyarray(histogram_map)
    n_patterns = histogram_map.shape[-1]
    histograms = histogram_map.reshape(-1, n_patterns)
    return np.mean(histograms, axis=0)

def get_lbpfs(data):
    """Get LBP frequency histograms."""
    for d in data:
        d['cancer_slice_lbpf'] = get_lbpf(d['cancer_slice'][...,0])
        d['cancer_slice_lbpf_avg'] = avg_lbpf_map(d['cancer_slice_lbpf'])
        d['cancer_lbpf_avg'] = avg_lbpf_map(get_lbpf(d['cancer_roi']))
        d['normal_lbpf_avg'] = avg_lbpf_map(get_lbpf(d['normal_roi']))

def get_distances(data, model):
    """Get LBP frequence histogram distances."""
    for d in data:
        sample = d['cancer_slice_lbpf']
        #model = d['cancer_lbpf_avg']
        print sample.shape
        dist = np.zeros_like(sample[...,0])
        for y, x in np.ndindex(dist.shape):
            dist[y,x] = dwi.texture.lbpf_dist(sample[y,x], model,
                    method='log-likelihood')
        d['distance_map'] = dist

def draw(data, param, filename):
    import matplotlib
    import matplotlib.pyplot as plt
    import pylab as pl

    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'nearest'
    n_cols, n_rows = 2, 1
    fig = plt.figure(figsize=(n_cols*6, n_rows*6))

    CANCER_COLOR = (1.0, 0.0, 0.0, 1.0)
    NORMAL_COLOR = (0.0, 1.0, 0.0, 1.0)
    AUTO_COLOR = (1.0, 1.0, 0.0, 1.0)

    pmap = data['cancer_slice'][...,0].copy()
    dwi.util.clip_pmap(pmap, ['ADCm'])

    cancer_pos = data['cancer_mask'].where()[0][1:3]
    normal_pos = data['normal_mask'].where()[0][1:3]

    ax1 = fig.add_subplot(1, n_cols, 1)
    ax1.set_title('ADCm')
    plt.imshow(pmap)

    ax3 = fig.add_subplot(1, n_cols, 2)
    ax3.set_title('LBP freq dist')
    view = np.zeros(pmap.shape + (3,), dtype=float)
    view[...,0] = pmap / pmap.max()
    view[...,1] = pmap / pmap.max()
    view[...,2] = pmap / pmap.max()
    #for i, a in enumerate(pmap):
    #    for j, v in enumerate(a):
    #        if v < dwi.autoroi.ADCM_MIN:
    #            view[i,j,:] = [0.5, 0, 0]
    #        elif v > dwi.autoroi.ADCM_MAX:
    #            view[i,j,:] = [0, 0.5, 0]
    plt.imshow(view)
    plt.imshow(get_roi_layer(pmap, cancer_pos, CANCER_COLOR), alpha=0.7)
    plt.imshow(get_roi_layer(pmap, normal_pos, NORMAL_COLOR), alpha=0.7)

    plt.tight_layout()
    print 'Writing figure:', filename
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


args = parse_args()

print 'Reading dataset...'
data = dwi.dataset.dataset_read_samplelist('samples_train.txt', cases=args.cases, scans=['1a', '2a'])
dwi.dataset.dataset_read_patientinfo(data, 'patients.txt')
dwi.dataset.dataset_read_subregions(data, 'bounding_box_100_10pad')
dwi.dataset.dataset_read_pmaps(data, 'results_Mono_combinedDICOM', 'ADCm')
#dwi.dataset.dataset_read_prostate_masks(data, 'masks_prostate')
dwi.dataset.dataset_read_roi_masks(data, 'masks_rois', shape=(5,5))
get_roi_slices(data)
for d in data:
    print d['case'], d['scan'], d['score'], d['image'].shape
    #print d['subregion'], dwi.util.subwindow_shape(d['subregion'])

print 'Calculating texture properties...'
get_lbpfs(data)
model = avg_lbpf_map([d['cancer_lbpf_avg'] for d in data])
get_distances(data, model)

print 'Plotting...'
l = []
for d in data:
    imgs = [d['cancer_slice'][...,0], d['distance_map']]
    l.append(imgs)
imgs = [[d['cancer_slice'][...,0], d['distance_map']] for d in data]
ylabels=[d['case'] for d in data]
xlabels=['adc', 'dist']
dwi.plot.show_images(l, ylabels, xlabels, outfile='dist.png')
for d in data:
    draw(d, 'dist_fig/dist_%s.png' % (d['case'], d['scan']))
