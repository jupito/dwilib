#!/usr/bin/env python2

"""Find most interesting ROI's in a DWI image."""

import argparse
import glob
import numpy as np

import dwi.autoroi
import dwi.files
import dwi.mask
import dwi.patient
import dwi.util

IN_SUBREGION_DIR = 'bounding_box_100_10pad'
IN_ROI_MASK_DIR = 'masks_rois'
IN_PROSTATE_MASK_DIR = 'masks_prostate'
IN_PATIENTS_FILE = 'patients.txt'

OUT_MASK_DIR = 'masks_auto'
OUT_IMAGE_DIR = 'find_roi_images'

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument('--verbose', '-v',
            action='count',
            help='increase verbosity')
    p.add_argument('--samplelist', default='samples_all.txt',
            help='sample list file')
    p.add_argument('--pmapdir', default='results_Mono_combinedDICOM',
            help='input parametric map directory')
    p.add_argument('--param', default='ADCm',
            help='image parameter to use')
    p.add_argument('--roidim', metavar='I', nargs=3, type=int, default=[1,5,5],
            help='dimensions of wanted ROI (3 integers; default 1 5 5)')
    p.add_argument('--algparams', metavar='I', nargs=3, type=int,
            default=[5,10,1000],
            help='algorithm parameters (ROI side min, max, number of ROIs)')
    p.add_argument('--cases', metavar='I', nargs='*', type=int, default=[],
            help='case numbers')
    p.add_argument('--scans', metavar='S', nargs='*', default=[],
            help='scan identifiers')
    p.add_argument('--outmask',
            help='output mask file')
    p.add_argument('--outfig',
            help='output figure file')
    p.add_argument('--clip', action='store_true',
            help='clip image intensity values on load')
    args = p.parse_args()
    return args

def clip_image(img, params):
    """Clip parameter-specific intensity outliers in-place."""
    for i in range(img.shape[-1]):
        if params[i].startswith('ADC'):
            img[...,i].clip(0, 0.002, out=img[...,i])
        elif params[i].startswith('K'):
            img[...,i].clip(0, 2, out=img[...,i])

def read_data(samplelist_file, patients_file, image_dir, subregion_dir,
        roi_mask_dir, prostate_mask_dir, param, cases=[], scans=[], clip=False):
    samples = dwi.util.read_sample_list(samplelist_file)
    patientsinfo = dwi.patient.read_patients_file(patients_file)
    data = []
    for sample in samples:
        case = sample['case']
        if cases and not case in cases:
            continue
        score = dwi.patient.get_patient(patientsinfo, case).score
        for scan in sample['scans']:
            if scans and not scan in scans:
                continue
            subregion = dwi.files.read_subregion(subregion_dir, case, scan)
            masks = dwi.files.read_roi_masks(roi_mask_dir, case, scan)
            cancer_mask, normal_mask = masks['ca'], masks['n']
            prostate_mask = dwi.files.read_prostate_mask(prostate_mask_dir,
                    case, scan)
            image = dwi.files.read_dicom_pmap(image_dir, case, scan, param)
            cropped_cancer_mask = cancer_mask.crop(subregion)
            cropped_normal_mask = normal_mask.crop(subregion)
            cropped_prostate_mask = prostate_mask.crop(subregion)
            cropped_image = dwi.util.crop_image(image, subregion).copy()
            if clip:
                clip_image(cropped_image, [param])
            d = dict(case=case, scan=scan, score=score,
                    subregion=subregion,
                    cancer_mask=cropped_cancer_mask,
                    normal_mask=cropped_normal_mask,
                    prostate_mask=cropped_prostate_mask,
                    original_shape=image.shape,
                    image=cropped_image)
            data.append(d)
            assert d['cancer_mask'].array.shape ==\
                    d['normal_mask'].array.shape ==\
                    d['prostate_mask'].array.shape ==\
                    d['image'].shape[0:3]
    return data

###

def draw_roi(img, pos, color):
    """Draw a rectangle ROI on a layer."""
    y, x = pos
    img[y:y+5, x:x+5] = color

def get_roi_layer(img, pos, color):
    """Get a layer with a rectangle ROI for drawing."""
    layer = np.zeros(img.shape + (4,))
    draw_roi(layer, pos, color)
    return layer

def draw(data, param, filename):
    import matplotlib
    import matplotlib.pyplot as plt
    import pylab as pl

    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['image.interpolation'] = 'nearest'
    n_cols, n_rows = 3, 1
    fig = plt.figure(figsize=(n_cols*6, n_rows*6))

    CANCER_COLOR = (1.0, 0.0, 0.0, 1.0)
    NORMAL_COLOR = (0.0, 1.0, 0.0, 1.0)
    AUTO_COLOR = (1.0, 1.0, 0.0, 1.0)

    slice_index = data['roi_corner'][0]
    pmap = data['image'][slice_index].copy()
    clip_image(pmap, [param])
    pmap = pmap[...,0]

    cancer_pos = (-1, -1)
    normal_pos = (-1, -1)
    distance = -1
    auto_pos = (data['roi_coords'][1][0], data['roi_coords'][2][0])
    if 'cancer_mask' in data:
        cancer_pos = data['cancer_mask'].where()[0][1:3]
        distance = dwi.util.distance(cancer_pos, auto_pos)
    if 'normal_mask' in data:
        normal_pos = data['normal_mask'].where()[0][1:3]

    ax1 = fig.add_subplot(1, n_cols, 1)
    ax1.set_title('Slice %i %s' % (slice_index, param))
    plt.imshow(pmap)

    ax2 = fig.add_subplot(1, n_cols, 2)
    ax2.set_title('Calculated score map')
    scoremap = data['scoremap'][slice_index]
    scoremap /= scoremap.max()
    imgray = plt.imshow(pmap, alpha=1)
    imjet = plt.imshow(scoremap, alpha=0.8, cmap='jet')

    ax3 = fig.add_subplot(1, n_cols, 3)
    ax3.set_title('ROIs: %s, %s, distance: %.2f' % (cancer_pos, auto_pos,
            distance))
    view = np.zeros(pmap.shape + (3,), dtype=float)
    view[...,0] = pmap / pmap.max()
    view[...,1] = pmap / pmap.max()
    view[...,2] = pmap / pmap.max()
    for i, a in enumerate(pmap):
        for j, v in enumerate(a):
            if v < dwi.autoroi.ADCM_MIN:
                view[i,j,:] = [0.5, 0, 0]
            elif v > dwi.autoroi.ADCM_MAX:
                view[i,j,:] = [0, 0.5, 0]
    plt.imshow(view)
    if 'cancer_mask' in data:
        plt.imshow(get_roi_layer(pmap, cancer_pos, CANCER_COLOR), alpha=0.7)
    if 'normal_mask' in data:
        plt.imshow(get_roi_layer(pmap, normal_pos, NORMAL_COLOR), alpha=0.7)
    plt.imshow(get_roi_layer(pmap, auto_pos, AUTO_COLOR), alpha=0.7)

    fig.colorbar(imgray, ax=ax1, shrink=0.65)
    fig.colorbar(imjet, ax=ax2, shrink=0.65)
    fig.colorbar(imgray, ax=ax3, shrink=0.65)

    plt.tight_layout()
    print 'Writing figure:', filename
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def write_mask(d, filename):
    """Write mask. XXX: Here only single-slice ones."""
    slice_index = d['roi_corner'][0]
    a = np.zeros((d['original_shape'][1:3]), dtype=int)
    _, y, x = d['roi_coords']
    y_offset, x_offset = d['subregion'][2], d['subregion'][4]
    y = (y[0]+y_offset, y[1]+y_offset)
    x = (x[0]+x_offset, x[1]+x_offset)
    a[y[0]:y[1], x[0]:x[1]] = 1
    mask = dwi.mask.Mask(slice_index+1, a)
    print 'Writing mask:', filename
    mask.write(filename)


args = parse_args()
print 'Reading data...'
data = read_data(args.samplelist, IN_PATIENTS_FILE, args.pmapdir,
        IN_SUBREGION_DIR, IN_ROI_MASK_DIR, IN_PROSTATE_MASK_DIR, args.param,
        args.cases, args.scans, args.clip)
sidemin, sidemax, n_rois = args.algparams
if sidemin > sidemax:
    raise Exception('Invalid ROI size limits')

for d in data:
    print '{case} {scan}: {score} {subregion}'.format(**d)
    if args.verbose:
        print d['image'].shape
        print map(lambda m: len(m.selected_slices()), [d['cancer_mask'],
                d['normal_mask'], d['prostate_mask']])
    d.update(dwi.autoroi.find_roi(d['image'], args.roidim, [args.param],
            prostate_mask=d['prostate_mask'], sidemin=sidemin, sidemax=sidemax,
            n_rois=n_rois))
    print '{case} {scan}: Optimal ROI at {roi_corner}'.format(**d)
    draw(d, args.param, args.outfig or
            OUT_IMAGE_DIR+'/{case}_{scan}.png'.format(**d))
    write_mask(d, args.outmask or
            OUT_MASK_DIR+'/{case}_{scan}_auto.mask'.format(**d))

#if args.verbose:
#    for i, p in enumerate(params):
#        z, y, x = coords
#        a = img[z[0]:z[1], y[0]:y[1], x[0]:x[1], i]
#        print p, a.min(), a.max(), np.median(a)
#        print dwi.util.fivenum(a.flatten())
#        a = img[..., i]
#        print p, a.min(), a.max(), np.median(a)
#        print dwi.util.fivenum(a.flatten())
