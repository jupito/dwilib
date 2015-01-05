#!/usr/bin/env python2

"""Find most interesting ROI's in a DWI image."""

import argparse
import glob
import numpy as np

import dwi.autoroi
import dwi.dicomfile
import dwi.mask
import dwi.patient
import dwi.util

PARAMS = ['ADCm']

IN_SUBREGION_DIR = 'bounding_box_100_10pad'
IN_MASK_DIR = 'dicom_masks'
IN_IMAGE_DIR = 'results_Mono_combinedDICOM'
IN_SAMPLELIST_FILE = 'samples_all.txt'
IN_SUBWINDOWS_FILE = 'subwindows.txt'
IN_PATIENTS_FILE = 'patients.txt'

OUT_MASK_DIR = 'masks_auto2'
OUT_IMAGE_DIR = 'find_roi2_images'

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument('--verbose', '-v',
            action='count',
            help='increase verbosity')
    p.add_argument('--roidim', metavar='I', nargs=3, type=int, default=[1,5,5],
            help='dimensions of wanted ROI (3 integers; default 1 5 5)')
    p.add_argument('--cases', metavar='I', nargs='*', type=int, default=[],
            help='case numbers')
    p.add_argument('--scan',
            help='scan id')
    p.add_argument('--outmask',
            help='output mask file')
    p.add_argument('--outpic',
            help='output graphic file')
    args = p.parse_args()
    return args

def read_subregion(case, scan):
    """Read subregion definition."""
    d = dict(c=case, s=scan)
    s = IN_SUBREGION_DIR + '/{c}_*_{s}_*.txt'.format(**d)
    paths = glob.glob(s)
    if len(paths) != 1:
        raise Exception('Subregion file confusion: %s' % s)
    subregion = dwi.util.read_subregion_file(paths[0])
    return subregion

def read_dicom_masks(case, scan):
    """Read cancer and normal masks in DICOM format.
    
    Cancer mask path ends with "_ca", normal with "_n". Unless these names
    exist, first two are used. Some cases have a third mask which has cancer,
    they are ignored for now.
    """
    d = dict(c=case, s=scan)
    s = IN_MASK_DIR + '/{c}_*_{s}_D_*'.format(**d)
    cancer_path, normal_path, other_paths = None, None, []
    paths = sorted(glob.glob(s), key=str.lower)
    if len(paths) < 2:
        raise Exception('Not all masks were not found: %s' % s)
    for path in paths:
        if not cancer_path and path.lower().endswith('ca'):
            cancer_path = path
        elif not normal_path and path.lower().endswith('n'):
            normal_path = path
        else:
            other_paths.append(path)
    if not cancer_path:
        cancer_path = other_paths.pop(0)
    if not normal_path:
        normal_path = other_paths.pop(0)
    masks = map(dwi.mask.read_dicom_mask, [cancer_path, normal_path])
    return tuple(masks)

def read_prostate_mask(case, scan):
    """Read 3D prostate mask in DICOM format.
    
    The first multi-slice mask with proper pathname is used.
    """
    d = dict(c=case, s=scan)
    IN_PROSTATE_MASK_DIR = 'masks_prostate'
    s = IN_PROSTATE_MASK_DIR + '/{c}_*_{s}_*'.format(**d)
    paths = sorted(glob.glob(s))
    for path in paths:
        mask = dwi.mask.read_mask(path)
        if len(mask.selected_slices()) > 1:
            return mask
    raise Exception('Multi-slice prostate mask not found: %s' % s)

def read_image(case, scan, param):
    d = dict(c=case, s=scan, p=param)
    s = IN_IMAGE_DIR + '/{c}_*_{s}/{c}_*_{s}_{p}'.format(**d)
    paths = glob.glob(s)
    if len(paths) != 1:
        raise Exception('Image path confusion: %s' % s)
    d = dwi.dicomfile.read_dir(paths[0])
    image = d['image']
    #image = image.squeeze(axis=3) # Remove single subvalue dimension.
    return image

def clip_outliers(img):
    """Clip parameter-specific intensity outliers in-place."""
    for i in range(img.shape[-1]):
        if PARAMS[i].startswith('ADC'):
            img[...,i].clip(0, 0.002, out=img[...,i])
        elif PARAMS[i].startswith('K'):
            img[...,i].clip(0, 2, out=img[...,i])

def read_data(cases):
    samples = dwi.util.read_sample_list(IN_SAMPLELIST_FILE)
    subwindows = dwi.util.read_subwindows(IN_SUBWINDOWS_FILE)
    patientsinfo = dwi.patient.read_patients_file(IN_PATIENTS_FILE)
    data = []
    for sample in samples:
        case = sample['case']
        if cases and not case in cases:
            continue
        score = dwi.patient.get_patient(patientsinfo, case).score
        for scan in sample['scans']:
            subwindow = subwindows[(case, scan)]
            slice_index = subwindow[0] # Make it zero-based.
            subregion = read_subregion(case, scan)
            cancer_mask, normal_mask = read_dicom_masks(case, scan)
            prostate_mask = read_prostate_mask(case, scan)
            image = read_image(case, scan, PARAMS[0])
            cropped_cancer_mask = cancer_mask.crop(subregion)
            cropped_normal_mask = normal_mask.crop(subregion)
            cropped_prostate_mask = prostate_mask.crop(subregion)
            cropped_image = dwi.util.crop_image(image, subregion).copy()
            #cropped_image = cropped_image[[slice_index],...] # TODO: one slice
            clip_outliers(cropped_image)
            d = dict(case=case, scan=scan, score=score,
                    subwindow=subwindow,
                    slice_index=slice_index,
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
    img[y:y+4:4, x] = color
    img[y:y+4:4, x+4] = color
    img[y, x:x+4:4] = color
    img[y+4, x:x+5:4] = color

def get_roi_layer(img, pos, color):
    """Get a layer with a rectangle ROI for drawing."""
    layer = np.zeros(img.shape + (4,))
    draw_roi(layer, pos, color)
    return layer

def draw(data):
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
    ax1.set_title('Slice %i %s' % (slice_index, PARAMS[0]))
    iview = data['image'][slice_index,:,:,0]
    plt.imshow(iview)

    ax2 = fig.add_subplot(1, n_cols, 2)
    ax2.set_title('Calculated score map')
    iview = data['image'][slice_index,...,0]
    pview = data['sum_scoremaps'][slice_index,...,0]
    pview /= pview.max()
    imgray = plt.imshow(iview, alpha=1)
    imjet = plt.imshow(pview, alpha=0.8, cmap='jet')

    ax3 = fig.add_subplot(1, n_cols, 3)
    ax3.set_title('ROIs: %s, %s, distance: %.2f' % (cancer_pos, auto_pos,
            distance))
    iview = data['image'][slice_index,:,:,0]
    #plt.imshow(iview)
    view = np.zeros(iview.shape + (3,), dtype=float)
    view[...,0] = iview / iview.max()
    view[...,1] = iview / iview.max()
    view[...,2] = iview / iview.max()
    for i, a in enumerate(iview):
        for j, v in enumerate(a):
            if v < dwi.autoroi.ADCM_MIN:
                view[i,j,:] = [0.5, 0, 0]
            elif v > dwi.autoroi.ADCM_MAX:
                view[i,j,:] = [0, 0.5, 0]
    plt.imshow(view)
    if 'cancer_mask' in data:
        plt.imshow(get_roi_layer(iview, cancer_pos, CANCER_COLOR), alpha=0.8)
    if 'normal_mask' in data:
        plt.imshow(get_roi_layer(iview, normal_pos, NORMAL_COLOR), alpha=0.8)
    plt.imshow(get_roi_layer(iview, auto_pos, AUTO_COLOR), alpha=0.8)

    fig.colorbar(imgray, ax=ax1, shrink=0.65)
    fig.colorbar(imjet, ax=ax2, shrink=0.65)
    fig.colorbar(imgray, ax=ax3, shrink=0.65)

    plt.tight_layout()
    filename = OUT_IMAGE_DIR + '/{case}_{scan}.png'.format(**data)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def write_mask(d):
    """Write mask. XXX: Here only single-slice ones."""
    slice_index = d['roi_corner'][0]
    a = np.zeros((d['original_shape'][1:3]), dtype=int)
    _, y, x = d['roi_coords']
    y_offset, x_offset = d['subregion'][2], d['subregion'][4]
    y = (y[0]+y_offset, y[1]+y_offset)
    x = (x[0]+x_offset, x[1]+x_offset)
    a[y[0]:y[1], x[0]:x[1]] = 1
    mask = dwi.mask.Mask(slice_index+1, a)
    filename = OUT_MASK_DIR + '/{case}_{scan}_auto2.mask'.format(**d)
    mask.write(filename)


args = parse_args()

print 'Reading data...'
data = read_data(args.cases)

for d in data:
    print
    print d['case'], d['scan'], d['score'], d['subwindow'], d['subregion']
    if args.verbose:
        print d['image'].shape
        print map(lambda m: len(m.selected_slices()), [d['cancer_mask'],
                d['normal_mask'], d['prostate_mask']])
    d.update(dwi.autoroi.find_roi(d['image'], args.roidim, PARAMS,
            prostate_mask=d['prostate_mask']))
    print 'Optimal ROI: {} at {}'.format(d['roi_coords'], d['roi_corner'])
    draw(d)
    write_mask(d)

#if args.verbose:
#    for i, p in enumerate(params):
#        z, y, x = coords
#        a = img[z[0]:z[1], y[0]:y[1], x[0]:x[1], i]
#        print p, a.min(), a.max(), np.median(a)
#        print dwi.util.fivenum(a.flatten())
#        a = img[..., i]
#        print p, a.min(), a.max(), np.median(a)
#        print dwi.util.fivenum(a.flatten())
