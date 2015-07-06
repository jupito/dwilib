"""Dataset, directory and file structures."""

import glob

import numpy as np

import dwi.dicomfile
import dwi.files
import dwi.mask
import dwi.patient
import dwi.util

def read_subregion(directory, case, scan):
    """Read subregion definition."""
    d = dict(d=directory, c=case, s=scan)
    path = dwi.util.sglob('{d}/{c}_*{s}_*.txt'.format(**d))
    subregion = dwi.files.read_subregion_file(path)
    return subregion

def read_roi_masks(directory, case, scan, keys=['ca', 'n', 'ca2']):
    """Read cancer and normal ROI masks.

    Mask path ends with '_ca' for cancer ROI, '_n' for normal ROI, or '_ca2' for
    an optional second cancer ROI.

    A dictionary is returned, with the ending as key and mask as value.
    """
    d = dict(d=directory, c=case, s=scan)
    s = '{d}/{c}_*_{s}_[Dd]_*'.format(**d)
    masks = {}
    paths = glob.iglob(s)
    for path in paths:
        for key in keys:
            if path.lower().endswith('_' + key):
                masks[key] = dwi.mask.read_mask(path)
    if not ('ca' in masks and 'n' in masks):
        raise Exception('Mask for cancer or normal ROI was not found: %s' % s)
    return masks

def read_prostate_mask(directory, case, scan):
    """Read 3D prostate mask in DICOM format.

    The first multi-slice mask with proper pathname is used.
    """
    d = dict(d=directory, c=case, s=scan)
    s = '{d}/{c}_*_{s}_*'.format(**d)
    paths = sorted(glob.glob(s))
    for path in paths:
        mask = dwi.mask.read_mask(path)
        if len(mask.selected_slices()) > 1:
            return mask
    raise Exception('Multi-slice prostate mask not found: %s' % s)

def read_lesion_masks(directory, case, scan):
    """Read 3D prostate mask in DICOM format.

    The first multi-slice mask with proper pathname is used.
    """
    d = dict(d=directory, c=case, s=scan)
    s = '{d}/PCa_masks_*/{c}_*{s}_*'.format(**d)
    paths = sorted(glob.glob(s))
    if len(paths) == 0:
        raise Exception('Lesion masks not found: %s' % s)
    masks = [dwi.mask.read_mask(p) for p in paths]
    return masks

def read_dicom_pmap(directory, case, scan, param):
    """Read a single-parameter pmap in DICOM format."""
    d = dict(d=directory, c=case, s=scan, p=param)
    if param == 'raw':
        # There's no actual parameter, only single 'raw' value (used for T2).
        s = '{d}/{c}_*_{s}*'
    else:
        s = '{d}/{c}_*_{s}/{c}_*_{s}*_{p}'
    path = dwi.util.sglob(s.format(**d))
    d = dwi.dicomfile.read_dir(path)
    image = d['image']
    #image = image.squeeze(axis=3) # Remove single subvalue dimension.
    return image

#def read_dicom_pmaps(samplelist_file, patients_file, image_dir, subregion_dir,
#        prostate_mask_dir, roi_mask_dir, param, cases=[], scans=[], clip=False):
#    """Read pmaps in DICOM format and other data."""
#    # XXX Obsolete
#    samples = dwi.files.read_sample_list(samplelist_file)
#    patientsinfo = dwi.files.read_patients_file(patients_file)
#    data = []
#    for sample in samples:
#        case = sample['case']
#        if cases and not case in cases:
#            continue
#        score = dwi.patient.get_patient(patientsinfo, case).score
#        for scan in sample['scans']:
#            if scans and not scan in scans:
#                continue
#            image = read_dicom_pmap(image_dir, case, scan, param)
#            original_shape = image.shape
#            prostate_mask = read_prostate_mask(prostate_mask_dir, case, scan)
#            roi_masks = read_roi_masks(roi_mask_dir, case, scan)
#            cancer_mask, normal_mask = roi_masks['ca'], roi_masks['n']
#            subregion = None
#            if subregion_dir:
#                subregion = read_subregion(subregion_dir, case, scan)
#                image = dwi.util.crop_image(image, subregion).copy()
#                prostate_mask = prostate_mask.crop(subregion)
#                cancer_mask = cancer_mask.crop(subregion)
#                normal_mask = normal_mask.crop(subregion)
#            if clip:
#                dwi.util.clip_pmap(image, [param])
#            d = dict(case=case, scan=scan, score=score,
#                    image=image,
#                    original_shape=original_shape,
#                    subregion=subregion,
#                    prostate_mask=prostate_mask,
#                    cancer_mask=cancer_mask,
#                    normal_mask=normal_mask,
#                    )
#            data.append(d)
#            assert (d['image'].shape[0:3] ==
#                    d['prostate_mask'].array.shape ==
#                    d['cancer_mask'].array.shape ==
#                    d['normal_mask'].array.shape)
#    return data

def dataset_read_samplelist(samplelist_file, cases=(), scans=()):
    """Create a new dataset from a sample list file, optionally including only
    mentioned cases and scans."""
    samples = dwi.files.read_sample_list(samplelist_file)
    data = []
    for sample in samples:
        case = sample['case']
        if cases and not case in cases:
            continue
        for scan in sample['scans']:
            if scans and not scan in scans:
                continue
            data.append(dict(case=case, scan=scan))
    return data

def dataset_read_samples(cases_scans):
    """Create a new dataset from a list of (sample, scan) tuples."""
    data = [dict(case=c, scan=s) for c, s in cases_scans]
    return data

def dataset_read_patientinfo(data, patients_file):
    """Add patient info to dataset."""
    patientsinfo = dwi.files.read_patients_file(patients_file)
    for d in data:
        d['score'] = dwi.patient.get_patient(patientsinfo, d['case']).score

def dataset_read_subregions(data, subregion_dir):
    """Add subregions to dataset."""
    for d in data:
        d['subregion'] = read_subregion(subregion_dir, d['case'], d['scan'])

def dataset_read_pmaps(data, image_dir, params):
    """Add pmaps to dataset (after optional subregions)."""
    for d in data:
        images = []
        for param in params:
            image = read_dicom_pmap(image_dir, d['case'], d['scan'], param)
            if 'subregion' in d:
                d['original_shape'] = image.shape
                image = dwi.util.crop_image(image, d['subregion']).copy()
            images.append(image)
        d['image'] = np.concatenate(images, axis=-1)

def dataset_read_prostate_masks(data, prostate_mask_dir):
    """Add prostate masks to dataset (after pmaps)."""
    for d in data:
        mask = read_prostate_mask(prostate_mask_dir, d['case'], d['scan'])
        if 'subregion' in d:
            mask = mask.crop(d['subregion'])
        assert d['image'].shape[0:3] == mask.array.shape
        roi = mask.selected(d['image'])
        d.update(prostate_mask=mask, prostate_roi=roi)

def dataset_read_lesion_masks(data, mask_dir):
    """Add lesion masks to dataset (after pmaps)."""
    for d in data:
        masks = read_lesion_masks(mask_dir, d['case'], d['scan'])
        if 'subregion' in d:
            masks = [m.crop(d['subregion']) for m in masks]
        for m in masks:
            assert d['image'].shape[0:3] == m.array.shape
        d.update(lesion_masks=masks)

def dataset_read_roi_masks(data, roi_mask_dir, shape=None):
    """Add ROI masks to dataset (after pmaps)."""
    for d in data:
        masks = read_roi_masks(roi_mask_dir, d['case'], d['scan'])
        cmask, nmask = masks['ca'], masks['n']
        if 'subregion' in d:
            cmask = cmask.crop(d['subregion'])
            nmask = nmask.crop(d['subregion'])
        assert d['image'].shape[0:3] == cmask.array.shape == nmask.array.shape
        croi = cmask.selected(d['image'])
        nroi = nmask.selected(d['image'])
        if shape == '2d':
            croi.shape = dwi.util.make2d(croi.size)
            nroi.shape = dwi.util.make2d(nroi.size)
        elif shape:
            croi.shape = shape
            nroi.shape = shape
        d.update(cancer_mask=cmask, normal_mask=nmask, cancer_roi=croi,
                normal_roi=nroi)
