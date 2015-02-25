"""Directory and file structures."""

import glob

import dwi.dicomfile
import dwi.mask
import dwi.util

def read_subregion(directory, case, scan):
    """Read subregion definition."""
    d = dict(d=directory, c=case, s=scan)
    path = dwi.util.sglob('{d}/{c}_*_{s}_*.txt'.format(**d))
    subregion = dwi.util.read_subregion_file(path)
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

def read_dicom_pmap(directory, case, scan, param):
    """Read a single-parameter pmap in DICOM format."""
    d = dict(d=directory, c=case, s=scan, p=param)
    path = dwi.util.sglob('{d}/{c}_*_{s}/{c}_*_{s}_{p}'.format(**d))
    d = dwi.dicomfile.read_dir(path)
    image = d['image']
    #image = image.squeeze(axis=3) # Remove single subvalue dimension.
    return image

def read_dicom_pmaps(samplelist_file, patients_file, image_dir, subregion_dir,
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
            original_shape = image.shape
            cropped_cancer_mask = cancer_mask.crop(subregion)
            cropped_normal_mask = normal_mask.crop(subregion)
            cropped_prostate_mask = prostate_mask.crop(subregion)
            cropped_image = dwi.util.crop_image(image, subregion).copy()
            if clip:
                dwi.util.clip_pmap(cropped_image, [param])
            d = dict(case=case, scan=scan, score=score,
                    subregion=subregion,
                    cancer_mask=cropped_cancer_mask,
                    normal_mask=cropped_normal_mask,
                    prostate_mask=cropped_prostate_mask,
                    original_shape=original_shape,
                    image=cropped_image)
            data.append(d)
            assert d['cancer_mask'].array.shape ==\
                    d['normal_mask'].array.shape ==\
                    d['prostate_mask'].array.shape ==\
                    d['image'].shape[0:3]
    return data
