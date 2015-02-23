"""Directory and file structures."""

import glob

import dwi.dicomfile

def read_subregion(directory, case, scan):
    """Read subregion definition."""
    d = dict(d=directory, c=case, s=scan)
    s = '{d}/{c}_*_{s}_*.txt'.format(**d)
    paths = glob.glob(s)
    if len(paths) != 1:
        raise Exception('Subregion file confusion: %s' % s)
    subregion = dwi.util.read_subregion_file(paths[0])
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
    d = dict(d=directory, c=case, s=scan, p=param)
    s = '{d}/{c}_*_{s}/{c}_*_{s}_{p}'.format(**d)
    paths = glob.glob(s)
    if len(paths) != 1:
        raise Exception('Image path confusion: %s' % s)
    d = dwi.dicomfile.read_dir(paths[0])
    image = d['image']
    #image = image.squeeze(axis=3) # Remove single subvalue dimension.
    return image
