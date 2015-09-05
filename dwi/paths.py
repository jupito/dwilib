"""Common path names."""

from __future__ import absolute_import, division, print_function
import os.path

import dwi.util


def samplelist_path(mode, samplelist):
    return 'patients_{m[0]}_{l}.txt'.format(m=mode, l=samplelist)


def pmap_path(mode, case=None, scan=None, fmt='dicom'):
    if fmt == 'hdf5' or 'std' in mode:
        path = 'images/{m}'
        if case is not None and scan is not None:
            path += '/{c}-{s}.h5'
        return path.format(m=mode, c=case, s=scan)
    elif fmt == 'dicom':
        if len(mode) == 1:
            path = 'dicoms/{m[0]}_*'
        else:
            path = 'dicoms/{m[1]}_*'
        if case is not None and scan is not None:
            if len(mode) == 1:
                path += '/{c}_*_{s}*'
            else:
                path += '/{c}_*_{s}/{c}_*_{s}*_{m[2]}'
        path = path.format(m=mode, c=case, s=scan)
        return dwi.util.sglob(path, typ='dir')
    else:
        raise ValueError('Unknown format: {}'.format(fmt))


def subregion_path(mode, case=None, scan=None):
    path = 'subregions/{m[0]}'
    if case is not None and scan is not None:
        path += '/{c}_{s}_subregion10.txt'
    return path.format(m=mode, c=case, s=scan)


def mask_path(mode, masktype, case, scan, lesion=None, algparams=()):
    """Return path and deps of masks of different types."""
    d = dict(m=mode, mt=masktype, c=case, s=scan, l=lesion,
             ap_='_'.join(algparams))
    do_glob = True
    if masktype == 'prostate':
        path = 'masks_{mt}/{m[0]}/{c}_*_{s}*'
    elif masktype == 'lesion':
        path = 'masks_{mt}/{m[0]}/PCa_masks_{m[0]}_{l}*/{c}_*{s}_*'
    elif masktype in ('CA', 'N'):
        path = 'masks_roi/{m[0]}/{c}_*_{s}_D_{mt}'
    elif masktype == 'auto':
        path = 'masks_{mt}/{m}/{ap_}/{c}_{s}_auto.mask'
        do_glob = False  # Don't require existence, can be generated.
    else:
        raise Exception('Unknown mask type: {mt}'.format(**d))
    path = path.format(**d)
    if do_glob:
        path = dwi.util.sglob(path)
    return path


def roi_path(mode, masktype, case=None, scan=None, lesion=None, algparams=()):
    """Return whole ROI path or part of it."""
    d = dict(m=mode, mt=masktype, c=case, s=scan, l=lesion,
             ap_='_'.join(algparams))
    if masktype == 'lesion':
        return 'rois_lesion/{m}/{c}-{s}-{l}.h5'.format(**d)
    components = ['rois_{mt}', '{m}']
    if algparams:
        components.append('{ap_}')
    if case is not None and scan is not None:
        components.append('{c}_x_x_{s}_{m}_{mt}.txt')
    components = [x.format(**d) for x in components]
    return os.path.join(*components)


def texture_path(mode, case, scan, lesion, masktype, slices, portion, method,
                 winsize, algparams=(), voxel='mean'):
    """Return path to texture file."""
    path = 'texture_{mt}/{m}_{slices}_{portion}_{vx}'
    if voxel == 'mean':
        ext = 'txt'
    else:
        ext = 'h5'
    if masktype in ('prostate', 'lesion', 'CA', 'N'):
        path += '/{c}_{s}_{l}_{mth}-{ws}.{ext}'
    elif masktype == 'auto':
        path += '/{ap_}/{c}_{s}_{l}_{mth}-{ws}.{ext}'
    else:
        raise Exception('Unknown mask type: {mt}'.format(mt=masktype))
    return path.format(m=mode, c=case, s=scan, l=lesion, mt=masktype,
                       slices=slices, portion=portion, mth=method, ws=winsize,
                       ap_='_'.join(algparams), vx=voxel, ext=ext)


def std_cfg_path(mode):
    """Return path to standardization configuration file."""
    return 'stdcfg_{m}.txt'.format(m=mode)
