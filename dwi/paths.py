"""Common path names."""

from __future__ import absolute_import, division, print_function

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


def mask_path(mode, masktype, case, scan, lesion=None, algparams=(),
              fmt='dicom'):
    """Return path and deps of masks of different types."""
    d = dict(m=mode, mt=masktype, c=case, s=scan, l=lesion,
             ap_='_'.join(algparams))
    if fmt == 'hdf5':
        return 'masks_{mt}/{m}/{c}_{s}_{l}.h5'.format(**d)
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
    if masktype == 'image':
        return pmap_path(mode, case, scan)  # No ROI, but the whole image.
    d = dict(m=mode, mt=masktype, c=case, s=scan, l=lesion,
             ap_='_'.join(algparams))
    path = 'rois_{mt}/{m}'
    if algparams:
        path += '/{ap_}'
    if case is not None and scan is not None:
        if masktype == 'lesion':
            path += '/{c}-{s}-{l}.h5'
        else:
            path += '/{c}_x_x_{s}_{m}_{mt}.txt'
    return path.format(**d)


def texture_path(mode, case, scan, lesion, masktype, slices, portion, method,
                 winsize, algparams=(), voxel='mean'):
    """Return path to texture file."""
    if voxel == 'mean':
        ext = 'txt'
    else:
        ext = 'h5'
    path = 'texture_{mt}/{m}_{slices}_{portion}_{vx}'
    if masktype == 'auto':
        path += '/{ap_}'
    path += '/{c}_{s}_{l}'
    if method is not None and winsize is not None:
        path += '_{mth}-{ws}'
    path += '.{ext}'
    return path.format(m=mode, c=case, s=scan, l=lesion, mt=masktype,
                       slices=slices, portion=portion, mth=method, ws=winsize,
                       ap_='_'.join(algparams), vx=voxel, ext=ext)


def std_cfg_path(mode):
    """Return path to standardization configuration file."""
    return 'stdcfg_{m}.txt'.format(m=mode)


def histogram_path(mode, roi, samplelist):
    """Return path to histogram plot."""
    return 'histograms/{m}_{r}_{s}.png'.format(m=mode, r=roi, s=samplelist)
