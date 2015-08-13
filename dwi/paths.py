"""Common path names."""

from __future__ import absolute_import, division, print_function
import os.path

import dwi.util


def samplelist_path(mode, samplelist):
    return 'patients_{m.modality}_{l}.txt'.format(m=mode, l=samplelist)


def pmap_path(mode, case=None, scan=None):
    path = dwi.util.sglob('dicoms_{m.model}_*'.format(m=mode), typ='dir')
    if case is not None and scan is not None:
        if mode.param == 'raw':
            # There's no actual parameter, only single 'raw' value (for T2).
            s = '/{c}_*_{s}*'
        else:
            s = '/{c}_*_{s}/{c}_*_{s}*_{m.param}'
        path += s.format(m=mode, c=case, s=scan)
    return dwi.util.sglob(path, typ='dir')


def subregion_path(mode, case=None, scan=None):
    path = 'subregions'
    if case is not None and scan is not None:
        path += '/{c}_{s}_subregion10.txt'.format(c=case, s=scan)
    return path


def mask_path(mode, masktype, case, scan, lesion=None, algparams=[]):
    """Return path and deps of masks of different types."""
    d = dict(m=mode, mt=masktype, c=case, s=scan, l=lesion,
             ap_='_'.join(algparams))
    do_glob = True
    if masktype == 'prostate':
        path = 'masks_{mt}_{m.modality}/{c}_*_{s}*'
    elif masktype == 'lesion':
        path = 'masks_{mt}_{m.modality}/PCa_masks_{m.modality}_{l}*/{c}_*{s}_*'
    elif masktype in ('CA', 'N'):
        path = 'masks_rois/{c}_*_{s}_D_{mt}'
    elif masktype == 'auto':
        path = 'masks_{mt}_{m}/{ap_}/{c}_{s}_auto.mask'
        do_glob = False  # Don't require existence, can be generated.
    else:
        raise Exception('Unknown mask type: {mt}'.format(**d))
    path = path.format(**d)
    if do_glob:
        path = dwi.util.sglob(path)
    return path


def roi_path(mode, masktype, case=None, scan=None, algparams=[]):
    """Return whole ROI path or part of it."""
    d = dict(m=mode, mt=masktype, c=case, s=scan, ap_='_'.join(algparams))
    components = ['rois_{mt}_{m}']
    if algparams:
        components.append('{ap_}')
    if case is not None and scan is not None:
        components.append('{c}_x_x_{s}_{m.model}_{m.param}_{mt}.txt')
    components = [x.format(**d) for x in components]
    return os.path.join(*components)


def texture_path(mode, case, scan, lesion, masktype, slices, portion,
                 algparams=()):
    """Return path to texture file."""
    path = 'texture_{mt}_{m.model}_{m.param}_{slices}_{portion}'
    if masktype in ('lesion', 'CA', 'N'):
        path += '/{c}_{s}_{l}.txt'
    elif masktype == 'auto':
        path += '/{ap_}/{c}_{s}_{l}.txt'
    else:
        raise Exception('Unknown mask type: {mt}'.format(mt=masktype))
    return path.format(m=mode, c=case, s=scan, l=lesion, mt=masktype,
                       slices=slices, portion=portion, ap_='_'.join(algparams))


def texture_path_new(mode, case, scan, lesion, masktype, slices, portion,
                     method, winsize, algparams=(), voxel='mean'):
    """Return path to texture file."""
    path = 'texture_{mt}_{m.model}_{m.param}_{slices}_{portion}_{vx}'
    if voxel == 'mean':
        ext = 'txt'
    else:
        ext = 'h5'
    if masktype in ('lesion', 'CA', 'N'):
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
