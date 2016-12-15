"""PyDoIt file for automating tasks."""

# TODO: Merge cases_scans() and lesions(). Remove read_sample_list().

from __future__ import absolute_import, division, print_function
from collections import defaultdict
from itertools import product
from os.path import dirname

from doit import get_var
# from doit.tools import check_timestamp_unchanged
from doit.tools import create_folder

import dwi.files
from dwi.paths import (samplelist_path, pmap_path, subregion_path, mask_path,
                       roi_path, std_cfg_path, texture_path, histogram_path,
                       grid_path)
import dwi.patient
import dwi.shell
import dwi.util


def words(string, sep=','):
    """Split string into stripped words."""
    return [x.strip() for x in string.split(sep)]


# Imaging modes.
DEFAULT_MODE = 'DWI-Mono-ADCm'
MODES = [dwi.util.ImageMode(_) for _ in words(get_var('mode', DEFAULT_MODE))]
MODE = MODES[0]

# Sample lists (train, test, etc).
SAMPLELISTS = words(get_var('samplelist', 'all'))
SAMPLELIST = SAMPLELISTS[0]


def name(*items):
    """A task name consisting of items."""
    s = '_'.join('{}' for _ in items)
    return s.format(*items)


def folders(*paths):
    """A PyDoIt action that creates the folders for given file names """
    return [(create_folder, [dirname(x)]) for x in paths]


def texture_methods():
    return dwi.rcParams['texture.methods']


def texture_winsizes(masktype, mode, method):
    if method.endswith('_all'):
        return ['all']
    elif method.endswith('_mbb'):
        return ['mbb']
    elif method == 'sobel':
        return [3]  # Sobel convolution kernel is always 3x3 voxels.
    elif masktype in ('CA', 'N'):
        return [3, 5]  # These ROIs are always 5x5 voxels.
    elif mode[0] in ('T2', 'T2w'):
        return range(*dwi.rcParams['texture.winsizes.large'])
    else:
        return range(*dwi.rcParams['texture.winsizes.small'])


def texture_methods_winsizes(mode, masktype):
    for method in texture_methods():
        for winsize in texture_winsizes(masktype, mode, method):
            yield method, winsize


def texture_params():
    masktypes = ['lesion']
    slices = ['maxfirst', 'all']
    portion = [1, 0]
    return product(masktypes, slices, portion)


def find_roi_param_combinations(mode):
    """Generate all find_roi.py parameter combinations."""
    find_roi_params = [
        [1, 2, 3],  # ROI depth min
        [1, 2, 3],  # ROI depth max
        range(2, 13),  # ROI side min (3 was not good)
        range(3, 13),  # ROI side max
        range(250, 2000, 250) + [50, 100, 150, 200],  # Number of ROIs
        ]
    if mode[0] == 'DWI':
        if SAMPLELIST == 'test':
            params = [
                (2, 3, 10, 10, 500),  # Mono: corr, auc
                (2, 3, 10, 10, 1750),  # Mono: corr
                (2, 3, 11, 11, 750),  # Mono: corr
                # (2, 3, 2, 2, 250),  # Kurt: auc
                # (2, 3, 9, 9, 1000),  # Kurt: corr
                # (2, 3, 12, 12, 1750),  # Kurt: corr
                # (2, 3, 5, 5, 500),  # Kurt K: corr, auc
                ]
        else:
            params = product(*find_roi_params)
        for t in params:
            if t[0] <= t[1] and t[2] == t[3]:
                yield [str(x) for x in t]


def cases_scans(mode, samplelist):
    """Generate all case, scan pairs."""
    samples = dwi.files.read_sample_list(samplelist_path(mode, samplelist))
    for sample in samples:
        case = sample['case']
        for scan in sample['scans']:
            yield case, scan


def lesions(mode, samplelist):
    """Generate all case, scan, lesion# (1-based) combinations."""
    patients = dwi.files.read_patients_file(samplelist_path(mode, samplelist))
    for p in patients:
        for scan in p.scans:
            for i, _ in enumerate(p.lesions):
                yield p.num, scan, i+1


#
# Tasks.
#


def task_standardize_train():
    """Standardize MRI images: training phase.

    Pay attention to the sample list: all samples should be used.
    """
    if MODE[0] != 'T2w':
        return
    mode = MODE - 'std'
    std_cfg = std_cfg_path(mode)
    # inpaths = [pmap_path(mode, c, s) for c, s in cases_scans(mode, 'all')]
    inpaths = [roi_path(mode, 'prostate', c, s) for c, s in cases_scans(mode,
                                                                        'all')]
    cmd = dwi.shell.standardize_train(inpaths, std_cfg, 'none')
    yield {
        'name': name(mode),
        'actions': [cmd],
        'file_dep': inpaths,
        'targets': [std_cfg],
        'clean': True,
        }


def task_standardize_transform():
    """Standardize MRI images: transform phase."""
    if MODE[0] != 'T2w':
        return
    mode = MODE - 'std'
    sl = SAMPLELIST
    cfgpath = std_cfg_path(mode)
    for case, scan in cases_scans(mode, sl):
        inpath = pmap_path(mode, case, scan)
        mask = mask_path(mode, 'prostate', case, scan)
        outpath = pmap_path(mode + 'std', case, scan)
        cmd = dwi.shell.standardize_transform(cfgpath, inpath, outpath,
                                              mask=mask)
        yield {
            'name': name(mode, case, scan),
            'actions': folders(outpath) + [cmd],
            'file_dep': [cfgpath, inpath, mask],
            'targets': [outpath],
            'clean': True,
            }


def task_make_subregion():
    """Make minimum bounding box + 10 voxel subregions from prostate masks."""
    for mode, sl in product(MODES, SAMPLELISTS):
        if str(mode) == 'DWI-Mono-ADCm':
            for case, scan in cases_scans(mode, sl):
                mask = mask_path(mode, 'prostate', case, scan)
                subregion = subregion_path(mode, case, scan)
                cmd = dwi.shell.make_subregion(mask, subregion)
                yield {
                    'name': name(mode, case, scan),
                    'actions': folders(subregion) + [cmd],
                    'file_dep': [mask],
                    'targets': [subregion],
                    'clean': True,
                    }


# def task_fit():
#     """Fit models to imaging data."""
#     for mode, sl in product(MODES, SAMPLELISTS):
#         if mode[0] in ('T2',):
#             for c, s in cases_scans(mode, sl):
#                 inpath = pmap_path(mode, c, s)
#                 outpath = pmap_path(mode+'fitted', c, s, fmt='h5')
#                 model = 'T2'
#                 mask = mask_path(mode, 'prostate', c, s)
#                 mbb = (0, 20, 20)
#                 cmd = dwi.shell.fit(inpath, outpath, model, mask=mask,
#                                     mbb=mbb)
#                 yield {
#                     'name': name(mode, c, s, model),
#                     'actions': folders(outpath) + [cmd],
#                     'file_dep': [inpath, mask],
#                     'targets': [outpath],
#                     'clean': True,
#                     }


def get_task_select_roi_lesion(mode, case, scan, lesion):
    """Select ROIs from the pmap DICOMs based on masks."""
    masktype = 'lesion'
    mask = mask_path(mode, 'lesion', case, scan, lesion=lesion)
    roi = roi_path(mode, masktype, case, scan, lesion=lesion)
    pmap = pmap_path(mode, case, scan)
    d = dict(mask=mask, keepmasked=True)
    if 'std' in mode:
        d['astype'] = 'float32'  # Integers cannot have nans.
    cmd = dwi.shell.select_voxels(pmap, roi, **d)
    return {
        'name': name(mode, masktype, case, scan, lesion),
        'actions': folders(roi) + [cmd],
        'file_dep': [mask, pmap],
        'targets': [roi],
        'clean': True,
        }


def get_task_select_roi_manual(mode, case, scan, masktype):
    """Select ROIs from the pmap DICOMs based on masks."""
    mask = mask_path(mode, masktype, case, scan)
    roi = roi_path(mode, masktype, case, scan)
    pmap = pmap_path(mode, case, scan)
    d = dict(mask=mask, keepmasked=(masktype == 'prostate'))
    if 'std' in mode:
        d['astype'] = 'float32'  # Integers cannot have nans.
    cmd = dwi.shell.select_voxels(pmap, roi, **d)
    return {
        'name': name(mode, masktype, case, scan),
        'actions': folders(roi) + [cmd],
        'file_dep': [mask, pmap],
        'targets': [roi],
        'clean': True,
        }


def get_task_select_roi_auto(mode, case, scan, algparams):
    """Select ROIs from the pmap DICOMs based on masks."""
    ap_ = '_'.join(algparams)
    mask = mask_path(mode, 'auto', case, scan, algparams=algparams)
    roi = roi_path(mode, 'auto', case, scan, algparams=algparams)
    pmap = pmap_path(mode, case, scan)
    cmd = dwi.shell.select_voxels(pmap, roi, mask=mask, keepmasked=False)
    return {
        'name': name(mode, ap_, case, scan),
        'actions': folders(roi) + [cmd],
        'file_dep': [mask, pmap],
        'targets': [roi],
        'clean': True,
        }


def task_select_roi_lesion():
    """Select lesion ROIs from the pmap DICOMs."""
    for mode, sl in product(MODES, SAMPLELISTS):
        for c, s, l in lesions(mode, sl):
            yield get_task_select_roi_lesion(mode, c, s, l)


def task_select_roi_manual():
    """Select manually selected ROIs from the pmap DICOMs."""
    for mode, sl in product(MODES, SAMPLELISTS):
        masktypes = ('prostate',)
        if mode[0] == 'DWI':
            masktypes += ('CA', 'N')
        for mt in masktypes:
            for c, s in cases_scans(mode, sl):
                try:
                    yield get_task_select_roi_manual(mode, c, s, mt)
                except IOError as e:
                    print('select_roi_manual', e)


def task_select_roi_auto():
    """Select automatic ROIs from the pmap DICOMs."""
    for mode, sl in product(MODES, SAMPLELISTS):
        if mode[0] == 'DWI':
            for algparams in find_roi_param_combinations(mode):
                for c, s in cases_scans(mode, sl):
                    try:
                        yield get_task_select_roi_auto(mode, c, s, algparams)
                    except IOError as e:
                        print('select_roi_auto', e)


def task_select_roi():
    """Select all ROIs task group."""
    return {
        'actions': None,
        'task_dep': ['select_roi_lesion', 'select_roi_manual',
                     'select_roi_auto'],
        }


def get_task_texture(mode, masktype, case, scan, lesion, slices, portion,
                     method, winsize, voxel):
    """Generate texture features."""
    inpath = pmap_path(mode, case, scan)
    deps = [inpath]
    mask = mask_path(mode, masktype, case, scan, lesion=lesion)
    if mask is not None:
        deps.append(mask)
    outfile = texture_path(mode, case, scan, lesion, masktype, slices, portion,
                           method, winsize, voxel=voxel)
    cmd = dwi.shell.get_texture(mode, inpath, method, winsize, slices, portion,
                                outfile, voxel, mask=mask)
    return {
        'name': name(mode, masktype, slices, portion, case, scan, lesion,
                     method, winsize, voxel),
        'actions': folders(outfile) + [cmd],
        'file_dep': deps,
        'targets': [outfile],
        'clean': True,
        }


def task_texture():
    """Generate texture features."""
    for mode, sl in product(MODES, SAMPLELISTS):
        for mt, slices, portion in texture_params():
            for c, s, l in lesions(mode, sl):
                for mth, ws in texture_methods_winsizes(mode, mt):
                    yield get_task_texture(mode, mt, c, s, l, slices, portion,
                                           mth, ws, 'mean')
                    yield get_task_texture(mode, mt, c, s, l, slices, portion,
                                           mth, ws, 'all')
        mt = 'prostate'
        for c, s in cases_scans(mode, sl):
            for mth, ws in texture_methods_winsizes(mode, mt):
                # yield get_task_texture(mode, mt, c, s, None, 'maxfirst', 0,
                #                        mth, ws, 'all')
                yield get_task_texture(mode, mt, c, s, None, 'all', 0, mth, ws,
                                       'all')
        mt = 'all'
        for c, s in cases_scans(mode, sl):
            for mth, ws in texture_methods_winsizes(mode, mt):
                yield get_task_texture(mode, mt, c, s, None, 'all', 0, mth, ws,
                                       'all')


def task_merge_textures():
    """Merge texture methods into singe file per case/scan/lesion."""
    for mode, sl in product(MODES, SAMPLELISTS):
        for mt, slices, portion in texture_params():
            for c, s, l in lesions(mode, sl):
                infiles = [texture_path(mode, c, s, l, mt, slices,
                                        portion, mth, ws) for mth, ws in
                           texture_methods_winsizes(mode, mt)]
                outfile = texture_path(mode, c, s, l, mt+'_merged',
                                       slices, portion, None, None)
                cmd = dwi.shell.select_voxels(' '.join(infiles), outfile)
                yield {
                    'name': name(mode, c, s, l, mt, slices, portion),
                    'actions': folders(outfile) + [cmd],
                    'file_dep': infiles,
                    'targets': [outfile],
                    }


def get_task_histogram(mode, masktype, samplelist):
    if masktype == 'lesion':
        it = lesions(mode, samplelist)
    else:
        it = cases_scans(mode, samplelist)
    inpaths = [roi_path(mode, masktype, *x) for x in it]
    figpath = histogram_path(mode, masktype, samplelist)
    cmd = dwi.shell.histogram(inpaths, figpath, param=0)
    return {
        'name': name(mode, masktype, samplelist),
        'actions': folders(figpath) + [cmd],
        'file_dep': inpaths,
        'targets': [figpath],
        }


def task_histogram():
    """Plot image histograms."""
    for mode, sl in product(MODES, SAMPLELISTS):
        for mt in ('image', 'prostate', 'lesion'):
            yield get_task_histogram(mode, mt, sl)


def get_task_grid(mode, c, s, ls, mt, lt=None, mbb=None, nanbg=False,
                  fmt='txt'):
    """Grid classifier."""
    pmap = pmap_path(mode, c, s)
    prostate = mask_path(mode, 'prostate', c, s)
    lesion = [mask_path(mode, 'lesion', c, s, x) for x in ls]
    out, target = grid_path(mode, c, s, mt, ['raw'], fmt=fmt)
    if fmt == 'h5':
        target = out
    z = 5
    # z = 1
    d = dict(mbb=mbb, voxelsize=None, winsize=z, voxelspacing=(z, 1, 1),
             lesiontypes=lt, use_centroid=False, nanbg=nanbg)
    cmd = dwi.shell.grid(pmap, 0, prostate, lesion, out, **d)
    return {
        'name': name(mode, c, s, mt),
        'actions': folders(out) + [cmd],
        'file_dep': [pmap, prostate] + lesion,
        'targets': [target],
        }


def get_task_grid_texture(mode, c, s, ls, mt, mth, ws, lt=None, mbb=None,
                          nanbg=False, fmt='txt'):
    """Grid classifier."""
    pmap = texture_path(mode, c, s, None, mt, 'all', 0, mth, ws, voxel='all')
    prostate = mask_path(mode, 'prostate', c, s)
    lesion = [mask_path(mode, 'lesion', c, s, x) for x in ls]
    out, target = grid_path(mode, c, s, mt, [mth, ws], fmt=fmt)
    if fmt == 'h5':
        target = out
    z = 5
    # z = 1
    d = dict(mbb=mbb, voxelsize=None, winsize=z, voxelspacing=(z, 1, 1),
             lesiontypes=lt, use_centroid=False, nanbg=nanbg)
    cmd = dwi.shell.grid(pmap, None, prostate, lesion, out, **d)
    return {
        'name': name(mode, c, s, mt, mth, ws),
        'actions': folders(out) + [cmd],
        'file_dep': [pmap, prostate] + lesion,
        'targets': [target],
        }


def task_grid():
    """Grid classifier."""
    for mode, sl in product(MODES, SAMPLELISTS):
        lesioninfo = defaultdict(list)
        for c, s, l in dwi.patient.iterlesions(samplelist_path(mode, sl)):
            c, l, lt = c.num, l.index + 1, l.location
            lesioninfo[(c, s)].append((l, lt))
        for k, v in lesioninfo.iteritems():
            c, s = k
            ls, lt = [x[0] for x in v], [x[1] for x in v]

            d = dict(lt=lt, mbb=None, fmt='h5')

            mt = 'prostate'
            d['nanbg'] = True
            yield get_task_grid(mode, c, s, ls, mt, **d)
            for mth, ws in texture_methods_winsizes(mode, mt):
                yield get_task_grid_texture(mode, c, s, ls, mt, mth, ws, **d)

            mt = 'all'
            d['nanbg'] = False
            yield get_task_grid(mode, c, s, ls, mt, **d)
            for mth, ws in texture_methods_winsizes(mode, mt):
                yield get_task_grid_texture(mode, c, s, ls, mt, mth, ws, **d)
