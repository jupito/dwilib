"""PyDoIt file for automating tasks."""

from __future__ import absolute_import, division, print_function
from collections import defaultdict
from itertools import chain, product
from os.path import dirname

from doit import get_var
from doit.tools import check_timestamp_unchanged, create_folder

import dwi.files
from dwi.paths import (samplelist_path, pmap_path, subregion_path, mask_path,
                       roi_path, std_cfg_path, texture_path, histogram_path,
                       grid_path)
import dwi.patient
import dwi.shell
import dwi.util


# Backends:
# dbm: (default) It uses python dbm module.
# json: Plain text using a json structure, it is slow but good for debugging.
# sqlite3: (experimental) very slow implementation, support concurrent access.

DOIT_CONFIG = {
    'backend': 'sqlite3',
    'default_tasks': [],
    'verbosity': 1,
    'num_process': 7,
    }


def words(string, sep=','):
    """Split string into stripped words."""
    return [x.strip() for x in string.split(sep)]


# Imaging modes.
MODES = [dwi.patient.ImageMode(_) for _ in words(get_var('mode',
                                                         'DWI-Mono-ADCm'))]
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
    return [
        # 'stats',
        # 'haralick',
        # 'moment',
        # 'haralick_mbb',

        'glcm',
        'glcm_mbb',
        'lbp',
        'hog',
        'gabor',
        'haar',
        'hu',
        'zernike',
        'sobel',
        'stats_all',
        ]


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
        return xrange(3, 30, 4)
    else:
        return xrange(3, 16, 2)


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
        xrange(2, 13),  # ROI side min (3 was not good)
        xrange(3, 13),  # ROI side max
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


def path_deps(*paths):
    """Return list of path dependencies, i.e. the file(s) itself or the
    directory contents.
    """
    # paths = [dwi.util.sglob(x) for x in paths]  # First make sure all exist.
    for i, path in enumerate(paths):
        if '*' in path or ('[' in path and ']' in path):
            paths[i] = dwi.util.sglob(path)
    paths = list(chain(f for p in paths for f in dwi.util.walker(p)))
    # paths = [x for x in paths if not isdir(x)]
    return paths


#
# Tasks.
#


def task_standardize_train():
    """Standardize MRI images: training phase.

    Pay attention to the sample list: all samples should be used.
    """
    mode = MODE - 'std'
    std_cfg = std_cfg_path(mode)
    # inpaths = [pmap_path(mode, c, s) for c, s in cases_scans(mode, 'all')]
    inpaths = [roi_path(mode, 'prostate', c, s) for c, s in cases_scans(mode,
                                                                        'all')]
    yield {
        'name': name(mode),
        'actions': [dwi.shell.standardize_train_cmd(inpaths, std_cfg, 'none')],
        'file_dep': path_deps(*inpaths),
        'targets': [std_cfg],
        'clean': True,
        }


def task_standardize_transform():
    """Standardize MRI images: transform phase."""
    mode = MODE - 'std'
    sl = SAMPLELIST
    cfgpath = std_cfg_path(mode)
    for case, scan in cases_scans(mode, sl):
        inpath = pmap_path(mode, case, scan)
        mask = mask_path(mode, 'prostate', case, scan)
        outpath = pmap_path(mode + 'std', case, scan)
        cmd = dwi.shell.standardize_transform_cmd(cfgpath, inpath, outpath,
                                                  mask=mask)
        yield {
            'name': name(mode, case, scan),
            'actions': folders(outpath) + [cmd],
            'file_dep': path_deps(cfgpath, inpath, mask),
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
                cmd = dwi.shell.make_subregion_cmd(mask, subregion)
                yield {
                    'name': name(mode, case, scan),
                    'actions': folders(subregion) + [cmd],
                    'file_dep': path_deps(mask),
                    'targets': [subregion],
                    'clean': True,
                    }


def task_fit():
    """Fit models to imaging data."""
    for mode, sl in product(MODES, SAMPLELISTS):
        if mode[0] in ('T2',):
            for c, s in cases_scans(mode, sl):
                inpath = pmap_path(mode, c, s)
                outpath = pmap_path(mode+'fitted', c, s, fmt='hdf5')
                model = 'T2'
                mask = mask_path(mode, 'prostate', c, s)
                mbb = (0, 20, 20)
                cmd = dwi.shell.fit_cmd(inpath, outpath, model, mask=mask,
                                        mbb=mbb)
                yield {
                    'name': name(mode, c, s, model),
                    'actions': folders(outpath) + [cmd],
                    'file_dep': path_deps(inpath, mask),
                    'targets': [outpath],
                    'clean': True,
                }


# def get_task_find_roi(mode, case, scan, algparams):
#     d = dict(m=mode, c=case, s=scan, ap_='_'.join(algparams))
#     outmask = mask_path(mode, 'auto', case, scan, algparams=algparams)
#     outfig = 'find_roi_images/{m}/{ap_}/{c}_{s}.png'.format(**d)
#     subregion = subregion_path(mode, case, scan)
#     mask_p = mask_path(mode, 'prostate', case, scan)
#     mask_c = mask_path(mode, 'CA', case, scan)
#     mask_n = mask_path(mode, 'N', case, scan)
#     cmd = dwi.shell.find_roi_cmd(mode, case, scan, algparams, outmask,
#                                  outfig)
#     return {
#         'name': '{m}_{ap_}_{c}_{s}'.format(**d),
#         'actions': folders(outmask, outfig) + [cmd],
#         'file_dep': path_deps(subregion, mask_p, mask_c, mask_n),
#         'targets': [outmask, outfig],
#         'clean': True,
#         }


# def task_find_roi():
#     """Find a cancer ROI automatically."""
#     for algparams in find_roi_param_combinations(MODE):
#         for case, scan in cases_scans(MODE, SAMPLELIST):
#             yield get_task_find_roi(MODE, case, scan, algparams)


def get_task_select_roi_lesion(mode, case, scan, lesion):
    """Select ROIs from the pmap DICOMs based on masks."""
    masktype = 'lesion'
    mask = mask_path(mode, 'lesion', case, scan, lesion=lesion)
    roi = roi_path(mode, masktype, case, scan, lesion=lesion)
    pmap = pmap_path(mode, case, scan)
    d = dict(mask=mask, keepmasked=True)
    if 'std' in mode:
        d['astype'] = 'float32'  # Integers cannot have nans.
    cmd = dwi.shell.select_voxels_cmd(pmap, roi, **d)
    return {
        'name': name(mode, masktype, case, scan, lesion),
        'actions': folders(roi) + [cmd],
        'file_dep': path_deps(mask, pmap),
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
    cmd = dwi.shell.select_voxels_cmd(pmap, roi, **d)
    return {
        'name': name(mode, masktype, case, scan),
        'actions': folders(roi) + [cmd],
        'file_dep': path_deps(mask, pmap),
        'targets': [roi],
        'uptodate': [check_timestamp_unchanged(pmap)],
        'clean': True,
        }


def get_task_select_roi_auto(mode, case, scan, algparams):
    """Select ROIs from the pmap DICOMs based on masks."""
    ap_ = '_'.join(algparams)
    mask = mask_path(mode, 'auto', case, scan, algparams=algparams)
    roi = roi_path(mode, 'auto', case, scan, algparams=algparams)
    pmap = pmap_path(mode, case, scan)
    cmd = dwi.shell.select_voxels_cmd(pmap, roi, mask=mask, keepmasked=False)
    return {
        'name': name(mode, ap_, case, scan),
        'actions': folders(roi) + [cmd],
        'file_dep': [mask, pmap],
        'targets': [roi],
        'uptodate': [check_timestamp_unchanged(pmap)],
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


# def get_task_autoroi_auc(mode, threshold):
#     """Evaluate auto-ROI prediction ability by ROC AUC with Gleason score."""
#     d = dict(m=mode, sl=SAMPLELIST, t=threshold)
#     outfile = 'autoroi_auc_{t}_{m}_{sl}.txt'.format(**d)
#     cmds = ['echo -n > {o}'.format(o=outfile)]
#     for algparams in find_roi_param_combinations(mode):
#         cmds.append(dwi.shell.auc_cmd(mode, threshold, algparams, outfile))
#     return {
#         'name': 'autoroi_auc_{sl}_{m}_{t}'.format(**d),
#         'actions': cmds,
#         'task_dep': ['select_roi_auto'],
#         'targets': [outfile],
#         'clean': True,
#         }


# def get_task_autoroi_correlation(mode, thresholds):
#     """Evaluate auto-ROI prediction ability by correlation with Gleason
#     score."""
#     d = dict(m=mode, sl=SAMPLELIST, t_=thresholds.replace(' ', ','))
#     outfile = 'autoroi_correlation_{t_}_{m}_{sl}.txt'.format(**d)
#     cmds = ['echo -n > {o}'.format(o=outfile)]
#     for algparams in find_roi_param_combinations(mode):
#         cmds.append(dwi.shell.correlation_cmd(mode, thresholds, algparams,
#                                               outfile))
#     return {
#         'name': 'autoroi_correlation_{sl}_{m}_{t_}'.format(**d),
#         'actions': cmds,
#         'task_dep': ['select_roi_auto'],
#         'targets': [outfile],
#         'clean': True,
#         }


# def task_evaluate_autoroi():
#     """Evaluate auto-ROI prediction ability."""
#     yield get_task_autoroi_auc(MODE, '3+3')
#     yield get_task_autoroi_auc(MODE, '3+4')
#     yield get_task_autoroi_correlation(MODE, '3+3 3+4')
#     yield get_task_autoroi_correlation(MODE, '')


def get_task_texture(mode, masktype, case, scan, lesion, slices, portion,
                     method, winsize, voxel):
    """Generate texture features."""
    inpath = pmap_path(mode, case, scan)
    mask = mask_path(mode, masktype, case, scan, lesion=lesion)
    outfile = texture_path(mode, case, scan, lesion, masktype, slices, portion,
                           method, winsize, voxel=voxel)
    cmd = dwi.shell.get_texture_cmd(mode, inpath, method, winsize, slices,
                                    portion, mask, outfile, voxel)
    return {
        'name': name(mode, masktype, slices, portion, case, scan, lesion,
                     method, winsize, voxel),
        'actions': folders(outfile) + [cmd],
        'file_dep': path_deps(inpath, mask),
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
                yield get_task_texture(mode, mt, c, s, None, 'maxfirst', 0,
                                       mth, ws, 'all')
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
                cmd = dwi.shell.select_voxels_cmd(' '.join(infiles), outfile)
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
    cmd = dwi.shell.histogram_cmd(inpaths, figpath)
    return {
        'name': name(mode, masktype, samplelist),
        'actions': folders(figpath) + [cmd],
        'file_dep': path_deps(*inpaths),
        'targets': [figpath],
    }


def task_histogram():
    """Plot image histograms."""
    for mode, sl in product(MODES, SAMPLELISTS):
        for mt in ('image', 'prostate', 'lesion'):
            yield get_task_histogram(mode, mt, sl)


def get_task_grid(mode, c, s, ls):
    """Grid classifier."""
    pmap = pmap_path(mode, c, s)
    prostate = mask_path(mode, 'prostate', c, s)
    lesion = [mask_path(mode, 'lesion', c, s, x) for x in ls]
    # fmt = 'h5'
    fmt = 'txt'
    out, target = grid_path(mode, c, s, ['raw'], fmt=fmt)
    cmd = dwi.shell.grid_cmd(pmap, None, prostate, lesion, out)
    return {
        'name': name(mode, c, s),
        'actions': folders(out) + [cmd],
        'file_dep': path_deps(pmap, prostate, *lesion),
        'targets': [target],
    }


def get_task_grid_texture(mode, c, s, ls, mth, ws):
    """Grid classifier."""
    pmap = texture_path(mode, c, s, None, 'prostate', 'all', 0, mth, ws,
                        voxel='all')
    prostate = mask_path(mode, 'prostate', c, s)
    lesion = [mask_path(mode, 'lesion', c, s, x) for x in ls]
    # fmt = 'h5'
    fmt = 'txt'
    out, target = grid_path(mode, c, s, [mth, ws], fmt=fmt)
    cmd = dwi.shell.grid_cmd(pmap, None, prostate, lesion, out)
    return {
        'name': name(mode, c, s, mth, ws),
        'actions': folders(out) + [cmd],
        'file_dep': path_deps(pmap, prostate, *lesion),
        'targets': [target],
    }


def task_grid():
    """Grid classifier."""
    for mode, sl in product(MODES, SAMPLELISTS):
        d = defaultdict(list)
        for c, s, l in lesions(mode, sl):
            d[(c, s)].append(l)
        for k, ls in d.iteritems():
            c, s = k
            yield get_task_grid(mode, c, s, ls)
            for mth, ws in texture_methods_winsizes(mode, 'prostate'):
                yield get_task_grid_texture(mode, c, s, ls, mth, ws)


# def task_check_mask_overlap():
#     """Check mask overlap."""
#     for mode, sl in product(MODES, SAMPLELISTS):
#         for c, s, l in lesions(mode, sl):
#             container = mask_path(mode, 'prostate', c, s)
#             other = mask_path(mode, 'lesion', c, s, l)
#             fig = 'maskoverlap/{m}/{c}-{s}-{l}.png'.format(m=mode[0], c=c,
#                                                            s=s, l=l)
#             cmd = dwi.shell.check_mask_overlap_cmd(container, other, fig)
#             yield {
#                 'name': name(mode, c, s, l),
#                 'actions': folders(fig) + [cmd],
#                 'file_dep': path_deps(container, other),
#                 'targets': [fig],
#             }
