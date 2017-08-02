"""PyDoIt file for automating tasks."""

import logging
from collections import defaultdict
from itertools import product

# from doit import get_var
# from doit.tools import check_timestamp_unchanged

import dwi.dataset
import dwi.paths
import dwi.shell
from dwi import rcParams
from dwi.doit import (_files, cases_scans, find_roi_param_combinations,
                      folders, lesions, taskname, texture_methods_winsizes,
                      texture_params)
from dwi.types import TextureSpec

DOIT_CONFIG = dwi.doit.get_config()
MODES = rcParams.modes
SAMPLELISTS = rcParams.samplelists

MODE = MODES[0]  # XXX: Only first mode used.
SAMPLELIST = SAMPLELISTS[0]  # XXX: Only first samplelist used.

logging.info('Imaging modes: %s', rcParams.modes)
logging.info('samplelists: %s', rcParams.samplelists)
logging.info('Using %d processes', DOIT_CONFIG['num_process'])


def task_standardize_train():
    """Standardize MRI images: training phase.

    Pay attention to the sample list: all samples should be used.
    """
    if MODE[0] != 'T2w':
        return
    # mode = MODE - 'std'
    mode = MODE[:-1] if MODE[-1] == 'std' else MODE
    paths = dwi.paths.Paths(mode)
    std_cfg = paths.std_cfg()
    inpaths = [paths.roi('prostate', case=c, scan=s) for c, s in
               cases_scans(mode, 'all')]
    cmd = dwi.shell.standardize_train(inpaths, std_cfg, 'none')
    yield {
        'name': taskname(mode),
        'actions': [cmd],
        'file_dep': _files(*inpaths),
        'targets': _files(std_cfg),
        'clean': True,
        }


def task_standardize_transform():
    """Standardize MRI images: transform phase."""
    if MODE[0] != 'T2w':
        return
    # mode = MODE - 'std'
    mode = MODE[:-1] if MODE[-1] == 'std' else MODE
    sl = SAMPLELIST
    paths = dwi.paths.Paths(mode)
    outpaths = dwi.paths.Paths(tuple(mode) + ('std',))
    cfgpath = paths.std_cfg()
    for case, scan in cases_scans(mode, sl):
        inpath = paths.pmap(case=case, scan=scan)
        mask = paths.mask('prostate', case, scan)
        outpath = outpaths.pmap(case=case, scan=scan)
        cmd = dwi.shell.standardize_transform(cfgpath, inpath, outpath,
                                              mask=mask)
        yield {
            'name': taskname(mode, case, scan),
            'actions': folders(outpath) + [cmd],
            'file_dep': _files(cfgpath, inpath, mask),
            'targets': _files(outpath),
            'clean': True,
            }


def task_make_subregion():
    """Make minimum bounding box + 10 voxel subregions from prostate masks."""
    for mode, sl in product(MODES, SAMPLELISTS):
        paths = dwi.paths.Paths(mode)
        if str(mode) == 'DWI-Mono-ADCm':
            for case, scan in cases_scans(mode, sl):
                mask = paths.mask('prostate', case, scan)
                subregion = paths.subregion(case=case, scan=scan)
                cmd = dwi.shell.make_subregion(mask, subregion)
                yield {
                    'name': taskname(mode, case, scan),
                    'actions': folders(subregion) + [cmd],
                    'file_dep': _files(mask),
                    'targets': _files(subregion),
                    'clean': True,
                    }


# def task_fit():
#     """Fit models to imaging data."""
#     for mode, sl in product(MODES, SAMPLELISTS):
#         paths = dwi.paths.Paths(mode)
#         outpaths = dwi.paths.Paths(mode + 'fitted')
#         if mode[0] in ('T2',):
#             for c, s in cases_scans(mode, sl):
#                 inpath = paths.pmap(case=c, scan=s)
#                 outpath = outpaths.pmap(case=c, scan=s, fmt='h5')
#                 model = 'T2'
#                 mask = paths.mask('prostate', c, s)
#                 mbb = (0, 20, 20)
#                 cmd = dwi.shell.fit(inpath, outpath, model, mask=mask,
#                                     mbb=mbb)
#                 yield {
#                     'name': taskname(mode, c, s, model),
#                     'actions': folders(outpath) + [cmd],
#                     'file_dep': _files(inpath, mask),
#                     'targets': _files(outpath),
#                     'clean': True,
#                     }


def get_task_select_roi_lesion(mode, case, scan, lesion):
    """Select ROIs from the pmap DICOMs based on masks."""
    paths = dwi.paths.Paths(mode)
    masktype = 'lesion'
    mask = paths.mask('lesion', case, scan, lesion=lesion)
    roi = paths.roi(masktype, case=case, scan=scan, lesion=lesion)
    pmap = paths.pmap(case=case, scan=scan)
    d = dict(mask=mask, keepmasked=True)
    if 'std' in mode:
        d['astype'] = 'float32'  # Integers cannot have nans.
    cmd = dwi.shell.select_voxels(pmap, roi, **d)
    return {
        'name': taskname(mode, masktype, case, scan, lesion),
        'actions': folders(roi) + [cmd],
        'file_dep': _files(mask, pmap),
        'targets': _files(roi),
        'clean': True,
        }


def get_task_select_roi_manual(mode, case, scan, masktype):
    """Select ROIs from the pmap DICOMs based on masks."""
    paths = dwi.paths.Paths(mode)
    mask = paths.mask(masktype, case, scan)
    roi = paths.roi(masktype, case=case, scan=scan)
    pmap = paths.pmap(case=case, scan=scan)
    d = dict(mask=mask, keepmasked=(masktype == 'prostate'))
    if 'std' in mode:
        d['astype'] = 'float32'  # Integers cannot have nans.
    cmd = dwi.shell.select_voxels(pmap, roi, **d)
    return {
        'name': taskname(mode, masktype, case, scan),
        'actions': folders(roi) + [cmd],
        'file_dep': _files(mask, pmap),
        'targets': _files(roi),
        'clean': True,
        }


def get_task_select_roi_auto(mode, case, scan, algparams):
    """Select ROIs from the pmap DICOMs based on masks."""
    paths = dwi.paths.Paths(mode)
    ap_ = '_'.join(algparams)
    mask = paths.mask('auto', case, scan, algparams=algparams)
    roi = paths.roi('auto', case=case, scan=scan, algparams=algparams)
    pmap = paths.pmap(case=case, scan=scan)
    cmd = dwi.shell.select_voxels(pmap, roi, mask=mask, keepmasked=False)
    return {
        'name': taskname(mode, ap_, case, scan),
        'actions': folders(roi) + [cmd],
        'file_dep': _files(mask, pmap),
        'targets': _files(roi),
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
                yield get_task_select_roi_manual(mode, c, s, mt)


def task_select_roi_auto():
    """Select automatic ROIs from the pmap DICOMs."""
    for mode, sl in product(MODES, SAMPLELISTS):
        if mode[0] == 'DWI':
            for algparams in find_roi_param_combinations(mode, sl):
                for c, s in cases_scans(mode, sl):
                    yield get_task_select_roi_auto(mode, c, s, algparams)


def task_select_roi():
    """Select all ROIs task group."""
    return {
        'actions': None,
        'task_dep': ['select_roi_lesion', 'select_roi_manual',
                     'select_roi_auto'],
        }


def get_task_texture(mode, masktype, case, scan, lesion, slices, portion,
                     tspec, voxel):
    """Generate texture features."""
    paths = dwi.paths.Paths(mode)
    inpath = paths.pmap(case=case, scan=scan)
    deps = [inpath]
    mask = paths.mask(masktype, case, scan, lesion=lesion)
    if mask is not None:
        deps.append(mask)
    outfile = paths.texture(case, scan, lesion, masktype, slices, portion,
                            tspec, voxel=voxel)
    cmd = dwi.shell.get_texture(mode, inpath, tspec, slices, portion, outfile,
                                voxel, mask=mask)
    return {
        'name': taskname(mode, masktype, slices, portion, case, scan, lesion,
                         tspec.method, tspec.winsize, voxel),
        'actions': folders(outfile) + [cmd],
        'file_dep': _files(*deps),
        'targets': _files(outfile),
        'clean': True,
        }


def task_texture():
    """Generate texture features."""
    for mode, sl in product(MODES, SAMPLELISTS):
        for mt, slices, portion, voxel in texture_params():
            for c, s, l in lesions(mode, sl):
                for tspec in texture_methods_winsizes(mode, mt):
                    yield get_task_texture(mode, mt, c, s, l, slices, portion,
                                           tspec, voxel)
        mt = 'prostate'
        for c, s in cases_scans(mode, sl):
            for tspec in texture_methods_winsizes(mode, mt):
                yield get_task_texture(mode, mt, c, s, None, 'maxfirst', 0,
                                       tspec, 'all')
                yield get_task_texture(mode, mt, c, s, None, 'all', 0, tspec,
                                       'all')
        mt = 'all'
        for c, s in cases_scans(mode, sl):
            for tspec in texture_methods_winsizes(mode, mt):
                yield get_task_texture(mode, mt, c, s, None, 'all', 0, tspec,
                                       'all')
        for mt in ['CA', 'N']:
            for c, s in cases_scans(mode, sl):
                for tspec in texture_methods_winsizes(mode, mt):
                    yield get_task_texture(mode, mt, c, s, None, 'all', 0,
                                           tspec, 'median')


def get_task_merge_textures(mode, mt, c, s, l, slices, portion, voxel):
    """Merge texture methods into singe file per case/scan/lesion."""
    paths = dwi.paths.Paths(mode)
    infiles = [paths.texture(c, s, l, mt, slices, portion, tspec, voxel=voxel)
               for tspec in texture_methods_winsizes(mode, mt)]
    outfile = paths.texture(c, s, l, mt+'_merged', slices, portion, None,
                            voxel=voxel)
    cmd = dwi.shell.select_voxels(' '.join(map(str, infiles)), outfile)
    return {
        'name': taskname(mode, c, s, l, mt, slices, portion,
                         voxel),
        'actions': folders(outfile) + [cmd],
        'file_dep': _files(*infiles),
        'targets': _files(outfile),
        }


def task_merge_textures():
    """Merge texture methods into singe file per case/scan/lesion."""
    for mode, sl in product(MODES, SAMPLELISTS):
        for mt, slices, portion, voxel in texture_params(voxels=['mean',
                                                                 'median']):
            for c, s, l in lesions(mode, sl):
                yield get_task_merge_textures(mode, mt, c, s, l, slices,
                                              portion, voxel)
        for mt in ['CA', 'N']:
            slices, portion, voxel, l = 'all', 0, 'median', None
            for c, s in cases_scans(mode, sl):
                yield get_task_merge_textures(mode, mt, c, s, l, slices,
                                              portion, voxel)


def get_task_histogram(mode, masktype, samplelist):
    """Plot image histograms."""
    paths = dwi.paths.Paths(mode)
    if masktype == 'lesion':
        it = (dict(case=c, scan=s, lesion=l) for c, s, l in
              lesions(mode, samplelist))
    else:
        it = (dict(case=c, scan=s) for c, s in cases_scans(mode, samplelist))
    inpaths = [paths.roi(masktype, **x) for x in it]
    figpath = paths.histogram(masktype, samplelist)
    cmd = dwi.shell.histogram(inpaths, figpath, params=None)
    return {
        'name': taskname(mode, masktype, samplelist),
        'actions': folders(figpath) + [cmd],
        'file_dep': _files(inpaths),
        'targets': _files(figpath),
        }


def task_histogram():
    """Plot image histograms."""
    for mode, sl in product(MODES, SAMPLELISTS):
        for mt in ('image', 'prostate', 'lesion'):
            yield get_task_histogram(mode, mt, sl)


def get_task_grid(mode, c, s, ls, mt, lt=None, mbb=None, nanbg=False,
                  fmt='txt'):
    """Grid classifier."""
    paths = dwi.paths.Paths(mode)
    pmap = paths.pmap(case=c, scan=s)
    prostate = paths.mask('prostate', c, s)
    lesion = [paths.mask('lesion', c, s, lesion=x) for x in ls]
    tspec = TextureSpec('raw', 1, None)
    out, target = paths.grid(c, s, mt, tspec, fmt=fmt)
    if fmt == 'h5':
        target = out
    z = 5
    # z = 1
    d = dict(mbb=mbb, voxelsize=None, winsize=z, voxelspacing=(z, 1, 1),
             lesiontypes=lt, use_centroid=False, nanbg=nanbg)
    cmd = dwi.shell.grid(pmap, 0, prostate, lesion, out, **d)
    return {
        'name': taskname(mode, c, s, mt),
        'actions': folders(out) + [cmd],
        'file_dep': _files(pmap, prostate, *lesion),
        'targets': _files(target),
        }


def get_task_grid_texture(mode, c, s, ls, mt, tspec, lt=None, mbb=None,
                          nanbg=False, fmt='txt'):
    """Grid classifier."""
    paths = dwi.paths.Paths(mode)
    pmap = paths.texture(c, s, None, mt, 'all', 0, tspec, voxel='all')
    prostate = paths.mask('prostate', c, s)
    lesion = [paths.mask('lesion', c, s, lesion=x) for x in ls]
    out, target = paths.grid(c, s, mt, tspec, fmt=fmt)
    if fmt == 'h5':
        target = out
    z = 5
    # z = 1
    d = dict(mbb=mbb, voxelsize=None, winsize=z, voxelspacing=(z, 1, 1),
             lesiontypes=lt, use_centroid=False, nanbg=nanbg)
    cmd = dwi.shell.grid(pmap, None, prostate, lesion, out, **d)
    return {
        'name': taskname(mode, c, s, mt, tspec.method, tspec.winsize),
        'actions': folders(out) + [cmd],
        'file_dep': _files(pmap, prostate, *lesion),
        'targets': _files(target),
        }


def task_grid():
    """Grid classifier."""
    for mode, sl in product(MODES, SAMPLELISTS):
        paths = dwi.paths.Paths(mode)
        lesioninfo = defaultdict(list)
        for c, s, l in dwi.dataset.iterlesions(str(paths.samplelist(sl))):
            c, l, lt = c.num, l.index + 1, l.location
            lesioninfo[(c, s)].append((l, lt))
        for k, v in lesioninfo.items():
            c, s = k
            ls, lt = [x[0] for x in v], [x[1] for x in v]

            d = dict(lt=lt, mbb=None, fmt='h5')

            mt = 'prostate'
            d['nanbg'] = True
            yield get_task_grid(mode, c, s, ls, mt, **d)
            for tspec in texture_methods_winsizes(mode, mt):
                yield get_task_grid_texture(mode, c, s, ls, mt, tspec, **d)

            mt = 'all'
            d['nanbg'] = False
            yield get_task_grid(mode, c, s, ls, mt, **d)
            for tspec in texture_methods_winsizes(mode, mt):
                yield get_task_grid_texture(mode, c, s, ls, mt, tspec, **d)


# def task_check_mask_overlap():
#     """Check mask overlap."""
#     for mode, sl in product(MODES, SAMPLELISTS):
#         for c, s, l in lesions(mode, sl):
#             container = paths.mask('prostate', c, s)
#             other = paths.mask('lesion', c, s, lesion=l)
#             d = dict(m=mode, c=c, s=s, l=l)
#             fig = 'maskoverlap/{m[0]}/{c}-{s}-{l}.png'.format(**d)
#             cmd = dwi.shell.check_mask_overlap(container, other, fig)
#             yield {
#                 'name': taskname(mode, c, s, l),
#                 'actions': folders(fig) + [cmd],
#                 'file_dep': _files(container, other),
#                 'targets': _files(fig),
#                 }
