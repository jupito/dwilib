"""PyDoIt file for automating tasks."""

from __future__ import absolute_import, division, print_function
from itertools import chain, product
from os.path import dirname
import re

from doit import get_var
from doit.tools import check_timestamp_unchanged, create_folder

import dwi.files
from dwi.paths import (samplelist_path, pmap_path, subregion_path, mask_path,
                       roi_path, std_cfg_path, texture_path, texture_path_new)
import dwi.patient
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

DWILIB = '~/src/dwilib/tools'

MODE = dwi.patient.ImageMode(get_var('mode', 'DWI-Mono-ADCm'))
SAMPLELIST = get_var('samplelist', 'all')  # Sample list (train, test, etc)


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
        'sobel_mbb',
        'stats_all',
        ]


def texture_winsizes_old(masktype, mode):
    if masktype in ('CA', 'N'):
        seq = [3, 5]
    elif mode.modality in ('T2', 'T2w'):
        seq = xrange(3, 30, 4)
    else:
        seq = xrange(3, 16, 2)
    return ' '.join(str(x) for x in seq)


def texture_winsizes_new(masktype, mode, method):
    if method.endswith('_all'):
        return ['all']
    elif method.endswith('_mbb'):
        return ['mbb']
    elif masktype in ('CA', 'N'):
        return [3, 5]
    elif mode.modality in ('T2', 'T2w'):
        return xrange(3, 30, 4)
    else:
        return xrange(3, 16, 2)


def texture_methods_winsizes_new(mode, masktype):
    for method in texture_methods():
        for winsize in texture_winsizes_new(masktype, mode, method):
            yield method, winsize


def texture_params():
    lesion = ['lesion']
    slices = ['maxfirst', 'all']
    portion = [1, 0]
    return product(lesion, slices, portion)


def find_roi_param_combinations(mode):
    """Generate all find_roi.py parameter combinations."""
    find_roi_params = [
        [1, 2, 3],  # ROI depth min
        [1, 2, 3],  # ROI depth max
        xrange(2, 13),  # ROI side min (3 was not good)
        xrange(3, 13),  # ROI side max
        range(250, 2000, 250) + [50, 100, 150, 200],  # Number of ROIs
        ]
    if mode.modality == 'DWI':
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


def cases_scans(mode):
    """Generate all case, scan pairs."""
    samples = dwi.files.read_sample_list(samplelist_path(mode, SAMPLELIST))
    for sample in samples:
        case = sample['case']
        for scan in sample['scans']:
            yield case, scan


def lesions(mode):
    """Generate all case, scan, lesion# (1-based) combinations."""
    patients = dwi.files.read_patients_file(samplelist_path(mode, SAMPLELIST))
    for p in patients:
        for scan in p.scans:
            for lesion in range(len(p.lesions)):
                yield p.num, scan, lesion+1


def path_deps(*paths):
    """Return list of path dependencies, i.e. the file(s) itself or the
    directory contents.
    """
    # paths = [dwi.util.sglob(x) for x in paths]  # First make sure all exist.
    for i, path in enumerate(paths):
        if '*' in path or ('[' in path and ']' in path):
            paths[i] = dwi.util.sglob(path)
    paths = chain(f for p in paths for f in dwi.util.walker(p))
    paths = list(paths)
    return paths


#
# Commands.
#


def standardize_train_cmd(mode, cfgfile, samplelist='all'):
    """Standardize MRI images: training phase.

    Pay attention to the sample list: probably all data should be used.
    """
    d = dict(prg=DWILIB+'/standardize.py', m=mode, o=cfgfile,
             slf=samplelist_path(mode, samplelist), pd=pmap_path(mode))
    cmd = ('{prg} --patients {slf} --pmapdir {pd}'
           ' --param {m.param} --outconf {o}')
    return cmd.format(**d)


def standardize_transform_cmd(cfgpath, inpath, outpath):
    """Standardize MRI images: transform phase."""
    cmd = '{prg} --transform {c} {i} {o}'
    return cmd.format(prg=DWILIB+'/standardize.py', c=cfgpath, i=inpath,
                      o=outpath)


def get_texture_cmd_old(mode, case, scan, methods, winsizes, slices, portion,
                        mask, outpath, std_cfg):
    d = dict(prg=DWILIB+'/get_texture.py', m=mode, c=case, s=scan,
             slices=slices, portion=portion,
             methods=' '.join(methods), winsizes=winsizes,
             pd=pmap_path(mode), mask=mask, o=outpath)
    cmd = ('{prg} -v'
           ' --pmapdir {pd} --param {m.param}'
           ' --case {c} --scan {s} --mask {mask}'
           ' --methods {methods} --winsizes {winsizes}'
           ' --slices {slices} --portion {portion}'
           ' --output {o}')
    if std_cfg:
        cmd += ' --std {}'.format(std_cfg)
    return cmd.format(**d)


def get_texture_cmd_new(inpath, method, winsize, slices, portion, mask,
                        outpath, voxel, std_cfg=None):
    d = dict(prg=DWILIB+'/get_texture_new.py', i=inpath, mask=mask,
             slices=slices, portion=portion, mth=method, ws=winsize,
             o=outpath, vx=voxel)
    cmd = ('{prg} -v'
           ' --input {i} --mask {mask}'
           ' --slices {slices} --portion {portion}'
           ' --method {mth} --winspec {ws} --voxel {vx}'
           ' --output {o}')
    if std_cfg:
        cmd += ' --std {}'.format(std_cfg)
    cmd = cmd.format(**d)
    return cmd


def find_roi_cmd(mode, case, scan, algparams, outmask, outfig):
    d = dict(prg=DWILIB+'/find_roi.py', m=mode,
             slf=samplelist_path(mode, SAMPLELIST), pd=pmap_path(mode),
             srd=subregion_path(mode), c=case, s=scan, ap=' '.join(algparams),
             outmask=outmask, outfig=outfig)
    return ('{prg} --patients {slf} --pmapdir {pd} --subregiondir {srd} '
            '--param {m.param} --cases {c} --scans {s} --algparams {ap} '
            '--outmask {outmask} --outfig {outfig}'.format(**d))


def make_subregion_cmd(mask, subregion):
    cmd = '{prg} -i {mask} --pad 10 -s {sr}'
    return cmd.format(prg=DWILIB+'/masktool.py', mask=mask, sr=subregion)


def select_voxels_cmd(inpath, outpath, mask=None, source_attrs=False):
    cmd = '{prg} -i {i} -o {o}'
    if mask:
        cmd += ' -m {m}'
    if source_attrs:
        cmd += ' --source_attrs'
    return cmd.format(prg=DWILIB+'/select_voxels.py', i=inpath, o=outpath,
                      m=mask)


def auc_cmd(mode, threshold, algparams, outfile):
    d = dict(prg=DWILIB+'/roc_auc.py', m=mode,
             slf=samplelist_path(mode, SAMPLELIST), t=threshold,
             i=roi_path(mode, 'auto', algparams=algparams),
             ap_='_'.join(algparams), o=outfile)
    return (r'echo `{prg} --patients {slf} --threshold {t} --voxel mean'
            '--autoflip --pmapdir {i}` {ap_} >> {o}'.format(**d))


def correlation_cmd(mode, thresholds, algparams, outfile):
    d = dict(prg=DWILIB+'/correlation.py', m=mode,
             slf=samplelist_path(mode, SAMPLELIST), t=thresholds,
             i=roi_path(mode, 'auto', algparams=algparams),
             ap_='_'.join(algparams), o=outfile)
    return (r'echo `{prg} --patients {slf} --thresholds {t} --voxel mean'
            '--pmapdir {i}` {ap_} >> {o}'.format(**d))


def mask_out_cmd(src, dst, mask):
    d = dict(prg=DWILIB+'/mask_out_dicom.py', src=src, dst=dst, mask=mask)
    rm = 'rm -Rf {dst}'.format(**d)  # Remove destination image
    cp = 'cp -R --no-preserve=all {src} {dst}'.format(**d)  # Copy source
    mask = '{prg} --mask {mask} --image {dst}'.format(**d)  # Mask image
    return [rm, cp, mask]


#
# Tasks.
#


def task_convert_data():
    """Convert DICOM data to HDF5."""
    mode = MODE
    paths = dwi.util.iglob('dicoms_{m.model}_*/*'.format(m=mode), typ='dir')
    for path in paths:
        p = re.compile(r"""
           dicoms_(?P<model>[a-z0-9]+)_.+
           /
           (?P<case>\d+) _
           (?P<name>[a-z_]+) _
           (?P<scan>[0-9]+[a-z]+) _?
           (?P<remark>\w+)?
           """, flags=re.VERBOSE | re.IGNORECASE)
        m = re.match(p, path)
        if m:
            case = int(m.group('case'))
            scan = m.group('scan').lower()
            outpath = pmap_path(mode, case, scan, new=True)
            cmd = select_voxels_cmd(path, outpath, source_attrs=True)
            yield {
                'name': name(mode, case, scan),
                'actions': folders(outpath) + [cmd],
                'targets': [outpath],
                'uptodate': [check_timestamp_unchanged(path)],
            }


def task_standardize_train():
    """Standardize MRI images: training phase."""
    for mode in [MODE]:
        std_cfg = std_cfg_path(mode)
        yield {
            'name': name(mode),
            'actions': [standardize_train_cmd(mode, std_cfg)],
            'targets': [std_cfg],
            'uptodate': [check_timestamp_unchanged(pmap_path(mode))],
            'clean': True,
            }


def task_standardize_transform():
    """Standardize MRI images: transform phase."""
    for mode in [MODE]:
        if not mode.standardize:
            continue
        cfgpath = std_cfg_path(mode)
        for case, scan in cases_scans(mode):
            inpath = pmap_path(mode, case, scan)
            outpath = pmap_path(str(mode) + '-std', case, scan, new=True)
            cmd = standardize_transform_cmd(cfgpath, inpath, outpath)
            yield {
                'name': name(case, scan),
                'actions': folders(outpath) + [cmd],
                'file_dep': path_deps(cfgpath, inpath),
                'targets': [outpath],
                'clean': True,
            }


def task_make_subregion():
    """Make minimum bounding box + 10 voxel subregions from prostate masks."""
    for case, scan in cases_scans(MODE):
        mask = mask_path(MODE, 'prostate', case, scan)
        subregion = subregion_path(MODE, case, scan)
        cmd = make_subregion_cmd(mask, subregion)
        yield {
            'name': name(case, scan),
            'actions': folders(subregion) + [cmd],
            'file_dep': path_deps(mask),
            'targets': [subregion],
            'clean': True,
            }


def get_task_find_roi(mode, case, scan, algparams):
    d = dict(m=mode, c=case, s=scan, ap_='_'.join(algparams))
    outmask = mask_path(mode, 'auto', case, scan, algparams=algparams)
    outfig = 'find_roi_images_{m}/{ap_}/{c}_{s}.png'.format(**d)
    subregion = subregion_path(mode, case, scan)
    mask_p = mask_path(mode, 'prostate', case, scan)
    mask_c = mask_path(mode, 'CA', case, scan)
    mask_n = mask_path(mode, 'N', case, scan)
    cmd = find_roi_cmd(mode, case, scan, algparams, outmask, outfig)
    return {
        'name': '{m}_{ap_}_{c}_{s}'.format(**d),
        'actions': folders(outmask, outfig) + [cmd],
        'file_dep': path_deps(subregion, mask_p, mask_c, mask_n),
        'targets': [outmask, outfig],
        'clean': True,
        }


def task_find_roi():
    """Find a cancer ROI automatically."""
    for algparams in find_roi_param_combinations(MODE):
        for case, scan in cases_scans(MODE):
            yield get_task_find_roi(MODE, case, scan, algparams)


def get_task_select_roi_lesion(mode, case, scan, lesion):
    """Select ROIs from the pmap DICOMs based on masks."""
    masktype = 'lesion'
    mask = mask_path(mode, 'lesion', case, scan, lesion=lesion)
    roi = roi_path(str(mode) + '-std', masktype, case, scan, lesion=lesion)
    pmap = pmap_path(str(mode) + '-std', case, scan, new=True)
    cmd = select_voxels_cmd(pmap, roi, mask=mask)
    return {
        'name': name(mode, masktype, case, scan, lesion),
        'actions': folders(roi) + [cmd],
        'file_dep': path_deps(mask),
        'targets': [roi],
        'clean': True,
        }


def get_task_select_roi_manual(mode, case, scan, masktype):
    """Select ROIs from the pmap DICOMs based on masks."""
    mask = mask_path(mode, masktype, case, scan)
    roi = roi_path(mode, masktype, case, scan)
    pmap = pmap_path(mode, case, scan)
    cmd = select_voxels_cmd(pmap, roi, mask=mask)
    return {
        'name': name(mode, masktype, case, scan),
        'actions': folders(roi) + [cmd],
        'file_dep': path_deps(mask),
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
    cmd = select_voxels_cmd(pmap, roi, mask=mask)
    return {
        'name': name(mode, ap_, case, scan),
        'actions': folders(roi) + [cmd],
        'file_dep': [mask],
        'targets': [roi],
        'uptodate': [check_timestamp_unchanged(pmap)],
        'clean': True,
        }


def task_select_roi_lesion():
    """Select lesion ROIs from the pmap DICOMs."""
    for c, s, l in lesions(MODE):
        yield get_task_select_roi_lesion(MODE, c, s, l)


def task_select_roi_manual():
    """Select cancer ROIs from the pmap DICOMs."""
    for masktype in ('CA', 'N'):
        for case, scan in cases_scans(MODE):
            try:
                yield get_task_select_roi_manual(MODE, case, scan, masktype)
            except IOError as e:
                print('select_roi_manual', e)


def task_select_roi_auto():
    """Select automatic ROIs from the pmap DICOMs."""
    for algparams in find_roi_param_combinations(MODE):
        for case, scan in cases_scans(MODE):
            try:
                yield get_task_select_roi_auto(MODE, case, scan, algparams)
            except IOError as e:
                print('select_roi_auto', e)


def task_select_roi():
    """Select all ROIs task group."""
    return {
        'actions': None,
        'task_dep': ['select_roi_manual', 'select_roi_auto'],
        }


def get_task_autoroi_auc(mode, threshold):
    """Evaluate auto-ROI prediction ability by ROC AUC with Gleason score."""
    d = dict(m=mode, sl=SAMPLELIST, t=threshold)
    outfile = 'autoroi_auc_{t}_{m}_{sl}.txt'.format(**d)
    cmds = ['echo -n > {o}'.format(o=outfile)]
    for algparams in find_roi_param_combinations(mode):
        cmds.append(auc_cmd(mode, threshold, algparams, outfile))
    return {
        'name': 'autoroi_auc_{sl}_{m}_{t}'.format(**d),
        'actions': cmds,
        'task_dep': ['select_roi_auto'],
        'targets': [outfile],
        'clean': True,
        }


def get_task_autoroi_correlation(mode, thresholds):
    """Evaluate auto-ROI prediction ability by correlation with Gleason
    score."""
    d = dict(m=mode, sl=SAMPLELIST, t_=thresholds.replace(' ', ','))
    outfile = 'autoroi_correlation_{t_}_{m}_{sl}.txt'.format(**d)
    cmds = ['echo -n > {o}'.format(o=outfile)]
    for algparams in find_roi_param_combinations(mode):
        cmds.append(correlation_cmd(mode, thresholds, algparams, outfile))
    return {
        'name': 'autoroi_correlation_{sl}_{m}_{t_}'.format(**d),
        'actions': cmds,
        'task_dep': ['select_roi_auto'],
        'targets': [outfile],
        'clean': True,
        }


def task_evaluate_autoroi():
    """Evaluate auto-ROI prediction ability."""
    yield get_task_autoroi_auc(MODE, '3+3')
    yield get_task_autoroi_auc(MODE, '3+4')
    yield get_task_autoroi_correlation(MODE, '3+3 3+4')
    yield get_task_autoroi_correlation(MODE, '')


def get_task_texture_manual_old(mode, masktype, case, scan, lesion, slices,
                                portion):
    """Generate texture features."""
    methods = texture_methods()
    winsizes = texture_winsizes_old(masktype, mode)
    mask = mask_path(mode, masktype, case, scan, lesion=lesion)
    outfile = texture_path(mode, case, scan, lesion, masktype, slices,
                           portion)
    std_cfg = None
    deps = path_deps(mask)
    if mode.standardize:
        std_cfg = std_cfg_path(mode)
        deps.append(std_cfg)
    cmd = get_texture_cmd_old(mode, case, scan, methods, winsizes, slices,
                              portion, mask, outfile, std_cfg)
    return {
        'name': name(mode, masktype, slices, portion, case, scan, lesion),
        'actions': folders(outfile) + [cmd],
        'file_dep': deps,
        'targets': [outfile],
        'clean': True,
        }


def get_task_texture_manual_new(mode, masktype, case, scan, lesion, slices,
                                portion, method, winsize, voxel):
    """Generate texture features."""
    inpath = pmap_path(mode, case, scan)
    mask = mask_path(mode, masktype, case, scan, lesion=lesion)
    outfile = texture_path_new(mode, case, scan, lesion, masktype, slices,
                               portion, method, winsize, voxel=voxel)
    # std_cfg = None
    deps = path_deps(mask)
    if mode.standardize:
        # std_cfg = std_cfg_path(mode)
        # deps.append(std_cfg)
        inpath = pmap_path(str(mode) + '-std', case, scan, new=True)
        deps.append(inpath)
    cmd = get_texture_cmd_new(inpath, method, winsize, slices, portion, mask,
                              outfile, voxel, std_cfg=None)
    return {
        'name': name(mode, masktype, slices, portion, case, scan, lesion,
                     method, winsize, voxel),
        'actions': folders(outfile) + [cmd],
        'file_dep': deps,
        'targets': [outfile],
        'clean': True,
        }


def get_task_texture_auto_old(mode, algparams, case, scan, lesion, slices,
                              portion):
    """Generate texture features."""
    masktype = 'auto'
    methods = texture_methods()
    winsizes = texture_winsizes_old(masktype, mode)
    mask = mask_path(mode, masktype, case, scan, lesion=lesion,
                     algparams=algparams)
    outfile = texture_path(mode, case, scan, lesion, masktype, slices,
                           portion, algparams)
    std_cfg = None
    deps = [mask]
    if mode.standardize:
        std_cfg = std_cfg_path(mode)
        deps.append(std_cfg)
    cmd = get_texture_cmd_old(mode, case, scan, methods, winsizes, slices,
                              portion, mask, outfile, std_cfg)
    return {
        'name': name(mode, '_'.join(algparams), slices, portion, case, scan,
                     lesion),
        'actions': folders(outfile) + [cmd],
        'file_dep': deps,
        'targets': [outfile],
        'clean': True,
        }


def task_texture_old():
    """Generate texture features."""
    for c, s, l in lesions(MODE):
        yield get_task_texture_manual_old(MODE, 'lesion', c, s, l, 'maxfirst',
                                          0)
        yield get_task_texture_manual_old(MODE, 'lesion', c, s, l, 'maxfirst',
                                          1)
        yield get_task_texture_manual_old(MODE, 'lesion', c, s, l, 'all', 0)
        yield get_task_texture_manual_old(MODE, 'lesion', c, s, l, 'all', 1)
        if MODE.modality in ('T2', 'T2w'):
            continue  # Do only lesion for those.
        yield get_task_texture_manual_old(MODE, 'CA', c, s, l, 'maxfirst', 1)
        yield get_task_texture_manual_old(MODE, 'N', c, s, l, 'maxfirst', 1)
        for ap in find_roi_param_combinations(MODE):
            yield get_task_texture_auto_old(MODE, ap, c, s, l, 'maxfirst', 1)


def task_texture_new():
    """Generate texture features."""
    mode = MODE
    for mt, slices, portion in texture_params():
        for case, scan, lesion in lesions(mode):
            for mth, ws in texture_methods_winsizes_new(mode, 'lesion'):
                yield get_task_texture_manual_new(mode, mt, case, scan,
                                                  lesion, slices, portion,
                                                  mth, ws, 'mean')
                yield get_task_texture_manual_new(mode, mt, case, scan,
                                                  lesion, slices, portion,
                                                  mth, ws, 'all')
    # FIXME: Clean the horrible kludge.
    for c, s in cases_scans(mode):
        for mth, ws in texture_methods_winsizes_new(mode, 'prostate'):
            yield get_task_texture_manual_new(mode, 'prostate', c, s, None,
                                              'maxfirst', 0, mth, ws, 'all')
            yield get_task_texture_manual_new(mode, 'prostate', c, s, None,
                                              'all', 0, mth, ws, 'all')


def task_merge_textures():
    """Merge texture methods into singe file per case/scan/lesion."""
    mode = MODE
    for mt, slices, portion in texture_params():
        for case, scan, lesion in lesions(mode):
            infiles = [texture_path_new(mode, case, scan, lesion, mt, slices,
                                        portion, mth, ws) for mth, ws in
                       texture_methods_winsizes_new(mode, mt)]
            outfile = texture_path_new(mode, case, scan, lesion, mt, slices,
                                       portion, 'merged', 'merged')
            outfile = 'merged_' + outfile
            cmd = select_voxels_cmd(' '.join(infiles), outfile)
            yield {
                'name': name(mode, case, scan, lesion, mt, slices, portion),
                'actions': folders(outfile) + [cmd],
                'file_dep': infiles,
                'targets': [outfile],
            }


def get_task_mask_prostate(mode, case, scan, imagetype, postfix,
                           param='DICOM'):
    """Generate DICOM images with everything but prostate zeroed."""
    imagedir = 'dicoms_{}'.format(mode.modality)
    outdir = 'dicoms_masked_{}'.format(mode.modality)
    d = dict(c=case, s=scan, id=imagedir, od=outdir, it=imagetype,
             pox=postfix, p=param)
    mask = mask_path(mode, 'prostate', case, scan)
    src = dwi.util.sglob('{id}_*/{c}_*{it}_{s}{pox}/{p}'.format(**d))
    dst = '{od}/{c}{it}_{s}'.format(**d)
    cmds = mask_out_cmd(src, dst, mask)
    return {
        'name': name(case, scan),
        'actions': folders(dst) + cmds,
        # 'file_dep':  # TODO
        # 'targets':  # TODO
        }


def task_mask_prostate_DWI():
    """Generate DICOM images with everything but prostate zeroed."""
    for case, scan in cases_scans(MODE):
        try:
            mode = dwi.patient.ImageMode('DWI-SI-raw')
            yield get_task_mask_prostate(mode, case, scan, '_hB', '')
            # mode = dwi.patient.ImageMode('SPAIR-SPAIR-raw')
            # yield get_task_mask_prostate(mode, case, scan, '', '_all')
        except IOError as e:
            print('mask_prostate_DWI', e)


def task_mask_prostate_T2():
    """Generate DICOM images with everything but prostate zeroed."""
    for case, scan in cases_scans(MODE):
        try:
            mode = dwi.patient.ImageMode('T2-T2-raw')
            yield get_task_mask_prostate(mode, case, scan, '', '*')
            # mode = dwi.patient.ImageMode('T2f-T2f-raw')
            # yield get_task_mask_prostate(mode, case, scan, '', '*', '*_Rho')
            # mode = dwi.patient.ImageMode('T2w-T2w-raw')
            # yield get_task_mask_prostate(mode, case, scan, '', '*')
        except IOError as e:
            print('mask_prostate_T2', e)


def task_all():
    """Do all essential things."""
    return {
        'actions': None,
        'task_dep': ['select_roi', 'evaluate_autoroi'],
        }
