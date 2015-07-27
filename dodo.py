"""PyDoIt file for automating tasks."""

from __future__ import absolute_import, division, print_function
from glob import glob
from itertools import chain, product
import os
from os.path import dirname, isdir, join

from doit import get_var
from doit.tools import check_timestamp_unchanged, create_folder

import dwi.files
import dwi.patient
import dwi.util

#Backends:
#dbm: (default) It uses python dbm module.
#json: Plain text using a json structure, it is slow but good for debugging.
#sqlite3: (experimental) very slow implementation, support concurrent access.

DOIT_CONFIG = {
    'backend': 'sqlite3',
    'default_tasks': [],
    'verbosity': 1,
    'num_process': 7,
    }

DWILIB = '~/src/dwilib/tools'

MODE = dwi.patient.ImageMode(*get_var('mode', 'DWI-Mono-ADCm').split('-'))
SAMPLELIST = get_var('samplelist', 'all')  # Sample list (train, test, etc)

FIND_ROI_PARAMS = [
    [1, 2, 3],  # ROI depth min
    [1, 2, 3],  # ROI depth max
    range(2, 13),  # ROI side min (3 was not good)
    range(3, 13),  # ROI side max
    range(250, 2000, 250) + [50, 100, 150, 200],  # Number of ROIs
    ]

def texture_methods(mode):
    return [
        #'stats',
        #'haralick',
        #'moment',
        #'haralick_mbb',

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

def texture_winsizes(masktype, mode):
    if masktype in ('CA', 'N'):
        l = [3, 5]
    elif mode.modality in ('T2', 'T2w'):
        l = range(3, 30, 4)
    else:
        l = range(3, 16, 2)
    return ' '.join(str(x) for x in l)

#def texture_winsizes_new(masktype, mode, method):
#    if method.endswith('_all') or method.endswith('_mbb'):
#        l = [0]
#    elif masktype in ('CA', 'N'):
#        l = [3, 5]
#    elif mode.modality in ('T2', 'T2w'):
#        l = range(3, 30, 4)
#    else:
#        l = range(3, 16, 2)
#    return ' '.join(str(x) for x in l)

def find_roi_param_combinations(mode):
    """Generate all find_roi.py parameter combinations."""
    if mode.modality == 'DWI':
        if SAMPLELIST == 'test':
            params = [
                (2,3,10,10,500),  # Mono: corr, auc
                (2,3,10,10,1750),  # Mono: corr
                (2,3,11,11,750),  # Mono: corr
                #(2,3,2,2,250),  # Kurt: auc
                #(2,3,9,9,1000),  # Kurt: corr
                #(2,3,12,12,1750),  # Kurt: corr
                #(2,3,5,5,500),  # Kurt K: corr, auc
                ]
        else:
            params = product(*FIND_ROI_PARAMS)
        for t in params:
            if t[0] <= t[1] and t[2] == t[3]:
                yield [str(x) for x in t]

def samplelist_file(mode, samplelist=SAMPLELIST):
    return 'patients_{m.modality}_{l}.txt'.format(m=mode, l=samplelist)

def pmapdir_dicom(mode):
    return dwi.util.sglob('dicoms_{m.model}_*'.format(m=mode), typ='dir')

def pmap_dicom(mode, case, scan):
    pd = pmapdir_dicom(mode)
    if mode.param == 'raw':
        # There's no actual parameter, only single 'raw' value (used for T2).
        s = '{pd}/{c}_*_{s}*'
    else:
        s = '{pd}/{c}_*_{s}/{c}_*_{s}*_{m.param}'
    path = s.format(pd=pd, m=mode, c=case, s=scan)
    return dwi.util.sglob(path, typ='dir')

def subregion_dir(mode):
    return 'subregions'

def subregion_path(mode, case, scan):
    return '{srd}/{c}_{s}_subregion10.txt'.format(srd=subregion_dir(mode),
                                                  c=case, s=scan)

def mask_path(mode, masktype, case, scan, lesion=None, algparams=[]):
    """Return path and deps of masks of different types."""
    d = dict(m=mode, mt=masktype, c=case, s=scan, l=lesion,
             ap_='_'.join(algparams))
    do_glob = True
    if masktype == 'prostate':
        path = 'masks_prostate_{m.modality}/{c}_*_{s}*'
    elif masktype == 'lesion':
        if mode.model in ('T2', 'T2w'):
            path = 'masks_lesion_{m.model}/PCa_masks_{m.model}_{l}*/{c}_*{s}_*'
        else:
            path = 'masks_lesion_DWI/PCa_masks_DWI_{l}*/{c}_*{s}_*'
    elif masktype in ('CA', 'N'):
        path = 'masks_rois/{c}_*_{s}_D_{mt}'
    elif masktype == 'auto':
        # Don't require existence, can be generated.
        do_glob = False
        path = 'masks_auto_{m.model}_{m.param}/{ap_}/{c}_{s}_auto.mask'
    else:
        raise Exception('Unknown mask type: {mt}'.format(**d))
    path = path.format(**d)
    if do_glob:
        path = dwi.util.sglob(path)
    return path

def roi_path(mode, masktype, case=None, scan=None, algparams=[]):
    """Return whole ROI path or part of it."""
    d = dict(m=mode, mt=masktype, c=case, s=scan, ap_='_'.join(algparams))
    components = ['rois_{mt}_{m.model}_{m.param}']
    if algparams:
        components.append('{ap_}')
    if case is not None and scan is not None:
        components.append('{c}_x_x_{s}_{m.model}_{m.param}_{mt}.txt')
    components = [x.format(**d) for x in components]
    return join(*components)

def texture_path(mode, case, scan, lesion, masktype, slices, portion,
                 algparams=()):
    """Return path to texture file."""
    if masktype in ('lesion', 'CA', 'N'):
        path = 'texture_{mt}_{m.model}_{m.param}_{slices}_{portion}/{c}_{s}_{l}.txt'
    elif masktype == 'auto':
        path = 'texture_{mt}_{m.model}_{m.param}_{slices}_{portion}/{ap_}/{c}_{s}_{l}.txt'
    else:
        raise Exception('Unknown mask type: {mt}'.format(mt=masktype))
    return path.format(m=mode, c=case, s=scan, l=lesion, mt=masktype,
                       slices=slices, portion=portion, ap_='_'.join(algparams))

#def texture_path_new(mode, case, scan, lesion, masktype, slices, portion,
#                     method, winsize, algparams=()):
#    """Return path to texture file."""
#    if masktype in ('lesion', 'CA', 'N'):
#        path = 'texture_{mt}_{m.model}_{m.param}_{slices}_{portion}/{c}_{s}_{l}_{mth}_{ws}.txt'
#    elif masktype == 'auto':
#        path = 'texture_{mt}_{m.model}_{m.param}_{slices}_{portion}/{ap_}/{c}_{s}_{l}_{mth}_{ws}.txt'
#    else:
#        raise Exception('Unknown mask type: {mt}'.format(**d))
#    return path.format(m=mode, c=case, s=scan, l=lesion, mt=masktype,
#                       slices=slices, portion=portion, mth=method, ws=winsize,
#                       ap_='_'.join(algparams))

def cases_scans(mode):
    """Generate all case, scan pairs."""
    samples = dwi.files.read_sample_list(samplelist_file(mode))
    for sample in samples:
        case = sample['case']
        for scan in sample['scans']:
            yield case, scan

def lesions(mode):
    """Generate all case, scan, lesion# (1-based) combinations."""
    patients = dwi.files.read_patients_file(samplelist_file(mode))
    for p in patients:
        for scan in p.scans:
            for lesion in range(len(p.lesions)):
                yield p.num, scan, lesion+1

def path_deps(*paths):
    """Return list of path dependencies, i.e. the file(s) itself or the
    directory contents.
    """
    # First make sure all exist.
    #paths = [dwi.util.sglob(x) for x in paths]
    for i, path in enumerate(paths):
        if '*' in path or ('[' in path and ']' in path):
            paths[i] = dwi.util.sglob(path)
    paths = chain(f for p in paths for f in dwi.util.walker(p))
    paths = list(paths)
    return paths

def get_texture_cmd(mode, case, scan, methods, winsizes, slices, portion,
                    mask, outpath):
    GET_TEXTURE = DWILIB+'/get_texture.py'
    d = dict(prg=GET_TEXTURE, m=mode, c=case, s=scan,
             slices=slices, portion=portion,
             methods=' '.join(methods), winsizes=winsizes,
             pd=pmapdir_dicom(mode), mask=mask, o=outpath)
    cmd = ('{prg} -v'
           ' --pmapdir {pd} --param {m.param} --case {c} --scan {s} --mask {mask}'
           ' --methods {methods} --winsizes {winsizes}'
           ' --slices {slices} --portion {portion}'
           ' --output {o}')
    if mode.model == 'T2w':
        cmd += ' --std stdcfg_{m.model}.txt'
    return cmd.format(**d)

#def get_texture_cmd_new(mode, case, scan, method, winsize, slices, portion,
#                        mask, outpath):
#    GET_TEXTURE_NEW = DWILIB+'/get_texture_new.py'
#    d = dict(m=mode, c=case, s=scan,
#             slices=slices, portion=portion, mth=method, ws=winsize,
#             pd=pmapdir_dicom(mode), mask=mask, o=outpath)
#    cmd = ('{prg} -v'
#           ' --pmapdir {pd} --param {m.param} --case {c} --scan {s} --mask {mask}'
#           ' --slices {slices} --portion {portion}'
#           ' --method {mth} --winsize {ws} --voxel mean'
#           ' --output {o}')
#    if mode.model == 'T2w':
#        cmd += ' --std stdcfg_{m.model}.txt'
#    cmd = cmd.format(prg=GET_TEXTURE_NEW, **d)
#    return cmd

#def task_anonymize():
#    """Anonymize imaging data."""
#    ANON = DWILIB+'/anonymize_dicom.py'
#    files = glob('dicoms/*/DICOMDIR') + glob('dicoms/*/DICOM/*')
#    files.sort()
#    for f in files:
#        cmd = '{prg} -v -i {f}'.format(prg=ANON, f=f)
#        yield {
#           'name': f,
#           'actions': [cmd],
#           #'file_dep': [f],
#           }

#def fit_cmd(model, subwindow, infiles, outfile):
#    PMAP = DWILIB+'/pmap.py'
#    d = dict(prg=PMAP, m=model, sw=' '.join(str(x) for x in subwindow),
#             i=' '.join(infiles), o=outfile)
#    s = '{prg} -m {m} -s {sw} -d {i} -o {o}'.format(**d)
#    return s
#
#def task_fit():
#    """Fit models to imaging data."""
#    MODELS = ('Si SiN Mono MonoN Kurt KurtN Stretched StretchedN '
#              'Biexp BiexpN'.split())
#    SUBWINDOWS = dwi.files.read_subwindows('subwindows.txt')
#    for case, scan in SUBWINDOWS.keys():
#        subwindow = SUBWINDOWS[(case, scan)]
#        d = dict(c=case, s=scan)
#        infiles = sorted(glob('dicoms/{c}_*_hB_{s}*/DICOM/*'.format(**d)))
#        if len(infiles) == 0:
#            continue
#        for model in MODELS:
#            d['m'] = model
#            outfile = 'pmaps/pmap_{c}_{s}_{m}.txt'.format(**d)
#            cmd = fit_cmd(model, subwindow, infiles, outfile)
#            yield {
#               'name': '{c}_{s}_{m}'.format(**d),
#               'actions': [(create_folder, [dirname(outfile)]),
#                           cmd],
#               'file_dep': infiles,
#               'targets': [outfile],
#               'clean': True,
#               }

def task_make_subregion():
    """Make minimum bounding box + 10 voxel subregions from prostate masks."""
    MASKTOOL = DWILIB+'/masktool.py'
    for case, scan in cases_scans(MODE):
        mask = mask_path(MODE, 'prostate', case, scan)
        subregion = subregion_path(MODE, case, scan)
        cmd = '{prg} -i {msk} --pad 10 -s {sr}'.format(prg=MASKTOOL, msk=mask,
                                                       sr=subregion)
        yield {
            'name': '{c}_{s}'.format(c=case, s=scan),
            'actions': [(create_folder, [dirname(subregion)]),
                        cmd],
            'file_dep': path_deps(mask),
            'targets': [subregion],
            'clean': True,
            }

def get_task_find_roi(mode, case, scan, algparams):
    FIND_ROI = DWILIB+'/find_roi.py'
    d = dict(prg=FIND_ROI, m=mode, slf=samplelist_file(mode),
             pd=pmapdir_dicom(mode), srd=subregion_dir(mode),
             c=case, s=scan, ap=' '.join(algparams), ap_='_'.join(algparams))
    mask = mask_path(mode, 'auto', case, scan, algparams=algparams)
    fig = 'find_roi_images_{m.model}_{m.param}/{ap_}/{c}_{s}.png'.format(**d)
    d.update(mask=mask, fig=fig)
    subregion = subregion_path(mode, case, scan)
    mask_p = mask_path(mode, 'prostate', case, scan)
    mask_c = mask_path(mode, 'CA', case, scan)
    mask_n = mask_path(mode, 'N', case, scan)
    cmd = ('{prg} --patients {slf} --pmapdir {pd} --subregiondir {srd} '
           '--param {m.param} --cases {c} --scans {s} --algparams {ap} '
           '--outmask {mask} --outfig {fig}'.format(**d))
    return {
        'name': '{m.model}_{m.param}_{ap_}_{c}_{s}'.format(**d),
        'actions': [(create_folder, [dirname(mask)]),
                    (create_folder, [dirname(fig)]),
                    cmd],
        'file_dep': path_deps(subregion, mask_p, mask_c, mask_n),
        'targets': [mask, fig],
        'clean': True,
        }

def task_find_roi():
    """Find a cancer ROI automatically."""
    for algparams in find_roi_param_combinations(MODE):
        for case, scan in cases_scans(MODE):
            yield get_task_find_roi(MODE, case, scan, algparams)

def select_voxels_cmd(mask, inpath, outpath):
    SELECT_VOXELS = DWILIB+'/select_voxels.py'
    return '{prg} -m {m} -i "{i}" -o "{o}"'.format(prg=SELECT_VOXELS,
        m=mask, i=inpath, o=outpath)

def get_task_select_roi_manual(mode, case, scan, masktype):
    """Select ROIs from the pmap DICOMs based on masks."""
    d = dict(m=mode, c=case, s=scan, mt=masktype)
    mask = mask_path(mode, masktype, case, scan)
    roi = roi_path(mode, masktype, case, scan)
    pmap = pmap_dicom(mode, case, scan)
    cmd = select_voxels_cmd(mask, pmap, roi)
    return {
        'name': '{m.model}_{m.param}_{mt}_{c}_{s}'.format(**d),
        'actions': [(create_folder, [dirname(roi)]),
                    cmd],
        'file_dep': path_deps(mask),
        'targets': [roi],
        'uptodate': [check_timestamp_unchanged(pmap)],
        'clean': True,
        }

def get_task_select_roi_auto(mode, case, scan, algparams):
    """Select ROIs from the pmap DICOMs based on masks."""
    d = dict(m=mode, c=case, s=scan, mt='auto', ap_='_'.join(algparams))
    mask = mask_path(mode, 'auto', case, scan, algparams=algparams)
    roi = roi_path(mode, 'auto', case, scan, algparams=algparams)
    pmap = pmap_dicom(mode, case, scan)
    cmd = select_voxels_cmd(mask, pmap, roi)
    return {
        'name': '{m.model}_{m.param}_{ap_}_{c}_{s}'.format(**d),
        'actions': [(create_folder, [dirname(roi)]),
                    cmd],
        'file_dep': path_deps(mask),
        'targets': [roi],
        'uptodate': [check_timestamp_unchanged(pmap)],
        'clean': True,
        }

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
    CALC_AUC = DWILIB+'/roc_auc.py'
    d = dict(m=mode, sl=SAMPLELIST, slf=samplelist_file(mode), prg=CALC_AUC,
             t=threshold)
    d['o'] = 'autoroi_auc_{t}_{m.model}_{m.param}_{sl}.txt'.format(**d)
    cmds = ['echo -n > {o}'.format(**d)]
    for algparams in find_roi_param_combinations(MODE):
        d['ap_'] = '_'.join(algparams)
        d['i'] = roi_path(mode, 'auto', algparams=algparams)
        s = r'echo `{prg} --patients {slf} --threshold {t} --voxel mean --autoflip --pmapdir {i}` {ap_} >> {o}'
        cmds.append(s.format(**d))
    return {
        'name': 'autoroi_auc_{sl}_{m.model}_{m.param}_{t}'.format(**d),
        'actions': cmds,
        'task_dep': ['select_roi_auto'],
        'targets': [d['o']],
        'clean': True,
        }

def get_task_autoroi_correlation(mode, thresholds):
    """Evaluate auto-ROI prediction ability by correlation with Gleason
    score."""
    CORRELATION = DWILIB+'/correlation.py'
    d = dict(sl=SAMPLELIST, slf=samplelist_file(mode), prg=CORRELATION,
             m=mode, t=thresholds, t_=thresholds.replace(' ', ','))
    d['o'] = 'autoroi_correlation_{t_}_{m.model}_{m.param}_{sl}.txt'.format(**d)
    cmds = ['echo -n > {o}'.format(**d)]
    for algparams in find_roi_param_combinations(MODE):
        d['ap_'] = '_'.join(algparams)
        d['i'] = roi_path(mode, 'auto', algparams=algparams)
        s = r'echo `{prg} --patients {slf} --thresholds {t} --voxel mean --pmapdir {i}` {ap_} >> {o}'
        cmds.append(s.format(**d))
    return {
        'name': 'autoroi_correlation_{sl}_{m.model}_{m.param}_{t_}'.format(**d),
        'actions': cmds,
        'task_dep': ['select_roi_auto'],
        'targets': [d['o']],
        'clean': True,
        }

def task_evaluate_autoroi():
    """Evaluate auto-ROI prediction ability."""
    yield get_task_autoroi_auc(MODE, '3+3')
    yield get_task_autoroi_auc(MODE, '3+4')
    yield get_task_autoroi_correlation(MODE, '3+3 3+4')
    yield get_task_autoroi_correlation(MODE, '')

def get_task_texture_manual(mode, masktype, case, scan, lesion, slices,
                            portion):
    """Generate texture features."""
    d = dict(m=mode, mt=masktype, c=case, s=scan, l=lesion,
             slices=slices, portion=portion)
    methods = texture_methods(mode)
    winsizes = texture_winsizes(masktype, mode)
    mask = mask_path(mode, masktype, case, scan, lesion=lesion)
    outfile = texture_path(mode, case, scan, lesion, masktype, slices, portion)
    cmd = get_texture_cmd(mode, case, scan, methods, winsizes, slices, portion,
                          mask, outfile)
    return {
        'name': '{m.model}_{m.param}_{mt}_{slices}_{portion}_{c}_{s}_{l}'.format(**d),
        'actions': [(create_folder, [dirname(outfile)]),
                    cmd],
        'file_dep': path_deps(mask),
        'targets': [outfile],
        'clean': True,
        }

#def get_task_texture_manual_new(mode, masktype, case, scan, lesion, slices,
#        portion, method, winsize):
#    """Generate texture features."""
#    d = dict(m=mode, mt=masktype, c=case, s=scan, l=lesion,
#             slices=slices, portion=portion, mth=method, ws=winsize)
#    mask = mask_path(mode, masktype, case, scan, lesion=lesion)
#    outfile = texture_path_new(mode, case, scan, lesion, masktype, slices,
#                              portion, method, winsize)
#    cmd = get_texture_cmd(mode, case, scan, method, winsize, slices, portion,
#                          mask, outfile)
#    return {
#        'name': '{m.model}_{m.param}_{mt}_{slices}_{portion}_{c}_{s}_{l}_{mth}_{ws}'.format(**d),
#        'actions': [(create_folder, [dirname(outfile)]),
#                    cmd],
#        'file_dep': path_deps(mask),
#        'targets': [outfile],
#        'clean': True,
#        }

def get_task_texture_auto(mode, algparams, case, scan, lesion, slices, portion):
    """Generate texture features."""
    masktype = 'auto'
    d = dict(m=mode, mt=masktype, c=case, s=scan, l=lesion,
             ap_='_'.join(algparams), slices=slices, portion=portion)
    methods = texture_methods(mode)
    winsizes = texture_winsizes(masktype, mode)
    mask = mask_path(mode, masktype, case, scan, lesion=lesion, algparams=algparams)
    outfile = texture_path(mode, case, scan, lesion, masktype, slices, portion,
                           algparams)
    cmd = get_texture_cmd(mode, case, scan, methods, winsizes, slices, portion,
                          mask, outfile)
    return {
        'name': '{m.model}_{m.param}_{ap_}_{slices}_{portion}_{c}_{s}_{l}'.format(**d),
        'actions': [(create_folder, [dirname(outfile)]),
                    cmd],
        'file_dep': path_deps(mask),
        'targets': [outfile],
        'clean': True,
        }

def task_texture():
    """Generate texture features."""
    for case, scan, lesion in lesions(MODE):
        yield get_task_texture_manual(MODE, 'lesion', case, scan, lesion, 'maxfirst', 0)
        yield get_task_texture_manual(MODE, 'lesion', case, scan, lesion, 'maxfirst', 1)
        yield get_task_texture_manual(MODE, 'lesion', case, scan, lesion, 'all', 0)
        yield get_task_texture_manual(MODE, 'lesion', case, scan, lesion, 'all', 1)
        if MODE.model in ('T2', 'T2w'):
            continue  # Do only lesion for these.
        yield get_task_texture_manual(MODE, 'CA', case, scan, lesion, 'maxfirst', 1)
        yield get_task_texture_manual(MODE, 'N', case, scan, lesion, 'maxfirst', 1)
        for ap in find_roi_param_combinations(MODE):
            yield get_task_texture_auto(MODE, ap, case, scan, lesion, 'maxfirst', 1)

#def task_texture_new():
#    """Generate texture features."""
#    for case, scan, lesion in lesions(MODE):
#        for mth in texture_methods(MODE):
#            for ws in texture_winsizes_new('lesion', MODE, mth):
#                yield get_task_texture_manual_new(MODE, 'lesion', case, scan,
#                        lesion, 'maxfirst', 0, mth, ws)

def get_task_mask_prostate(modality, case, scan, imagetype, postfix,
                           param='DICOM'):
    """Generate DICOM images with everything but prostate zeroed."""
    MASK_OUT_DICOM = DWILIB+'/mask_out_dicom.py'
    imagedir = 'dicoms_{}'.format(modality)
    maskdir = 'masks_prostate_{}'.format(modality)
    outdir = 'dicoms_masked_{}'.format(modality)
    d = dict(prg=MASK_OUT_DICOM, c=case, s=scan, md=maskdir, id=imagedir,
             od=outdir, it=imagetype, pox=postfix, p=param)
    d['mask'] = dwi.util.sglob('{md}/{c}_*_{s}*'.format(**d))
    d['img_src'] = dwi.util.sglob('{id}_*/{c}_*{it}_{s}{pox}/{p}'.format(**d))
    d['img_dst'] = '{od}/{c}{it}_{s}'.format(**d)
    cmd_rm = 'rm -Rf {img_dst}'.format(**d)
    cmd_cp = 'cp -R --no-preserve=all {img_src} {img_dst}'.format(**d)
    cmd_mask = '{prg} --mask {mask} --image {img_dst}'.format(**d)
    return {
        'name': '{c}_{s}'.format(**d),
        'actions': [(create_folder, [dirname(d['img_dst'])]),
                    cmd_rm,  # Remove destination image dir
                    cmd_cp,  # Copy source image dir to destination
                    cmd_mask],  # Mask destination image
        #'file_dep':  # TODO
        #'targets':  # TODO
        }

def task_mask_prostate_DWI():
    """Generate DICOM images with everything but prostate zeroed."""
    for case, scan in cases_scans(MODE):
        try:
            yield get_task_mask_prostate('DWI', case, scan, '_hB', '')
            #yield get_task_mask_prostate('SPAIR', case, scan, '', '_all')
        except IOError as e:
            print('mask_prostate_DWI', e)

def task_mask_prostate_T2():
    """Generate DICOM images with everything but prostate zeroed."""
    for case, scan in cases_scans(MODE):
        try:
            yield get_task_mask_prostate('T2', case, scan, '', '*')
            #yield get_task_mask_prostate('T2f', case, scan, '', '*', '*_Rho')
            #yield get_task_mask_prostate('T2w', case, scan, '', '*')
        except IOError as e:
            print('mask_prostate_T2', e)

def task_all():
    """Do all essential things."""
    return {
        'actions': None,
        'task_dep': ['select_roi', 'evaluate_autoroi', 'texture'],
        }
