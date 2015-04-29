"""PyDoIt file for automating tasks."""

import glob
import itertools
import os
from os.path import dirname

from doit import get_var
#from doit.tools import Interactive
from doit.tools import check_timestamp_unchanged
from doit.tools import create_folder
#from doit.tools import result_dep

import dwi.util

"""
Backends:
dbm: (default) It uses python dbm module.
json: Plain text using a json structure, it is slow but good for debugging.
sqlite3: (experimental) very slow implementation, support concurrent access.
"""

DOIT_CONFIG = {
    'backend': 'sqlite3',
    'default_tasks': [],
    'verbosity': 1,
    'num_process': 7,
    }

DWILIB = '~/src/dwilib/tools'
PMAP = DWILIB+'/pmap.py'
ANON = DWILIB+'/anonymize_dicom.py'
FIND_ROI = DWILIB+'/find_roi.py'
COMPARE_MASKS = DWILIB+'/compare_masks.py'
SELECT_VOXELS = DWILIB+'/select_voxels.py'
CALC_AUC = DWILIB+'/roc_auc.py'
CORRELATION = DWILIB+'/correlation.py'
MASKTOOL = DWILIB+'/masktool.py'
GET_TEXTURE = DWILIB+'/get_texture.py'
MASK_OUT_DICOM = DWILIB+'/mask_out_dicom.py'

SUBREGION_DIR = 'subregions'

MODELS = 'Si SiN Mono MonoN Kurt KurtN Stretched StretchedN '\
        'Biexp BiexpN'.split()
DEFAULT_PARAMS = dict(Mono='ADCm', MonoN='ADCmN',
        Kurt='ADCk', KurtN='ADCkN',
        T2='raw')
MODEL = get_var('model', 'Mono')
PARAM = get_var('param', DEFAULT_PARAMS[MODEL])

SAMPLELIST = get_var('samplelist', 'all') # Sample list (train, test, etc)
SUBWINDOWS = dwi.util.read_subwindows('subwindows.txt')

FIND_ROI_PARAMS = [
        [1, 2, 3], # ROI depth min
        [1, 2, 3], # ROI depth max
        range(2, 13), # ROI side min (3 was not good)
        range(3, 13), # ROI side max
        range(250, 2000, 250) + [50, 100, 150, 200], # Number of ROIs
]

def texture_methods(model=MODEL):
    return ' '.join([
        'stats',
        'glcm',
        #'haralick',
        'lbp',
        'hog',
        'gabor',
        'moment',
        'haar',
        'sobel',
        'glcm_mbb',
        #'haralick_mbb',
        ])

def texture_winsizes(model=MODEL):
    if model == 'T2':
        return ' '.join(map(str, range(3, 36, 4)))
    else:
        return ' '.join(map(str, range(3, 16, 2)))

def find_roi_param_combinations():
    """Generate all find_roi.py parameter combinations."""
    if SAMPLELIST == 'test':
        params = [
                (2,3,10,10,500), # Mono: corr, auc
                (2,3,10,10,1750), # Mono: corr
                (2,3,11,11,750), # Mono: corr
                #(2,3,2,2,250), # Kurt: auc
                #(2,3,9,9,1000), # Kurt: corr
                #(2,3,12,12,1750), # Kurt: corr
                #(2,3,5,5,500), # Kurt K: corr, auc
                ]
    else:
        params = itertools.product(*FIND_ROI_PARAMS)
    for t in params:
        if t[0] <= t[1] and t[2] == t[3]:
            yield map(str, t)

def pmapdir_dicom(model):
    s = 'dicoms_{m}_*'.format(m=model)
    path = dwi.util.sglob(s, typ='dir')
    return path

def pmap_dicom(**d):
    s = 'results_{m}_combinedDICOM/{c}_*_{s}/{c}_*_{s}_{p}'
    return dwi.util.sglob(s.format(**d), typ='dir')

def samplelist_file(samplelist):
    return 'samples_%s.txt' % samplelist

def mask_path(d):
    do_glob = True
    if d['mt'] == 'lesion':
        if d['m'] == 'T2':
            path = 'masks_lesion_T2/PCa_masks_*_[O1]*/{c}_*{s}_*'
        else:
            path = 'masks_lesion/PCa_masks_*_[O1]*/{c}_*{s}_*'
        deps = '{}/*'.format(path)
    elif d['mt'] == 'CA' or d['mt'] == 'N':
        path = 'masks_rois/{c}_*_{s}_D_{mt}'
        deps = '{}/*'.format(path)
    elif d['mt'] == 'auto':
        # Don't require existence, can be generated.
        do_glob = False
        path = 'masks_auto_{m}_{p}/{ap_}/{c}_{s}_auto.mask'
        deps = [path]
    else:
        raise Exception('Unknown mask type: {mt}'.format(**d))
    path = path.format(**d)
    if do_glob:
        path = dwi.util.sglob(path)
        deps = glob.glob(path)
    return path, deps

SAMPLES = dwi.util.read_sample_list(samplelist_file(SAMPLELIST))

def cases_scans():
    """Generate all case, scan pairs."""
    for sample in SAMPLES:
        case = sample['case']
        for scan in sample['scans']:
            yield case, scan

def fit_cmd(model, subwindow, infiles, outfile):
    d = dict(prg=PMAP, m=model, sw=' '.join(map(str, subwindow)),
            i=' '.join(infiles), o=outfile)
    s = '{prg} -m {m} -s {sw} -d {i} -o {o}'.format(**d)
    return s

# Tasks

#def task_anonymize():
#    """Anonymize imaging data."""
#    files = glob.glob('dicoms/*/DICOMDIR') + glob.glob('dicoms/*/DICOM/*')
#    files.sort()
#    for f in files:
#        cmd = '{prg} -v -i {f}'.format(prg=ANON, f=f)
#        yield {
#                'name': f,
#                'actions': [cmd],
#                #'file_dep': [f],
#                }

def task_fit():
    """Fit models to imaging data."""
    for case, scan in SUBWINDOWS.keys():
        subwindow = SUBWINDOWS[(case, scan)]
        d = dict(c=case, s=scan)
        infiles = sorted(glob.glob('dicoms/{c}_*_hB_{s}*/DICOM/*'.format(**d)))
        if len(infiles) == 0:
            continue
        for model in MODELS:
            d['m'] = model
            outfile = 'pmaps/pmap_{c}_{s}_{m}.txt'.format(**d)
            cmd = fit_cmd(model, subwindow, infiles, outfile)
            yield {
                    'name': '{c}_{s}_{m}'.format(**d),
                    'actions': [(create_folder, [dirname(outfile)]),
                            cmd],
                    'file_dep': infiles,
                    'targets': [outfile],
                    'clean': True,
                    }

def task_make_subregion():
    """Make minimum bounding box + 10 voxel subregions from prostate masks."""
    for case, scan in cases_scans():
        d = dict(prg=MASKTOOL, c=case, s=scan)
        d.update(i='masks_prostate/{c}_*_{s}_*'.format(**d),
                o='subregions/{c}_{s}_subregion10.txt'.format(**d))
        file_deps = glob.glob('{i}/*'.format(**d))
        cmd = '{prg} -i {i} --pad 10 -s {o}'.format(**d)
        yield {
                'name': '{c}_{s}'.format(**d),
                'actions': [(create_folder, [dirname(d['o'])]),
                        cmd],
                'file_dep': file_deps,
                'targets': [d['o']],
                'clean': True,
                }

def get_task_find_roi(samplelist, case, scan, model, param, algparams):
    d = dict(prg=FIND_ROI, slf=samplelist_file(samplelist),
            pd=pmapdir_dicom(model), srd=SUBREGION_DIR, m=model, p=param,
            c=case, s=scan, ap=' '.join(algparams), ap_='_'.join(algparams))
    maskpath = 'masks_auto_{m}_{p}/{ap_}/{c}_{s}_auto.mask'.format(**d)
    figpath = 'find_roi_images_{m}_{p}/{ap_}/{c}_{s}.png'.format(**d)
    d.update(mp=maskpath, fp=figpath)
    file_deps = []
    file_deps += ['{srd}/{c}_{s}_subregion10.txt'.format(**d)]
    file_deps += glob.glob('masks_prostate/{c}_*_{s}_*/*'.format(**d))
    file_deps += glob.glob('masks_rois/{c}_*_{s}_*/*'.format(**d))
    cmd = '{prg} --samplelist {slf} --pmapdir {pd} --subregiondir {srd} '\
            '--param {p} --cases {c} --scans {s} --algparams {ap} '\
            '--outmask {mp} --outfig {fp}'.format(**d)
    return {
            'name': '{m}_{p}_{ap_}_{c}_{s}'.format(**d),
            'actions': [(create_folder, [dirname(maskpath)]),
                        (create_folder, [dirname(figpath)]),
                    cmd],
            'file_dep': file_deps,
            'targets': [maskpath, figpath],
            'clean': True,
            }

def task_find_roi():
    """Find a cancer ROI automatically."""
    for algparams in find_roi_param_combinations():
        for case, scan in cases_scans():
            yield get_task_find_roi(SAMPLELIST, case, scan, MODEL, PARAM,
                    algparams)

def get_task_select_roi_manual(case, scan, model, param, masktype):
    """Select ROIs from the pmap DICOMs based on masks."""
    d = dict(c=case, s=scan, m=model, p=param, mt=masktype)
    maskpath = dwi.util.sglob('masks_rois/{c}_*_{s}_D_{mt}'.format(**d))
    outpath = 'rois_{mt}_{m}_{p}/{c}_x_x_{s}_{m}_{p}_{mt}.txt'.format(**d)
    inpath = pmap_dicom(**d)
    args = [SELECT_VOXELS]
    args += ['-m %s' % maskpath]
    args += ['-i "%s"' % inpath]
    args += ['-o "%s"' % outpath]
    cmd = ' '.join(args)
    return {
            'name': '{m}_{p}_{mt}_{c}_{s}'.format(**d),
            'actions': [(create_folder, [dirname(outpath)]),
                    cmd],
            #'file_dep': [maskpath],
            'targets': [outpath],
            'uptodate': [check_timestamp_unchanged(inpath),
                    check_timestamp_unchanged(maskpath)],
            'clean': True,
            }

def get_task_select_roi_auto(case, scan, model, param, algparams):
    """Select ROIs from the pmap DICOMs based on masks."""
    d = dict(c=case, s=scan, m=model, p=param, mt='auto',
            ap_='_'.join(algparams))
    maskpath = 'masks_{mt}_{m}_{p}/{ap_}/{c}_{s}_{mt}.mask'.format(**d)
    outpath = 'rois_{mt}_{m}_{p}/{ap_}/{c}_x_x_{s}_{m}_{p}_{mt}.txt'.format(**d)
    inpath = pmap_dicom(**d)
    args = [SELECT_VOXELS]
    args += ['-m %s' % maskpath]
    args += ['-i "%s"' % inpath]
    args += ['-o "%s"' % outpath]
    cmd = ' '.join(args)
    return {
            'name': '{m}_{p}_{ap_}_{c}_{s}'.format(**d),
            'actions': [(create_folder, [dirname(outpath)]),
                    cmd],
            'file_dep': [maskpath],
            'targets': [outpath],
            'uptodate': [check_timestamp_unchanged(inpath)],
            'clean': True,
            }

def task_select_roi_manual():
    """Select cancer ROIs from the pmap DICOMs."""
    for masktype in ['CA', 'N']:
        for case, scan in cases_scans():
            try:
                yield get_task_select_roi_manual(case, scan, MODEL, PARAM,
                        masktype)
            except IOError, e:
                print e

def task_select_roi_auto():
    """Select automatic ROIs from the pmap DICOMs."""
    for algparams in find_roi_param_combinations():
        for case, scan in cases_scans():
            try:
                yield get_task_select_roi_auto(case, scan, MODEL, PARAM, algparams)
            except IOError, e:
                print e

def task_select_roi():
    """Select all ROIs task group."""
    return {
            'actions': None,
            'task_dep': ['select_roi_manual', 'select_roi_auto'],
            }

def get_task_autoroi_auc(samplelist, model, param, threshold):
    """Evaluate auto-ROI prediction ability by ROC AUC with Gleason score."""
    d = dict(sl=samplelist, slf=samplelist_file(samplelist), prg=CALC_AUC,
            m=model, p=param, t=threshold)
    d['o'] = 'autoroi_auc_{t}_{m}_{p}_{sl}.txt'.format(**d)
    cmds = ['echo -n > {o}'.format(**d)]
    for algparams in find_roi_param_combinations():
        d['ap_'] = '_'.join(algparams)
        d['i'] = 'rois_auto_{m}_{p}/{ap_}'.format(**d)
        s = r'echo `{prg} --patients patients.txt --samplelist {slf} --threshold {t} --voxel mean --autoflip --pmapdir {i}` {ap_} >> {o}'
        cmds.append(s.format(**d))
    return {
            'name': 'autoroi_auc_{sl}_{m}_{p}_{t}'.format(**d),
            'actions': cmds,
            'task_dep': ['select_roi_auto'],
            'targets': [d['o']],
            'clean': True,
            }

def get_task_autoroi_correlation(samplelist, model, param, thresholds):
    """Evaluate auto-ROI prediction ability by correlation with Gleason
    score."""
    d = dict(sl=samplelist, slf=samplelist_file(samplelist), prg=CORRELATION,
            m=model, p=param, t=thresholds, t_=thresholds.replace(' ', ','))
    d['o'] = 'autoroi_correlation_{t_}_{m}_{p}_{sl}.txt'.format(**d)
    cmds = ['echo -n > {o}'.format(**d)]
    for algparams in find_roi_param_combinations():
        d['ap_'] = '_'.join(algparams)
        d['i'] = 'rois_auto_{m}_{p}/{ap_}'.format(**d)
        s = r'echo `{prg} --patients patients.txt --samplelist {slf} --thresholds {t} --voxel mean --pmapdir {i}` {ap_} >> {o}'
        cmds.append(s.format(**d))
    return {
            'name': 'autoroi_correlation_{sl}_{m}_{p}_{t_}'.format(**d),
            'actions': cmds,
            'task_dep': ['select_roi_auto'],
            'targets': [d['o']],
            'clean': True,
            }

def task_evaluate_autoroi():
    """Evaluate auto-ROI prediction ability."""
    yield get_task_autoroi_auc(SAMPLELIST, MODEL, PARAM, '3+3')
    yield get_task_autoroi_auc(SAMPLELIST, MODEL, PARAM, '3+4')
    yield get_task_autoroi_correlation(SAMPLELIST, MODEL, PARAM, '3+3 3+4')
    yield get_task_autoroi_correlation(SAMPLELIST, MODEL, PARAM, '')

def get_task_texture_manual(model, param, masktype, case, scan):
    """Generate texture features."""
    d = dict(prg=GET_TEXTURE, methods=texture_methods(model),
            winsizes=texture_winsizes(model), pd=pmapdir_dicom(model), m=model,
            p=param, mt=masktype, c=case, s=scan)
    d['slices'] = 'maxfirst'
    if masktype == 'lesion':
        d['portion'] = 0 # Window center must be inside lesion.
    else:
        d['portion'] = 1 # Whole window must be inside lesion if possible.
    d['mask'], mask_deps = mask_path(d)
    d['o'] = 'texture_{mt}_{m}_{p}/{c}_{s}.txt'.format(**d)
    cmd = '{prg} --methods {methods} --winsizes {winsizes}'\
            ' --pmapdir {pd} --param {p} --case {c} --scan {s} --mask {mask}'\
            ' --slices {slices} --portion {portion}'\
            ' --output {o}'.format(**d)
    return {
            'name': '{m}_{p}_{mt}_{c}_{s}'.format(**d),
            'actions': [(create_folder, [dirname(d['o'])]),
                    cmd],
            'file_dep': mask_deps,
            'targets': [d['o']],
            'clean': True,
            }

def get_task_texture_auto(model, param, algparams, case, scan):
    """Generate texture features."""
    d = dict(prg=GET_TEXTURE, methods=texture_methods(model),
            winsizes=texture_winsizes(model), pd=pmapdir_dicom(model), m=model,
            p=param, mt='auto', c=case, s=scan, ap_='_'.join(algparams))
    d['slices'] = 'maxfirst'
    d['portion'] = 1 # Whole window must be inside lesion if possible.
    d['mask'], mask_deps = mask_path(d)
    d['o'] = 'texture_{mt}_{m}_{p}/{ap_}/{c}_{s}.txt'.format(**d)
    cmd = '{prg} --methods {methods} --winsizes {winsizes}'\
            ' --pmapdir {pd} --param {p} --case {c} --scan {s} --mask {mask}'\
            ' --slices {slices} --portion {portion}'\
            ' --output {o}'.format(**d)
    return {
            'name': '{m}_{p}_{ap_}_{c}_{s}'.format(**d),
            'actions': [(create_folder, [dirname(d['o'])]),
                    cmd],
            'file_dep': mask_deps,
            'targets': [d['o']],
            'clean': True,
            }

def task_texture():
    """Generate texture features."""
    for case, scan in cases_scans():
        for masktype in ['lesion']:
            yield get_task_texture_manual(MODEL, PARAM, masktype, case, scan)
        if MODEL == 'T2':
            continue # Skip the rest.
        for masktype in ['CA', 'N']:
            yield get_task_texture_manual(MODEL, PARAM, masktype, case, scan)
        for algparams in find_roi_param_combinations():
            yield get_task_texture_auto(MODEL, PARAM, algparams, case, scan)

def get_task_mask_prostate(case, scan, maskdir, imagedir, outdir, imagetype,
        postfix, param='DICOM'):
    """Generate DICOM images with everything but prostate zeroed."""
    d = dict(prg=MASK_OUT_DICOM, c=case, s=scan, md=maskdir, id=imagedir,
            od=outdir, it=imagetype, pox=postfix, p=param)
    d['mask'] = dwi.util.sglob('{md}/{c}_*_{s}*'.format(**d))
    d['img_src'] = dwi.util.sglob('{id}/{c}_*{it}_{s}{pox}/{p}'.format(**d))
    d['img_dst'] = '{od}/{c}{it}_{s}'.format(**d)
    cmd_rm = 'rm -Rf {img_dst}'.format(**d)
    cmd_cp = 'cp -R --no-preserve=all {img_src} {img_dst}'.format(**d)
    cmd_mask = '{prg} --mask {mask} --image {img_dst}'.format(**d)
    return {
            'name': '{c}_{s}'.format(**d),
            'actions': [(create_folder, [dirname(d['img_dst'])]),
                    cmd_rm, # Remove destination image dir
                    cmd_cp, # Copy source image dir to destination
                    cmd_mask], # Mask destination image
            #'file_dep': # TODO
            #'targets': # TODO
            }

def task_mask_prostate():
    """Generate DICOM images with everything but prostate zeroed."""
    for case, scan in cases_scans():
        try:
            yield get_task_mask_prostate(case, scan, 'masks_prostate', 'dicoms', 'dicoms_masked_DWI', '_hB', '')
            #yield get_task_mask_prostate(case, scan, 'masks_prostate', 'new/for_jussi_data_missing_04_01_2015/SPAIR_f_12b_highb', 'dicoms_masked_DWI_missing', '', '_all')
        except IOError, e:
            print e

def task_mask_prostate_T2():
    """Generate DICOM images with everything but prostate zeroed."""
    for case, scan in cases_scans():
        try:
            yield get_task_mask_prostate(case, scan, 'masks_prostate_T2', 'dicoms_T2_data_for_72cases_03_05_2015_no65', 'dicoms_masked_T2', '', '_T2')
            #yield get_task_mask_prostate(case, scan, 'masks_prostate_T2', 'dicoms_T2_data_for_72cases_03_05_2015_no65_FITTED', 'dicoms_masked_T2_rho', '', '_T2', '*_Rho')
            #yield get_task_mask_prostate(case, scan, 'masks_prostate_T2W', 'dicoms_T2W_TSE_2.5m', 'dicoms_masked_T2W', '', '*')
        except IOError, e:
            print e

def task_all():
    """Do all essential things."""
    return {
            'actions': None,
            'task_dep': ['select_roi', 'evaluate_autoroi', 'texture'],
            }
