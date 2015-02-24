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
    }

DWILIB = '~/src/dwilib/tools'
PMAP = DWILIB+'/pmap.py'
ANON = DWILIB+'/anonymize_dicom.py'
FIND_ROI = DWILIB+'/find_roi.py'
COMPARE_MASKS = DWILIB+'/compare_masks.py'
SELECT_VOXELS = DWILIB+'/select_voxels.py'
CALC_AUC = DWILIB+'/roc_auc.py'
CORRELATION = DWILIB+'/correlation.py'

MODELS = 'Si SiN Mono MonoN Kurt KurtN Stretched StretchedN '\
        'Biexp BiexpN'.split()
DEFAULT_PARAMS = dict(Mono='ADCm', MonoN='ADCmN', Kurt='ADCk', KurtN='ADCkN')
MODEL = get_var('model', 'Mono')
PARAM = get_var('param', DEFAULT_PARAMS[MODEL])
PMAPDIR_DICOM = 'results_{m}_combinedDICOM'.format(m=MODEL)

SAMPLELIST = get_var('samplelist', 'all') # Sample list (train, test, etc)
SUBWINDOWS = dwi.util.read_subwindows('subwindows.txt')

FIND_ROI_PARAMS = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], # ROI side min (3 was not good)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], # ROI side max
        range(250, 2000, 250) + [50, 100, 150, 200], # Number of ROIs
]

def find_roi_param_combinations():
    """Generate all find_roi.py parameter combinations."""
    if SAMPLELIST == 'test':
        params = [
                (10,10,500), # Mono: auc
                (11,11,750), # Mono: corr
                (2,2,250), # Kurt: auc
                #(9,9,1000), # Kurt: corr
                (12,12,1750), # Kurt: corr
                ]
    else:
        params = itertools.product(*FIND_ROI_PARAMS)
    for t in params:
        #if t[0] <= t[1]:
        if t[0] == t[1]:
            yield map(str, t)

def samplelist_file(samplelist):
    return 'samples_%s.txt' % samplelist

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

def get_task_find_roi(pmapdir, samplelist, case, scan, model, param, algparams):
    d = dict(prg=FIND_ROI, slf=samplelist_file(samplelist), pd=pmapdir, m=model,
            p=param, c=case, s=scan, ap=' '.join(algparams),
            ap_='_'.join(algparams))
    maskpath = 'masks_auto_{m}_{p}/{ap_}/{c}_{s}_auto.mask'.format(**d)
    figpath = 'find_roi_images_{m}_{p}/{ap_}/{c}_{s}.png'.format(**d)
    d.update(mp=maskpath, fp=figpath)
    file_deps = [d['slf']]
    file_deps += glob.glob('masks_prostate/{c}_*_{s}_*/*'.format(**d))
    file_deps += glob.glob('masks_rois/{c}_*_{s}_*/*'.format(**d))
    cmd = '{prg} --samplelist {slf} --pmapdir {pd} --param {p} --cases {c} --scans {s} '\
            '--algparams {ap} --outmask {mp} --outfig {fp}'.format(**d)
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
            yield get_task_find_roi(PMAPDIR_DICOM, SAMPLELIST, case, scan,
                    MODEL, PARAM, algparams)

def get_task_select_roi_manual(case, scan, model, param, masktype):
    """Select ROIs from the pmap DICOMs based on masks."""
    d = dict(c=case, s=scan, m=model, p=param, mt=masktype)
    maskpath = dwi.util.sglob('masks_rois/{c}_*_{s}_D_{mt}'.format(**d))
    outpath = 'rois_{mt}_{m}_{p}/{c}_x_x_{s}_{m}_{p}_{mt}.txt'.format(**d)
    inpath = dwi.util.sglob('results_{m}_combinedDICOM/{c}_*_{s}/{c}_*_{s}_{p}'.format(**d))
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
            'uptodate': [check_timestamp_unchanged(maskpath),
                    check_timestamp_unchanged(inpath)],
            'clean': True,
            }

def get_task_select_roi_auto(case, scan, model, param, algparams):
    """Select ROIs from the pmap DICOMs based on masks."""
    d = dict(c=case, s=scan, m=model, p=param, mt='auto',
            ap_='_'.join(algparams))
    maskpath = 'masks_{mt}_{m}_{p}/{ap_}/{c}_{s}_{mt}.mask'.format(**d)
    outpath = 'rois_{mt}_{m}_{p}/{ap_}/{c}_x_x_{s}_{m}_{p}_{mt}.txt'.format(**d)
    inpath = dwi.util.sglob('results_{m}_combinedDICOM/{c}_*_{s}/{c}_*_{s}_{p}'.format(**d))
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
            'clean': True,
            }

def task_select_roi_manual():
    """Select cancer ROIs from the pmap DICOMs."""
    for masktype in ['CA', 'N']:
        for case, scan in cases_scans():
            yield get_task_select_roi_manual(case, scan, MODEL, PARAM, masktype)

def task_select_roi_auto():
    """Select automatic ROIs from the pmap DICOMs."""
    for algparams in find_roi_param_combinations():
        for case, scan in cases_scans():
            yield get_task_select_roi_auto(case, scan, MODEL, PARAM, algparams)

def task_select_roi():
    """Select all ROIs task group."""
    return {
            'actions': None,
            'task_dep': ['select_roi_manual', 'select_roi_auto'],
            }

#def task_evaluate_autoroi_OLD():
#    """Evaluate auto-ROI prediction ability by ROC AUC and correlation with
#    Gleason score."""
#    outfile = 'autoroi_evaluation_%s_%s_%s.txt' % (MODEL, PARAM, SAMPLELIST)
#    d = dict(slf=SAMPLELIST_FILE, prg_auc=CALC_AUC, prg_corr=CORRELATION,
#            m=MODEL, p=PARAM, o=outfile)
#    cmds = ['echo -n > {o}'.format(**d)]
#    for algparams in find_roi_param_combinations():
#        d['ap'] = ' '.join(algparams)
#        d['ap_'] = '_'.join(algparams)
#        d['i'] = 'rois_auto_{m}_{p}/{ap_}'.format(**d)
#        s = 'echo -n {ap} >> {o}'
#        cmds.append(s.format(**d))
#        s = r'echo -n \\t`{prg_auc} --patients patients.txt --samplelist {slf} --threshold 3+3 --average --autoflip --pmapdir {i}` >> {o}'
#        cmds.append(s.format(**d))
#        s = r'echo -n \\t`{prg_auc} --patients patients.txt --samplelist {slf} --threshold 3+4 --average --autoflip --pmapdir {i}` >> {o}'
#        cmds.append(s.format(**d))
#        s = r'echo -n \\t`{prg_corr} --patients patients.txt --samplelist {slf} --thresholds 3+3 3+4 --average --pmapdir {i}` >> {o}'
#        cmds.append(s.format(**d))
#        s = r'echo \\t`{prg_corr} --patients patients.txt --samplelist {slf} --thresholds --average --pmapdir {i}` >> {o}'
#        cmds.append(s.format(**d))
#    return {
#            'actions': cmds,
#            'task_dep': ['select_roi_auto'],
#            'targets': [outfile],
#            'clean': True,
#            }

def get_task_autoroi_auc(samplelist, model, param, threshold):
    """Evaluate auto-ROI prediction ability by ROC AUC with Gleason score."""
    d = dict(sl=samplelist, slf=samplelist_file(samplelist), prg=CALC_AUC,
            m=model, p=param, t=threshold)
    d['o'] = 'autoroi_auc_{t}_{m}_{p}_{sl}.txt'.format(**d)
    cmds = ['echo -n > {o}'.format(**d)]
    for algparams in find_roi_param_combinations():
        d['ap_'] = '_'.join(algparams)
        d['i'] = 'rois_auto_{m}_{p}/{ap_}'.format(**d)
        s = r'echo `{prg} --patients patients.txt --samplelist {slf} --threshold {t} --average --autoflip --pmapdir {i}` {ap_} >> {o}'
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
        s = r'echo `{prg} --patients patients.txt --samplelist {slf} --thresholds {t} --average --pmapdir {i}` {ap_} >> {o}'
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
