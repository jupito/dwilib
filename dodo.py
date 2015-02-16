"""PyDoIt file for automating tasks."""

import glob
import itertools
import os
from os.path import dirname

from doit import get_var
#from doit.tools import Interactive
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
SAMPLELIST_FILE = 'samples_%s.txt' % SAMPLELIST
SAMPLES = dwi.util.read_sample_list(SAMPLELIST_FILE)
SUBWINDOWS = dwi.util.read_subwindows('subwindows.txt')

FIND_ROI_PARAMS = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # ROI side min (3 was not good)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], # ROI side max
        range(250, 2000, 250) + [50, 100, 150, 200], # Number of ROIs
]

def find_roi_param_combinations():
    if SAMPLELIST == 'test':
        #if MODEL == 'Mono':
        #    yield (10,10,500) # Mono: corr, auc
        #elif MODEL == 'Kurt':
        #    yield (9,9,1000) # Kurt: corr
        #    yield (2,2,250) # Kurt: auc
        yield (10,10,500) # Mono: auc
        yield (11,11,750) # Mono: corr
        yield (2,2,250) # Kurt: auc
        #yield (9,9,1000) # Kurt: corr
        yield (12,12,1750) # Kurt: corr
    else:
        for params in itertools.product(*FIND_ROI_PARAMS):
            #if params[0] <= params[1]:
            if params[0] == params[1]:
                yield params

# Common functions

def subwindow_to_str(subwindow):
    return ' '.join(map(str, subwindow))

def fit_cmd(model, subwindow, infiles, outfile):
    d = dict(prg=PMAP,
            m=model,
            sw=subwindow_to_str(subwindow),
            i=' '.join(infiles),
            o=outfile,
            )
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
#                'file_dep': [f],
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

def get_task_find_roi(case, scan, algparams):
    d = dict(prg=FIND_ROI, slf=SAMPLELIST_FILE, pd=PMAPDIR_DICOM, m=MODEL,
            p=PARAM, c=case, s=scan, ap=' '.join(algparams),
            ap_='_'.join(algparams))
    maskpath = 'masks_auto_{m}_{p}/{ap_}/{c}_{s}_auto.mask'.format(**d)
    figpath = 'find_roi_images_{m}_{p}/{ap_}/{c}_{s}.png'.format(**d)
    d.update(mp=maskpath, fp=figpath)
    file_deps = [SAMPLELIST_FILE]
    file_deps += glob.glob('masks_prostate/{c}_*_{s}_*/*'.format(**d))
    file_deps += glob.glob('masks_rois/{c}_*_{s}_*/*'.format(**d))
    cmd = '{prg} --samplelist {slf} --pmapdir {pd} --param {p} --cases {c} --scans {s} '\
            '--algparams {ap} --outmask {mp} --outfig {fp}'.format(**d)
    return {
            'name': '{ap_}_{c}_{s}'.format(**d),
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
        for sample in SAMPLES:
            case = sample['case']
            for scan in sample['scans']:
                yield get_task_find_roi(case, scan, map(str, algparams))

## Deprecated.
#def task_compare_masks():
#    """Compare ROI masks."""
#    for case, scan in SUBWINDOWS.keys():
#        subwindow = SUBWINDOWS[(case, scan)]
#        d = dict(prg=COMPARE_MASKS,
#                c=case,
#                s=scan,
#                w=subwindow_to_str(subwindow),
#                )
#        #file1 = 'masks/{c}_{s}_ca.mask'.format(**d)
#        file1 = glob.glob('masks/{c}_{s}_1_*.mask'.format(**d))[0]
#        file2 = 'masks_auto/{c}_{s}_auto.mask'.format(**d)
#        outfile = 'roi_comparison/{c}_{s}.txt'.format(**d)
#        d['f1'] = file1
#        d['f2'] = file2
#        d['o'] = outfile
#        cmd = '{prg} -s {w} {f1} {f2} > {o}'.format(**d)
#        if not os.path.exists(file1):
#            print 'Missing mask file %s' % file1
#            continue
#        yield {
#                'name': '{c}_{s}'.format(**d),
#                'actions': [(create_folder, [dirname(outfile)]),
#                        cmd],
#                'file_dep': [file1, file2],
#                'targets': [outfile],
#                'clean': True,
#                }

def get_task_select_roi_dicom_mask(case, scan, model, param, masktype,
        subwindow=None):
    """Select ROIs from the pmap DICOMs based on masks."""
    d = dict(c=case, s=scan, m=model, p=param, mt=masktype, sw=subwindow)
    maskpath = 'masks_rois/{c}_*_{s}_D_{mt}'.format(**d)
    outpath = 'rois_{mt}_{m}_{p}/{c}_x_x_{s}_{m}_{p}_{mt}.txt'.format(**d)
    s = 'results_{m}_combinedDICOM/{c}_*_{s}/{c}_*_{s}_{p}'.format(**d)
    inpath = glob.glob(s)[0]
    args = [SELECT_VOXELS]
    #args += ['-v']
    if subwindow:
        args += ['-s %s' % subwindow_to_str(subwindow)]
    args += ['-m %s' % maskpath]
    args += ['-i "%s"' % inpath]
    args += ['-o "%s"' % outpath]
    cmd = ' '.join(args)
    return {
            'name': '{c}_{s}_{mt}'.format(**d),
            'actions': [(create_folder, [dirname(outpath)]),
                    cmd],
            #'file_dep': [maskpath],
            'targets': [outpath],
            'clean': True,
            }

def get_task_select_roi(case, scan, model, param, masktype, algparams=[],
        subwindow=None):
    """Select ROIs from the pmap DICOMs based on masks."""
    d = dict(c=case, s=scan, m=model, p=param, mt=masktype,
            ap_='_'.join(algparams), sw=subwindow)
    maskpath = 'masks_{mt}_{m}_{p}/{ap_}/{c}_{s}_{mt}.mask'.format(**d)
    outpath = 'rois_{mt}_{m}_{p}/{ap_}/{c}_x_x_{s}_{m}_{p}_{mt}.txt'.format(**d)
    s = 'results_{m}_combinedDICOM/{c}_*_{s}/{c}_*_{s}_{p}'.format(**d)
    inpath = glob.glob(s)[0]
    args = [SELECT_VOXELS]
    #args += ['-v']
    if subwindow:
        args += ['-s %s' % subwindow_to_str(subwindow)]
    args += ['-m %s' % maskpath]
    args += ['-i "%s"' % inpath]
    args += ['-o "%s"' % outpath]
    cmd = ' '.join(args)
    return {
            'name': '{ap_}_{c}_{s}'.format(**d),
            'actions': [(create_folder, [dirname(outpath)]),
                    cmd],
            'file_dep': [maskpath],
            'targets': [outpath],
            'clean': True,
            }

def task_select_roi_manual():
    """Select cancer ROIs from the pmap DICOMs."""
    for sample in SAMPLES:
        case = sample['case']
        for scan in sample['scans']:
            yield get_task_select_roi_dicom_mask(case, scan, MODEL, PARAM, 'CA')
            yield get_task_select_roi_dicom_mask(case, scan, MODEL, PARAM, 'N')

def task_select_roi_auto():
    """Select automatic ROIs from the pmap DICOMs."""
    for algparams in find_roi_param_combinations():
        for sample in SAMPLES:
            case = sample['case']
            for scan in sample['scans']:
                yield get_task_select_roi(case, scan, MODEL, PARAM, 'auto',
                        algparams=map(str, algparams))

def task_evaluate_autoroi():
    """Evaluate auto-ROI prediction ability by ROC AUC and correlation with
    Gleason score."""
    outfile = 'autoroi_evaluation_%s_%s.txt' % (MODEL, SAMPLELIST)
    d = dict(slf=SAMPLELIST_FILE, prg_auc=CALC_AUC, prg_corr=CORRELATION,
            m=MODEL, p=PARAM, o=outfile)
    cmds = ['echo -n > {o}'.format(**d)]
    for algparams in find_roi_param_combinations():
        d['ap'] = ' '.join(map(str, algparams))
        d['ap_'] = '_'.join(map(str, algparams))
        d['i'] = 'rois_auto_{m}_{p}/{ap_}'.format(**d)
        s = 'echo -n {ap} >> {o}'
        cmds.append(s.format(**d))
        s = r'echo -n \\t`{prg_auc} --patients patients.txt --samplelist {slf} --threshold 3+3 --average --autoflip --pmapdir {i}` >> {o}'
        cmds.append(s.format(**d))
        s = r'echo -n \\t`{prg_auc} --patients patients.txt --samplelist {slf} --threshold 3+4 --average --autoflip --pmapdir {i}` >> {o}'
        cmds.append(s.format(**d))
        s = r'echo -n \\t`{prg_corr} --patients patients.txt --samplelist {slf} --thresholds 3+3 3+4 --average --pmapdir {i}` >> {o}'
        cmds.append(s.format(**d))
        s = r'echo \\t`{prg_corr} --patients patients.txt --samplelist {slf} --thresholds --average --pmapdir {i}` >> {o}'
        cmds.append(s.format(**d))
    return {
            'actions': cmds,
            'task_dep': ['select_roi_auto'],
            'targets': [outfile],
            'clean': True,
            }
