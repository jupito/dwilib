"""PyDoIt file for automating tasks."""

import glob
import os

#from doit.tools import Interactive
from doit.tools import create_folder
#from doit.tools import result_dep

"""
Backends:
dbm: (default) It uses python dbm module.
json: Plain text using a json structure, it is slow but good for debugging.
sqlite3: (experimental) very slow implementation, support concurrent access.
"""

DOIT_CONFIG = {
    'backend': 'sqlite3',
    'default_tasks': [],
    'verbosity': 2,
    }

DWILIB = '~/src/dwilib'
PMAP = DWILIB+'/pmap.py'
ANON = DWILIB+'/anonymize_dicom.py'
FIND_ROI = DWILIB+'/find_roi.py'
COMPARE_MASKS = DWILIB+'/compare_masks.py'
SELECT_VOXELS = DWILIB+'/select_voxels.py'
MODELS = 'Si SiN Mono MonoN Kurt KurtN Stretched StretchedN '\
        'Biexp BiexpN'.split()

# Common functions

# TODO: Call from common modules instead.
def read_sample_list(filename):
    """Read a list of samples from file."""
    import re
    entries = []
    p = re.compile(r'(\d+)\s+(\w+)\s+([\w,]+)')
    with open(filename, 'r') as f:
        for line in f:
            m = p.match(line.strip())
            if m:
                case, name, scans = m.groups()
                case = int(case)
                name = name.lower()
                scans = tuple(sorted(scans.lower().split(',')))
                d = dict(case=case, name=name, scans=scans)
                entries.append(d)
    return entries

def read_subwindows(filename):
    r = {}
    with open(filename, 'rb') as f:
        for line in f.xreadlines():
            line = line.strip()
            if line.startswith('#'):
                continue
            words = line.split()
            if len(words) != 8:
                raise Exception('Cannot parse subwindow: %s.' % line)
            case, scan, subwindow = int(words[0]), words[1], map(int, words[2:])
            r[(case, scan)] = subwindow
    return r

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
        outdir = 'pmaps'
        d = dict(c=case, s=scan, od=outdir)
        infiles = sorted(glob.glob('dicoms/{c}_*_hB_{s}*/DICOM/*'.format(**d)))
        if len(infiles) == 0:
            continue
        for model in MODELS:
            d['m'] = model
            outfile = '{od}/pmap_{c}_{s}_{m}.txt'.format(**d)
            cmd = fit_cmd(model, subwindow, infiles, outfile)
            yield {
                    'name': '{c}_{s}_{m}'.format(**d),
                    'actions': [(create_folder, [outdir]),
                            cmd],
                    'file_dep': infiles,
                    'targets': [outfile],
                    'clean': True,
                    }

def task_find_roi():
    """Find ROIs."""
    for case, scan in SUBWINDOWS.keys():
        outdir = 'masks_auto'
        d = dict(prg=FIND_ROI, c=case, s=scan, od=outdir)
        infile = 'pmaps/pmap_{c}_{s}_MonoN.txt'.format(**d)
        inmask = glob.glob('masks/{c}_{s}_1_*.mask'.format(**d))[0]
        outfile = '{od}/{c}_{s}_auto.mask'.format(**d)
        graphicfile = '{od}/autoroi_{c}_{s}.png'.format(**d)
        d['i'] = infile
        d['m'] = inmask
        d['o'] = outfile
        d['g'] = graphicfile
        cmd = '{prg} -i {i} -m {m} -o {o} -g {g}'.format(**d)
        if not os.path.exists(infile):
            continue
        yield {
                'name': '{c}_{s}'.format(**d),
                'actions': [(create_folder, [outdir]),
                        cmd],
                'file_dep': [infile],
                'targets': [outfile, graphicfile],
                'clean': True,
                }

def task_compare_masks():
    """Compare ROI masks."""
    for case, scan in SUBWINDOWS.keys():
        subwindow = SUBWINDOWS[(case, scan)]
        d = dict(prg=COMPARE_MASKS,
                c=case,
                s=scan,
                w=subwindow_to_str(subwindow),
                )
        #file1 = 'masks/{c}_{s}_ca.mask'.format(**d)
        file1 = glob.glob('masks/{c}_{s}_1_*.mask'.format(**d))[0]
        file2 = 'masks_auto/{c}_{s}_auto.mask'.format(**d)
        outdir = 'roi_comparison'
        outfile = os.path.join(outdir, '{c}_{s}.txt'.format(**d))
        d['f1'] = file1
        d['f2'] = file2
        d['o'] = outfile
        cmd = '{prg} -s {w} {f1} {f2} > {o}'.format(**d)
        if not os.path.exists(file1):
            print 'Missing mask file %s' % file1
            continue
        yield {
                'name': '{c}_{s}'.format(**d),
                'actions': [(create_folder, [outdir]),
                        cmd],
                'file_dep': [file1, file2],
                'targets': [outfile],
                'clean': True,
                }

def get_task_select_roi(case, scan, model, param, subwindow, mask):
    d = dict(c=case, s=scan, m=model, p=param, subwindow=subwindow, mask=mask)
    outdir = 'rois_{mask}'.format(**d)
    s = os.path.join('masks_{mask}', '{c}_{s}_*.mask')
    maskpath = glob.glob(s.format(**d))[0]
    s = os.path.join('results_{m}_combinedDICOM', '{c}_*_{s}',
            '{c}_*_{s}_{p}')
    inpath = glob.glob(s.format(**d))[0]
    s = os.path.join(outdir, '{c}_{s}_{m}_{p}_{mask}.txt')
    outpath = s.format(**d)
    args = [SELECT_VOXELS]
    #args += ['-v']
    if subwindow:
        args += ['-s %s' % subwindow_to_str(subwindow)]
    args += ['-m "%s"' % maskpath]
    args += ['-i "%s"' % inpath]
    args += ['-o "%s"' % outpath]
    cmd = ' '.join(args)
    return {
            'name': '{c}_{s}'.format(**d),
            'actions': [(create_folder, [outdir]),
                    cmd],
            'file_dep': [maskpath],
            'targets': [outpath],
            'clean': True,
            }

def task_select_roi_cancer():
    """Select cancer ROIs from the pmap DICOMs."""
    for sample in SAMPLES_ALL:
        for scan in sample['scans']:
            case = sample['case']
            subwin = None
            mask = 'cancer'
            yield get_task_select_roi(case, scan, 'Mono', 'ADCm', subwin, mask)

def task_select_roi_auto():
    """Select automatic ROIs from the pmap DICOMs."""
    for sample in SAMPLES_ALL:
        for scan in sample['scans']:
            case = sample['case']
            subwin = SUBWINDOWS[(case, scan)]
            mask = 'auto'
            yield get_task_select_roi(case, scan, 'Mono', 'ADCm', subwin, mask)

SAMPLES_ALL = read_sample_list('samples_all.txt')
SUBWINDOWS = read_subwindows('subwindows.txt')
