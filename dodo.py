"""PyDoIt file for automating tasks."""

import glob
import os

#from doit.tools import Interactive
from doit.tools import create_folder
#from doit.tools import result_dep

DOIT_CONFIG = {
    'default_tasks': [],
    'verbosity': 2,
    }

DWILIB = '~/src/dwilib'
PMAP = DWILIB+'/pmap.py'
ANON = DWILIB+'/anonymize_dicom.py'
FIND_ROI = DWILIB+'/find_roi.py'
COMPARE_MASKS = DWILIB+'/compare_masks.py'
MODELS = 'Si SiN Mono MonoN Kurt KurtN Stretched StretchedN '\
        'Biexp BiexpN'.split()

# Common functions

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
        outfile = '{od}/{c}_{s}_auto.mask'.format(**d)
        d['i'] = infile
        d['o'] = outfile
        cmd = '{prg} -i {i} -o {o}'.format(**d)
        if not os.path.exists(infile):
            continue
        yield {
                'name': '{c}_{s}'.format(**d),
                'actions': [(create_folder, [outdir]),
                        cmd],
                'file_dep': [infile],
                'targets': [outfile],
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
        file1 = 'masks/{c}_{s}_ca.mask'.format(**d)
        file2 = 'masks_auto/{c}_{s}_auto.mask'.format(**d)
        outdir = 'roi_comparison'
        outfile = os.path.join(outdir, '{c}_{s}.txt'.format(**d))
        d['f1'] = file1
        d['f2'] = file2
        d['o'] = outfile
        cmd = '{prg} -s {w} {f1} {f2} > {o}'.format(**d)
        if not os.path.exists(file1) or not os.path.exists(file2):
            continue
        yield {
                'name': '{c}_{s}'.format(**d),
                'actions': [(create_folder, [outdir]),
                        cmd],
                'file_dep': [file1, file2],
                'targets': [outfile],
                }

SUBWINDOWS = read_subwindows('subwindows.txt')
