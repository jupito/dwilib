#!/usr/bin/env python2

"""Fit models to imaging data and write resulting parametric maps as ASCII
files."""

import os.path
import sys
import argparse

from dwi import dwimage
from dwi import models

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description =
            'Produce parameter maps by fitting curves.')
    p.add_argument('--listmodels', '-l', action='store_true',
            help='list models')
    p.add_argument('--models', '-m', nargs='+', default=[],
            help='models to use')
    p.add_argument('--input', '-i', nargs='+', default=[],
            help='input files')
    p.add_argument('--dicom', '-d', nargs='+', default=[],
            help='input DICOM files')
    p.add_argument('--roi', '-r', metavar='i', nargs=6, default=[],
            required=False, type=int, help='ROI (6 integers)')
    p.add_argument('--average', '-a', action='store_true',
            help='average voxels into one')
    p.add_argument('--verbose', '-v', action='count',
            help='be more verbose')
    args = p.parse_args()
    return args

def write_pmap_ascii(dwi, model, params, pmap):
    """Write parameter images to an ASCII file."""
    filename = '%s_%s.txt' % (os.path.basename(dwi.filename), model)
    print 'Writing parameters to %s...' % filename
    with open(filename, 'w') as f:
        write_pmap_ascii_head(dwi, model, params, f)
        write_pmap_ascii_body(pmap, f)

def write_pmap_ascii_head(dwi, model, params, f):
    f.write('subwindow: [%s]\n' % ' '.join(map(str, dwi.subwindow)))
    f.write('number: %d\n' % dwi.number)
    f.write('bset: [%s]\n' % ' '.join(map(str, dwi.bset)))
    f.write('ROIslice: %s\n' % dwi.roislice)
    f.write('name: %s\n' % dwi.name)
    f.write('executiontime: %d s\n' % dwi.execution_time())
    f.write('description: %s %s\n' % (os.path.basename(dwi.filename),
        repr(model)))
    f.write('model: %s\n' % model.name)
    f.write('parameters: %s\n' % ' '.join(map(str, params)))

def write_pmap_ascii_body(pmap, f):
    for p in pmap:
        f.write(' '.join(map(repr, p)) + '\n')

def log(str):
    sys.stderr.write(str)
    sys.stderr.flush()

def fit_dwi(model, dwi):
    if args.roi:
        dwi = dwi.get_roi(args.roi)
    if args.verbose:
        print dwi
    logger = log if args.verbose > 1 else None
    if not model.params:
        model.params = ['SI%dN' % b for b in dwi.bset]
    params = model.params + ['RMSE']
    pmap = dwi.fit_whole(model, log=logger, mean=args.average)
    write_pmap_ascii(dwi, model, params, pmap)

def fit_ascii(model, filename):
    dwis = dwimage.load(filename, 1)
    for dwi in dwis:
        fit_dwi(model, dwi)

def fit_dicom(model, filenames):
    dwis = dwimage.load_dicom(filenames)
    for dwi in dwis:
        fit_dwi(model, dwi)


args = parse_args()

if args.listmodels:
    for model in models.Models:
        print '{n}: {d}'.format(n=model.name, d=model.desc)
    print '{n}: {d}'.format(n='all', d='all models')
    print '{n}: {d}'.format(n='normalized', d='all normalized models')

selected_models = args.models
if 'all' in selected_models:
    selected_models += [m.name for m in models.Models]
elif 'normalized' in selected_models:
    selected_models += 'SiN MonoN KurtN StretchedN BiexpN'.split()

for model in models.Models:
    if model.name in selected_models:
        for filename in args.input:
            fit_ascii(model, filename)
        if args.dicom:
            fit_dicom(model, args.dicom)
