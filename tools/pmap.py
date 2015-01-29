#!/usr/bin/env python2

"""Produce parametric maps by fitting one or more diffusion models to imaging
data. Multiple input images can be provided in ASCII format. Single input image
can be provided as a group of DICOM files. Output is written in ASCII files
named by input and model."""

import os.path
import sys
import argparse

from dwi import dwimage
from dwi import models

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument('-v', '--verbose',
            action='count',
            help='increase verbosity')
    p.add_argument('-l', '--listmodels',
            action='store_true',
            help='list available models')
    p.add_argument('-a', '--average',
            action='store_true',
            help='average input voxels into one')
    p.add_argument('-s', '--subwindow', metavar='I',
            nargs=6, default=[], required=False, type=int,
            help='use subwindow (specified by 6 one-based indices)')
    p.add_argument('-m', '--models', metavar='MODEL',
            nargs='+', default=[],
            help='models to use')
    p.add_argument('-i', '--input', metavar='FILENAME',
            nargs='+', default=[],
            help='input ASCII files')
    p.add_argument('-d', '--dicom', metavar='PATHNAME',
            nargs='+', default=[],
            help='input DICOM files or directories')
    p.add_argument('-o', '--output', metavar='FILENAME',
            required=False,
            help='output file (for single model only)')
    args = p.parse_args()
    return args

def write_pmap_ascii(dwi, model, params, pmap):
    """Write parameter images to an ASCII file."""
    if args.output:
        filename = args.output
    else:
        filename = 'pmap_%s_%s.txt' % (os.path.basename(dwi.filename), model)
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
    f.write('description: %s %s\n' % (dwi.filename, repr(model)))
    f.write('model: %s\n' % model.name)
    f.write('parameters: %s\n' % ' '.join(map(str, params)))

def write_pmap_ascii_body(pmap, f):
    for p in pmap:
        f.write(' '.join(map(repr, p)) + '\n')

def log(str):
    sys.stderr.write(str)
    sys.stderr.flush()

def fit_dwi(model, dwi):
    if args.subwindow:
        dwi = dwi.get_roi(args.subwindow, onebased=True)
    if args.verbose:
        print dwi
    logger = log if args.verbose > 1 else None
    if not model.params:
        model.params = ['SI%dN' % b for b in dwi.bset]
    params = model.params + ['RMSE']
    #pmap = dwi.fit_whole(model, log=logger, mean=args.average)
    pmap = dwi.fit(model, average=args.average)
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

if args.output and len(args.models) > 1:
    raise 'Error: one output file, several models.'

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
