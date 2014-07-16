#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

# Fit and write parametric maps as ASCII files.

import os.path
import sys
import argparse
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab

import dwimage
import fit

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description =
            'Produce parameter maps by fitting curves.')
    p.add_argument('--models', '-m', nargs='+', required=True,
            help='models to use')
    p.add_argument('--input', '-i', nargs='+', required=True,
            help='input files')
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
    f.write('executiontime: %d s\n' % dwi.execution_time)
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

def fit_file(model, filename):
    dwis = dwimage.load(filename, 1)
    for dwi in dwis:
        if args.verbose:
            print dwi
        if not model.params:
            model.params = ['SI%dN' % b for b in dwi.bset]
        params = model.params + ['RMSE']
        pmap = dwi.fit_whole(model, log=None, mean=False)
        write_pmap_ascii(dwi, model, params, pmap)


args = parse_args()

selected_models = args.models
if 'all' in selected_models:
    selected_models += [m.name for m in fit.Models]
elif 'normalized' in selected_models:
    selected_models += 'SiN MonoN KurtN StretchedN BiexpN'.split()

for model in fit.Models:
    if model.name in selected_models:
        for filename in args.input:
            fit_file(model, filename)
