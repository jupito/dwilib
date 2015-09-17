#!/usr/bin/env python2

"""Produce parametric maps by fitting one or more diffusion models to imaging
data. Multiple input images can be provided in ASCII format. Single input image
can be provided as a group of DICOM files. Output is written in ASCII files
named by input and model."""

from __future__ import absolute_import, division, print_function
import os.path
import sys
import argparse

import dwi.dwimage
import dwi.models


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-v', '--verbose', action='count',
                   help='increase verbosity')
    p.add_argument('-l', '--listmodels', action='store_true',
                   help='list available models')
    p.add_argument('-a', '--average', action='store_true',
                   help='average input voxels into one')
    p.add_argument('-s', '--subwindow', metavar='I', nargs=6, default=[],
                   required=False, type=int,
                   help='use subwindow (specified by 6 one-based indices)')
    p.add_argument('-m', '--models', metavar='MODEL', nargs='+', default=[],
                   help='models to use')
    p.add_argument('-i', '--input', metavar='FILENAME', nargs='+', default=[],
                   help='input ASCII files or DICOM directories')
    p.add_argument('-o', '--output', metavar='FILENAME', required=False,
                   help='output file (for single model only)')
    return p.parse_args()


def write_pmap_ascii(img, model, params, pmap, output):
    """Write parameter images to an ASCII file."""
    if output:
        filename = output
    else:
        filename = 'pmap_%s_%s.txt' % (os.path.basename(img.filename), model)
    print('Writing parameters to %s...' % filename)
    with open(filename, 'w') as f:
        write_pmap_ascii_head(img, model, params, f)
        write_pmap_ascii_body(pmap, f)


def write_pmap_ascii_head(img, model, params, f):
    f.write('subwindow: [%s]\n' % ' '.join(str(x) for x in img.subwindow))
    f.write('number: %d\n' % img.number)
    f.write('bset: [%s]\n' % ' '.join(str(x) for x in img.bset))
    f.write('ROIslice: %s\n' % img.roislice)
    f.write('name: %s\n' % img.name)
    f.write('executiontime: %d s\n' % img.execution_time())
    f.write('description: %s %s\n' % (img.filename, repr(model)))
    f.write('model: %s\n' % model.name)
    f.write('parameters: %s\n' % ' '.join(str(x) for x in params))


def write_pmap_ascii_body(pmap, f):
    for p in pmap:
        f.write(' '.join(repr(x) for x in p) + '\n')


def log(s):
    sys.stderr.write(s)
    sys.stderr.flush()


def fit_dwi(model, img, subwindow, average, verbose, output):
    if subwindow:
        img = img.get_roi(subwindow, onebased=True)
    if verbose:
        print(img)
    # logger = log if verbose > 1 else None
    if not model.params:
        if model.name == 'Si':
            model.params = ['SI%d' % b for b in img.bset]
        elif model.name == 'SiN':
            model.params = ['SI%dN' % b for b in img.bset]
    params = model.params + ['RMSE']
    # pmap = img.fit_whole(model, log=logger, mean=average)
    pmap = img.fit(model, average=average)
    write_pmap_ascii(img, model, params, pmap, output)


def fit(model, path, subwindow, average, verbose, output):
    dwis = dwi.dwimage.load(path)
    for img in dwis:
        fit_dwi(model, img, subwindow, average, verbose, output)


def main():
    args = parse_args()

    if args.output and len(args.models) > 1:
        raise Exception('Error: one output file, several models.')

    if args.listmodels:
        for model in dwi.models.Models:
            print('{n}: {d}'.format(n=model.name, d=model.desc))
        print('{n}: {d}'.format(n='all', d='all models'))
        print('{n}: {d}'.format(n='normalized', d='all normalized models'))

    selected_models = args.models
    if 'all' in selected_models:
        selected_models += [m.name for m in dwi.models.Models]
    elif 'normalized' in selected_models:
        selected_models += 'SiN MonoN KurtN StretchedN BiexpN'.split()

    for model in dwi.models.Models:
        if model.name in selected_models:
            for path in args.input:
                fit(model, path, args.subwindow, args.average, args.verbose,
                    args.output)


if __name__ == '__main__':
    main()
