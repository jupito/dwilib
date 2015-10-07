#!/usr/bin/env python2

"""Produce parametric maps by fitting one or more diffusion models to imaging
data. Multiple input images can be provided in ASCII format. Single input image
can be provided as a group of DICOM files. Output is written in ASCII files
named by input and model."""

from __future__ import absolute_import, division, print_function
import os.path
import argparse

import numpy as np

import dwi.dwimage
import dwi.files
import dwi.mask
import dwi.models


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-v', '--verbose', action='count',
                   help='increase verbosity')
    p.add_argument('--listmodels', action='store_true',
                   help='list available models')
    p.add_argument('--average', action='store_true',
                   help='average input voxels into one')
    p.add_argument('--mask',
                   help='mask file')
    p.add_argument('--subwindow', metavar='I', nargs=6, type=int,
                   help='use subwindow (specified by 6 one-based indices)')
    p.add_argument('--mbb', metavar='I', nargs=3, type=int,
                   help='use minimum bounding box around mask '
                   'with padding on three axes')
    p.add_argument('-m', '--models', metavar='MODEL', nargs='+', default=[],
                   help='models to use')
    p.add_argument('-i', '--input', metavar='PATH', nargs='+', default=[],
                   help='input files (or DICOM directories)')
    p.add_argument('-o', '--output', metavar='PATH',
                   help='output file (for single model and input only)')
    return p.parse_args()


def fit(image, timepoints, model):
    """Fit model to image."""
    shape = image.shape[:-1]
    image = image.reshape(-1, len(timepoints))
    assert len(timepoints) == len(image[0]), len(image[0])
    # self.start_execution()
    pmap = model.fit(timepoints, image)
    # self.end_execution()
    pmap.shape = shape + (pmap.shape[-1],)
    return pmap


def get_timepoints(model, attrs):
    """Get timepoints to use."""
    if model.name == 'T2':
        timepoints = attrs['echotimes']
    else:
        timepoints = attrs['bset']
    return timepoints


def fix_T2(image, attrs):
    # There may be an already fitted fake 'zero echo time.'
    if attrs['echotimes'][0] == 0:
        attrs['echotimes'] = attrs['echotimes'][1:]
        image = image[..., 1:]
    assert 0 not in attrs['echotimes'], attrs['echotimes']
    return image, attrs


def get_params(model, timepoints):
    """Get model parameters."""
    if model.params:
        params = [str(x) for x in model.params]
    else:
        if model.name == 'Si':
            model.params = ['SI%d' % x for x in timepoints]
        elif model.name == 'SiN':
            model.params = ['SI%dN' % x for x in timepoints]
        else:
            raise ValueError('Unknown model parameters {}'.format(model))
    params.append('RMSE')
    return params


def main():
    args = parse_args()

    if args.output:
        if len(args.models) > 1 or len(args.input) > 1:
            raise ValueError('Error: one output, several models or inputs.')

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
    models = [x for x in dwi.models.Models if x.name in selected_models]

    for inpath in args.input:
        image, attrs = dwi.files.read_pmap(inpath)
        assert image.ndim == 4, image.ndim
        if args.verbose:
            print('Read image', image.shape, image.dtype)
        if args.mask:
            if args.verbose:
                print('Applying mask', args.mask)
            mask = dwi.mask.read_mask(args.mask)
            if args.mbb:
                mbb = mask.bounding_box(args.mbb)
                if args.verbose:
                    print('Using minimum bounding box {m}'.format(m=mbb))
                z, y, x = [slice(*t) for t in mbb]
                mask.array[z, y, x] = True
            image = mask.apply_mask(image, value=np.nan)
            attrs['mask'] = args.mask
        if args.subwindow:
            if args.verbose:
                print('Using subwindow', args.subwindow)
            # image = dwi.util.crop_image(image, args.subwindow, onebased=True)
            image = dwi.util.select_subwindow(image, args.subwindow,
                                              onebased=True)
            print(image.shape, np.count_nonzero(np.isnan(image)))
            attrs['subwindow'] = args.subwindow
        if args.average:
            image = np.mean(image, axis=(0, 1, 2), keepdims=True)
        for model in models:
            if model.name == 'T2':
                image, attrs = fix_T2(image, attrs)
            if args.verbose:
                n = np.count_nonzero(-np.isnan(image[..., 0]))
                print('Fitting {m} to {n} voxels'.format(m=model.name, n=n))
            timepoints = get_timepoints(model, attrs)
            params = get_params(model, timepoints)
            outpath = (args.output or
                       'pmap_{i}_{m}.txt'.format(i=os.path.basename(inpath),
                                                 m=model.name))
            pmap = fit(image, timepoints, model)
            d = dict(attrs)
            d.update(model=model.name, parameters=params)
            dwi.files.write_pmap(outpath, pmap, d)
            if args.verbose:
                print('Wrote', outpath)


if __name__ == '__main__':
    main()
