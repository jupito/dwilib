#!/usr/bin/python3

"""Produce parametric maps by fitting one or more models to imaging data."""

import argparse

import numpy as np

import dwi.files
import dwi.mask
import dwi.models


def parse_args(models):
    """Parse command-line arguments."""
    formatter = argparse.RawDescriptionHelpFormatter
    p = argparse.ArgumentParser(description=__doc__, formatter_class=formatter,
                                epilog='Available models:\n'+'\n'.join(models))
    p.add_argument('-v', '--verbose', action='count',
                   help='increase verbosity')
    p.add_argument('--input', metavar='PATH', required=True,
                   help='input image file or DICOM directory')
    p.add_argument('--output', metavar='PATH', required=True,
                   help='output pmap file')
    p.add_argument('--params', type=int, nargs='+',
                   help='included parameter indices')
    p.add_argument('--average', action='store_true',
                   help='average input voxels into one')
    p.add_argument('--mask',
                   help='mask file')
    p.add_argument('--subwindow', metavar='I', nargs=6, type=int,
                   help='use subwindow (specified by 6 one-based indices)')
    p.add_argument('--mbb', metavar='I', nargs=3, type=int,
                   help='use minimum bounding box around mask '
                   'with padding on three axes')
    p.add_argument('--model', required=True,
                   help='model to use')
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
    models = ['{n}: {d}'.format(n=x.name, d=x.desc) for x in dwi.models.Models]
    args = parse_args(models)

    model = [x for x in dwi.models.Models if x.name == args.model][0]

    image, attrs = dwi.files.read_pmap(args.input, params=args.params)
    assert image.ndim == 4, image.ndim
    if args.verbose:
        print('Read image', image.shape, image.dtype, args.input)
        print('Parameters', attrs['parameters'])
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
            attrs['mbb'] = args.mbb
        image = mask.apply_mask(image, value=np.nan)
        attrs['mask'] = args.mask
    if args.subwindow:
        if args.verbose:
            print('Using subwindow', args.subwindow)
        # image = dwi.util.crop_image(image, args.subwindow, onebased=True)
        image = dwi.util.select_subwindow(image, args.subwindow, onebased=True)
        print(image.shape, np.count_nonzero(np.isnan(image)))
        attrs['subwindow'] = args.subwindow
    if args.average:
        image = np.mean(image, axis=(0, 1, 2), keepdims=True)

    if model.name == 'T2':
        image, attrs = fix_T2(image, attrs)
    if args.verbose:
        n = np.count_nonzero(-np.isnan(image[..., 0]))
        print('Fitting {m} to {n} voxels'.format(m=model.name, n=n))
        print('Guesses:', [len(p.guesses(1)) for p in model.params])
    timepoints = get_timepoints(model, attrs)
    params = get_params(model, timepoints)
    pmap = fit(image, timepoints, model)
    d = dict(attrs)
    d.update(parameters=params, source=args.input, model=model.name,
             description=repr(model))
    dwi.files.write_pmap(args.output, pmap, d)
    if args.verbose:
        print('Wrote', pmap.shape, pmap.dtype, args.output)


if __name__ == '__main__':
    main()
