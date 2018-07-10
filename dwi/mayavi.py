"""Mayavi stuff."""

# http://docs.enthought.com/mayavi/mayavi/auto/example_mri.html

import logging

import numpy as np
from mayavi import mlab

import dwi.files
import dwi.image
import dwi.util
from dwi.files import Path


def transform_axes(pmap):
    axes = (0, 2, 1)
    pmap = np.transpose(pmap, axes=axes)
    # spacing = pmap.info['attrs']['voxel_spacing']
    spacing = pmap.spacing
    # pmap.info['attrs']['voxel_spacing'] = [spacing[x] for x in axes]
    pmap.spacing = [spacing[x] for x in axes]
    return pmap


def read_image(path):
    img = dwi.image.Image.read(path, params=[0], dtype=np.float32)[..., 0]
    return img


def read_mask(path):
    img = read_image(path)
    img = img.clip(0, 1, out=img)
    return img


def show_mask(masks, images):
    # bb = np.logical_or(*masks).mbb(100)
    bb = masks[0].mbb(100)
    masks = [x[bb] for x in masks]
    images = [x[bb] for x in images]

    logging.warning('### %s', [np.count_nonzero(x) for x in masks])

    fig = mlab.figure()

    for i, a in enumerate(masks):
        src = mlab.pipeline.scalar_field(a)
        src.name += ': {}'.format(Path(a.info['path']).name)
        logging.warning(src.name)
        # src.spacing = a.info['attrs']['voxel_spacing']
        src.spacing = a.spacing
        src.update_image_data = True
        blur = mlab.pipeline.user_defined(src, filter='ImageGaussianSmooth')
        voi = mlab.pipeline.extract_grid(blur)
        d = dict(contours=[0.3])
        if i == 0:
            d.update(color=(0, 0, 1), opacity=0.2)
        elif i == 1:
            d.update(color=(1, 1, 0), opacity=0.4)
        else:
            d.update(color=(1, 0, 0), opacity=0.8)
        mlab.pipeline.iso_surface(voi, **d)

    for a in images:
        src = mlab.pipeline.scalar_field(a)
        src.name += ': {}'.format(a.info['path'])
        # src.spacing = a.info['attrs']['voxel_spacing']
        src.spacing = a.spacing
        src.update_image_data = True
        blur = mlab.pipeline.user_defined(src, filter='ImageGaussianSmooth')
        voi = mlab.pipeline.extract_grid(blur)
        d = dict(contours=5, colormap='viridis', opacity=0.3)
        mlab.pipeline.iso_surface(voi, **d)

    mlab.view(0, 0)
    mlab.orientation_axes(figure=fig)
    mlab.show()
