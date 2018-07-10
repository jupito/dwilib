#!/usr/bin/python3

"""Experimentation on Visvis."""

# https://github.com/almarklein/visvis/wiki/Visvis_basics
# /usr/lib/python3.6/site-packages/visvis/examples/surfaceFromRandomPoints.py

import logging

# import numpy as np
import visvis as vv

from dwi.files import Path
from dwi.image import Image
from dwi import util


def plot(image):
    # ax = vv.gca()
    # ms = vv.Mesh(ax)
    logging.warning([image.shape, image.spacing])
    vol = image[:, :, :, 0]
    logging.warning([vol.min(), vol.max()])
    vol = util.normalize(vol, 'ADCm')
    logging.warning([vol.min(), vol.max()])
    vol = vv.Aarray(vol, image.spacing)

    cmap = None
    # cmap = vv.CM_VIRIDIS
    render_style = 'mip'
    # render_style = 'iso'
    # render_style = 'ray'
    # render_style = 'edgeray'
    # render_style = 'litray'

    vv.figure()
    vv.xlabel('x axis')
    vv.ylabel('y axis')
    vv.zlabel('z axis')

    a1 = vv.subplot(111)
    t1 = vv.volshow(vol, cm=cmap, renderStyle=render_style)
    t1.isoThreshold = 0.7
    vv.title(render_style)

    # a1.camera = a2.camera = a3.camera
    vv.ColormapEditor(a1)


def main():
    p = Path('/mri/images/DWI-Mono/42-1a/42-1a_ADCm.zip')
    image = Image.read(p, dtype='float32')
    image = image[image.mbb()]

    app = vv.use()
    # vv.figure()
    # vv.title(p.name)
    plot(image)
    app.Run()


if __name__ == '__main__':
    main()
