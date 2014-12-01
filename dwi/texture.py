"""Texture analysis."""

import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage.util import view_as_windows

PROPNAMES = ['contrast', 'dissimilarity', 'homogeneity', 'energy',
        'correlation', 'ASM']

def get_coprops(windows):
    props = np.zeros((len(windows), len(PROPNAMES)))
    for i, win in enumerate(windows):
        #win = skimage.img_as_ubyte(win)
        if win.min() == 0:
            props[i].fill(0)
            continue
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = greycomatrix(win, distances, angles, 256, symmetric=True,
                normed=True)
        for j, propname in enumerate(PROPNAMES):
            a = greycoprops(glcm, propname)
            props[i,j] = np.mean(a)
    return props

def get_texture_pmap(img, win_step):
    pmap = np.zeros((len(PROPNAMES)+1, img.shape[0]/win_step+1,
        img.shape[1]/win_step+1))
    windows = view_as_windows(img, (5,5), step=win_step)
    #props = get_coprops(windows.reshape(-1,5,5))
    #props.shape = (windows.shape[0], windows.shape[1], len(PROPNAMES))
    for i, row in enumerate(windows):
        for j, win in enumerate(row):
            if win.min() > 0:
                v = get_coprops([win])[0]
            else:
                v = 0
            pmap[0,i,j] = np.median(win)
            pmap[1:,i,j] = v
    return pmap
