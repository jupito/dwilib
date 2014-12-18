"""Automatic ROI search."""

ADCM_MIN = 0.00050680935535585281
ADCM_MAX = 0.0017784125828491648

import numpy as np

def get_score_param(img, param):
    """Return parameter score of given ROI."""
    if param.startswith('ADC'):
        #r = 1-np.mean(img)
        r = 1./(np.mean(img)-0.0008)
        #if np.min(img) < 0.0002:
        #    r = 0
        if (img < ADCM_MIN).any() or (img > ADCM_MAX).any():
            r = 0
    elif param.startswith('K'):
        r = np.mean(img)/1000
    elif param.startswith('score'):
        r = np.mean(img)
    else:
        r = 0 # Unknown parameter
    return r

def get_score(img, params):
    """Return total score of given ROI."""
    scores = [get_score_param(i, p) for i, p in zip(img.T, params)]
    r = sum(scores)
    return r

def get_roi_scores(img, d, params):
    """Return array of all scores for each possible ROI of given dimension."""
    scores_shape = tuple((img.shape[i]-d[i]+1 for i in range(3)))
    scores = np.zeros(scores_shape)
    scores.fill(np.nan)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            for k in range(scores.shape[2]):
                z = (i, i+d[0])
                y = (j, j+d[1])
                x = (k, k+d[2])
                roi = img[z[0]:z[1], y[0]:y[1], x[0]:x[1], :]
                scores[i,j,k] = get_score(roi, params)
    return scores

def get_scoremap(img, d, params, n_rois):
    """Return array like original image, with scores of n_rois best ROI's."""
    scores = get_roi_scores(img, d, params)
    indices = scores.ravel().argsort()[::-1] # Sort ROI's by descending score.
    indices = indices[0:n_rois] # Select best ones.
    indices = [np.unravel_index(i, scores.shape) for i in indices]
    scoremap = np.zeros(img.shape[0:3] + (1,))
    for z, y, x in indices:
        scoremap[z:z+d[0], y:y+d[1], x:x+d[2], 0] += scores[z,y,x]
    return scoremap

def find_roi(img, roidim, params):
    #dims = [(1,1,1)]
    dims = [(1,i,i) for i in range(5, 10)]
    #n_rois = 2000
    n_rois = 70*70/2
    scoremaps = [get_scoremap(img, d, params, n_rois) for d in dims]
    sum_scoremaps = sum(scoremaps)
    roimap = get_scoremap(sum_scoremaps, roidim, ['score'], 1)
    corner = [axis[0] for axis in roimap[...,0].nonzero()]
    coords = [(x, x+d) for x, d in zip(corner, roidim)]
    d = dict(sum_scoremaps=sum_scoremaps, roi_coords=coords)
    return d
