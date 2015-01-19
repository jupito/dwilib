"""Automatic ROI search."""

ADCM_MIN = 0.00050680935535585281
ADCM_MAX = 0.0017784125828491648

import numpy as np

import dwi.util

def get_score_param(img, param):
    """Return parameter score of given ROI."""
    if param.startswith('ADC'):
        #r = 1-np.mean(img)
        #r = 1./(np.mean(img)-0.0008)
        r = 1./np.mean(img)
        #if np.min(img) < 0.0002:
        #    r = 0
        #if (img < ADCM_MIN).any() or (img > ADCM_MAX).any():
        #    r = 0
    elif param.startswith('K'):
        r = np.mean(img)/1000
    elif param.startswith('score'):
        r = np.mean(img)
    elif param == 'prostate_mask':
        # Ban areas more than a certain amount outside of prostate.
        if float(img.sum())/img.size > 0.20:
            r = 0
        else:
            r = -np.inf
    else:
        r = 0 # Unknown parameter
    return r

def get_score(img, params):
    """Return total score of given ROI."""
    scores = [get_score_param(img[...,i], p) for i, p in enumerate(params)]
    r = sum(scores)
    return r

def get_roi_scores(img, d, params):
    """Return array of all scores for each possible ROI of given dimension."""
    scores_shape = tuple([img.shape[i]-d[i]+1 for i in range(3)])
    scores = np.zeros(scores_shape)
    scores.fill(np.nan)
    for z, y, x in np.ndindex(scores.shape):
        roi = img[z:z+d[0], y:y+d[1], x:x+d[2], :]
        scores[z,y,x] = get_score(roi, params)
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

def find_roi(img, roidim, params, prostate_mask=None, n_rois=1000,
        siderange=(5, 10)):
    #dims = [(1,1,1)]
    dims = [(2,i,i) for i in range(*siderange)]
    dims += [(3,i,i) for i in range(*siderange)]
    #dims = dwi.util.combinations([range(2, 4), range(*siderange), range(*siderange)])
    #print dims
    if prostate_mask:
        # Add mask to image as an extra parameter.
        mask = prostate_mask.array.view()
        mask.shape += (1,)
        img = np.concatenate((img, mask), axis=3)
        params = params + ['prostate_mask']
    scoremaps = [get_scoremap(img, d, params, n_rois) for d in dims]
    sum_scoremaps = sum(scoremaps)
    roimap = get_scoremap(sum_scoremaps, roidim, ['score'], 1)
    # Get first nonzero position at each axis.
    corner = [axis[0] for axis in roimap[...,0].nonzero()]
    # Convert to [(start, stop), ...] notation.
    coords = [(x, x+d) for x, d in zip(corner, roidim)]
    d = dict(scoremap=sum_scoremaps[...,0], roi_corner=corner,
            roi_coords=coords)
    return d
