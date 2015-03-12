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
        if np.mean(img) > 0:
            r = 1./np.mean(img)
        else:
            r = 0
        # NOTE The following min/max limit seems to make things worse.
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
    elif param == 'prostate_mask_strict':
        # Ban areas even partly outside of prostate.
        if img.all():
            r = 0
        else:
            r = -np.inf
    else:
        r = 0 # Unknown parameter
    return r

def get_roi_scores(img, d, params):
    """Return array of all scores for each possible ROI of given dimension."""
    shape = [img.shape[i]-d[i]+1 for i in range(3)] + [len(params)]
    scores = np.empty(shape)
    for z, y, x, i in np.ndindex(scores.shape):
        roi = img[z:z+d[0], y:y+d[1], x:x+d[2], i]
        scores[z,y,x,i] = get_score_param(roi, params[i])
    return scores

def get_scoremap(img, d, params, n_rois):
    """Return array like original image, with scores of n_rois best ROI's."""
    scores = get_roi_scores(img, d, params)
    scores = np.sum(scores, axis=-1) # Sum scores parameter-wise.
    indices = scores.ravel().argsort() # Sort ROI's by score.
    indices = indices[-n_rois:] # Select best ones.
    indices = [np.unravel_index(i, scores.shape) for i in indices]
    scoremap = np.zeros(img.shape[0:3] + (1,))
    for z, y, x in indices:
        scoremap[z:z+d[0], y:y+d[1], x:x+d[2], 0] += scores[z,y,x]
    return scoremap

def add_mask(img, mask):
    """Add mask to image as an extra parameter."""
    m = mask.array.view()
    m.shape += (1,)
    img = np.concatenate((img, m), axis=3)
    return img

def find_roi(img, roidim, params, prostate_mask=None, sidemin=5, sidemax=10,
        n_rois=1000):
    assert sidemin <= sidemax

    # Draw score map.
    dims = [(2,i,i) for i in range(sidemin, sidemax+1)]
    dims += [(3,i,i) for i in range(sidemin, sidemax+1)]
    #dims = dwi.util.combinations([range(2, 4), range(*siderange), range(*siderange)])
    #print dims
    if prostate_mask:
        img = add_mask(img, prostate_mask)
        params = params + ['prostate_mask']
    scoremaps = [get_scoremap(img, d, params, n_rois) for d in dims]
    scoremap = sum(scoremaps)

    # Find optimal ROI.
    scoremap_params = ['score']
    if prostate_mask:
        scoremap = add_mask(scoremap, prostate_mask)
        scoremap_params += ['prostate_mask_strict']
    roimap = get_scoremap(scoremap, roidim, scoremap_params, 1)

    # Get first nonzero position at each axis.
    corner = [axis[0] for axis in roimap[...,0].nonzero()]
    # Convert to [(start, stop), ...] notation.
    coords = [(x, x+d) for x, d in zip(corner, roidim)]

    d = dict(scoremap=scoremap[...,0], roi_corner=corner, roi_coords=coords)
    return d
