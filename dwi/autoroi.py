"""Automatic ROI search."""

import numpy as np

from .types import AlgParams

ADCM_MIN = 0.00050680935535585281
ADCM_MAX = 0.0017784125828491648


def get_score_param(img, param):
    """Return parameter score of given ROI."""
    if param.startswith('ADC'):
        # return 1 - np.mean(img)
        # return 1 / (np.mean(img) - 0.0008)
        if np.mean(img) > 0:
            return 1 / np.mean(img)
        return 0
        # NOTE The following min/max limit seems to make things worse.
        # if (img < ADCM_MIN).any() or (img > ADCM_MAX).any():
        #     return 0
    elif param.startswith('K'):
        return np.mean(img) / 1000
    elif param.startswith('score'):
        return np.mean(img)
    elif param == 'prostate_mask':
        # Ban areas more than a certain amount outside of prostate.
        if img.sum() / img.size > 0.20:
            return 1
        return -1e20
    elif param == 'prostate_mask_strict':
        # Ban areas even partly outside of prostate.
        if img.all():
            return 1
        return -1e20
    return 0  # Unknown parameter


def get_roi_scores(img, d, params):
    """Return array of all scores for each possible ROI of given dimension."""
    shape = [img.shape[i]-d[i]+1 for i in range(3)] + [len(params)]
    scores = np.empty(shape, dtype=np.float32)
    for z, y, x, i in np.ndindex(scores.shape):
        roi = img[z:z+d[0], y:y+d[1], x:x+d[2], i]
        scores[z, y, x, i] = get_score_param(roi, params[i])
    return scores


def scale_scores(scores):
    """Scale scores in-place."""
    scores[...] /= scores[...].max()
    # import sklearn.preprocessing
    # shape = scores.shape
    # a = scores.ravel()
    # sklearn.preprocessing.scale(a, copy=False)
    # a.shape = shape
    # scores[...] = a


def get_scoremap(img, d, params, nrois):
    """Return array like original image, with scores of nrois best ROI's."""
    scores = get_roi_scores(img, d, params)
    for i in range(len(params)):
        scale_scores(scores[..., i])
    scores = np.sum(scores, axis=-1)  # Sum scores parameter-wise.
    indices = scores.ravel().argsort()  # Sort ROI's by score.
    indices = indices[-nrois:]  # Select best ones.
    indices = [np.unravel_index(i, scores.shape) for i in indices]
    scoremap = np.zeros(img.shape[0:3] + (1,), dtype=np.float32)
    for z, y, x in indices:
        scoremap[z:z+d[0], y:y+d[1], x:x+d[2], 0] += scores[z, y, x]
    return scoremap


def add_mask(img, mask):
    """Add mask to image as an extra parameter."""
    m = mask.array.view()
    m.shape += (1,)
    return np.concatenate((img, m), axis=3)


def find_roi(img, roidim, params, prostate_mask=None, ap=None):
    if ap is None:
        ap = AlgParams(depthmin=2, depthmax=3, sidemin=10, sidemax=10,
                       nrois=500)
    assert ap.depthmin <= ap.depthmax
    assert ap.sidemin <= ap.sidemax

    # Draw score map.
    dims = [(j, i, i) for i in range(ap.sidemin, ap.sidemax+1)
            for j in range(ap.depthmin, ap.depthmax+1)]
    if prostate_mask:
        img = add_mask(img, prostate_mask)
        params = params + ['prostate_mask']
    scoremaps = [get_scoremap(img, d, params, ap.nrois) for d in dims]
    scoremap = sum(scoremaps)

    # Find optimal ROI.
    scoremap_params = ['score']
    if prostate_mask:
        scoremap = add_mask(scoremap, prostate_mask)
        scoremap_params += ['prostate_mask_strict']
    roimap = get_scoremap(scoremap, roidim, scoremap_params, 1)

    # Get first nonzero position at each axis.
    corner = [axis[0] for axis in roimap[..., 0].nonzero()]
    # Convert to [(start, stop), ...] notation.
    coords = [(x, x+d) for x, d in zip(corner, roidim)]

    return dict(algparams=ap, scoremap=scoremap[..., 0], roi_corner=corner,
                roi_coords=coords)
