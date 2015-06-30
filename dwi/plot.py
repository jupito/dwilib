"""Plotting."""

import numpy as np
import pylab as pl

import dwi.util

def show_images(Imgs, ylabels=[], xlabels=[], vmin=None, vmax=None,
        outfile=None):
    """Show a grid of images. Imgs is an array of columns of rows of images."""
    pl.rcParams['image.cmap'] = 'gray'
    pl.rcParams['image.aspect'] = 'equal'
    pl.rcParams['image.interpolation'] = 'none'
    ncols, nrows = max(len(imgs) for imgs in Imgs), len(Imgs)
    fig = pl.figure(figsize=(ncols*6, nrows*6))
    for i, imgs in enumerate(Imgs):
        for j, img in enumerate(imgs):
            ax = fig.add_subplot(nrows, ncols, i*ncols+j+1)
            ax.set_title('%i, %i' % (i, j))
            if ylabels:
                ax.set_ylabel(ylabels[i])
            if xlabels:
                ax.set_xlabel(xlabels[j])
            pl.imshow(img, vmin=vmin, vmax=vmax)
    pl.tight_layout()
    if outfile:
        pl.savefig(outfile, bbox_inches='tight')
    else:
        pl.show()
    pl.close()

def plot_rocs(X, Y, params, autoflip=False, outfile=None):
    """Plot several ROCs."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    n_rows, n_cols = len(params), 1
    pl.figure(figsize=(n_cols*6, n_rows*6))
    for x, y, param, row in zip(X, Y, params, range(n_rows)):
        fpr, tpr, auc = dwi.util.calculate_roc_auc(y, x, autoflip=autoflip)
        pl.subplot2grid((n_rows, n_cols), (row, 0))
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive rate')
        pl.ylabel('True Positive rate')
        pl.title('%s' % param)
        pl.legend(loc='lower right')
    pl.tight_layout()
    if outfile:
        pl.savefig(outfile, bbox_inches='tight')
    else:
        pl.show()
    pl.close()
