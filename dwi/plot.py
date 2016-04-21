"""Plotting."""

from __future__ import absolute_import, division, print_function
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab as pl

import dwi.util


log = logging.getLogger(__name__)


# plt.rcParams['image.aspect'] = 'equal'
plt.rcParams['image.cmap'] = 'viridis'
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['savefig.dpi'] = '100'


def show_images(Imgs, ylabels=None, xlabels=None, vmin=None, vmax=None,
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
    if outfile is not None:
        log.info('Plotting to %s', outfile)
        pl.savefig(outfile, bbox_inches='tight')
    else:
        pl.show()
    pl.close()


def plot_rocs(X, Y, params=None, autoflip=False, outfile=None):
    """Plot multiple ROCs."""
    if params is None:
        params = [str(i) for i in range(len(X))]
    assert len(X) == len(Y) == len(params)
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
    if outfile is not None:
        log.info('Plotting to %s', outfile)
        pl.savefig(outfile, bbox_inches='tight')
    else:
        pl.show()
    pl.close()


def generate_plots(nrows=1, ncols=1, titles=None, xlabels=None, ylabels=None,
                   title_kwargs=None, path=None):
    """Generate subfigures, yielding each context for plotting."""
    if titles is None:
        # Invent missing titles.
        titles = [str(x) for x in range(ncols * nrows)]
    if len(titles) == 1:
        # Multiply single title.
        titles = titles * (ncols * nrows)
    # assert len(titles) == nrows * ncols
    fig = plt.figure(figsize=(ncols*6, nrows*6))
    for i, title in enumerate(titles):
        ax = fig.add_subplot(nrows, ncols, i+1)
        if title is not None:
            ax.set_title(title, **(title_kwargs or {}))
        if xlabels is not None:
            ax.set_xlabel(xlabels[i])
        if ylabels is not None:
            ax.set_ylabel(ylabels[i])
        yield plt
    plt.tight_layout()
    if path is not None:
        log.info('Plotting to %s', path)
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_grid(im, centroid, base=5, color=(1, 0, 0, 1), linestyle=(0, (1, 1)),
              linewidth=1):
    """Draw a centered grid on image, with good default style."""
    assert len(centroid) == 2, centroid
    for ax, c in zip([im.axes.yaxis, im.axes.xaxis], centroid):
        ax.set_major_locator(mpl.ticker.IndexLocator(base=base,
                                                     offset=round(c) % base))
    im.axes.grid(b=True, color=color[:3], alpha=color[-1], linestyle=linestyle,
                 linewidth=linewidth)
