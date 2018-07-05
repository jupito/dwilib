"""Plotting."""

from contextlib import contextmanager
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
import pylab as pl

import dwi.files
import dwi.stats
import dwi.util

log = logging.getLogger(__name__)

# plt.rcParams['image.aspect'] = 'equal'
# plt.rcParams['image.cmap'] = 'viridis'
# plt.rcParams['image.interpolation'] = 'none'
# plt.rcParams['savefig.dpi'] = '100'


def reverse_cmap(name):
    """Return the name of the reverse version of given Matplotlib colormap."""
    if name.endswith('_r'):
        return name[:-2]
    else:
        return name + '_r'


@contextmanager
def figure(*args, **kwargs):
    """Get a figure and close it afterwards."""
    # TODO: Check if this already exists in matplotlib.
    # TODO: Add tight_layout, save/show (as separate function?).
    fig = plt.figure(*args, **kwargs)
    try:
        yield fig
    finally:
        plt.close()


def show_images(Imgs, ylabels=None, xlabels=None, vmin=None, vmax=None,
                outfile=None):
    """Show a grid of images. Imgs is an array of columns of rows of images."""
    ncols, nrows = max(len(imgs) for imgs in Imgs), len(Imgs)
    fig = pl.figure(figsize=(ncols * 6, nrows * 6))
    for i, imgs in enumerate(Imgs):
        for j, img in enumerate(imgs):
            ax = fig.add_subplot(nrows, ncols, i * ncols + j + 1)
            ax.set_title('%i, %i' % (i, j))
            if ylabels:
                ax.set_ylabel(ylabels[i])
            if xlabels:
                ax.set_xlabel(xlabels[j])
            pl.imshow(img, vmin=vmin, vmax=vmax)
    pl.tight_layout()
    if outfile is not None:
        log.info('Plotting to %s', outfile)
        pl.savefig(str(outfile), bbox_inches='tight')
    else:
        pl.show()
    pl.close()


def plot_rocs(X, Y, params=None, autoflip=False, outfile=None):
    """Plot multiple ROCs."""
    if params is None:
        params = [str(i) for i in range(len(X))]
    assert len(X) == len(Y) == len(params)
    n_rows, n_cols = len(params), 1
    pl.figure(figsize=(n_cols * 6, n_rows * 6))
    for x, y, param, row in zip(X, Y, params, range(n_rows)):
        x = dwi.stats.scale_standard(x)
        fpr, tpr, auc = dwi.stats.calculate_roc_auc(y, x, autoflip=autoflip)
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
        pl.savefig(str(outfile), bbox_inches='tight')
    else:
        pl.show()
    pl.close()


def generate_plots(nrows=1, ncols=1, titles=None, xlabels=None, ylabels=None,
                   title_kwargs=None, suptitle=None, path=None):
    """Generate subfigures, yielding each context for plotting."""
    if titles is None:
        # Invent missing titles.
        titles = [str(x) for x in range(ncols * nrows)]
    if len(titles) == 1:
        # Multiply single title.
        titles = titles * (ncols * nrows)
    # assert len(titles) == nrows * ncols
    fig = plt.figure(figsize=(ncols * 6, nrows * 6))
    if suptitle:
        fig.suptitle(suptitle)
    for i, title in enumerate(titles):
        ax = fig.add_subplot(nrows, ncols, i + 1)
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
        dwi.files.ensure_dir(path)
        plt.savefig(str(path), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def noticks(plot):
    """Show no ticks."""
    # plot.xticks([])
    # plot.yticks([])
    plot.tick_params(
        axis='both', which='both',
        bottom='off', top='off', left='off', right='off',
        labelbottom='off', labeltop='off', labelleft='off', labelright='off',
        )


def plot_grid(im, centroid, base=5, color=(1, 0, 0, 1), linestyle=(0, (1, 1)),
              linewidth=1):
    """Draw a centered grid on image, with good default style."""
    assert len(centroid) == 2, centroid
    for ax, c in zip([im.axes.yaxis, im.axes.xaxis], centroid):
        ax.set_major_locator(mpl.ticker.IndexLocator(base=base,
                                                     offset=round(c) % base))
    im.axes.grid(b=True, color=color[:3], alpha=color[-1], linestyle=linestyle,
                 linewidth=linewidth)


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    # Size matched to graph.
    # Copied from http://stackoverflow.com/questions/18195758/
    #     set-matplotlib-colorbar-size-to-match-graph
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes('right', size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
