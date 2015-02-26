"""Plotting."""

import pylab as pl

def show_images(Imgs, ylabels=[], xlabels=[], outfile=None):
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
            pl.imshow(img)
    pl.tight_layout()
    if outfile:
        pl.savefig(outfile, bbox_inches='tight')
    else:
        pl.imshow(img)
        pl.show()
