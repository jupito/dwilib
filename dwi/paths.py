"""Common path names."""

# TODO: Get rid of messy globbing by by explicit data file naming.
# TODO: Move all functions to a class.

from __future__ import absolute_import, division, print_function

from .types import ImageMode, Path


def _fmt_tspec(tspec):
    parts = filter(None, [tspec.method, tspec.winsize, tspec.feature])
    return '-'.join(str(x) for x in parts)


class Paths(object):
    def __init__(self, mode):
        self.mode = ImageMode(mode)

    def samplelist(self, samplelist):
        """Return path to samplelist."""
        d = dict(m=self.mode, l=samplelist)
        return Path('patients') / 'patients_{m[0]}_{l}.txt'.format(**d)

    def pmap(self, case=None, scan=None, fmt='dicom'):
        """Return path to pmap."""
        if 'std' in self.mode or self.mode == 'T2-fitted':
            fmt = 'h5'  # TODO: Temporary redirection.
        d = dict(m=self.mode, c=case, s=scan)
        path = 'images/{m[:2]}'
        if fmt == 'h5':
            if case is not None and scan is not None:
                path += '/{c}-{s}.h5'
            return Path(path.format(**d))
        elif fmt == 'dicom':
            if case is not None and scan is not None:
                if self.mode == 'DWI':
                    path += '/{c}_hB_{s}.zip'
                elif len(self.mode) == 1:
                    path += '/{c}_{s}*'
                else:
                    path += '/{c}_*_{s}/{c}_*_{s}*_{m[2]}.zip'
            path, = Path().glob(path.format(**d))
            return path
        else:
            raise ValueError('Unknown format: {}'.format(fmt))

    def subregion(self, case=None, scan=None):
        """Return path to subregion file. XXX: Obsolete."""
        d = dict(m=self.mode, c=case, s=scan)
        path = Path('subregions') / '{m[0]}'.format(**d)
        if case is not None and scan is not None:
            path /= '{c}_{s}_subregion10.txt'.format(**d)
        return path

    def mask(self, masktype, case, scan, lesion=None, algparams=()):
        """Return path and deps of masks of different types."""
        if masktype == 'all':
            return None
        d = dict(m=self.mode, mt=masktype, c=case, s=scan, l=lesion,
                 ap_='_'.join(algparams))
        do_glob = True
        path = 'masks'
        if masktype == 'prostate':
            path += '/{mt}/{m[0]}/{c}_*_{s}*.h5'
        elif masktype == 'lesion':
            path += '/{mt}/{m[0]}/lesion{l}/{c}_*{s}_{m[0]}.h5'
        elif masktype in ('CA', 'N'):
            path += '/roi/{m[0]}/{c}_*_{s}_D_{mt}.h5'
        elif masktype == 'auto':
            path += '/{mt}/{m}/{ap_}/{c}_{s}_auto.mask'
            do_glob = False  # Don't require existence, can be generated.
        else:
            raise Exception('Unknown mask type: {mt}'.format(**d))
        path = path.format(**d)
        if do_glob:
            path, = Path().glob(path)
        return Path(path)

    def roi(self, masktype, case=None, scan=None, lesion=None, algparams=()):
        """Return whole ROI path or part of it."""
        if masktype == 'image':
            return self.pmap(case, scan)  # No ROI, but the whole image.
        d = dict(m=self.mode, mt=masktype, c=case, s=scan, l=lesion)
        path = Path('rois/{mt}/{m}'.format(**d))
        if algparams:
            path /= '_'.join(algparams)
        if case is not None and scan is not None:
            if masktype == 'prostate':
                s = '{c}-{s}.h5'
            elif masktype == 'lesion':
                s = '{c}-{s}-{l}.h5'
            else:
                s = '{c}_x_x_{s}_{m}_{mt}.txt'
            path /= s.format(**d)
        return path

    def texture(self, case, scan, lesion, masktype, slices, portion, tspec,
                algparams=(), voxel='mean'):
        """Return path to texture file."""
        if tspec is not None and tspec.method == 'raw':
            # 'Raw' texture method is actually just the source image.
            return self.pmap(case, scan)
        d = dict(m=self.mode, c=case, s=scan, l=lesion, mt=masktype,
                 slices=slices, portion=portion, tspec=tspec, vx=voxel)
        if voxel == 'all':
            suffix = '.h5'
        else:
            suffix = 'txt'
        path = Path('texture_{mt}/{m}_{slices}_{portion}_{vx}'.format(**d))
        if masktype == 'auto':
            path /= '_'.join(algparams)
        path /= '{c}_{s}_{l}'.format(**d)
        if tspec is not None:
            path /= '_{tspec.method}-{tspec.winsize}'.format(**d)
        return path.with_suffix(suffix)

    def std_cfg(self):
        """Return path to standardization configuration file."""
        return Path('stdcfg_{m}.txt'.format(m=self.mode))

    def histogram(self, roi, samplelist):
        """Return path to histogram plot."""
        return Path('histograms/{m}_{r}_{s}.png'.format(m=self.mode, r=roi,
                                                        s=samplelist))

    def grid(self, case, scan, mt, tspec, fmt='txt'):
        """Return path to the first of the grid files.

        FIXME: The first element in the resulting tuple no more exists as file.
        """
        path = Path('grid_{mt}/{m}'.format(mt=mt, m=self.mode))
        if tspec is not None:
            path /= _fmt_tspec(tspec)
        if case is not None and scan is not None:
            path /= '{c}-{s}.{f}'.format(c=case, s=scan, f=fmt)
        target = '{r}-0{e}'.format(r=path.stem, e=path.suffix)
        return path, target


# The rest are for compatibility.


def samplelist_path(mode, samplelist):
    return str(Paths(mode).samplelist(samplelist))


def pmap_path(mode, **kwargs):
    return str(Paths(mode).pmap(**kwargs))


def mask_path(mode, masktype, case, scan, **kwargs):
    return str(Paths(mode).mask(masktype, case, scan, **kwargs))


def roi_path(mode, masktype, **kwargs):
    return str(Paths(mode).roi(masktype, **kwargs))


def texture_path(mode, case, scan, lesion, masktype, slices, portion, tspec,
                 **kwargs):
    return str(Paths(mode).texture(case, scan, lesion, masktype, slices,
                                   portion, tspec, **kwargs))
