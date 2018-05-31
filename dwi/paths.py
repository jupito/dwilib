"""Common path names."""

from .types import AlgParams, ImageMode, ImageTarget, Path


class Paths(object):
    def __init__(self, mode, base='.'):
        self.mode = ImageMode(mode)
        self.base = Path(base)  # TODO: The `work` directory now, later `/mri`.

    @property
    def work(self):
        return self.base / 'work'

    def samplelist(self, samplelist):
        """Return path to samplelist."""
        d = dict(m=self.mode, sl=samplelist)
        return self.work / 'patients' / 'patients_{m[0]}_{sl}.txt'.format(**d)

    def pmap(self, case=None, scan=None, fmt='dicom'):
        """Return path to pmap."""
        if 'std' in self.mode or self.mode == 'T2-fitted':
            fmt = 'h5'  # TODO: Temporary redirection.
        trg = ImageTarget(case, scan, None)
        d = dict(m=self.mode, t=trg)
        path = self.base / 'images' / str(self.mode[:2])
        if any(trg):
            if fmt == 'h5':
                s = '{t}.h5'
            elif fmt == 'dicom':
                if self.mode in ['DWI', 'T2', 'T2w']:
                    s = '{t}.zip'
                else:
                    s = '{t}/{t}_{m[2]}.zip'
            else:
                raise ValueError('Unknown format: {}'.format(fmt))
            path /= s.format(**d)
        return path

    def subregion(self, case=None, scan=None):
        """Return path to subregion file. XXX: Obsolete."""
        d = dict(c=case, s=scan)
        path = self.work / 'subregions' / str(self.mode[0])
        if case is not None and scan is not None:
            path /= '{c}_{s}_subregion10.txt'.format(**d)
        return path

    def mask(self, masktype, case, scan, lesion=None, algparams=()):
        """Return path and deps of masks of different types."""
        if masktype == 'all':
            return None
        d = dict(m=self.mode, mt=masktype, c=case, s=scan, l=lesion)
        path = self.base / 'masks'
        if d['m'][0] == 'DWI':
            d['m'] = ImageMode(['DWI_hB'] + list(d['m'][1:]))
        if masktype == 'prostate':
            s = '{mt}/{m[0]}/{c}-{s}.h5'
        elif masktype == 'lesion':
            s = '{mt}/{m[0]}/lesion{l}/{c}-{s}.h5'
        elif masktype in ['CA', 'N']:
            s = 'roi/{m[0]}/{c}-{s}_{mt}.h5'
        elif masktype == 'auto':
            d['ap'] = AlgParams(*algparams)
            s = '{mt}/{m}/{ap}/{c}-{s}_auto.mask'
        else:
            raise Exception('Unknown mask type: {mt}'.format(**d))
        return path / s.format(**d)

    def roi(self, masktype, case=None, scan=None, lesion=None, algparams=()):
        """Return whole ROI path or part of it."""
        if masktype == 'image':
            return self.pmap(case, scan)  # No ROI, but the whole image.
        d = dict(m=self.mode, mt=masktype, c=case, s=scan, l=lesion)
        path = self.work / 'rois' / masktype / str(self.mode)
        if algparams:
            path /= str(AlgParams(*algparams))
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
                 slices=slices, portion=portion, tspec=tspec, vx=voxel,
                 fmt='txt')
        if voxel == 'all':
            d['fmt'] = 'h5'
        path = (self.work / 'texture' / masktype /
                '{m}_{slices}_{portion}_{vx}'.format(**d))
        if masktype == 'auto':
            path /= str(AlgParams(*algparams))
        if tspec is None:
            s = '{c}_{s}_{l}.{fmt}'
        else:
            s = '{c}_{s}_{l}_{tspec.method}-{tspec.winsize}.{fmt}'
        return path / s.format(**d)

    def std_cfg(self):
        """Return path to standardization configuration file."""
        return self.work / 'stdcfg_{m}.txt'.format(m=self.mode)

    def histogram(self, roi, samplelist):
        """Return path to histogram plot."""
        return (self.work / 'histograms' /
                '{m}_{r}_{s}.png'.format(m=self.mode, r=roi, s=samplelist))

    def grid(self, case, scan, masktype, tspec, fmt='txt'):
        """Return path to the first of the grid files.

        FIXME: The first element in the resulting tuple no more exists as file.
        """
        path = self.work / 'grid' / masktype / str(self.mode)
        if tspec is not None:
            path /= str(tspec)
        if case is not None and scan is not None:
            path /= '{c}-{s}.{f}'.format(c=case, s=scan, f=fmt)
        target = '{r}-0{e}'.format(r=path.stem, e=path.suffix)
        return path, target

    def histology(self, case):
        """Return path to histology image."""
        # NOTE: Doesn't use `self.mode`.
        return (self.work / 'hist' /
                'ALL_renamed_RALP').glob('{c}_*.*'.format(c=case))


# The rest are for compatibility.


def samplelist_path(mode, samplelist):
    return str(Paths(mode).samplelist(samplelist))


def pmap_path(mode, *args, **kwargs):
    return str(Paths(mode).pmap(*args, **kwargs))


def mask_path(mode, masktype, case, scan, *args, **kwargs):
    return str(Paths(mode).mask(masktype, case, scan, *args, **kwargs))


def roi_path(mode, masktype, **kwargs):
    return str(Paths(mode).roi(masktype, **kwargs))


def texture_path(mode, case, scan, lesion, masktype, slices, portion, tspec,
                 **kwargs):
    return str(Paths(mode).texture(case, scan, lesion, masktype, slices,
                                   portion, tspec, **kwargs))
