"""Operations regarding miscellaneous files."""

from __future__ import absolute_import, division, print_function
import os.path
import re

import numpy as np


COMMENT_PREFIX = '#'


def toline(iterable):
    """Convert an iterable into a line."""
    return ' '.join(str(x) for x in iterable) + '\n'


def valid_lines(filename):
    """Read and yield lines that are neither empty nor comments."""
    with open(filename, 'rU') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(COMMENT_PREFIX):
                yield line


def read_subwindows(filename):
    """Read a list of subwindows from file, return in a dictionary."""
    r = {}
    for line in valid_lines(filename):
        words = line.split()
        if len(words) != 8:
            raise Exception('Cannot parse subwindow: %s' % line)
        case, scan = int(words[0]), words[1]
        subwindow = [int(x) for x in words[2:]]
        r[(case, scan)] = subwindow
    return r


def read_patients_file(filename, include_lines=False):
    """Load a list of patients.

    Row format: num name scan1,scan2,... score [location]
    """
    from dwi.patient import GleasonScore, Lesion, Patient
    patients = []
    p = re.compile(r"""
                   (?P<num>\d+) \s+
                   (?P<name>\w+) \s+
                   (?P<scans>[\w,]+) \s+
                   (?P<score>\d\+\d(\+\d)?) \s* (?P<location>\w+)? \s*
                   ((?P<score2>\d\+\d(\+\d)?) \s+ (?P<location2>\w+))? \s*
                   ((?P<score3>\d\+\d(\+\d)?) \s+ (?P<location3>\w+))?
                   """,
                   flags=re.VERBOSE)
    for line in valid_lines(filename):
        m = p.match(line)
        if m:
            num = int(m.group('num'))
            name = m.group('name').lower()
            scans = sorted(m.group('scans').lower().split(','))
            score = GleasonScore(m.group('score'))
            lesions = [Lesion(0, score, 'xx')]
            if m.group('location'):
                # New-style, multi-lesion file.
                lesions = []
                lesions.append(Lesion(0, GleasonScore(m.group('score')),
                                      m.group('location').lower()))
                if m.group('score2'):
                    lesions.append(Lesion(1, GleasonScore(m.group('score2')),
                                          m.group('location2').lower()))
                if m.group('score3'):
                    lesions.append(Lesion(2, GleasonScore(m.group('score3')),
                                          m.group('location3').lower()))
            patient = Patient(num, name, scans, lesions)
            if include_lines:
                patient.line = line
            patients.append(patient)
        else:
            raise Exception('Invalid line in patients file: %s', line)
    return sorted(patients)


def read_sample_list(filename):
    """Read a list of samples from file."""
    entries = []
    p = re.compile(r'(\d+)\s+(\w+)\s+([\w,]+)')
    for line in valid_lines(filename):
        m = p.match(line)
        if m:
            case, name, scans = m.groups()
            case = int(case)
            name = name.lower()
            scans = tuple(sorted(scans.lower().split(',')))
            d = dict(case=case, name=name, scans=scans)
            entries.append(d)
    return entries


def read_mask_file(filename):
    """Read a ROI mask file. XXX: Deprecated, use dwi.mask module instead."""
    arrays = []
    for line in valid_lines(filename):
        if line[0] == '0' or line[0] == '1':
            a = np.array(list(line), dtype=int)
            arrays.append(a)
    mask = np.array(arrays)
    return mask


def read_subregion_file(filename):
    """Read a subregion definition from file.

    It's formatted as one voxel index per line, zero-based, in order of
    y_first, y_last, x_first, x_last, z_first, z_last. The "last" ones need +1
    to get Python-like start:stop indices. They are returned as (z_start,
    z_stop, y_start, y_stop, x_start, x_stop).
    """
    entries = []
    for line in valid_lines(filename):
        entries.append(int(float(line)))
    if len(entries) != 6:
        raise Exception('Invalid subregion file: %s' % filename)
    entries = entries[4:6] + entries[0:4]  # Move z indices to front.
    # Add one to "last" indices get Python-like start:stop indices.
    entries[1] += 1
    entries[3] += 1
    entries[5] += 1
    return tuple(entries)


def write_comment(f, text):
    """Write zero or more lines to file with comment prefix."""
    for line in text.splitlines():
        f.write('{p} {s}\n'.format(p=COMMENT_PREFIX, s=line))


def write_subregion_file(filename, win, comment=''):
    """Write a subregion definition to file.

    It's formatted as one voxel index per line, zero-based, in order of
    y_first, y_last, x_first, x_last, z_first, z_last.
    """
    if len(win) != 6:
        raise Exception('Invalid subregion: %s' % win)
    entries = [win[2], win[3]-1, win[4], win[5]-1, win[0], win[1]-1]
    with open(filename, 'w') as f:
        write_comment(f, comment)
        for entry in entries:
            f.write('%i\n' % entry)


def write_pmap(filename, pmap, attrs, fmt=None):
    """Write parametric map file either as HDF5 or ASCII."""
    pmap = np.asanyarray(pmap)
    if pmap.ndim < 2:
        raise Exception('Not enough dimensions: {}'.format(pmap.shape))
    if 'parameters' not in attrs:
        attrs['parameters'] = [str(i) for i in range(pmap.shape[-1])]
    if 'shape' not in attrs:
        attrs['shape'] = pmap.shape
    if 'dtype' not in attrs:
        attrs['dtype'] = str(pmap.dtype)
    if pmap.shape[-1] != len(attrs['parameters']):
        raise Exception('Number of values and parameters mismatch')
    assert all(isinstance(x, str) for x in
               attrs['parameters']), attrs['parameters']
    if fmt is None:
        fmt = os.path.splitext(filename)[1][1:]
    if fmt in ['hdf5', 'h5']:
        import dwi.hdf5
        dwi.hdf5.write_hdf5(filename, pmap, attrs)
    elif fmt in ['txt', 'ascii']:
        import dwi.asciifile
        pmap = pmap.reshape((-1, pmap.shape[-1]))  # Can't keep shape.
        dwi.asciifile.write_ascii_file(filename, pmap, None, attrs=attrs)
    else:
        raise Exception('Unknown format: {}'.format(fmt))


def pick_params(pmap, attrs, params):
    """Select a subset of parameters by their indices."""
    params = list(params)
    pmap = pmap[..., params]
    if 'bset' in attrs and len(attrs['bset']) == len(attrs['parameters']):
        attrs['bset'] = [attrs['bset'][x] for x in params]
    if 'echotimes' in attrs and (len(attrs['echotimes']) ==
                                 len(attrs['parameters'])):
        attrs['echotimes'] = [attrs['echotimes'][x] for x in params]
    attrs['parameters'] = [attrs['parameters'][x] for x in params]
    return pmap, attrs


def read_pmap(pathname, ondisk=False, fmt=None, params=None):
    """Read a parametric map.

    With parameter ondisk it will not be read into memory. Parameter params
    tells which parameter indices should be included.
    """
    if fmt is None:
        fmt = os.path.splitext(pathname)[1][1:]
    if fmt in ['hdf5', 'h5']:
        import dwi.hdf5
        pmap, attrs = dwi.hdf5.read_hdf5(pathname, ondisk=ondisk)
    elif fmt in ['txt', 'ascii']:
        import dwi.asciifile
        attrs, pmap = dwi.asciifile.read_ascii_file(pathname)
        if 'parameters' in attrs:
            attrs['parameters'] = attrs['parameters'].split()
    else:
        import dwi.dicomfile
        d = dwi.dicomfile.read_dir(pathname)
        pmap = d.pop('image')
        attrs = dict(d)
    if 'parameters' not in attrs:
        attrs['parameters'] = range(pmap.shape[-1])
    attrs['parameters'] = [str(x) for x in attrs['parameters']]
    if params is not None:
        pmap, attrs = pick_params(pmap, attrs, params)
    return pmap, attrs


def read_mask(path, expected_voxel_spacing=None, n_dec=3, container=None,
              allowed_outside=0.2):
    """Read pmap as a mask.

    Optionally expect voxel spacing to match up to a certain number of
    decimals. The optional parameter allowed_outside indicates how much of the
    smaller mask (lesion) volume may be outside of a larger container mask
    (prostate) without an error being raised.
    """
    mask, attrs = read_pmap(path)
    mask = mask[..., 0].astype(np.bool)
    if expected_voxel_spacing is not None:
        voxel_spacing = [round(x, n_dec) for x in attrs['voxel_spacing']]
        expected_voxel_spacing = [round(x, n_dec) for x in
                                  expected_voxel_spacing]
        if voxel_spacing != expected_voxel_spacing:
            raise ValueError('Expected voxel spacing {}, got {}'.format(
                expected_voxel_spacing, voxel_spacing))
    if container is not None:
        portion_outside_container = (np.count_nonzero(mask[~container]) /
                                     np.count_nonzero(mask))
        if portion_outside_container > allowed_outside:
            s = '{}: Portion of selected voxels outside container is {:%}'
            raise ValueError(s.format(path, portion_outside_container))
    return mask
