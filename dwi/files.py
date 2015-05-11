import re

import dwi.util

"""Operations regarding miscellaneous files."""

COMMENT_PREFIX = '#'

def read_subwindows(filename):
    """Read a list of subwindows from file, return in a dictionary."""
    r = {}
    with open(filename, 'rU') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(COMMENT_PREFIX):
                continue
            words = line.split()
            if len(words) != 8:
                raise Exception('Cannot parse subwindow: %s' % line)
            case, scan, subwindow = int(words[0]), words[1], map(int, words[2:])
            r[(case, scan)] = subwindow
    return r

def read_sample_list(filename):
    """Read a list of samples from file."""
    entries = []
    p = re.compile(r'(\d+)\s+(\w+)\s+([\w,]+)')
    with open(filename, 'rU') as f:
        for line in f:
            m = p.match(line.strip())
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
    with open(filename, 'rU') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] == '0' or line[0] == '1':
                a = np.array(list(line), dtype=int)
                arrays.append(a)
    mask = np.array(arrays)
    return mask

def read_subregion_file(filename):
    """Read a subregion definition from file.

    It's formatted as one voxel index per line, zero-based, in order of y_first,
    y_last, x_first, x_last, z_first, z_last. The "last" ones need +1 to get
    Python-like start:stop indices. They are returned as (z_start, z_stop,
    y_start, y_stop, x_start, x_stop).
    """
    entries = []
    with open(filename, 'rU') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(COMMENT_PREFIX):
                continue
            entries.append(int(float(line)))
    if len(entries) != 6:
        raise Exception('Invalid subregion file: %s' % filename)
    entries = entries[4:6] + entries[0:4] # Move z indices to front.
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

    It's formatted as one voxel index per line, zero-based, in order of y_first,
    y_last, x_first, x_last, z_first, z_last.
    """
    if len(win) != 6:
        raise Exception('Invalid subregion: %s' % win)
    entries = [win[2], win[3]-1, win[4], win[5]-1, win[0], win[1]-1]
    with open(filename, 'w') as f:
        write_comment(f, comment)
        for entry in entries:
            f.write('%i\n' % entry)
