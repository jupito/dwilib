#!/usr/bin/python3

"""Read CPR files."""

import logging
import re
import sys

import numpy as np

import dwi.files
from dwi.types import Path

"""
function cprs = plot_data(name, cprs)

    global Maskdir;

    if exist(Maskdir) ~= 7
        mkdir(Maskdir);
    end

    %read all lines
    fprintf('%d Opening file %s\n', cprs, name.cpr)

    if exist(name.cpr) ~= 2
        fprintf(2, 'File %s was not found\n', name.cpr);
        return;
    end

    fid=fopen(name.cpr);
    lines = {};
    while 1
        tline = fgetl(fid);
        if ~ischar(tline), break, end
        lines{end+1} = tline;
    end
    fclose(fid);

    indexes = strfind(name.cpr,'_');
    name.cpr = name.cpr(1:indexes(end)+1);

    %handle data
    %read hard-coded line for image data dimensions
    %dimensions_line = lines{324};
    %dim_start = strfind(dimensions_line,'[');
    %dim_end = strfind(dimensions_line,']');
    %dim_str = dimensions_line(dim_start+1:dim_end-1);
    %dim = sscanf(dim_str,'%d')';

    %read hard-coded line for image data
    for i = 1:length(lines)
        if length(strfind(lines{i},'<connectionString id="')) > 0
            DICOM_line = lines{i};
            filename_start = strfind(DICOM_line,'>');
            filename_end = strfind(DICOM_line,'<');
            filename_str = DICOM_line(filename_start(1)+1:filename_end(2)-1);
            break;
        end
    end

    dim = name.dim;
    %resolve lines for 2D ROIs
    ROIs = {};
    for l_i = 1:length(lines)
%        if (length(strfind(lines{l_i},'Type="Carimas.Data.VOIs.ROI_ListVOI"
%>')) > 0 && ...
%            length(strfind(lines{l_i},'Name="')) > 0)
%            inds = strfind(lines{l_i},'"');
%            ROIs{end+1}.name = lines{l_i}(inds(3)+1:inds(4)-1);
%            ROIs{end}.ROI_lines = [];
%        end
        if length(strfind(lines{l_i},'Type="Carimas.Data.VOIs.ROI" >')) > 0
            inds = strfind(lines{l_i},'"');
            ROIs{end+1}.name = lines{l_i}(inds(3)+1:inds(4)-1);
            ROIs{end}.ROI_lines = [];
            ROIs{end}.ROI_lines(end+1) = l_i;
        end
    end
    for ROI_i = 1:length(ROIs)
        ROIs{ROI_i}.ROIslices = [];
        ROIs{ROI_i}.ROIs = {};
        ROIs{ROI_i}.no = ROI_i;
        for r_i = 1:length(ROIs{ROI_i}.ROI_lines)
            %resolve name
            name_line = lines{ROIs{ROI_i}.ROI_lines(r_i)};
            str_tag_indexes = strfind(name_line,'"');
            ROIs{ROI_i}.ROIs{end+1}.name =
name_line(str_tag_indexes(3)+1:str_tag_indexes(4)-1);

            %resolve packed mask data
            mask_line = lines{ROIs{ROI_i}.ROI_lines(r_i)+4};
            mask_start = strfind(mask_line,'[');
            mask_end = strfind(mask_line,']');
            packedmask = sscanf(mask_line(mask_start+1:mask_end-1), '%d');

            %unpack mask data
            position = 0;
            mask = zeros(dim(1),dim(2),dim(3));
            ROIslice = -1;
            for mask_i = 1:length(packedmask)
                if packedmask(mask_i) > 0
                    pos_z = floor(position/(dim(1)*dim(2)));
                    pos_y = floor((position-dim(1)*dim(2)*pos_z)/dim(1));
                    pos_x = (position-dim(1)*dim(2)*pos_z)-dim(1)*pos_y;
                    mask(pos_x+1:pos_x + packedmask(mask_i), pos_y+1, pos_z+1)
= 1;
                    ROIslice = pos_z+1;
                    position = position + packedmask(mask_i);
                else
                    position = position - packedmask(mask_i);
                end
            end
            ROIs{ROI_i}.ROIs{end}.mask = mask(:,:,ROIslice)';
            ROIs{ROI_i}.ROIs{end}.vol = length(mask(mask > 0));
            ROIs{ROI_i}.ROIslices(end+1) = ROIslice;
        end
    end

    img = zeros(name.dim);
    if length(ROIs) == 0
        fprintf(2, 'No ROIs was found\n', name.cpr);
        return;
    end
    for ROI_i = 1:length(ROIs)
        %name: 'Copied'
        %mask: [224x224 double]
        %ROIslice: 6
        %vol: 25
        try
            if length(strfind(name.name, '1st')) > 0
                img = name.fun{1}(img, ROIs{ROI_i});
            else
                img = name.fun{2}(img, ROIs{ROI_i});
            end
        catch
            fprintf(2, 'Failed to create mask\n');
            continue;
        end
    end
    nii = make_nii(img, [name.siz], name.orig, 4, ROIs{1}.name);
    sname = name.name;
    sname = strrep(sname, '1st_1st', '1st');
    sname = strrep(sname, '2nd_2nd', '2nd');
    if length(strfind(sname, 'Hr_b_')) == 0
        sname = strrep(sname, 'Hr_', 'Hr_b_');
    end

    if exist([Maskdir filesep name.casename]) ~= 7
        mkdir([Maskdir filesep name.casename]);
    end
    if strcmp(name.cpr(end-3:end),'_2nd') == 1
        nii_filename = [sname '_2nd.nii'];
    else
        nii_filename = [sname '.nii'];
    end
    fprintf('Writing %s\n', [Maskdir filesep name.casename filesep
nii_filename]);
    try
        save_nii(nii, [Maskdir filesep name.casename filesep nii_filename]);
    catch
        fprintf(2, 'Failed to save mask\n');
    end
    cprs = cprs + 1;
end
"""


# class CPRFile(object):
#     def __init__(self, path):
#         self.path = Path(path)
#
#     @staticmethod
#     def parse_file(path):


def read_cpr(path):
    """Read and parse a CPR file. Return masks, which are (id, RLE data).

    Note: Python 3.6.4 docs say its XML module is not secure, se we use regexp.
    """
    # Example: <Mask id="123">[12 34 56]</Mask>
    mask_pattern = r'<Mask\s.*?id="(.*?)".*?>.*?\[(.*?)\].*?</Mask>'
    text = path.read_text()
    matches = re.finditer(mask_pattern, text, flags=re.DOTALL)

    def parse_match(m):
        number, mask = m.groups()
        number = int(number)
        mask = [int(x) for x in mask.split()]
        return number, mask

    masks = [parse_match(x) for x in matches]
    return masks


def parse_mask(mask):
    """Decode a run length encoded mask into an array."""
    lst = []
    for length in mask:
        n = int(length > 0)  # Run is 1 if positive, 0 if negative.
        lst.extend([n] * abs(length))
    return np.array(lst, dtype=np.bool)


def main(path, shape, outdir, fmt='h5'):
    logging.info(path)
    masks = read_cpr(path)
    logging.info(len(masks))
    # logging.info(masks)
    # logging.info(masks[-1][1])
    # logging.info(len(parse_mask(masks[-1][1])))
    for i, m in enumerate(masks, 1):
        number, mask = m
        logging.info('Mask: %i, $i', number, len(mask))
        mask = parse_mask(mask)
        try:
            mask.shape = shape + (1,)
        except ValueError as e:
            logging.error('%s: %s', e, path)
            continue
        assert mask.ndim == 4, mask.shape
        outname = '{}.{}-{}.{}'.format(path.name, i, number, fmt)
        outpath = outdir / outname
        attrs = {}
        print('Writing mask shape {}: {}'.format(mask.shape, outpath))
        dwi.files.write_pmap(outpath, mask, attrs)


if __name__ == '__main__':
    # Arguments: input file; output directory; shape (eg 20,224,224).
    logging.basicConfig(level=logging.INFO)
    path, outdir, shape = sys.argv[1:]
    logging.info(path, outdir, shape, sys.argv[1:])
    path = Path(path)
    shape = tuple(int(x) for x in shape.split(','))
    outdir = Path(outdir)
    main(path, shape, outdir)
