#!/usr/bin/python3

"""Anonymize DICOM files. Write in-place or into another directory."""

import argparse
import os

import dicom


# ftp://medical.nema.org/medical/dicom/final/sup55_ft.pdf
# Digital Imaging and Communications in Medicine (DICOM)
# Supplement 55: Attribute Level Confidentiality (including De-identification)
# Table X.1-1
# Basic Application Level Confidentiality Profile Attributes
CONFIDENTIAL_TAGS = {
    (0x0008, 0x0014): "Instance Creator UID",
    (0x0008, 0x0018): "SOP Instance UID",
    (0x0008, 0x0050): "Accession Number",
    (0x0008, 0x0080): "Institution Name",
    (0x0008, 0x0081): "Institution Address",
    (0x0008, 0x0090): "Referring Physician's Name",
    (0x0008, 0x0092): "Referring Physician's Address",
    (0x0008, 0x0094): "Referring Physician's Telephone Numbers",
    (0x0008, 0x1010): "Station Name",
    (0x0008, 0x1030): "Study Description",
    (0x0008, 0x103E): "Series Description",
    (0x0008, 0x1040): "Institutional Department Name",
    (0x0008, 0x1048): "Physician(s) of Record",
    (0x0008, 0x1050): "Performing Physicians' Name",
    (0x0008, 0x1060): "Name of Physician(s) Reading Study",
    (0x0008, 0x1070): "Operators' Name",
    (0x0008, 0x1080): "Admitting Diagnoses Description",
    (0x0008, 0x1155): "Referenced SOP Instance UID",
    (0x0008, 0x2111): "Derivation Description",
    (0x0010, 0x0010): "Patient's Name",
    (0x0010, 0x0020): "Patient ID",
    (0x0010, 0x0030): "Patient's Birth Date",
    (0x0010, 0x0032): "Patient's Birth Time",
    (0x0010, 0x0040): "Patient's Sex",
    (0x0010, 0x1000): "Other Patient Ids",
    (0x0010, 0x1001): "Other Patient Names",
    (0x0010, 0x1010): "Patient's Age",
    (0x0010, 0x1020): "Patient's Size",
    (0x0010, 0x1030): "Patient's Weight",
    (0x0010, 0x1090): "Medical Record Locator",
    (0x0010, 0x2160): "Ethnic Group",
    (0x0010, 0x2180): "Occupation",
    (0x0010, 0x21B0): "Additional Patient's History",
    (0x0010, 0x4000): "Patient Comments",
    (0x0018, 0x1000): "Device Serial Number",
    (0x0018, 0x1030): "Protocol Name",
    (0x0020, 0x000D): "Study Instance UID",
    (0x0020, 0x000E): "Series Instance UID",
    (0x0020, 0x0010): "Study ID",
    (0x0020, 0x0052): "Frame of Reference UID",
    (0x0020, 0x0200): "Synchronization Frame of Reference UID",
    (0x0020, 0x4000): "Image Comments",
    (0x0040, 0x0275): "Request Attributes Sequence",
    (0x0040, 0xA124): "UID",
    (0x0040, 0xA730): "Content Sequence",
    (0x0088, 0x0140): "Storage Media File-set UID",
    (0x3006, 0x0024): "Referenced Frame of Reference UID",
    (0x3006, 0x00C2): "Related Frame of Reference UID",
}


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-v', '--verbose', action='count',
                   help='increase verbosity')
    p.add_argument('-n', '--dry-run', action='store_true',
                   help='do not write any changes')
    p.add_argument('-i', '--input', metavar='FILENAME', nargs='+',
                   help='input DICOM files')
    p.add_argument('-o', '--output', metavar='PATHNAME',
                   help='output directory')
    return p.parse_args()


def delete_confidential(dataset, elem):
    """Delete confidential elements from dataset."""
    if elem.tag in CONFIDENTIAL_TAGS:
        deleted.append((elem.tag, elem.name))
        del dataset[elem.tag]


args = parse_args()
for infile in args.input:
    if args.output:
        outfile = os.path.join(args.output, os.path.basename(infile))
    else:
        outfile = infile
    deleted = []
    f = dicom.read_file(infile)
    f.walk(delete_confidential, recursive=True)
    if deleted:
        if args.verbose > 1:
            for elem in deleted:
                print('{i}: {t} {n}'.format(i=infile, t=elem[0], n=elem[1]))
        if args.verbose:
            d = dict(i=infile, o=outfile, n=len(deleted))
            print('{i}: Deleted {n} elements, writing to {o}.'.format(**d))
        if not args.dry_run:
            f.save_as(outfile, WriteLikeOriginal=True)
