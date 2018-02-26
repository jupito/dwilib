#!/bin/sh

# Convert a bunch of mask files to HDF5 format.

outdir=out
fmt=h5

mkdir -p "$outdir"

for x in "$@"; do
    select_voxels.py -v --astype bool --keepmasked --source_attrs \
        -i "$x" -o "$outdir/$(basename "$x" .zip).$fmt"
done
