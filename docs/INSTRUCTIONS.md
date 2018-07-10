Instructions for using dwilib
=============================



Files
-----

Directory "tools" contains tools for different tasks which are described below.
They expect to find the library directory "dwi" along the usual path of Python
libraries, so modify the environment variable PYTHONPATH as needed.

Directory "scripts" contains some smaller scripts that were used at some point
when calculating things for our research.


### Model fitting
* pmap.py -- Fit diffusion models to imaging data (see below).


### Statistical
* compare_masks.py -- Compare ROI masks.
* correlation.py -- Test parameter correlation with Gleason scores.
* find_roi.py -- Automatic ROI search.
* reproducibility.py -- Calculate reproducibility measures, like coefficient of
  repeatability (CR) and intraclass correlation coefficient (ICC(3,1)).
* roc_auc.py -- Calculate and compare diagnostic ROC AUCs.


### Plotting
* compare_boxplot.py
* compare_hist.py
* compare_pmaps.py
* draw_boxplot.py
* draw_pmap.py
* draw_pmaps.py
* draw_roc.py
* plot.py
* plot_gleason.py


### DICOM tools
* anonymize_dicom.py -- Anonymize DICOM files.
* info_dicom.py -- Show information on DICOM files.
* view_dicom.py -- View DICOM DWI images.


### Other
* dodo.py -- Build tool for [DoIt](http://pydoit.org/).
* info.py -- Print information on columns of numbers.
* make_ascii.py
* masktool.py -- Inspect and manage masks.
* print_all.py
* test-rls.py


### Old and deprecated
* compare_auc.py -- Calculate and compare diagnostic ROC AUCs.



Fitting models to data
----------------------

Program `pmap.py` is used for fitting the diffusion models to imaging data. It
generates parametric maps, hence the name. Invoking `pmap.py --help` lists
possible arguments for the program. Use `-l` to list all available models.

    usage: pmap.py [-h] [-v] [-l] [-a] [-s I I I I I I] [-m MODEL [MODEL ...]]
                   [-i FILENAME [FILENAME ...]] [-d PATHNAME [PATHNAME ...]]
                   [-o FILENAME]

    Produce parametric maps by fitting one or more diffusion models to imaging
    data. Multiple input images can be provided in ASCII format. Single input
    image can be provided as a group of DICOM files. Output is written in ASCII
    files named by input and model.

    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         increase verbosity
      -l, --listmodels      list available models
      -a, --average         average input voxels into one
      -s I I I I I I, --subwindow I I I I I I
                            use subwindow (specified by 6 one-based indices)
      -m MODEL [MODEL ...], --models MODEL [MODEL ...]
                            models to use
      -i FILENAME [FILENAME ...], --input FILENAME [FILENAME ...]
                            input ASCII files
      -d PATHNAME [PATHNAME ...], --dicom PATHNAME [PATHNAME ...]
                            input DICOM files or directories
      -o FILENAME, --output FILENAME
                            output file (for single model only)

Subwindow specification (`-s`) is a list of six one-based index numbers in the
following order: first included slice, next excluded slice, first included row,
next excluded row, first included column, next excluded column. For example,
`-s 9 10 70 120 80 140` includes the 9th slice with an area of 50x60 voxels
starting from the 70th row and 80th column.

Output files are in simple ASCII format. The header contains information about
the subwindow, b-values, parameter names, etc. After that, each line represents
one voxel with its fitted parameter values separated by space, and ending in
the root mean square error (RMSE).

Examples:

    pmap.py -s 11 12 85 135 80 140 -m MonoN KurtN -d 10_1a/DICOM/

fits the normalized Monoexponential ADC and Kurtosis models to a one-slice
subwindow of a DICOM image from directory `10_1a/DICOM`, outputting each
parametric map to its on file.

    pmap.py -s 11 12 85 135 80 140 -m normalized -d 10_1a/DICOM/

fits all normalized models.

    pmap.py -m SiN -d 10_1a/DICOM/ -o out.txt

generates the normalized signal intensity curves for the whole image into file
`out.txt`.



Correlation and ROC AUC analysis
--------------------------------

Programs `correlation.py` and `roc_auc.py` can be used to calculate pmap
correlation with Gleason scores and ROC AUC based on how well it discriminates
Gleason score groups.

These tools require a samplelist file, which contains the list of included
samples as lines with the form `case name scan1,scan2`, and a patient list
file, which contains also Gleason scores on lines like `case name scan1,scan2
score`. Example files are provided in the source tree.

Some examples:

    correlation.py -v --patients patients.txt --thresholds 3+3 3+4 --voxel mean --pmapdir pmaps
    correlation.py -v --patients patients.txt --thresholds --voxel mean --pmapdir pmaps

These calculate Spearman correlation coefficients for samples mentioned in
`patients.txt`, with pmap files in directory `pmaps`, grouping Gleason scores to
three groups, or no groups at all. In first variation, parameter `--thresholds
3+3 3+4` sets the score groups to those smaller than equal to 3+3, 3+4, and
those greater than 3+4.

Parametric map filenames should have the form `{c}_*_{s}_*.txt`, where `{c}` is
case number and `{s}` is scan identifier.

Use parameter `-v` to get also the p-value and confidence interval. Yet another
`-v` add more information output, and `-h` gives help. Parameter `--voxel`
selects the voxel (number line in ASCII files) to use. It can be a zero-based
voxel index number or a string: `mean` and `median` average all voxels in each
file, the default value `all` takes all voxels independently. Optional
parameter `--patients` changes the patient list filename from its default
`patients.txt`.

In order to get ROC AUCs you can type something like:

    roc_auc.py -v --samplelist samples.txt --patients patients.txt --threshold 3+3 --nboot 5000 --voxel mean --autoflip --pmapdir pmaps

This calculates ROC AUCs for samples mentioned in samplelist file
`samples.txt`, with pmap files in directory `pmaps`, grouping Gleason scores to
two groups, those less or equal to 3+3, and those greater than 3+3.

Bootstrapped AUCs are also calculated, here with 5000 bootsraps. Parameter
`--autoflip` just flips the data when necessary to make the AUC always greater
than 0.5. If you give more than one pmap directory, the statistical difference
of AUCs is calculated for them.



Reproducibility measures
------------------------

Program `reproducibility.py` can be used to calculate reproducibility measures,
most importantly the coefficient of reproducibility (CR) and intra-class
correlation (ICC(3,1)).

This tool still uses the old interface, so the samples are given as a list of
pmap files instead of a samplelist file and a directory.

An example:

    reproducibility.py -v --voxel 0 -b 10000 -m pmaps/*_MonoN.txt

This calculates various numbers (see below), using the first voxel and 10000
bootstraps. With parameter `--voxel`, you can set the voxel index or set it to
`mean` or `median` of all voxels.

Parametric map filenames should have the form `*{c}*{s}*`, where `{c}` is case
number, and `{s}` is scan identifier.

Measures:

* avg[lower-upper]: value mean with confidence interval
* msd/avg: mean squared difference
* CI/avg: confidence interval per mean
* wCV: within-patient coefficient of variance
* CoR/avg: coefficient of repeatability per mean
* ICC: intra-class correlation, form ICC(3,1)
* bsICC[lower-upper]: bootstrapped intra-class correlation with confidence
interval

