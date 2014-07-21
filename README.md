dwilib
======

Tools in development to support analysis of Diffusion-weighted imaging (DWI) data, with focus on prostate cancer.

Currently it can only read the data in custom ASCII format, but support for DICOM is under way.

Features
--------
- Perform model fitting (ADC, Kurtosis, etc.)
- Calculate correlation by Gleason scores
- Calculate and compare diagnostic ROC AUCs
- Calculate reproducibility measures
- Plotting schemes

Requirements
------------
- NumPy
- SciPy
- Scikit-Learn
- leastsqbound-scipy
- Matplotlib
- Pydicom

Todo
----
- Clean up the messy parts
- Documentation
- DICOM support
- Proper pipelining
- Regression classification
