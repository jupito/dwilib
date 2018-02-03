dwilib
======

*NEW: Switched to Python version 3, it is now required and assumed everywhere!*

Tools in development to support analysis of Diffusion-Weighted Imaging (DWI)
data, with focus on prostate cancer.

This software is being developed as part of a research project at the `Magnetic
Resonance Imaging Research Center <http://mrc.utu.fi/>`_, Turku, Finland.

Note: This is very much under construction. The program code is being written on
demand basis, for custom needs at in-house projects. It has grown organically at
the same time as the programmer has been studying the subject and learning the
tools. So the code is quite messy in many places.


Features
--------
- Read input data as DICOM, or in custom ASCII or MATLAB formats
- Perform model fitting (Monoexponential ADC, Kurtosis, Stretched exponential,
  Biexponential)
- Calculate correlation with Gleason score
- Calculate and compare diagnostic ROC AUCs
- Calculate reproducibility measures
- Plotting schemes
- Viewer for multi-slice, multi-b-value DWI DICOM files (uses the Matplotlib GUI
  widget)
- Build tool for automated pipelining of data processing tasks


Todo or In Progress
-------------------
- Improved documentation
- Improved build tool
- Autonomous tumor delineation/ROI placement
- Regression classification


Requirements
------------
- Python 3.4
- pathlib2 (or newer Python)
- NumPy
- SciPy
- `Scikit-Learn <http://scikit-learn.org/>`_


Optional requirements
---------------------
- `leastsqbound-scipy <https://github.com/jjhelmus/leastsqbound-scipy>`_ (fitting)
- `Matplotlib <http://matplotlib.org/>`_ (plotting)
- `Pydicom <https://code.google.com/p/pydicom/>`_ (reading DICOM files)
- `h5py <http://www.h5py.org/>`_ (handling HDF5 files)
- `scikit-image <http://scikit-image.org/>`_ (texture analysis)
- `Mahotas <http://luispedro.org/software/mahotas/>`_ (texture analysis)
- `PyDoIt <http://pydoit.org/>`_ (task management)
