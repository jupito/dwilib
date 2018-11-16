"""A setuptools based setup module."""

# See:
# https://packaging.python.org/
# https://packaging.python.org/tutorials/packaging-projects/
# https://setuptools.readthedocs.io/en/latest/setuptools.html
# https://packaging.python.org/tutorials/distributing-packages/
# https://github.com/pypa/sampleproject

from setuptools import find_packages, setup


# Get the long description from the README file
with open('README.rst') as f:
    long_description = f.read()

setup(
    name='dwilib',
    version='0.1.0.dev2',
    description='Research tools for MRI-based CAD of cancer',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/jupito/dwilib',
    author='Jussi Toivonen',
    author_email='jupito@iki.fi',
    license='MIT',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Utilities',
    ],
    keywords='medical imaging cancer mri',

    python_requires='>=3.4',
    # # https://packaging.python.org/en/latest/requirements.html
    # install_requires=[
    #     # 'doit',
    #     'h5py',
    #     'joblib',
    #     'leastsqbound',
    #     'mahotas',
    #     'matplotlib',
    #     # 'nibabel',
    #     'numpy',
    #     # 'Pillow',
    #     'pandas',
    #     'pydicom',
    #     'scikit-image',
    #     'scikit-learn',
    #     'scipy',
    #     # 'seaborn',
    #     # 'tabulate',
    #     # 'xarray',
    # ],

    # packages=find_packages(exclude=['contrib', 'doc', 'tests']),
    packages=find_packages(),
    package_data={
        # 'dwilib': ['examples/doit.cfg'],
        # 'dwilib': ['examples/doit.cfg'],
        # '': [
        #     'doc/*',
        #     'examples/*',
        #     'tools',
        #     ],
    },
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
)
