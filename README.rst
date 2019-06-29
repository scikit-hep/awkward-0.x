.. image:: docs/source/logo-300px.png

awkward-array
=============

.. inclusion-marker-1-do-not-remove

Manipulate arrays of complex data structures as easily as Numpy.

.. inclusion-marker-1-5-do-not-remove

Calculations with rectangular, numerical data are simpler and faster in Numpy than traditional for loops. Consider, for instance,

.. code-block:: python

    all_r = []
    for x, y in zip(all_x, all_y):
        all_r.append(sqrt(x**2 + y**2))

versus

.. code-block:: python

    all_r = sqrt(all_x**2 + all_y**2)

Not only is the latter easier to read, it's hundreds of times faster than the for loop (and provides opportunities for hidden vectorization and parallelization). However, the Numpy abstraction stops at rectangular arrays of numbers or character strings. While it's possible to put arbitrary Python data in a Numpy array, Numpy's ``dtype=object`` is essentially a fixed-length list: data are not contiguous in memory and operations are not vectorized.

Awkward-array is a pure Python+Numpy library for manipulating complex data structures as you would Numpy arrays. Even if your data structures

* contain variable-length lists (jagged/ragged),
* are deeply nested (record structure),
* have different data types in the same list (heterogeneous),
* are masked, bit-masked, or index-mapped (nullable),
* contain cross-references or even cyclic references,
* need to be Python class instances on demand,
* are not defined at every point (sparse),
* are not contiguous in memory,
* should not be loaded into memory all at once (lazy),

this library can access them as `columnar data structures <https://towardsdatascience.com/the-beauty-of-column-oriented-data-2945c0c9f560>`__, with the efficiency of Numpy arrays. They may be converted from JSON or Python data, loaded from "awkd" files, `HDF5 <https://www.hdfgroup.org>`__, `Parquet <https://parquet.apache.org>`__, or `ROOT <https://root.cern>`__ files, or they may be views into memory buffers like `Arrow <https://arrow.apache.org>`__.

.. inclusion-marker-2-do-not-remove

Installation
============

Install awkward like any other Python package:

.. code-block:: bash

    pip install awkward                       # maybe with sudo or --user, or in virtualenv
    pip install awkward-numba                 # optional: integration with and optimization by Numba

or install with `conda <https://conda.io/en/latest/miniconda.html>`__:

.. code-block:: bash

    conda config --add channels conda-forge   # if you haven't added conda-forge already
    conda install awkward
    conda install awkward-numba               # optional: integration with and optimization by Numba

The base ``awkward`` package requires only `Numpy <https://scipy.org/install.html>`__  (1.13.1+), but ``awkward-numba`` additionally requires `Numba <https://numba.pydata.org/numba-doc/dev/user/installing.html>`__.

Recommended packages:
---------------------

- `pyarrow <https://arrow.apache.org/docs/python/install.html>`__ to view Arrow and Parquet data as awkward-arrays
- `h5py <https://www.h5py.org>`__ to read and write awkward-arrays in HDF5 files
- `Pandas <https://pandas.pydata.org>`__ as an alternative view

.. inclusion-marker-3-do-not-remove

Tutorial
========

Run this tutorial on Binder (TODO).

**Table of contents:**

- TODO

Reference
=========

For a list of all functions, classes, methods, and their parameters, click below.


TODO: insert sketchwork.py here, converted to REsT.


Acknowledgements
================

Support for this work was provided by NSF cooperative agreement OAC-1836650 (IRIS-HEP), grant OAC-1450377 (DIANA/HEP) and PHY-1520942 (US-CMS LHC Ops).

Thanks especially to the gracious help of `awkward-array contributors <https://github.com/scikit-hep/awkward-array/graphs/contributors>`__!
