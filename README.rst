awkward-array
=============

.. inclusion-marker-1-do-not-remove

Manipulate arrays of complex data structures as easily as Numpy.

.. inclusion-marker-1-5-do-not-remove

awkward-array is a pure Python+Numpy library for manipulating complex data structures as you would Numpy arrays. Even if your data structures

- contain variable-length lists (jagged or ragged),
- are deeply nested (records or structs),
- have different data types in the same list (heterogeneous),
- are masked, bit-masked, or index-mapped (nullable),
- contain cross-references or even cyclic references,
- need to be Python class instances on demand,
- are not defined at every point (sparse),
- are not contiguous in memory,
- should not be loaded into memory all at once (lazy),

this library can access them with the efficiency of Numpy arrays. They may be converted from JSON or Python data, loaded from ZIP, `HDF5 <https://www.h5py.org>`__, `Parquet <https://parquet.apache.org>`__, or `ROOT <https://root.cern>`__ files, or they may be views into memory buffers like `Arrow <https://arrow.apache.org>`__.

Consider this monstrosity:

.. code-block:: python

    import awkward
    array = awkward.fromiter([[1.1, 2.2, None, 3.3, None],
                              [4.4, [5.5]],
                              [{"x": 6, "y": {"z": 7}}, None, {"x": 8, "y": {"z": 9}}]
                             ])

It's a list of lists; the first contains numbers and ``None``, the second contains a sub-sub-list, and the third defines nested records. If we print this out, we see that it is called a ``JaggedArray``:

.. code-block:: python

    array
    # returns <JaggedArray [[1.1 2.2 None 3.3 None] [4.4 [5.5]] [<Row 0> None <Row 1>]] at 79093e598f98>

and we get the full Python structure back by calling ``array.tolist()``:

.. code-block:: python

    array.tolist()
    # returns [[1.1, 2.2, None, 3.3, None],
    #          [4.4, [5.5]],
    #          [{'x': 6, 'y': {'z': 7}}, None, {'x': 8, 'y': {'z': 9}}]]

But we can also manipulate it as though it were a Numpy array. We can, for instance, take the first two elements of each sub-list (slicing the second dimension):

.. code-block:: python

    array[:, :2]
    # returns <JaggedArray [[1.1 2.2] [4.4 [5.5]] [<Row 0> None]] at 79093e5ab080>

or the last two:

.. code-block:: python
    
    array[:, -2:]
    # returns <JaggedArray [[3.3 None] [4.4 [5.5]] [None <Row 1>]] at 79093e5ab3c8>

Internally, the data has been rearranged into a ``columnar <https://towardsdatascience.com/the-beauty-of-column-oriented-data-2945c0c9f560>`__ form, with all values at a given level of hierarchy in the same array. Numpy-like slicing, masking, and fancy indexing are translated into Numpy operations on these internal arrays: they are _not_ implemented with Python for loops!

To see some of this structure, ask for the content of the array:

.. code-block:: python

    array.content
    # returns <IndexedMaskedArray [1.1 2.2 None ... <Row 0> None <Row 1>] at 79093e598ef0>

Notice that the boundaries between sub-lists are gone: they exist only at the ``JaggedArray`` level. This ``IndexedMaskedArray`` level handles the ``None`` values in the data. If we dig further, we'll find a ``UnionArray`` to handle the mixture of sub-lists and sub-sub-lists and record structures. If we dig far enough, we'll find the numerical data:

.. code-block:: python

    array.content.content.contents[0]
    # returns array([1.1, 2.2, 3.3, 4.4])
    array.content.content.contents[1].content
    # returns array([5.5])

Perhaps most importantly, Numpy's universal functions (operations that apply to every element in an array) can be used on our array. This, too, goes straight to the columnar data and preserves structure.

.. code-block:: python
    
    array + 100
    # returns <JaggedArray [[101.1 102.2 None 103.3 None]
    #                       [104.4 [105.5]]
    #                       [<Row 0> None <Row 1>]] at 724509ffe2e8>

    (array + 100).tolist()
    # returns [[101.1, 102.2, None, 103.3, None],
    #          [104.4, [105.5]],
    #          [{'x': 106, 'y': {'z': 107}}, None, {'x': 108, 'y': {'z': 109}}]]

    numpy.sin(array)
    # returns <JaggedArray [[0.8912073600614354 0.8084964038195901 None -0.1577456941432482 None]
    #                       [-0.951602073889516 [-0.70554033]]
    #                       [<Row 0> None <Row 1>]] at 70a40c3a61d0>

Rather than matching the speed of compiled code, this can exceed the speed of compiled code on non-columnar data because the operation may be vectorized on awkward-array's underlying columnar arrays.

.. inclusion-marker-2-do-not-remove

Installation
============

Install awkward-array like any other Python package:

.. code-block:: bash

    pip install awkward

or similar (use ``sudo``, ``--user``, ``virtualenv``, or pip-in-conda if you wish).

Strict dependencies:
====================

- `Python <http://docs.python-guide.org/en/latest/starting/installation/>`__ (2.7+, 3.4+)
- `Numpy <https://scipy.org/install.html>`__

Recommended dependencies:
=========================

- `Numba and LLVM <http://numba.pydata.org/numba-doc/latest/user/installing.html>`__ to JIT-compile functions (requires a particular version of LLVM, follow instructions)
- `Dask <http://dask.pydata.org/en/latest/install.html>`__ to distribute work on arrays
- `bcolz <http://bcolz.blosc.org/en/latest/install.html>`__ for on-the-fly compression
- `pyarrow and Arrow-C++ <https://arrow.apache.org/docs/python/install.html>`__ for interoperability with other applications and fast Parquet reading/writing

.. inclusion-marker-3-do-not-remove

Tutorial
========

**Table of contents:**

(...)

Interactive tutorial
====================

.. Run `this tutorial <https://mybinder.org/v2/gh/scikit-hep/histbook/master?filepath=binder%2Ftutorial.ipynb>`__ on Binder.

(...)

Reference documentation
=======================

(...)

Getting started
---------------

Install awkward-arrays...
