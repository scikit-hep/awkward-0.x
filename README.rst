.. image:: docs/source/logo-300px.png

awkward-array
=============

.. inclusion-marker-1-do-not-remove

Manipulate arrays of complex data structures as easily as Numpy.

.. inclusion-marker-1-5-do-not-remove

awkward-array is a pure Python+Numpy library for manipulating complex data structures as you would Numpy arrays. Even if your data structures

* contain variable-length lists (jagged or ragged),
* are deeply nested (record structure),
* have different data types in the same list (heterogeneous),
* are masked, bit-masked, or index-mapped (nullable),
* contain cross-references or even cyclic references,
* need to be Python class instances on demand,
* are not defined at every point (sparse),
* are not contiguous in memory,
* should not be loaded into memory all at once (lazy),

this library can access them with the efficiency of Numpy arrays. They may be converted from JSON or Python data, loaded from "awkd" files, `HDF5 <https://www.hdfgroup.org>`__, `Parquet <https://parquet.apache.org>`__, or `ROOT <https://root.cern>`__ files, or they may be views into memory buffers like `Arrow <https://arrow.apache.org>`__.

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

Internally, the data has been rearranged into a `columnar <https://towardsdatascience.com/the-beauty-of-column-oriented-data-2945c0c9f560>`__ form, with all values at a given level of hierarchy in the same array. Numpy-like slicing, masking, and fancy indexing are translated into Numpy operations on these internal arrays: they are *not* implemented with Python for loops!

To see some of this structure, ask for the content of the array:

.. code-block:: python

    array.content
    # returns <IndexedMaskedArray [1.1 2.2 None ... <Row 0> None <Row 1>] at 79093e598ef0>

Notice that the boundaries between sub-lists are gone: they exist only at the ``JaggedArray`` level. This ``IndexedMaskedArray`` level handles the ``None`` values in the data. If we dig further, we'll find a ``UnionArray`` to handle the mixture of sub-lists and sub-sub-lists and record structures. If we dig deeply enough, we'll find the numerical data:

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

Rather than matching the speed of compiled code, this can exceed the speed of compiled code (on non-columnar data) because the operation may be vectorized on awkward-array's underlying columnar arrays.

(To do: performance example to substantiate that claim.)

.. inclusion-marker-2-do-not-remove

Installation
============

Install awkward like any other Python package:

.. code-block:: bash

    pip install awkward                       # maybe with sudo or --user, or in virtualenv
    pip install awkward-numba                 # optional: some methods accelerated by Numba

or install with `conda <https://conda.io/en/latest/miniconda.html>`__:

.. code-block:: bash

    conda config --add channels conda-forge   # if you haven't added conda-forge already
    conda install awkward
    conda install awkward-numba               # optional: some methods accelerated by Numba

The base ``awkward`` package requires only `Numpy <https://scipy.org/install.html>`__  (1.13.1+), but ``awkward-numba`` additionally requires `Numba <https://numba.pydata.org/numba-doc/dev/user/installing.html>`__.

Recommended packages:
---------------------

- `pyarrow <https://arrow.apache.org/docs/python/install.html>`__ to view Arrow and Parquet data as awkward-arrays
- `h5py <https://www.h5py.org>`__ to read and write awkward-arrays in HDF5 files

(To do: integration with `Dask <https://pandas.pydata.org>`__, `Pandas <https://pandas.pydata.org>`__, and `Numba <https://pandas.pydata.org>`__.)

.. inclusion-marker-3-do-not-remove

Tutorial
========

**Table of contents:**

* `JSON log data processing example <#json-log-data-processing-example>`__

* Features

  - `Jaggedness <#jaggedness>`__
  - `Record structure <#record-structure>`__
  - `Heterogeneous arrays <#heterogeneous-arrays>`__
  - `Masking <#masking>`__
  - `Cross-references <#cross-references>`__
  - `Class instances and methods <#class-instances-and-methods>`__
  - `Indirection <#indirection>`__
  - `Sparseness <#sparseness>`__
  - `Non-contiguousness <#non-contiguousness>`__
  - `Laziness <#laziness>`__

* `Serialization, reading and writing files <#serialization-reading-and-writing-files>`__

* Detailed particle physics examples

  - `Jagged Lorentz vector arrays; Z peak <#jagged-lorentz-vector-arrays-z-peak>`__
  - `Particle isolation cuts <#particle-isolation-cuts>`__
  - `Generator/reconstructed matching <#generatorreconstructed-matching>`__

(Parquet exoplanets is in the serialization section.)

Interactive tutorial
--------------------

.. Run `this tutorial <https://mybinder.org/v2/gh/scikit-hep/histbook/master?filepath=binder%2Ftutorial.ipynb>`__ on Binder.

(...)

JSON log data processing example
--------------------------------

Jaggedness
----------

Record structure
----------------

Heterogeneous arrays
--------------------

Masking
-------

Cross-references
----------------

Class instances and methods
---------------------------

Indirection
-----------

Sparseness
----------

Non-contiguousness
------------------

Laziness
--------

Serialization, reading and writing files
----------------------------------------

Jagged Lorentz vector arrays; Z peak
------------------------------------

Particle isolation cuts
-----------------------

Generator/reconstructed matching
--------------------------------
