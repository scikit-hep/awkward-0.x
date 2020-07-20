.. image:: docs/source/logo-300px.png

This is a deprecated version of Awkward Array
=============================================

See `scikit-hep/awkward-1.0 <https://github.com/scikit-hep/awkward-1.0#readme>`__ for the latest version of Awkward Array. Old and new versions are available as separate packages,

.. code-block:: bash

    pip install awkward    # old
    pip install awkward1   # new

because the interface has changed. Later this year, "Awkward 1" will simply become the ``awkward`` package with version number 1.0. Then the two packages will shift to

.. code-block:: bash

    pip install awkward    # new
    pip install awkward0   # old

You can adopt the new library gradually. If you want to use some of its features without completely switching over, you can use `ak.from_awkward0 <https://awkward-array.readthedocs.io/en/latest/_auto/ak.from_awkward0.html>`__ and `ak.to_awkward0 <https://awkward-array.readthedocs.io/en/latest/_auto/ak.to_awkward0.html>`__ with the new library loaded as

.. code-block:: python

    import awkward1 as ak

awkward-array
=============

.. image:: https://travis-ci.org/scikit-hep/awkward-array.svg?branch=master
   :target: https://travis-ci.org/scikit-hep/awkward-array

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3275017.svg
   :target: https://doi.org/10.5281/zenodo.3275017

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/scikit-hep/awkward-array/master?urlpath=lab/tree/binder%2Ftutorial.ipynb

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

**Note:** feedback on this project informs the development of `awkward-1.0 <https://github.com/jpivarski/awkward-1.0>`__, a reimplementation in C++ with a simpler user interface, coming in 2020. Leave comments about the future of awkward-array there (as GitHub issues or in the Google Docs).

.. inclusion-marker-2-do-not-remove

Installation
============

Install awkward like any other Python package:

.. code-block:: bash

    pip install awkward                       # maybe with sudo or --user, or in virtualenv

or install with `conda <https://conda.io/en/latest/miniconda.html>`__:

.. code-block:: bash

    conda config --add channels conda-forge   # if you haven't added conda-forge already
    conda install awkward

The base ``awkward`` package requires only `Numpy <https://scipy.org/install.html>`__  (1.13.1+).

Recommended packages:
---------------------

- `pyarrow <https://arrow.apache.org/docs/python/install.html>`__ to view Arrow and Parquet data as awkward-arrays
- `h5py <https://www.h5py.org>`__ to read and write awkward-arrays in HDF5 files
- `Pandas <https://pandas.pydata.org>`__ as an alternative view

.. inclusion-marker-3-do-not-remove

Questions
=========

If you have a question about how to use awkward-array that is not answered in the document below, I recommend asking your question on `StackOverflow <https://stackoverflow.com/questions/tagged/awkward-array>`__ with the ``[awkward-array]`` tag. (I get notified of questions with this tag.)

.. raw:: html

   <p align="center"><a href="https://stackoverflow.com/questions/tagged/awkward-array"><img src="https://cdn.sstatic.net/Sites/stackoverflow/company/img/logos/so/so-logo.png" width="30%"></a></p>

If you believe you have found a bug in awkward-array, post it on the `GitHub issues tab <https://github.com/scikit-hep/awkward-array/issues>`__.

Tutorial
========

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/scikit-hep/awkward-array/master?urlpath=lab/tree/binder%2Ftutorial.ipynb

**Table of contents:**

* `Introduction <#introduction>`__

* `Overview with sample datasets <#overview-with-sample-datasets>`__

  * `NASA exoplanets from a Parquet file <#nasa-exoplanets-from-a-parquet-file>`__

  * `NASA exoplanets from an Arrow buffer <#nasa-exoplanets-from-an-arrow-buffer>`__

  * `Relationship to Pandas <#relationship-to-pandas>`__

  * `LHC data from a ROOT file <#lhc-data-from-a-root-file>`__

* `Awkward-array data model <#awkward-array-data-model>`__

  * `Mutability <#mutability>`__

  * `Relationship to Arrow <#relationship-to-arrow>`__

* `High-level operations common to all classes <#high-level-operations-common-to-all-classes>`__

  * `Slicing with square brackets <#slicing-with-square-brackets>`__

  * `Assigning with square brackets <#assigning-with-square-brackets>`__

  * `Numpy-like broadcasting <#numpy-like-broadcasting>`__

  * `Support for Numpy universal functions (ufuncs) <#support-for-numpy-universal-functions-ufuncs>`__

  * `Global switches <#global-switches>`__

  * `Generic properties and methods <#generic-properties-and-methods>`__

  * `Reducers <#reducers>`__

  * `Properties and methods for jaggedness <#properties-and-methods-for-jaggedness>`__

  * `Properties and methods for tabular columns <#properties-and-methods-for-tabular-columns>`__

  * `Properties and methods for missing values <#properties-and-methods-for-missing-values>`__

  * `Functions for structure manipulation <#functions-for-structure-manipulation>`__

* `Functions for input/output and conversion <#functions-for-inputoutput-and-conversion>`__

* `High-level types <#high-level-types>`__

* `Low-level layouts <#low-level-layouts>`__
    
Introduction
------------

Numpy is great for exploratory data analysis because it encourages the analyst to calculate one operation at a time, rather than one datum at a time. To compute an expression like

.. raw:: html

    <p align="center"><img src="https://latex.codecogs.com/svg.latex?m%3D%5Csqrt%7B(E_1%2BE_2)%5E2-(p_%7Bx1%7D%2Bp_%7Bx2%7D)%5E2-(p_%7By1%7D%2Bp_%7By2%7D)%5E2-(p_%7Bz1%7D%2Bp_%7Bz2%7D)%5E2%7D" title="m=\sqrt{(E_1+E_2)^2-(p_{x1}+p_{x2})^2-(p_{y1}+p_{y2})^2-(p_{z1}+p_{z2})^2}" /></p>

you might first compute ``sqrt((px1 + px2)**2 + (py1 + py2)**2)`` for all data (which is a meaningful quantity: ``pt``), then compute ``sqrt(pt**2 + (pz1 + pz2)**2)`` (another meaningful quantity: ``p``), then compute the whole expression as ``sqrt((E1 + E2)**2 - p**2)``. Performing each step separately on all data lets you plot and cross-check distributions of partial computations, to discover surprises as early as possible.

This order of data processing is called "columnar" in the sense that a dataset may be visualized as a table in which rows are repeated measurements and columns are the different measurable quantities (same layout as `Pandas DataFrames <https://pandas.pydata.org>`__). It is also called "vectorized" in that a Single (virtual) Instruction is applied to Multiple Data (virtual SIMD). Numpy can be hundreds to thousands of times faster than pure Python because it avoids the overhead of handling Python instructions in the loop over numbers. Most data processing languages (R, MATLAB, IDL, all the way back to APL) work this way: an interactive interpreter controlling fast, array-at-a-time math.

However, it's difficult to apply this methodology to non-rectangular data. If your dataset has nested structure, a different number of values per row, different data types in the same column, or cross-references or even circular references, Numpy can't help you.

If you try to make an array with non-trivial types:


.. code-block:: python3

    import numpy
    nested = numpy.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}])
    nested
    # array([{'x': 1, 'y': 1.1}, {'x': 2, 'y': 2.2}, {'x': 3, 'y': 3.3},
    #        {'x': 4, 'y': 4.4}, {'x': 5, 'y': 5.5}], dtype=object)

Numpy gives up and returns a ``dtype=object`` array, which means Python objects and pure Python processing. You don't get the columnar operations or the performance boost.

For instance, you might want to say


.. code-block:: python3

    try:
        nested + 100
    except Exception as err:
        print(type(err), str(err))
    # <class 'TypeError'> unsupported operand type(s) for +: 'dict' and 'int'

but there is no vectorized addition for an array of dicts because there is no addition for dicts defined in pure Python. Numpy is not using its vectorized routines—it's calling Python code on each element.

The same applies to variable-length data, such as lists of lists, where the inner lists have different lengths. This is a more serious shortcoming than the above because the list of dicts (Python's equivalent of an "`array of structs <https://en.wikipedia.org/wiki/AOS_and_SOA>`__") could be manually reorganized into two numerical arrays, ``"x"`` and ``"y"`` (a "`struct of arrays <https://en.wikipedia.org/wiki/AOS_and_SOA>`__"). Not so with a list of variable-length lists.

.. code-block:: python3

    varlen = numpy.array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]])
    varlen
    # array([list([1.1, 2.2, 3.3]), list([]), list([4.4, 5.5]), list([6.6]),
    #        list([7.7, 8.8, 9.9])], dtype=object)

As before, we get a ``dtype=object`` without vectorized methods.

.. code-block:: python3

    try:
        varlen + 100
    except Exception as err:
        print(type(err), str(err))
    # <class 'TypeError'> can only concatenate list (not "int") to list

What's worse, this array looks purely numerical and could have been made by a process that was *supposed* to create equal-length inner lists.

Awkward-array provides a way of talking about these data structures as arrays.

.. code-block:: python3

    import awkward
    nested = awkward.fromiter([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}])
    nested
    # <Table [<Row 0> <Row 1> <Row 2> <Row 3> <Row 4>] at 0x7f25e80a01d0>

This ``Table`` is a columnar data structure with the same meaning as the Python data we built it with. To undo ``awkward.fromiter``, call ``.tolist()``.

.. code-block:: python3

    nested.tolist()
    # [{'x': 1, 'y': 1.1},
    #  {'x': 2, 'y': 2.2},
    #  {'x': 3, 'y': 3.3},
    #  {'x': 4, 'y': 4.4},
    #  {'x': 5, 'y': 5.5}]

Values at the same position of the tree structure are contiguous in memory: this is a struct of arrays.

.. code-block:: python3

    nested.contents["x"]
    # array([1, 2, 3, 4, 5])

    nested.contents["y"]
    # array([1.1, 2.2, 3.3, 4.4, 5.5])

Having a structure like this means that we can perform vectorized operations on the whole structure with relatively few Python instructions (number of Python instructions scales with the complexity of the data type, not with the number of values in the dataset).

.. code-block:: python3

    (nested + 100).tolist()
    # [{'x': 101, 'y': 101.1},
    #  {'x': 102, 'y': 102.2},
    #  {'x': 103, 'y': 103.3},
    #  {'x': 104, 'y': 104.4},
    #  {'x': 105, 'y': 105.5}]

    (nested + numpy.arange(100, 600, 100)).tolist()
    # [{'x': 101, 'y': 101.1},
    #  {'x': 202, 'y': 202.2},
    #  {'x': 303, 'y': 303.3},
    #  {'x': 404, 'y': 404.4},
    #  {'x': 505, 'y': 505.5}]

It's less obvious that variable-length data can be represented in a columnar format, but it can.

.. code-block:: python3

    varlen = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]])
    varlen
    # <JaggedArray [[1.1 2.2 3.3] [] [4.4 5.5] [6.6] [7.7 8.8 9.9]] at 0x7f25bc7b1438>

Unlike Numpy's ``dtype=object`` array, the inner lists are *not* Python lists and the numerical values *are* contiguous in memory. This is made possible by representing the structure (where each inner list starts and stops) in one array and the values in another.

.. code-block:: python3

    varlen.counts, varlen.content
    # (array([3, 0, 2, 1, 3]), array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))

(For fast random access, the more basic representation is ``varlen.offsets``, which is in turn a special case of a ``varlen.starts, varlen.stops`` pair. These details are discussed below.)

A structure like this can be broadcast like Numpy with a small number of Python instructions (scales with the complexity of the data type, not the number of values).

.. code-block:: python3

    varlen + 100
    # <JaggedArray [[101.1 102.2 103.3] [] [104.4 105.5] [106.6] [107.7 108.8 109.9]] at 0x7f25bc7b1400>

    varlen + numpy.arange(100, 600, 100)
    # <JaggedArray [[101.1 102.2 103.3] [] [304.4 305.5] [406.6] [507.7 508.8 509.9]] at 0x7f25bc7b1da0>

You can even slice this object as though it were multidimensional (each element is a tensor of the same rank, but with different numbers of dimensions).

.. code-block:: python3

    # Skip the first two inner lists; skip the last value in each inner list that remains.
    varlen[2:, :-1]
    # <JaggedArray [[4.4] [] [7.7 8.8]] at 0x7f25bc755588>

The data are not rectangular, so some inner lists might have as many elements as your selection. Don't worry—you'll get error messages.

.. code-block:: python3

    try:
        varlen[:, 1]
    except Exception as err:
        print(type(err), str(err))
    # <class 'IndexError'> index 1 is out of bounds for jagged min size 0

Masking with the ``.counts`` is handy because all the Numpy advanced indexing rules apply (in an extended sense) to jagged arrays.

.. code-block:: python3

    varlen[varlen.counts > 1, 1]
    # array([2.2, 5.5, 8.8])

I've only presented the two most important awkward classes, ``Table`` and ``JaggedArray`` (and not how they combine). Each class is presented in more detail below. For now, I'd just like to point out that you can make crazy complicated data structures

.. code-block:: python3

    crazy = awkward.fromiter([[1.21, 4.84, None, 10.89, None],
                              [19.36, [30.25]],
                              [{"x": 36, "y": {"z": 49}}, None, {"x": 64, "y": {"z": 81}}]
                             ])

and they vectorize and slice as expected.

.. code-block:: python3

    numpy.sqrt(crazy).tolist()
    # [[1.1, 2.2, None, 3.3000000000000003, None],
    #  [4.4, [5.5]],
    #  [{'x': 6.0, 'y': {'z': 7.0}}, None, {'x': 8.0, 'y': {'z': 9.0}}]]

This is because any awkward array can be the content of any other awkward array. Like Numpy, the features of awkward-array are simple, yet compose nicely to let you build what you need.

Overview with sample datasets
-----------------------------

Many of the examples in this tutorial use ``awkward.fromiter`` to make awkward arrays from lists and ``array.tolist()`` to turn them back into lists (or dicts for ``Table``, tuples for ``Table`` with anonymous fields, Python objects for ``ObjectArrays``, etc.). These should be considered slow methods, since Python instructions are executed in the loop, but that's a necessary part of examining or building Python objects.

Ideally, you'd want to get your data from a binary, columnar source and produce binary, columnar output, or convert only once and reuse the converted data. `Parquet <https://parquet.apache.org>`__ is a popular columnar format for storing data on disk and `Arrow <https://arrow.apache.org>`__ is a popular columnar format for sharing data in memory (between functions or applications). `ROOT <https://root.cern>`__ is a popular columnar format for particle physicists, and `uproot <https://github.com/scikit-hep/uproot>`__ natively produces awkward arrays from ROOT files.

`HDF5 <https://www.hdfgroup.org>`__ and its Python library `h5py <https://www.h5py.org/>`__ are columnar, but only for rectangular arrays, unlike the others mentioned here. Awkward-array can *wrap* HDF5 with an interpretation layer to store columnar data structures, but then the awkward-array library wuold be needed to read the data back in a meaningful way. Awkward also has a native file format, ``.awkd`` files, which are simply ZIP archives of columns as binary blobs and metadata (just as Numpy's ``.npz`` is a ZIP of arrays with metadata). The HDF5, awkd, and pickle serialization procedures use the same protocol, which has backward and forward compatibility features.

NASA exoplanets from a Parquet file
"""""""""""""""""""""""""""""""""""

Let's start by opening a Parquet file. Awkward reads Parquet through the `pyarrow <https://arrow.apache.org/docs/python>`__ module, which is an optional dependency, so be sure you have it installed before trying the next line.

.. code-block:: python3

    stars = awkward.fromparquet("tests/samples/exoplanets.parquet")
    stars
    # <ChunkedArray [<Row 0> <Row 1> <Row 2> ... <Row 2932> <Row 2933> <Row 2934>] at 0x7f25b9c67780>

(There is also an ``awkward.toparquet`` that takes the file name and array as arguments.)

Columns are accessible with square brackets and strings

.. code-block:: python3

    stars["name"]
    # <ChunkedArray ['11 Com' '11 UMi' '14 And' ... 'tau Gem' 'ups And' 'xi Aql'] at 0x7f25b9c67dd8>

or by dot-attribute (if the name doesn't have weird characters and doesn't conflict with a method or property name).

.. code-block:: python3

    stars.ra, stars.dec
    # (<ChunkedArray [185.179276 229.27453599999998 352.822571 ... 107.78488200000001 24.199345 298.56201200000004] at 0x7f25b94ccf28>,
    #  <ChunkedArray [17.792868 71.823898 39.236198 ... 30.245163 41.40546 8.461452] at 0x7f25b94cca90>)

This file contains data about extrasolar planets and their host stars. As such, it's a ``Table`` full of Numpy arrays and ``JaggedArrays``. The star attributes (`"name"`, `"ra"` or right ascension in degrees, `"dec"` or declination in degrees, `"dist"` or distance in parsecs, `"mass"` in multiples of the sun's mass, and `"radius"` in multiples of the sun's radius) are plain Numpy arrays and the planet attributes (`"name"`, `"orbit"` or orbital distance in AU, `"eccen"` or eccentricity, `"period"` or periodicity in days, `"mass"` in multiples of Jupyter's mass, and `"radius"` in multiples of Jupiter's radius) are jagged because each star may have a different number of planets.

.. code-block:: python3

    stars.planet_name
    # <ChunkedArray [['b'] ['b'] ['b'] ... ['b'] ['b' 'c' 'd'] ['b']] at 0x7f25b94dc550>

    stars.planet_period, stars.planet_orbit
    # (<ChunkedArray [[326.03] [516.21997] [185.84] ... [305.5] [4.617033 241.258 1276.46] [136.75]] at 0x7f25b94cccc0>,
    #  <ChunkedArray [[1.29] [1.53] [0.83] ... [1.17] [0.059222000000000004 0.827774 2.51329] [0.68]] at 0x7f25b94cc978>)

For large arrays, only the first and last values are printed: the second-to-last star has three planets; all the other stars shown here have one planet.

These arrays are called ``ChunkedArrays`` because the Parquet file is lazily read in chunks (Parquet's row group structure). The ``ChunkedArray`` (subdivides the file) contains ``VirtualArrays`` (read one chunk on demand), which generate the ``JaggedArrays``. This is an illustration of how each awkward class provides one feature, and you get desired behavior by combining them.

The ``ChunkedArrays`` and ``VirtualArrays`` support the same Numpy-like access as ``JaggedArray``, so we can compute with them just as we would any other array.

.. code-block:: python3

    # distance in parsecs → distance in light years
    stars.dist * 3.26156
    # <ChunkedArray [304.5318572 410.0433232 246.5413204 ... 367.38211839999997 43.7375196 183.5279812] at 0x7f25b94cce80>

    # for all stars, drop the first planet
    stars.planet_mass[:, 1:]
    # <ChunkedArray [[] [] [] ... [] [1.981 4.132] []] at 0x7f25b94ccf60>

NASA exoplanets from an Arrow buffer
""""""""""""""""""""""""""""""""""""

The pyarrow implementation of Arrow is more complete than its implementation of Parquet, so we can use more features in the Arrow format, such as nested tables.

Unlike Parquet, which is intended as a file format, Arrow is a memory format. You might get an Arrow buffer as the output of another function, through interprocess communication, from a network RPC call, a message bus, etc. Arrow can be saved as files, though this isn't common. In this case, we'll get it from a file.

.. code-block:: python3

    import pyarrow
    arrow_buffer = pyarrow.ipc.open_file(open("tests/samples/exoplanets.arrow", "rb")).get_batch(0)
    stars = awkward.fromarrow(arrow_buffer)
    stars
    # <Table [<Row 0> <Row 1> <Row 2> ... <Row 2932> <Row 2933> <Row 2934>] at 0x7f25b94f2518>

(There is also an ``awkward.toarrow`` that takes an awkward array as its only argument, returning the relevant Arrow structure.)

This file is structured differently. Instead of jagged arrays of numbers like ``"planet_mass"``, ``"planet_period"``, and ``"planet_orbit"``, this file has a jagged table of ``"planets"``. A jagged table is a ``JaggedArray`` of ``Table``.

.. code-block:: python3

    stars["planets"]
    # <JaggedArray [[<Row 0>] [<Row 1>] [<Row 2>] ... [<Row 3928>] [<Row 3929> <Row 3930> <Row 3931>] [<Row 3932>]] at 0x7f25b94fb080>

Notice that the square brackets are nested, but the contents are ``<Row>`` objects. The second-to-last star has three planets, as before.

We can find the non-jagged ``Table`` in the ``JaggedArray.content``.

.. code-block:: python3

    stars["planets"].content
    # <Table [<Row 0> <Row 1> <Row 2> ... <Row 3930> <Row 3931> <Row 3932>] at 0x7f25b94f2d68>

When viewed as Python lists and dicts, the ``'planets'`` field is a list of planet dicts, each with its own fields.

.. code-block:: python3

    stars[:2].tolist()
    # [{'dec': 17.792868,
    #   'dist': 93.37,
    #   'mass': 2.7,
    #   'name': '11 Com',
    #   'planets': [{'eccen': 0.231,
    #     'mass': 19.4,
    #     'name': 'b',
    #     'orbit': 1.29,
    #     'period': 326.03,
    #     'radius': nan}],
    #   'ra': 185.179276,
    #   'radius': 19.0},
    #  {'dec': 71.823898,
    #   'dist': 125.72,
    #   'mass': 2.78,
    #   'name': '11 UMi',
    #   'planets': [{'eccen': 0.08,
    #     'mass': 14.74,
    #     'name': 'b',
    #     'orbit': 1.53,
    #     'period': 516.21997,
    #     'radius': nan}],
    #   'ra': 229.27453599999998,
    #   'radius': 29.79}]

Despite being packaged in an arguably more intuitive way, we can still get jagged arrays of numbers by requesting ``"planets"`` and a planet attribute (two column selections) without specifying which star or which parent.

.. code-block:: python3

    stars.planets.name
    # <JaggedArray [['b'] ['b'] ['b'] ... ['b'] ['b' 'c' 'd'] ['b']] at 0x7f25b94dc780>

    stars.planets.mass
    # <JaggedArray [[19.4] [14.74] [4.8] ... [20.6] [0.6876 1.981 4.132] [2.8]] at 0x7f25b94fb240>

Even though the ``Table`` is hidden inside the ``JaggedArray``, its ``columns`` pass through to the top.

.. code-block:: python3

    stars.columns
    # ['dec', 'dist', 'mass', 'name', 'planets', 'ra', 'radius']

    stars.planets.columns
    # ['eccen', 'mass', 'name', 'orbit', 'period', 'radius']

For a more global view of the structures contained within one of these arrays, print out its high-level type. ("High-level" because it presents logical distinctions, like jaggedness and tables, but not physical distinctions, like chunking and virtualness.)

.. code-block:: python3

    print(stars.type)
    # [0, 2935) -> 'dec'     -> float64
    #              'dist'    -> float64
    #              'mass'    -> float64
    #              'name'    -> <class 'str'>
    #              'planets' -> [0, inf) -> 'eccen'  -> float64
    #                                       'mass'   -> float64
    #                                       'name'   -> <class 'str'>
    #                                       'orbit'  -> float64
    #                                       'period' -> float64
    #                                       'radius' -> float64
    #              'ra'      -> float64
    #              'radius'  -> float64

The above should be read like a function's data type: ``argument type -> return type`` for the function that takes an index in square brackets and returns something else. For example, the first ``[0, 2935)`` means that you could put any non-negative integer less than ``2935`` in square brackets after ``stars``, like this:

.. code-block:: python3

    stars[1734]
    # <Row 1734>

and get an object that would take ``'dec'``, ``'dist'``, ``'mass'``, ``'name'``, ``'planets'``, ``'ra'``, or ``'radius'`` in its square brackets. The return type depends on which of those strings you provide.

.. code-block:: python3

    stars[1734]["mass"]   # type is float64
    # 0.54

    stars[1734]["name"]   # type is <class 'str'>
    # 'Kepler-186'

    stars[1734]["planets"]
    # <Table [<Row 2192> <Row 2193> <Row 2194> <Row 2195> <Row 2196>] at 0x7f25b94dc438>

The planets have their own table structure:

.. code-block:: python3

    print(stars[1734]["planets"].type)
    # [0, 5) -> 'eccen'  -> float64
    #           'mass'   -> float64
    #           'name'   -> <class 'str'>
    #           'orbit'  -> float64
    #           'period' -> float64
    #           'radius' -> float64

Notice that within the context of ``stars``, the ``planets`` could take any non-negative integer ``[0, inf)``, but for a particular star, the allowed domain is known with more precision: ``[0, 5)``. This is because ``stars["planets"]`` is a jagged array—a different number of planets for each star—but one ``stars[1734]["planets"]`` is a simple array—five planets for *this* star.

Passing a non-negative integer less than 5 to this array, we get an object that takes one of six strings: : ``'eccen'``, ``'mass'``, ``'name'``, ``'orbit'``, ``'period'``, and ``'radius'``.

.. code-block:: python3

    stars[1734]["planets"][4]
    # <Row 2196>

and the return type of these depends on which string you provide.

.. code-block:: python3

    stars[1734]["planets"][4]["period"]   # type is float
    # 129.9441

    stars[1734]["planets"][4]["name"]   # type is <class 'str'>
    # 'f'

    stars[1734]["planets"][4].tolist()
    # {'eccen': 0.04,
    #  'mass': nan,
    #  'name': 'f',
    #  'orbit': 0.432,
    #  'period': 129.9441,
    #  'radius': 0.10400000000000001}

(Incidentally, this is a `potentially habitable exoplanet <https://www.nasa.gov/ames/kepler/kepler-186f-the-first-earth-size-planet-in-the-habitable-zone>`__, the first ever discovered.)

.. code-block:: python3

    stars[1734]["name"], stars[1734]["planets"][4]["name"]
    # ('Kepler-186', 'f')

Some of these arguments "commute" and others don't. Dimensional axes have a particular order, so you can't request a planet by its row number before selecting a star, but you can swap a column-selection (string) and a row-selection (integer). For a rectangular table, it's easy to see how you can slice column-first or row-first, but it even works when the table is jagged.

.. code-block:: python3

    stars["planets"]["name"][1734][4]
    # 'f'

    stars[1734]["planets"][4]["name"]
    # 'f'

None of these intermediate slices actually process data, so you can slice in any order that is logically correct without worrying about performance. Projections, even multi-column projections

.. code-block:: python3

    orbits = stars["planets"][["name", "eccen", "orbit", "period"]]
    orbits[1734].tolist()
 
In this representation, each star's attributes must be duplicated for all of its planets, and it is not possible to show stars that have no planets (not present in this dataset), but the information is preserved in a way that Pandas can recognize and operate on. (For instance, .unstack() would widen each planet attribute into a separate column per planet and simplify the index to strictly one row per star.)
The limitation is that only a single jagged structure can be represented by a DataFrame. The structure can be arbitrarily deep in Tables (which add depth to the column names),

.. code-block:: python3

    array = awkward.fromiter([{"a": {"b": 1, "c": {"d": [2]}}, "e": 3},

    stars[1734]["planets"][4]["name"]
    # 'f'

None of these intermediate slices actually process data, so you can slice in any order that is logically correct without worrying about performance. Projections,
even multi-column projections

.. code-block:: python3

    orbits = stars["planets"][["name", "eccen", "orbit", "period"]]
    orbits[1734].tolist()
    # [{'name': 'b', 'eccen': nan, 'orbit': 0.0343, 'period': 3.8867907},
    #  {'name': 'c', 'eccen': nan, 'orbit': 0.0451, 'period': 7.267302},
    #  {'name': 'd', 'eccen': nan, 'orbit': 0.0781, 'period': 13.342996},
    #  {'name': 'e', 'eccen': nan, 'orbit': 0.11, 'period': 22.407704},
    #  {'name': 'f', 'eccen': 0.04, 'orbit': 0.432, 'period': 129.9441}]

are a useful way to restructure data without incurring a runtime cost.

Relationship to Pandas
""""""""""""""""""""""

Arguably, this kind of dataset could be manipulated as a `Pandas DataFrame <https://pandas.pydata.org>`__ instead of awkward arrays. Despite the variable number of planets per star, the exoplanets dataset could be flattened into a rectangular DataFrame, in which the distinction between solar systems is represented by a two-component index (leftmost pair of columns below), a `MultiIndex <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html>`__.

.. code-block:: python3

    awkward.topandas(stars, flatten=True)[-9:]

.. raw:: html

      <table border="0" class="dataframe">
        <thead>
          <tr>
            <th></th>
            <th></th>
            <th>dec</th>
            <th>dist</th>
            <th>mass</th>
            <th>name</th>
            <th colspan="6" halign="left">planets</th>
            <th>ra</th>
            <th>radius</th>
          </tr>
          <tr>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>eccen</th>
            <th>mass</th>
            <th>name</th>
            <th>orbit</th>
            <th>period</th>
            <th>radius</th>
            <th></th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th rowspan="4" valign="top">2931</th>
            <th>0</th>
            <td>-15.937480</td>
            <td>3.60</td>
            <td>0.78</td>
            <td>49</td>
            <td>0.1800</td>
            <td>0.01237</td>
            <td>101</td>
            <td>0.538000</td>
            <td>162.870000</td>
            <td>NaN</td>
            <td>26.017012</td>
            <td>NaN</td>
          </tr>
          <tr>
            <th>1</th>
            <td>-15.937480</td>
            <td>3.60</td>
            <td>0.78</td>
            <td>49</td>
            <td>0.1600</td>
            <td>0.01237</td>
            <td>102</td>
            <td>1.334000</td>
            <td>636.130000</td>
            <td>NaN</td>
            <td>26.017012</td>
            <td>NaN</td>
          </tr>
          <tr>
            <th>2</th>
            <td>-15.937480</td>
            <td>3.60</td>
            <td>0.78</td>
            <td>49</td>
            <td>0.0600</td>
            <td>0.00551</td>
            <td>103</td>
            <td>0.133000</td>
            <td>20.000000</td>
            <td>NaN</td>
            <td>26.017012</td>
            <td>NaN</td>
          </tr>
          <tr>
            <th>3</th>
            <td>-15.937480</td>
            <td>3.60</td>
            <td>0.78</td>
            <td>49</td>
            <td>0.2300</td>
            <td>0.00576</td>
            <td>104</td>
            <td>0.243000</td>
            <td>49.410000</td>
            <td>NaN</td>
            <td>26.017012</td>
            <td>NaN</td>
          </tr>
          <tr>
            <th>2932</th>
            <th>0</th>
            <td>30.245163</td>
            <td>112.64</td>
            <td>2.30</td>
            <td>53</td>
            <td>0.0310</td>
            <td>20.60000</td>
            <td>98</td>
            <td>1.170000</td>
            <td>305.500000</td>
            <td>NaN</td>
            <td>107.784882</td>
            <td>26.80</td>
          </tr>
          <tr>
            <th rowspan="3" valign="top">2933</th>
            <th>0</th>
            <td>41.405460</td>
            <td>13.41</td>
            <td>1.30</td>
            <td>48</td>
            <td>0.0215</td>
            <td>0.68760</td>
            <td>98</td>
            <td>0.059222</td>
            <td>4.617033</td>
            <td>NaN</td>
            <td>24.199345</td>
            <td>1.56</td>
          </tr>
          <tr>
            <th>1</th>
            <td>41.405460</td>
            <td>13.41</td>
            <td>1.30</td>
            <td>48</td>
            <td>0.2596</td>
            <td>1.98100</td>
            <td>99</td>
            <td>0.827774</td>
            <td>241.258000</td>
            <td>NaN</td>
            <td>24.199345</td>
            <td>1.56</td>
          </tr>
          <tr>
            <th>2</th>
            <td>41.405460</td>
            <td>13.41</td>
            <td>1.30</td>
            <td>48</td>
            <td>0.2987</td>
            <td>4.13200</td>
            <td>100</td>
            <td>2.513290</td>
            <td>1276.460000</td>
            <td>NaN</td>
            <td>24.199345</td>
            <td>1.56</td>
          </tr>
          <tr>
            <th>2934</th>
            <th>0</th>
            <td>8.461452</td>
            <td>56.27</td>
            <td>2.20</td>
            <td>55</td>
            <td>0.0000</td>
            <td>2.80000</td>
            <td>98</td>
            <td>0.680000</td>
            <td>136.750000</td>
            <td>NaN</td>
            <td>298.562012</td>
            <td>12.00</td>
          </tr>
        </tbody>
      </table>

In this representation, each star's attributes must be duplicated for all of its planets, and it is not possible to show stars that have no planets (not present in this dataset), but the information is preserved in a way that Pandas can recognize and operate on. (For instance, ``.unstack()`` would widen each planet attribute into a separate column per planet and simplify the index to strictly one row per star.)

The limitation is that only a single jagged structure can be represented by a DataFrame. The structure can be arbitrarily deep in ``Tables`` (which add depth to the column names),

.. code-block:: python3

    array = awkward.fromiter([{"a": {"b": 1, "c": {"d": [2]}}, "e": 3},
                              {"a": {"b": 4, "c": {"d": [5, 5.1]}}, "e": 6},
                              {"a": {"b": 7, "c": {"d": [8, 8.1, 8.2]}}, "e": 9}])
    awkward.topandas(array, flatten=True)

.. raw:: html    

      <table border="0" class="dataframe">
        <thead>
          <tr>
            <th></th>
            <th></th>
            <th colspan="2" halign="left">a</th>
            <th>e</th>
          </tr>
          <tr>
            <th></th>
            <th></th>
            <th>b</th>
            <th>c</th>
            <th></th>
          </tr>
          <tr>
            <th></th>
            <th></th>
            <th></th>
            <th>d</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>0</th>
            <th>0</th>
            <td>1</td>
            <td>2.0</td>
            <td>3</td>
          </tr>
          <tr>
            <th rowspan="2" valign="top">1</th>
            <th>0</th>
            <td>4</td>
            <td>5.0</td>
            <td>6</td>
          </tr>
          <tr>
            <th>1</th>
            <td>4</td>
            <td>5.1</td>
            <td>6</td>
          </tr>
          <tr>
            <th rowspan="3" valign="top">2</th>
            <th>0</th>
            <td>7</td>
            <td>8.0</td>
            <td>9</td>
          </tr>
          <tr>
            <th>1</th>
            <td>7</td>
            <td>8.1</td>
            <td>9</td>
          </tr>
          <tr>
            <th>2</th>
            <td>7</td>
            <td>8.2</td>
            <td>9</td>
          </tr>
        </tbody>
      </table>

and arbitrarily deep in ``JaggedArrays`` (which add depth to the row names),

.. code-block:: python3

    array = awkward.fromiter([{"a": 1, "b": [[2.2, 3.3, 4.4], [], [5.5, 6.6]]},
                              {"a": 10, "b": [[1.1], [2.2, 3.3], [], [4.4]]},
                              {"a": 100, "b": [[], [9.9]]}])
    awkward.topandas(array, flatten=True)

.. raw:: html
    
      <table border="0" class="dataframe">
        <thead>
          <tr>
            <th></th>
            <th></th>
            <th></th>
            <th>a</th>
            <th>b</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th rowspan="5" valign="top">0</th>
            <th rowspan="3" valign="top">0</th>
            <th>0</th>
            <td>1</td>
            <td>2.2</td>
          </tr>
          <tr>
            <th>1</th>
            <td>1</td>
            <td>3.3</td>
          </tr>
          <tr>
            <th>2</th>
            <td>1</td>
            <td>4.4</td>
          </tr>
          <tr>
            <th rowspan="2" valign="top">2</th>
            <th>0</th>
            <td>1</td>
            <td>5.5</td>
          </tr>
          <tr>
            <th>1</th>
            <td>1</td>
            <td>6.6</td>
          </tr>
          <tr>
            <th rowspan="4" valign="top">1</th>
            <th>0</th>
            <th>0</th>
            <td>10</td>
            <td>1.1</td>
          </tr>
          <tr>
            <th rowspan="2" valign="top">1</th>
            <th>0</th>
            <td>10</td>
            <td>2.2</td>
          </tr>
          <tr>
            <th>1</th>
            <td>10</td>
            <td>3.3</td>
          </tr>
          <tr>
            <th>3</th>
            <th>0</th>
            <td>10</td>
            <td>4.4</td>
          </tr>
          <tr>
            <th>2</th>
            <th>1</th>
            <th>0</th>
            <td>100</td>
            <td>9.9</td>
          </tr>
        </tbody>
      </table>

and they can even have two ``JaggedArrays`` at the same level if their number of elements is the same (at all levels of depth).

.. code-block:: python3

    array = awkward.fromiter([{"a": [[1.1, 2.2, 3.3], [], [4.4, 5.5]], "b": [[1, 2, 3], [], [4, 5]]},
                              {"a": [[1.1], [2.2, 3.3], [], [4.4]],    "b": [[1], [2, 3], [], [4]]},
                              {"a": [[], [9.9]],                       "b": [[], [9]]}])
    awkward.topandas(array, flatten=True)

.. raw:: html

      <table border="0" class="dataframe">
        <thead>
          <tr>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th>a</th>
            <th>b</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th rowspan="5" valign="top">0</th>
            <th rowspan="3" valign="top">0</th>
            <th>0</th>
            <th>0</th>
            <td>1.1</td>
            <td>1</td>
          </tr>
          <tr>
            <th>1</th>
            <th>1</th>
            <td>2.2</td>
            <td>2</td>
          </tr>
          <tr>
            <th>2</th>
            <th>2</th>
            <td>3.3</td>
            <td>3</td>
          </tr>
          <tr>
            <th rowspan="2" valign="top">2</th>
            <th>0</th>
            <th>0</th>
            <td>4.4</td>
            <td>4</td>
          </tr>
          <tr>
            <th>1</th>
            <th>1</th>
            <td>5.5</td>
            <td>5</td>
          </tr>
          <tr>
            <th rowspan="4" valign="top">1</th>
            <th>0</th>
            <th>0</th>
            <th>0</th>
            <td>1.1</td>
            <td>1</td>
          </tr>
          <tr>
            <th rowspan="2" valign="top">1</th>
            <th>0</th>
            <th>0</th>
            <td>2.2</td>
            <td>2</td>
          </tr>
          <tr>
            <th>1</th>
            <th>1</th>
            <td>3.3</td>
            <td>3</td>
          </tr>
          <tr>
            <th>3</th>
            <th>0</th>
            <th>0</th>
            <td>4.4</td>
            <td>4</td>
          </tr>
          <tr>
            <th>2</th>
            <th>1</th>
            <th>0</th>
            <th>0</th>
            <td>9.9</td>
            <td>9</td>
          </tr>
        </tbody>
      </table>

But if there are two ``JaggedArrays`` with *different* structure at the same level, a single DataFrame cannot represent them.

.. code-block:: python3

    array = awkward.fromiter([{"a": [1, 2, 3], "b": [1.1, 2.2]},
                              {"a": [1],       "b": [1.1, 2.2, 3.3]},
                              {"a": [1, 2],    "b": []}])
    try:
        awkward.topandas(array, flatten=True)
    except Exception as err:
        print(type(err), str(err))
    # <class 'ValueError'> this array has more than one jagged array structure

To describe data like these, you'd need two DataFrames, and any calculations involving both ``"a"`` and ``"b"`` would have to include a join on those DataFrames. Awkward arrays are not limited in this way: the last ``array`` above is a valid awkward array and is useful for calculations that mix ``"a"`` and ``"b"``.

LHC data from a ROOT file
"""""""""""""""""""""""""

Particle physicsts need structures like these—in fact, they have been a staple of particle physics analyses for decades. The `ROOT <https://root.cern>`__ file format was developed in the mid-90's to serialize arbitrary C++ data structures in a columnar way (replacing ZEBRA and similar Fortran projects that date back to the 70's). The `PyROOT <https://root.cern.ch/pyroot>`__ library dynamically wraps these objects to present them in Python, though with a performance penalty. The `uproot <https://github.com/scikit-hep/uproot>`__ library reads columnar data directly from ROOT files in Python without intermediary C++.

.. code-block:: python3

    import uproot
    events = uproot.open("http://scikit-hep.org/uproot/examples/HZZ-objects.root")["events"].lazyarrays()
    events
    # <Table [<Row 0> <Row 1> <Row 2> ... <Row 2418> <Row 2419> <Row 2420>] at 0x781189cd7b70>

    events.columns
    # ['jetp4',
    #  'jetbtag',
    #  'jetid',
    #  'muonp4',
    #  'muonq',
    #  'muoniso',
    #  'electronp4',
    #  'electronq',
    #  'electroniso',
    #  'photonp4',
    #  'photoniso',
    #  'MET',
    #  'MC_bquarkhadronic',
    #  'MC_bquarkleptonic',
    #  'MC_wdecayb',
    #  'MC_wdecaybbar',
    #  'MC_lepton',
    #  'MC_leptonpdgid',
    #  'MC_neutrino',
    #  'num_primaryvertex',
    #  'trigger_isomu24',
    #  'eventweight']

This is a typical particle physics dataset (though small!) in that it represents the momentum and energy (``"p4"`` for `Lorentz 4-momentum <https://en.wikipedia.org/wiki/Four-vector>`__) of several different species of particles: ``"jet"``, ``"muon"``, ``"electron"``, and ``"photon"``. Each collision can produce a different number of particles in each species. Other variables, such as missing transverse energy or ``"MET"``, have one value per collision event. Events with zero particles in a species are valuable for the event-level data.

.. code-block:: python3

    # The first event has two muons.
    events.muonp4
    # <ChunkedArray [[TLorentzVector(-52.899, -11.655, -8.1608, 54.779) TLorentzVector(37.738, 0.69347, -11.308, 39.402)] [TLorentzVector(-0.81646, -24.404, 20.2, 31.69)] [TLorentzVector(48.988, -21.723, 11.168, 54.74) TLorentzVector(0.82757, 29.801, 36.965, 47.489)] ... [TLorentzVector(-29.757, -15.304, -52.664, 62.395)] [TLorentzVector(1.1419, 63.61, 162.18, 174.21)] [TLorentzVector(23.913, -35.665, 54.719, 69.556)]] at 0x781189cd7fd0>

    # The first event has zero jets.
    events.jetp4
    # <ChunkedArray [[] [TLorentzVector(-38.875, 19.863, -0.89494, 44.137)] [] ... [TLorentzVector(-3.7148, -37.202, 41.012, 55.951)] [TLorentzVector(-36.361, 10.174, 226.43, 229.58) TLorentzVector(-15.257, -27.175, 12.12, 33.92)] []] at 0x781189cd7be0>

    # Every event has exactly one MET.
    events.MET
    # <ChunkedArray [TVector2(5.9128, 2.5636) TVector2(24.765, -16.349) TVector2(-25.785, 16.237) ... TVector2(18.102, 50.291) TVector2(79.875, -52.351) TVector2(19.714, -3.5954)] at 0x781189cfe780>

Unlike the exoplanet data, these events cannot be represented as a DataFrame because of the different numbers of particles in each species and because zero-particle events have value. Even with just ``"muonp4"``, ``"jetp4"``, and ``"MET"``, there is no translation.

.. code-block:: python3

    try:
        awkward.topandas(events[["muonp4", "jetp4", "MET"]], flatten=True)
    except Exception as err:
        print(type(err), str(err))
    # <class 'NameError'> name 'awkward' is not defined

It could be described as a collection of DataFrames, in which every operation relating particles in the same event would require a join. But that would make analysis harder, not easier. An event has meaning on its own.

.. code-block:: python3

    events[0].tolist()
    # {'jetp4': [],
    #  'jetbtag': [],
    #  'jetid': [],
    #  'muonp4': [TLorentzVector(-52.899, -11.655, -8.1608, 54.779),
    #   TLorentzVector(37.738, 0.69347, -11.308, 39.402)],
    #  'muonq': [1, -1],
    #  'muoniso': [4.200153350830078, 2.1510612964630127],
    #  'electronp4': [],
    #  'electronq': [],
    #  'electroniso': [],
    #  'photonp4': [],
    #  'photoniso': [],
    #  'MET': TVector2(5.9128, 2.5636),
    #  'MC_bquarkhadronic': TVector3(0, 0, 0),
    #  'MC_bquarkleptonic': TVector3(0, 0, 0),
    #  'MC_wdecayb': TVector3(0, 0, 0),
    #  'MC_wdecaybbar': TVector3(0, 0, 0),
    #  'MC_lepton': TVector3(0, 0, 0),
    #  'MC_leptonpdgid': 0,
    #  'MC_neutrino': TVector3(0, 0, 0),
    #  'num_primaryvertex': 6,
    #  'trigger_isomu24': True,
    #  'eventweight': 0.009271008893847466}

Particle physics isn't alone in this: analyzing JSON-formatted log files in production systems or allele likelihoods in genomics are two other fields where variable-length, nested structures can help. Arbitrary data structures are useful and working with them in columns provides a new way to do exploratory data analysis: one array at a time.

Awkward-array data model
------------------------

Awkward array features are provided by a suite of classes that each extend Numpy arrays in one small way. These classes may then be composed to combine features.

In this sense, Numpy arrays are awkward-array's most basic array class. A Numpy array is a small Python object that points to a large, contiguous region of memory, and, as much as possible, operations replace or change the small Python object, not the big data buffer. Therefore, many Numpy operations are *views*, rather than *in-place operations* or *copies*, leaving the original value intact but returning a new value that is linked to the original. Assigning to arrays and in-place operations are allowed, but they are more complicated to use because one must be aware of which arrays are views and which are copies.

Awkward-array's model is to treat all arrays as though they were immutable, favoring views over copies, and not providing any high-level in-place operations on low-level memory buffers (i.e. no in-place assignment).

Numpy provides complete control over the interpretation of an ``N`` dimensional array. A Numpy array has a `dtype <https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html>`__ to interpret bytes as signed and unsigned integers of various bit-widths, floating-point numbers, booleans, little endian and big endian, fixed-width bytestrings (for applications such as 6-byte MAC addresses or human-readable strings with padding), or `record arrays <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`__ for contiguous structures. A Numpy array has a `pointer <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.ctypes.html>`__ to the first element of its data buffer (``array.ctypes.data``) and a `shape <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html>`__ to describe its ``N`` dimensions as a rank-``N`` tensor. Only ``shape[0]`` is the length as returned by the Python function ``len``. Furthermore, an `order <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flags.html>`__ flag determines if rank > 1 arrays are laid out in "C" order or "Fortran" order. A Numpy array also has a `stride <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.strides.html>`__ to determine how many bytes separate one element from the next. (Data in a Numpy array need not be strictly contiguous, but they must be regular: the number of bytes seprating them is a constant.) This stride may even be negative to describe a reversed view of an array, which allows any ``slice`` of an array, even those with ``skip != 1`` to be a view, rather than a copy. Numpy arrays also have flags to determine whether they `own <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flags.html>`__ their data buffer (and should therefore delete it when the Python object goes out of scope) and whether the data buffer is `writable <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flags.html>`__.


The biggest restriction on this data model is that Numpy arrays are strictly rectangular. The ``shape`` and ``stride`` are constants, enforcing a regular layout. Awkward's ``JaggedArray`` is a generalization of Numpy's rank-2 arrays—that is, arrays of arrays—in that the inner arrays of a ``JaggedArray`` may all have different lengths. For higher ranks, such as arrays of arrays of arrays, put a ``JaggedArray`` inside another as its ``content``. An important special case of ``JaggedArray`` is ``StringArray``, whose ``content`` is interpreted as characters (with or without encoding), which represents an array of strings without unnecessary padding, as in Numpy's case.

Although Numpy's `record arrays <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`__ present a buffer as a table, with differently typed, named columns, that table must be contiguous or interleaved (with non-trivial ``strides``) in memory: an `array of structs <https://en.wikipedia.org/wiki/AOS_and_SOA>`__. Awkward's ``Table`` provides the same interface, except that each column may be anywhere in memory, stored in a ``contents`` dict mapping field names to arrays. This is a true generalization: a ``Table`` may be a wrapped view of a Numpy record array, but not vice-versa. Use a ``Table`` anywhere you'd have a record/class/struct in non-columnar data structures. A ``Table`` with anonymous (integer-valued, rather than string-valued) fields is like an array of strongly typed tuples.

Numpy has a `masked array <https://docs.scipy.org/doc/numpy/reference/maskedarray.html>`__ module for nullable data—values that may be "missing" (like Python's ``None``). Naturally, the only kinds of arrays Numpy can mask are subclasses of its own ``ndarray``, and we need to be able to mask any awkward array, so the awkward library defines its own ``MaskedArray``. Additionally, we sometimes want to mask with bits, rather than bytes (e.g. for Arrow compatibility), so there's a ``BitMaskedArray``, and sometimes we want to mask large structures without using memory for the masked-out values, so there's an ``IndexedMaskedArray`` (fusing the functionality of a ``MaskedArray`` with an ``IndexedArray``).

Numpy has no provision for an array containing different data types ("heterogeneous"), but awkward-array has a ``UnionArray``. The ``UnionArray`` stores data for each type as separate ``contents`` and identifies the types and positions of each element in the ``contents`` using ``tags`` and ``index`` arrays (equivalent to Arrow's `dense union type <https://arrow.apache.org/docs/memory_layout.html#dense-union-type>`__ with ``types`` and ``offsets`` buffers). As a data type, unions are a counterpart to records or tuples (making ``UnionArray`` a counterpart to ``Table``): each record/tuple contains *all* of its ``contents`` but a union contains *any* of its ``contents``. (Note that a ``UnionArray`` may be the best way to interleave two arrays, even if they have the same type. Heterogeneity is not a necessary feature of a ``UnionArray``.)

Numpy has a ``dtype=object`` for arrays of Python objects, but awkward's ``ObjectArray`` creates Python objects on demand from array data. A large dataset of some ``Point`` class, containing floating-point members ``x`` and ``y``, can be stored as an ``ObjectArray`` of a ``Table`` of ``x`` and ``y`` with much less memory than a Numpy array of ``Point`` objects. The ``ObjectArray`` has a ``generator`` function that produces Python objects from array elements.  ``StringArray`` is also a special case of ``ObjectArray``, which instantiates variable-length character contents as Python strings.

Although an ``ObjectArray`` can save memory, creating Python objects in a loop may still use more computation time than is necessary. Therefore, awkward arrays can also have vectorized ``Methods``—bound functions that operate on the array data, rather than instantiating every Python object in an ``ObjectArray``. Although an ``ObjectArray`` is a good use-case for ``Methods``, any awkward array can have them. (The second most common case being a ``JaggedArray`` of ``ObjectArrays``.)

The nesting of awkward arrays within awkward arrays need not be tree-like: they can have cross-references and cyclic references (using ordinary Python assignment). ``IndexedArray`` can aid in building complex structures: it is simply an integer ``index`` that would be applied to its ``content`` with `integer array indexing <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#integer-array-indexing>`__ to get any element. ``IndexedArray`` is the equivalent of a pointer in non-columnar data structures.

The counterpart of an ``IndexedArray`` is a ``SparseArray``: whereas an ``IndexedArray`` consists of pointers *to* elements of its ``content``, a ``SparseArray`` consists of pointers *from* elements of its content, representing a very large array in terms of its non-zero (or non-``default``) elements. Awkward's ``SparseArray`` is a `coordinate format (COO) <https://scipy-lectures.org/advanced/scipy_sparse/coo_matrix.html>`__, one-dimensional array.

Another limitation of Numpy is that arrays cannot span multiple memory buffers. Awkward's ``ChunkedArray`` represents a single logical array made of physical ``chunks`` that may be anywhere in memory. A ``ChunkedArray``'s ``chunksizes`` may be known or unknown. One application of ``ChunkedArray`` is to append data to an array without allocating on every call: ``AppendableArray`` allocates memory in equal-sized chunks.

Another application of ``ChunkedArray`` is to lazily load data in chunks. Awkward's ``VirtualArray`` calls its ``generator`` function to materialize an array when needed, and a ``ChunkedArray`` of ``VirtualArrays`` is a classic lazy-loading array, used to gradually read Parquet and ROOT files. In most libraries, lazy-loading is not a part of the data but a feature of the reading interface. Nesting virtualness makes it possible to load ``Tables`` within ``Tables``, where even the columns of the inner ``Tables`` are on-demand.

For more details, see `array classes <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc>`__.

* `Jaggedness <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#jaggedness>`__

  * `JaggedArray <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#jaggedarray>`__

  * `Helper functions <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#helper-functions>`__

* `Product types <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#product-types>`__

  * `Table <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#table>`__

* `Sum types <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#sum-types>`__

  * `UnionArray <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#unionarray>`__

* `Option types <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#option-types>`__

  * `MaskedArray <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#maskedarray>`__

  * `BitMaskedArray <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#bitmaskedarray>`__

  * `IndexedMaskedArray <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#indexedmaskedarray>`__

* `Indirection <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#indirection>`__

  * `IndexedArray <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#indexedarray>`__

  * `SparseArray <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#sparsearray>`__

  * `Helper functions <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#helper-functions-1>`__

* `Opaque objects <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#opaque-objects>`__

  * `Mix-in Methods <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#mix-in-methods>`__

  * `ObjectArray <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#objectarray>`__

  * `StringArray <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#stringarray>`__

* `Non-contiguousness <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#non-contiguousness>`__

  * `ChunkedArray <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#chunkedarray>`__

  * `AppendableArray <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#appendablearray>`__

* `Laziness <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#laziness>`__

  * `VirtualArray <https://github.com/scikit-hep/awkward-array/blob/master/docs/classes.adoc#virtualarray>`__

Mutability
""""""""""

Awkward arrays are considered immutable in the sense that elements of the data cannot be modified in-place. That is, assignment with square brackets at an integer index raises an error. Awkward does not prevent the underlying Numpy arrays from being modified in-place, though that can lead to confusing results—the behavior is left undefined. The reason for this omission in functionality is that the internal representation of columnar data structures is more constrained than their non-columnar counterparts: some in-place modification can't be defined, and others have surprising side-effects.

However, the Python objects representing awkward arrays can be changed in-place. Each class has properties defining its structure, such as ``content``, and these may be replaced at any time. (Replacing properties does not change values in any Numpy arrays.) In fact, this is the only way to build cyclic references: an object in Python must be assigned to a name before that name can be used as a reference.

Awkward arrays are appendable, but only through ``AppendableArray``, and ``Table`` columns may be added, changed, or removed. The only use of square-bracket assignment (i.e. ``__setitem__``) is to modify ``Table`` columns.

Awkward arrays produced by an external program may grow continuously, as long as more deeply nested arrays are filled first. That is, the ``content`` of a ``JaggedArray`` must be updated before updating its structure arrays (``starts`` and ``stops``). The definitions of awkward array validity allow for nested elements with no references pointing at them ("unreachable" elements), but not for references pointing to a nested element that doesn't exist.

Relationship to Arrow
"""""""""""""""""""""

`Apache Arrow <https://arrow.apache.org>`__ is a cross-language, columnar memory format for complex data structures. There is intentionally a high degree of overlap between awkward-array and Arrow. But whereas Arrow's focus is data portability, awkward's focus is computation: it would not be unusual to get data from Arrow, compute something with awkward-array, then return it to another Arrow buffer. For this reason, ``awkward.fromarrow`` is a zero-copy view. Awkward's data representation is broader than Arrow's, so ``awkward.toarrow`` does, in general, perform a copy.

The main difference between awkward-array and Arrow is that awkward-array does not require all arrays to be included within a contiguous memory buffer, though libraries like `pyarrow <https://arrow.apache.org/docs/python>`__ relax this criterion while building a compliant Arrow buffer. This restriction does imply that Arrow cannot encode cross-references or cyclic dependencies.

Arrow also doesn't have the luxury of relying on Numpy to define its `primitive arrays <https://arrow.apache.org/docs/memory_layout.html#primitive-value-arrays>`__, so it has a fixed endianness, has no regular tensors without expressing it as a jagged array, and requires 32-bit integers for indexing, instead of taking whatever integer type a user provides.

`Nullability <https://arrow.apache.org/docs/memory_layout.html#null-bitmaps>`__ is an optional property of every data type in Arrow, but it's a structure element in awkward. Similarly, `dictionary encoding <https://arrow.apache.org/docs/memory_layout.html#dictionary-encoding>`__ is built into Arrow as a fundamental property, but it would be built from an ``IndexedArray`` in awkward. Chunking and lazy-loading are supported by readers such as `pyarrow <https://arrow.apache.org/docs/python>`__, but they're not part of the Arrow data model.

The following list translates awkward-array classes and features to their Arrow counterparts, if possible.

* ``JaggedArray``: Arrow's `list type <https://arrow.apache.org/docs/memory_layout.html#list-type>`__.
* ``Table``: Arrow's `struct type <https://arrow.apache.org/docs/memory_layout.html#struct-type>`__, though columns can be added to or removed from awkward ``Tables`` whereas Arrow is strictly immutable.
* ``BitMaskedArray``: every data type in Arrow potentially has a `null bitmap <https://arrow.apache.org/docs/memory_layout.html#null-bitmaps>`__, though it's an explicit array structure in awkward. (Arrow has no counterpart for Awkward's ``MaskedArray`` or ``IndexedMaskedArray``.)
* ``UnionArray``: directly equivalent to Arrow's `dense union <https://arrow.apache.org/docs/memory_layout.html#dense-union-type>`__. Arrow also has a `sparse union <https://arrow.apache.org/docs/memory_layout.html#sparse-union-type>`__, which awkward-array only has as a ``UnionArray.fromtags`` constructor that builds the dense union on the fly from a sparse union.
* ``ObjectArray`` and ``Methods``: no counterpart because Arrow must be usable in any language.
* ``StringArray``: "string" is a logical type built on top of Arrow's `list type <https://arrow.apache.org/docs/memory_layout.html#list-type>`__.
* ``IndexedArray``: no counterpart (though its role in building `dictionary encoding <https://arrow.apache.org/docs/memory_layout.html#dictionary-encoding>`__ is built into Arrow as a fundamental property).
* ``SparseArray``: no counterpart.
* ``ChunkedArray``: no counterpart (though a reader may deal with non-contiguous data).
* ``AppendableArray``: no counterpart; Arrow is strictly immutable.
* ``VirtualArray``: no counterpart (though a reader may lazily load data).

High-level operations: common to all classes
--------------------------------------------

There are three levels of abstraction in awkward-array: high-level operations for data analysis, low-level operations for engineering the structure of the data, and implementation details. Implementation details are handled in the usual way for Python: if exposed at all, class, method, and function names begin with underscores and are not guaranteed to be stable from one release to the next.

The distinction between high-level operations and low-level operations is more subtle and developed as awkward-array was put to use. Data analysts care about the logical structure of the data—whether it is jagged, what the column names are, whether certain values could be ``None``, etc. Data engineers (or an analyst in "engineering mode") care about contiguousness, how much data are in memory at a given time, whether strings are dictionary-encoded, whether arrays have unreachable elements, etc. The dividing line is between high-level types and low-level array layout (both of which are defined in their own sections below). The following awkward classes have the same high-level type as their content:

* ``IndexedArray`` because indirection to type ``T`` has type ``T``,
* ``SparseArray`` because a lookup of elements with type ``T`` has type ``T``,
* ``ChunkedArray`` because the chunks, which must have the same type as each other, collectively have that type when logically concatenated,
* ``AppendableArray`` because it's a special case of ``ChunkedArray``,
* ``VirtualArray`` because it produces an array of a given type on demand,
* ``UnionArray`` has the same type as its ``contents`` *only if* all ``contents`` have the same type as each other.

All other classes, such as ``JaggedArray``, have a logically distinct type from their contents.

This section describes a suite of operations that are common to all awkward classes. For some high-level types, the operation is meaningless or results in an error, such as the jagged ``counts`` of an array that is not jagged at any level, or the ``columns`` of an array that contains no tables, but the operation has a well-defined action on every array class. To use these operations, you do need to understand the high-level type of your data, but not whether it is wrapped in an ``IndexedArray``, a ``SparseArray``, a ``ChunkedArray``, an ``AppendableArray``, or a ``VirtualArray``.

Slicing with square brackets
""""""""""""""""""""""""""""

The primary operation for all classes is slicing with square brackets. This is the operation defined by Python's ``__getitem__`` method. It is so basic that high-level types are defined in terms of what they return when a scalar argument is passed in square brakets.

Just as Numpy's slicing reproduces but generalizes Python sequence behavior, awkward-array reproduces (most of) `Numpy's slicing behavior <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`__ and generalizes it in certain cases. An integer argument, a single slice argument, a single Numpy array-like of booleans or integers, and a tuple of any of the above is handled just like Numpy. Awkward-array does not handle ellipsis (because the depth of an awkward array can be different on different branches of a ``Table`` or ``UnionArray``) or ``None`` (because it's not always possible to insert a ``newaxis``). Numpy `record arrays <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`__ accept a string or sequence of strings as a column argument if it is the only argument, not in a tuple with other types. Awkward-array accepts a string or sequence of strings if it contains a ``Table`` at some level.

An integer argument selects one element from the top-level array (starting at zero), changing the type by decreasing rank or jaggedness by one level.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8], [9.9]])
    a[0]
    # array([1.1, 2.2, 3.3])

Negative indexes count backward from the last element,

.. code-block:: python3

    a[-1]
    # array([9.9])

and the index (after translating negative indexes) must be at least zero and less than the length of the top-level array.

.. code-block:: python3

    try:
        a[-6]
    except Exception as err:
        print(type(err), str(err))
    # <class 'IndexError'> index -6 is out of bounds for axis 0 with size 5

A slice selects a range of elements from the top-level array, maintaining the array's type. The first index is the inclusive starting point (starting at zero) and the second index is the exclusive endpoint.

.. code-block:: python3

    a[2:4]
    # <JaggedArray [[4.4 5.5] [6.6 7.7 8.8]] at 0x7811883f8390>

Python's slice syntax (above) or literal ``slice`` objects may be used.

.. code-block:: python3

    a[slice(2, 4)]
    # <JaggedArray [[4.4 5.5] [6.6 7.7 8.8]] at 0x7811883f8630>

Negative indexes count backward from the last element and endpoints may be omitted.

.. code-block:: python3

    a[-2:]
    # <JaggedArray [[6.6 7.7 8.8] [9.9]] at 0x7811883f8978>

Start and endpoints beyond the array are not errors: they are truncated.

.. code-block:: python3

    a[2:100]
    # <JaggedArray [[4.4 5.5] [6.6 7.7 8.8] [9.9]] at 0x7811883f8be0>

A skip value (third index of the slice) sets the stride for indexing, allowing you to skip elements, and this skip can be negative. It cannot, however, be zero.

.. code-block:: python3

    a[::-1]
    # <JaggedArray [[9.9] [6.6 7.7 8.8] [4.4 5.5] [] [1.1 2.2 3.3]] at 0x7811883f8ef0>

A Numpy array-like of booleans with the same length as the array may be used to filter elements. Numpy has a specialized `numpy.compress <https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html>`__ function for this operation, but the only way to get it in awkward-array is through square brackets.

.. code-block:: python3

    a[[True, True, False, True, False]]
    # <JaggedArray [[1.1 2.2 3.3] [] [6.6 7.7 8.8]] at 0x781188407278>

A Numpy array-like of integers with the same length as the array may be used to select a collection of indexes. Numpy has a specialized `numpy.take <https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html>`__ function for this operation, but the only way to get it in awkward-array is through square brakets. Negative indexes and repeated elements are handled in the same way as Numpy.

.. code-block:: python3

    a[[-1, 0, 1, 2, 2, 2]]
    # <JaggedArray [[9.9] [1.1 2.2 3.3] [] [4.4 5.5] [4.4 5.5] [4.4 5.5]] at 0x781188407550>

A tuple of length ``N`` applies selections to the first ``N`` levels of rank or jaggedness. Our example array has only two levels, so we can apply two kinds of indexes.

.. code-block:: python3

    a[2:, 0]
    # array([4.4, 6.6, 9.9])

    a[[True, False, True, True, False], ::-1]
    # <JaggedArray [[3.3 2.2 1.1] [5.5 4.4] [8.8 7.7 6.6]] at 0x7811884079e8>

    a[[0, 3, 0], 1::]
    # <JaggedArray [[2.2 3.3] [7.7 8.8] [2.2 3.3]] at 0x781188407cc0>

As described in Numpy's `advanced indexing <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing>`__, advanced indexes (boolean or integer arrays) are broadcast and iterated as one:

.. code-block:: python3

    a[[0, 3], [True, False, True]]
    # array([1.1, 8.8])

Awkward array has two extensions beyond Numpy, both of which affect only jagged data. If an array is jagged and a jagged array of booleans with the same structure (same length at all levels) is passed in square brackets, only inner arrays would be filtered.

.. code-block:: python3

    a    = awkward.fromiter([[  1.1,   2.2,  3.3], [], [ 4.4,  5.5], [ 6.6,  7.7,   8.8], [  9.9]])
    mask = awkward.fromiter([[False, False, True], [], [True, True], [True, True, False], [False]])
    a[mask]
    # <JaggedArray [[3.3] [] [4.4 5.5] [6.6 7.7] []] at 0x7811883f8f60>

Similarly, if an array is jagged and a jagged array of integers with the same structure is passed in square brackets, only inner arrays would be filtered/duplicated/rearranged.

.. code-block:: python3

    a     = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8], [9.9]])
    index = awkward.fromiter([[2, 2, 2, 2], [], [1, 0], [2, 1, 0], []])
    a[index]
    # <JaggedArray [[3.3 3.3 3.3 3.3] [] [5.5 4.4] [8.8 7.7 6.6] []] at 0x78118847acf8>

Although all of the above use a ``JaggedArray`` as an example, the principles are general: you should get analogous results with jagged tables, masked jagged arrays, etc. Non-jagged arrays only support Numpy-like slicing.

If an array contains a ``Table``, it can be selected with a string or a sequence of strings, just like Numpy `record arrays <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`__.

.. code-block:: python3

    a = awkward.fromiter([{"x": 1, "y": 1.1, "z": "one"}, {"x": 2, "y": 2.2, "z": "two"}, {"x": 3, "y": 3.3, "z": "three"}])
    a
    # <Table [<Row 0> <Row 1> <Row 2>] at 0x7811883930f0>

    a["x"]
    # array([1, 2, 3])

    a[["z", "y"]].tolist()
    # [{'z': 'one', 'y': 1.1}, {'z': 'two', 'y': 2.2}, {'z': 'three', 'y': 3.3}]

Like Numpy, integer indexes and string indexes commute if the integer index corresponds to a structure outside the ``Table`` (this condition is always met for Numpy record arrays).

.. code-block:: python3

    a["y"][1]
    # 2.2

    a[1]["y"]
    # 2.2

    a = awkward.fromiter([[{"x": 1, "y": 1.1, "z": "one"}, {"x": 2, "y": 2.2, "z": "two"}], [], [{"x": 3, "y": 3.3, "z": "three"}]])
    a
    # <JaggedArray [[<Row 0> <Row 1>] [] [<Row 2>]] at 0x781188407358>

    a["y"][0][1]
    # 2.2

    a[0]["y"][1]
    # 2.2

    a[0][1]["y"]
    # 2.2

but not

.. code-block:: python3

    a = awkward.fromiter([{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.1, 2.2]}, {"x": 3, "y": [3.1, 3.2, 3.3]}])
    a
    # <Table [<Row 0> <Row 1> <Row 2>] at 0x7811883934a8>

    a["y"][2][1]
    # 3.2

    a[2]["y"][1]
    # 3.2

    try:
        a[2][1]["y"]
    except Exception as err:
        print(type(err), str(err))
    # <class 'AttributeError'> no column named '_util_isstringslice'

because

.. code-block:: python3

    a[2].tolist()
    # {'x': 3, 'y': [3.1, 3.2, 3.3]}

cannot take a ``1`` argument before ``"y"``.

Just as integer indexes can be alternated with string/sequence of string indexes, so can slices, arrays, and tuples of slices and arrays.

.. code-block:: python3

    a["y"][:, 0]
    # array([1.1, 2.1, 3.1])

Generally speaking, string and sequence of string indexes are *column* indexes, while all other types are *row* indexes.

Assigning with square brackets
""""""""""""""""""""""""""""""

As discussed above, awkward arrays are generally immutable with few exceptions. Row assignment is only possible via appending to an ``AppendableArray``. Column assignment, reassignment, and deletion are in general allowed. The syntax for assigning and reassigning columns is through assignment to a square bracket expression. This operation is defined by Python's ``__setitem__`` method. The syntax for deleting columns is through the ``del`` operators on a square bracket expression. This operation is defined by Python's ``__delitem__`` method.

Since only columns can be changed, only strings and sequences of strings are allowed as indexes.

.. code-block:: python3

    a = awkward.fromiter([[{"x": 1, "y": 1.1, "z": "one"}, {"x": 2, "y": 2.2, "z": "two"}], [], [{"x": 3, "y": 3.3, "z": "three"}]])
    a
    # <JaggedArray [[<Row 0> <Row 1>] [] [<Row 2>]] at 0x7811883905c0>

    a["a"] = awkward.fromiter([[100, 200], [], [300]])
    a.tolist()
    # [[{'x': 1, 'y': 1.1, 'z': 'one', 'a': 100},
    #   {'x': 2, 'y': 2.2, 'z': 'two', 'a': 200}],
    #  [],
    #  [{'x': 3, 'y': 3.3, 'z': 'three', 'a': 300}]]

    del a["a"]
    a.tolist()
    # [[{'x': 1, 'y': 1.1, 'z': 'one'}, {'x': 2, 'y': 2.2, 'z': 'two'}],
    #  [],
    #  [{'x': 3, 'y': 3.3, 'z': 'three'}]]

    a[["a", "b"]] = awkward.fromiter([[{"first": 100, "second": 111}, {"first": 200, "second": 222}], [], [{"first": 300, "second": 333}]])
    a.tolist()
    # [[{'x': 1, 'y': 1.1, 'z': 'one', 'a': 100, 'b': 111},
    #   {'x': 2, 'y': 2.2, 'z': 'two', 'a': 200, 'b': 222}],
    #  [],
    #  [{'x': 3, 'y': 3.3, 'z': 'three', 'a': 300, 'b': 333}]]

Note that the names of the columns on the right-hand side of the assignment are irrelevant; we're setting two columns, there needs to be two columns on the right. Columns can be anonymous:

.. code-block:: python3

    a[["a", "b"]] = awkward.Table(awkward.fromiter([[100, 200], [], [300]]), awkward.fromiter([[111, 222], [], [333]]))
    a.tolist()
    # [[{'x': 1, 'y': 1.1, 'z': 'one', 'a': 100, 'b': 111},
    #   {'x': 2, 'y': 2.2, 'z': 'two', 'a': 200, 'b': 222}],
    #  [],
    #  [{'x': 3, 'y': 3.3, 'z': 'three', 'a': 300, 'b': 333}]]

Another thing to note is that the structure (lengths at all levels of jaggedness) must match if the depth is the same.

.. code-block:: python3

    try:
        a["c"] = awkward.fromiter([[100, 200, 300], [400], [500, 600]])
    except Exception as err:
        print(type(err), str(err))
    # <class 'ValueError'> cannot broadcast JaggedArray to match JaggedArray with a different counts

But if the right-hand side is shallower and can be *broadcasted* to the left-hand side, it will be. (See below for broadcasting.)

.. code-block:: python3

    a["c"] = awkward.fromiter([100, 200, 300])
    a.tolist()
    # [[{'x': 1, 'y': 1.1, 'z': 'one', 'a': 100, 'b': 111, 'c': 100},
    #   {'x': 2, 'y': 2.2, 'z': 'two', 'a': 200, 'b': 222, 'c': 100}],
    #  [],
    #  [{'x': 3, 'y': 3.3, 'z': 'three', 'a': 300, 'b': 333, 'c': 300}]]

Numpy-like broadcasting
"""""""""""""""""""""""

In assignments and mathematical operations between higher-rank and lower-rank arrays, Numpy repeats values in the lower-rank array to "fit," if possible, before applying the operation. This is called `boradcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`__. For example,

.. code-block:: python3

    numpy.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]) + 100
    # array([[101.1, 102.2, 103.3],
    #        [104.4, 105.5, 106.6]])

Singletons are also expanded to fit.

.. code-block:: python3

    numpy.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]) + numpy.array([[100], [200]])
    # array([[101.1, 102.2, 103.3],
    #        [204.4, 205.5, 206.6]])

Awkward arrays have the same feature, but this has particularly useful effects for jagged arrays. In an operation involving two arrays of different depths of jaggedness, the shallower one expands to fit the deeper one.

.. code-block:: python3

    awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]]) + awkward.fromiter([100, 200, 300])
    # <JaggedArray [[101.1 102.2 103.3] [] [304.4 305.5]] at 0x781188390940>

Note that the ``100`` was broadcasted to all three of the elements of the first inner array, ``200`` was broadcasted to no elements in the second inner array (because the second inner array is empty), and ``300`` was broadcasted to all two of the elements of the third inner array.

This is the columnar equivalent to accessing a variable defined outside of an inner loop.

.. code-block:: python3

    jagged = [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
    flat = [100, 200, 300]
    for i in range(3):
        for j in range(len(jagged[i])):
            # j varies in this loop, but i is constant
            print(i, j, jagged[i][j] + flat[i])
    # 0 0 101.1
    # 0 1 102.2
    # 0 2 103.3
    # 2 0 304.4
    # 2 1 305.5

Many translations of non-columnar code to columnar code has this form. It's often surprising to users that they don't have to do anything special to get this feature (e.g. ``cross``).

Support for Numpy universal functions (ufuncs)
""""""""""""""""""""""""""""""""""""""""""""""

Numpy's key feature of array-at-a-time programming is mainly provided by "universal functions" or "ufuncs." This is a special class of function that applies a scalars → scalar kernel independently to aligned elements of internal arrays to return a same-shape output array. That is, for a scalars → scalar function ``f(x1, ..., xN) → y``, the ufunc takes ``N`` input arrays of the same ``shape`` and returns one output array with that ``shape`` in which ``output[i] = f(input1[i], ..., inputN[i])`` for all ``i``.

.. code-block:: python3

    # N = 1
    numpy.sqrt(numpy.array([1, 4, 9, 16, 25]))
    # array([1., 2., 3., 4., 5.])

    # N = 2
    numpy.add(numpy.array([[1.1, 2.2], [3.3, 4.4]]), numpy.array([[100, 200], [300, 400]]))
    # array([[101.1, 202.2],
    #        [303.3, 404.4]])

Keep in mind that a ufunc is not simply a function that has this property, but a specially named class, deriving from a type in the Numpy library.

.. code-block:: python3

    numpy.sqrt, numpy.add
    # (<ufunc 'sqrt'>, <ufunc 'add'>)

    isinstance(numpy.sqrt, numpy.ufunc), isinstance(numpy.add, numpy.ufunc)
    # (True, True)

This class of functions can be overridden, and awkward-array overrides them to recognize and properly handle awkward arrays.

.. code-block:: python3

    numpy.sqrt(awkward.fromiter([[1, 4, 9], [], [16, 25]]))
    # <JaggedArray [[1.0 2.0 3.0] [] [4.0 5.0]] at 0x7811883f88d0>

    numpy.add(awkward.fromiter([[[1.1], 2.2], [], [3.3, None]]), awkward.fromiter([[[100], 200], [], [None, 300]]))
    # <JaggedArray [[[101.1] 202.2] [] [None None]] at 0x7811883f8d68>

Only the primary action of the ufunc (``ufunc.__call__``) has been overridden; methods like ``ufunc.at``, ``ufunc.reduce``, and ``ufunc.reduceat`` are not supported. Also, the in-place ``out`` parameter is not supported because awkward array data cannot be changed in-place.

For awkward arrays, the input arguments to a ufunc must all have the same structure or, if shallower, be broadcastable to the deepest structure. (See above for "broadcasting.") The scalar function is applied to elements at the same positions within this structure from different input arrays. The output array has this structure, populated by return values of the scalar function.

* Rectangular arrays must have the same shape, just as in Numpy. A scalar can be broadcasted (expanded) to have the same shape as the arrays.
* Jagged arrays must have the same number of elements in all inner arrays. A rectangular array with the same outer shape (i.e. containing scalars instead of inner arrays) can be broadcasted to inner arrays with the same lengths.
* Tables must have the same sets of columns (though not necessarily in the same order). There is no broadcasting of missing columns.
* Missing values (``None`` from ``MaskedArrays``) transform to missing values in every ufunc. That is, ``None + 5`` is ``None``, ``None + None`` is ``None``, etc.
* Different data types (through a ``UnionArray``) must be compatible at every site where values are included in the calculation. For instance, input arrays may contain tables with different sets of columns, but all inputs at index ``i`` must have the same sets of columns as each other:

.. code-block:: python3

    numpy.add(awkward.fromiter([{"x": 1, "y": 1.1}, {"y": 1.1, "z": 100}]),
              awkward.fromiter([{"x": 3, "y": 3.3}, {"y": 3.3, "z": 300}])).tolist()
    # [{'x': 4, 'y': 4.4}, {'y': 4.4, 'z': 400}]

Unary and binary operations on awkward arrays, such as ``-x``, ``x + y``, and ``x**2``, are actually Numpy ufuncs, so all of the above applies to them as well (such as broadcasting the scalar ``2`` in ``x**2``).

Remember that only ufuncs have been overridden by awkward-array: other Numpy functions such as ``numpy.concatenate`` are ignorant of awkward arrays and will attempt to convert them to Numpy first. In some cases, that may be what you want, but in many, especially any cases involving jagged arrays, it will be a major performance loss and a loss of functionality: jagged arrays turn into Numpy ``dtype=object`` arrays containing Numpy arrays, which can be a very large number of Python objects and doesn't behave as a multidimensional array.

You can check to see if a function from Numpy is a ufunc with ``isinstance``.

.. code-block:: python3

    isinstance(numpy.concatenate, numpy.ufunc)
    # False

and you can prevent accidental conversions to Numpy by setting ``allow_tonumpy`` to ``False``, either on one array or globally on a whole class of awkward arrays. (See "global switches" below.)

.. code-block:: python3

    x = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    y = awkward.fromiter([[6.6, 7.7, 8.8], [9.9]])
    numpy.concatenate([x, y])
    # array([array([1.1, 2.2, 3.3]), array([], dtype=float64),
    #        array([4.4, 5.5]), array([6.6, 7.7, 8.8]), array([9.9])],
    #       dtype=object)

    x.allow_tonumpy = False
    try:
        numpy.concatenate([x, y])
    except Exception as err:
        print(type(err), str(err))
    # <class 'RuntimeError'> awkward.array.base.AwkwardArray.allow_tonumpy is False; refusing to convert to Numpy

Global switches
"""""""""""""""

The ``AwkwardArray`` abstract base class has the following switches to turn off sometmes-undesirable behavior. These switches could be set on the ``AwkwardArray`` class itself, affecting all awkward arrays, or they could be set on a particular class like ``JaggedArray`` to only affect ``JaggedArray`` instances, or they could be set on a particular instance, to affect only that instance.

* ``allow_tonumpy`` (default is ``True``); if ``False``, forbid any action that would convert an awkward array into a Numpy array (with a likely loss of performance and functionality).
* ``allow_iter`` (default is ``True``); if ``False``, forbid any action that would iterate over an awkward array in Python (except printing a few elements as part of its string representation).
* ``check_prop_valid`` (default is ``True``); if ``False``, skip the single-property validity checks in array constructors and when setting properties.
* ``check_whole_valid`` (default is ``True``); if ``False``, skip the whole-array validity checks that are typically called before methods that need them.

.. code-block:: python3

    awkward.AwkwardArray.check_prop_valid
    # True

    awkward.JaggedArray.check_whole_valid
    # True

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    numpy.array(a)
    # array([array([1.1, 2.2, 3.3]), array([], dtype=float64),
    #        array([4.4, 5.5])], dtype=object)

    a.allow_tonumpy = False
    try:
        numpy.array(a)
    except Exception as err:
        print(type(err), str(err))
    # <class 'RuntimeError'> awkward.array.base.AwkwardArray.allow_tonumpy is False; refusing to convert to Numpy

    list(a)
    # [array([1.1, 2.2, 3.3]), array([], dtype=float64), array([4.4, 5.5])]

    a.allow_iter = False
    try:
        list(a)
    except Exception as err:
        print(type(err), str(err))
    # <class 'RuntimeError'> awkward.array.base.AwkwardArray.allow_iter is False; refusing to iterate

    a
    # <JaggedArray [[1.1 2.2 3.3] [] [4.4 5.5]] at 0x78118847ae10>

Generic properties and methods
""""""""""""""""""""""""""""""

All awkward arrays have the following properties and methods.

* ``type``: the high-level type of the array. (See below for a detailed description of high-level types.)

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    b = awkward.fromiter([[1.1, 2.2, None, 3.3, None],
                          [4.4, [5.5]],
                          [{"x": 6, "y": {"z": 7}}, None, {"x": 8, "y": {"z": 9}}]
                         ])

    a.type
    # ArrayType(3, inf, dtype('float64'))

    print(a.type)
    # [0, 3) -> [0, inf) -> float64

    b.type
    # ArrayType(3, inf, OptionType(UnionType(dtype('float64'), ArrayType(inf, dtype('float64')), TableType(x=dtype('int64'), y=TableType(z=dtype('int64'))))))

    print(b.type)
    # [0, 3) -> [0, inf) -> ?((float64             |
    #                          [0, inf) -> float64 |
    #                          'x' -> int64
    #                          'y' -> 'z' -> int64 ))

* ``layout``: the low-level layout of the array. (See below for a detailed description of low-level layouts.)

.. code-block:: python3

    a.layout
    #  layout
    # [    ()] JaggedArray(starts=layout[0], stops=layout[1], content=layout[2])
    # [     0]   ndarray(shape=3, dtype=dtype('int64'))
    # [     1]   ndarray(shape=3, dtype=dtype('int64'))
    # [     2]   ndarray(shape=5, dtype=dtype('float64'))

    b.layout
    #  layout
    # [           ()] JaggedArray(starts=layout[0], stops=layout[1], content=layout[2])
    # [            0]   ndarray(shape=3, dtype=dtype('int64'))
    # [            1]   ndarray(shape=3, dtype=dtype('int64'))
    # [            2]   IndexedMaskedArray(mask=layout[2, 0], content=layout[2, 1], maskedwhen=-1)
    # [         2, 0]     ndarray(shape=10, dtype=dtype('int64'))
    # [         2, 1]     UnionArray(tags=layout[2, 1, 0], index=layout[2, 1, 1], contents=[layout[2, 1, 2], layout[2, 1, 3], layout[2, 1, 4]])
    # [      2, 1, 0]       ndarray(shape=7, dtype=dtype('uint8'))
    # [      2, 1, 1]       ndarray(shape=7, dtype=dtype('int64'))
    # [      2, 1, 2]       ndarray(shape=4, dtype=dtype('float64'))
    # [      2, 1, 3]       JaggedArray(starts=layout[2, 1, 3, 0], stops=layout[2, 1, 3, 1], content=layout[2, 1, 3, 2])
    # [   2, 1, 3, 0]         ndarray(shape=1, dtype=dtype('int64'))
    # [   2, 1, 3, 1]         ndarray(shape=1, dtype=dtype('int64'))
    # [   2, 1, 3, 2]         ndarray(shape=1, dtype=dtype('float64'))
    # [      2, 1, 4]       Table(x=layout[2, 1, 4, 0], y=layout[2, 1, 4, 1])
    # [   2, 1, 4, 0]         ndarray(shape=2, dtype=dtype('int64'))
    # [   2, 1, 4, 1]         Table(z=layout[2, 1, 4, 1, 0])
    # [2, 1, 4, 1, 0]           ndarray(shape=2, dtype=dtype('int64'))

* ``dtype``: the `Numpy dtype <https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html>`__ that this array would have if cast as a Numpy array. Numpy dtypes cannot fully specify awkward arrays: use the ``type`` for an analyst-friendly description of the data type or ``layout`` for details about how the arrays are represented.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    a.dtype   # the closest Numpy dtype to a jagged array is dtype=object ('O')
    # dtype('O')

    numpy.array(a)
    # array([array([1.1, 2.2, 3.3]), array([], dtype=float64),
    #        array([4.4, 5.5])], dtype=object)

* ``shape``: the `Numpy shape <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html>`__ that this array would have if cast as a Numpy array. This only specifies the first regular dimensions, not any jagged dimensions or regular dimensions nested within awkward structures. The Python length (``__len__``) of the array is the first element of this ``shape``.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    a.shape
    # (3,)

    len(a)
    # 3

The following ``JaggedArray`` has two fixed-size dimensions at the top, followed by a jagged dimension inside of that. The shape only represents the first few dimensions.

.. code-block:: python3

    a = awkward.JaggedArray.fromcounts([[3, 0], [2, 4]], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    a
    # <JaggedArray [[[1.1 2.2 3.3] []] [[4.4 5.5] [6.6 7.7 8.8 9.9]]] at 0x7811883bc0b8>

    a.shape
    # (2, 2)

    len(a)
    # 2

    print(a.type)
    # [0, 2) -> [0, 2) -> [0, inf) -> float64

Also, a dimension can effectively be fixed-size, but represented by a ``JaggedArray``. The ``shape`` does not encompass any dimensions represented by a ``JaggedArray``.

.. code-block:: python3

    # Same structure, but it's JaggedArrays all the way down.
    b = a.structure1d()
    b
    # <JaggedArray [[[1.1 2.2 3.3] []] [[4.4 5.5] [6.6 7.7 8.8 9.9]]] at 0x781188407240>

    b.shape
    # (2,)

* ``size``: the product of ``shape``, as in Numpy.

.. code-block:: python3

    a.shape
    # (2, 2)

    a.size
    # 4

* ``nbytes``: the total number of bytes in all memory buffers referenced by the array, not including bytes in Python objects (which are Python-implementation dependent, not even available in PyPy). Same as the Numpy property of the same name.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    a.nbytes
    # 72

    a.offsets.nbytes + a.content.nbytes
    # 72

* ``tolist()``: converts the array into Python objects: ``lists`` for arrays, ``dicts`` for table rows, ``tuples`` for table rows with anonymous fields and a ``rowname`` of ``"tuple"``, ``None`` for missing data, and Python objects from ``ObjectArrays``. This is an approximate inverse of ``awkward.fromiter``.

.. code-block:: python3

    awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).tolist()
    # [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    awkward.fromiter([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]).tolist()
    # [{'x': 1, 'y': 1.1}, {'x': 2, 'y': 2.2}, {'x': 3, 'y': 3.3}]

    awkward.Table.named("tuple", [1, 2, 3], [1.1, 2.2, 3.3]).tolist()
    # [(1, 1.1), (2, 2.2), (3, 3.3)]

    awkward.fromiter([[1.1, 2.2, None], [], [None, 3.3]]).tolist()
    # [[1.1, 2.2, None], [], [None, 3.3]]

    class Point:
        def __init__(self, x, y):
            self.x, self.y = x, y
        def __repr__(self):
            return f"Point({self.x}, {self.y})"

    a = awkward.fromiter([[Point(1, 1.1), Point(2, 2.2), Point(3, 3.3)], [], [Point(4, 4.4), Point(5, 5.5)]])
    a
    # <JaggedArray [[Point(1, 1.1) Point(2, 2.2) Point(3, 3.3)] [] [Point(4, 4.4) Point(5, 5.5)]] at 0x7811883bccf8>

    a.tolist()
    # [[Point(1, 1.1), Point(2, 2.2), Point(3, 3.3)],
    #  [],
    #  [Point(4, 4.4), Point(5, 5.5)]]

* ``valid(exception=False, message=False)``: manually invoke the whole-array validity checks on the top-level array (not recursively). With the default options, this function returns ``True`` if valid and ``False`` if not. If ``exception=True``, it returns nothing on success and raises the appropriate exception on failure. If ``message=True``, it returns ``None`` on success and the error string on failure. (TODO: ``recursive=True``?)

.. code-block:: python3

    a = awkward.JaggedArray.fromcounts([3, 0, 2], [1.1, 2.2, 3.3, 4.4])  # content array is too short
    a.valid()
    # False

    try:
        a.valid(exception=True)
    except Exception as err:
        print(type(err), str(err))
    # <class 'ValueError'> maximum offset 5 is beyond the length of the content (4)

    a.valid(message=True)
    # "<class 'ValueError'>: maximum offset 5 is beyond the length of the content (4)"

* ``astype(dtype)``: convert *nested Numpy arrays* into the given type while maintaining awkward structure.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    a.astype(numpy.int32)
    # <JaggedArray [[1 2 3] [] [4 5]] at 0x7811883b9898>

* ``regular()``: convert the awkward array into a Numpy array and (unlike ``numpy.array(awkward_array)``) raise an error if it cannot be faithfully represented.

.. code-block:: python3

    # This JaggedArray happens to have equal-sized inner arrays.
    a = awkward.fromiter([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]])
    a
    # <JaggedArray [[1.1 2.2 3.3] [4.4 5.5 6.6] [7.7 8.8 9.9]] at 0x781188390240>

    a.regular()
    # array([[1.1, 2.2, 3.3],
    #        [4.4, 5.5, 6.6],
    #        [7.7, 8.8, 9.9]])

    # This one does not.
    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    a
    # <JaggedArray [[1.1 2.2 3.3] [] [4.4 5.5]] at 0x7811883b9c18>

    try:
        a.regular()
    except Exception as err:
        print(type(err), str(err))
    # <class 'ValueError'> jagged array is not regular: different elements have different counts

* ``copy(optional constructor arguments...)``: copy an awkward array object, non-recursively and without copying memory buffers, possibly replacing some of its parameters. If the class is an awkward subclass or has mix-in methods, they are propagated to the copy.

.. code-block:: python3

    class Special:
        def get(self, index):
            try:
                return self[index]
            except IndexError:
                return None

    JaggedArrayMethods = awkward.Methods.mixin(Special, awkward.JaggedArray)

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    a.__class__ = JaggedArrayMethods
    a
    # <JaggedArrayMethods [[1.1 2.2 3.3] [] [4.4 5.5]] at 0x7811883bc2b0>

    a.get(2)
    # array([4.4, 5.5])

    a.get(3)

    b = a.copy(content=[100, 200, 300, 400, 500])
    b
    # <JaggedArrayMethods [[100 200 300] [] [400 500]] at 0x7811883c5908>

    b.get(2)
    # array([400, 500])

    b.get(3)

Internally, all the methods that return views of the array (like slicing) use ``copy`` to retain the special methods.

.. code-block:: python3

    c = a[1:]
    c
    # <JaggedArrayMethods [[] [4.4 5.5]] at 0x7811883c5be0>

    c.get(1)
    # array([4.4, 5.5])

    c.get(2)

* ``deepcopy(optional constructor arguments...)``: like ``copy``, except that it recursively copies all internal structure, including memory buffers associated with Numpy arrays.

.. code-block:: python3

    b = a.deepcopy(content=[100, 200, 300, 400, 500])
    b
    # <JaggedArrayMethods [[100 200 300] [] [400 500]] at 0x781188355748>

    # Modify the structure of a (not recommended; this is a demo).
    a.starts[0] = 1
    a
    # <JaggedArrayMethods [[2.2 3.3] [] [4.4 5.5]] at 0x7811883bc2b0>

    # But b is not modified. (If it were, it would start with 200.)
    b
    # <JaggedArrayMethods [[100 200 300] [] [400 500]] at 0x781188355748>

* ``empty_like(optional constructor arguments...)``
* ``zeros_like(optional constructor arguments...)``
* ``ones_like(optional constructor arguments...)``: recursively copies structure, replacing contents with new uninitialized buffers, new buffers full of zeros, or new buffers full of ones. Not usually used in analysis, but needed for implementation.

.. code-block:: python3

    d = a.zeros_like()
    d
    # <JaggedArrayMethods [[0.0 0.0] [] [0.0 0.0]] at 0x7811883c59b0>

    e = a.ones_like()
    e
    # <JaggedArrayMethods [[1.0 1.0] [] [1.0 1.0]] at 0x78118847a2b0>

Reducers
""""""""

All awkward arrays also have a complete set of reducer methods. Reducers can be found in Numpy as well (as array methods and as free-standing functions), but they're not called out as a special class the way that universal functions ("ufuncs") are. Reducers decrease the rank or jaggedness of an array by one dimension, replacing subarrays with scalars. Examples include ``sum``, ``min``, and ``max``, but any monoid (associative operation with an identity) can be a reducer.

In awkward-array, reducers are only array methods (not free-standing functions) and unlike Numpy, they do not take an ``axis`` parameter. When a reducer is called at any level, it reduces the innermost dimension. (Since outer dimensions can be jagged, this is the only dimension that can be meaningfully reduced.)

.. code-block:: python3

    a = awkward.fromiter([[[[1, 2], [3]], [[4, 5]]], [[[], [6, 7, 8, 9]]]])
    a
    # <JaggedArray [[[[1 2] [3]] [[4 5]]] [[[] [6 7 8 9]]]] at 0x7811883b9470>

    a.sum()
    # <JaggedArray [[[3 3] [9]] [[0 30]]] at 0x7811883bc4a8>

    a.sum().sum()
    # <JaggedArray [[6 9] [30]] at 0x7811883bc048>

    a.sum().sum().sum()
    # array([15, 30])

    a.sum().sum().sum().sum()
    # 45

In the following example, "the deepest axis" of different fields in the table are at different depths: singly jagged in ``"x"`` and doubly jagged array in ``"y"``. The ``sum`` reduces each depth by one, producing a flat array ``"x"`` and a singly jagged array in ``"y"``.

.. code-block:: python3

    a = awkward.fromiter([{"x": [], "y": [[0.1, 0.2], [], [0.3]]}, {"x": [1, 2, 3], "y": [[0.4], [], [0.5, 0.6]]}])
    a.tolist()
    # [{'x': [], 'y': [[0.1, 0.2], [], [0.3]]},
    #  {'x': [1, 2, 3], 'y': [[0.4], [], [0.5, 0.6]]}]

    a.sum().tolist()
    [{'x': 0, 'y': [0.3, 0.0, 0.3]},
     {'x': 6, 'y': [0.4, 0.0, 1.1]}]

This sum cannot be reduced again because ``"x"`` is not jagged (would reduce to a scalar) and ``"y"`` is (would reduce to an array). The result cannot be scalar in one field (a single row, not a collection) and an array in another field (a collection).

.. code-block:: python3

    try:
        a.sum().sum()
    except Exception as err:
        print(type(err), str(err))
    # <class 'ValueError'> some Table columns are jagged and others are not

A table can be reduced if all of its fields are jagged or if all of its fields are not jagged; here's an example of the latter.

.. code-block:: python3

    a = awkward.fromiter([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])
    a.tolist()
    # [{'x': 1, 'y': 1.1}, {'x': 2, 'y': 2.2}, {'x': 3, 'y': 3.3}]

    a.sum()
    # <sum {'x': 6, 'y': 6.6}>

The resulting object is a scalar row—for your convenience, it has been labeled with the reducer that produced it.

.. code-block:: python3

    isinstance(a.sum(), awkward.Table.Row)
    # True

``UnionArrays`` are even more constrained: they can only be reduced if they have primitive (Numpy) type.

.. code-block:: python3

    a = awkward.fromiter([1, 2, 3, {"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}])
    a
    # <UnionArray [1 2 3 <Row 0> <Row 1>] at 0x781188355550>

    try:
        a.sum()
    except Exception as err:
        print(type(err), str(err))
    # <class 'TypeError'> cannot reduce a UnionArray of non-primitive type

    a = awkward.UnionArray.fromtags([0, 0, 0, 1, 1],
                                    [numpy.array([1, 2, 3], dtype=numpy.int32),
                                     numpy.array([4, 5], dtype=numpy.float64)])
    a
    # <UnionArray [1 2 3 4.0 5.0] at 0x781188355da0>

    a.sum()
    # 15.0

In all reducers, ``NaN`` in floating-point arrays and ``None`` in ``MaskedArrays`` are skipped, so these reducers are more like ``numpy.nansum``, ``numpy.nanmax``, and ``numpy.nanmin``, but generalized to all nullable types.

.. code-block:: python3

    a = awkward.fromiter([[[[1.1, numpy.nan], [2.2]], [[None, 3.3]]], [[[], [None, numpy.nan, None]]]])
    a
    # <JaggedArray [[[[1.1 nan] [2.2]] [[None 3.3]]] [[[] [None nan None]]]] at 0x78118835c7b8>

    a.sum()
    # <JaggedArray [[[1.1 2.2] [3.3]] [[0.0 0.0]]] at 0x781188355a20>

    a = awkward.fromiter([[{"x": 1, "y": 1.1}, None, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": numpy.nan}]])
    a.tolist()
    # [[{'x': 1, 'y': 1.1}, None, {'x': 3, 'y': 3.3}], [], [{'x': 4, 'y': nan}]]

    a.sum().tolist()
    # [{'x': 4, 'y': 4.4}, {'x': 0, 'y': 0.0}, {'x': 4, 'y': 0.0}]

The following reducers are defined as methods on all awkward arrays.

* ``reduce(ufunc, identity)``: generic reducer, calls ``ufunc.reduceat`` and returns ``identity`` for empty arrays.

.. code-block:: python3

    # numba.vectorize makes new ufuncs (requires type signatures and a kernel function)
    import numba
    @numba.vectorize([numba.int64(numba.int64, numba.int64)])
    def sum_mod_10(x, y):
        return (x + y) % 10

    a = awkward.fromiter([[1, 2, 3], [], [4, 5, 6], [7, 8, 9, 10]])
    a.sum()
    # array([ 6,  0, 15, 34])

    a.reduce(sum_mod_10, 0)
    # array([6, 0, 5, 4])

    # Missing (None) values are ignored.
    a = awkward.fromiter([[1, 2, None, 3], [], [None, None, None], [7, 8, 9, 10]])
    a.reduce(sum_mod_10, 0)
    # array([6, 0, 0, 4])

* ``any()``: boolean reducer, returns ``True`` if any (logical or) of the elements of an array are ``True``, returns ``False`` for empty arrays.

.. code-block:: python3

    a = awkward.fromiter([[False, False], [True, True], [True, False], []])
    a.any()
    # array([False,  True,  True, False])

    # Missing (None) values are ignored.
    a = awkward.fromiter([[False, None], [True, None], [None]])
    a.any()
    # array([False,  True, False])

* ``all()``: boolean reducer, returns ``True`` if all (logical and) of the elements of an array are ``True``, returns ``True`` for empty arrays.

.. code-block:: python3

    a = awkward.fromiter([[False, False], [True, True], [True, False], []])
    a.all()
    # array([False,  True, False,  True])

    # Missing (None) values are ignored.
    a = awkward.fromiter([[False, None], [True, None], [None]])
    a.all()
    # array([False,  True,  True])

* ``count()``: returns the (integer) number of elements in an array, skipping ``None`` and ``NaN``.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, None], [], [3.3, numpy.nan]])
    a.count()
    # array([2, 0, 1])

* ``count_nonzero()``: returns the (integer) number of non-zero elements in an array, skipping ``None`` and ``NaN``.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, None, 0], [], [3.3, numpy.nan, 0]])
    a.count_nonzero()
    # array([2, 0, 1])

* ``sum()``: returns the sum of each array, skipping ``None`` and ``NaN``, returning 0 for empty arrays.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, None], [], [3.3, numpy.nan]])
    a.sum()
    # array([3.3, 0. , 3.3])

* ``prod()``: returns the product (multiplication) of each array, skipping ``None`` and ``NaN``, returning 1 for empty arrays.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, None], [], [3.3, numpy.nan]])
    a.prod()
    # array([2.42, 1.  , 3.3 ])

* ``min()``: returns the minimum number in each array, skipping ``None`` and ``NaN``, returning infinity or the largest possible integer for empty arrays. (Note that Numpy raises errors for empty arrays.)

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, None], [], [3.3, numpy.nan]])
    a.min()
    # array([1.1, inf, 3.3])

    a = awkward.fromiter([[1, 2, None], [], [3]])
    a.min()
    # array([                  1, 9223372036854775807,                   3])

The identity of minimization is ``inf`` for floating-point values and ``9223372036854775807`` for ``int64`` because minimization with any other value would return the other value. This is more convenient for data analysts than raising an error because empty inner arrays are common.

* ``max()``: returns the maximum number in each array, skipping ``None`` and ``NaN``, returning negative infinity or the smallest possible integer for empty arrays. (Note that Numpy raises errors for empty arrays.)

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, None], [], [3.3, numpy.nan]])
    a.max()
    # array([ 2.2, -inf,  3.3])

    a = awkward.fromiter([[1, 2, None], [], [3]])
    a.max()
    # array([                   2, -9223372036854775808,                    3])

The identity of maximization is ``-inf`` for floating-point values and ``-9223372036854775808`` for ``int64`` because maximization with any other value would return the other value. This is more convenient for data analysts than raising an error because empty inner arrays are common.

Note that the maximization-identity for unsigned types is ``0``.

.. code-block:: python3

    a = awkward.JaggedArray.fromcounts([3, 0, 2], numpy.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=numpy.uint16))
    a
    # <JaggedArray [[1 2 3] [] [4 5]] at 0x78112c0e9a58>

    a.max()
    # array([3, 0, 5], dtype=uint16)

Functions like mean and standard deviation aren't true reducers because they're not associative (``mean(mean(x1, x2, x3), mean(x4, x5))`` is not equal to ``mean(mean(x1, x2), mean(x3, x4, x5))``). However, they're useful methods that exist on all awkward arrays, defined in terms of reducers.

* ``moment(n, weight=None)``: returns the ``n``th moment of each array (a floating-point value), skipping ``None`` and ``NaN``, returning ``NaN`` for empty arrays. If ``weight`` is given, it is taken as an array of weights, which may have the same structure as the ``array`` or be broadcastable to it, though any broadcasted weights would have no effect on the moment.

.. code-block:: python3

    a = awkward.fromiter([[1, 2, 3], [], [4, 5]])

    a.moment(1)
    # array([2. , nan, 4.5])

    a.moment(2)
    # array([ 4.66666667,         nan, 20.5       ])

Here is the first moment (mean) with a weight broadcasted from a scalar and from a non-jagged array, to show how it doesn't affect the result. The moment is calculated over an inner array, so if a constant value is broadcasted to all elements of that inner array, they all get the same weight.

.. code-block:: python3

    a.moment(1)
    # array([2. , nan, 4.5])

    a.moment(1, 100)
    # array([2. , nan, 4.5])

    a.moment(1, numpy.array([100, 200, 300]))
    # array([2. , nan, 4.5])

Only when the weight varies across an inner array does it have an effect.

.. code-block:: python3

    a.moment(1, awkward.fromiter([[1, 10, 100], [], [0, 100]]))
    # array([2.89189189,        nan, 5.        ])

* ``mean(weight=None)``: returns the mean of each array (a floating-point value), skipping ``None`` and ``NaN``, returning ``NaN`` for empty arrays, using optional ``weight`` as above.

.. code-block:: python3

    a = awkward.fromiter([[1, 2, 3], [], [4, 5]])
    a.mean()
    # array([2. , nan, 4.5])

* ``var(weight=None, ddof=0)``: returns the variance of each array (a floating-point value), skipping ``None`` and ``NaN``, returning ``NaN`` for empty arrays, using optional ``weight`` as above. The ``ddof`` or "Delta Degrees of Freedom" replaces a divisor of ``N`` (count or sum of weights) with a divisor of ``N - ddof``, following `numpy.var <https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html>`__.

.. code-block:: python3

    a = awkward.fromiter([[1, 2, 3], [], [4, 5]])
    a.var()
    # array([0.66666667,        nan, 0.25      ])

    a.var(ddof=1)
    # array([1. , nan, 0.5])

* ``std(weight=None, ddof=0)``: returns the standard deviation of each array, the square root of the variance described above.

.. code-block:: python3

    a.std()
    # array([0.81649658,        nan, 0.5       ])

    a.std(ddof=1)
    # array([1.        ,        nan, 0.70710678])

Properties and methods for jaggedness
"""""""""""""""""""""""""""""""""""""

All awkward arrays have these methods, but they provide information about the first nested ``JaggedArray`` within a structure. If, for instance, the ``JaggedArray`` is within some structure that doesn't affect high-level type (e.g. ``IndexedArray``, ``ChunkedArray``, ``VirtualArray``), then the methods are passed through to the ``JaggedArray``. If it's nested within something that does change type, but can meaningfully pass on the call, such as ``MaskedArray``, then that's what they do. If, however, it reaches a ``Table``, which may have some jagged columns and some non-jagged columns, the propagation stops.

* ``counts``: Numpy array of the number of elements in each inner array of the shallowest ``JaggedArray``. The ``counts`` may have rank > 1 if there are any fixed-size dimensions before the ``JaggedArray``.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
    a.counts
    # array([3, 0, 2, 4])

    # MaskedArrays return -1 for missing values.
    a = awkward.fromiter([[1.1, 2.2, 3.3], [], None, [6.6, 7.7, 8.8, 9.9]])
    a.counts
    # array([ 3,  0, -1,  4])

A missing inner array (counts is ``-1``) is distinct from an empty inner array (counts is ``0``), but if you want to ensure that you're working with data that have at least ``N`` elements, ``counts >= N`` works.

.. code-block:: python3

    a.counts >= 1
    # array([ True, False, False,  True])

    a[a.counts >= 1]
    # <MaskedArray [[1.1 2.2 3.3] [6.6 7.7 8.8 9.9]] at 0x78112c0d54a8>

    # UnionArrays return -1 for non-jagged arrays mixed with jagged arrays.
    a = awkward.fromiter([[1.1, 2.2, 3.3], [], 999, [6.6, 7.7, 8.8, 9.9]])
    a.counts
    # array([ 3,  0, -1,  4])

    # Same for tabular data, regardless of whether they contain nested jagged arrays.
    a = awkward.fromiter([[1.1, 2.2, 3.3], [], {"x": 1, "y": [1.1, 1.2, 1.3]}, [6.6, 7.7, 8.8, 9.9]])
    a.counts
    # array([ 3,  0, -1,  4])

Note! This means that pure ``Tables`` will always return zeros for counts, regardless of what they contain.

.. code-block:: python3

    a = awkward.fromiter([{"x": [], "y": []}, {"x": [1], "y": [1.1]}, {"x": [1, 2], "y": [1.1, 2.2]}])
    a.counts
    # array([-1, -1, -1])

If all of the columns of a ``Table`` are ``JaggedArrays`` with the same structure, you probably want to zip them into a single ``JaggedArray``.

.. code-block:: python3

    b = awkward.JaggedArray.zip(x=a.x, y=a.y)
    b
    # <JaggedArray [[] [<Row 0>] [<Row 1> <Row 2>]] at 0x78112c0dc7f0>

    b.counts
    # array([0, 1, 2])

* ``flatten(axis=0)``: removes one level of structure (losing information about boundaries between inner arrays) at a depth of jaggedness given by ``axis``.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
    a.flatten()
    # array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])

Unlike a ``JaggedArray``'s ``content``, which is part of its low-level layout, ``flatten()`` performs a high-level logical operation. Here's an example of the distinction.

.. code-block:: python3

    # JaggedArray with an unusual but valid structure.
    a = awkward.JaggedArray([3, 100, 0, 6], [6, 100, 2, 10],
                            [4.4, 5.5, 999, 1.1, 2.2, 3.3, 6.6, 7.7, 8.8, 9.9, 123])
    a
    # <JaggedArray [[1.1 2.2 3.3] [] [4.4 5.5] [6.6 7.7 8.8 9.9]] at 0x78112c127cf8>

    a.flatten()   # gives you a logically flattened array
    # array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])

    a.content     # gives you an internal structure component of the array
    # array([  4.4,   5.5, 999. ,   1.1,   2.2,   3.3,   6.6,   7.7,   8.8,
    #          9.9, 123. ])

In many cases, the output of ``flatten()`` corresponds to the output of ``content``, but be aware of the difference and use the one you want.

With ``flatten(axis=1)``, we can internally flatten nested ``JaggedArrays``.

.. code-block:: python3

    a = awkward.fromiter([[[1.1, 2.2], [3.3]], [], [[4.4, 5.5]], [[6.6, 7.7, 8.8], [], [9.9]]])
    a
    # <JaggedArray [[[1.1 2.2] [3.3]] [] [[4.4 5.5]] [[6.6 7.7 8.8] [] [9.9]]] at 0x78112c127208>

    a.flatten(axis=0)
    # <JaggedArray [[1.1 2.2] [3.3] [4.4 5.5] [6.6 7.7 8.8] [] [9.9]] at 0x78112c1276a0>

    a.flatten(axis=1)
    # <JaggedArray [[1.1 2.2 3.3] [] [4.4 5.5] [6.6 7.7 8.8 9.9]] at 0x78112c127320>

Even if a ``JaggedArray``'s inner structure is due to a fixed-shape Numpy array, the ``axis`` parameter propagates down and does the right thing.

.. code-block:: python3

    a = awkward.JaggedArray.fromcounts(numpy.array([3, 0, 2]),
                                       numpy.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]))
    a
    # <JaggedArray [[[1 1] [2 2] [3 3]] [] [[4 4] [5 5]]] at 0x78112c0d5ac8>

    type(a.content)
    # numpy.ndarray

    a.flatten(axis=1)
    # <JaggedArray [[1 1 2 2 3 3] [] [4 4 5 5]] at 0x78112c0d5a20>

But, unlike Numpy, we can't ask for an ``axis`` starting from the other end (with a negative index). The "deepest array" is not a well-defined concept for awkward arrays.

.. code-block:: python3

    try:
        a.flatten(axis=-1)
    except Exception as err:
        print(type(err), str(err))
    # <class 'TypeError'> axis must be a non-negative integer (can't count from the end)

    a = awkward.fromiter([[[1.1, 2.2], [3.3]], [], None, [[6.6, 7.7, 8.8], [], [9.9]]])
    a
    # <MaskedArray [[[1.1 2.2] [3.3]] [] None [[6.6 7.7 8.8] [] [9.9]]] at 0x78112c0d51d0>

    a.flatten(axis=1)
    # <JaggedArray [[1.1 2.2 3.3] [] [6.6 7.7 8.8 9.9]] at 0x78112c0dcfd0>

* ``pad(length, maskedwhen=True, clip=False)``: ensures that each inner array has at least ``length`` elements by filling in the empty spaces with ``None`` (i.e. by inserting a ``MaskedArray`` layer). The ``maskedwhen`` parameter determines whether ``mask[i] == True`` means the element is ``None`` (``maskedwhen=True``) or not ``None`` (``maskedwhen=False``). Setting ``maskedwhen`` doesn't change the logical meaning of the array. If ``clip=True``, then the inner arrays will have exactly ``length`` elements (by clipping the ones that are too long). Even though this results in regular sizes, they are still represented by a ``JaggedArray``.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
    a
    # <JaggedArray [[1.1 2.2 3.3] [] [4.4 5.5] [6.6 7.7 8.8 9.9]] at 0x78112c127be0>

    a.pad(3)
    # <JaggedArray [[1.1 2.2 3.3] [None None None] [4.4 5.5 None] [6.6 7.7 8.8 9.9]] at 0x78112c122588>

    a.pad(3, maskedwhen=False)
    # <JaggedArray [[1.1 2.2 3.3] [None None None] [4.4 5.5 None] [6.6 7.7 8.8 9.9]] at 0x78112c122c18>

    a.pad(3, clip=True)
    # <JaggedArray [[1.1 2.2 3.3] [None None None] [4.4 5.5 None] [6.6 7.7 8.8]] at 0x78112c127940>

If you want to get rid of the ``MaskedArray`` layer, replace ``None`` with some value.

.. code-block:: python3

    a.pad(3).fillna(-999)
    # <JaggedArray [[1.1 2.2 3.3] [-999.0 -999.0 -999.0] [4.4 5.5 -999.0] [6.6 7.7 8.8 9.9]] at 0x78112c0dc0b8>

If you want to make an effectively regular array into a real Numpy array, use ``regular``.

.. code-block:: python3

    a.pad(3, clip=True).fillna(0).regular()
    # array([[1.1, 2.2, 3.3],
    #        [0. , 0. , 0. ],
    #        [4.4, 5.5, 0. ],
    #        [6.6, 7.7, 8.8]])

If a ``JaggedArray`` is nested within some other type, ``pad`` will propagate down to it.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], None, [4.4, 5.5], None])
    a
    # <MaskedArray [[1.1 2.2 3.3] [] None [4.4 5.5] None] at 0x78112c0d52b0>

    a.pad(3)
    # <MaskedArray [[1.1 2.2 3.3] [None None None] None [4.4 5.5 None] None] at 0x78112c0e9908>

    a = awkward.Table(x=[[1, 1], [2, 2], [3, 3], [4, 4]],
                      y=awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]))
    a.tolist()
    # [{'x': [1, 1], 'y': [1.1, 2.2, 3.3]},
    #  {'x': [2, 2], 'y': []},
    #  {'x': [3, 3], 'y': [4.4, 5.5]},
    #  {'x': [4, 4], 'y': [6.6, 7.7, 8.8, 9.9]}]

    a.pad(3).tolist()
    # [{'x': [1, 1, None], 'y': [1.1, 2.2, 3.3]},
    #  {'x': [2, 2, None], 'y': [None, None, None]},
    #  {'x': [3, 3, None], 'y': [4.4, 5.5, None]},
    #  {'x': [4, 4, None], 'y': [6.6, 7.7, 8.8, 9.9]}]

    a.pad(3, clip=True).tolist()
    # [{'x': [1, 1, None], 'y': [1.1, 2.2, 3.3]},
    #  {'x': [2, 2, None], 'y': [None, None, None]},
    #  {'x': [3, 3, None], 'y': [4.4, 5.5, None]},
    #  {'x': [4, 4, None], 'y': [6.6, 7.7, 8.8]}]

If you pass a ``pad`` through a ``Table``, be sure that every field in each record is a nested array (and therefore can be padded).

.. code-block:: python3

    a = awkward.Table(x=[1, 2, 3, 4],
                      y=awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]))
    a.tolist()
    # [{'x': 1, 'y': [1.1, 2.2, 3.3]},
    #  {'x': 2, 'y': []},
    #  {'x': 3, 'y': [4.4, 5.5]},
    #  {'x': 4, 'y': [6.6, 7.7, 8.8, 9.9]}]

    try:
        a.pad(3)
    except Exception as err:
        print(type(err), str(err))
    # <class 'ValueError'> pad cannot be applied to scalars

The same goes for ``UnionArrays``.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3, [1, 2, 3]], [], [4.4, 5.5, [4, 5]]])
    a
    # <JaggedArray [[1.1 2.2 3.3 [1 2 3]] [] [4.4 5.5 [4 5]]] at 0x7811883c5d30>

    a.pad(5)
    # <JaggedArray [[1.1 2.2 3.3 [1 2 3] None] [None None None None None] [4.4 5.5 [4 5] None None]] at 0x78112c0e9a20>

    a = awkward.UnionArray.fromtags([0, 0, 0, 1, 1],
                                    [awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]]),
                                     awkward.fromiter([[100, 101], [102]])])
    a
    # <UnionArray [[1.1 2.2 3.3] [] [4.4 5.5] [100 101] [102]] at 0x78112c0bed30>

    a.pad(3)
    # <UnionArray [[1.1 2.2 3.3] [None None None] [4.4 5.5 None] [100 101 None] [102 None None]] at 0x78112c0bedd8>

    a = awkward.UnionArray.fromtags([0, 0, 0, 1, 1],
                                    [awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]]),
                                     awkward.fromiter([100, 200])])
    a
    # <UnionArray [[1.1 2.2 3.3] [] [4.4 5.5] 100 200] at 0x78112c0e9b00>

    try:
        a.pad(3)
    except Exception as err:
        print(type(err), str(err))
    # <class 'ValueError'> pad cannot be applied to scalars

The general behavior of ``pad`` is to replace the shallowest ``JaggedArray`` with a ``JaggedArray`` containing a ``MaskedArray``. The one exception to this type signature is that ``StringArrays`` are padded with characters.

.. code-block:: python3

    a = awkward.fromiter(["one", "two", "three"])
    a
    # <StringArray ['one' 'two' 'three'] at 0x78112c0dcb00>

    a.pad(4, clip=True)
    # <StringArray ['one ' 'two ' 'thre'] at 0x78112c1222b0>

    a.pad(4, maskedwhen=b".", clip=True)
    # <StringArray ['one.' 'two.' 'thre'] at 0x78112c122f98>

    a.pad(4, maskedwhen=b"\x00", clip=True)
    # <StringArray ['one\x00' 'two\x00' 'thre'] at 0x78112c122be0>

* ``argmin()`` and ``argmax()``: returns the index of the minimum or maximum value in a non-jagged array or the indexes where each inner array is minimized or maximized. The jagged structure of the return value consists of empty arrays for each empty array and singleton arrays for non-empty ones, consisting of a single index in an inner array. This is the form needed to extract one element from each inner array using jagged indexing.

.. code-block:: python3

    a = awkward.fromiter([[-3.3, 5.5, -8.8], [], [-6.6, 0.0, 2.2, 3.3], [], [2.2, -2.2, 4.4]])
    absa = abs(a)

    a
    # <JaggedArray [[-3.3 5.5 -8.8] [] [-6.6 0.0 2.2 3.3] [] [2.2 -2.2 4.4]] at 0x78112c0beb70>

    absa
    # <JaggedArray [[3.3 5.5 8.8] [] [6.6 0.0 2.2 3.3] [] [2.2 2.2 4.4]] at 0x78112c0bec18>

    index = absa.argmax()
    index
    # <JaggedArray [[2] [] [0] [] [2]] at 0x78112c0d0128>

    absa[index]
    # <JaggedArray [[8.8] [] [6.6] [] [4.4]] at 0x78112c122c50>

    a[index]
    # <JaggedArray [[-8.8] [] [-6.6] [] [4.4]] at 0x78112c0d5eb8>

* ``cross(other, nested=False)`` and ``argcross(other, nested=False)``: returns jagged tuples representing the `cross-join <https://en.wikipedia.org/wiki/Join_(SQL)#Cross_join>`__ of `array[i]` and `other[i]` separately for each `i`. If `nested=True`, the result is doubly jagged so that each element of the output corresponds to exactly one element in the original `array`.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
    b = awkward.fromiter([["one", "two"], ["three"], ["four", "five", "six"], ["seven"]])
    a.cross(b)
    # <JaggedArray [[(1.1, one) (1.1, two) (2.2, one) (2.2, two) (3.3, one) (3.3, two)] [] [(4.4, four) (4.4, five) (4.4, six) (5.5, four) (5.5, five) (5.5, six)] [(6.6, seven) (7.7, seven) (8.8, seven) (9.9, seven)]] at 0x78112c0e9550>

    a.cross(b, nested=True)
    # <JaggedArray [[[(1.1, one) (1.1, two)] [(2.2, one) (2.2, two)] [(3.3, one) (3.3, two)]] [] [[(4.4, four) (4.4, five) (4.4, six)] [(5.5, four) (5.5, five) (5.5, six)]] [[(6.6, seven)] [(7.7, seven)] [(8.8, seven)] [(9.9, seven)]]] at 0x78112c0be978>

The "arg" version returns indexes at which the appropriate objects may be found, as usual.

.. code-block:: python3

    a.argcross(b)
    # <JaggedArray [[(0, 0) (0, 1) (1, 0) (1, 1) (2, 0) (2, 1)] [] [(0, 0) (0, 1) (0, 2) (1, 0) (1, 1) (1, 2)] [(0, 0) (1, 0) (2, 0) (3, 0)]] at 0x78112c122470>

    a.argcross(b, nested=True)
    # <JaggedArray [[[(0, 0) (0, 1)] [(1, 0) (1, 1)] [(2, 0) (2, 1)]] [] [[(0, 0) (0, 1) (0, 2)] [(1, 0) (1, 1) (1, 2)]] [[(0, 0)] [(1, 0)] [(2, 0)] [(3, 0)]]] at 0x78112c122dd8>

This method is good to use with ``unzip``, which separates the ``Table`` of tuples into a left half and a right half.

.. code-block:: python3

    left, right = a.cross(b).unzip()
    left, right
    # (<JaggedArray [[1.1 1.1 2.2 2.2 3.3 3.3] [] [4.4 4.4 4.4 5.5 5.5 5.5] [6.6 7.7 8.8 9.9]] at 0x78112c0be278>,
    #  <JaggedArray [['one' 'two' 'one' 'two' 'one' 'two'] [] ['four' 'five' 'six' 'four' 'five' 'six'] ['seven' 'seven' 'seven' 'seven']] at 0x78112c0d0470>)

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
    b = awkward.fromiter([[1, 2], [3], [4, 5, 6], [7]])
    left, right = a.cross(b, nested=True).unzip()
    left, right
    # (<JaggedArray [[[1.1 1.1] [2.2 2.2] [3.3 3.3]] [] [[4.4 4.4 4.4] [5.5 5.5 5.5]] [[6.6] [7.7] [8.8] [9.9]]] at 0x78112c127048>,
    #  <JaggedArray [[[1 2] [1 2] [1 2]] [] [[4 5 6] [4 5 6]] [[7] [7] [7] [7]]] at 0x78112c127630>)

This can be handy if a subsequent function takes two jagged arrays as arguments.

.. code-block:: python3

    distance = round(abs(left - right), 1)
    distance
    # <JaggedArray [[[0.1 0.9] [1.2 0.2] [2.3 1.3]] [] [[0.4 0.6 1.6] [1.5 0.5 0.5]] [[0.4] [0.7] [1.8] [2.9]]] at 0x78112c0bec88>

Cross with ``nested=True``, followed by some calculation on the pairs and then some reducer, is a common pattern. Because of the ``nested=True`` and the reducer, the resulting array has the same structure as the original.

.. code-block:: python3

    distance.min()
    # <JaggedArray [[0.1 0.2 1.3] [] [0.4 0.5] [0.4 0.7 1.8 2.9]] at 0x78112c0d50f0>

    round(a + distance.min(), 1)
    # <JaggedArray [[1.2 2.4 4.6] [] [4.8 6.0] [7.0 8.4 10.6 12.8]] at 0x78112c122518>

* ``pairs(nested=False)`` and ``argpairs(nested=False)``: returns jagged tuples representing the `self-join <https://en.wikipedia.org/wiki/Join_(SQL)#Self-join>`__ removing duplicates but not same-object pairs (i.e. a self-join with ``i1 <= i2``) for each inner array separately.

.. code-block:: python3

    a = awkward.fromiter([["a", "b", "c"], [], ["d", "e"]])
    a.pairs()
    # <JaggedArray [[(a, a) (a, b) (a, c) (b, b) (b, c) (c, c)] [] [(d, d) (d, e) (e, e)]] at 0x78112c127898>

The "arg" and ``nested=True`` versions have the same meanings as with ``cross`` (above).

.. code-block:: python3

    a.argpairs()
    # <JaggedArray [[(0, 0) (0, 1) (0, 2) (1, 1) (1, 2) (2, 2)] [] [(0, 0) (0, 1) (1, 1)]] at 0x78112c0d0978>

    a.pairs(nested=True)
    # <JaggedArray [[[(a, a) (a, b) (a, c)] [(b, b) (b, c)] [(c, c)]] [] [[(d, d) (d, e)] [(e, e)]]] at 0x78112c0be2b0>

Just as with ``cross`` (above), this is good to combine with ``unzip`` and maybe a reducer.

.. code-block:: python3

    a.pairs().unzip()
    # (<JaggedArray [['a' 'a' 'a' 'b' 'b' 'c'] [] ['d' 'd' 'e']] at 0x78112c0d08d0>,
    #  <JaggedArray [['a' 'b' 'c' 'b' 'c' 'c'] [] ['d' 'e' 'e']] at 0x78112c0d0fd0>)

* ``distincts(nested=False)`` and ``argdistincts(nested=False)``: returns jagged tuples representing the `self-join <https://en.wikipedia.org/wiki/Join_(SQL)#Self-join>`__ removing duplicates and same-object pairs (i.e. a self-join with ``i1 < i2``) for each inner array separately.

.. code-block:: python3

    a = awkward.fromiter([["a", "b", "c"], [], ["d", "e"]])
    a.distincts()
    # <JaggedArray [[(a, b) (a, c) (b, c)] [] [(d, e)]] at 0x78112c127080>

The "arg" and ``nested=True`` versions have the same meanings as with ``cross`` (above).

.. code-block:: python3

    a.argdistincts()
    # <JaggedArray [[(0, 1) (0, 2) (1, 2)] [] [(0, 1)]] at 0x78112c0d04e0>

    a.distincts(nested=True)
    # <JaggedArray [[[(a, b) (a, c)] [(b, c)]] [] [[(d, e)]]] at 0x78112c0d0a58>

Just as with ``cross`` (above), this is good to combine with ``unzip`` and maybe a reducer.

.. code-block:: python3

    a.distincts().unzip()
    # (<JaggedArray [['a' 'a' 'b'] [] ['d']] at 0x78112c11e908>,
    #  <JaggedArray [['b' 'c' 'c'] [] ['e']] at 0x78112c11e518>)

* ``choose(n)`` and ``argchoose(n)``: returns jagged tuples for distinct combinations of ``n`` elements from every inner array separately. ``array.choose(2)`` is the same as ``array.distincts()`` apart from order.

.. code-block:: python3

    a = awkward.fromiter([["a", "b", "c"], [], ["d", "e"], ["f", "g", "h", "i", "j"]])
    a
    # <JaggedArray [['a' 'b' 'c'] [] ['d' 'e'] ['f' 'g' 'h' 'i' 'j']] at 0x78112c0d0400>

    a.choose(2)
    # <JaggedArray [[(a, b) (a, c) (b, c)] [] [(d, e)] [(f, g) (f, h) (g, h) ... (g, j) (h, j) (i, j)]] at 0x78112c11e0f0>

    a.choose(3)
    # <JaggedArray [[(a, b, c)] [] [] [(f, g, h) (f, g, i) (f, h, i) ... (f, i, j) (g, i, j) (h, i, j)]] at 0x78114c6e46a0>

    a.choose(4)
    # <JaggedArray [[] [] [] [(f, g, h, i) (f, g, h, j) (f, g, i, j) (f, h, i, j) (g, h, i, j)]] at 0x78112c0d0cc0>

The "arg" version has the same meaning as ``cross`` (above), but there is no ``nested=True`` because of the order.

.. code-block:: python3

    a.argchoose(2)
    # <JaggedArray [[(0, 1) (0, 2) (1, 2)] [] [(0, 1)] [(0, 1) (0, 2) (1, 2) ... (1, 4) (2, 4) (3, 4)]] at 0x78112c11e2b0>

Just as with ``cross`` (above), this is good to combine with ``unzip`` and maybe a reducer.

.. code-block:: python3

    a.choose(2).unzip()
    # (<JaggedArray [['a' 'a' 'b'] [] ['d'] ['f' 'f' 'g' ... 'g' 'h' 'i']] at 0x78112c11e5c0>,
    #  <JaggedArray [['b' 'c' 'c'] [] ['e'] ['g' 'h' 'h' ... 'j' 'j' 'j']] at 0x78112c0f7ac8>)

    a.choose(3).unzip()
    # (<JaggedArray [['a'] [] [] ['f' 'f' 'f' ... 'f' 'g' 'h']] at 0x78112c0dc5f8>,
    #  <JaggedArray [['b'] [] [] ['g' 'g' 'h' ... 'i' 'i' 'i']] at 0x78112c0dc3c8>,
    #  <JaggedArray [['c'] [] [] ['h' 'i' 'i' ... 'j' 'j' 'j']] at 0x78112c0dc6d8>)

    a.choose(4).unzip()
    # (<JaggedArray [[] [] [] ['f' 'f' 'f' 'f' 'g']] at 0x78112c0d0eb8>,
    #  <JaggedArray [[] [] [] ['g' 'g' 'g' 'h' 'h']] at 0x78112c11e550>,
    #  <JaggedArray [[] [] [] ['h' 'h' 'i' 'i' 'i']] at 0x78112c11e2e8>,
    #  <JaggedArray [[] [] [] ['i' 'j' 'j' 'j' 'j']] at 0x78112c11e4a8>)

* ``JaggedArray.zip(columns...)``: combines jagged arrays with the same structure into a single jagged array. The columns may be unnamed (resulting in a jagged array of tuples) or named with keyword arguments or dict keys (resulting in a jagged array of a table with named columns).

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    b = awkward.fromiter([[100, 200, 300], [], [400, 500]])
    awkward.JaggedArray.zip(a, b)
    # <JaggedArray [[(1.1, 100) (2.2, 200) (3.3, 300)] [] [(4.4, 400) (5.5, 500)]] at 0x78112c0f71d0>

    awkward.JaggedArray.zip(x=a, y=b).tolist()
    # [[{'x': 1.1, 'y': 100}, {'x': 2.2, 'y': 200}, {'x': 3.3, 'y': 300}],
    #  [],
    #  [{'x': 4.4, 'y': 400}, {'x': 5.5, 'y': 500}]]

    awkward.JaggedArray.zip({"x": a, "y": b}).tolist()
    # [[{'x': 1.1, 'y': 100}, {'x': 2.2, 'y': 200}, {'x': 3.3, 'y': 300}],
    #  [],
    #  [{'x': 4.4, 'y': 400}, {'x': 5.5, 'y': 500}]]

Not all of the arguments need to be jagged; those that aren't will be broadcasted to the right shape.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    b = awkward.fromiter([100, 200, 300])
    awkward.JaggedArray.zip(a, b)
    # <JaggedArray [[(1.1, 100) (2.2, 100) (3.3, 100)] [] [(4.4, 300) (5.5, 300)]] at 0x78112c0f7c18>

    awkward.JaggedArray.zip(a, 1000)
    # <JaggedArray [[(1.1, 1000) (2.2, 1000) (3.3, 1000)] [] [(4.4, 1000) (5.5, 1000)]] at 0x78112c0f72e8>

Properties and methods for tabular columns
""""""""""""""""""""""""""""""""""""""""""

All awkward arrays have these methods, but they provide information about the first nested ``Table`` within a structure. If, for instance, the ``Table`` is within some structure that doesn't affect high-level type (e.g. ``IndexedArray``, ``ChunkedArray``, ``VirtualArray``), then the methods are passed through to the ``Table``. If it's nested within something that does change type, but can meaningfully pass on the call, such as ``MaskedArray``, then that's what they do.

* ``columns``: the names of the columns at the first tabular level of depth.

.. code-block:: python3

    a = awkward.fromiter([{"x": 1, "y": 1.1, "z": "one"}, {"x": 2, "y": 2.2, "z": "two"}, {"x": 3, "y": 3.3, "z": "three"}])
    a.tolist()
    # [{'x': 1, 'y': 1.1, 'z': 'one'},
    #  {'x': 2, 'y': 2.2, 'z': 'two'},
    #  {'x': 3, 'y': 3.3, 'z': 'three'}]

    a.columns
    # ['x', 'y', 'z']

    a = awkward.Table(x=[1, 2, 3],
                      y=[1.1, 2.2, 3.3],
                      z=awkward.Table(a=[4, 5, 6], b=[4.4, 5.5, 6.6]))
    a.tolist()
    # [{'x': 1, 'y': 1.1, 'z': {'a': 4, 'b': 4.4}},
    #  {'x': 2, 'y': 2.2, 'z': {'a': 5, 'b': 5.5}},
    #  {'x': 3, 'y': 3.3, 'z': {'a': 6, 'b': 6.6}}]

    a.columns
    # ['x', 'y', 'z']

    a["z"].columns
    # ['a', 'b']

    a.z.columns
    # ['a', 'b']

* ``unzip()``: returns a tuple of projections through each of the columns (in the same order as the ``columns`` property).

.. code-block:: python3

    a = awkward.fromiter([{"x": 1, "y": 1.1, "z": "one"}, {"x": 2, "y": 2.2, "z": "two"}, {"x": 3, "y": 3.3, "z": "three"}])
    a.unzip()
    # (array([1, 2, 3]),
    #  array([1.1, 2.2, 3.3]),
    #  <StringArray ['one' 'two' 'three'] at 0x78112c0d02b0>)

The ``unzip`` method is the opposite of the ``Table`` constructor,

.. code-block:: python3

    a = awkward.Table(x=[1, 2, 3],
                      y=[1.1, 2.2, 3.3],
                      z=awkward.fromiter(["one", "two", "three"]))
    a.tolist()
    # [{'x': 1, 'y': 1.1, 'z': 'one'},
    #  {'x': 2, 'y': 2.2, 'z': 'two'},
    #  {'x': 3, 'y': 3.3, 'z': 'three'}]

    a.unzip()
    # (array([1, 2, 3]),
    #  array([1.1, 2.2, 3.3]),
    #  <StringArray ['one' 'two' 'three'] at 0x78112c115a20>)

but it is also the opposite of ``JaggedArray.zip``.

.. code-block:: python3

    b = awkward.JaggedArray.zip(x=awkward.fromiter([[1, 2, 3], [], [4, 5]]),
                                y=awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]]),
                                z=awkward.fromiter([["a", "b", "c"], [], ["d", "e"]]))
    b.tolist()
    # [[{'x': 1, 'y': 1.1, 'z': 'a'},
    #   {'x': 2, 'y': 2.2, 'z': 'b'},
    #   {'x': 3, 'y': 3.3, 'z': 'c'}],
    #  [],
    #  [{'x': 4, 'y': 4.4, 'z': 'd'}, {'x': 5, 'y': 5.5, 'z': 'e'}]]

    b.unzip()
    # (<JaggedArray [[1 2 3] [] [4 5]] at 0x78112c14fe10>,
    #  <JaggedArray [[1.1 2.2 3.3] [] [4.4 5.5]] at 0x78112c14f9b0>,
    #  <JaggedArray [['a' 'b' 'c'] [] ['d' 'e']] at 0x78112c14fa90>)

``JaggedArray.zip`` produces a jagged array of ``Table`` whereas the ``Table`` constructor produces just a ``Table``, and these are distinct things, though they can both be inverted by the same function because row indexes and column indexes commute:

.. code-block:: python3

    b[0]["y"]
    # array([1.1, 2.2, 3.3])

    b["y"][0]
    # array([1.1, 2.2, 3.3])

So ``unzip`` turns a flat ``Table`` into a tuple of flat arrays (opposite of the ``Table`` constructor) and it turns a jagged ``Table`` into a tuple of jagged arrays (opposite of ``JaggedArray.zip``).

* ``istuple``: an array of tuples is a special kind of ``Table``, one whose ``rowname`` is ``"tuple"`` and columns are ``"0"``, ``"1"``, ``"2"``, etc. If these conditions are met, ``istuple`` is ``True``; otherwise, ``False``.

.. code-block:: python3

    a = awkward.Table(x=[1, 2, 3],
                      y=[1.1, 2.2, 3.3],
                      z=awkward.fromiter(["one", "two", "three"]))
    a.tolist()
    # [{'x': 1, 'y': 1.1, 'z': 'one'},
    #  {'x': 2, 'y': 2.2, 'z': 'two'},
    #  {'x': 3, 'y': 3.3, 'z': 'three'}]

    a.istuple
    # False

    a = awkward.Table([1, 2, 3],
                      [1.1, 2.2, 3.3],
                      awkward.fromiter(["one", "two", "three"]))
    a.tolist()
    # [(1, 1.1, 'one'), (2, 2.2, 'two'), (3, 3.3, 'three')]

    a.istuple
    # True

Even though the following tuples are inside of a jagged array, the first level of ``Table`` is a tuple, so ``istuple`` is ``True``.

.. code-block:: python3

    b = awkward.JaggedArray.zip(awkward.fromiter([[1, 2, 3], [], [4, 5]]),
                                awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]]),
                                awkward.fromiter([["a", "b", "c"], [], ["d", "e"]]))
    b
    # <JaggedArray [[(1, 1.1, a) (2, 2.2, b) (3, 3.3, c)] [] [(4, 4.4, d) (5, 5.5, e)]] at 0x78112c0d0e48>

    b.istuple
    # True

* ``i0`` through ``i9``: one of the two conditions for a ``Table`` to be a ``tuple`` is that columns are named ``"0"``, ``"1"``, ``"2"``, etc. Columns like that could be selected with ``["0"]`` at the risk of being misread as ``[0]``, and they could not be selected with attribute dot-access because pure numbers are not valid Python attributes. However, ``i0`` through ``i9`` are provided as shortcuts (overriding any columns with these exact names) for the first 10 tuple slots.

.. code-block:: python3

    a = awkward.Table([1, 2, 3],
                      [1.1, 2.2, 3.3],
                      awkward.fromiter(["one", "two", "three"]))
    a.tolist()
    # [(1, 1.1, 'one'), (2, 2.2, 'two'), (3, 3.3, 'three')]

    a.i0
    # array([1, 2, 3])

    a.i1
    # array([1.1, 2.2, 3.3])

    a.i2
    # <StringArray ['one' 'two' 'three'] at 0x78112c14fe80>

* ``flattentuple()``: calling ``cross`` repeatedly can result in tuples nested within tuples; this flattens them at all levels, turning all ``(i, (j, k))`` into ``(i, j, k)``. Whereas ``array.flatten()`` removes one level of structure from the rows (losing information), ``array.flattentuple()`` removes all levels of structure from the columns (renaming them, but not losing information).

.. code-block:: python3

    a = awkward.Table([1, 2, 3], [1, 2, 3], awkward.Table(awkward.Table([1, 2, 3], [1, 2, 3]), [1, 2, 3]))
    a.tolist()
    # [(1, 1, ((1, 1), 1)), (2, 2, ((2, 2), 2)), (3, 3, ((3, 3), 3))]

    a.flattentuple().tolist()
    # [(1, 1, 1, 1, 1), (2, 2, 2, 2, 2), (3, 3, 3, 3, 3)]

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
    b = awkward.fromiter([[100, 200], [300], [400, 500, 600], [700]])
    c = awkward.fromiter([["a"], ["b", "c"], ["d"], ["e", "f"]])

The ``cross`` method internally calls ``flattentuples()`` if it detects that one of its arguments is the result of a ``cross``.

.. code-block:: python3

    a.cross(b).cross(c).tolist()
    # [[(1.1, 100, 'a'),
    #   (1.1, 200, 'a'),
    #   (2.2, 100, 'a'),
    #   (2.2, 200, 'a'),
    #   (3.3, 100, 'a'),
    #   (3.3, 200, 'a')],
    #  [],
    #  [(4.4, 400, 'd'),
    #   (4.4, 500, 'd'),
    #   (4.4, 600, 'd'),
    #   (5.5, 400, 'd'),
    #   (5.5, 500, 'd'),
    #   (5.5, 600, 'd')],
    #  [(6.6, 700, 'e'),
    #   (6.6, 700, 'f'),
    #   (7.7, 700, 'e'),
    #   (7.7, 700, 'f'),
    #   (8.8, 700, 'e'),
    #   (8.8, 700, 'f'),
    #   (9.9, 700, 'e'),
    #   (9.9, 700, 'f')]]

Properties and methods for missing values
"""""""""""""""""""""""""""""""""""""""""

All awkward arrays have these methods, but they provide information about the first nested ``MaskedArray`` within a structure. If, for instance, the ``MaskedArray`` is within some structure that doesn't affect high-level type (e.g. ``IndexedArray``, ``ChunkedArray``, ``VirtualArray``), then the methods are passed through to the ``MaskedArray``. If it's nested within something that does change type, but can meaningfully pass on the call, such as ``JaggedArray``, then that's what they do.

* ``boolmask(maskedwhen=None)``: returns a Numpy array of booleans indicating which elements are missing ("masked") and which are not. If ``maskedwhen=True``, a ``True`` value in the Numpy array means missing/masked; if ``maskedwhen=False``, a ``False`` value in the Numpy array means missing/masked. If no value is passed (or ``None``), the ``MaskedArray``'s own ``maskedwhen`` property is used (which is by default ``True``). Non-``MaskedArrays`` are assumed to have a ``maskedwhen`` of ``True`` (the default).

.. code-block:: python3

    a = awkward.fromiter([1, 2, None, 3, 4, None, None, 5])
    a.boolmask()
    # array([False, False,  True, False, False,  True,  True, False])

    a.boolmask(maskedwhen=False)
    # array([ True,  True, False,  True,  True, False, False,  True])

``MaskedArrays`` inside of ``JaggedArrays`` or ``Tables`` are hidden.

.. code-block:: python3

    a = awkward.fromiter([[1.1, None, 2.2], [], [3.3, 4.4, None, 5.5]])
    a.boolmask()
    # array([False, False, False])

    a.flatten().boolmask()
    # array([False,  True, False, False, False,  True, False])

    a = awkward.fromiter([{"x": 1, "y": 1.1}, {"x": None, "y": 2.2}, {"x": None, "y": 3.3}, {"x": 4, "y": None}])
    a.boolmask()
    # array([False, False, False, False])

    a.x.boolmask()
    # array([False,  True,  True, False])

    a.y.boolmask()
    # array([False, False, False,  True])

* ``ismasked`` and ``isunmasked``: shortcut for ``boolmask(maskedwhen=True)`` and ``boolmask(maskedwhen=False)`` as a property, which is more appropriate for analysis.

.. code-block:: python3

    a = awkward.fromiter([1, 2, None, 3, 4, None, None, 5])
    a.ismasked
    # array([False, False,  True, False, False,  True,  True, False])

    a.isunmasked
    # array([ True,  True, False,  True,  True, False, False,  True])

* ``fillna(value)``: turn a ``MaskedArray`` into a non-``MaskedArray`` by replacing ``None`` with ``value``. Applies to the outermost ``MaskedArray``, but it passes through ``JaggedArrays`` and into all ``Table`` columns.

.. code-block:: python3

    a = awkward.fromiter([1, 2, None, 3, 4, None, None, 5])
    a.fillna(999)
    # array([  1,   2, 999,   3,   4, 999, 999,   5])

    a = awkward.fromiter([[1.1, None, 2.2], [], [3.3, 4.4, None, 5.5]])
    a.fillna(999)
    # <JaggedArray [[1.1 999.0 2.2] [] [3.3 4.4 999.0 5.5]] at 0x78112c0859b0>

    a = awkward.fromiter([{"x": 1, "y": 1.1}, {"x": None, "y": 2.2}, {"x": None, "y": 3.3}, {"x": 4, "y": None}])
    a.fillna(999).tolist()
    # [{'x': 1, 'y': 1.1},
    #  {'x': 999, 'y': 2.2},
    #  {'x': 999, 'y': 3.3},
    #  {'x': 4, 'y': 999.0}]

Functions for structure manipulation
""""""""""""""""""""""""""""""""""""

Only one structure-manipulation function (for now) is defined at top-level in awkward-array: ``awkward.concatenate``.

* ``awkward.concatenate(arrays, axis=0)``: concatenate two or more ``arrays``. If ``axis=0``, the arrays are concatenated lengthwise (the resulting length is the sum of the lengths of each of the ``arrays``). If ``axis=1``, each inner array is concatenated: the input ``arrays`` must all be jagged with the same outer array length. (Values of ``axis`` greater than ``1`` are not yet supported.)

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    b = awkward.fromiter([[100, 200], [300], [400, 500, 600]])
    awkward.concatenate([a, b])
    # <JaggedArray [[1.1 2.2 3.3] [] [4.4 5.5] [100.0 200.0] [300.0] [400.0 500.0 600.0]] at 0x78112c122c88>

    awkward.concatenate([a, b], axis=1)
    # <JaggedArray [[1.1 2.2 3.3 100.0 200.0] [300.0] [4.4 5.5 400.0 500.0 600.0]] at 0x78112c425978>

    a = awkward.fromiter([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])
    b = awkward.fromiter([{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}])
    awkward.concatenate([a, b]).tolist()
    # [{'x': 1, 'y': 1.1},
    #  {'x': 2, 'y': 2.2},
    #  {'x': 3, 'y': 3.3},
    #  {'x': 4, 'y': 4.4},
    #  {'x': 5, 'y': 5.5}]

If the arrays have different types, their concatenation is a ``UnionArray``.

.. code-block:: python3

    a = awkward.fromiter([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])
    b = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    awkward.concatenate([a, b]).tolist()
    # [{'x': 1, 'y': 1.1},
    #  {'x': 2, 'y': 2.2},
    #  {'x': 3, 'y': 3.3},
    #  [1.1, 2.2, 3.3],
    #  [],
    #  [4.4, 5.5]]

    a = awkward.fromiter([1, None, 2])
    b = awkward.fromiter([None, 3, None])
    awkward.concatenate([a, b])
    # <MaskedArray [1 None 2 None 3 None] at 0x78112c085da0>

    import awkward, numpy
    a = awkward.fromiter(["one", "two", "three"])
    b = awkward.fromiter(["four", "five", "six"])
    awkward.concatenate([a, b])
    # <StringArray ['one' 'two' 'three' 'four' 'five' 'six'] at 0x78112c14f7f0>

    awkward.concatenate([a, b], axis=1)
    # <StringArray ['onefour' 'twofive' 'threesix'] at 0x78112c115518>

Functions for input/output and conversion
-----------------------------------------

Most of the functions defined at the top-level of the library are conversion functions.

* ``awkward.fromiter(iterable, awkwardlib=None, dictencoding=False, maskedwhen=True)``: convert Python or JSON data into awkward arrays. Not a fast function: it necessarily involves a Python for loop. The ``awkwardlib`` determines which awkward module to use to make arrays. If ``dictencoding`` is ``True``, bytes and strings will be "dictionary-encoded" in Arrow/Parquet terms—this is an ``IndexedArray`` in awkward. The ``maskedwhen`` parameter determines whether ``MaskedArrays`` have a mask that is ``True`` when data are missing or ``False`` when data are missing.

.. code-block:: python3

    # We have been using this function all along, but why not another example?
    complicated = awkward.fromiter([[1.1, 2.2, None, 3.3, None],
                                    [4.4, [5.5]],
                                    [{"x": 6, "y": {"z": 7}}, None, {"x": 8, "y": {"z": 9}}]
                                   ])
    complicated
    # <JaggedArray [[1.1 2.2 None 3.3 None] [4.4 [5.5]] [<Row 0> None <Row 1>]] at 0x78112c0ef438>

The fact that this nested, row-wise data have been converted into columnar arrays can be seen by inspecting its ``layout``.

.. code-block:: python3

    complicated.layout
    #  layout
    # [           ()] JaggedArray(starts=layout[0], stops=layout[1], content=layout[2])
    # [            0]   ndarray(shape=3, dtype=dtype('int64'))
    # [            1]   ndarray(shape=3, dtype=dtype('int64'))
    # [            2]   IndexedMaskedArray(mask=layout[2, 0], content=layout[2, 1], maskedwhen=-1)
    # [         2, 0]     ndarray(shape=10, dtype=dtype('int64'))
    # [         2, 1]     UnionArray(tags=layout[2, 1, 0], index=layout[2, 1, 1], contents=[layout[2, 1, 2], layout[2, 1, 3], layout[2, 1, 4]])
    # [      2, 1, 0]       ndarray(shape=7, dtype=dtype('uint8'))
    # [      2, 1, 1]       ndarray(shape=7, dtype=dtype('int64'))
    # [      2, 1, 2]       ndarray(shape=4, dtype=dtype('float64'))
    # [      2, 1, 3]       JaggedArray(starts=layout[2, 1, 3, 0], stops=layout[2, 1, 3, 1], content=layout[2, 1, 3, 2])
    # [   2, 1, 3, 0]         ndarray(shape=1, dtype=dtype('int64'))
    # [   2, 1, 3, 1]         ndarray(shape=1, dtype=dtype('int64'))
    # [   2, 1, 3, 2]         ndarray(shape=1, dtype=dtype('float64'))
    # [      2, 1, 4]       Table(x=layout[2, 1, 4, 0], y=layout[2, 1, 4, 1])
    # [   2, 1, 4, 0]         ndarray(shape=2, dtype=dtype('int64'))
    # [   2, 1, 4, 1]         Table(z=layout[2, 1, 4, 1, 0])
    # [2, 1, 4, 1, 0]           ndarray(shape=2, dtype=dtype('int64'))

    for index, node in complicated.layout.items():
        if node.cls == numpy.ndarray:
            print("[{0:>13s}] {1}".format(", ".join(repr(i) for i in index), repr(node.array)))
    # [            0] array([0, 5, 7])
    # [            1] array([ 5,  7, 10])
    # [         2, 0] array([ 0,  1, -1,  2, -1,  3,  4,  5, -1,  6])
    # [      2, 1, 0] array([0, 0, 0, 0, 1, 2, 2], dtype=uint8)
    # [      2, 1, 1] array([0, 1, 2, 3, 0, 0, 1])
    # [      2, 1, 2] array([1.1, 2.2, 3.3, 4.4])
    # [   2, 1, 3, 0] array([0])
    # [   2, 1, 3, 1] array([1])
    # [   2, 1, 3, 2] array([5.5])
    # [   2, 1, 4, 0] array([6, 8])
    # [2, 1, 4, 1, 0] array([7, 9])

The number of arrays in this object scales with the complexity of its data type, but not with the size of the dataset. If it were as complicated as it is now but billions of elements long, it would still contain 11 Numpy arrays, and operations on it would scale as Numpy scales. However, converting a billion Python objects to these 11 arrays would be a large up-front cost.

More detail on the row-wise to columnar conversion process is given in `docs/fromiter.adoc <https://github.com/scikit-hep/awkward-array/blob/master/docs/fromiter.adoc>`__.

* ``load(file, awkwardlib=None, whitelist=awkward.persist.whitelist, cache=None, schemasuffix=".json")``: loads data from an "awkd" (special ZIP) file. This function is like ``numpy.load``, but for awkward arrays. If the file contains a single object, that object will be read immediately; if it has a collection of named arrays, it will return a loader that loads those arrays on demand. The ``awkwardlib`` determines the module to use to define arrays, the ``whitelist`` is where you can provide a list of functions that may be called in this process, ``cache`` is a global cache object assigned to ``VirtualArrays``, and ``schemasuffix`` determines the file name pattern to look for objects inside the ZIP file.

* ``save(file, array, name=None, mode="a", compression=awkward.persist.compression, delimiter="-", suffix=".raw", schemasuffix=".json")``: saves data to an "awkd" (special ZIP) file. This function is like ``numpy.savez`` and is the reverse of ``load`` (above). The ``array`` may be a single object or a dict of named arrays, the ``name`` is a name to use inside the file, ``mode="a"`` means create or append to an existing file, refusing to overwrite data while ``mode="w"`` overwrites data, ``compression`` is a compression policy (set of rules determining which arrays to compress and how), and the rest of the arguments determine file names within the ZIP: ``delimiter`` between name components, ``suffix`` for array data, and ``schemasuffix`` for the schemas that tell ``load`` how to find all other data.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    b = awkward.fromiter([[1.1, 2.2, None, 3.3, None],
                          [4.4, [5.5]],
                          [{"x": 6, "y": {"z": 7}}, None, {"x": 8, "y": {"z": 9}}]
                         ])

    awkward.save("single.awkd", a, mode="w")

    awkward.load("single.awkd")
    # <JaggedArray [[1.1 2.2 3.3] [] [4.4 5.5]] at 0x78112c14ff98>

    awkward.save("multi.awkd", {"a": a, "b": b}, mode="w")

    multi = awkward.load("multi.awkd")

    multi["a"]
    # <JaggedArray [[1.1 2.2 3.3] [] [4.4 5.5]] at 0x78112c0906d8>

    multi["b"]
    # <JaggedArray [[1.1 2.2 None 3.3 None] [4.4 [5.5]] [<Row 0> None <Row 1>]] at 0x78112c0906a0>

Only ``save`` has a ``compression`` parameter because only the writing process gets to decide how arrays are compressed. We don't use ZIP's built-in compression, but use Python compression functions and encode the choice in the metadata. If ``compression=True``, all arrays will be compressed with zlib; if ``compression=False``, ``None``, or ``[]``, none will. In general, ``compression`` is a list of rules; the first rule that is satisfied by a given array uses the specified compress/decompress pair of functions. Here's the default policy:

.. code-block:: python3

    awkward.persist.compression
    # [{'minsize': 8192,
    #   'types': [numpy.bool_, bool, numpy.integer],
    #   'contexts': '*',
    #   'pair': (<function zlib.compress(data, /, level=-1)>,
    #    ('zlib', 'decompress'))}]

The default policy has only one rule. If any array has a minimum size (``minsize``) of 8 kB (``8192`` bytes), a numeric type (``array.dtype.type``) that is a subclass of ``numpy.bool_``, ``bool``, or ``numpy.integer``, and is in any awkward-array context (``JaggedArray.starts``, ``MaskedArray.mask``, etc.), then it will be compressed with ``zip.compress`` and decompressed with ``('zlib', 'decompress')``. The compression function is given as an object—the Python function that will be called to transform byte strings into compressed byte strings—but the decompression function is given as a location in Python's namespace: a tuple of nested objects, the first of which is a fully qualified module name (submodules separated by dots). This is because only the *location* of the decompression function needs to be written to the file.

The saved awkward array consists of a collection of byte strings for Numpy arrays (2 for object ``a`` and 11 for object ``b``, above) and JSON-formatted metadata that reconstructs the nested hierarchy of awkward classes around those Numpy arrays. This metadata includes information such as which byte strings should be decompressed and how, but also which awkward constructors to call to fit everything together. As such, the JSON metadata is code, a limited language without looping or function definitions (i.e. not Turing complete) but with the ability to call any Python function.

Using a mini-language as metadata gives us great capacity for backward and forward compatibility (new or old ways of encoding things are simply calling different functions), but it does raise the danger of malicious array files calling unwanted Python functions. For this reason, ``load`` refuses to call any functions not specified in a ``whitelist``. The default whitelist consists of functions known to be safe:

.. code-block:: python3

    awkward.persist.whitelist
    # [['numpy', 'frombuffer'],
    #  ['zlib', 'decompress'],
    #  ['lzma', 'decompress'],
    #  ['backports.lzma', 'decompress'],
    #  ['lz4.block', 'decompress'],
    #  ['awkward', '*Array'],
    #  ['awkward', 'Table'],
    #  ['awkward', 'numpy', 'frombuffer'],
    #  ['awkward.util', 'frombuffer'],
    #  ['awkward.persist'],
    #  ['awkward.arrow', '_ParquetFile', 'fromjson'],
    #  ['uproot_methods.classes.*'],
    #  ['uproot_methods.profiles.*'],
    #  ['uproot.tree', '_LazyFiles'],
    #  ['uproot.tree', '_LazyTree'],
    #  ['uproot.tree', '_LazyBranch']]

The format of each item in the whitelist is a list of nested objects, the first of which being a fully qualified module name (submodules separated by dots). For instance, in the ``awkward.arrow`` submodule, there is a class named ``_ParquetFile`` and it has a static method ``fromjson`` that is deemed to be safe. Patterns of safe names are can be wildcarded, such as ``['awkward', '*Array']`` and ``['uproot_methods.classes.*']``.

You can add your own functions, and forward compatibility (using data made by a new version in an old version of awkward-array) often dictates that you must add a function manually. The error message explains how to do this.

The same serialization format is used when you pickle an awkward array or save it in an HDF5 file. More detail on the metadata mini-language is given in `docs/serialization.adoc <https://github.com/scikit-hep/awkward-array/blob/master/docs/serialization.adoc>`__.

* ``hdf5(group, awkwardlib=None, compression=awkward.persist.compression, whitelist=awkward.persist.whitelist, cache=None)``: wrap a ``h5py.Group`` as an awkward-aware group, to save awkward arrays to HDF5 files and to read them back again. The options have the same meaning as ``load`` and ``save``.

Unlike "awkd" (special ZIP) files, HDF5 files can be written and overwritten like a database, rather than write-once files.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    b = awkward.fromiter([[1.1, 2.2, None, 3.3, None],
                          [4.4, [5.5]],
                          [{"x": 6, "y": {"z": 7}}, None, {"x": 8, "y": {"z": 9}}]
                         ])

    import h5py
    f = h5py.File("awkward.hdf5", "w")
    f
    # <HDF5 file "awkward.hdf5" (mode r+)>

    g = awkward.hdf5(f)
    g
    # <awkward.hdf5 '/' (0 members)>

    g["array"] = a

    g["array"]
    # <JaggedArray [[1.1 2.2 3.3] [] [4.4 5.5]] at 0x781115141320>

    del g["array"]

    g["array"] = b

    g["array"]
    # <JaggedArray [[1.1 2.2 None 3.3 None] [4.4 [5.5]] [<Row 0> None <Row 1>]] at 0x7811883b9198>

The HDF5 format does not include columnar representations of arbitrary nested data, as awkward-array does, so what we're actually storing are plain Numpy arrays and the metadata necessary to reconstruct the awkward array.

.. code-block:: python3

    # Reopen file, without wrapping it as awkward.hdf5 this time.
    f = h5py.File("awkward.hdf5", "r")
    f
    # <HDF5 file "awkward.hdf5" (mode r+)>

    f["array"]
    # <HDF5 group "/array" (9 members)>

    f["array"].keys()
    # <KeysViewHDF5 ['1', '12', '14', '16', '19', '4', '7', '9', 'schema.json']>

The "schema.json" array is the JSON metadata, containing directives like ``{"call": ["awkward", "JaggedArray", "fromcounts"]}`` and ``{"read": "1"}`` meaning the array named ``"1"``, etc.

.. code-block:: python3

    import json
    json.loads(f["array"]["schema.json"][:].tostring())
    # {'awkward': '0.12.0rc1',
    #  'schema': {'call': ['awkward', 'JaggedArray', 'fromcounts'],
    #   'args': [{'call': ['awkward', 'numpy', 'frombuffer'],
    #     'args': [{'read': '1'}, {'dtype': 'int64'}, {'json': 3, 'id': 2}],
    #     'id': 1},
    #    {'call': ['awkward', 'IndexedMaskedArray'],
    #     'args': [{'call': ['awkward', 'numpy', 'frombuffer'],
    #       'args': [{'read': '4'}, {'dtype': 'int64'}, {'json': 10, 'id': 5}],
    #       'id': 4},
    #      {'call': ['awkward', 'UnionArray', 'fromtags'],
    #       'args': [{'call': ['awkward', 'numpy', 'frombuffer'],
    #         'args': [{'read': '7'}, {'dtype': 'uint8'}, {'json': 7, 'id': 8}],
    #         'id': 7},
    #        {'list': [{'call': ['awkward', 'numpy', 'frombuffer'],
    #           'args': [{'read': '9'}, {'dtype': 'float64'}, {'json': 4, 'id': 10}],
    #           'id': 9},
    #          {'call': ['awkward', 'JaggedArray', 'fromcounts'],
    #           'args': [{'call': ['awkward', 'numpy', 'frombuffer'],
    #             'args': [{'read': '12'},
    #              {'dtype': 'int64'},
    #              {'json': 1, 'id': 13}],
    #             'id': 12},
    #            {'call': ['awkward', 'numpy', 'frombuffer'],
    #             'args': [{'read': '14'}, {'dtype': 'float64'}, {'ref': 13}],
    #             'id': 14}],
    #           'id': 11},
    #          {'call': ['awkward', 'Table', 'frompairs'],
    #           'args': [{'pairs': [['x',
    #               {'call': ['awkward', 'numpy', 'frombuffer'],
    #                'args': [{'read': '16'},
    #                 {'dtype': 'int64'},
    #                 {'json': 2, 'id': 17}],
    #                'id': 16}],
    #              ['y',
    #               {'call': ['awkward', 'Table', 'frompairs'],
    #                'args': [{'pairs': [['z',
    #                    {'call': ['awkward', 'numpy', 'frombuffer'],
    #                     'args': [{'read': '19'}, {'dtype': 'int64'}, {'ref': 17}],
    #                     'id': 19}]]},
    #                 {'json': 0}],
    #                'id': 18}]]},
    #            {'json': 0}],
    #           'id': 15}]}],
    #       'id': 6},
    #      {'json': -1}],
    #     'id': 3}],
    #   'id': 0},
    #  'prefix': 'array/'}

Without awkward-array, these objects can't be meaningfully read back from the HDF5 file.

* ``awkward.fromarrow(arrow, awkwardlib=None)``: convert an `Apache Arrow <https://arrow.apache.org>`__ formatted buffer to an awkward array (zero-copy). The ``awkwardlib`` parameter has the same meaning as above.

* ``awkward.toarrow(array)``: convert an awkward array to an Apache Arrow buffer, if possible (involving a data copy, but no Python loops).

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    b = awkward.fromiter([[1.1, 2.2, None, 3.3, None],
                          [4.4, [5.5]],
                          [{"x": 6, "y": {"z": 7}}, None, {"x": 8, "y": {"z": 9}}]
                         ])

    awkward.toarrow(a)
    # <pyarrow.lib.ListArray object at 0x78110846b1a8>
    # [
    #   [
    #     1.1,
    #     2.2,
    #     3.3
    #   ],
    #   [],
    #   [
    #     4.4,
    #     5.5
    #   ]
    # ]

    awkward.fromarrow(awkward.toarrow(a))
    # <JaggedArray [[1.1 2.2 3.3] [] [4.4 5.5]] at 0x78110846d550>

    awkward.toarrow(b)
    # <pyarrow.lib.ListArray object at 0x78110846b6d0>
    # [
    #   -- is_valid: all not null
    #   -- type_ids:     [
    #       0,
    #       0,
    #       2,
    #       0,
    #       2
    #     ]
    #   -- value_offsets:     [
    #       0,
    #       1,
    #       1,
    #       2,
    #       1
    #     ]
    #   -- child 0 type: double
    #     [
    #       1.1,
    #       2.2,
    #       3.3,
    #       4.4
    #     ]
    #   -- child 1 type: list<item: double>
    #     [
    #       [
    #         5.5
    #       ]
    #     ]
    #   -- child 2 type: struct<x: int64, y: struct<z: int64>>
    #     -- is_valid: all not null
    #     -- child 0 type: int64
    #       [
    #         6,
    #         8
    #       ]
    #     -- child 1 type: struct<z: int64>
    #       -- is_valid: all not null
    #       -- child 0 type: int64
    #         [
    #           7,
    #           9
    #         ],
    #   -- is_valid: all not null
    #   -- type_ids:     [
    #       0,
    #       1
    #     ]
    #   -- value_offsets:     [
    #       3,
    #       0
    #     ]
    #   -- child 0 type: double
    #     [
    #       1.1,
    #       2.2,
    #       3.3,
    #       4.4
    #     ]
    #   -- child 1 type: list<item: double>
    #     [
    #       [
    #         5.5
    #       ]
    #     ]
    #   -- child 2 type: struct<x: int64, y: struct<z: int64>>
    #     -- is_valid: all not null
    #     -- child 0 type: int64
    #       [
    #         6,
    #         8
    #       ]
    #     -- child 1 type: struct<z: int64>
    #       -- is_valid: all not null
    #       -- child 0 type: int64
    #         [
    #           7,
    #           9
    #         ],
    #   -- is_valid: all not null
    #   -- type_ids:     [
    #       2,
    #       2,
    #       2
    #     ]
    #   -- value_offsets:     [
    #       0,
    #       1,
    #       1
    #     ]
    #   -- child 0 type: double
    #     [
    #       1.1,
    #       2.2,
    #       3.3,
    #       4.4
    #     ]
    #   -- child 1 type: list<item: double>
    #     [
    #       [
    #         5.5
    #       ]
    #     ]
    #   -- child 2 type: struct<x: int64, y: struct<z: int64>>
    #     -- is_valid: all not null
    #     -- child 0 type: int64
    #       [
    #         6,
    #         8
    #       ]
    #     -- child 1 type: struct<z: int64>
    #       -- is_valid: all not null
    #       -- child 0 type: int64
    #         [
    #           7,
    #           9
    #         ]
    # ]

    awkward.fromarrow(awkward.toarrow(b))
    # <JaggedArray [[1.1 2.2 <Row 1> 3.3 <Row 1>] [4.4 [5.5]] [<Row 0> <Row 1> <Row 1>]] at 0x78110846de48>

Unlike HDF5, Arrow is capable of columnar jagged arrays, nullable values, nested structures, etc. If you save an awkward array in Arrow format, someone else can read it without the awkward-array library. There are a few awkward array classes that don't have an Arrow equivalent, though. Below is a list of all translations.

* Numpy array → Arrow `BooleanArray <https://arrow.apache.org/docs/python/generated/pyarrow.BooleanArray.html>`__, `IntegerArray <https://arrow.apache.org/docs/python/generated/pyarrow.IntegerArray.html>`__, or `FloatingPointArray <https://arrow.apache.org/docs/python/generated/pyarrow.FloatingPointArray.html>`__.
* ``JaggedArray`` → Arrow `ListArray <https://arrow.apache.org/docs/python/generated/pyarrow.ListArray.html>`__.
* ``StringArray`` → Arrow `StringArray <https://arrow.apache.org/docs/python/generated/pyarrow.StringArray.html>`__.
* ``Table`` → Arrow `Table <https://arrow.apache.org/docs/python/generated/pyarrow.Table.html>`__ at top-level, but an Arrow `StructArray <https://arrow.apache.org/docs/python/generated/pyarrow.StructArray.html>`__ if nested.
* ``MaskedArray`` → missing data mask (nullability in Arrow is an array attribute, rather than an array wrapper).
* ``IndexedMaskedArray`` → unfolded into a simple mask before the Arrow translation.
* ``IndexedArray`` → Arrow `DictionaryArray <https://arrow.apache.org/docs/python/generated/pyarrow.DictionaryArray.html>`__.
* ``SparseArray`` → converted to a dense array before the Arrow translation.
* ``ObjectArray`` → Pythonic interpretation is discarded before the Arrow translation.
* ``UnionArray`` → Arrow dense `UnionArray <https://arrow.apache.org/docs/python/generated/pyarrow.UnionArray.html>`__ if possible, sparse UnionArray if necessary.
* ``ChunkedArray`` (including ``AppendableArray``) → Arrow `RecordBatches <https://arrow.apache.org/docs/python/generated/pyarrow.RecordBatch.html>`__, but only at top-level: nested ``ChunkedArrays`` cannot be converted.
* ``VirtualArray`` → array gets materialized before the Arrow translation (i.e. the lazy-loading is not preserved).

Since Arrow is an in-memory format, both ``toarrow`` and ``fromarrow`` are side-effect-free functions with a return value. Functions that write to files have a side-effect (the state of your disk changing) and no return value. Once you've made your Arrow buffer, you have to figure out what to do with it. (You may want to `write it to a stream <https://arrow.apache.org/docs/python/ipc.html>`__ for interprocess communication.)

* ``awkward.fromparquet(where, awkwardlib=None)``: reads from a Parquet file (at filename/URI ``where``) into an awkward array, through pyarrow. The ``awkwardlib`` parameter has the same meaning as above.

* ``awkward.toparquet(where, array, schema=None)``: writes an awkward array to a Parquet file (at filename/URI ``where``), through pyarrow. The Parquet ``schema`` may be inferred from the awkward array or explicitly specified.

Like Arrow and unlike HDF5, Parquet natively stores complex data structures in a columnar format and doesn't need to be wrapped by an interpretation layer like ``awkward.hdf5``. Like HDF5 and unlike Arrow, Parquet is a file format, intended for storage.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    b = awkward.fromiter([[1.1, 2.2, None, 3.3, None],
                          [4.4, [5.5]],
                          [{"x": 6, "y": {"z": 7}}, None, {"x": 8, "y": {"z": 9}}]
                         ])

    awkward.toparquet("dataset.parquet", a)

    a2 = awkward.fromparquet("dataset.parquet")
    a2
    # <ChunkedArray [[1.1 2.2 3.3] [] [4.4 5.5]] at 0x78110846dc50>

Notice that we get a ``ChunkedArray`` back. This is because ``awkward.fromparquet`` is lazy-loading the Parquet file, which might be very large (not in this case, obviously). It's actually a ``ChunkedArray`` (one `row group <https://parquet.apache.org/documentation/latest/#unit-of-parallelization>`__ per chunk) of ``VirtualArrays``, and each ``VirtualArray`` is read when it is accessed (for instance, to print it above).

.. code-block:: python3

    a2.chunks
    # [<VirtualArray [[1.1 2.2 3.3] [] [4.4 5.5]] at 0x78110846dc18>]

    a2.chunks[0].array
    # <BitMaskedArray [[1.1 2.2 3.3] [] [4.4 5.5]] at 0x78110849aeb8>

The next layer of new structure is that the jagged array is bit-masked. Even though none of the values are nullable, this is an artifact of the way Parquet formats columnar data.

.. code-block:: python3

    a2.chunks[0].array.content
    # <JaggedArray [[1.1 2.2 3.3] [] [4.4 5.5]] at 0x78110849a518>

    a2.layout
    #  layout
    # [           ()] ChunkedArray(chunks=[layout[0]], chunksizes=[3])
    # [            0]   VirtualArray(generator=<awkward.arrow._ParquetFile object at 0x78110846df98>, args=(0, ''), kwargs={}, array=layout[0, 0])
    # [         0, 0]     BitMaskedArray(mask=layout[0, 0, 0], content=layout[0, 0, 1], maskedwhen=False, lsborder=True)
    # [      0, 0, 0]       ndarray(shape=1, dtype=dtype('uint8'))
    # [      0, 0, 1]       JaggedArray(starts=layout[0, 0, 1, 0], stops=layout[0, 0, 1, 1], content=layout[0, 0, 1, 2])
    # [   0, 0, 1, 0]         ndarray(shape=3, dtype=dtype('int32'))
    # [   0, 0, 1, 1]         ndarray(shape=3, dtype=dtype('int32'))
    # [   0, 0, 1, 2]         BitMaskedArray(mask=layout[0, 0, 1, 2, 0], content=layout[0, 0, 1, 2, 1], maskedwhen=False, lsborder=True)
    # [0, 0, 1, 2, 0]           ndarray(shape=1, dtype=dtype('uint8'))
    # [0, 0, 1, 2, 1]           ndarray(shape=5, dtype=dtype('float64'))

Fewer types can be written to Parquet files than Arrow buffers, since pyarrow does not yet have a complete Arrow → Parquet transformation.

.. code-block:: python3

    try:
        awkward.toparquet("dataset2.parquet", b)
    except Exception as err:
        print(type(err), str(err))
    # <class 'pyarrow.lib.ArrowNotImplementedError'> Unhandled type for Arrow to Parquet schema conversion: union[dense]<0: double=0, 1: list<item: double>=1, 2: struct<x: int64, y: struct<z: int64>>=2>

* ``awkward.topandas(array, flatten=False)``: convert the array into a Pandas DataFrame (if tabular) or a Pandas Series (otherwise). If ``flatten=False``, wrap the awkward arrays as a new Pandas extension type (not fully implemented). If ``flatten=True``, convert the jaggedness and nested tables into row and column ``pandas.MultiIndex`` without introducing any new types (not always possible).

.. code-block:: python3

    a = awkward.Table(x=awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]),
                      y=awkward.fromiter([100, 200, 300, 400]))
    df = awkward.topandas(a)
    df

.. raw:: html

      <table border="0" class="dataframe">
        <thead>
          <tr style="text-align: right;">
            <th></th>
            <th>x</th>
            <th>y</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>0</th>
            <td>[1.1 2.2 3.3]</td>
            <td>100</td>
          </tr>
          <tr>
            <th>1</th>
            <td>[]</td>
            <td>200</td>
          </tr>
          <tr>
            <th>2</th>
            <td>[4.4 5.5]</td>
            <td>300</td>
          </tr>
          <tr>
            <th>3</th>
            <td>[6.6 7.7 8.8 9.9]</td>
            <td>400</td>
          </tr>
        </tbody>
      </table>

.. code-block:: python3

    df.x
    # 0        [1.1 2.2 3.3]
    # 1                   []
    # 2            [4.4 5.5]
    # 3    [6.6 7.7 8.8 9.9]
    # Name: x, dtype: awkward

Note that the ``dtype`` is ``awkward``. The array has not been converted into Numpy ``dtype=object`` (which would imply a performance loss); it has been wrapped as a container that Pandas recognizes. You can get the awkward array back the same way you would a Numpy array:

.. code-block:: python3

    df.x.values
    # <JaggedSeries [[1.1 2.2 3.3] [] [4.4 5.5] [6.6 7.7 8.8 9.9]] at 0x78110846d400>

(``JaggedSeries`` is a thin wrapper on ``JaggedArray``; they behave the same way.)

The value of this is that awkward slice semantics can be applied to data in Pandas.

.. code-block:: python3

    df[1:]

.. raw:: html

       <table border="0" class="dataframe">
         <thead>
           <tr style="text-align: right;">
             <th></th>
             <th>x</th>
             <th>y</th>
           </tr>
         </thead>
         <tbody>
           <tr>
             <th>1</th>
             <td>[]</td>
             <td>200</td>
           </tr>
           <tr>
             <th>2</th>
             <td>[4.4 5.5]</td>
             <td>300</td>
           </tr>
           <tr>
             <th>3</th>
             <td>[6.6 7.7 8.8 9.9]</td>
             <td>400</td>
           </tr>
         </tbody>
       </table>

.. code-block:: python3

    df.x[df.x.values.counts > 0]
    # 0        [1.1 2.2 3.3]
    # 2            [4.4 5.5]
    # 3    [6.6 7.7 8.8 9.9]
    # Name: x, dtype: awkward

However, Pandas has a (limited) way of handling jaggedness and nested tables, with ``pandas.MultiIndex`` rows and columns, respectively.

.. code-block:: python3

    # Nested tables become MultiIndex-valued column names.
    array = awkward.fromiter([{"a": {"b": 1, "c": {"d": [2]}}, "e": 3},
                              {"a": {"b": 4, "c": {"d": [5, 5.1]}}, "e": 6},
                              {"a": {"b": 7, "c": {"d": [8, 8.1, 8.2]}}, "e": 9}])
    df = awkward.topandas(array, flatten=True)
    df

.. raw:: html

      <table border="0" class="dataframe">
        <thead>
          <tr>
            <th></th>
            <th></th>
            <th colspan="2" halign="left">a</th>
            <th>e</th>
          </tr>
          <tr>
            <th></th>
            <th></th>
            <th>b</th>
            <th>c</th>
            <th></th>
          </tr>
          <tr>
            <th></th>
            <th></th>
            <th></th>
            <th>d</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>0</th>
            <th>0</th>
            <td>1</td>
            <td>2.0</td>
            <td>3</td>
          </tr>
          <tr>
            <th rowspan="2" valign="top">1</th>
            <th>0</th>
            <td>4</td>
            <td>5.0</td>
            <td>6</td>
          </tr>
          <tr>
            <th>1</th>
            <td>4</td>
            <td>5.1</td>
            <td>6</td>
          </tr>
          <tr>
            <th rowspan="3" valign="top">2</th>
            <th>0</th>
            <td>7</td>
            <td>8.0</td>
            <td>9</td>
          </tr>
          <tr>
            <th>1</th>
            <td>7</td>
            <td>8.1</td>
            <td>9</td>
          </tr>
          <tr>
            <th>2</th>
            <td>7</td>
            <td>8.2</td>
            <td>9</td>
          </tr>
        </tbody>
      </table>

.. code-block:: python3

    # Jagged arrays become MultiIndex-valued rows (index).
    array = awkward.fromiter([{"a": 1, "b": [[2.2, 3.3, 4.4], [], [5.5, 6.6]]},
                              {"a": 10, "b": [[1.1], [2.2, 3.3], [], [4.4]]},
                              {"a": 100, "b": [[], [9.9]]}])
    df = awkward.topandas(array, flatten=True)
    df

.. raw:: html

      <table border="0" class="dataframe">
        <thead>
          <tr>
            <th></th>
            <th></th>
            <th></th>
            <th>a</th>
            <th>b</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th rowspan="5" valign="top">0</th>
            <th rowspan="3" valign="top">0</th>
            <th>0</th>
            <td>1</td>
            <td>2.2</td>
          </tr>
          <tr>
            <th>1</th>
            <td>1</td>
            <td>3.3</td>
          </tr>
          <tr>
            <th>2</th>
            <td>1</td>
            <td>4.4</td>
          </tr>
          <tr>
            <th rowspan="2" valign="top">2</th>
            <th>0</th>
            <td>1</td>
            <td>5.5</td>
          </tr>
          <tr>
            <th>1</th>
            <td>1</td>
            <td>6.6</td>
          </tr>
          <tr>
            <th rowspan="4" valign="top">1</th>
            <th>0</th>
            <th>0</th>
            <td>10</td>
            <td>1.1</td>
          </tr>
          <tr>
            <th rowspan="2" valign="top">1</th>
            <th>0</th>
            <td>10</td>
            <td>2.2</td>
          </tr>
          <tr>
            <th>1</th>
            <td>10</td>
            <td>3.3</td>
          </tr>
          <tr>
            <th>3</th>
            <th>0</th>
            <td>10</td>
            <td>4.4</td>
          </tr>
          <tr>
            <th>2</th>
            <th>1</th>
            <th>0</th>
            <td>100</td>
            <td>9.9</td>
          </tr>
        </tbody>
      </table>

The advantage of this is that no new column types are introduced, and Pandas already has functions for managing structure in its ``MultiIndex``. For instance, this structure can be unstacked into Pandas's columns.

.. code-block:: python3

    df.unstack()

.. raw:: html

      <table border="0" class="dataframe">
        <thead>
          <tr>
            <th></th>
            <th></th>
            <th colspan="3" halign="left">a</th>
            <th colspan="3" halign="left">b</th>
          </tr>
          <tr>
            <th></th>
            <th></th>
            <th>0</th>
            <th>1</th>
            <th>2</th>
            <th>0</th>
            <th>1</th>
            <th>2</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th rowspan="2" valign="top">0</th>
            <th>0</th>
            <td>1.0</td>
            <td>1.0</td>
            <td>1.0</td>
            <td>2.2</td>
            <td>3.3</td>
            <td>4.4</td>
          </tr>
          <tr>
            <th>2</th>
            <td>1.0</td>
            <td>1.0</td>
            <td>NaN</td>
            <td>5.5</td>
            <td>6.6</td>
            <td>NaN</td>
          </tr>
          <tr>
            <th rowspan="3" valign="top">1</th>
            <th>0</th>
            <td>10.0</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>1.1</td>
            <td>NaN</td>
            <td>NaN</td>
          </tr>
          <tr>
            <th>1</th>
            <td>10.0</td>
            <td>10.0</td>
            <td>NaN</td>
            <td>2.2</td>
            <td>3.3</td>
            <td>NaN</td>
          </tr>
          <tr>
            <th>3</th>
            <td>10.0</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>4.4</td>
            <td>NaN</td>
            <td>NaN</td>
          </tr>
          <tr>
            <th>2</th>
            <th>1</th>
            <td>100.0</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>9.9</td>
            <td>NaN</td>
            <td>NaN</td>
          </tr>
        </tbody>
      </table>

.. code-block:: python3

    df.unstack().unstack()

.. raw:: html

      <table border="0" class="dataframe">
        <thead>
          <tr>
            <th></th>
            <th colspan="10" halign="left">a</th>
            <th>...</th>
            <th colspan="10" halign="left">b</th>
          </tr>
          <tr>
            <th></th>
            <th colspan="4" halign="left">0</th>
            <th colspan="4" halign="left">1</th>
            <th colspan="2" halign="left">2</th>
            <th>...</th>
            <th colspan="2" halign="left">0</th>
            <th colspan="4" halign="left">1</th>
            <th colspan="4" halign="left">2</th>
          </tr>
          <tr>
            <th></th>
            <th>0</th>
            <th>1</th>
            <th>2</th>
            <th>3</th>
            <th>0</th>
            <th>1</th>
            <th>2</th>
            <th>3</th>
            <th>0</th>
            <th>1</th>
            <th>...</th>
            <th>2</th>
            <th>3</th>
            <th>0</th>
            <th>1</th>
            <th>2</th>
            <th>3</th>
            <th>0</th>
            <th>1</th>
            <th>2</th>
            <th>3</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>0</th>
            <td>1.0</td>
            <td>NaN</td>
            <td>1.0</td>
            <td>NaN</td>
            <td>1.0</td>
            <td>NaN</td>
            <td>1.0</td>
            <td>NaN</td>
            <td>1.0</td>
            <td>NaN</td>
            <td>...</td>
            <td>5.5</td>
            <td>NaN</td>
            <td>3.3</td>
            <td>NaN</td>
            <td>6.6</td>
            <td>NaN</td>
            <td>4.4</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
          </tr>
          <tr>
            <th>1</th>
            <td>10.0</td>
            <td>10.0</td>
            <td>NaN</td>
            <td>10.0</td>
            <td>NaN</td>
            <td>10.0</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>...</td>
            <td>NaN</td>
            <td>4.4</td>
            <td>NaN</td>
            <td>3.3</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
          </tr>
          <tr>
            <th>2</th>
            <td>NaN</td>
            <td>100.0</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>...</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
          </tr>
        </tbody>
      </table>

It is also possible to get `Pandas Series and DataFrames through Arrow <https://arrow.apache.org/docs/python/pandas.html>`__, though this doesn't handle jagged arrays well: they get converted into Numpy ``dtype=object`` arrays.

.. code-block:: python3

    df = awkward.toarrow(array).to_pandas()
    df

.. raw:: html

      <table border="0" class="dataframe">
        <thead>
          <tr style="text-align: right;">
            <th></th>
            <th>a</th>
            <th>b</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>0</th>
            <td>1</td>
            <td>[[2.2, 3.3, 4.4], [], [5.5, 6.6]]</td>
          </tr>
          <tr>
            <th>1</th>
            <td>10</td>
            <td>[[1.1], [2.2, 3.3], [], [4.4]]</td>
          </tr>
          <tr>
            <th>2</th>
            <td>100</td>
            <td>[[], [9.9]]</td>
          </tr>
        </tbody>
      </table>

.. code-block:: python3

    df.b
    # 0    [[2.2, 3.3, 4.4], [], [5.5, 6.6]]
    # 1       [[1.1], [2.2, 3.3], [], [4.4]]
    # 2                          [[], [9.9]]
    # Name: b, dtype: object

    df.b[0]
    # array([array([2.2, 3.3, 4.4]), array([], dtype=float64),
    #        array([5.5, 6.6])], dtype=object)

High-level types
----------------

The high-level type of an array describes its characteristics in terms of what it *represents*, a *logical* view of the data. By contrast, the layouts (below) describe the nested arrays themselves, a *physical* view of the data.

The logical view of Numpy arrays is described in terms of ``shape`` and ``dtype``. The awkward type of a Numpy array is presented a little differently.

.. code-block:: python3

    a = numpy.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
    t = awkward.type.fromarray(a)
    t
    # ArrayType(3, 2, dtype('float64'))

Above is the object-form of the high-level type and object that ``takes`` arguments ``to`` return values.

.. code-block:: python3

    t.takes
    # 3

    t.to
    # ArrayType(2, dtype('float64'))

    t.to.to
    # dtype('float64')

High-level type objects also have a printable form for human readability.

.. code-block:: python3

    print(t)
    # [0, 3) -> [0, 2) -> float64

The above should be read like a function's data type: ``argument type -> return type`` for the function that takes an index in square brackets and returns something else. For example, the first ``[0, 3)`` means that you could put any non-negative integer less than ``3`` in square brackets after the array, like this:

.. code-block:: python3

    a[2]
    # array([5.5, 6.6])

The second ``[0, 2)`` means that the next argument can be any non-negative integer less than ``2``.

.. code-block:: python3

    a[2][1]
    # 6.6

And then you have a Numpy ``dtype``.

The reason high-level types are expressed like this, instead of Numpy ``shape`` and ``dtype`` is to generalize to arbitrary objects.

.. code-block:: python3

    a = awkward.fromiter([{"x": 1, "y": []}, {"x": 2, "y": [1.1, 2.2]}, {"x": 3, "y": [1.1, 2.2, 3.3]}])
    print(a.type)
    # [0, 3) -> 'x' -> int64
    #           'y' -> [0, inf) -> float64

In the above, you could call ``a[2]["x"]`` to get ``3`` or ``a[2]["y"][1]`` to get ``2.2``, but the types and even number of allowed arguments depend on which path you take. Numpy's ``shape`` and ``dtype`` have no equivalent.

Also in the above, the allowed argument for the jagged array is specified as ``[0, inf)``, which doesn't literally mean any value up to infinity is allowed—the constraint simply isn't specific because it depends on the details of the jagged array. Even specifying the maximum length of any sublist (``a["y"].counts.max()``) would require a calculation that scales with the size of the dataset, which can be infeasible in some cases. Instead, ``[0, inf)`` simply means "jagged."

Fixed-length arrays inside of ``JaggedArrays`` or ``Tables`` are presented with known upper limits:

.. code-block:: python3

    a = awkward.Table(x=[[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]],
                      y=awkward.fromiter([[1, 2, 3], [], [4, 5]]))
    print(a.type)
    # [0, 3) -> 'x' -> [0, 2) -> float64
    #           'y' -> [0, inf) -> int64

Whereas each value of a ``Table`` row (`product type <https://en.wikipedia.org/wiki/Product_type>`__) contains a member of every one of its fields, each value of a ``UnionArray`` item (`sum type <https://en.wikipedia.org/wiki/Tagged_union>`__) contains a member of exactly one of its possibilities. The distinction is drawn as the lack or presence of a vertical bar (meaning "or": ``|``).

.. code-block:: python3

    a = awkward.fromiter([{"x": 1, "y": "one"}, {"x": 2, "y": "two"}, {"x": 3, "y": "three"}])
    print(a.type)
    # [0, 3) -> 'x' -> int64
    #           'y' -> <class 'str'>

    a = awkward.fromiter([1, 2, 3, "four", "five", "six"])
    print(a.type)
    # [0, 6) -> (int64         |
    #            <class 'str'> )

The parenthesis is to keep ``Table`` fields from being mixed up with ``UnionArray`` possibilities.

.. code-block:: python3

    a = awkward.fromiter([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": "three"}, {"x": 4, "y": "four"}])
    print(a.type)
    # [0, 4) -> 'x' -> int64
    #           'y' -> (float64       |
    #                   <class 'str'> )

As in mathematics, products and the adjacency operator take precedence over sums.

.. code-block:: python3

    a = awkward.fromiter([1, 2, 3, {"x": 4.4, "y": "four"}, {"x": 5.5, "y": "five"}, {"x": 6.6, "y": "six"}])
    print(a.type)
    # [0, 6) -> (int64                |
    #            'x' -> float64
    #            'y' -> <class 'str'> )

Missing data, represented by ``MaskedArrays``, ``BitMaskedArrays``, or ``IndexedMaskedArrays``, are called "option types" in the high-level type language.

.. code-block:: python3

    a = awkward.fromiter([1, 2, 3, None, None, 4, 5])
    print(a.type)
    # [0, 7) -> ?(int64)

    # Inner arrays could be missing values.
    a = awkward.fromiter([[1.1, 2.2, 3.3], None, [4.4, 5.5]])
    print(a.type)
    # [0, 3) -> ?([0, inf) -> float64)

    # Numbers in those arrays could be missing values.
    a = awkward.fromiter([[1.1, 2.2, None], [], [4.4, 5.5]])
    print(a.type)
    # [0, 3) -> [0, inf) -> ?(float64)

Cross-references and cyclic references are expressed in awkward type objects by creating the same graph structure among the type objects as the arrays. Thus,

.. code-block:: python3

    tree = awkward.fromiter([
        {"value": 1.23, "left":    1, "right":    2},     # node 0
        {"value": 3.21, "left":    3, "right":    4},     # node 1
        {"value": 9.99, "left":    5, "right":    6},     # node 2
        {"value": 3.14, "left":    7, "right": None},     # node 3
        {"value": 2.71, "left": None, "right":    8},     # node 4
        {"value": 5.55, "left": None, "right": None},     # node 5
        {"value": 8.00, "left": None, "right": None},     # node 6
        {"value": 9.00, "left": None, "right": None},     # node 7
        {"value": 0.00, "left": None, "right": None},     # node 8
    ])
    left = tree.contents["left"].content
    right = tree.contents["right"].content
    left[(left < 0) | (left > 8)] = 0         # satisfy overzealous validity checks
    right[(right < 0) | (right > 8)] = 0
    tree.contents["left"].content = awkward.IndexedArray(left, tree)
    tree.contents["right"].content = awkward.IndexedArray(right, tree)

    tree[0].tolist()
    # {'left': {'left': {'left': {'left': None, 'right': None, 'value': 9.0},
    #    'right': None,
    #    'value': 3.14},
    #   'right': {'left': None,
    #    'right': {'left': None, 'right': None, 'value': 0.0},
    #    'value': 2.71},
    #   'value': 3.21},
    #  'right': {'left': {'left': None, 'right': None, 'value': 5.55},
    #   'right': {'left': None, 'right': None, 'value': 8.0},
    #   'value': 9.99},
    #  'value': 1.23}

In the print-out, labels (``T0 :=``, ``T1 :=``, ``T2 :=``) are inserted to indicate where cross-references begin and end.

.. code-block:: python3

    print(tree.type)
    # [0, 9) -> 'left'  -> T0 := ?(T1 := 'left'  -> T0
    #                                    'right' -> T2 := ?(T1)
    #                                    'value' -> float64)
    #           'right' -> T2
    #           'value' -> float64

The ``ObjectArray`` class turns awkward array structures into Python objects on demand. From an analysis point of view, the elements of the array *are* Python objects, and that is reflected in the type.

.. code-block:: python3

    class Point:
        def __init__(self, x, y):
            self.x, self.y = x, y
        def __repr__(self):
            return "Point({0}, {1})".format(self.x, self.y)

    a = awkward.fromiter([Point(0, 0), Point(3, 2), Point(1, 1), Point(2, 4), Point(0, 0)])
    a
    # <ObjectArray [Point(0, 0) Point(3, 2) Point(1, 1) Point(2, 4) Point(0, 0)] at 0x781106089390>

    print(a.type)
    # [0, 5) -> <function ObjectFillable.finalize.<locals>.make at 0x781106085a60>

In summary,

* each element of a Numpy ``shape`` like ``(i, j, k)`` becomes a functional argument: ``[0, i) -> [0, j) -> [0, k)``;
* high-level types terminate on Numpy ``dtypes`` or ``ObjectArray`` functions;
* columns of a ``Table`` are presented adjacent to one another: the type is field 1 *and* field 2 *and* field 3, etc.;
* possibilities of a ``UnionArray`` are separated by vertical bars ``|``: the type is possibility 1 *or* possibility 2 *or* possibility 3, etc.;
* nullable types are indicated by a question mark;
* cross-references and cyclic references are maintained in the type objects, printed with labels.

Low-level layouts
-----------------

The layout of an array describes how it is constructed in terms of Numpy arrays and other parameters. It has more information than a high-level type (above), more that would typically be needed for data analysis, but very necessary for data engineering.

A ``Layout`` object is a mapping from position tuples to ``LayoutNodes``. The screen representation is sufficient for reading.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    t = a.layout
    t
    #  layout
    # [    ()] JaggedArray(starts=layout[0], stops=layout[1], content=layout[2])
    # [     0]   ndarray(shape=3, dtype=dtype('int64'))
    # [     1]   ndarray(shape=3, dtype=dtype('int64'))
    # [     2]   ndarray(shape=5, dtype=dtype('float64'))

    t[2]
    # <LayoutNode [(2,)] ndarray>

    t[2].array
    # array([1.1, 2.2, 3.3, 4.4, 5.5])

    a = awkward.fromiter([[[1.1, 2.2], [3.3]], [], [[4.4, 5.5]]])
    t = a.layout
    t
    #  layout
    # [    ()] JaggedArray(starts=layout[0], stops=layout[1], content=layout[2])
    # [     0]   ndarray(shape=3, dtype=dtype('int64'))
    # [     1]   ndarray(shape=3, dtype=dtype('int64'))
    # [     2]   JaggedArray(starts=layout[2, 0], stops=layout[2, 1], content=layout[2, 2])
    # [  2, 0]     ndarray(shape=3, dtype=dtype('int64'))
    # [  2, 1]     ndarray(shape=3, dtype=dtype('int64'))
    # [  2, 2]     ndarray(shape=5, dtype=dtype('float64'))

    t[2]
    # <LayoutNode [(2,)] JaggedArray>

    t[2].array
    # <JaggedArray [[1.1 2.2] [3.3] [4.4 5.5]] at 0x7811060a1208>

    t[2, 2].array
    # array([1.1, 2.2, 3.3, 4.4, 5.5])

Classes like ``IndexedArray``, ``SparseArray``, ``ChunkedArray``, ``AppendableArray``, and ``VirtualArray`` don't change the high-level type of an array, but they do change the layout. Consider, for instance, an array made with ``awkward.fromiter`` and an array read by ``awkward.fromparquet``.

.. code-block:: python3

    a = awkward.fromiter([[1.1, 2.2, None, 3.3], [], None, [4.4, 5.5]])

    awkward.toparquet("tmp.parquet", a)

    b = awkward.fromparquet("tmp.parquet")

At first, it terminates at ``VirtualArray`` because the data haven't been read—we don't know what arrays are associated with it.

.. code-block:: python3

    b.layout
    #  layout
    # [    ()] ChunkedArray(chunks=[layout[0]], chunksizes=[4])
    # [     0]   VirtualArray(generator=<awkward.arrow._ParquetFile object at 0x781106089668>, args=(0, ''), kwargs={})

But after reading,

.. code-block:: python3

    b
    # <ChunkedArray [[1.1 2.2 None 3.3] [] [] [4.4 5.5]] at 0x7811060890b8>

The layout shows that it has more structure than ``a``.

.. code-block:: python3

    b.layout
    #  layout
    # [           ()] ChunkedArray(chunks=[layout[0]], chunksizes=[4])
    # [            0]   VirtualArray(generator=<awkward.arrow._ParquetFile object at 0x781106089668>, args=(0, ''), kwargs={}, array=layout[0, 0])
    # [         0, 0]     BitMaskedArray(mask=layout[0, 0, 0], content=layout[0, 0, 1], maskedwhen=False, lsborder=True)
    # [      0, 0, 0]       ndarray(shape=1, dtype=dtype('uint8'))
    # [      0, 0, 1]       JaggedArray(starts=layout[0, 0, 1, 0], stops=layout[0, 0, 1, 1], content=layout[0, 0, 1, 2])
    # [   0, 0, 1, 0]         ndarray(shape=4, dtype=dtype('int32'))
    # [   0, 0, 1, 1]         ndarray(shape=4, dtype=dtype('int32'))
    # [   0, 0, 1, 2]         BitMaskedArray(mask=layout[0, 0, 1, 2, 0], content=layout[0, 0, 1, 2, 1], maskedwhen=False, lsborder=True)
    # [0, 0, 1, 2, 0]           ndarray(shape=1, dtype=dtype('uint8'))
    # [0, 0, 1, 2, 1]           ndarray(shape=6, dtype=dtype('float64'))

.. code-block:: python3

    a.layout
    #  layout
    # [     ()] MaskedArray(mask=layout[0], content=layout[1], maskedwhen=True)
    # [      0]   ndarray(shape=4, dtype=dtype('bool'))
    # [      1]   JaggedArray(starts=layout[1, 0], stops=layout[1, 1], content=layout[1, 2])
    # [   1, 0]     ndarray(shape=4, dtype=dtype('int64'))
    # [   1, 1]     ndarray(shape=4, dtype=dtype('int64'))
    # [   1, 2]     MaskedArray(mask=layout[1, 2, 0], content=layout[1, 2, 1], maskedwhen=True)
    # [1, 2, 0]       ndarray(shape=6, dtype=dtype('bool'))
    # [1, 2, 1]       ndarray(shape=6, dtype=dtype('float64'))

However, they have the same high-level type.

.. code-block:: python3

    print(b.type)
    # [0, 4) -> ?([0, inf) -> ?(float64))

    print(a.type)
    # [0, 4) -> ?([0, inf) -> ?(float64))

Cross-references and cyclic references are also encoded in the ``layout``, as references to previously seen indexes.

.. code-block:: python3

    tree.layout
    #  layout
    # [     ()] Table(left=layout[0], right=layout[1], value=layout[2])
    # [      0]   MaskedArray(mask=layout[0, 0], content=layout[0, 1], maskedwhen=True)
    # [   0, 0]     ndarray(shape=9, dtype=dtype('bool'))
    # [   0, 1]     IndexedArray(index=layout[0, 1, 0], content=layout[0, 1, 1])
    # [0, 1, 0]       ndarray(shape=9, dtype=dtype('int64'))
    # [0, 1, 1]       -> layout[()]
    # [      1]   MaskedArray(mask=layout[1, 0], content=layout[1, 1], maskedwhen=True)
    # [   1, 0]     ndarray(shape=9, dtype=dtype('bool'))
    # [   1, 1]     IndexedArray(index=layout[1, 1, 0], content=layout[1, 1, 1])
    # [1, 1, 0]       ndarray(shape=9, dtype=dtype('int64'))
    # [1, 1, 1]       -> layout[()]
    # [      2]   ndarray(shape=9, dtype=dtype('float64'))

Acknowledgements
================

Support for this work was provided by NSF cooperative agreement OAC-1836650 (IRIS-HEP), grant OAC-1450377 (DIANA/HEP) and PHY-1520942 (US-CMS LHC Ops).

Thanks especially to the gracious help of `awkward-array contributors <https://github.com/scikit-hep/awkward-array/graphs/contributors>`__!
