# %%markdown
# # Introduction
#
# Numpy is great for exploratory data analysis because it encourages the analyst to calculate one operation at a time, rather than one datum at a time. To compute an expression like
#
# .. math::
#
#     m = \\sqrt{(E_1 + E_2)^2 - (p_{x1} + p_{x2})^2 - (p_{y1} + p_{y2})^2 - (p_{z1} + p_{z2})^2}
#
# the analyst might first compute :math:`\\sqrt{(p_{x1} + p_{x2})^2 + (p_{y1} + p_{y2})^2}` for all data (which has a meaning: :math:`p_T`), then compute :math:`\\sqrt{{p_T}^2 + (p_{z1} + p_{z2})^2}` for all data (which has a meaning: :math:`|p|`), then compute the whole expression as :math:`\\sqrt{(E_1 + E_2)^2 - |p|^2}`. Performing each step separately on all data allows the analyzer to plot and cross-check distributions of partial computations, to discover surprises as early as possible.
#
# This order of data processing is called "columnar" in the sense that a dataset may be visualized as a table in which rows are repeated measurements and columns are the different measurable quantities (same layout as `Pandas DataFrames <https://pandas.pydata.org>`__). It is also called "vectorized" in that a Single (virtual) Instruction is applied to Multiple Data (virtual SIMD). Numpy can be hundreds to thousands of times faster than pure Python because it avoids the overhead of handling Python instructions in the loop over numbers. Most data processing languages (R, MATLAB, IDL, all the way back to APL) work this way: an interactive interpreter with fast, vectorized math.
#
# However, it's difficult to apply this to non-rectangular data. If your dataset has nested structure, a different number of values per row, different data types in the same column, or cross-references or even circular references, Numpy can't help you.
#
# If you try to make an array with non-trivial types:

# %%
import numpy
a = numpy.array([[1.1, 2.2, "three"], [], ["four", [5]]])
a

# %%markdown
# Numpy gives up and returns a ``dtype=object`` array, which means Python objects and pure Python processing. Neither the columnar operations nor the performance boost apply.
#
# That's what **awkward-array** is for: it generalizes Numpy's array language to complex data types. It's for analyzing data that are not only structured in the sense of having correlations among their values, but also in the sense of having non-trivial data structures. It's for arrays that are just, well, awkward.

# %%
import awkward
a = awkward.fromiter([[1.1, 2.2, "three"], [], ["four", [5]]])

# %%markdown
# Instead of relying on Python, ``awkward.fromiter`` converts these data structures into an internally columnar one, using Numpy for each column. The resulting class is called a "jagged" (or "ragged") array:

# %%
a

# %%markdown
# The structure (3 items in the first list, 0 in the second, and 2 in the third) is stored in one array and the data in another.

# %%
a.counts, a.content

# %%markdown
# The heterogeneity (different data types) of the content is encoded in additional arrays, one for each type (the third of which is another jagged array).

# %%
a.content.contents

# %%markdown
# This decomposition is variously called "splitting" (by particle physicists) or "shredding" or "striping" (by  `Arrow <https://arrow.apache.org>`__ and `Parquet <https://parquet.apache.org>`__ developers).
#
# Most importantly, you can do Numpy vectorized operations,

# %%
a = awkward.fromiter([[1, 4, None], [], [9, {"x": 16, "y": 25}]])
numpy.sqrt(a).tolist()   # numbers at all levels of depth are square-rooted

# %%markdown
# multidimensional slicing,

# %%
a = awkward.fromiter([[1], [], [2, 3], [4, 5, [6]]])
a[2:, 1:]   # drop the first two outer lists and the first element of every remaining list

# %%markdown
# and broadcasting,

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
a * 10   # multiply all numbers by the scalar 10

# %%markdown
# including jagged-only extensions of these concepts,

# %%
a + numpy.array([100, 200, 300])   # add 100 to the whole first list and 300 to the last

# %%
a.sum()   # reduce (by summation) to get a scalar per inner list

# %%markdown
# This tutorial starts with a data analyst's perspective—using awkward-array to manipulate data—and then focuses on each awkward-array type. Like Numpy, the features of this library are deliberately simple, yet compositional. Any awkward array may be the content of any other awkward array. Building and analyzing complex data structures is up to you.

# %%markdown
# # Getting data and initial exploration
#
# A lot of the examples in this tutorial use ``awkward.fromiter`` to make awkward arrays from lists and ``array.tolist()`` to turn them back into lists (or dicts for structures, tuples for structures with anonymous fields, Python objects for ``ObjectArrays``, etc.). This should be thought of as a slow method, since Python instructions are executed in the loop, but that's the only way to convert to and from Python objects. I only use it for small datasets, though if you have JSON-formatted data, ``awkward.fromiter`` may be a necessary preprocessing step.
#
# Ideally, data should be provided in a `columnar format <https://towardsdatascience.com/the-beauty-of-column-oriented-data-2945c0c9f560>`__ or converted only once. `Parquet <https://parquet.apache.org>`__ is a popular columnar format for storing data on disk and `Arrow <https://arrow.apache.org>`__ is a popular columnar format for sharing data in memory (between functions or applications). `ROOT <https://root.cern>`__ is a popular columnar format for particle physicists, and `uproot <https://github.com/scikit-hep/uproot>`__ natively produces awkward arrays from ROOT files. `HDF5 <https://www.hdfgroup.org>`__ and its Python library `h5py <https://www.h5py.org/>`__ are only columnar for rectangular arrays. All of the others at least do jagged arrays.

# %%markdown
# ## Parquet files
#
# If you have a Parquet file, open it with

# %%
stars = awkward.fromparquet("tests/samples/exoplanets.parquet")
stars

# %%markdown
# Columns are accessible with square brackets and strings

# %%
stars["name"]

# %%markdown
# or by attribute (if it doesn't have weird characters and doesn't conflict with a method or property name)

# %%
stars.ra, stars.dec

# %%markdown
# In this dataset, each star has one or more planets (the second-to-last has three planets). Thus, the planet attributes are jagged arrays.

# %%
stars.planet_name

# %%
stars.planet_period, stars.planet_orbit

# %%markdown
# These arrays are called ``ChunkedArrays`` because the Parquet file is lazily read in chunks (Parquet's row group structure). They support the same Numpy features as vectorized computation

# %%
# distance in parsecs → distance in light years
stars.dist * 3.26156

# %%markdown
# and multidimensional slicing.

# %%
stars.planet_mass[:, 1:]

# %%markdown
# ## Arrow buffers
#
# Parquet and Arrow go through the `pyarrow <https://arrow.apache.org/docs/python>`__ library, which has support for more data structures in Arrow than Parquet.
#
# But whereas it's obvious that Parquet will come from a file on disk, Arrow buffers may come from anywhere: the output of another function, interprocess communication, a network call, or even a file.

# %%
import pyarrow
arrow_buffer = pyarrow.ipc.open_file(open("tests/samples/exoplanets.arrow", "rb")).get_batch(0)
stars = awkward.fromarrow(arrow_buffer)
stars

# %%markdown
# Since Arrow has more features, this file has been packed more intuitively: planets are objects within a jagged array.

# %%
stars["planets"]

# %%markdown
# When viewed as Python lists and dicts, the ``'planets'`` field is a list of planet dicts, each with its own fields.

# %%
stars[:2].tolist()

# %%markdown
# But the cross-cutting view of each planet attribute we had earlier is still immediately accessible: selecting a column of a table inside a jagged array produces a jagged array of that attribute.

# %%
stars.planets.name

# %%
stars.planets.mass

# %%markdown
# The ``stars`` and ``stars.planets`` are both tables, though ``stars.planets`` is a jagged table (a ``Table`` inside a ``JaggedArray``). This structure keeps their columns separate.

# %%
stars.columns

# %%
stars.planets.columns

# %%markdown
# For a more global view of the structures contained within one of these arrays, print out its type.

# %%
print(stars.type)

# %%markdown
# The above should be read like a function's data type, where the function in question is indexing with square brackets. The ``[0, 2935)`` means that you could put any non-negative integer less than ``2935`` in square brackets after ``stars``.

# %%
stars[1734]

# %%markdown
# The ``'dec'``, ``'dist'``, ``'mass'``, ``'name'``, ``'planets'``, ``'ra'``, and ``'radius'`` at the next level indicate that the next square bracket can contain one of those strings (or, as stated above, use the dot notation).

# %%
stars[1734]["mass"]   # type is float64

# %%
stars[1734]["name"]   # type is <class 'str'>

# %%
stars[1734]["planets"]

# %%markdown
# The ``'planets'`` field can take another index whose type is ``[0, inf)``, any non-negative integer. That is, it's a jagged array—every subarray can have a different length and those lengths would be too numerous to specify. All of them have finite upper limits, and requesting an index out of range would raise an error.

# %%
stars[1734]["planets"][4]

# %%markdown
# Within this jagged array, there are six string fields: ``'eccen'``, ``'mass'``, ``'name'``, ``'orbit'``, ``'period'``, and ``'radius'``. Passing any one of these strings in square brackets returns a ``float64`` or a ``<class 'str'>``.

# %%
stars[1734]["planets"][4]["period"]

# %%
stars[1734]["planets"][4]["name"]

# %%
stars[1734]["planets"][4].tolist()

# %%markdown
# Incidentally, this was the `first potentially habitable exoplanet <https://www.nasa.gov/ames/kepler/kepler-186f-the-first-earth-size-planet-in-the-habitable-zone>`__` discovered.

# %%
stars[1734]["name"], stars[1734]["planets"][4]["name"]

# %%markdown
# Another way to describe the cross-cutting view of an attribute as a jagged array is to say that

# %%
stars["planets"]["name"][1734][4]

# %%
stars[1734]["planets"][4]["name"]
