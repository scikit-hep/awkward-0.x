# %%markdown
# # Introduction
#
# Numpy is great for exploratory data analysis because it encourages the analyst to calculate one operation at a time, rather than one datum at a time. To compute an expression like
#
# .. math::
#
#     m = \\sqrt{(E_1 + E_2)^2 - (p_{x1} + p_{x2})^2 - (p_{y1} + p_{y2})^2 - (p_{z1} + p_{z2})^2}
#
# you might first compute :math:`\\sqrt{(p_{x1} + p_{x2})^2 + (p_{y1} + p_{y2})^2}` for all data (which is a meaningful quantity: :math:`p_T`), then compute :math:`\\sqrt{{p_T}^2 + (p_{z1} + p_{z2})^2}` for all data (another meaningful quantity: :math:`|p|`), then compute the whole expression as :math:`\\sqrt{(E_1 + E_2)^2 - |p|^2}`. Performing each step separately on all data lets you plot and cross-check distributions of partial computations, to discover surprises as early as possible.
#
# This order of data processing is called "columnar" in the sense that a dataset may be visualized as a table in which rows are repeated measurements and columns are the different measurable quantities (same layout as `Pandas DataFrames <https://pandas.pydata.org>`__). It is also called "vectorized" in that a Single (virtual) Instruction is applied to Multiple Data (virtual SIMD). Numpy can be hundreds to thousands of times faster than pure Python because it avoids the overhead of handling Python instructions in the loop over numbers. Most data processing languages (R, MATLAB, IDL, all the way back to APL) work this way: an interactive interpreter controlling fast, array-at-a-time math.
#
# However, it's difficult to apply this methodology to non-rectangular data. If your dataset has nested structure, a different number of values per row, different data types in the same column, or cross-references or even circular references, Numpy can't help you.
#
# If you try to make an array with non-trivial types:

# %%
import numpy
nested = numpy.array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}])
nested

# %%markdown
# Numpy gives up and returns a ``dtype=object`` array, which means Python objects and pure Python processing. You don't get the columnar operations or the performance boost.
#
# For instance, you might want to say

# %%
try:
    nested + 100
except Exception as err:
    print(type(err), str(err))

# %%markdown
# but there is no vectorized addition for an array of dicts because there is no addition for dicts defined in pure Python. Numpy is not using its vectorized routines—it's calling Python code on each element.
#
# The same applies to variable-length data, such as lists of lists, where the inner lists have different lengths. This is a more serious shortcoming than the above because the list of dicts (Python's equivalent of an "`array of structs <https://en.wikipedia.org/wiki/AOS_and_SOA>`__`") could be manually reorganized into two numerical arrays, ``"x"`` and ``"y"`` (a "`struct of arrays <https://en.wikipedia.org/wiki/AOS_and_SOA>`__"). Not so with a list of variable-length lists.

# %%
varlen = numpy.array([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]])
varlen

# %%markdown
# As before, we get a ``dtype=object`` without vectorized methods.

# %%
try:
    varlen + 100
except Exception as err:
    print(type(err), str(err))

# %%markdown
# What's worse, this array looks purely numerical and could have been made by a process that was *supposed* to create equal-length inner lists.
#
# Awkward-array provides a way of talking about these data structures as arrays.

# %%
import awkward
nested = awkward.fromiter([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}, {"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}])
nested

# %%markdown
# This ``Table`` is a columnar data structure with the same meaning as the Python data we built it with. To undo ``awkward.fromiter``, call ``.tolist()``.

# %%
nested.tolist()

# %%markdown
# Values at the same position of the tree structure are contiguous in memory: this is a struct of arrays.

# %%
nested.contents["x"]

# %%
nested.contents["y"]

# %%markdown
# Having a structure like this means that we can perform vectorized operations on the whole structure with relatively few Python instructions (number of Python instructions scales with the complexity of the data type, not with the number of values in the dataset).

# %%
(nested + 100).tolist()

# %%
(nested + numpy.arange(100, 600, 100)).tolist()

# %%markdown
# It's less obvious that variable-length data can be represented in a columnar format, but it can.

# %%
varlen = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]])
varlen

# %%markdown
# Unlike Numpy's ``dtype=object`` array, the inner lists are *not* Python lists and the numerical values *are* contiguous in memory. This is made possible by representing the structure (where each inner list starts and stops) in one array and the values in another.

# %%
varlen.counts, varlen.content

# %%markdown
# (For fast random access, the more basic representation is ``varlen.offsets``, which is in turn a special case of a ``varlen.starts, varlen.stops`` pair. These details are discussed below.)
#
# A structure like this can be broadcast like Numpy with a small number of Python instructions (scales with the complexity of the data type, not the number of values).

# %%
varlen + 100

# %%
varlen + numpy.arange(100, 600, 100)

# %%markdown
# You can even slice this object as though it were multidimensional (each element is a tensor of the same rank, but with different numbers of dimensions).

# %%
# Skip the first two inner lists; skip the last value in each inner list that remains.
varlen[2:, :-1]

# %%markdown
# The data are not rectangular, so some inner lists might have as many elements as your selection. Don't worry—you'll get error messages.

# %%
try:
    varlen[:, 1]
except Exception as err:
    print(type(err), str(err))

# %%markdown
# Masking with the ``.counts`` is handy because all the Numpy advanced indexing rules apply (in an extended sense) to jagged arrays.

# %%
varlen[varlen.counts > 1, 1]

# %%markdown
# I've only presented the two most important awkward classes, ``Table`` and ``JaggedArray`` (and not how they combine). Each class is presented in more detail below. For now, I'd just like to point out that you can make crazy complicated data structures

# %%
crazy = awkward.fromiter([[1.21, 4.84, None, 10.89, None],
                          [19.36, [30.25]],
                          [{"x": 36, "y": {"z": 49}}, None, {"x": 64, "y": {"z": 81}}]
                         ])

# %%markdown
# and they vectorize and slice as expected.

# %%
numpy.sqrt(crazy).tolist()

# %%markdown
# This is because any awkward array can be the content of any other awkward array. Like Numpy, the features of awkward-array are simple, yet compose nicely to let you build what you need.

# %%markdown
# # Overview with sample datasets
#
# Many of the examples in this tutorial use ``awkward.fromiter`` to make awkward arrays from lists and ``array.tolist()`` to turn them back into lists (or dicts for ``Table``, tuples for ``Table`` with anonymous fields, Python objects for ``ObjectArrays``, etc.). These should be considered slow methods, since Python instructions are executed in the loop, but that's a necessary part of examining or building Python objects.
#
# Ideally, you'd want to get your data from a binary, columnar source and produce binary, columnar output, or convert only once and reuse the converted data. `Parquet <https://parquet.apache.org>`__ is a popular columnar format for storing data on disk and `Arrow <https://arrow.apache.org>`__ is a popular columnar format for sharing data in memory (between functions or applications). `ROOT <https://root.cern>`__ is a popular columnar format for particle physicists, and `uproot <https://github.com/scikit-hep/uproot>`__ natively produces awkward arrays from ROOT files.
#
# `HDF5 <https://www.hdfgroup.org>`__ and its Python library `h5py <https://www.h5py.org/>`__ are columnar, but only for rectangular arrays, unlike the others mentioned here. Awkward-array can *wrap* HDF5 with an interpretation layer to store columnar data structures, but then the awkward-array library wuold be needed to read the data back in a meaningful way. Awkward also has a native file format, ``.awkd`` files, which are simply ZIP archives of columns as binary blobs and metadata (just as Numpy's ``.npz`` is a ZIP of arrays with metadata). The HDF5, awkd, and pickle serialization procedures use the same protocol, which has backward and forward compatibility features.

# %%markdown
# ## NASA exoplanets from a Parquet file
#
# Let's start by opening a Parquet file. Awkward reads Parquet through the `pyarrow <https://arrow.apache.org/docs/python>`__ module, which is an optional dependency, so be sure you have it installed before trying the next line.

# %%
stars = awkward.fromparquet("tests/samples/exoplanets.parquet")
stars

# %%markdown
# (There is also an ``awkward.toparquet`` that takes the file name and array as arguments.)
#
# Columns are accessible with square brackets and strings

# %%
stars["name"]

# %%markdown
# or by dot-attribute (if the name doesn't have weird characters and doesn't conflict with a method or property name).

# %%
stars.ra, stars.dec

# %%markdown
# This file contains data about extrasolar planets and their host stars. As such, it's a ``Table`` full of Numpy arrays and ``JaggedArrays``. The star attributes (`"name"`, `"ra"` or right ascension in degrees, `"dec"` or declination in degrees, `"dist"` or distance in parsecs, `"mass"` in multiples of the sun's mass, and `"radius"` in multiples of the sun's radius) are plain Numpy arrays and the planet attributes (`"name"`, `"orbit"` or orbital distance in AU, `"eccen"` or eccentricity, `"period"` or periodicity in days, `"mass"` in multiples of Jupyter's mass, and `"radius"` in multiples of Jupiter's radius) are jagged because each star may have a different number of planets.

# %%
stars.planet_name

# %%
stars.planet_period, stars.planet_orbit

# %%markdown
# For large arrays, only the first and last values are printed: the second-to-last star has three planets; all the other stars shown here have one planet.
#
# These arrays are called ``ChunkedArrays`` because the Parquet file is lazily read in chunks (Parquet's row group structure). The ``ChunkedArray`` (subdivides the file) contains ``VirtualArrays`` (read one chunk on demand), which generate the ``JaggedArrays``. This is an illustration of how each awkward class provides one feature, and you get desired behavior by combining them.
#
# The ``ChunkedArrays`` and ``VirtualArrays`` support the same Numpy-like access as ``JaggedArray``, so we can compute with them just as we would any other array.

# %%
# distance in parsecs → distance in light years
stars.dist * 3.26156

# %%
# for all stars, drop the first planet
stars.planet_mass[:, 1:]

# %%markdown
# ## NASA exoplanets from an Arrow buffer
#
# The pyarrow implementation of Arrow is more complete than its implementation of Parquet, so we can use more features in the Arrow format, such as nested tables.
#
# Unlike Parquet, which is intended as a file format, Arrow is a memory format. You might get an Arrow buffer as the output of another function, through interprocess communication, from a network RPC call, a message bus, etc. Arrow can be saved as files, though this isn't common. In this case, we'll get it from a file.

# %%
import pyarrow
arrow_buffer = pyarrow.ipc.open_file(open("tests/samples/exoplanets.arrow", "rb")).get_batch(0)
stars = awkward.fromarrow(arrow_buffer)
stars

# %%markdown
# (There is also an ``uproot.toarrow`` that takes an awkward array as its only argument, returning the relevant Arrow structure.)
#
# This file is structured differently. Instead of jagged arrays of numbers like ``"planet_mass"``, ``"planet_period"``, and ``"planet_orbit"``, this file has a jagged table of ``"planets"``. A jagged table is a ``JaggedArray`` of ``Table``.

# %%
stars["planets"]

# %%markdown
# Notice that the square brackets are nested, but the contents are ``<Row>`` objects. The second-to-last star has three planets, as before.
#
# We can find the non-jagged ``Table`` in the ``JaggedArray.content``.

# %%
stars["planets"].content

# %%markdown
# When viewed as Python lists and dicts, the ``'planets'`` field is a list of planet dicts, each with its own fields.

# %%
stars[:2].tolist()

# %%markdown
# Despite being packaged in an arguably more intuitive way, we can still get jagged arrays of numbers by requesting ``"planets"`` and a planet attribute (two column selections) without specifying which star or which parent.

# %%
stars.planets.name

# %%
stars.planets.mass

# %%markdown
# Even though the ``Table`` is hidden inside the ``JaggedArray``, its ``columns`` pass through to the top.

# %%
stars.columns

# %%
stars.planets.columns

# %%markdown
# For a more global view of the structures contained within one of these arrays, print out its high-level type. ("High-level" because it presents logical distinctions, like jaggedness and tables, but not physical distinctions, like chunking and virtualness.)

# %%
print(stars.type)

# %%markdown
# The above should be read like a function's data type: ``argument type -> return type`` for the function that takes an index in square brackets and returns something else. For example, the first ``[0, 2935)`` means that you could put any non-negative integer less than ``2935`` in square brackets after ``stars``, like this:

# %%
stars[1734]

# %%markdown
# and get an object that would take ``'dec'``, ``'dist'``, ``'mass'``, ``'name'``, ``'planets'``, ``'ra'``, or ``'radius'`` in its square brackets. The return type depends on which of those strings you provide.

# %%
stars[1734]["mass"]   # type is float64

# %%
stars[1734]["name"]   # type is <class 'str'>

# %%
stars[1734]["planets"]

# %%markdown
# The planets have their own table structure:

# %%
print(stars[1734]["planets"].type)

# %%markdown
# Notice that within the context of ``stars``, the ``planets`` could take any non-negative integer ``[0, inf)``, but for a particular star, the allowed domain is known with more precision: ``[0, 5)``. This is because ``stars["planets"]`` is a jagged array—a different number of planets for each star—but one ``stars[1734]["planets"]`` is a simple array—five planets for *this* star.
#
# Passing a non-negative integer less than 5 to this array, we get an object that takes one of six strings: : ``'eccen'``, ``'mass'``, ``'name'``, ``'orbit'``, ``'period'``, and ``'radius'``.

# %%
stars[1734]["planets"][4]

# %%markdown
# and the return type of these depends on which string you provide.

# %%
stars[1734]["planets"][4]["period"]   # type is float

# %%
stars[1734]["planets"][4]["name"]   # type is <class 'str'>

# %%
stars[1734]["planets"][4].tolist()

# %%markdown
# (Incidentally, this is a `potentially habitable exoplanet <https://www.nasa.gov/ames/kepler/kepler-186f-the-first-earth-size-planet-in-the-habitable-zone>`__`, the first ever discovered.)

# %%
stars[1734]["name"], stars[1734]["planets"][4]["name"]

# %%markdown
# Some of these arguments "commute" and others don't. Dimensional axes have a particular order, so you can't request a planet by its row number before selecting a star, but you can swap a column-selection (string) and a row-selection (integer). For a rectangular table, it's easy to see how you can slice column-first or row-first, but it even works when the table is jagged.

# %%
stars["planets"]["name"][1734][4]

# %%
stars[1734]["planets"][4]["name"]

# %%markdown
# None of these intermediate slices actually process data, so you can slice in any order that is logically correct without worrying about performance. Projections, even multi-column projections

# %%
orbits = stars["planets"][["name", "eccen", "orbit", "period"]]
orbits[1734].tolist()

# %%markdown
# are a useful way to restructure data without incurring a runtime cost.

# %%markdown
# ## Relationship to Pandas
#
# Arguably, this kind of dataset could be manipulated as a `Pandas DataFrame <https://pandas.pydata.org>`__ instead of awkward arrays. Despite the variable number of planets per star, the exoplanets dataset could be flattened into a rectangular DataFrame, in which the distinction between solar systems is represented by a two-component index (leftmost pair of columns below), a `MultiIndex <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html>`__.

# %%
awkward.topandas(stars, flatten=True)[-9:]

# %%markdown
# In this representation, each star's attributes must be duplicated for all of its planets, and it is not possible to show stars that have no planets (not present in this dataset), but the information is preserved in a way that Pandas can recognize and operate on. (For instance, ``.unstack()`` would widen each planet attribute into a separate column per planet and simplify the index to strictly one row per star.)
#
# The limitation is that only a single jagged structure can be represented by a DataFrame. The structure can be arbitrarily deep in ``Tables`` (which add depth to the column names),

# %%
array = awkward.fromiter([{"a": {"b": 1, "c": {"d": [2]}}, "e": 3},
                          {"a": {"b": 4, "c": {"d": [5, 5.1]}}, "e": 6},
                          {"a": {"b": 7, "c": {"d": [8, 8.1, 8.2]}}, "e": 9}])
awkward.topandas(array, flatten=True)

# %%markdown
# and arbitrarily deep in ``JaggedArrays`` (which add depth to the row names),

# %%
array = awkward.fromiter([{"a": 1, "b": [[2.2, 3.3, 4.4], [], [5.5, 6.6]]},
                          {"a": 10, "b": [[1.1], [2.2, 3.3], [], [4.4]]},
                          {"a": 100, "b": [[], [9.9]]}])
awkward.topandas(array, flatten=True)

# %%markdown
# and they can even have two ``JaggedArrays`` at the same level if their number of elements is the same (at all levels of depth).

# %%
array = awkward.fromiter([{"a": [[1.1, 2.2, 3.3], [], [4.4, 5.5]], "b": [[1, 2, 3], [], [4, 5]]},
                          {"a": [[1.1], [2.2, 3.3], [], [4.4]],    "b": [[1], [2, 3], [], [4]]},
                          {"a": [[], [9.9]],                       "b": [[], [9]]}])
awkward.topandas(array, flatten=True)

# %%markdown
# But if there are two ``JaggedArrays`` with *different* structure at the same level, a single DataFrame cannot represent them.

# %%
array = awkward.fromiter([{"a": [1, 2, 3], "b": [1.1, 2.2]},
                          {"a": [1],       "b": [1.1, 2.2, 3.3]},
                          {"a": [1, 2],    "b": []}])
try:
    awkward.topandas(array, flatten=True)
except Exception as err:
    print(type(err), str(err))

# %%markdown
# To describe data like these, you'd need two DataFrames, and any calculations involving both ``"a"`` and ``"b"`` would have to include a join on those DataFrames. Awkward arrays are not limited in this way: the last ``array`` above is a valid awkward array and is useful for calculations that mix ``"a"`` and ``"b"``.

# %%markdown
# ## LHC data from a ROOT file
#
# Particle physicsts need structures like these—in fact, they have been a staple of particle physics analyses for decades. The `ROOT <https://root.cern>`__ file format was developed in the mid-90's to serialize arbitrary C++ data structures in a columnar way (replacing ZEBRA and similar Fortran projects that date back to the 70's). The `PyROOT <https://root.cern.ch/pyroot>`__ library dynamically wraps these objects to present them in Python, though with a performance penalty. The `uproot <https://github.com/scikit-hep/uproot>`__ library reads columnar data directly from ROOT files in Python without intermediary C++.

# %%
import uproot
events = uproot.open("http://scikit-hep.org/uproot/examples/HZZ-objects.root")["events"].lazyarrays()
events

# %%
events.columns

# %%markdown
# This is a typical particle physics dataset (though small!) in that it represents the momentum and energy (``"p4"`` for `Lorentz 4-momentum <https://en.wikipedia.org/wiki/Four-vector`__) of several different species of particles: ``"jet"``, ``"muon"``, ``"electron"``, and ``"photon"``. Each collision can produce a different number of particles in each species. Other variables, such as missing transverse energy or ``"MET"``, have one value per collision event. Events with zero particles in a species are valuable for the event-level data.

# %%
# The first event has two muons.
events.muonp4

# %%
# The first event has zero jets.
events.jetp4

# %%
# Every event has exactly one MET.
events.MET

# %%markdown
# Unlike the exoplanet data, these events cannot be represented as a DataFrame because of the different numbers of particles in each species and because zero-particle events have value. Even with just ``"muonp4"``, ``"jetp4"``, and ``"MET"``, there is no translation.

# %%
try:
    awkward.topandas(events[["muonp4", "jetp4", "MET"]], flatten=True)
except Exception as err:
    print(type(err), str(err))

# %%markdown
# It could be described as a collection of DataFrames, in which every operation relating particles in the same event would require a join. But that would make analysis harder, not easier. An event has meaning on its own.

# %%
events[0].tolist()

# %%markdown
# Particle physics isn't alone in this: analyzing JSON-formatted log files in production systems or allele likelihoods in genomics are two other fields where variable-length, nested structures can help. Arbitrary data structures are useful and working with them in columns provides a new way to do exploratory data analysis: one array at a time.

# %%markdown
# # Awkward-array data model
#
# Awkward array features are provided by a suite of classes that each extend Numpy arrays in one small way. These classes may then be composed to combine features.
#
# In this sense, Numpy arrays are awkward-array's most basic array class. A Numpy array is a small Python object that points to a large, contiguous region of memory, and, as much as possible, operations replace or change the small Python object, not the big data buffer. Therefore, many Numpy operations are *views*, rather than *in-place operations* or *copies*, leaving the original value intact but returning a new value that is linked to the original. Assigning to arrays and in-place operations are allowed, but they are more complicated to use because one must be aware of which arrays are views and which are copies.
#
# Awkward-array's model is to treat all arrays as though they were immutable, favoring views over copies, and not providing any high-level in-place operations on low-level memory buffers (i.e. no in-place assignment).
#
# Numpy provides complete control over the interpretation of an ``N`` dimensional array. A Numpy array has a `dtype <https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html>`__ to interpret bytes as signed and unsigned integers of various bit-widths, floating-point numbers, booleans, little endian and big endian, fixed-width bytestrings (for applications such as 6-byte MAC addresses or human-readable strings with padding), or `record arrays <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`__ for contiguous structures. A Numpy array has a `pointer <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.ctypes.html>`__ to the first element of its data buffer (``array.ctypes.data``) and a `shape <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html>`__ to describe its ``N`` dimensions as a rank-``N`` tensor. Only ``shape[0]`` is the length as returned by the Python function ``len``. Furthermore, an `order <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flags.html>`__ flag determines if rank > 1 arrays are laid out in "C" order or "Fortran" order. A Numpy array also has a `stride <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.strides.html>`__` to determine how many bytes separate one element from the next. (Data in a Numpy array need not be strictly contiguous, but they must be regular: the number of bytes seprating them is a constant.) This stride may even be negative to describe a reversed view of an array, which allows any ``slice`` of an array, even those with ``skip != 1`` to be a view, rather than a copy. Numpy arrays also have flags to determine whether they `own <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flags.html>`__` their data buffer (and should therefore delete it when the Python object goes out of scope) and whether the data buffer is `writable <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flags.html>`__.

# %%markdown
#
# The biggest restriction on this data model is that Numpy arrays are strictly rectangular. The ``shape`` and ``stride`` are constants, enforcing a regular layout. Awkward's ``JaggedArray`` is a generalization of Numpy's rank-2 arrays—that is, arrays of arrays—in that the inner arrays of a ``JaggedArray`` may all have different lengths. For higher ranks, such as arrays of arrays of arrays, put a ``JaggedArray`` inside another as its ``content``. An important special case of ``JaggedArray`` is ``StringArray``, whose ``content`` is interpreted as characters (with or without encoding), which represents an array of strings without unnecessary padding, as in Numpy's case.
#
# Although Numpy's `record arrays <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`__ present a buffer as a table, with differently typed, named columns, that table must be contiguous or interleaved (with non-trivial ``strides``) in memory: an `array of structs <https://en.wikipedia.org/wiki/AOS_and_SOA>`__. Awkward's ``Table`` provides the same interface, except that each column may be anywhere in memory, stored in a ``contents`` dict mapping field names to arrays. This is a true generalization: a ``Table`` may be a wrapped view of a Numpy record array, but not vice-versa. Use a ``Table`` anywhere you'd have a record/class/struct in non-columnar data structures. A ``Table`` with anonymous (integer-valued, rather than string-valued) fields is like an array of strongly typed tuples.
#
# Numpy has a `masked array <https://docs.scipy.org/doc/numpy/reference/maskedarray.html>`__ module for nullable data—values that may be "missing" (like Python's ``None``). Naturally, the only kinds of arrays Numpy can mask are subclasses of its own ``ndarray``, and we need to be able to mask any awkward array, so the awkward library defines its own ``MaskedArray``. Additionally, we sometimes want to mask with bits, rather than bytes (e.g. for Arrow compatibility), so there's a ``BitMaskedArray``, and sometimes we want to mask large structures without using memory for the masked-out values, so there's an ``IndexedMaskedArray`` (fusing the functionality of a ``MaskedArray`` with an ``IndexedArray``).
#
# Numpy has no provision for an array containing different data types ("heterogeneous"), but awkward-array has a ``UnionArray``. The ``UnionArray`` stores data for each type as separate ``contents`` and identifies the types and positions of each element in the ``contents`` using ``tags`` and ``index`` arrays (equivalent to Arrow's `dense union type <https://arrow.apache.org/docs/memory_layout.html#dense-union-type>`__ with ``types`` and ``offsets`` buffers). As a data type, unions are a counterpart to records or tuples (making ``UnionArray`` a counterpart to ``Table``): each record/tuple contains *all* of its ``contents`` but a union contains *any* of its ``contents``. (Note that a ``UnionArray`` may be the best way to interleave two arrays, even if they have the same type. Heterogeneity is not a necessary feature of a ``UnionArray``.)
#
# Numpy has a ``dtype=object`` for arrays of Python objects, but awkward's ``ObjectArray`` creates Python objects on demand from array data. A large dataset of some ``Point`` class, containing floating-point members ``x`` and ``y``, can be stored as an ``ObjectArray`` of a ``Table`` of ``x`` and ``y`` with much less memory than a Numpy array of ``Point`` objects. The ``ObjectArray`` has a ``generator`` function that produces Python objects from array elements.  ``StringArray`` is also a special case of ``ObjectArray``, which instantiates variable-length character contents as Python strings.
#
# Although an ``ObjectArray`` can save memory, creating Python objects in a loop may still use more computation time than is necessary. Therefore, awkward arrays can also have vectorized ``Methods``—bound functions that operate on the array data, rather than instantiating every Python object in an ``ObjectArray``. Although an ``ObjectArray`` is a good use-case for ``Methods``, any awkward array can have them. (The second most common case being a ``JaggedArray`` of ``ObjectArrays``.)
#
# The nesting of awkward arrays within awkward arrays need not be tree-like: they can have cross-references and cyclic references (using ordinary Python assignment). ``IndexedArray`` can aid in building complex structures: it is simply an integer ``index`` that would be applied to its ``content`` with `integer array indexing <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#integer-array-indexing>`__ to get any element. ``IndexedArray`` is the equivalent of a pointer in non-columnar data structures.
#
# The counterpart of an ``IndexedArray`` is a ``SparseArray``: whereas an ``IndexedArray`` consists of pointers *to* elements of its ``content``, a ``SparseArray`` consists of pointers *from* elements of its content, representing a very large array in terms of its non-zero (or non-``default``) elements. Awkward's ``SparseArray`` is a `coordinate format (COO) <https://scipy-lectures.org/advanced/scipy_sparse/coo_matrix.html>`__, one-dimensional array.
#
# Another limitation of Numpy is that arrays cannot span multiple memory buffers. Awkward's ``ChunkedArray`` represents a single logical array made of physical ``chunks`` that may be anywhere in memory. A ``ChunkedArray``'s ``chunksizes`` may be known or unknown. One application of ``ChunkedArray`` is to append data to an array without allocating on every call: ``AppendableArray`` allocates memory in equal-sized chunks.
#
# Another application of ``ChunkedArray`` is to lazily load data in chunks. Awkward's ``VirtualArray`` calls its ``generator`` function to materialize an array when needed, and a ``ChunkedArray`` of ``VirtualArrays`` is a classic lazy-loading array, used to gradually read Parquet and ROOT files. In most libraries, lazy-loading is not a part of the data but a feature of the reading interface. Nesting virtualness makes it possible to load ``Tables`` within ``Tables``, where even the columns of the inner ``Tables`` are on-demand.

# %%markdown
# ## Mutability
#
# Awkward arrays are considered immutable in the sense that elements of the data cannot be modified in-place. That is, assignment with square brackets at an integer index raises an error. Awkward does not prevent the underlying Numpy arrays from being modified in-place, though that can lead to confusing results—the behavior is left undefined. The reason for this omission in functionality is that the internal representation of columnar data structures is more constrained than their non-columnar counterparts: some in-place modification can't be defined, and others have surprising side-effects.
#
# However, the Python objects representing awkward arrays can be changed in-place. Each class has properties defining its structure, such as ``content``, and these may be replaced at any time. (Replacing properties does not change values in any Numpy arrays.) In fact, this is the only way to build cyclic references: an object in Python must be assigned to a name before that name can be used as a reference.
#
# Awkward arrays are appendable, but only through ``AppendableArray``, and ``Table`` columns may be added, changed, or removed. The only use of square-bracket assignment (i.e. ``__setitem__``) is to modify ``Table`` columns.
#
# Awkward arrays produced by an external program may grow continuously, as long as more deeply nested arrays are filled first. That is, the ``content`` of a ``JaggedArray`` must be updated before updating its structure arrays (``starts`` and ``stops``). The definitions of awkward array validity allow for nested elements with no references pointing at them ("unreachable" elements), but not for references pointing to a nested element that doesn't exist.

# %%markdown
# ## Relationship to Arrow
#
# `Apache Arrow <https://arrow.apache.org>`__ is a cross-language, columnar memory format for complex data structures. There is intentionally a high degree of overlap between awkward-array and Arrow. But whereas Arrow's focus is data portability, awkward's focus is computation: it would not be unusual to get data from Arrow, compute something with awkward-array, then return it to another Arrow buffer. For this reason, ``awkward.fromarrow`` is a zero-copy view. Awkward's data representation is broader than Arrow's, so ``awkward.toarrow`` does, in general, perform a copy.
#
# The main difference between awkward-array and Arrow is that awkward-array does not require all arrays to be included within a contiguous memory buffer, though libraries like `pyarrow <https://arrow.apache.org/docs/python>`__ relax this criterion while building a compliant Arrow buffer. This restriction does imply that Arrow cannot encode cross-references or cyclic dependencies.
#
# Arrow also doesn't have the luxury of relying on Numpy to define its `primitive arrays <https://arrow.apache.org/docs/memory_layout.html#primitive-value-arrays>`__, so it has a fixed endianness, has no regular tensors without expressing it as a jagged array, and requires 32-bit integers for indexing, instead of taking whatever integer type a user provides.
#
# `Nullability <https://arrow.apache.org/docs/memory_layout.html#null-bitmaps>`__ is an optional property of every data type in Arrow, but it's a structure element in awkward. Similarly, `dictionary encoding <https://arrow.apache.org/docs/memory_layout.html#dictionary-encoding>`__ is built into Arrow as a fundamental property, but it would be built from an ``IndexedArray`` in awkward. Chunking and lazy-loading are supported by readers such as `pyarrow <https://arrow.apache.org/docs/python>`__, but they're not part of the Arrow data model.
#
# The following list translates awkward-array classes and features to their Arrow counterparts, if possible.
#
# * ``JaggedArray``: Arrow's `list type <https://arrow.apache.org/docs/memory_layout.html#list-type>`__.
# * ``Table``: Arrow's `struct type <https://arrow.apache.org/docs/memory_layout.html#struct-type>`__, though columns can be added to or removed from awkward ``Tables`` whereas Arrow is strictly immutable.
# * ``BitMaskedArray``: every data type in Arrow potentially has a `null bitmap <https://arrow.apache.org/docs/memory_layout.html#null-bitmaps>`__, though it's an explicit array structure in awkward. (Arrow has no counterpart for Awkward's ``MaskedArray`` or ``IndexedMaskedArray``.)
# * ``UnionArray``: directly equivalent to Arrow's `dense union <https://arrow.apache.org/docs/memory_layout.html#dense-union-type>`__. Arrow also has a `sparse union <https://arrow.apache.org/docs/memory_layout.html#sparse-union-type>`__, which awkward-array only has as a ``UnionArray.fromtags`` constructor that builds the dense union on the fly from a sparse union.
# * ``ObjectArray`` and ``Methods``: no counterpart because Arrow must be usable in any language.
# * ``StringArray``: "string" is a logical type built on top of Arrow's `list type <https://arrow.apache.org/docs/memory_layout.html#list-type>`__.
# * ``IndexedArray``: no counterpart (though its role in building `dictionary encoding <https://arrow.apache.org/docs/memory_layout.html#dictionary-encoding>`__ is built into Arrow as a fundamental property).
# * ``SparseArray``: no counterpart.
# * ``ChunkedArray``: no counterpart (though a reader may deal with non-contiguous data).
# * ``AppendableArray``: no counterpart; Arrow is strictly immutable.
# * ``VirtualArray``: no counterpart (though a reader may lazily load data).

# %%markdown
# # High-level operations: common to all classes
#
# There are three levels of abstraction in awkward-array: high-level operations for data analysis, low-level operations for engineering the structure of the data, and implementation details. Implementation details are handled in the usual way for Python: if exposed at all, class, method, and function names begin with underscores and are not guaranteed to be stable from one release to the next. There is more than one implementation of awkward: the original awkward library, which depends only on Numpy, awkward-numba, which uses Numba to just-in-time compile its operations, and awkward-cpp, which has precompiled operations. Each has its own implementation details.
#
# The distinction between high-level operations and low-level operations is more subtle and developed as awkward-array was put to use. Data analysts care about the logical structure of the data—whether it is jagged, what the column names are, whether certain values could be ``None``, etc. Data engineers (or an analyst in "engineering mode") care about contiguousness, how much data are in memory at a given time, whether strings are dictionary-encoded, whether arrays have unreachable elements, etc. The dividing line is between high-level types and low-level array layout (both of which are defined in their own sections below). The following awkward classes have the same high-level type as their content:
#
# * ``IndexedArray`` because indirection to type ``T`` has type ``T``,
# * ``SparseArray`` because a lookup of elements with type ``T`` has type ``T``,
# * ``ChunkedArray`` because the chunks, which must have the same type as each other, collectively have that type when logically concatenated,
# * ``AppendableArray`` because it's a special case of ``ChunkedArray``,
# * ``VirtualArray`` because it produces an array of a given type on demand,
# * ``UnionArray`` has the same type as its ``contents`` *only if* all ``contents`` have the same type as each other.
#
# All other classes, such as ``JaggedArray``, have a logically distinct type from their contents.
#
# This section describes a suite of operations that are common to all awkward classes. For some high-level types, the operation is meaningless or results in an error, such as the jagged ``counts`` of an array that is not jagged at any level, or the ``columns`` of an array that contains no tables, but the operation has a well-defined action on every array class. To use these operations, you do need to understand the high-level type of your data, but not whether it is wrapped in an ``IndexedArray``, a ``SparseArray``, a ``ChunkedArray``, an ``AppendableArray``, or a ``VirtualArray``.

# %%markdown
# ## Slicing with square brackets
#
# The primary operation for all classes is slicing with square brackets. This is the operation defined by Python's ``__getitem__`` method. It is so basic that high-level types are defined in terms of what they return when a scalar argument is passed in square brakets.
#
# Just as Numpy's slicing reproduces but generalizes Python sequence behavior, awkward-array reproduces (most of) `Numpy's slicing behavior <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`__ and generalizes it in certain cases. An integer argument, a single slice argument, a single Numpy array-like of booleans or integers, and a tuple of any of the above is handled just like Numpy. Awkward-array does not handle ellipsis (because the depth of an awkward array can be different on different branches of a ``Table`` or ``UnionArray``) or ``None`` (because it's not always possible to insert a ``newaxis``). Numpy `record arrays <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`__ accept a string or sequence of strings as a column argument if it is the only argument, not in a tuple with other types. Awkward-array accepts a string or sequence of strings if it contains a ``Table`` at some level.
#
# An integer argument selects one element from the top-level array (starting at zero), changing the type by decreasing rank or jaggedness by one level.

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8], [9.9]])
a[0]

# %%markdown
# Negative indexes count backward from the last element,

# %%
a[-1]

# %%markdown
# and the index (after translating negative indexes) must be at least zero and less than the length of the top-level array.

# %%
try:
    a[-6]
except Exception as err:
    print(type(err), str(err))

# %%markdown
# A slice selects a range of elements from the top-level array, maintaining the array's type. The first index is the inclusive starting point (starting at zero) and the second index is the exclusive endpoint.

# %%
a[2:4]

# %%markdown
# Python's slice syntax (above) or literal ``slice`` objects may be used.

# %%
a[slice(2, 4)]

# %%markdown
# Negative indexes count backward from the last element and endpoints may be omitted.

# %%
a[-2:]

# %%markdown
# Start and endpoints beyond the array are not errors: they are truncated.

# %%
a[2:100]

# %%markdown
# A skip value (third index of the slice) sets the stride for indexing, allowing you to skip elements, and this skip can be negative. It cannot, however, be zero.

# %%
a[::-1]

# %%markdown
# A Numpy array-like of booleans with the same length as the array may be used to filter elements. Numpy has a specialized `numpy.compress <https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html>`__ function for this operation, but the only way to get it in awkward-array is through square brackets.

# %%
a[[True, True, False, True, False]]

# %%markdown
# A Numpy array-like of integers with the same length as the array may be used to select a collection of indexes. Numpy has a specialized `numpy.take <https://docs.scipy.org/doc/numpy/reference/generated/numpy.take.html>`__ function for this operation, but the only way to get it in awkward-array is through square brakets. Negative indexes and repeated elements are handled in the same way as Numpy.

# %%
a[[-1, 0, 1, 2, 2, 2]]

# %%markdown
# A tuple of length ``N`` applies selections to the first ``N`` levels of rank or jaggedness. Our example array has only two levels, so we can apply two kinds of indexes.

# %%
a[2:, 0]

# %%
a[[True, False, True, True, False], ::-1]

# %%
a[[0, 3, 0], 1::]

# %%markdown
# As described in Numpy's `advanced indexing <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing>`__, advanced indexes (boolean or integer arrays) are broadcast and iterated as one:

# %%
a[[0, 3], [True, False, True]]

# %%markdown
# Awkward array has two extensions beyond Numpy, both of which affect only jagged data. If an array is jagged and a jagged array of booleans with the same structure (same length at all levels) is passed in square brackets, only inner arrays would be filtered.

# %%
a    = awkward.fromiter([[  1.1,   2.2,  3.3], [], [ 4.4,  5.5], [ 6.6,  7.7,   8.8], [  9.9]])
mask = awkward.fromiter([[False, False, True], [], [True, True], [True, True, False], [False]])
a[mask]

# %%markdown
# Similarly, if an array is jagged and a jagged array of integers with the same structure is passed in square brackets, only inner arrays would be filtered/duplicated/rearranged.

# %%
a     = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8], [9.9]])
index = awkward.fromiter([[2, 2, 2, 2], [], [1, 0], [2, 1, 0], []])
a[index]

# %%markdown
# Although all of the above use a ``JaggedArray`` as an example, the principles are general: you should get analogous results with jagged tables, masked jagged arrays, etc. Non-jagged arrays only support Numpy-like slicing.
#
# If an array contains a ``Table``, it can be selected with a string or a sequence of strings, just like Numpy `record arrays <https://docs.scipy.org/doc/numpy/user/basics.rec.html>`__.

# %%
a = awkward.fromiter([{"x": 1, "y": 1.1, "z": "one"}, {"x": 2, "y": 2.2, "z": "two"}, {"x": 3, "y": 3.3, "z": "three"}])
a

# %%
a["x"]

# %%
a[["z", "y"]].tolist()

# %%markdown
# Like Numpy, integer indexes and string indexes commute if the integer index corresponds to a structure outside the ``Table`` (this condition is always met for Numpy record arrays).

# %%
a["y"][1]

# %%
a[1]["y"]

# %%
a = awkward.fromiter([[{"x": 1, "y": 1.1, "z": "one"}, {"x": 2, "y": 2.2, "z": "two"}], [], [{"x": 3, "y": 3.3, "z": "three"}]])
a

# %%
a["y"][0][1]

# %%
a[0]["y"][1]

# %%
a[0][1]["y"]

# %%markdown
# but not

# %%
a = awkward.fromiter([{"x": 1, "y": [1.1]}, {"x": 2, "y": [2.1, 2.2]}, {"x": 3, "y": [3.1, 3.2, 3.3]}])
a

# %%
a["y"][2][1]

# %%
a[2]["y"][1]

# %%
try:
    a[2][1]["y"]
except Exception as err:
    print(type(err), str(err))

# %%markdown
because

# %%
a[2].tolist()

# %%markdown
# cannot take a ``1`` argument before ``"y"``.
#
# Just as integer indexes can be alternated with string/sequence of string indexes, so can slices, arrays, and tuples of slices and arrays.

# %%
a["y"][:, 0]

# %%markdown
# Generally speaking, string and sequence of string indexes are *column* indexes, while all other types are *row* indexes.

# %%markdown
# ## Assigning with square brackets
#
# As discussed above, awkward arrays are generally immutable with few exceptions. Row assignment is only possible via appending to an ``AppendableArray``. Column assignment, reassignment, and deletion are in general allowed. The syntax for assigning and reassigning columns is through assignment to a square bracket expression. This operation is defined by Python's ``__setitem__`` method. The syntax for deleting columns is through the ``del`` operators on a square bracket expression. This operation is defined by Python's ``__delitem__`` method.
#
# Since only columns can be changed, only strings and sequences of strings are allowed as indexes.

# %%
a = awkward.fromiter([[{"x": 1, "y": 1.1, "z": "one"}, {"x": 2, "y": 2.2, "z": "two"}], [], [{"x": 3, "y": 3.3, "z": "three"}]])
a

# %%
a["a"] = awkward.fromiter([[100, 200], [], [300]])
a.tolist()

# %%
del a["a"]
a.tolist()

# %%
a[["a", "b"]] = awkward.fromiter([[{"first": 100, "second": 111}, {"first": 200, "second": 222}], [], [{"first": 300, "second": 333}]])
a.tolist()

# %%markdown
# Note that the names of the columns on the right-hand side of the assignment are irrelevant; we're setting two columns, there needs to be two columns on the right. Columns can be anonymous:

# %%
a[["a", "b"]] = awkward.Table(awkward.fromiter([[100, 200], [], [300]]), awkward.fromiter([[111, 222], [], [333]]))
a.tolist()

# %%markdown
# Another thing to note is that the structure (lengths at all levels of jaggedness) must match if the depth is the same.

# %%
try:
    a["c"] = awkward.fromiter([[100, 200, 300], [400], [500, 600]])
except Exception as err:
    print(type(err), str(err))

# %%markdown
# But if the right-hand side is shallower and can be *broadcasted* to the left-hand side, it will be. (See below for broadcasting.)

# %%
a["c"] = awkward.fromiter([100, 200, 300])
a.tolist()

# %%markdown
# ## Numpy-like broadcasting
#
# In assignments and mathematical operations between higher-rank and lower-rank arrays, Numpy repeats values in the lower-rank array to "fit," if possible, before applying the operation. This is called `boradcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`__. For example,

# %%
numpy.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]) + 100

# %%markdown
Singletons are also expanded to fit.

# %%
numpy.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]) + numpy.array([[100], [200]])

# %%markdown
# Awkward arrays have the same feature, but this has particularly useful effects for jagged arrays. In an operation involving two arrays of different depths of jaggedness, the shallower one expands to fit the deeper one.

# %%
awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]]) + awkward.fromiter([100, 200, 300])

# %%markdown
# Note that the ``100`` was broadcasted to all three of the elements of the first inner array, ``200`` was broadcasted to no elements in the second inner array (because the second inner array is empty), and ``300`` was broadcasted to all two of the elements of the third inner array.
#
# This is the columnar equivalent to accessing a variable defined outside of an inner loop.

# %%
jagged = [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
flat = [100, 200, 300]
for i in range(3):
    for j in range(len(jagged[i])):
        # j varies in this loop, but i is constant
        print(i, j, jagged[i][j] + flat[i])

# %%markdown
# Many translations of non-columnar code to columnar code has this form. It's often surprising to users that they don't have to do anything special to get this feature (e.g. ``cross``).

# %%markdown
# ## Support for Numpy universal functions (ufuncs)
#
# Numpy's key feature of array-at-a-time programming is mainly provided by "universal functions" or "ufuncs." This is a special class of function that applies a scalars → scalar kernel independently to aligned elements of internal arrays to return a same-shape output array. That is, for a scalars → scalar function ``f(x1, ..., xN) → y``, the ufunc takes ``N`` input arrays of the same ``shape`` and returns one output array with that ``shape`` in which ``output[i] = f(input1[i], ..., inputN[i])`` for all ``i``.

# %%
# N = 1
numpy.sqrt(numpy.array([1, 4, 9, 16, 25]))

# %%
# N = 2
numpy.add(numpy.array([[1.1, 2.2], [3.3, 4.4]]), numpy.array([[100, 200], [300, 400]]))

# %%markdown
# Keep in mind that a ufunc is not simply a function that has this property, but a specially named class, deriving from a type in the Numpy library.

# %%
numpy.sqrt, numpy.add

# %%
isinstance(numpy.sqrt, numpy.ufunc), isinstance(numpy.add, numpy.ufunc)

# %%markdown
# This class of functions can be overridden, and awkward-array overrides them to recognize and properly handle awkward arrays.

# %%
numpy.sqrt(awkward.fromiter([[1, 4, 9], [], [16, 25]]))

# %%
numpy.add(awkward.fromiter([[[1.1], 2.2], [], [3.3, None]]), awkward.fromiter([[[100], 200], [], [None, 300]]))

# %%markdown
# Only the primary action of the ufunc (``ufunc.__call__``) has been overridden; methods like ``ufunc.at``, ``ufunc.reduce``, and ``ufunc.reduceat`` are not supported. Also, the in-place ``out`` parameter is not supported because awkward array data cannot be changed in-place.
#
# For awkward arrays, the input arguments to a ufunc must all have the same structure or, if shallower, be broadcastable to the deepest structure. (See above for "broadcasting.") The scalar function is applied to elements at the same positions within this structure from different input arrays. The output array has this structure, populated by return values of the scalar function.
#
# * Rectangular arrays must have the same shape, just as in Numpy. A scalar can be broadcasted (expanded) to have the same shape as the arrays.
# * Jagged arrays must have the same number of elements in all inner arrays. A rectangular array with the same outer shape (i.e. containing scalars instead of inner arrays) can be broadcasted to inner arrays with the same lengths.
# * Tables must have the same sets of columns (though not necessarily in the same order). There is no broadcasting of missing columns.
# * Missing values (``None`` from ``MaskedArrays``) transform to missing values in every ufunc. That is, ``None + 5`` is ``None``, ``None + None`` is ``None``, etc.
# * Different data types (through a ``UnionArray``) must be compatible at every site where values are included in the calculation. For instance, input arrays may contain tables with different sets of columns, but all inputs at index ``i`` must have the same sets of columns as each other:

# %%
numpy.add(awkward.fromiter([{"x": 1, "y": 1.1}, {"y": 1.1, "z": 100}]),
          awkward.fromiter([{"x": 3, "y": 3.3}, {"y": 3.3, "z": 300}])).tolist()

# %%markdown
# Unary and binary operations on awkward arrays, such as ``-x``, ``x + y``, and ``x**2``, are actually Numpy ufuncs, so all of the above applies to them as well (such as broadcasting the scalar ``2`` in ``x**2``).
#
# Remember that only ufuncs have been overridden by awkward-array: other Numpy functions such as ``numpy.concatenate`` are ignorant of awkward arrays and will attempt to convert them to Numpy first. In some cases, that may be what you want, but in many, especially any cases involving jagged arrays, it will be a major performance loss and a loss of functionality: jagged arrays turn into Numpy ``dtype=object`` arrays containing Numpy arrays, which can be a very large number of Python objects and doesn't behave as a multidimensional array.
#
# You can check to see if a function from Numpy is a ufunc with ``isinstance``.

# %%
isinstance(numpy.concatenate, numpy.ufunc)

# %%markdown
# and you can prevent accidental conversions to Numpy by setting ``allow_tonumpy`` to ``False``, either on one array or globally on a whole class of awkward arrays. (See "global switches" below.)

# %%
x = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
y = awkward.fromiter([[6.6, 7.7, 8.8], [9.9]])
numpy.concatenate([x, y])

# %%
x.allow_tonumpy = False
try:
    numpy.concatenate([x, y])
except Exception as err:
    print(type(err), str(err))

# %%markdown
# ## Global switches
#
# The ``AwkwardArray`` abstract base class has the following switches to turn off sometmes-undesirable behavior. These switches could be set on the ``AwkwardArray`` class itself, affecting all awkward arrays, or they could be set on a particular class like ``JaggedArray`` to only affect ``JaggedArray`` instances, or they could be set on a particular instance, to affect only that instance.
#
# * ``allow_tonumpy`` (default is ``True``); if ``False``, forbid any action that would convert an awkward array into a Numpy array (with a likely loss of performance and functionality).
# * ``allow_iter`` (default is ``True``); if ``False``, forbid any action that would iterate over an awkward array in Python (except printing a few elements as part of its string representation).
# * ``check_prop_valid`` (default is ``True``); if ``False``, skip the single-property validity checks in array constructors and when setting properties.
# * ``check_whole_valid`` (default is ``True``); if ``False``, skip the whole-array validity checks that are typically called before methods that need them.

# %%
awkward.AwkwardArray.check_prop_valid

# %%
awkward.JaggedArray.check_whole_valid

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
numpy.array(a)

# %%
a.allow_tonumpy = False
try:
    numpy.array(a)
except Exception as err:
    print(type(err), str(err))

# %%
list(a)

# %%
a.allow_iter = False
try:
    list(a)
except Exception as err:
    print(type(err), str(err))

# %%
a

# %%markdown
# ## Generic properties and methods
#
# All awkward arrays have the following properties and methods.

# %%markdown
# * ``type``: the high-level type of the array. (See below for a detailed description of high-level types.)

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
b = awkward.fromiter([[1.1, 2.2, None, 3.3, None],
                      [4.4, [5.5]],
                      [{"x": 6, "y": {"z": 7}}, None, {"x": 8, "y": {"z": 9}}]
                     ])

# %%
a.type

# %%
print(a.type)

# %%
b.type

# %%
print(b.type)

# %%markdown
# * ``layout``: the low-level layout of the array. (See below for a detailed description of low-level layouts.)
a.layout

# %%
b.layout

# %%markdown
# * ``dtype``: the `Numpy dtype <https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html>`__ that this array would have if cast as a Numpy array. Numpy dtypes cannot fully specify awkward arrays: use the ``type`` for an analyst-friendly description of the data type or ``layout`` for details about how the arrays are represented.

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
a.dtype   # the closest Numpy dtype to a jagged array is dtype=object ('O')

# %%
numpy.array(a)

# %%markdown
# * ``shape``: the `Numpy shape <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html>`__ that this array would have if cast as a Numpy array. This only specifies the first regular dimensions, not any jagged dimensions or regular dimensions nested within awkward structures. The Python length (``__len__``) of the array is the first element of this ``shape``.

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
a.shape

# %%
len(a)

# %%markdown
# The following ``JaggedArray`` has two fixed-size dimensions at the top, followed by a jagged dimension inside of that. The shape only represents the first few dimensions.

# %%
a = awkward.JaggedArray.fromcounts([[3, 0], [2, 4]], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
a

# %%
a.shape

# %%
len(a)

# %%
print(a.type)

# %%markdown
# Also, a dimension can effectively be fixed-size, but represented by a ``JaggedArray``. The ``shape`` does not encompass any dimensions represented by a ``JaggedArray``.

# %%
# Same structure, but it's JaggedArrays all the way down.
b = a.structure1d()
b

# %%
b.shape

# %%markdown
# * ``size``: the product of ``shape``, as in Numpy.

# %%
a.shape

# %%
a.size

# %%markdown
# * ``nbytes``: the total number of bytes in all memory buffers referenced by the array, not including bytes in Python objects (which are Python-implementation dependent, not even available in PyPy). Same as the Numpy property of the same name.

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
a.nbytes

# %%
a.offsets.nbytes + a.content.nbytes

# %%markdown
# * ``tolist()``: converts the array into Python objects: ``lists`` for arrays, ``dicts`` for table rows, ``tuples`` for table rows with anonymous fields and a ``rowname`` of ``"tuple"``, ``None`` for missing data, and Python objects from ``ObjectArrays``. This is an approximate inverse of ``awkward.fromiter``.

# %%
awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]]).tolist()

# %%
awkward.fromiter([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]).tolist()

# %%
awkward.Table.named("tuple", [1, 2, 3], [1.1, 2.2, 3.3]).tolist()

# %%
awkward.fromiter([[1.1, 2.2, None], [], [None, 3.3]]).tolist()

# %%
class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

a = awkward.fromiter([[Point(1, 1.1), Point(2, 2.2), Point(3, 3.3)], [], [Point(4, 4.4), Point(5, 5.5)]])
a

# %%
a.tolist()

# %%markdown
# * ``valid(exception=False, message=False)``: manually invoke the whole-array validity checks on the top-level array (not recursively). With the default options, this function returns ``True`` if valid and ``False`` if not. If ``exception=True``, it returns nothing on success and raises the appropriate exception on failure. If ``message=True``, it returns ``None`` on success and the error string on failure. (TODO: ``recursive=True``?)

# %%
a = awkward.JaggedArray.fromcounts([3, 0, 2], [1.1, 2.2, 3.3, 4.4])  # content array is too short
a.valid()

# %%
try:
    a.valid(exception=True)
except Exception as err:
    print(type(err), str(err))

# %%
a.valid(message=True)

# %%markdown
# * ``astype(dtype)``: convert *nested Numpy arrays* into the given type while maintaining awkward structure.

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
a.astype(numpy.int32)

# %%markdown
# * ``regular()``: convert the awkward array into a Numpy array and (unlike ``numpy.array(awkward_array)``) raise an error if it cannot be faithfully represented.

# %%
# This JaggedArray happens to have equal-sized inner arrays.
a = awkward.fromiter([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]])
a

# %%
a.regular()

# %%
# This one does not.
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
a

# %%
try:
    a.regular()
except Exception as err:
    print(type(err), str(err))

# %%markdown
# * ``copy(optional constructor arguments...)``: copy an awkward array object, non-recursively and without copying memory buffers, possibly replacing some of its parameters. If the class is an awkward subclass or has mix-in methods, they are propagated to the copy.

# %%
class Special:
    def get(self, index):
        try:
            return self[index]
        except IndexError:
            return None

JaggedArrayMethods = awkward.Methods.mixin(Special, awkward.JaggedArray)

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
a.__class__ = JaggedArrayMethods
a

# %%
a.get(2)

# %%
a.get(3)

# %%
b = a.copy(content=[100, 200, 300, 400, 500])
b

# %%
b.get(2)

# %%
b.get(3)

# %%markdown
# Internally, all the methods that return views of the array (like slicing) use ``copy`` to retain the special methods.

# %%
c = a[1:]
c

# %%
c.get(1)

# %%
c.get(2)

# %%markdown
# * ``deepcopy(optional constructor arguments...)``: like ``copy``, except that it recursively copies all internal structure, including memory buffers associated with Numpy arrays.

# %%
b = a.deepcopy(content=[100, 200, 300, 400, 500])
b

# %%
# Modify the structure of a (not recommended; this is a demo).
a.starts[0] = 1
a

# %%
# But b is not modified. (If it were, it would start with 200.)
b

# %%markdown
# * ``empty_like(optional constructor arguments...)``
# * ``zeros_like(optional constructor arguments...)``
# * ``ones_like(optional constructor arguments...)``: recursively copies structure, replacing contents with new uninitialized buffers, new buffers full of zeros, or new buffers full of ones. Not usually used in analysis, but needed for implementation.

# %%
d = a.zeros_like()
d

# %%
e = a.ones_like()
e

# %%markdown
# ## Reducers
#
# All awkward arrays also have a complete set of reducer methods. Reducers can be found in Numpy as well (as array methods and as free-standing functions), but they're not called out as a special class the way that universal functions ("ufuncs") are. Reducers decrease the rank or jaggedness of an array by one dimension, replacing subarrays with scalars. Examples include ``sum``, ``min``, and ``max``, but any monoid (associative operation with an identity) can be a reducer.
#
# In awkward-array, reducers are only array methods (not free-standing functions) and unlike Numpy, they do not take an ``axis`` parameter. When a reducer is called at any level, it reduces the innermost dimension. (Since outer dimensions can be jagged, this is the only dimension that can be meaningfully reduced.)

# %%
a = awkward.fromiter([[[[1, 2], [3]], [[4, 5]]], [[[], [6, 7, 8, 9]]]])
a

# %%
a.sum()

# %%
a.sum().sum()

# %%
a.sum().sum().sum()

# %%
a.sum().sum().sum().sum()

# %%markdown
# In the following example, "the deepest axis" of different fields in the table are at different depths: singly jagged in ``"x"`` and doubly jagged array in ``"y"``. The ``sum`` reduces each depth by one, producing a flat array ``"x"`` and a singly jagged array in ``"y"``.

# %%
a = awkward.fromiter([{"x": [], "y": [[0.1, 0.2], [], [0.3]]}, {"x": [1, 2, 3], "y": [[0.4], [], [0.5, 0.6]]}])
a.tolist()

# %%
a.sum().tolist()

# %%markdown
# This sum cannot be reduced again because ``"x"`` is not jagged (would reduce to a scalar) and ``"y"`` is (would reduce to an array). The result cannot be scalar in one field (a single row, not a collection) and an array in another field (a collection).

# %%
try:
    a.sum().sum()
except Exception as err:
    print(type(err), str(err))

# %%markdown
# A table can be reduced if all of its fields are jagged or if all of its fields are not jagged; here's an example of the latter.

# %%
a = awkward.fromiter([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])
a.tolist()

# %%
a.sum()

# %%markdown
# The resulting object is a scalar row—for your convenience, it has been labeled with the reducer that produced it.

# %%
isinstance(a.sum(), awkward.Table.Row)

# %%markdown
# ``UnionArrays`` are even more constrained: they can only be reduced if they have primitive (Numpy) type.

# %%
a = awkward.fromiter([1, 2, 3, {"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}])
a

# %%
try:
    a.sum()
except Exception as err:
    print(type(err), str(err))

# %%
a = awkward.UnionArray.fromtags([0, 0, 0, 1, 1],
                                [numpy.array([1, 2, 3], dtype=numpy.int32),
                                 numpy.array([4, 5], dtype=numpy.float64)])
a

# %%
a.sum()

# %%markdown
# In all reducers, ``NaN`` in floating-point arrays and ``None`` in ``MaskedArrays`` are skipped, so these reducers are more like ``numpy.nansum``, ``numpy.nanmax``, and ``numpy.nanmin``, but generalized to all nullable types.

# %%
a = awkward.fromiter([[[[1.1, numpy.nan], [2.2]], [[None, 3.3]]], [[[], [None, numpy.nan, None]]]])
a

# %%
a.sum()

# %%
a = awkward.fromiter([[{"x": 1, "y": 1.1}, None, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": numpy.nan}]])
a.tolist()

# %%
a.sum().tolist()

# %%markdown
# The following reducers are defined as methods on all awkward arrays.

# %%markdown
# * ``reduce(ufunc, identity)``: generic reducer, calls ``ufunc.reduceat`` and returns ``identity`` for empty arrays.

# %%
# numba.vectorize makes new ufuncs (requires type signatures and a kernel function)
import numba
@numba.vectorize([numba.int64(numba.int64, numba.int64)])
def sum_mod_10(x, y):
    return (x + y) % 10

# %%
a = awkward.fromiter([[1, 2, 3], [], [4, 5, 6], [7, 8, 9, 10]])
a.sum()

# %%
a.reduce(sum_mod_10, 0)

# %%
# Missing (None) values are ignored.
a = awkward.fromiter([[1, 2, None, 3], [], [None, None, None], [7, 8, 9, 10]])
a.reduce(sum_mod_10, 0)

# %%markdown
# * ``any()``: boolean reducer, returns ``True`` if any (logical or) of the elements of an array are ``True``, returns ``False`` for empty arrays.

# %%
a = awkward.fromiter([[False, False], [True, True], [True, False], []])
a.any()

# %%
# Missing (None) values are ignored.
a = awkward.fromiter([[False, None], [True, None], [None]])
a.any()

# %%markdown
# * ``all()``: boolean reducer, returns ``True`` if all (logical and) of the elements of an array are ``True``, returns ``True`` for empty arrays.

# %%
a = awkward.fromiter([[False, False], [True, True], [True, False], []])
a.all()

# %%
# Missing (None) values are ignored.
a = awkward.fromiter([[False, None], [True, None], [None]])
a.all()

# %%markdown
# * ``count()``: returns the (integer) number of elements in an array, skipping ``None`` and ``NaN``.

# %%
a = awkward.fromiter([[1.1, 2.2, None], [], [3.3, numpy.nan]])
a.count()

# %%markdown
# * ``count_nonzero()``: returns the (integer) number of non-zero elements in an array, skipping ``None`` and ``NaN``.

# %%
a = awkward.fromiter([[1.1, 2.2, None, 0], [], [3.3, numpy.nan, 0]])
a.count_nonzero()

# %%markdown
# * ``sum()``: returns the sum of each array, skipping ``None`` and ``NaN``, returning 0 for empty arrays.

# %%
a = awkward.fromiter([[1.1, 2.2, None], [], [3.3, numpy.nan]])
a.sum()

# %%markdown
# * ``prod()``: returns the product (multiplication) of each array, skipping ``None`` and ``NaN``, returning 1 for empty arrays.

# %%
a = awkward.fromiter([[1.1, 2.2, None], [], [3.3, numpy.nan]])
a.prod()

# %%markdown
# * ``min()``: returns the minimum number in each array, skipping ``None`` and ``NaN``, returning infinity or the largest possible integer for empty arrays. (Note that Numpy raises errors for empty arrays.)

# %%
a = awkward.fromiter([[1.1, 2.2, None], [], [3.3, numpy.nan]])
a.min()

# %%
a = awkward.fromiter([[1, 2, None], [], [3]])
a.min()

# %%markdown
# The identity of minimization is ``inf`` for floating-point values and ``9223372036854775807`` for ``int64`` because minimization with any other value would return the other value. This is more convenient for data analysts than raising an error because empty inner arrays are common.

# %%markdown
# * ``max()``: returns the maximum number in each array, skipping ``None`` and ``NaN``, returning negative infinity or the smallest possible integer for empty arrays. (Note that Numpy raises errors for empty arrays.)

# %%
a = awkward.fromiter([[1.1, 2.2, None], [], [3.3, numpy.nan]])
a.max()

# %%
a = awkward.fromiter([[1, 2, None], [], [3]])
a.max()

# %%markdown
# The identity of maximization is ``-inf`` for floating-point values and ``-9223372036854775808`` for ``int64`` because maximization with any other value would return the other value. This is more convenient for data analysts than raising an error because empty inner arrays are common.
#
# Note that the maximization-identity for unsigned types is ``0``.

# %%
a = awkward.JaggedArray.fromcounts([3, 0, 2], numpy.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=numpy.uint16))
a

# %%
a.max()

# %%markdown
# Functions like mean and standard deviation aren't true reducers because they're not associative (``mean(mean(x1, x2, x3), mean(x4, x5))`` is not equal to ``mean(mean(x1, x2), mean(x3, x4, x5))``). However, they're useful methods that exist on all awkward arrays, defined in terms of reducers.

# %%markdown
# * ``moment(n, weight=None)``: returns the ``n``th moment of each array (a floating-point value), skipping ``None`` and ``NaN``, returning ``NaN`` for empty arrays. If ``weight`` is given, it is taken as an array of weights, which may have the same structure as the ``array`` or be broadcastable to it, though any broadcasted weights would have no effect on the moment.

# %%
a = awkward.fromiter([[1, 2, 3], [], [4, 5]])

# %%
a.moment(1)

# %%
a.moment(2)

# %%markdown
# Here is the first moment (mean) with a weight broadcasted from a scalar and from a non-jagged array, to show how it doesn't affect the result. The moment is calculated over an inner array, so if a constant value is broadcasted to all elements of that inner array, they all get the same weight.

# %%
a.moment(1)

# %%
a.moment(1, 100)

# %%
a.moment(1, numpy.array([100, 200, 300]))

# %%markdown
# Only when the weight varies across an inner array does it have an effect.

# %%
a.moment(1, awkward.fromiter([[1, 10, 100], [], [0, 100]]))

# %%markdown
# * ``mean(weight=None)``: returns the mean of each array (a floating-point value), skipping ``None`` and ``NaN``, returning ``NaN`` for empty arrays, using optional ``weight`` as above.

# %%
a = awkward.fromiter([[1, 2, 3], [], [4, 5]])
a.mean()

# %%markdown
# * ``var(weight=None, ddof=0)``: returns the variance of each array (a floating-point value), skipping ``None`` and ``NaN``, returning ``NaN`` for empty arrays, using optional ``weight`` as above. The ``ddof`` or "Delta Degrees of Freedom" replaces a divisor of ``N`` (count or sum of weights) with a divisor of ``N - ddof``, following `numpy.var <https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html>`__.

# %%
a = awkward.fromiter([[1, 2, 3], [], [4, 5]])
a.var()

# %%
a.var(ddof=1)

# %%markdown
# * ``std(weight=None, ddof=0)``: returns the standard deviation of each array, the square root of the variance described above.

# %%
a.std()

# %%
a.std(ddof=1)

# %%markdown
# ## Properties and methods for jaggedness
#
# All awkward arrays have these methods, but they provide information about the first nested ``JaggedArray`` within a structure. If, for instance, the ``JaggedArray`` is within some structure that doesn't affect high-level type (e.g. ``IndexedArray``, ``ChunkedArray``, ``VirtualArray``), then the methods are passed through to the ``JaggedArray``. If it's nested within something that does change type, but can meaningfully pass on the call, such as ``MaskedArray``, then that's what they do. If, however, it reaches a ``Table``, which may have some jagged columns and some non-jagged columns, the propagation stops.
#
# * ``counts``: Numpy array of the number of elements in each inner array of the shallowest ``JaggedArray``. The ``counts`` may have rank > 1 if there are any fixed-size dimensions before the ``JaggedArray``.

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
a.counts

# %%
# MaskedArrays return -1 for missing values.
a = awkward.fromiter([[1.1, 2.2, 3.3], [], None, [6.6, 7.7, 8.8, 9.9]])
a.counts

# %%markdown
# A missing inner array (counts is ``-1``) is distinct from an empty inner array (counts is ``0``), but if you want to ensure that you're working with data that have at least ``N`` elements, ``counts >= N`` works.

# %%
a.counts >= 1

# %%
a[a.counts >= 1]

# %%
# UnionArrays return -1 for non-jagged arrays mixed with jagged arrays.
a = awkward.fromiter([[1.1, 2.2, 3.3], [], 999, [6.6, 7.7, 8.8, 9.9]])
a.counts

# %%
# Same for tabular data, regardless of whether they contain nested jagged arrays.
a = awkward.fromiter([[1.1, 2.2, 3.3], [], {"x": 1, "y": [1.1, 1.2, 1.3]}, [6.6, 7.7, 8.8, 9.9]])
a.counts

# %%markdown
# Note! This means that pure ``Tables`` will always return zeros for counts, regardless of what they contain.

# %%
a = awkward.fromiter([{"x": [], "y": []}, {"x": [1], "y": [1.1]}, {"x": [1, 2], "y": [1.1, 2.2]}])
a.counts

# %%markdown
# If all of the columns of a ``Table`` are ``JaggedArrays`` with the same structure, you probably want to zip them into a single ``JaggedArray``.

# %%
b = awkward.JaggedArray.zip(x=a.x, y=a.y)
b

# %%
b.counts

# %%markdown
# * ``flatten(axis=0)``: removes one level of structure (losing information about boundaries between inner arrays) at a depth of jaggedness given by ``axis``.

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
a.flatten()

# %%markdown
# Unlike a ``JaggedArray``'s ``content``, which is part of its low-level layout, ``flatten()`` performs a high-level logical operation. Here's an example of the distinction.

# %%
# JaggedArray with an unusual but valid structure.
a = awkward.JaggedArray([3, 100, 0, 6], [6, 100, 2, 10],
                        [4.4, 5.5, 999, 1.1, 2.2, 3.3, 6.6, 7.7, 8.8, 9.9, 123])
a

# %%
a.flatten()   # gives you a logically flattened array

# %%
a.content     # gives you an internal structure component of the array

# %%markdown
# In many cases, the output of ``flatten()`` corresponds to the output of ``content``, but be aware of the difference and use the one you want.
#
# With ``flatten(axis=1)``, we can internally flatten nested ``JaggedArrays``.

# %%
a = awkward.fromiter([[[1.1, 2.2], [3.3]], [], [[4.4, 5.5]], [[6.6, 7.7, 8.8], [], [9.9]]])
a

# %%
a.flatten(axis=0)

# %%
a.flatten(axis=1)

# %%markdown
# Even if a ``JaggedArray``'s inner structure is due to a fixed-shape Numpy array, the ``axis`` parameter propagates down and does the right thing.

# %%
a = awkward.JaggedArray.fromcounts(numpy.array([3, 0, 2]),
                                   numpy.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]))
a

# %%
type(a.content)

# %%
a.flatten(axis=1)

# %%markdown
# But, unlike Numpy, we can't ask for an ``axis`` starting from the other end (with a negative index). The "deepest array" is not a well-defined concept for awkward arrays.

# %%
try:
    a.flatten(axis=-1)
except Exception as err:
    print(type(err), str(err))

# %%
a = awkward.fromiter([[[1.1, 2.2], [3.3]], [], None, [[6.6, 7.7, 8.8], [], [9.9]]])
a

# %%
a.flatten(axis=1)

# %%markdown
# * ``pad(length, maskedwhen=True, clip=False)``: ensures that each inner array has at least ``length`` elements by filling in the empty spaces with ``None`` (i.e. by inserting a ``MaskedArray`` layer). The ``maskedwhen`` parameter determines whether ``mask[i] == True`` means the element is ``None`` (``maskedwhen=True``) or not ``None`` (``maskedwhen=False``). Setting ``maskedwhen`` doesn't change the logical meaning of the array. If ``clip=True``, then the inner arrays will have exactly ``length`` elements (by clipping the ones that are too long). Even though this results in regular sizes, they are still represented by a ``JaggedArray``.

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
a

# %%
a.pad(3)

# %%
a.pad(3, maskedwhen=False)

# %%
a.pad(3, clip=True)

# %%markdown
# If you want to get rid of the ``MaskedArray`` layer, replace ``None`` with some value.

# %%
a.pad(3).fillna(-999)

# %%markdown
# If you want to make an effectively regular array into a real Numpy array, use ``regular``.

# %%
a.pad(3, clip=True).fillna(0).regular()

# %%markdown
# If a ``JaggedArray`` is nested within some other type, ``pad`` will propagate down to it.

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], None, [4.4, 5.5], None])
a

# %%
a.pad(3)

# %%
a = awkward.Table(x=[[1, 1], [2, 2], [3, 3], [4, 4]],
                  y=awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]))
a.tolist()

# %%
a.pad(3).tolist()

# %%
a.pad(3, clip=True).tolist()

# %%markdown
# If you pass a ``pad`` through a ``Table``, be sure that every field in each record is a nested array (and therefore can be padded).

# %%
a = awkward.Table(x=[1, 2, 3, 4],
                  y=awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]))
a.tolist()

# %%
try:
    a.pad(3)
except Exception as err:
    print(type(err), str(err))

# %%markdown
# The same goes for ``UnionArrays``.

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3, [1, 2, 3]], [], [4.4, 5.5, [4, 5]]])
a

# %%
a.pad(5)

# %%
a = awkward.UnionArray.fromtags([0, 0, 0, 1, 1],
                                [awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]]),
                                 awkward.fromiter([[100, 101], [102]])])
a

# %%
a.pad(3)

# %%
a = awkward.UnionArray.fromtags([0, 0, 0, 1, 1],
                                [awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]]),
                                 awkward.fromiter([100, 200])])
a

# %%
try:
    a.pad(3)
except Exception as err:
    print(type(err), str(err))

# %%markdown
# The general behavior of ``pad`` is to replace the shallowest ``JaggedArray`` with a ``JaggedArray`` containing a ``MaskedArray``. The one exception to this type signature is that ``StringArrays`` are padded with characters.

# %%
a = awkward.fromiter(["one", "two", "three"])
a

# %%
a.pad(4, clip=True)

# %%
a.pad(4, maskedwhen=b".", clip=True)

# %%
a.pad(4, maskedwhen=b"\x00", clip=True)

# %%markdown
# * ``argmin()`` and ``argmax()``: returns the index of the minimum or maximum value in a non-jagged array or the indexes where each inner array is minimized or maximized. The jagged structure of the return value consists of empty arrays for each empty array and singleton arrays for non-empty ones, consisting of a single index in an inner array. This is the form needed to extract one element from each inner array using jagged indexing.

# %%
a = awkward.fromiter([[-3.3, 5.5, -8.8], [], [-6.6, 0.0, 2.2, 3.3], [], [2.2, -2.2, 4.4]])
absa = abs(a)

# %%
a

# %%
absa

# %%
index = absa.argmax()
index

# %%
absa[index]

# %%
a[index]

# %%markdown
# * ``cross(other, nested=False)`` and ``argcross(other, nested=False)``: returns jagged tuples representing the `cross-join <https://en.wikipedia.org/wiki/Join_(SQL)#Cross_join>`__ of `array[i]` and `other[i]` separately for each `i`. If `nested=True`, the result is doubly jagged so that each element of the output corresponds to exactly one element in the original `array`.

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
b = awkward.fromiter([["one", "two"], ["three"], ["four", "five", "six"], ["seven"]])
a.cross(b)

# %%
a.cross(b, nested=True)

# %%markdown
# The "arg" version returns indexes at which the appropriate objects may be found, as usual.

# %%
a.argcross(b)

# %%
a.argcross(b, nested=True)

# %%markdown
# This method is good to use with ``unzip``, which separates the ``Table`` of tuples into a left half and a right half.

# %%
left, right = a.cross(b).unzip()
left, right

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
b = awkward.fromiter([[1, 2], [3], [4, 5, 6], [7]])
left, right = a.cross(b, nested=True).unzip()
left, right

# %%markdown
# This can be handy if a subsequent function takes two jagged arrays as arguments.

# %%
distance = round(abs(left - right), 1)
distance

# %%markdown
# Cross with ``nested=True``, followed by some calculation on the pairs and then some reducer, is a common pattern. Because of the ``nested=True`` and the reducer, the resulting array has the same structure as the original.

# %%
distance.min()

# %%
round(a + distance.min(), 1)

# %%markdown
# * ``pairs(nested=False)`` and ``argpairs(nested=False)``: returns jagged tuples representing the `self-join <https://en.wikipedia.org/wiki/Join_(SQL)#Self-join>`__ removing duplicates but not same-object pairs (i.e. a self-join with ``i1 <= i2``) for each inner array separately.

# %%
a = awkward.fromiter([["a", "b", "c"], [], ["d", "e"]])
a.pairs()

# %%markdown
# The "arg" and ``nested=True`` versions have the same meanings as with ``cross`` (above).

# %%
a.argpairs()

# %%
a.pairs(nested=True)

# %%markdown
# Just as with ``cross`` (above), this is good to combine with ``unzip`` and maybe a reducer.

# %%
a.pairs().unzip()

# %%markdown
# * ``distincts(nested=False)`` and ``argdistincts(nested=False)``: returns jagged tuples representing the `self-join <https://en.wikipedia.org/wiki/Join_(SQL)#Self-join>`__ removing duplicates and same-object pairs (i.e. a self-join with ``i1 < i2``) for each inner array separately.

# %%
a = awkward.fromiter([["a", "b", "c"], [], ["d", "e"]])
a.distincts()

# %%markdown
# The "arg" and ``nested=True`` versions have the same meanings as with ``cross`` (above).

# %%
a.argdistincts()

# %%
a.distincts(nested=True)

# %%markdown
# Just as with ``cross`` (above), this is good to combine with ``unzip`` and maybe a reducer.

# %%
a.distincts().unzip()

# %%markdown
# * ``choose(n)`` and ``argchoose(n)``: returns jagged tuples for distinct combinations of ``n`` elements from every inner array separately. ``array.choose(2)`` is the same as ``array.distincts()`` apart from order.

# %%
a = awkward.fromiter([["a", "b", "c"], [], ["d", "e"], ["f", "g", "h", "i", "j"]])
a

# %%
a.choose(2)

# %%
a.choose(3)

# %%
a.choose(4)

# %%markdown
# The "arg" version has the same meaning as ``cross`` (above), but there is no ``nested=True`` because of the order.

# %%
a.argchoose(2)

# %%markdown
# Just as with ``cross`` (above), this is good to combine with ``unzip`` and maybe a reducer.

# %%
a.choose(2).unzip()

# %%
a.choose(3).unzip()

# %%
a.choose(4).unzip()

# %%markdown
# * ``JaggedArray.zip(columns...)``: combines jagged arrays with the same structure into a single jagged array. The columns may be unnamed (resulting in a jagged array of tuples) or named with keyword arguments or dict keys (resulting in a jagged array of a table with named columns).

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
b = awkward.fromiter([[100, 200, 300], [], [400, 500]])
awkward.JaggedArray.zip(a, b)

# %%
awkward.JaggedArray.zip(x=a, y=b).tolist()

# %%
awkward.JaggedArray.zip({"x": a, "y": b}).tolist()

# %%markdown
# Not all of the arguments need to be jagged; those that aren't will be broadcasted to the right shape.

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
b = awkward.fromiter([100, 200, 300])
awkward.JaggedArray.zip(a, b)

# %%
awkward.JaggedArray.zip(a, 1000)

# %%markdown
# ## Properties and methods for tabular columns
#
# All awkward arrays have these methods, but they provide information about the first nested ``Table`` within a structure. If, for instance, the ``Table`` is within some structure that doesn't affect high-level type (e.g. ``IndexedArray``, ``ChunkedArray``, ``VirtualArray``), then the methods are passed through to the ``Table``. If it's nested within something that does change type, but can meaningfully pass on the call, such as ``MaskedArray``, then that's what they do.
#
# * ``columns``: the names of the columns at the first tabular level of depth.

# %%
a = awkward.fromiter([{"x": 1, "y": 1.1, "z": "one"}, {"x": 2, "y": 2.2, "z": "two"}, {"x": 3, "y": 3.3, "z": "three"}])
a.tolist()

# %%
a.columns

# %%
a = awkward.Table(x=[1, 2, 3],
                  y=[1.1, 2.2, 3.3],
                  z=awkward.Table(a=[4, 5, 6], b=[4.4, 5.5, 6.6]))
a.tolist()

# %%
a.columns

# %%
a["z"].columns

# %%
a.z.columns

# %%markdown
# * ``unzip()``: returns a tuple of projections through each of the columns (in the same order as the ``columns`` property).

# %%
a = awkward.fromiter([{"x": 1, "y": 1.1, "z": "one"}, {"x": 2, "y": 2.2, "z": "two"}, {"x": 3, "y": 3.3, "z": "three"}])
a.unzip()

# %%markdown
# The ``unzip`` method is the opposite of the ``Table`` constructor,

# %%
a = awkward.Table(x=[1, 2, 3],
                  y=[1.1, 2.2, 3.3],
                  z=awkward.fromiter(["one", "two", "three"]))
a.tolist()

# %%
a.unzip()

# %%markdown
# but it is also the opposite of ``JaggedArray.zip``.

# %%
b = awkward.JaggedArray.zip(x=awkward.fromiter([[1, 2, 3], [], [4, 5]]),
                            y=awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]]),
                            z=awkward.fromiter([["a", "b", "c"], [], ["d", "e"]]))
b.tolist()

# %%
b.unzip()

# %%markdown
# ``JaggedArray.zip`` produces a jagged array of ``Table`` whereas the ``Table`` constructor produces just a ``Table``, and these are distinct things, though they can both be inverted by the same function because row indexes and column indexes commute:

# %%
b[0]["y"]

# %%
b["y"][0]

# %%markdown
# So ``unzip`` turns a flat ``Table`` into a tuple of flat arrays (opposite of the ``Table`` constructor) and it turns a jagged ``Table`` into a tuple of jagged arrays (opposite of ``JaggedArray.zip``).
#
# * ``istuple``: an array of tuples is a special kind of ``Table``, one whose ``rowname`` is ``"tuple"`` and columns are ``"0"``, ``"1"``, ``"2"``, etc. If these conditions are met, ``istuple`` is ``True``; otherwise, ``False``.

# %%
a = awkward.Table(x=[1, 2, 3],
                  y=[1.1, 2.2, 3.3],
                  z=awkward.fromiter(["one", "two", "three"]))
a.tolist()

# %%
a.istuple

# %%
a = awkward.Table([1, 2, 3],
                  [1.1, 2.2, 3.3],
                  awkward.fromiter(["one", "two", "three"]))
a.tolist()

# %%
a.istuple

# %%markdown
# Even though the following tuples are inside of a jagged array, the first level of ``Table`` is a tuple, so ``istuple`` is ``True``.

# %%
b = awkward.JaggedArray.zip(awkward.fromiter([[1, 2, 3], [], [4, 5]]),
                            awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]]),
                            awkward.fromiter([["a", "b", "c"], [], ["d", "e"]]))
b

# %%
b.istuple

# %%markdown
# * ``i0`` through ``i9``: one of the two conditions for a ``Table`` to be a ``tuple`` is that columns are named ``"0"``, ``"1"``, ``"2"``, etc. Columns like that could be selected with ``["0"]`` at the risk of being misread as ``[0]``, and they could not be selected with attribute dot-access because pure numbers are not valid Python attributes. However, ``i0`` through ``i9`` are provided as shortcuts (overriding any columns with these exact names) for the first 10 tuple slots.

# %%
a = awkward.Table([1, 2, 3],
                  [1.1, 2.2, 3.3],
                  awkward.fromiter(["one", "two", "three"]))
a.tolist()

# %%
a.i0

# %%
a.i1

# %%
a.i2

# %%markdown
# * ``flattentuple()``: calling ``cross`` repeatedly can result in tuples nested within tuples; this flattens them at all levels, turning all ``(i, (j, k))`` into ``(i, j, k)``. Whereas ``array.flatten()`` removes one level of structure from the rows (losing information), ``array.flattentuple()`` removes all levels of structure from the columns (renaming them, but not losing information).

# %%
a = awkward.Table([1, 2, 3], [1, 2, 3], awkward.Table(awkward.Table([1, 2, 3], [1, 2, 3]), [1, 2, 3]))
a.tolist()

# %%
a.flattentuple().tolist()

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]])
b = awkward.fromiter([[100, 200], [300], [400, 500, 600], [700]])
c = awkward.fromiter([["a"], ["b", "c"], ["d"], ["e", "f"]])

# %%markdown
# The ``cross`` method internally calls ``flattentuples()`` if it detects that one of its arguments is the result of a ``cross``.

# %%
a.cross(b).cross(c).tolist()

# %%markdown
# ## Properties and methods for missing values
#
# All awkward arrays have these methods, but they provide information about the first nested ``MaskedArray`` within a structure. If, for instance, the ``MaskedArray`` is within some structure that doesn't affect high-level type (e.g. ``IndexedArray``, ``ChunkedArray``, ``VirtualArray``), then the methods are passed through to the ``MaskedArray``. If it's nested within something that does change type, but can meaningfully pass on the call, such as ``JaggedArray``, then that's what they do.
#
# * ``boolmask(maskedwhen=None)``: returns a Numpy array of booleans indicating which elements are missing ("masked") and which are not. If ``maskedwhen=True``, a ``True`` value in the Numpy array means missing/masked; if ``maskedwhen=False``, a ``False`` value in the Numpy array means missing/masked. If no value is passed (or ``None``), the ``MaskedArray``'s own ``maskedwhen`` property is used (which is by default ``True``). Non-``MaskedArrays`` are assumed to have a ``maskedwhen`` of ``True`` (the default).

# %%
a = awkward.fromiter([1, 2, None, 3, 4, None, None, 5])
a.boolmask()

# %%
a.boolmask(maskedwhen=False)

# %%markdown
# ``MaskedArrays`` inside of ``JaggedArrays`` or ``Tables`` are hidden.

# %%
a = awkward.fromiter([[1.1, None, 2.2], [], [3.3, 4.4, None, 5.5]])
a.boolmask()

# %%
a.flatten().boolmask()

# %%
a = awkward.fromiter([{"x": 1, "y": 1.1}, {"x": None, "y": 2.2}, {"x": None, "y": 3.3}, {"x": 4, "y": None}])
a.boolmask()

# %%
a.x.boolmask()

# %%
a.y.boolmask()

# %%markdown
# * ``ismasked`` and ``isunmasked``: shortcut for ``boolmask(maskedwhen=True)`` and ``boolmask(maskedwhen=False)`` as a property, which is more appropriate for analysis.

# %%
a = awkward.fromiter([1, 2, None, 3, 4, None, None, 5])
a.ismasked

# %%
a.isunmasked

# %%markdown
# * ``fillna(value)``: turn a ``MaskedArray`` into a non-``MaskedArray`` by replacing ``None`` with ``value``. Applies to the outermost ``MaskedArray``, but it passes through ``JaggedArrays`` and into all ``Table`` columns.

# %%
a = awkward.fromiter([1, 2, None, 3, 4, None, None, 5])
a.fillna(999)

# %%
a = awkward.fromiter([[1.1, None, 2.2], [], [3.3, 4.4, None, 5.5]])
a.fillna(999)

# %%
a = awkward.fromiter([{"x": 1, "y": 1.1}, {"x": None, "y": 2.2}, {"x": None, "y": 3.3}, {"x": 4, "y": None}])
a.fillna(999).tolist()

# %%markdown
# ## Functions for structure manipulation
#
# Only one structure-manipulation function (for now) is defined at top-level in awkward-array: ``awkward.concatenate``.
#
# * ``awkward.concatenate(arrays, axis=0)``: concatenate two or more ``arrays``. If ``axis=0``, the arrays are concatenated lengthwise (the resulting length is the sum of the lengths of each of the ``arrays``). If ``axis=1``, each inner array is concatenated: the input ``arrays`` must all be jagged with the same outer array length. (Values of ``axis`` greater than ``1`` are not yet supported.)

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
b = awkward.fromiter([[100, 200], [300], [400, 500, 600]])
awkward.concatenate([a, b])

# %%
awkward.concatenate([a, b], axis=1)

# %%
a = awkward.fromiter([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])
b = awkward.fromiter([{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}])
awkward.concatenate([a, b]).tolist()

# %%markdown
# If the arrays have different types, their concatenation is a ``UnionArray``.

# %%
a = awkward.fromiter([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}])
b = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
awkward.concatenate([a, b]).tolist()

# %%
a = awkward.fromiter([1, None, 2])
b = awkward.fromiter([None, 3, None])
awkward.concatenate([a, b])

# %%
import awkward, numpy
a = awkward.fromiter(["one", "two", "three"])
b = awkward.fromiter(["four", "five", "six"])
awkward.concatenate([a, b])

# %%
awkward.concatenate([a, b], axis=1)

# %%markdown
# # Functions for input/output and conversion
#
# Most of the functions defined at the top-level of the library are conversion functions.
#
# * ``awkward.fromiter(iterable, awkwardlib=None, dictencoding=False, maskedwhen=True)``: convert Python or JSON data into awkward arrays. Not a fast function: it necessarily involves a Python for loop. The ``awkwardlib`` determines which awkward module to use to make arrays (``awkward`` is the default, but ``awkward.numba`` and ``awkward.cpp`` are alternatives). If ``dictencoding`` is ``True``, bytes and strings will be "dictionary-encoded" in Arrow/Parquet terms—this is an ``IndexedArray`` in awkward. The ``maskedwhen`` parameter determines whether ``MaskedArrays`` have a mask that is ``True`` when data are missing or ``False`` when data are missing.

# %%
# We have been using this function all along, but why not another example?
complicated = awkward.fromiter([[1.1, 2.2, None, 3.3, None],
                                [4.4, [5.5]],
                                [{"x": 6, "y": {"z": 7}}, None, {"x": 8, "y": {"z": 9}}]
                               ])
complicated

# %%markdown
# The fact that this nested, row-wise data have been converted into columnar arrays can be seen by inspecting its ``layout``.

# %%
complicated.layout

# %%
for index, node in complicated.layout.items():
    if node.cls == numpy.ndarray:
        print("[{0:>13s}] {1}".format(", ".join(repr(i) for i in index), repr(node.array)))

# %%markdown
# The number of arrays in this object scales with the complexity of its data type, but not with the size of the dataset. If it were as complicated as it is now but billions of elements long, it would still contain 11 Numpy arrays, and operations on it would scale as Numpy scales. However, converting a billion Python objects to these 11 arrays would be a large up-front cost.
#
# More detail on the row-wise to columnar conversion process is given in `docs/fromiter.adoc <https://github.com/scikit-hep/awkward-array/blob/master/docs/fromiter.adoc>`__.

# %%markdown
# * ``load(file, awkwardlib=None, whitelist=awkward.persist.whitelist, cache=None, schemasuffix=".json")``: loads data from an "awkd" (special ZIP) file. This function is like ``numpy.load``, but for awkward arrays. If the file contains a single object, that object will be read immediately; if it has a collection of named arrays, it will return a loader that loads those arrays on demand. The ``awkwardlib`` determines the module to use to define arrays, the ``whitelist`` is where you can provide a list of functions that may be called in this process, ``cache`` is a global cache object assigned to ``VirtualArrays``, and ``schemasuffix`` determines the file name pattern to look for objects inside the ZIP file.
#
# * ``save(file, array, name=None, mode="a", compression=awkward.persist.compression, delimiter="-", suffix=".raw", schemasuffix=".json")``: saves data to an "awkd" (special ZIP) file. This function is like ``numpy.savez`` and is the reverse of ``load`` (above). The ``array`` may be a single object or a dict of named arrays, the ``name`` is a name to use inside the file, ``mode="a"`` means create or append to an existing file, refusing to overwrite data while ``mode="w"`` overwrites data, ``compression`` is a compression policy (set of rules determining which arrays to compress and how), and the rest of the arguments determine file names within the ZIP: ``delimiter`` between name components, ``suffix`` for array data, and ``schemasuffix`` for the schemas that tell ``load`` how to find all other data.

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
b = awkward.fromiter([[1.1, 2.2, None, 3.3, None],
                      [4.4, [5.5]],
                      [{"x": 6, "y": {"z": 7}}, None, {"x": 8, "y": {"z": 9}}]
                     ])

# %%
awkward.save("single.awkd", a, mode="w")

# %%
awkward.load("single.awkd")

# %%
awkward.save("multi.awkd", {"a": a, "b": b}, mode="w")

# %%
multi = awkward.load("multi.awkd")

# %%
multi["a"]

# %%
multi["b"]

# %%markdown
# Only ``save`` has a ``compression`` parameter because only the writing process gets to decide how arrays are compressed. We don't use ZIP's built-in compression, but use Python compression functions and encode the choice in the metadata. If ``compression=True``, all arrays will be compressed with zlib; if ``compression=False``, ``None``, or ``[]``, none will. In general, ``compression`` is a list of rules; the first rule that is satisfied by a given array uses the specified compress/decompress pair of functions. Here's the default policy:

# %%
awkward.persist.compression

# %%markdown
# The default policy has only one rule. If any array has a minimum size (``minsize``) of 8 kB (``8192`` bytes), a numeric type (``array.dtype.type``) that is a subclass of ``numpy.bool_``, ``bool``, or ``numpy.integer``, and is in any awkward-array context (``JaggedArray.starts``, ``MaskedArray.mask``, etc.), then it will be compressed with ``zip.compress`` and decompressed with ``('zlib', 'decompress')``. The compression function is given as an object—the Python function that will be called to transform byte strings into compressed byte strings—but the decompression function is given as a location in Python's namespace: a tuple of nested objects, the first of which is a fully qualified module name (submodules separated by dots). This is because only the *location* of the decompression function needs to be written to the file.
#
# The saved awkward array consists of a collection of byte strings for Numpy arrays (2 for object ``a`` and 11 for object ``b``, above) and JSON-formatted metadata that reconstructs the nested hierarchy of awkward classes around those Numpy arrays. This metadata includes information such as which byte strings should be decompressed and how, but also which awkward constructors to call to fit everything together. As such, the JSON metadata is code, a limited language without looping or function definitions (i.e. not Turing complete) but with the ability to call any Python function.
#
# Using a mini-language as metadata gives us great capacity for backward and forward compatibility (new or old ways of encoding things are simply calling different functions), but it does raise the danger of malicious array files calling unwanted Python functions. For this reason, ``load`` refuses to call any functions not specified in a ``whitelist``. The default whitelist consists of functions known to be safe:

# %%
awkward.persist.whitelist

# %%markdown
# The format of each item in the whitelist is a list of nested objects, the first of which being a fully qualified module name (submodules separated by dots). For instance, in the ``awkward.arrow`` submodule, there is a class named ``_ParquetFile`` and it has a static method ``fromjson`` that is deemed to be safe. Patterns of safe names are can be wildcarded, such as ``['awkward', '*Array']`` and ``['uproot_methods.classes.*']``.
#
# You can add your own functions, and forward compatibility (using data made by a new version in an old version of awkward-array) often dictates that you must add a function manually. The error message explains how to do this.
#
# The same serialization format is used when you pickle an awkward array or save it in an HDF5 file. More detail on the metadata mini-language is given in `docs/serialization.adoc <https://github.com/scikit-hep/awkward-array/blob/master/docs/serialization.adoc>`__.

# %%markdown
# * ``hdf5(group, awkwardlib=None, compression=awkward.persist.compression, whitelist=awkward.persist.whitelist, cache=None)``: wrap a ``h5py.Group`` as an awkward-aware group, to save awkward arrays to HDF5 files and to read them back again. The options have the same meaning as ``load`` and ``save``.
#
# Unlike "awkd" (special ZIP) files, HDF5 files can be written and overwritten like a database, rather than write-once files.

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
b = awkward.fromiter([[1.1, 2.2, None, 3.3, None],
                      [4.4, [5.5]],
                      [{"x": 6, "y": {"z": 7}}, None, {"x": 8, "y": {"z": 9}}]
                     ])

# %%
import h5py
f = h5py.File("awkward.hdf5", "w")
f

# %%
g = awkward.hdf5(f)
g

# %%
g["array"] = a

# %%
g["array"]

# %%
del g["array"]

# %%
g["array"] = b

# %%
g["array"]

# %%markdown
# The HDF5 format does not include columnar representations of arbitrary nested data, as awkward-array does, so what we're actually storing are plain Numpy arrays and the metadata necessary to reconstruct the awkward array.

# %%
# Reopen file, without wrapping it as awkward.hdf5 this time.
f = h5py.File("awkward.hdf5", "r")
f

# %%
f["array"]

# %%
f["array"].keys()

# %%markdown
# The "schema.json" array is the JSON metadata, containing directives like ``{"call": ["awkward", "JaggedArray", "fromcounts"]}`` and ``{"read": "1"}`` meaning the array named ``"1"``, etc.

# %%
import json
json.loads(f["array"]["schema.json"][:].tostring())

# %%markdown
# Without awkward-array, these objects can't be meaningfully read back from the HDF5 file.

# %%markdown
# * ``awkward.fromarrow(arrow, awkwardlib=None)``: convert an `Apache Arrow <https://arrow.apache.org>`__ formatted buffer to an awkward array (zero-copy). The ``awkwardlib`` parameter has the same meaning as above.
#
# * ``awkward.toarrow(array)``: convert an awkward array to an Apache Arrow buffer, if possible (involving a data copy, but no Python loops).

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
b = awkward.fromiter([[1.1, 2.2, None, 3.3, None],
                      [4.4, [5.5]],
                      [{"x": 6, "y": {"z": 7}}, None, {"x": 8, "y": {"z": 9}}]
                     ])

# %%
awkward.toarrow(a)

# %%
awkward.fromarrow(awkward.toarrow(a))

# %%
awkward.toarrow(b)

# %%
awkward.fromarrow(awkward.toarrow(b))

# %%markdown
# Unlike HDF5, Arrow is capable of columnar jagged arrays, nullable values, nested structures, etc. If you save an awkward array in Arrow format, someone else can read it without the awkward-array library. There are a few awkward array classes that don't have an Arrow equivalent, though. Below is a list of all translations.
#
# * Numpy array → Arrow `BooleanArray <https://arrow.apache.org/docs/python/generated/pyarrow.BooleanArray.html>`__, `IntegerArray <https://arrow.apache.org/docs/python/generated/pyarrow.IntegerArray.html>`__, or `FloatingPointArray <https://arrow.apache.org/docs/python/generated/pyarrow.FloatingPointArray.html>`__.
# * ``JaggedArray`` → Arrow `ListArray <https://arrow.apache.org/docs/python/generated/pyarrow.ListArray.html>`__.
# * ``StringArray`` → Arrow `StringArray <https://arrow.apache.org/docs/python/generated/pyarrow.StringArray.html>`__.
# * ``Table`` → Arrow `Table <https://arrow.apache.org/docs/python/generated/pyarrow.Table.html>`__ at top-level, but an Arrow `StructArray <https://arrow.apache.org/docs/python/generated/pyarrow.StructArray.html>`__ if nested.
# * ``MaskedArray`` → missing data mask (nullability in Arrow is an array attribute, rather than an array wrapper).
# * ``IndexedMaskedArray`` → unfolded into a simple mask before the Arrow translation.
# * ``IndexedArray`` → Arrow `DictionaryArray <https://arrow.apache.org/docs/python/generated/pyarrow.DictionaryArray.html>`__.
# * ``SparseArray`` → converted to a dense array before the Arrow translation.
# * ``ObjectArray`` → Pythonic interpretation is discarded before the Arrow translation.
# * ``UnionArray`` → Arrow dense `UnionArray <https://arrow.apache.org/docs/python/generated/pyarrow.UnionArray.html>`__ if possible, sparse UnionArray if necessary.
# * ``ChunkedArray`` (including ``AppendableArray``) → Arrow `RecordBatches <https://arrow.apache.org/docs/python/generated/pyarrow.RecordBatch.html>`__, but only at top-level: nested ``ChunkedArrays`` cannot be converted.
# * ``VirtualArray`` → array gets materialized before the Arrow translation (i.e. the lazy-loading is not preserved).

# %%markdown
# Since Arrow is an in-memory format, both ``toarrow`` and ``fromarrow`` are side-effect-free functions with a return value. Functions that write to files have a side-effect (the state of your disk changing) and no return value. Once you've made your Arrow buffer, you have to figure out what to do with it. (You may want to `write it to a stream <https://arrow.apache.org/docs/python/ipc.html>`__ for interprocess communication.)

# %%markdown
# * ``awkward.fromparquet(where, awkwardlib=None)``: reads from a Parquet file (at filename/URI ``where``) into an awkward array, through pyarrow. The ``awkwardlib`` parameter has the same meaning as above.
#
# * ``awkward.toparquet(where, array, schema=None)``: writes an awkward array to a Parquet file (at filename/URI ``where``), through pyarrow. The Parquet ``schema`` may be inferred from the awkward array or explicitly specified.
#
# Like Arrow and unlike HDF5, Parquet natively stores complex data structures in a columnar format and doesn't need to be wrapped by an interpretation layer like ``awkward.hdf5``. Like HDF5 and unlike Arrow, Parquet is a file format, intended for storage.

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
b = awkward.fromiter([[1.1, 2.2, None, 3.3, None],
                      [4.4, [5.5]],
                      [{"x": 6, "y": {"z": 7}}, None, {"x": 8, "y": {"z": 9}}]
                     ])

# %%
awkward.toparquet("dataset.parquet", a)

# %%
a2 = awkward.fromparquet("dataset.parquet")
a2

# %%markdown
# Notice that we get a ``ChunkedArray`` back. This is because ``awkward.fromparquet`` is lazy-loading the Parquet file, which might be very large (not in this case, obviously). It's actually a ``ChunkedArray`` (one `row group <https://parquet.apache.org/documentation/latest/#unit-of-parallelization>`__ per chunk) of ``VirtualArrays``, and each ``VirtualArray`` is read when it is accessed (for instance, to print it above).

# %%
a2.chunks

# %%
a2.chunks[0].array

# %%markdown
# The next layer of new structure is that the jagged array is bit-masked. Even though none of the values are nullable, this is an artifact of the way Parquet formats columnar data.

# %%
a2.chunks[0].array.content

# %%
a2.layout

# %%markdown
# Fewer types can be written to Parquet files than Arrow buffers, since pyarrow does not yet have a complete Arrow → Parquet transformation.

# %%
try:
    awkward.toparquet("dataset2.parquet", b)
except Exception as err:
    print(type(err), str(err))

# %%markdown
# * ``awkward.topandas(array, flatten=False)``: convert the array into a Pandas DataFrame (if tabular) or a Pandas Series (otherwise). If ``flatten=False``, wrap the awkward arrays as a new Pandas extension type (not fully implemented). If ``flatten=True``, convert the jaggedness and nested tables into row and column ``pandas.MultiIndex`` without introducing any new types (not always possible).

# %%
a = awkward.Table(x=awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]),
                  y=awkward.fromiter([100, 200, 300, 400]))
df = awkward.topandas(a)
df

# %%
df.x

# %%markdown
# Note that the ``dtype`` is ``awkward``. The array has not been converted into Numpy ``dtype=object`` (which would imply a performance loss); it has been wrapped as a container that Pandas recognizes. You can get the awkward array back the same way you would a Numpy array:

# %%
df.x.values

# %%markdown
# (``JaggedSeries`` is a thin wrapper on ``JaggedArray``; they behave the same way.)
#
# The value of this is that awkward slice semantics can be applied to data in Pandas.

# %%
df[1:]

# %%
df.x[df.x.values.counts > 0]

# %%markdown
# However, Pandas has a (limited) way of handling jaggedness and nested tables, with ``pandas.MultiIndex`` rows and columns, respectively.

# %%
# Nested tables become MultiIndex-valued column names.
array = awkward.fromiter([{"a": {"b": 1, "c": {"d": [2]}}, "e": 3},
                          {"a": {"b": 4, "c": {"d": [5, 5.1]}}, "e": 6},
                          {"a": {"b": 7, "c": {"d": [8, 8.1, 8.2]}}, "e": 9}])
df = awkward.topandas(array, flatten=True)
df

# %%
# Jagged arrays become MultiIndex-valued rows (index).
array = awkward.fromiter([{"a": 1, "b": [[2.2, 3.3, 4.4], [], [5.5, 6.6]]},
                          {"a": 10, "b": [[1.1], [2.2, 3.3], [], [4.4]]},
                          {"a": 100, "b": [[], [9.9]]}])
df = awkward.topandas(array, flatten=True)
df

# %%markdown
# The advantage of this is that no new column types are introduced, and Pandas already has functions for managing structure in its ``MultiIndex``. For instance, this structure can be unstacked into Pandas's columns.

# %%
df.unstack()

# %%
df.unstack().unstack()

# %%markdown
# It is also possible to get `Pandas Series and DataFrames through Arrow <https://arrow.apache.org/docs/python/pandas.html>`__, though this doesn't handle jagged arrays well: they get converted into Numpy ``dtype=object`` arrays.

# %%
df = awkward.toarrow(array).to_pandas()
df

# %%
df.b

# %%
df.b[0]

# %%markdown
# # High-level types
#
# The high-level type of an array describes its characteristics in terms of what it *represents*, a *logical* view of the data. By contrast, the layouts (below) describe the nested arrays themselves, a *physical* view of the data.
#
# The logical view of Numpy arrays is described in terms of ``shape`` and ``dtype``. The awkward type of a Numpy array is presented a little differently.

# %%
a = numpy.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
t = awkward.type.fromarray(a)
t

# %%markdown
# Above is the object-form of the high-level type and object that ``takes`` arguments ``to`` return values.

# %%
t.takes

# %%
t.to

# %%
t.to.to

# %%markdown
# High-level type objects also have a printable form for human readability.

# %%
print(t)

# %%markdown
# The above should be read like a function's data type: ``argument type -> return type`` for the function that takes an index in square brackets and returns something else. For example, the first ``[0, 3)`` means that you could put any non-negative integer less than ``3`` in square brackets after the array, like this:

# %%
a[2]

# %%markdown
# The second ``[0, 2)`` means that the next argument can be any non-negative integer less than ``2``.

# %%
a[2][1]

# %%markdown
# And then you have a Numpy ``dtype``.
#
# The reason high-level types are expressed like this, instead of Numpy ``shape`` and ``dtype`` is to generalize to arbitrary objects.

# %%
a = awkward.fromiter([{"x": 1, "y": []}, {"x": 2, "y": [1.1, 2.2]}, {"x": 3, "y": [1.1, 2.2, 3.3]}])
print(a.type)

# %%markdown
# In the above, you could call ``a[2]["x"]`` to get ``3`` or ``a[2]["y"][1]`` to get ``2.2``, but the types and even number of allowed arguments depend on which path you take. Numpy's ``shape`` and ``dtype`` have no equivalent.
#
# Also in the above, the allowed argument for the jagged array is specified as ``[0, inf)``, which doesn't literally mean any value up to infinity is allowed—the constraint simply isn't specific because it depends on the details of the jagged array. Even specifying the maximum length of any sublist (``a["y"].counts.max()``) would require a calculation that scales with the size of the dataset, which can be infeasible in some cases. Instead, ``[0, inf)`` simply means "jagged."
#
# Fixed-length arrays inside of ``JaggedArrays`` or ``Tables`` are presented with known upper limits:

# %%
a = awkward.Table(x=[[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]],
                  y=awkward.fromiter([[1, 2, 3], [], [4, 5]]))
print(a.type)

# Whereas each value of a ``Table`` row (`product type <https://en.wikipedia.org/wiki/Product_type>`__) contains a member of every one of its fields, each value of a ``UnionArray`` item (`sum type <https://en.wikipedia.org/wiki/Tagged_union>`__) contains a member of exactly one of its possibilities. The distinction is drawn as the lack or presence of a vertical bar (meaning "or": ``|``).

# %%
a = awkward.fromiter([{"x": 1, "y": "one"}, {"x": 2, "y": "two"}, {"x": 3, "y": "three"}])
print(a.type)

# %%
a = awkward.fromiter([1, 2, 3, "four", "five", "six"])
print(a.type)

# %%markdown
# The parenthesis is to keep ``Table`` fields from being mixed up with ``UnionArray`` possibilities.

# %%
a = awkward.fromiter([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": "three"}, {"x": 4, "y": "four"}])
print(a.type)

# %%markdown
# As in mathematics, products and the adjacency operator take precedence over sums.

# %%
a = awkward.fromiter([1, 2, 3, {"x": 4.4, "y": "four"}, {"x": 5.5, "y": "five"}, {"x": 6.6, "y": "six"}])
print(a.type)

# %%markdown
# Missing data, represented by ``MaskedArrays``, ``BitMaskedArrays``, or ``IndexedMaskedArrays``, are called "option types" in the high-level type language.

# %%
a = awkward.fromiter([1, 2, 3, None, None, 4, 5])
print(a.type)

# %%
# Inner arrays could be missing values.
a = awkward.fromiter([[1.1, 2.2, 3.3], None, [4.4, 5.5]])
print(a.type)

# %%
# Numbers in those arrays could be missing values.
a = awkward.fromiter([[1.1, 2.2, None], [], [4.4, 5.5]])
print(a.type)

# %%markdown
# Cross-references and cyclic references are expressed in awkward type objects by creating the same graph structure among the type objects as the arrays. Thus,

# %%
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

# %%markdown
# In the print-out, labels (``T0 :=``, ``T1 :=``, ``T2 :=``) are inserted to indicate where cross-references begin and end.

# %%
print(tree.type)

# %%markdown
# The ``ObjectArray`` class turns awkward array structures into Python objects on demand. From an analysis point of view, the elements of the array *are* Python objects, and that is reflected in the type.

# %%
class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __repr__(self):
        return "Point({0}, {1})".format(self.x, self.y)

a = awkward.fromiter([Point(0, 0), Point(3, 2), Point(1, 1), Point(2, 4), Point(0, 0)])
a

# %%
print(a.type)

# %%markdown
# In summary,
#
# * each element of a Numpy ``shape`` like ``(i, j, k)`` becomes a functional argument: ``[0, i) -> [0, j) -> [0, k)``;
# * high-level types terminate on Numpy ``dtypes`` or ``ObjectArray`` functions;
# * columns of a ``Table`` are presented adjacent to one another: the type is field 1 *and* field 2 *and* field 3, etc.;
# * possibilities of a ``UnionArray`` are separated by vertical bars ``|``: the type is possibility 1 *or* possibility 2 *or* possibility 3, etc.;
# * nullable types are indicated by a question mark;
# * cross-references and cyclic references are maintained in the type objects, printed with labels.

# %%markdown
# # Low-level layouts
#
# The layout of an array describes how it is constructed in terms of Numpy arrays and other parameters. It has more information than a high-level type (above), more that would typically be needed for data analysis, but very necessary for data engineering.
#
# A ``Layout`` object is a mapping from position tuples to ``LayoutNodes``. The screen representation is sufficient for reading.

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
t = a.layout
t

# %%
t[2]

# %%
t[2].array

# %%
a = awkward.fromiter([[[1.1, 2.2], [3.3]], [], [[4.4, 5.5]]])
t = a.layout
t

# %%
t[2]

# %%
t[2].array

# %%
t[2, 2].array

# %%markdown
# Classes like ``IndexedArray``, ``SparseArray``, ``ChunkedArray``, ``AppendableArray``, and ``VirtualArray`` don't change the high-level type of an array, but they do change the layout. Consider, for instance, an array made with ``awkward.fromiter`` and an array read by ``awkward.fromparquet``.

# %%
a = awkward.fromiter([[1.1, 2.2, None, 3.3], [], None, [4.4, 5.5]])

# %%
awkward.toparquet("tmp.parquet", a)

# %%
b = awkward.fromparquet("tmp.parquet")

# %%markdown
# At first, it terminates at ``VirtualArray`` because the data haven't been read—we don't know what arrays are associated with it.

# %%
b.layout

# %%markdown
# But after reading,

# %%
b

# %%markdown
# The layout shows that it has more structure than ``a``.

# %%
b.layout

# %%
a.layout

# %%markdown
# However, they have the same high-level type.

# %%
print(b.type)

# %%
print(a.type)

# %%markdown
# Cross-references and cyclic references are also encoded in the ``layout``, as references to previously seen indexes.

# %%
tree.layout

# %%markdown
# # Details of each array class

# %%markdown
# ## JaggedArray: variable-length lists

# %%markdown
# ## Table: nested records

# %%markdown
# ## MaskedArray: nullable data

# %%markdown
# ## UnionArray: heterogeneous lists

# %%markdown
# ## ObjectArray and Methods: interactivity in Python

# %%markdown
# ## StringArray: strings

# %%markdown
# ## IndexedArray: cross-references and circular references

# %%markdown
# ## SparseArray: sparse data

# %%markdown
# ## ChunkedArray: non-contiguous data

# %%markdown
# ## AppendableArray: efficiently add rows of data

# %%markdown
# ## VirtualArray: data on demand

# %%markdown
# # Applications

# %%markdown
# ## Decision tree as an awkward array

# %%markdown
# ## Mixed-source data with persistvirtual

# %%markdown
# ## Using Pandas with awkward arrays

# %%markdown
# ## Using Numba with awkward arrays

# %%markdown
# ## Flattening awkard arrays for machine learning
