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
# This decomposition is variously called "splitting" (by particle physicists) or "shredding" or "striping" (by `Arrow <https://arrow.apache.org>`__ and `Parquet <https://parquet.apache.org/>`__ developers).
#
# Most importantly, you can do Numpy vectorized operations,

# %%
a = awkward.fromiter([[1, 4, None], [], [9, {"x": 16, "y": 25}]])
numpy.sqrt(a).tolist()

# %%markdown
# multidimensional slicing,

# %%
a = awkward.fromiter([[1], [], [2, 3], [4, 5, [6]]])
a[2:, 1:]

# %%markdown
# and broadcasting,

# %%
a = awkward.fromiter([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
a * 10

# %%markdown
# including jagged-only extensions of these concepts,

# %%
a + numpy.array([100, 200, 300])

# %%
a.sum()

# %%markdown
# This tutorial starts with a data analyst's perspective—using awkward-array to manipulate data—and then focuses on each awkward-array type. Like Numpy, the features of this library are deliberately simple, yet compositional. Any awkward array may be the content of any other awkward array. Building and analyzing complex data structures is up to you.

# %%markdown
# # Example data analysis: NASA exoplanets
#
# HERE...
