# %%markdown
#
# Numpy is great for exploratory data analysis because it encourages the analyst to calculate one operation at a time, rather than one datum at a time. To compute an expression like
#
# .. math::
#
#     m = \\sqrt{(E_1 + E_2)^2 - (p_{x1} + p_{x2})^2 - (p_{y1} + p_{y2})^2 - (p_{z1} + p_{z2})^2}
#
# the analyst might first compute :math:`\\sqrt{(p_{x1} + p_{x2})^2 + (p_{y1} + p_{y2})^2}` for all data (which has a meaning: :math:`p_T`), then compute :math:`\\sqrt{{p_T}^2 + (p_{z1} + p_{z2})^2}` for all data (which has a meaning: :math:`|p|`), then compute everything as :math:`\\sqrt{(E_1 + E_2)^2 - |p|^2}`. Performing each step on all data allows the analyzer to plot and cross-check each distribution, to discover surprises as early as possible.
# 
# This order of data processing is called "columnar" in the sense that a dataset may be visualized as a table in which rows are repeated measurements and columns are the different measurable quantities (same layout as `Pandas DataFrames <http://pandas.pydata.org>`__). It is also called "vectorized" in that a Single (Python) Instruction is applied to Multiple Data (SIMD). Numpy can be hundreds to thousands of times faster than pure Python because it avoids the overhead of handling Python instructions in the loop over data. Many data processing languages—R, MATLAB, IDL, all the way back to APL—consist of an interactive interpreter with fast, vectorized math.
#
# 
