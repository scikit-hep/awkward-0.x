#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import distutils.version

import numpy
if distutils.version.LooseVersion(numpy.__version__) < distutils.version.LooseVersion("1.13.1"):
    raise ImportError("Numpy 1.13.1 or later required")

import warnings
warnings.warn(
    """Consider switching from 'awkward' to 'awkward1', since the new interface will become the default later this year (2020).

    pip install -U awkward1

In Python:

    >>> import awkward1 as ak
    >>> new_style_array = ak.from_awkward0(old_style_array)
    >>> old_style_array = ak.to_awkward0(new_style_array)
""",
    DeprecationWarning
)

from awkward.array.base import AwkwardArray
from awkward.array.chunked import ChunkedArray, AppendableArray
from awkward.array.indexed import IndexedArray, SparseArray
from awkward.array.jagged import JaggedArray
from awkward.array.masked import MaskedArray, BitMaskedArray, IndexedMaskedArray
from awkward.array.objects import Methods, ObjectArray, StringArray
from awkward.array.table import Table
from awkward.array.union import UnionArray
from awkward.array.virtual import VirtualArray

def concatenate(arrays, axis=0):
    return AwkwardArray.concatenate(arrays, axis=axis)

from awkward.generate import fromiter

from awkward.persist import serialize, deserialize, save, load, hdf5

from awkward.arrow import toarrow, fromarrow, toparquet, fromparquet
from awkward.util import topandas

# convenient access to the version number
from awkward.version import __version__

__all__ = ["numpy", "AwkwardArray", "ChunkedArray", "AppendableArray", "IndexedArray", "SparseArray", "JaggedArray", "MaskedArray", "BitMaskedArray", "IndexedMaskedArray", "Methods", "ObjectArray", "Table", "UnionArray", "VirtualArray", "StringArray", "fromiter", "serialize", "deserialize", "save", "load", "hdf5", "toarrow", "fromarrow", "toparquet", "fromparquet", "topandas", "__version__"]

__path__ = __import__("pkgutil").extend_path(__path__, __name__)
