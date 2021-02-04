#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-0.x/blob/master/LICENSE

import distutils.version

import numpy
if distutils.version.LooseVersion(numpy.__version__) < distutils.version.LooseVersion("1.13.1"):
    raise ImportError("Numpy 1.13.1 or later required")

from awkward0.array.base import AwkwardArray
from awkward0.array.chunked import ChunkedArray, AppendableArray
from awkward0.array.indexed import IndexedArray, SparseArray
from awkward0.array.jagged import JaggedArray
from awkward0.array.masked import MaskedArray, BitMaskedArray, IndexedMaskedArray
from awkward0.array.objects import Methods, ObjectArray, StringArray
from awkward0.array.table import Table
from awkward0.array.union import UnionArray
from awkward0.array.virtual import VirtualArray

def concatenate(arrays, axis=0):
    return AwkwardArray.concatenate(arrays, axis=axis)

from awkward0.generate import fromiter

from awkward0.persist import serialize, deserialize, save, load, hdf5

from awkward0.arrow import toarrow, fromarrow, toparquet, fromparquet
from awkward0.util import topandas

# convenient access to the version number
from awkward0.version import __version__

__all__ = ["numpy", "AwkwardArray", "ChunkedArray", "AppendableArray", "IndexedArray", "SparseArray", "JaggedArray", "MaskedArray", "BitMaskedArray", "IndexedMaskedArray", "Methods", "ObjectArray", "Table", "UnionArray", "VirtualArray", "StringArray", "fromiter", "serialize", "deserialize", "save", "load", "hdf5", "toarrow", "fromarrow", "toparquet", "fromparquet", "topandas", "__version__"]

__path__ = __import__("pkgutil").extend_path(__path__, __name__)
