#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import distutils.version

import numpy
if distutils.version.LooseVersion(numpy.__version__) < distutils.version.LooseVersion("1.13.1"):
    raise ImportError("Numpy 1.13.1 or later required")

from awkward.array.chunked import ChunkedArray, AppendableArray
from awkward.array.indexed import IndexedArray, SparseArray
from awkward.array.jagged import JaggedArray
from awkward.array.masked import MaskedArray, BitMaskedArray, IndexedMaskedArray
from awkward.array.objects import Methods, ObjectArray, StringArray
from awkward.array.table import Table
from awkward.array.union import UnionArray
from awkward.array.virtual import VirtualArray

from awkward.generate import fromiter, fromiterchunks

from awkward.persist import serialize, deserialize, save, load, hdf5

from awkward.arrow import toarrow, fromarrow, toparquet, fromparquet

# convenient access to the version number
from awkward.version import __version__

__all__ = ["numpy", "ChunkedArray", "AppendableArray", "IndexedArray", "SparseArray", "JaggedArray", "MaskedArray", "BitMaskedArray", "IndexedMaskedArray", "Methods", "ObjectArray", "Table", "UnionArray", "VirtualArray", "StringArray", "fromiter", "fromiterchunks", "serialize", "deserialize", "save", "load", "hdf5", "toarrow", "fromarrow", "toparquet", "fromparquet", "__version__"]

__path__ = __import__("pkgutil").extend_path(__path__, __name__)
