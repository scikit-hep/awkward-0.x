 #!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import numpy

import distutils.version
import pandas

if distutils.version.LooseVersion(pandas.__version__) < distutils.version.LooseVersion('0.23.0'):
    raise ImportError('Pandas version 0.23.0 or later required')

from awkward.pandas.array.jagged import JaggedArrayPandas as JaggedArray
from awkward.array.chunked import ChunkedArray, AppendableArray
from awkward.array.indexed import IndexedArray, SparseArray
from awkward.array.masked import MaskedArray, BitMaskedArray, IndexedMaskedArray
from awkward.array.objects import Methods, ObjectArray, StringArray
from awkward.array.table import Table
from awkward.array.union import UnionArray
from awkward.array.virtual import VirtualArray

import awkward.generate

def fromiter(iterable, awkwardlib=None, **options):
    if awkwardlib is None:
        awkwardlib = "awkward.pandas"
    return awkward.generate.fromiter(iterable, awkwardlib=awkwardlib, **options)

def fromiterchunks(iterable, chunksize, awkwardlib=None, **options):
    if awkwardlib is None:
        awkwardlib = "awkward.pandas"
    return awkward.generate.fromiterchunks(iterable, chunksize, awkwardlib=awkwardlib, **options)

__all__ = ["numpy", "ChunkedArray", "AppendableArray", "IndexedArray", "SparseArray", "JaggedArray", "MaskedArray", "BitMaskedArray", "IndexedMaskedArray", "Methods", "ObjectArray", "Table", "UnionArray", "VirtualArray", "StringArray", "fromiter", "fromiterchunks", "serialize", "deserialize", "save", "load", "hdf5", "toarrow", "fromarrow", "toparquet", "fromparquet", "__version__"]

__path__ = __import__("pkgutil").extend_path(__path__, __name__)
