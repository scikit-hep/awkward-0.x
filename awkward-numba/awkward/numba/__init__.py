#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import numpy

from awkward.numba.array.base import NumbaMethods
from awkward.numba.array.chunked import ChunkedArrayNumba as ChunkedArray
from awkward.numba.array.chunked import AppendableArrayNumba as AppendableArray
from awkward.numba.array.indexed import IndexedArrayNumba as IndexedArray
from awkward.numba.array.indexed import SparseArrayNumba as SparseArray
from awkward.numba.array.jagged import JaggedArrayNumba as JaggedArray
from awkward.numba.array.masked import MaskedArrayNumba as MaskedArray
from awkward.numba.array.masked import BitMaskedArrayNumba as BitMaskedArray
from awkward.numba.array.masked import IndexedMaskedArrayNumba as IndexedMaskedArray
from awkward.numba.array.objects import MethodsNumba as Methods
from awkward.numba.array.objects import ObjectArrayNumba as ObjectArray
from awkward.numba.array.objects import StringArrayNumba as StringArray
from awkward.numba.array.table import TableNumba as Table
from awkward.numba.array.union import UnionArrayNumba as UnionArray
from awkward.numba.array.virtual import VirtualArrayNumba as VirtualArray

import awkward.generate
def fromiter(iterable, awkwardlib=None, **options):
    if awkwardlib is None:
        awkwardlib = "awkward.numba"
    return awkward.generate.fromiter(iterable, awkwardlib=awkwardlib, **options)

# convenient access to the version number
from awkward.version import __version__

__all__ = ["numpy", "AwkwardArray", "ChunkedArray", "AppendableArray", "IndexedArray", "SparseArray", "JaggedArray", "MaskedArray", "BitMaskedArray", "IndexedMaskedArray", "Methods", "ObjectArray", "Table", "UnionArray", "VirtualArray", "StringArray", "fromiter", "__version__"]
