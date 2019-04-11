#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import awkward.array.indexed
from .base import NumbaMethods

class IndexedArrayNumba(NumbaMethods, awkward.array.indexed.IndexedArray):
    pass

class SparseArrayNumba(NumbaMethods, awkward.array.indexed.SparseArray):
    pass
