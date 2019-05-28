#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

######################################################################## Numba-accelerated interface

import awkward.array.base
import awkward.array.jagged
from .base import CppMethods

from . import _jagged

class JaggedArrayCpp(CppMethods, awkward.array.jagged.JaggedArray):
    @classmethod
    def offsets2parents(cls, offsets):
        return getattr(_jagged, "offsets2parents_" + str(offsets.dtype))(offsets)

    def counts2offsets(cls, counts):
        return getattr(_jagged, "counts2offsets_" + str(counts.dtype))(counts)

    def startsstops2parents(cls, starts, stops):
        if starts.dtype is not stops.dtype:
            raise ValueError("starts and stops must be the same type")
        return getattr(_jagged, "startsstops2parents_" + str(stops.dtype))(starts, stops)

    def parents2startsstops(cls, parents, length=None):
        if length is None:
            length = parents.max() + 1
        return getattr(_jagged, "parents2startsstops_" + str(stops.dtype))(parents, length)
