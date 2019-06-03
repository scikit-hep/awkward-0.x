#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

######################################################################## Numba-accelerated interface

import awkward.array.base
import awkward.array.jagged
from .base import CppMethods
from ._jagged import JaggedArraySrc

class JaggedArrayCpp(CppMethods, JaggedArraySrc, awkward.array.jagged.JaggedArray):
    @classmethod
    def offsets2parents(cls, offsets):
        return getattr(JaggedArraySrc, "offsets2parents_" + str(offsets.dtype))(offsets)

    @classmethod
    def counts2offsets(cls, counts):
        return getattr(JaggedArraySrc, "counts2offsets_" + str(counts.dtype))(counts)

    @classmethod
    def startsstops2parents(cls, starts, stops):
        if starts.dtype is not stops.dtype:
            raise ValueError("starts and stops must be the same type")
        return getattr(JaggedArraySrc, "startsstops2parents_" + str(stops.dtype))(starts, stops)

    @classmethod
    def parents2startsstops(cls, parents, length=None):
        if length is None:
            length = parents.max() + 1
        return getattr(JaggedArraySrc, "parents2startsstops_" + str(parents.dtype))(parents, length)

    @classmethod
    def uniques2offsetsparents(cls, uniques):
        return getattr(JaggedArraySrc, "uniques2offsetsparents_" + str(uniques.dtype))(uniques)

    
