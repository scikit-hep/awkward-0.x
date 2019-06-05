#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

######################################################################## Numba-accelerated interface

import awkward.array.base
import awkward.array.jagged
from .base import CppMethods
from ._jagged import JaggedArraySrc

class JaggedArrayCpp(CppMethods, JaggedArraySrc, awkward.array.jagged.JaggedArray):
    @classmethod
    def startsstops2parents(cls, starts, stops):
        if starts.dtype is not stops.dtype:
            raise ValueError("starts and stops must be the same type")
        return getattr(JaggedArraySrc, "startsstops2parents")(starts, stops)

    @classmethod
    def parents2startsstops(cls, parents, length = None):
        if length is None:
            length = -1
        return getattr(JaggedArraySrc, "parents2startsstops")(parents, length)

    
