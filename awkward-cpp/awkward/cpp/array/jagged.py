#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

######################################################################## Numba-accelerated interface

import awkward.array.base
import awkward.array.jagged
from .base import CppMethods
from ._jagged import JaggedArraySrc

class JaggedArrayCpp(CppMethods, JaggedArraySrc, awkward.array.jagged.JaggedArray):
    @classmethod
    def parents2startsstops(cls, parents, length = None):
        if length is None:
            length = -1
        return getattr(JaggedArraySrc, "parents2startsstops")(parents, length)

    
