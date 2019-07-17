#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

######################################################################## Numba-accelerated interface

import awkward.array.base
import awkward.array.jagged
from .base import CppMethods
from .array_impl import JaggedArray

class JaggedArrayCpp(CppMethods, JaggedArray, awkward.array.jagged.JaggedArray):
    pass
