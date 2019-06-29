#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

######################################################################## Numba-accelerated interface

import awkward.array.base
import awkward.array.jagged
from .base import CppMethods
from .array_impl import JaggedArray

class JaggedArrayCpp(CppMethods, JaggedArray, awkward.array.jagged.JaggedArray):
    @classmethod
    def parents2startsstops(cls, parents, length = None):
        if length is None:
            length = -1
        return getattr(JaggedArray, "parents2startsstops")(parents, length)

    def __getitem__(self, where):
        if isinstance(where, slice):
            length = getattr(JaggedArray, "__len__")(self)
            start = 0
            stop = length
            step = 1
            if where.step is not None:
                step = where.step
            if step < 0:
                start = length - 1
                stop = -1 - length
            if where.start is not None:
                start = where.start
            if where.stop is not None:
                stop = where.stop
            return getattr(JaggedArray, "__getitem__")(self, start, stop, step)
        return getattr(JaggedArray, "__getitem__")(self, where)
