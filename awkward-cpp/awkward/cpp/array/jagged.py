#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

######################################################################## Numba-accelerated interface

import awkward.array.base
import awkward.array.jagged
import sys
from .base import CppMethods
from ._jagged import JaggedArraySrc

class JaggedArrayCpp(CppMethods, JaggedArraySrc, awkward.array.jagged.JaggedArray):
    @classmethod
    def parents2startsstops(cls, parents, length = None):
        if length is None:
            length = -1
        return getattr(JaggedArraySrc, "parents2startsstops")(parents, length)

    def __init__(self, starts, stops, content):
        if sys.byteorder is "big":
            if starts.dtype.byteorder is "<" or stops.dtype.byteorder is "<":
                raise TypeError("starts and stops must be of native byteorder")
        elif starts.dtype.byteorder is ">" or stops.dtype.byteorder is ">":
            raise TypeError("starts and stops must be of native byteorder")
        JaggedArraySrc.__init__(self, starts, stops, content)

    @JaggedArraySrc.starts.setter
    def starts(self, value):
        if sys.byteorder is "big":
            if value.dtype.byteorder is "<":
                raise TypeError("starts must be of native byteorder")
        elif value.dtype.byteorder is ">":
            raise TypeError("starts must be of native byteorder")
        return getattr(JaggedArraySrc, "set_starts")(self, value)

    @JaggedArraySrc.stops.setter
    def stops(self, value):
        if sys.byteorder is "big":
            if value.dtype.byteorder is "<":
                raise TypeError("stops must be of native byteorder")
        elif value.dtype.byteorder is ">":
            raise TypeError("stops must be of native byteorder")
        return getattr(JaggedArraySrc, "set_stops")(self, value)
