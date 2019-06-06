#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

######################################################################## Numba-accelerated interface

import awkward.array.base
import awkward.array.jagged
import numpy
from .base import CppMethods
from ._jagged import JaggedArraySrc
from ._jagged import CppNumPy

class JaggedArrayCpp(CppMethods, JaggedArraySrc, awkward.array.jagged.JaggedArray):
    @classmethod
    def practicemethod(cls, array):
        return NumPyCpp.fromCpp(getattr(JaggedArraySrc, "practicemethod"))(NumPyCpp.toCpp(array))
    
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
                                

class NumPyCpp:
    @classmethod
    def toCpp(cls, array):
        return getattr(CppNumPy, "create")(array.shape, array.dtype.char, array.data, array.strides, numpy.isfortran(array))

    @classmethod
    def fromCpp(cls, output):
        order = 'C'
        return numpy.ndarray(CppNumPy.get_shape(output), numpy.dtype(output.dtype), output.data, output.strides, order)
