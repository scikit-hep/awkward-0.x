#!/usr/bin/env python

# Copyright (c) 2018, DIANA-HEP
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import division

import unittest

import numpy as np

import awkward

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_method_mixin(self):
        class TypeArrayMethods(awkward.Methods):
            def _initObjectArray(self, table):
                awkward.ObjectArray.__init__(self, table, lambda row: Type(row["x"]))
                self.content.rowname = "Type"

            @property
            def x(self):
                return self["x"]

            @x.setter
            def x(self, value):
                self["x"] = value


        class TypeMethods(awkward.Methods):
            _arraymethods = TypeArrayMethods

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, value):
                self._x = value


        class TypeArray(TypeArrayMethods, awkward.ObjectArray):
            def __init__(self, x):
                self._initObjectArray(awkward.Table())
                self["x"] = x
                

        class Type(TypeMethods):
            def __init__(self, x):
                self._x = x
            

        counts = np.array([1, 4, 2, 0, 15])
        x = np.arange(np.sum(counts))
        array = TypeArray(x)
        assert np.all(array.x == x)

        JaggedTypeArray = awkward.Methods.mixin(TypeArrayMethods, awkward.JaggedArray)
        jagged_array = JaggedTypeArray.fromcounts(counts, array)
        assert np.all(jagged_array.x.flatten() == x)
        assert np.all(jagged_array.pairs()._0.x.counts == counts*(counts+1)//2)

