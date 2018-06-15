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

import numbers

import numpy

import awkward.base

class VirtualArray(awkward.base.AwkwardArray):
    def __init__(self, generator, dtype=None, shape=None):
        self.generator = generator
        self._array = None

        if dtype is None:
            self._dtype = dtype
        else:
            self._dtype = numpy.dtype(dtype)

        if shape is None or (isinstance(shape, tuple) and len(shape) != 0 and all(isinstance(x, (numbers.Integral, numpy.integer)) and x >= 0 for x in shape)):
            self._shape = shape
        else:
            raise TypeError("shape must be None (unknown) or a non-empty tuple of non-negative integers")

    def materialize(self):
        array = self.generator()
        if self._dtype is not None and self._dtype != array.dtype:
            raise ValueError("materialized array has dtype {0}, expected dtype {1}".format(array.dtype, self._dtype))
        if self._shape is not None and self._shape != array.shape:
            raise ValueError("materialized array has shape {0}, expected shape {1}".format(array.shape, self._shape))
        if len(array.shape) == 0:
            raise ValueError("materialized object is scalar: {0}".format(array))
        self._array, self._dtype, self._shape = array, array.dtype, array.shape

    def ensure_materialized(self):
        if self._array is None:
            self.materialize()

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape
    
    def __len__(self):
        if self._shape is None:
            self.ensure_materialized()
        return self._shape[0]

    def __getitem__(self, where):
        self.ensure_materialized()
        return self._array[where]

    def __setitem__(self, where, what):
        self.ensure_materialized()
        self._array[where] = what
        
class VirtualObjectArray(awkward.base.AwkwardArray):
    pass
