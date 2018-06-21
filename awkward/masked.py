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

import numpy

import awkward.base

class MaskedArray(awkward.base.AwkwardArray):
    def __init__(self, mask, content, writeable=True):
        self.mask = mask
        self.content = content
        self.writeable = writeable

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        value = self._toarray(value, self.MASKTYPE, (numpy.ndarray, awkward.base.AwkwardArray))

        if len(value.shape) != 1:
            raise TypeError("mask must have 1-dimensional shape")
        if value.shape[0] == 0:
            value = value.view(self.MASKTYPE)
        if not issubclass(value.dtype.type, (numpy.bool_, numpy.bool)):
            raise TypeError("mask must have boolean dtype")

        self._mask = value

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = self._toarray(value, self.CHARTYPE, (numpy.ndarray, awkward.base.AwkwardArray))

    @property
    def writeable(self):
        return self._writeable

    @writeable.setter
    def writeable(self, value):
        self._writeable = bool(value)

    @property
    def dtype(self):
        return self._content.dtype

    @property
    def shape(self):
        return self._content.shape



class BitMaskedArray(MaskedArray):
    pass

class FilteredArray(MaskedArray):
    pass

class BitFilteredArray(BitMaskedArray):
    pass
