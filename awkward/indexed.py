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

class indexedarray(AwkwardArray):
    indextype = numpy.dtype(numpy.int64)

    def __init__(self, index, content):
        self.index = index
        self.content = content

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        if not isinstance(value, AwkwardArray):
            value = numpy.array(value, dtype=getattr(value, "dtype", self.indextype), copy=False)
            if not issubclass(value.dtype.type, numpy.integer):
                raise TypeError("index must have integer dtype")
            if len(value.shape) != 1:
                raise TypeError("index must have 1-dimensional shape")

        self._index = value

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        if not isinstance(value, AwkwardArray):
            value = numpy.array(value, copy=False).reshape(-1)
        self._content = value

    def __getitem__(self, where):
        return self._content[self._index[where]]

class byteindexedarray(indexedarray):
    def __init__(self, index, content, dtype):
        super(byteindexedarray, self).__init__(index, content)
        self.dtype = dtype

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        try:
            memoryview(value)
        except TypeError:
            raise TypeError("content must support the buffer protocol")
        self._content = value

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = numpy.dtype(value)

    def __getitem__(self, where):
        starts = self._index[where]

        if len(starts.shape) == 0:
            return numpy.frombuffer(self._content, dtype=AwkwardArray.chartype)[starts : starts + self._dtype.itemsize].view(self._dtype)[0]

        else:
            out = numpy.empty(len(starts), dtype=self._dtype)
            to = numpy.arange(0, len(starts) * self._dtype.itemsize, self._dtype.itemsize)
            if len(out) != 0:
                for offset in range(self._dtype.itemsize):
                    numpy.frombuffer(out, dtype=AwkwardArray.chartype, offset=offset)[to] = numpy.frombuffer(self._content, dtype=AwkwardArray.chartype, offset=offset)[starts]
            return out
