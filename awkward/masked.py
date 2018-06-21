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

import collections
import numbers

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

    def __len__(self):
        return self._content.shape[0]

    def __getitem__(self, where):
        if self._isstring(where):
            return MaskedArray(self._mask, self._content[where], writeable=self._writeable)

        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        if isinstance(head, (numbers.Integral, numpy.integer)):
            if self._mask[head]:
                return numpy.ma.masked
            else:
                return self._content[where]

        else:
            return MaskedArray(self._mask[head], self._content[where], writeable=self._writeable)

    def __setitem__(self, where, what):
        if self._isstring(where):
            MaskedArray(self._mask, self._content[where], writeable=self._writeable)[:] = what
            return

        if not self._writeable:
            raise ValueError("assignment destination is read-only")

        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        if isinstance(what, numpy.ma.core.MaskedConstant) or (isinstance(what, collections.Sequence) and len(what) == 1 and isinstance(what[0], numpy.ma.core.MaskedConstant)):
            self._mask[head] = True
            
        elif isinstance(what, (collections.Sequence, numpy.ndarray, awkward.base.AwkwardArray)) and len(what) == 1:
            if isinstance(what[0], numpy.ma.core.MaskedConstant):
                self._mask[head] = True
            else:
                self._mask[head] = False
                self._content[where] = what[0]

        elif isinstance(what, MaskedArray):
            self._mask[head] = what._mask
            self._content[where] = what._content

        elif isinstance(what, collections.Sequence):
            self._mask[head] = [isinstance(x, numpy.ma.core.MaskedConstant) for x in what]
            self._content[where] = [x if not isinstance(x, numpy.ma.core.MaskedConstant) else 0 for x in what]

        elif isinstance(what, (numpy.ndarray, awkward.base.AwkwardArray)):
            self._mask[head] = False
            self._content[where] = what

        else:
            self._mask[head] = False
            self._content[where] = what

class BitMaskedArray(MaskedArray):
    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        value = self._toarray(value, self.BITMASKTYPE, (numpy.ndarray, awkward.base.AwkwardArray))

        if len(value.shape) != 1:
            raise TypeError("mask must have 1-dimensional shape")

        self._mask = value.view(self.BITMASKTYPE)

    def __getitem__(self, where):
        if self._isstring(where):
            return BitMaskedArray(self._mask, self._content[where], writeable=self._writeable)
        
        HERE

    def __setitem__(self, where, what):
        if self._isstring(where):
            BitMaskedArray(self._mask, self._content[where], writeable=self._writeable)[:] = what
            return

        THERE
