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

import awkward.array.base

class IndexedArray(awkward.array.base.AwkwardArray):
    def __init__(self, index, content):
        raise NotImplementedError

    def copy(self, index=None, content=None):
        raise NotImplementedError

    def deepcopy(self, index=None, content=None):
        raise NotImplementedError

    def empty_like(self, **overrides):
        raise NotImplementedError

    def zeros_like(self, **overrides):
        raise NotImplementedError

    def ones_like(self, **overrides):
        raise NotImplementedError

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        raise NotImplementedError

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        raise NotImplementedError

    @property
    def type(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def base(self):
        raise NotImplementedError

    def _valid(self):
        raise NotImplementedError

    def _argfields(self, function):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, where):
        raise NotImplementedError

    def __setitem__(self, where, what):
        raise NotImplementedError

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError

    @classmethod
    def concat(cls, first, *rest):
        raise NotImplementedError

    @classmethod
    def zip(cls, columns1={}, *columns2, **columns3):
        raise NotImplementedError

    @property
    def columns(self):
        raise NotImplementedError

    @property
    def allcolumns(self):
        raise NotImplementedError

    def pandas(self):
        raise NotImplementedError

# class IndexedArray(awkward.array.base.AwkwardArray):
#     def __init__(self, index, content):
#         self.index = index
#         self.content = content

#     @property
#     def index(self):
#         return self._index

#     @index.setter
#     def index(self, value):
#         value = self._toarray(value, self.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

#         if len(value.shape) != 1:
#             raise TypeError("index must have 1-dimensional shape")
#         if value.shape[0] == 0:
#             value = value.view(self.INDEXTYPE)
#         if not issubclass(value.dtype.type, numpy.integer):
#             raise TypeError("index must have integer dtype")

#         self._index = value

#     @property
#     def content(self):
#         return self._content

#     @content.setter
#     def content(self, value):
#         self._content = self._toarray(value, self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

#     @property
#     def dtype(self):
#         return self._content.dtype

#     @property
#     def shape(self):
#         return (len(self._index),) + self._content.shape[1:]

#     def __len__(self):
#         return len(self._index)
        
#     def __getitem__(self, where):
#         if self._isstring(where):
#             return IndexedArray(self._index, self._content[where])

#         return self._content[self._index[where]]

class ByteIndexedArray(IndexedArray):
    def __init__(self, index, content, dtype):
        raise NotImplementedError

    def copy(self, index=None, content=None):
        raise NotImplementedError

    def deepcopy(self, index=None, content=None):
        raise NotImplementedError

    def empty_like(self, **overrides):
        raise NotImplementedError

    def zeros_like(self, **overrides):
        raise NotImplementedError

    def ones_like(self, **overrides):
        raise NotImplementedError

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        raise NotImplementedError

    @property
    def type(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def base(self):
        raise NotImplementedError

    def _valid(self):
        raise NotImplementedError

    def _argfields(self, function):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, where):
        raise NotImplementedError

    def __setitem__(self, where, what):
        raise NotImplementedError

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError

    @classmethod
    def concat(cls, first, *rest):
        raise NotImplementedError

    @classmethod
    def zip(cls, columns1={}, *columns2, **columns3):
        raise NotImplementedError

    @property
    def columns(self):
        raise NotImplementedError

    @property
    def allcolumns(self):
        raise NotImplementedError

    def pandas(self):
        raise NotImplementedError

# class ByteIndexedArray(IndexedArray):
#     def __init__(self, index, content, dtype):
#         super(ByteIndexedArray, self).__init__(index, content)
#         self.dtype = dtype

#     @property
#     def content(self):
#         return self._content

#     @content.setter
#     def content(self, value):
#         self._content = self._toarray(value, self.CHARTYPE, numpy.ndarray).view(self.CHARTYPE).reshape(-1)

#     @property
#     def dtype(self):
#         return self._dtype

#     @dtype.setter
#     def dtype(self, value):
#         self._dtype = numpy.dtype(value)
        
#     def __getitem__(self, where):
#         if self._isstring(where):
#             return ByteIndexedArray(self._index, self._content[where], self._dtype)

#         starts = self._index[where]

#         if len(starts.shape) == 0:
#             pos, offset = divmod(starts, self._dtype.itemsize)
#             return numpy.frombuffer(self._content, dtype=self._dtype, count=(pos + 1), offset=offset)[pos]

#         else:
#             if len(starts) == 0:
#                 return numpy.empty(0, dtype=self._dtype)

#             else:
#                 hold = numpy.empty(len(starts), dtype=self._dtype)

#                 contidx = numpy.empty(len(starts) * self._dtype.itemsize, dtype=self.INDEXTYPE)
#                 contidx[::self._dtype.itemsize] = starts
#                 for offset in range(1, self._dtype.itemsize):
#                     contidx[offset::self._dtype.itemsize] = contidx[::self._dtype.itemsize] + offset
                
#                 holdidx = numpy.empty(len(starts) * self._dtype.itemsize, dtype=self.INDEXTYPE)
#                 holdidx[::self._dtype.itemsize] = numpy.arange(0, len(starts) * self._dtype.itemsize, self._dtype.itemsize)
#                 for offset in range(1, self._dtype.itemsize):
#                     holdidx[offset::self._dtype.itemsize] = holdidx[::self._dtype.itemsize] + offset

#                 numpy.frombuffer(hold, dtype=self.CHARTYPE)[holdidx] = self._content[contidx]
#                 return hold

class IndexedMaskedArray(IndexedArray):
    def __init__(self, index, content, maskedwhen=-1):
        raise NotImplementedError

    def copy(self, index=None, content=None):
        raise NotImplementedError

    def deepcopy(self, index=None, content=None):
        raise NotImplementedError

    def empty_like(self, **overrides):
        raise NotImplementedError

    def zeros_like(self, **overrides):
        raise NotImplementedError

    def ones_like(self, **overrides):
        raise NotImplementedError

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        raise NotImplementedError

    @property
    def type(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def base(self):
        raise NotImplementedError

    def _valid(self):
        raise NotImplementedError

    def _argfields(self, function):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, where):
        raise NotImplementedError

    def __setitem__(self, where, what):
        raise NotImplementedError

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError

    @classmethod
    def concat(cls, first, *rest):
        raise NotImplementedError

    @classmethod
    def zip(cls, columns1={}, *columns2, **columns3):
        raise NotImplementedError

    @property
    def columns(self):
        raise NotImplementedError

    @property
    def allcolumns(self):
        raise NotImplementedError

    def pandas(self):
        raise NotImplementedError

# class IndexedMaskedArray(IndexedArray):
#     def __init__(self, index, content, maskedwhen=-1):
#         super(IndexedMaskedArray, self).__init__(index, content)
#         self.maskedwhen = maskedwhen

#     @property
#     def maskedwhen(self):
#         return self._maskedwhen

#     @maskedwhen.setter
#     def maskedwhen(self, value):
#         if not isinstance(value, (numbers.Integral, numpy.integer)):
#             raise TypeError("maskedwhen must be an integer")
#         self._maskedwhen = value

#     def __getitem__(self, where):
#         if self._isstring(where):
#             return IndexedMaskedArray(self._index, self._content[where], maskedwhen=self._maskedwhen)

#         if not isinstance(where, tuple):
#             where = (where,)
#         head, tail = where[0], where[1:]

#         if isinstance(head, (numbers.Integral, numpy.integer)):
#             if self._index[head] == self._maskedwhen:
#                 return numpy.ma.masked
#             else:
#                 return self._content[self._singleton((self._index[head],) + tail)]
#         else:
#             return IndexedMaskedArray(self._index[head], self._content[self._singleton((slice(None),) + tail)], maskedwhen=self._maskedwhen)
