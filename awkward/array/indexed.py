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

import awkward.array.base
import awkward.util

def invert(permutation):
    permutation = permutation.reshape(-1)
    out = awkward.util.numpy.zeros(permutation.max() + 1, dtype=awkward.util.INDEXTYPE)
    identity = awkward.util.numpy.arange(len(permutation))
    out[permutation] = identity
    if not awkward.util.numpy.array_equal(out[permutation], identity):
        raise ValueError("cannot invert index; it contains duplicates")
    return out

class IndexedArray(awkward.array.base.AwkwardArray):
    def __init__(self, index, content):
        self.index = index
        self.content = content
        self._inverse = None
        self._isvalid = False

    def copy(self, index=None, content=None):
        out = self.__class__.__new__(self.__class__)
        out._index = self._index
        out._content = self._content
        out._inverse = self._inverse
        out._isvalid = self._isvalid
        if index is not None:
            out.index = index
        if content is not None:
            out.content = content
        return out

    def deepcopy(self, index=None, content=None):
        out = self.copy(index=index, content=content)
        out._index   = awkward.util.deepcopy(out._index)
        out._content = awkward.util.deepcopy(out._content)
        out._inverse = awkward.util.deepcopy(out._inverse)
        return out

    def empty_like(self, **overrides):
        mine = {}
        mine["index"] = overrides.pop("index", self._index)
        mine["content"] = overrides.pop("content", self._content)
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.empty_like(self._content), **mine)
        else:
            return self.copy(content=self._content.empty_like(**overrides), **mine)

    def zeros_like(self, **overrides):
        mine = {}
        mine["index"] = overrides.pop("index", self._index)
        mine["content"] = overrides.pop("content", self._content)
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.zeros_like(self._content), **mine)
        else:
            return self.copy(content=self._content.zeros_like(**overrides), **mine)

    def ones_like(self, **overrides):
        mine = {}
        mine["index"] = overrides.pop("index", self._index)
        mine["content"] = overrides.pop("content", self._content)
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.ones_like(self._content), **mine)
        else:
            return self.copy(content=self._content.ones_like(**overrides), **mine)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE)
        if not issubclass(value.dtype.type, awkward.util.numpy.integer):
            raise TypeError("index must have integer dtype")
        if (value < 0).any():
            raise ValueError("index must be a non-negative array")
        self._index = value
        self._inverse = None
        self._isvalid = False

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = awkward.util.toarray(value, awkward.util.DEFAULTTYPE)
        self._isvalid = False

    @property
    def dtype(self):
        return self._content.dtype

    @property
    def shape(self):
        return self._index.shape

    def __len__(self):
        return len(self._index)

    @property
    def type(self):
        return self._content.type

    @property
    def base(self):
        return self._content.base

    def _valid(self):
        if not self._isvalid:
            if len(self._index) != 0 and self._index.reshape(-1).max() > len(self._content):
                raise ValueError("maximum index ({0}) is beyond the length of the content ({1})".format(self._index.reshape(-1).max(), len(self._content)))
            self._isvalid = True

    def _argfields(self, function):
        if (isinstance(function, types.FunctionType) and function.__code__.co_argcount == 1) or isinstance(self._content, awkward.util.numpy.ndarray):
            return awkward.util._argfields(function)
        else:
            return self._content._argfields(function)

    def __iter__(self):
        self._valid()
        for i in self._index:
            yield self._content[i]

    def __getitem__(self, where):
        self._valid()

        if awkward.util.isstringslice(where):
            out = self.copy(self._index, self._content[where])
            out._isvalid = True
            return out

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[:len(self._index.shape)], where[len(self._index.shape):]

        where = self._index[where]
        if len(where.shape) != 0 and len(where) == 0:
            return awkward.util.numpy.empty(0, dtype=self._content.dtype)[tail]
        else:
            return self._content[(where,) + tail]

    def _invert(self, what):
        if what.shape != self._index.shape:
            raise ValueError("array to assign does not have the same shape as index")
        if self._inverse is None:
            self._inverse = invert(self._index)
        return IndexedArray(self._inverse, what)

    def __setitem__(self, where, what):
        self._valid()

        if isinstance(where, awkward.util.string):
            self._content[where] = self._invert(what)

        elif awkward.util.isstringslice(where):
            if len(where) != len(what):
                raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
            for x, y in zip(where, what):
                self._content[x] = self._invert(y)

        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self._valid()

        if method != "__call__":
            return NotImplemented

        inputs = list(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], IndexedArray):
                inputs[i] = inputs[i][:]

        return getattr(ufunc, method)(*inputs, **kwargs)

    @classmethod
    def concat(cls, first, *rest):
        raise NotImplementedError

    @property
    def columns(self):
        if isinstance(self._content, awkward.util.numpy.ndarray):
            raise TypeError("array has no Table, and hence no columns")
        return self._content.columns

    @property
    def allcolumns(self):
        if isinstance(self._content, awkward.util.numpy.ndarray):
            raise TypeError("array has no Table, and hence no columns")
        return self._content.allcolumns

    def pandas(self):
        import pandas

        self._valid()

        if isinstance(self._content, awkward.util.numpy.ndarray):
            return pandas.DataFrame(self._content[self._index])
        else:
            return self._content[self._index].pandas()

class ByteIndexedArray(IndexedArray):
    def __init__(self, index, content, dtype):
        super(ByteIndexedArray, self).__init__(index, content)
        self.dtype = dtype

    def copy(self, index=None, content=None, dtype=None):
        out = self.__class__.__new__(self.__class__)
        out._index = self._index
        out._content = self._content
        out._dtype = self._dtype
        if index is not None:
            out.index = index
        if content is not None:
            out.content = content
        if dtype is not None:
            out.dtype = dtype
        return out

    def deepcopy(self, index=None, content=None, dtype=None):
        out = self.copy(index=index, content=content)
        out._index   = awkward.util.deepcopy(out._index)
        out._content = awkward.util.deepcopy(out._content)
        return out

    def empty_like(self, **overrides):
        mine = {}
        mine["index"] = overrides.pop("index", self._index)
        mine["content"] = overrides.pop("content", self._content)
        mine["dtype"] = overrides.pop("dtype", self._dtype)
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.empty_like(self._content), **mine)
        else:
            return self.copy(content=self._content.empty_like(**overrides), **mine)

    def zeros_like(self, **overrides):
        mine = {}
        mine["index"] = overrides.pop("index", self._index)
        mine["content"] = overrides.pop("content", self._content)
        mine["dtype"] = overrides.pop("dtype", self._dtype)
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.zeros_like(self._content), **mine)
        else:
            return self.copy(content=self._content.zeros_like(**overrides), **mine)

    def ones_like(self, **overrides):
        mine = {}
        mine["index"] = overrides.pop("index", self._index)
        mine["content"] = overrides.pop("content", self._content)
        mine["dtype"] = overrides.pop("dtype", self._dtype)
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.ones_like(self._content), **mine)
        else:
            return self.copy(content=self._content.ones_like(**overrides), **mine)

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = awkward.util.toarray(value, awkward.util.CHARTYPE, awkward.util.numpy.ndarray).view(awkward.util.CHARTYPE).reshape(-1)

    @property
    def type(self):
        return awkward.type.ArrayType(self._index.shape, self._dtype)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = awkward.util.numpy.dtype(value)

    def _valid(self):
        if not self._isvalid:
            if len(self._index) != 0 and self._index.reshape(-1).max() > len(self._content):
                raise ValueError("maximum index ({0}) is beyond the length of the content ({1})".format(self._index.reshape(-1).max(), len(self._content)))

    def __iter__(self):
        self._valid()
        itemsize = self._dtype.itemsize
        for i in self._index:
            yield self._content[i : i + itemsize].view(self._dtype)[0]

    def __getitem__(self, where):
        self._valid()

        if awkward.util.isstringslice(where):
            raise IndexError("only integers, slices (`:`), and integer or boolean arrays are valid indices")

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[:len(self._index.shape)], where[len(self._index.shape):]

        if len(tail) != 0:
            raise IndexError("too many indices for array")

        starts = self._index[where]

        if len(starts.shape) == 0:
            return self._content[starts : starts + self._dtype.itemsize].view(self._dtype)[0]

        else:
            if len(starts) == 0:
                return awkward.util.numpy.empty(0, dtype=self._dtype)

            else:
                index = awkward.util.numpy.repeat(starts, self._dtype.itemsize)
                index += awkward.util.numpy.tile(awkward.util.numpy.arange(self._dtype.itemsize), len(starts))
                return self._content[index].view(self._dtype)

    def __setitem__(self, where, what):
        self._valid()
        if awkward.util.isstringslice(where):
            raise IndexError("only integers, slices (`:`), and integer or boolean arrays are valid indices")
        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def any(self):
        return self._content[self._index].any()

    def all(self):
        return self._content[self._index].all()

    @classmethod
    def concat(cls, first, *rest):
        raise NotImplementedError

    @property
    def columns(self):
        raise NotImplementedError

    @property
    def allcolumns(self):
        raise NotImplementedError

    def pandas(self):
        raise NotImplementedError

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
