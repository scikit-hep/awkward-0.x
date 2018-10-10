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

import functools

import awkward.array.base
import awkward.type
import awkward.util

class UnionArray(awkward.array.base.AwkwardArray):
    def __init__(self, tags, index, contents):
        self.tags = tags
        self.index = index
        self.contents = contents

    @classmethod
    def fromtags(cls, tags, contents):
        out = cls.__new__(cls)
        out.tags = tags
        out.index = awkward.util.numpy.empty(out._tags.shape, dtype=awkward.util.INDEXTYPE)
        out.contents = contents

        if out._tags.reshape(-1).max() >= len(out._contents):
            raise ValueError("maximum tag is {0} but there are only {1} contents arrays".format(out._tags.reshape(-1).max(), len(out._contents)))

        for tag, content in enumerate(out._contents):
            mask = (out._tags == tag)
            out._index[mask] = awkward.util.numpy.arange(awkward.util.numpy.count_nonzero(mask))

        return out

    def copy(self, tags=None, index=None, contents=None):
        out = self.__class__.__new__(self.__class__)
        out._tags = self._tags
        out._index = self._index
        out._contents = self._contents
        out._dtype = self._dtype
        out._isvalid = self._isvalid
        if tags is not None:
            out.tags = tags
        if index is not None:
            out.index = index
        if contents is not None:
            out.contents = contents
        return out

    def deepcopy(self, tags=None, index=None, contents=None):
        out = self.copy(tags=tags, index=index, contents=contents)
        out._tags = awkward.util.deepcopy(out._tags)
        out._index = awkward.util.deepcopy(out._index)
        out._contents = [awkward.util.deepcopy(x) for x in out._contents]            
        return out

    def empty_like(self, **overrides):
        return self.copy(contents=[awkward.util.numpy.empty_like(x) if isinstance(x, awkward.util.numpy.ndarray) else x.empty_like(**overrides) for x in self._contents])

    def zeros_like(self, **overrides):
        return self.copy(contents=[awkward.util.numpy.zeros_like(x) if isinstance(x, awkward.util.numpy.ndarray) else x.zeros_like(**overrides) for x in self._contents])

    def ones_like(self, **overrides):
        return self.copy(contents=[awkward.util.numpy.ones_like(x) if isinstance(x, awkward.util.numpy.ndarray) else x.ones_like(**overrides) for x in self._contents])

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, awkward.util.numpy.ndarray)
        if not issubclass(value.dtype.type, awkward.util.numpy.integer):
            raise TypeError("tags must have integer dtype")
        if (value < 0).any():
            raise ValueError("tags must be a non-negative array")
        self._tags = value
        self._isvalid = False

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, awkward.util.numpy.ndarray)
        if not issubclass(value.dtype.type, awkward.util.numpy.integer):
            raise TypeError("index must have integer dtype")
        if (value < 0).any():
            raise ValueError("index must be a non-negative array")
        self._index = value
        self._isvalid = False

    @property
    def contents(self):
        return self._contents

    @contents.setter
    def contents(self, value):
        value = tuple(awkward.util.toarray(x, awkward.util.DEFAULTTYPE) for x in value)
        if len(value) == 0:
            raise ValueError("contents must be a non-empty iterable")
        self._contents = value
        self._dtype = None
        self._isvalid = False

    @property
    def dtype(self):
        if self._dtype is None:
            if all(issubclass(x.dtype.type, (awkward.util.numpy.bool_, awkward.util.numpy.bool)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.bool_)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.int8)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.int8)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.uint8)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.uint8)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.int8, awkward.util.numpy.uint8, awkward.util.numpy.int16)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.int16)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.uint8, awkward.util.numpy.uint16)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.uint16)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.int8, awkward.util.numpy.uint8, awkward.util.numpy.int16, awkward.util.numpy.uint16, awkward.util.numpy.int32)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.int32)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.uint8, awkward.util.numpy.uint16, awkward.util.numpy.uint32)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.uint32)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.int8, awkward.util.numpy.uint8, awkward.util.numpy.int16, awkward.util.numpy.uint16, awkward.util.numpy.int32, awkward.util.numpy.uint32, awkward.util.numpy.int64)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.int64)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.uint8, awkward.util.numpy.uint16, awkward.util.numpy.uint32, awkward.util.numpy.uint64)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.uint64)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.float16)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.float16)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.float16, awkward.util.numpy.float32)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.float32)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.float16, awkward.util.numpy.float32, awkward.util.numpy.float64)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.float64)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.float16, awkward.util.numpy.float32, awkward.util.numpy.float64, awkward.util.numpy.float128)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.float128)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.integer, awkward.util.numpy.floating)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.float64)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.complex64)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.complex64)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.complex64, awkward.util.numpy.complex128)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.complex128)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.complex64, awkward.util.numpy.complex128, awkward.util.numpy.complex256)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.complex256)

            elif all(issubclass(x.dtype.type, (awkward.util.numpy.integer, awkward.util.numpy.floating, awkward.util.numpy.complexfloating)) for x in self._contents):
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.complex256)

            else:
                self._dtype = awkward.util.numpy.dtype(awkward.util.numpy.object_)

        return self._dtype

    @property
    def shape(self):
        first = self._contents[0].shape
        if self.dtype.kind == "O" or not all(x.shape == first for x in self._contents[1:]):
            return self._tags.shape
        else:
            return self._tags.shape + first

    def __len__(self):
        return len(self._tags)

    @property
    def type(self):
        return awkward.type.ArrayType(*(self._tags.shape + (functools.reduce(lambda a, b: a | b, [awkward.type.fromarray(x).to for x in self._contents]),)))

    def _valid(self):
        if not self._isvalid:
            if len(self._tags.shape) > len(self._index.shape):
                raise ValueError("tags length ({0}) must be less than or equal to index length ({1})".format(len(self._tags.shape), len(self._index.shape)))

            if self._tags.shape[1:] != self._index.shape[1:]:
                raise ValueError("tags dimensionality ({0}) must be equal to index dimensionality ({1})".format(self._tags.shape[1:], self._index.shape[1:]))

            if self._tags.reshape(-1).max() >= len(self._contents):
                raise ValueError("maximum tag is {0} but there are only {1} contents arrays".format(self._tags.reshape(-1).max(), len(self._contents)))

            index = self._index[:len(self._tags)]
            for tag, content in enumerate(self._contents):
                maxindex = index[self._tags == tag].reshape(-1).max()
                if maxindex >= len(content):
                    raise ValueError("maximum index ({0}) must be less than the length of all contents arrays ({1})".format(maxindex, len(content)))

            self._isvalid = True

    def __iter__(self):
        self._valid()
        # FIXME
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, where):
        self._valid()

        if awkward.util.isstringslice(where):
            contents = []
            for tag in awkward.util.numpy.unique(self._tags):
                contents.append(self._contents[tag][where])
            if len(contents) == 0:
                return self.copy(contents=[self._contents[0][where]])
            else:
                return self.copy(contents=contents)

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[:len(self._tags.shape)], where[len(self._tags.shape):]

        tags = self._tags[head]
        index = self._index[:len(self._tags)][head]

        if len(tags.shape) == len(index.shape) == 0:
            return self._contents[tags][(index,) + tail]
        else:
            if len(tags) == 0:
                return self._contents[0][(index,) + tail]
            elif (tags == tags[0]).all():
                return self._contents[tags[0]][(index,) + tail]
            else:
                return self.copy(tags=tags, index=index)
    
    def __setitem__(self, where, what):
        import awkward.array.index

        if what.shape[:len(self._tags.shape)] != self._tags.shape:
            raise ValueError("array to assign does not have the same starting shape as tags")

        if isinstance(where, awkward.util.string):
            for tag in awkward.util.numpy.unique(self._tags):
                inverseindex = awkward.array.index.invert(self._index[:len(self._tags)][self._tags == tag])
                self._contents[tag][where] = awkward.array.index.IndexedArray(inverseindex, what)

        elif awkward.util.isstringslice(where):
            if len(where) != len(what):
                raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
            for tag in awkward.util.numpy.unique(self._tags):
                inverseindex = awkward.array.index.invert(self._index[:len(self._tags)][self._tags == tag])
                for x, y in zip(where, what):
                    self._contents[tag][x] = awkward.array.index.IndexedArray(inverseindex, y)

        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __delitem__(self, where):
        if isinstance(where, awkward.util.string):
            for tag in awkward.util.numpy.unique(self._tags):
                del self._contents[tag][where]

        elif awkward.util.isstringslice(where):
            for tag in awkward.util.numpy.unique(self._tags):
                for x in where:
                    del self._contents[tag][x]

        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        HERE



        
    def any(self):
        self._valid()
        index = self._index[:len(self._tag)]
        for tag in awkward.util.numpy.unique(self._tags):
            if self._contents[tag][index[self._tags == tag]].any():
                return True
        return False

    def all(self):
        self._valid()
        index = self._index[:len(self._tag)]
        for tag in awkward.util.numpy.unique(self._tags):
            if not self._contents[tag][index[self._tags == tag]].all():
                return False
        return True

    @classmethod
    def concat(cls, first, *rest):
        raise NotImplementedError

    @property
    def base(self):
        return self._base

    @property
    def columns(self):
        out = None
        for content in self._contents:
            if out is None:
                out = content.columns
            else:
                out = [x for x in content.columns if x in out]
        return out

    @property
    def allcolumns(self):
        out = None
        for content in self._contents:
            if out is None:
                out = content.allcolumns
            else:
                out = [x for x in content.allcolumns if x in out]
        return out

    def pandas(self):
        raise NotImplementedError

# class UnionArray(awkward.array.base.AwkwardArray):
#     @classmethod
#     def fromtags(cls, tags, contents):
#         raise NotImplementedError

#     def __init__(self, tags, index, contents):
#         self.tags = tags
#         self.index = index
#         self.contents = contents

#     @property
#     def tags(self):
#         return self._tags

#     @tags.setter
#     def tags(self, value):
#         value = self._toarray(value, self.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

#         if len(value.shape) != 1:
#             raise TypeError("tags must have 1-dimensional shape")
#         if value.shape[0] == 0:
#             value = value.view(self.INDEXTYPE)
#         if not issubclass(value.dtype.type, numpy.integer):
#             raise TypeError("tags must have integer dtype")

#         self._tags = value

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
#     def contents(self):
#         return self._contents

#     @contents.setter
#     def contents(self, value):
#         self._contents = tuple(self._toarray(x, self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray)) for x in value)

#     @property
#     def dtype(self):
#         return numpy.dtype(object)

#     @property
#     def shape(self):
#         return (len(self._tags),)

#     def __len__(self):
#         return len(self._tags)

#     def __getitem__(self, where):
#         if self._isstring(where):
#             return UnionArray(self._tags, self._index, tuple(x[where] for x in self._contents))

#         if self._tags.shape != self._index.shape:
#             raise ValueError("tags shape ({0}) does not match index shape ({1})".format(self._tags.shape, self._index.shape))

#         if not isinstance(where, tuple):
#             where = (where,)
#         head, tail = where[0], where[1:]

#         tags = self._tags[head]
#         index = self._index[head]
#         assert tags.shape == index.shape

#         uniques = numpy.unique(tags)
#         if len(uniques) == 1:
#             return self._contents[uniques[0]][self._singleton((index,) + tail)]
#         else:
#             return UnionArray(tags, index, self._contents)
