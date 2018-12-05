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
import awkward.type
import awkward.util

class UnionArray(awkward.array.base.AwkwardArray):
    """
    UnionArray
    """

    def __init__(self, tags, index, contents):
        self.tags = tags
        self.index = index
        self.contents = contents

    @classmethod
    def fromtags(cls, tags, contents):
        out = cls.__new__(cls)
        out.tags = tags
        out.contents = contents

        if len(out._tags.reshape(-1)) > 0 and out._tags.reshape(-1).max() >= len(out._contents):
            raise ValueError("maximum tag is {0} but there are only {1} contents arrays".format(out._tags.reshape(-1).max(), len(out._contents)))

        index = awkward.util.numpy.full(out._tags.shape, -1, dtype=cls.INDEXTYPE)
        for tag, content in enumerate(out._contents):
            mask = (out._tags == tag)
            index[mask] = awkward.util.numpy.arange(awkward.util.numpy.count_nonzero(mask))

        out.index = index
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
    def issequential(self):
        self._valid()
        for tag in awkward.util.numpy.unique(self._tags):
            mask = self._tags == tag
            if not awkward.util.numpy.array_equal(self._index[mask], awkward.util.numpy.arange(awkward.util.numpy.count_nonzero(mask))):
                return False
        return True

    def __awkward_persist__(self, ident, fill, prefix, suffix, schemasuffix, storage, compression, **kwargs):
        self._valid()
        if self.issequential:
            return {"id": ident,
                    "call": ["awkward", self.__class__.__name__, "fromtags"],
                    "args": [fill(self._tags, self.__class__.__name__ + ".tags", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                             {"list": [fill(x, self.__class__.__name__ + ".contents", prefix, suffix, schemasuffix, storage, compression, **kwargs) for x in self._contents]}]}

        else:
            return {"id": ident,
                    "call": ["awkward", self.__class__.__name__],
                    "args": [fill(self._tags, self.__class__.__name__ + ".tags", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                             fill(self._index, self.__class__.__name__ + ".index", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                             {"list": [fill(x, self.__class__.__name__ + ".contents", prefix, suffix, schemasuffix, storage, compression, **kwargs) for x in self._contents]}]}

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, value):
        value = awkward.util.toarray(value, self.TAGTYPE, awkward.util.numpy.ndarray)
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
        value = awkward.util.toarray(value, self.INDEXTYPE, awkward.util.numpy.ndarray)
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
        try:
            iter(value)
        except TypeError:
            raise TypeError("contents must be iterable")
        value = tuple(awkward.util.toarray(x, self.DEFAULTTYPE) for x in value)
        if len(value) == 0:
            raise ValueError("contents must be non-empty")
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

    def __len__(self):
        return len(self._tags)

    def _gettype(self, seen):
        out = awkward.type.UnionType()
        for x in self._contents:
            out.append(awkward.type._fromarray(x, seen))
        for x in self._tags.shape[:0:-1]:
            out = awkward.type.ArrayType(x, out)
        return out

    def _valid(self):
        if not self._isvalid:
            if len(self._tags.shape) > len(self._index.shape):
                raise ValueError("tags length ({0}) must be less than or equal to index length ({1})".format(len(self._tags.shape), len(self._index.shape)))

            if self._tags.shape[1:] != self._index.shape[1:]:
                raise ValueError("tags dimensionality ({0}) must be equal to index dimensionality ({1})".format(self._tags.shape[1:], self._index.shape[1:]))

            if len(self._tags.reshape(-1)) > 0 and self._tags.reshape(-1).max() >= len(self._contents):
                raise ValueError("maximum tag is {0} but there are only {1} contents arrays".format(self._tags.reshape(-1).max(), len(self._contents)))

            index = self._index[:len(self._tags)]
            for tag in awkward.util.numpy.unique(self._tags):
                maxindex = index[self._tags == tag].reshape(-1).max()
                if maxindex >= len(self._contents[tag]):
                    raise ValueError("maximum index ({0}) must be less than the length of all contents arrays ({1})".format(maxindex, len(self._contents[tag])))

            self._isvalid = True

    def __iter__(self, checkiter=True):
        if checkiter:
            self._checkiter()
        self._valid()

        tags = self._tags
        lentags = len(self._tags)
        index = self._index
        contents = self._contents

        i = 0
        while i < lentags:
            yield contents[tags[i]][index[i]]
            i += 1

    def __getitem__(self, where):
        self._valid()

        if awkward.util.isstringslice(where):
            contents = []
            for tag in awkward.util.numpy.unique(self._tags):
                contents.append(self._contents[tag][where])
            # TODO: think about inheriting methods from contents[where]; all satisfying tags would have to have the same methods before promoting the output (generalized maybemixin?)
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
        import awkward.array.objects

        if method != "__call__":
            return NotImplemented

        tags = []
        for x in inputs:
            if isinstance(x, UnionArray):
                x._valid()
                tags.append(x._tags)
        assert len(tags) > 0

        if any(x.shape != tags[0].shape for x in tags[1:]):
            raise ValueError("cannot {0} UnionArrays because tag shapes differ".format(ufunc))

        combos = awkward.util.numpy.stack(tags, axis=-1).view([(str(i), x.dtype) for i, x in enumerate(tags)]).reshape(tags[0].shape)

        outtags = awkward.util.numpy.empty(tags[0].shape, dtype=self.TAGTYPE)
        outindex = awkward.util.numpy.empty(tags[0].shape, dtype=self.INDEXTYPE)

        out = None
        contents = {}
        types = {}
        for outtag, combo in enumerate(awkward.util.numpy.unique(combos)):
            mask = (combos == combo)
            outtags[mask] = outtag
            outindex[mask] = awkward.util.numpy.arange(awkward.util.numpy.count_nonzero(mask))

            result = getattr(ufunc, method)(*[x[mask] if isinstance(x, UnionArray) else x for x in inputs], **kwargs)

            if isinstance(result, tuple):
                if out is None:
                    out = list(result)
                for i, x in enumerate(result):
                    if isinstance(x, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray)):
                        if i not in contents:
                            contents[i] = []
                        contents[i].append(x)
                        types[i] = type(x)

            elif method == "at":
                pass

            else:
                if isinstance(result, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray)):
                    if None not in contents:
                        contents[None] = []
                    contents[None].append(result)
                    types[None] = type(result)

        if out is None:
            if None in contents:
                return awkward.array.objects.Methods.maybemixin(types[None], UnionArray)(outtags, outindex, contents[None])
            else:
                return None
        else:
            for i in range(len(out)):
                if i in contents:
                    out[i] = awkward.array.objects.Methods.maybemixin(types[i], UnionArray)(outtags, outindex, contents[i])
            return tuple(out)

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

    def astype(self, dtype):
        return self.copy(contents=[x.astype(dtype) for x in self._contents])

    def pandas(self):
        raise NotImplementedError
