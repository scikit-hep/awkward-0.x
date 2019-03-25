#!/usr/bin/env python

# Copyright (c) 2019, IRIS-HEP
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

        index = cls.numpy.full(out._tags.shape, -1, dtype=cls.INDEXTYPE)
        for tag, content in enumerate(out._contents):
            mask = (out._tags == tag)
            index[mask] = cls.numpy.arange(cls.numpy.count_nonzero(mask))

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
        out._tags = self._util_deepcopy(out._tags)
        out._index = self._util_deepcopy(out._index)
        out._contents = [self._util_deepcopy(x) for x in out._contents]            
        return out

    def empty_like(self, **overrides):
        return self.copy(contents=[self.numpy.empty_like(x) if isinstance(x, self.numpy.ndarray) else x.empty_like(**overrides) for x in self._contents])

    def zeros_like(self, **overrides):
        return self.copy(contents=[self.numpy.zeros_like(x) if isinstance(x, self.numpy.ndarray) else x.zeros_like(**overrides) for x in self._contents])

    def ones_like(self, **overrides):
        return self.copy(contents=[self.numpy.ones_like(x) if isinstance(x, self.numpy.ndarray) else x.ones_like(**overrides) for x in self._contents])

    @property
    def issequential(self):
        self._valid()
        for tag in self.numpy.unique(self._tags):
            mask = self._tags == tag
            if not self.numpy.array_equal(self._index[mask], self.numpy.arange(self.numpy.count_nonzero(mask))):
                return False
        return True

    def __awkward_persist__(self, ident, fill, prefix, suffix, schemasuffix, storage, compression, **kwargs):
        self._valid()
        if self.issequential:
            return {"id": ident,
                    "call": ["awkward", "UnionArray", "fromtags"],
                    "args": [fill(self._tags, "UnionArray.tags", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                             {"list": [fill(x, "UnionArray.contents", prefix, suffix, schemasuffix, storage, compression, **kwargs) for x in self._contents]}]}

        else:
            return {"id": ident,
                    "call": ["awkward", "UnionArray"],
                    "args": [fill(self._tags, "UnionArray.tags", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                             fill(self._index, "UnionArray.index", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                             {"list": [fill(x, "UnionArray.contents", prefix, suffix, schemasuffix, storage, compression, **kwargs) for x in self._contents]}]}

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, value):
        value = self._util_toarray(value, self.TAGTYPE, self.numpy.ndarray)
        tagsmax = value.max()
        if tagsmax <= self.numpy.iinfo(self.TAGTYPE).max:
            value = value.astype(self.TAGTYPE)
        elif tagsmax <= self.numpy.iinfo(self.numpy.uint8).max:
            value = value.astype(self.numpy.uint8)
        elif tagsmax <= self.numpy.iinfo(self.numpy.uint16).max:
            value = value.astype(self.numpy.uint16)
        elif tagsmax <= self.numpy.iinfo(self.numpy.uint32).max:
            value = value.astype(self.numpy.uint32)
        elif tagsmax <= self.numpy.iinfo(self.numpy.uint64).max:
            value = value.astype(self.numpy.uint64)
        else:
            raise ValueError("maximum tag must be at most {0}".format(self.numpy.iinfo(self.numpy.uint64).max))

        if self.check_prop_valid:
            if len(value) == 0:
                raise ValueError("tags must be non-empty")
            if not self._util_isintegertype(value.dtype.type):
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
        value = self._util_toarray(value, self.INDEXTYPE, self.numpy.ndarray)
        if self.check_prop_valid:
            if not self._util_isintegertype(value.dtype.type):
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
        if self.check_prop_valid:
            try:
                iter(value)
            except TypeError:
                raise TypeError("contents must be iterable")
        value = tuple(self._util_toarray(x, self.DEFAULTTYPE) for x in value)
        if self.check_prop_valid:
            if len(value) == 0:
                raise ValueError("contents must be non-empty")
        self._contents = value
        self._dtype = None
        self._isvalid = False

    @property
    def dtype(self):
        if self._dtype is None:
            if all(issubclass(x.dtype.type, (self.numpy.bool_, self.numpy.bool)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.bool_)

            elif all(issubclass(x.dtype.type, (self.numpy.int8)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.int8)

            elif all(issubclass(x.dtype.type, (self.numpy.uint8)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.uint8)

            elif all(issubclass(x.dtype.type, (self.numpy.int8, self.numpy.uint8, self.numpy.int16)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.int16)

            elif all(issubclass(x.dtype.type, (self.numpy.uint8, self.numpy.uint16)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.uint16)

            elif all(issubclass(x.dtype.type, (self.numpy.int8, self.numpy.uint8, self.numpy.int16, self.numpy.uint16, self.numpy.int32)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.int32)

            elif all(issubclass(x.dtype.type, (self.numpy.uint8, self.numpy.uint16, self.numpy.uint32)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.uint32)

            elif all(issubclass(x.dtype.type, (self.numpy.int8, self.numpy.uint8, self.numpy.int16, self.numpy.uint16, self.numpy.int32, self.numpy.uint32, self.numpy.int64)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.int64)

            elif all(issubclass(x.dtype.type, (self.numpy.uint8, self.numpy.uint16, self.numpy.uint32, self.numpy.uint64)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.uint64)

            elif all(issubclass(x.dtype.type, (self.numpy.float16)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.float16)

            elif all(issubclass(x.dtype.type, (self.numpy.float16, self.numpy.float32)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.float32)

            elif all(issubclass(x.dtype.type, (self.numpy.float16, self.numpy.float32, self.numpy.float64)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.float64)

            elif hasattr(self.numpy, "float128") and all(issubclass(x.dtype.type, (self.numpy.float16, self.numpy.float32, self.numpy.float64, self.numpy.float128)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.float128)

            elif all(issubclass(x.dtype.type, (self.numpy.integer, self.numpy.floating)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.float64)

            elif all(issubclass(x.dtype.type, (self.numpy.complex64)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.complex64)

            elif all(issubclass(x.dtype.type, (self.numpy.complex64, self.numpy.complex128)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.complex128)

            elif hasattr(self.numpy, "complex256") and all(issubclass(x.dtype.type, (self.numpy.complex64, self.numpy.complex128, self.numpy.complex256)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.complex256)

            elif all(issubclass(x.dtype.type, (self.numpy.integer, self.numpy.floating, self.numpy.complexfloating)) for x in self._contents):
                self._dtype = self.numpy.dtype(self.numpy.complex128)

            else:
                self._dtype = self.numpy.dtype(self.numpy.object_)

        return self._dtype

    def _getnbytes(self, seen):
        if id(self) in seen:
            return 0
        else:
            seen.add(id(self))
            return sum(x.nbytes if isinstance(x, self.numpy.ndarray) else x._getnbytes(seen) for x in self._contents)

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
        if self.check_whole_valid:
            if not self._isvalid:
                if len(self._tags.shape) > len(self._index.shape):
                    raise ValueError("tags length ({0}) must be less than or equal to index length ({1})".format(len(self._tags.shape), len(self._index.shape)))

                if self._tags.shape[1:] != self._index.shape[1:]:
                    raise ValueError("tags dimensionality ({0}) must be equal to index dimensionality ({1})".format(self._tags.shape[1:], self._index.shape[1:]))

                if len(self._tags.reshape(-1)) > 0 and self._tags.reshape(-1).max() >= len(self._contents):
                    raise ValueError("maximum tag is {0} but there are only {1} contents arrays".format(self._tags.reshape(-1).max(), len(self._contents)))

                index = self._index[:len(self._tags)]
                for tag in self.numpy.unique(self._tags):
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

        if self._util_isstringslice(where):
            contents = []
            for tag in self.numpy.unique(self._tags):
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
        if what.shape[:len(self._tags.shape)] != self._tags.shape:
            raise ValueError("array to assign does not have the same starting shape as tags")

        if isinstance(where, awkward.util.string):
            for tag in self.numpy.unique(self._tags):
                inverseindex = self.IndexedArray.invert(self._index[:len(self._tags)][self._tags == tag])
                self._contents[tag][where] = self.IndexedArray(inverseindex, what)

        elif self._util_isstringslice(where):
            if len(where) != len(what):
                raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
            for tag in self.numpy.unique(self._tags):
                inverseindex = self.IndexedArray.invert(self._index[:len(self._tags)][self._tags == tag])
                for x, y in zip(where, what):
                    self._contents[tag][x] = self.IndexedArray(inverseindex, y)

        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __delitem__(self, where):
        if isinstance(where, awkward.util.string):
            for tag in self.numpy.unique(self._tags):
                del self._contents[tag][where]

        elif self._util_isstringslice(where):
            for tag in self.numpy.unique(self._tags):
                for x in where:
                    del self._contents[tag][x]

        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            raise NotImplementedError("in-place operations not supported")

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

        combos = self.numpy.stack(tags, axis=-1).view([(str(i), x.dtype) for i, x in enumerate(tags)]).reshape(tags[0].shape)

        outtags = self.numpy.empty(tags[0].shape, dtype=self.TAGTYPE)
        outindex = self.numpy.empty(tags[0].shape, dtype=self.INDEXTYPE)

        out = None
        contents = {}
        types = {}
        for outtag, combo in enumerate(self.numpy.unique(combos)):
            mask = (combos == combo)
            outtags[mask] = outtag
            outindex[mask] = self.numpy.arange(self.numpy.count_nonzero(mask))

            result = getattr(ufunc, method)(*[x[mask] if isinstance(x, UnionArray) else x for x in inputs], **kwargs)

            if isinstance(result, tuple):
                if out is None:
                    out = list(result)
                for i, x in enumerate(result):
                    if isinstance(x, (self.numpy.ndarray, awkward.array.base.AwkwardArray)):
                        if i not in contents:
                            contents[i] = []
                        contents[i].append(x)
                        types[i] = type(x)

            elif method == "at":
                pass

            else:
                if isinstance(result, (self.numpy.ndarray, awkward.array.base.AwkwardArray)):
                    if None not in contents:
                        contents[None] = []
                    contents[None].append(result)
                    types[None] = type(result)

        if out is None:
            if None in contents:
                return self.Methods.maybemixin(types[None], UnionArray)(outtags, outindex, contents[None])
            else:
                return None
        else:
            for i in range(len(out)):
                if i in contents:
                    out[i] = self.Methods.maybemixin(types[i], UnionArray)(outtags, outindex, contents[i])
            return tuple(out)

    def _hasjagged(self):
        return all(self._util_hasjagged(x) for x in self._contents)

    def _reduce(self, ufunc, identity, dtype, regularaxis):
        if self._hasjagged():
            return self.copy(contents=[x._reduce(ufunc, identity, dtype, regularaxis) for x in self._contents])

        elif self.columns != []:
            out = awkward.array.table.Table()
            for n in self.columns:
                out[n] = self.copy(content=self[n])
            return out._reduce(ufunc, identity, dtype, regularaxis)

        else:
            return ufunc.reduce(self._prepare(identity, dtype))

    def _prepare(self, identity, dtype):
        if dtype is None and issubclass(self.dtype.type, (self.numpy.bool_, self.numpy.bool)):
            dtype = self.numpy.dtype(type(identity))
        if dtype is None:
            dtype = self.dtype

        out = None
        index = self._index[:len(self._tags)]
        for tag, content in enumerate(self._contents):
            if not isinstance(content, self.numpy.ndarray):
                content = content._prepare(identity, dtype)

            mask = (self._tags == tag)
            c = content[index[mask]]

            if out is None:
                out = self.numpy.full(self._tags.shape[:1] + c.shape[1:], identity, dtype=dtype)
            out[mask] = c

        return out

    @property
    def columns(self):
        out = None
        for content in self._contents:
            if isinstance(content, self.numpy.ndarray):
                return []

            if out is None:
                out = content.columns
            else:
                out = [x for x in content.columns if x in out]

        if out is None:
            return []
        else:
            return out

    def astype(self, dtype):
        return self.copy(contents=[x.astype(dtype) for x in self._contents])

    def fillna(self, value):
        return self.copy(contents=[self._util_fillna(x, value) for x in self._contents])
