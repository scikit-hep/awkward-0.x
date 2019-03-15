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

import pickle
import numbers

import awkward.array.base
import awkward.persist
import awkward.type
import awkward.util

class IndexedArray(awkward.array.base.AwkwardArrayWithContent):
    """
    IndexedArray
    """

    @classmethod
    def invert(cls, permutation):
        if permutation.size == 0:
            return cls.numpy.zeros(0, dtype=cls.IndexedArray.fget(None).INDEXTYPE)
        permutation = permutation.reshape(-1)
        out = cls.numpy.zeros(permutation.max() + 1, dtype=cls.IndexedArray.fget(None).INDEXTYPE)
        identity = cls.numpy.arange(len(permutation))
        out[permutation] = identity
        if not cls.numpy.array_equal(out[permutation], identity):
            raise ValueError("cannot invert index; it contains duplicates")
        return out

    def __init__(self, index, content):
        self.index = index
        self.content = content

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
        out._index   = self._util_deepcopy(out._index)
        out._content = self._util_deepcopy(out._content)
        out._inverse = self._util_deepcopy(out._inverse)
        return out

    def empty_like(self, **overrides):
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.empty_like(self._content))
        else:
            return self.copy(content=self._content.empty_like(**overrides))

    def zeros_like(self, **overrides):
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.zeros_like(self._content))
        else:
            return self.copy(content=self._content.zeros_like(**overrides))

    def ones_like(self, **overrides):
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.ones_like(self._content))
        else:
            return self.copy(content=self._content.ones_like(**overrides))

    def __awkward_persist__(self, ident, fill, prefix, suffix, schemasuffix, storage, compression, **kwargs):
        self._valid()
        return {"id": ident,
                "call": ["awkward", "IndexedArray"],
                "args": [fill(self._index, "IndexedArray.index", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                         fill(self._content, "IndexedArray.content", prefix, suffix, schemasuffix, storage, compression, **kwargs)]}

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
        self._inverse = None
        self._isvalid = False

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = self._util_toarray(value, self.DEFAULTTYPE)
        self._isvalid = False

    def _getnbytes(self, seen):
        if id(self) in seen:
            return 0
        else:
            seen.add(id(self))
            return self._index.nbytes + (self._content.nbytes if isinstance(self._content, self.numpy.ndarray) else self._content._getnbytes(seen))

    def __len__(self):
        return len(self._index)

    def _gettype(self, seen):
        out = awkward.type._fromarray(self._content, seen)
        for x in self._index.shape[:0:-1]:
            out = awkward.type.ArrayType(x, out)
        return out

    def _valid(self):
        if self.check_whole_valid:
            if not self._isvalid:
                if len(self._index) != 0 and self._index.reshape(-1).max() > len(self._content):
                    raise ValueError("maximum index ({0}) is beyond the length of the content ({1})".format(self._index.reshape(-1).max(), len(self._content)))

                self._isvalid = True

    def __iter__(self, checkiter=True):
        if checkiter:
            self._checkiter()
        self._valid()
        for i in self._index:
            yield self._content[i]

    def __getitem__(self, where):
        self._valid()

        if self._util_isstringslice(where):
            content = self._content[where]
            cls = awkward.array.objects.Methods.maybemixin(type(content), self.IndexedArray)
            out = cls.__new__(cls)
            out.__dict__.update(self.__dict__)
            out._content = content
            return out

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[:len(self._index.shape)], where[len(self._index.shape):]

        head = self._index[head]
        if len(head.shape) != 0 and len(head) == 0:
            return self.numpy.empty(0, dtype=self._content.dtype)[tail]
        else:
            return self._content[(head,) + tail]

    def _invert(self, what):
        if self._inverse is None:
            self._inverse = self.invert(self._index)
        return IndexedArray(self._inverse, what)

    def __setitem__(self, where, what):
        if what.shape[:len(self._index.shape)] != self._index.shape:
            raise ValueError("array to assign does not have the same starting shape as index")

        if isinstance(where, awkward.util.string):
            self._content[where] = self._invert(what)

        elif self._util_isstringslice(where):
            if len(where) != len(what):
                raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
            for x, y in zip(where, what):
                self._content[x] = self._invert(y)

        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            raise NotImplementedError("in-place operations not supported")

        if method != "__call__":
            return NotImplemented

        inputs = list(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], IndexedArray):
                inputs[i]._valid()
                inputs[i] = inputs[i][:]

        return getattr(ufunc, method)(*inputs, **kwargs)

    def _reduce(self, ufunc, identity, dtype, regularaxis):
        if self._util_hasjagged(self._content):
            return self.copy(content=self._content._reduce(ufunc, identity, dtype, regularaxis))

        elif isinstance(self._content, awkward.array.table.Table):
            out = awkward.array.table.Table()
            for n, x in self._content._contents.items():
                out[n] = self.copy(content=x)
            return out._reduce(ufunc, identity, dtype, regularaxis)

        else:
            return ufunc.reduce(self._prepare(identity, dtype))

    def _prepare(self, identity, dtype):
        if isinstance(self._content, self.numpy.ndarray):
            return self._content[self._index]
        else:
            return self._content._prepare(identity, dtype)[self._index]

class SparseArray(awkward.array.base.AwkwardArrayWithContent):
    """
    SparseArray
    """

    # TODO for 1.0: replace length with an indexshape

    def __init__(self, length, index, content, default=None):
        self.length = length
        self.index = index
        self.content = content
        self.default = default

    def copy(self, length=None, index=None, content=None, default=None):
        out = self.__class__.__new__(self.__class__)
        out._length = self._length
        out._index = self._index
        out._content = self._content
        out._default = self._default
        out._inverse = self._inverse
        out._isvalid = self._isvalid
        if length is not None:
            out.length = length
        if index is not None:
            out.index = index
        if content is not None:
            out.content = content
        if default is not None:
            out.default = default
        return out

    def deepcopy(self, length=None, index=None, content=None, default=None):
        out = self.copy(length=length, index=index, content=content, default=default)
        out._index = self._util_deepcopy(out._index)
        out._content = self._util_deepcopy(out._content)
        out._inverse = self._util_deepcopy(out._inverse)
        return out

    def empty_like(self, **overrides):
        mine = {}
        mine = overrides.pop("length", self._length)
        mine = overrides.pop("default", self._default)
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.empty_like(self._content), **mine)
        else:
            return self.copy(content=self._content.empty_like(**overrides), **mine)

    def zeros_like(self, **overrides):
        mine = {}
        mine = overrides.pop("length", self._length)
        mine = overrides.pop("default", self._default)
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.zeros_like(self._content), **mine)
        else:
            return self.copy(content=self._content.zeros_like(**overrides), **mine)

    def ones_like(self, **overrides):
        mine = {}
        mine = overrides.pop("length", self._length)
        mine = overrides.pop("default", self._default)
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.ones_like(self._content), **mine)
        else:
            return self.copy(content=self._content.ones_like(**overrides), **mine)

    def __awkward_persist__(self, ident, fill, prefix, suffix, schemasuffix, storage, compression, **kwargs):
        self._valid()
        
        if self._default is None:
            default = {"json": self._default}
        elif self._util_isinteger(self._default):
            default = {"json": int(self._default)}
        elif isinstance(self._default, (numbers.Real, self.numpy.floating)) and self.numpy.isfinite(self._default):
            default = {"json": float(self._default)}
        else:
            default = fill(self._default, "SparseArray.default", prefix, suffix, schemasuffix, storage, compression, **kwargs)

        return {"id": ident,
                "call": ["awkward", "SparseArray"],
                "args": [{"json": int(self._length)},
                         fill(self._index, "SparseArray.index", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                         fill(self._content, "SparseArray.content", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                         default]}

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        if self.check_prop_valid:
            if not self._util_isinteger(value):
                raise TypeError("length must be an integer")
            if value < 0:
                raise ValueError("length must be a non-negative integer") 
        self._length = value

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        value = self._util_toarray(value, self.INDEXTYPE, self.numpy.ndarray)
        if self.check_prop_valid:
            if not self._util_isintegertype(value.dtype.type):
                raise TypeError("index must have integer dtype")
            if len(value.shape) != 1:
                raise ValueError("index must be one-dimensional")
            if (value < 0).any():
                raise ValueError("index must be a non-negative array")
            if len(value) > 0 and not (value[1:] >= value[:-1]).all():
                raise ValueError("index must be monatonically increasing")
        self._index = value
        self._inverse = None
        self._isvalid = False

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = self._util_toarray(value, self.DEFAULTTYPE)
        self._isvalid = False

    @property
    def default(self):
        import awkward.array.jagged

        if self._default is None:
            if isinstance(self._content, awkward.array.jagged.JaggedArray):
                return self.JaggedArray([0], [0], self._content.content)
            elif self._content.shape[1:] == ():
                return self._content.dtype.type(0)
            else:
                return self.numpy.zeros(self._content.shape[1:], dtype=self._content.dtype)

        else:
            return self._default

        self._isvalid = False

    @default.setter
    def default(self, value):
        self._default = value

    def _gettype(self, seen):
        return awkward.type._fromarray(self._content, seen)

    def _getnbytes(self, seen):
        if id(self) in seen:
            return 0
        else:
            seen.add(id(self))
            return self._index.nbytes + (self._content.nbytes if isinstance(self._content, self.numpy.ndarray) else self._content._getnbytes(seen))

    def __len__(self):
        return self._length

    def _valid(self):
        if self.check_whole_valid:
            if not self._isvalid:
                if len(self._index) > len(self._content):
                    raise ValueError("length of index ({0}) must not be greater than the length of content ({1})".format(len(self._index), len(self._content)))

                self._isvalid = True

    def __iter__(self, checkiter=True):
        if checkiter:
            self._checkiter()
        self._valid()

        length = self._length
        index = self._index
        lenindex = len(self._index)
        content = self._content
        default = self.default

        i = 0
        j = self.numpy.searchsorted(index, 0, side="left")
        while i != length:
            if j == lenindex:
                yield default
            elif index[j] == i:
                yield content[j]
                while j != lenindex and index[j] == i:
                    j += 1
            else:
                yield default
            i += 1

    def __getitem__(self, where):
        self._valid()

        if self._util_isstringslice(where):
            content = self._content[where]
            cls = awkward.array.objects.Methods.maybemixin(type(content), self.SparseArray)
            out = cls.__new__(cls)
            out.__dict__.update(self.__dict__)
            out._content = content
            return out

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        if self._util_isinteger(head):
            original_head = head
            if head < 0:
                head += self._length
            if not 0 <= head < self._length:
                raise IndexError("index {0} is out of bounds for size {1}".format(original_head, length))

            match = self.numpy.searchsorted(self._index, head, side="left")

            if match < len(self._index) and self._index[match] == head:
                return self._content[(match,) + tail]
            elif tail == ():
                return self.default
            else:
                return self.default[tail]

        elif isinstance(head, slice):
            start, stop, step = head.indices(self._length)

            if step == 0:
                raise ValueError("slice step cannot be zero")
            elif step > 0:
                mask = (self._index < stop)
                mask &= (self._index >= start)
                index = self._index - start
            elif step < 0:
                mask = (self._index > stop)
                mask &= (self._index <= start)
                index = start - self._index

            if (step > 0 and stop - start > 0) or (step < 0 and stop - start < 0):
                d, m = divmod(abs(start - stop), abs(step))
                length = d + (1 if m != 0 else 0)
            else:
                length = 0
            
            if abs(step) > 1:
                index, remainder = self.numpy.divmod(index, abs(step))
                mask[remainder != 0] = False

            index = index[mask]
            content = self._content[mask]
            if step < 0:
                index = index[::-1]
                content = content[::-1]

            return self.copy(length=length, index=index, content=content)[tail]

        elif isinstance(head, SparseArray) and len(head.shape) == 1 and issubclass(head.dtype.type, (self.numpy.bool, self.numpy.bool_)):
            head._valid()
            if self._length != head._length:
                raise IndexError("boolean index did not match indexed array along dimension 0; dimension is {0} but corresponding boolean dimension is {1}".format(self._length, head._length))

            # the new index is a cumsum (starting at zero) of the boolean values
            index = self.numpy.cumsum(head._content)
            length = index[-1]
            index[1:] = index[:-1]
            index[0] = 0

            # find my sparse elements in the mask's sparse elements
            match1 = self.numpy.searchsorted(head._index, self._index, side="left")
            match1[match1 >= len(head._index)] = len(head._index) - 1
            content = self._content[self.numpy.logical_and(head._index[match1] == self._index, head._content[match1])]

            # find the mask's sparse elements in my sparse elements
            match2 = self.numpy.searchsorted(self._index, head._index, side="left")
            match2[match2 >= len(head._index)] = len(head._index) - 1
            index = index[self.numpy.logical_and(self._index[match2] == head._index, head._content)]

            return self.copy(length=length, index=index, content=content)

        else:
            head = self._util_toarray(head, self.INDEXTYPE)
            if len(head.shape) == 1 and issubclass(head.dtype.type, (self.numpy.bool, self.numpy.bool_)):
                if self._length != len(head):
                    raise IndexError("boolean index did not match indexed array along dimension 0; dimension is {0} but corresponding boolean dimension is {1}".format(self._length, len(head)))

                head = self.numpy.arange(self._length, dtype=self.INDEXTYPE)[head]

            if len(head.shape) == 1 and self._util_isintegertype(head.dtype.type):
                mask = (head < 0)
                if mask.any():
                    head[mask] += self._length
                if (head < 0).any() or (head >= self._length).any():
                    raise IndexError("indexes out of bounds for size {0}".format(self._length))
                
                match = self.numpy.searchsorted(self._index, head, side="left")
                match[match >= len(self._index)] = len(self._index) - 1
                explicit = (self._index[match] == head)

                tags = self.numpy.zeros(len(head), dtype=self.TAGTYPE)
                index = self.numpy.zeros(len(head), dtype=self.INDEXTYPE)
                tags[explicit] = 1
                index[explicit] = self.numpy.arange(self.numpy.count_nonzero(explicit))

                content = self._content[match[explicit]]
                default = self.numpy.array([self.default])
                return self.UnionArray(tags, index, [default, content])[tail]

            else:
                raise TypeError("cannot interpret shape {0}, dtype {1} as a fancy index or mask".format(head.shape, head.dtype))

    def _getinverse(self):
        if self._inverse is None:
            self._inverse = self.numpy.searchsorted(self._index, self.numpy.arange(self._length, dtype=self.INDEXTYPE), side="left")
            if len(self._index) > 0:
                self._inverse[self._index[-1] + 1 :] = len(self._index) - 1
        return self._inverse

    @property
    def dense(self):
        self._valid()

        if isinstance(self._content, self.numpy.ndarray):
            out = self.numpy.full(self.shape, self.default, dtype=self.dtype)
            if len(self._index) != 0:
                mask = self.boolmask(maskedwhen=True)
                out[mask] = self._content[self._inverse[mask]]
            return out

        else:
            raise NotImplementedError(type(self._content))

    def boolmask(self, maskedwhen=True):
        self._valid()

        if len(self._index) == 0:
            return self.numpy.empty(0, dtype=self.numpy.bool_)

        if maskedwhen:
            return self._index[self._getinverse()] == self.numpy.arange(self._length, dtype=self.INDEXTYPE)
        else:
            return self._index[self._getinverse()] != self.numpy.arange(self._length, dtype=self.INDEXTYPE)

    def _invert(self, what):
        if len(what) != self._length:
            raise ValueError("cannot assign array of length {0} to sparse table of length {1}".format(len(what), self._length))

        test = what[self.boolmask(maskedwhen=False)].any()
        while not isinstance(test, bool):
            test = test.any()

        if test:
            raise ValueError("cannot assign an array with non-zero elements in the undefined spots of a sparse table")

        return IndexedArray(self._inverse, what)

    def __setitem__(self, where, what):
        if isinstance(where, awkward.util.string):
            self._content[where] = self._invert(what)

        elif self._util_isstringslice(where):
            if len(where) != len(what):
                raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
            for x, y in zip(where, what):
                self._content[x] = self._invert(y)

        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            raise NotImplementedError("in-place operations not supported")

        if method != "__call__":
            return NotImplemented

        inputs = list(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], SparseArray):
                inputs[i]._valid()
                inputs[i] = inputs[i].dense   # FIXME: can do better (optimization)

        return getattr(ufunc, method)(*inputs, **kwargs)

    def _reduce(self, ufunc, identity, dtype, regularaxis):
        if self._util_hasjagged(self._content):
            return self.copy(content=self._content._reduce(ufunc, identity, dtype, regularaxis))

        elif isinstance(self._content, awkward.array.table.Table):
            out = awkward.array.table.Table()
            for n, x in self._content._contents.items():
                out[n] = self.copy(content=x)
            return out._reduce(ufunc, identity, dtype, regularaxis)

        else:
            return ufunc.reduce(self._prepare(identity, dtype))

    def _prepare(self, identity, dtype):
        if isinstance(self._content, self.numpy.ndarray):
            return self.dense
        else:
            return self.copy(content=self._content._prepare(identity, dtype)).dense
