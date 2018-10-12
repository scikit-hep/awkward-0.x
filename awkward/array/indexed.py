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

def invert(permutation):
    permutation = permutation.reshape(-1)
    out = awkward.util.numpy.zeros(permutation.max() + 1, dtype=awkward.util.INDEXTYPE)
    identity = awkward.util.numpy.arange(len(permutation))
    out[permutation] = identity
    if not awkward.util.numpy.array_equal(out[permutation], identity):
        raise ValueError("cannot invert index; it contains duplicates")
    return out

class IndexedArray(awkward.array.base.AwkwardArrayWithContent):
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
        out._index   = awkward.util.deepcopy(out._index)
        out._content = awkward.util.deepcopy(out._content)
        out._inverse = awkward.util.deepcopy(out._inverse)
        return out

    def empty_like(self, **overrides):
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.empty_like(self._content))
        else:
            return self.copy(content=self._content.empty_like(**overrides))

    def zeros_like(self, **overrides):
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.zeros_like(self._content))
        else:
            return self.copy(content=self._content.zeros_like(**overrides))

    def ones_like(self, **overrides):
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.ones_like(self._content))
        else:
            return self.copy(content=self._content.ones_like(**overrides))

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
        return awkward.type.ArrayType(*(self._index.shape + (awkward.type.fromarray(self._content).to,)))

    def _valid(self):
        if not self._isvalid:
            if len(self._index) != 0 and self._index.reshape(-1).max() > len(self._content):
                raise ValueError("maximum index ({0}) is beyond the length of the content ({1})".format(self._index.reshape(-1).max(), len(self._content)))
            self._isvalid = True

    def __iter__(self):
        self._valid()
        for i in self._index:
            yield self._content[i]

    def __getitem__(self, where):
        self._valid()

        if awkward.util.isstringslice(where):
            return self.copy(content=self._content[where])

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[:len(self._index.shape)], where[len(self._index.shape):]

        head = self._index[head]
        if len(head.shape) != 0 and len(head) == 0:
            return awkward.util.numpy.empty(0, dtype=self._content.dtype)[tail]
        else:
            return self._content[(head,) + tail]

    def _invert(self, what):
        if self._inverse is None:
            self._inverse = invert(self._index)
        return IndexedArray(self._inverse, what)

    def __setitem__(self, where, what):
        if what.shape[:len(self._index.shape)] != self._index.shape:
            raise ValueError("array to assign does not have the same starting shape as index")

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
        if method != "__call__":
            return NotImplemented

        inputs = list(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], IndexedArray):
                inputs[i]._valid()
                inputs[i] = inputs[i][:]

        return getattr(ufunc, method)(*inputs, **kwargs)

    @classmethod
    def concat(cls, first, *rest):
        raise NotImplementedError

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
        return awkward.type.ArrayType(*(self._index.shape + (self._dtype,)))

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
            self._isvalid = True

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
            if self._dtype.subdtype is None:
                return self._content[starts : starts + self._dtype.itemsize].view(self._dtype)[0]
            else:
                dt, sh = self._dtype.subdtype
                return self._content[starts : starts + self._dtype.itemsize].view(dt).reshape(sh)

        else:
            if len(starts) == 0:
                return awkward.util.numpy.empty(0, dtype=self._dtype)

            else:
                index = awkward.util.numpy.repeat(starts, self._dtype.itemsize)
                index += awkward.util.numpy.tile(awkward.util.numpy.arange(self._dtype.itemsize), len(starts))
                if self._dtype.subdtype is None:
                    return self._content[index].view(self._dtype)
                else:
                    dt, sh = self._dtype.subdtype
                    return self._content[index].view(dt).reshape((-1,) + sh)

    def __setitem__(self, where, what):
        if awkward.util.isstringslice(where):
            raise IndexError("only integers, slices (`:`), and integer or boolean arrays are valid indices")
        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __delitem__(self, where):
        if awkward.util.isstringslice(where):
            raise IndexError("only integers, slices (`:`), and integer or boolean arrays are valid indices")
        else:
            raise TypeError("invalid index for removing column from Table: {0}".format(where))

    def any(self):
        return self._content[self._index].any()

    def all(self):
        return self._content[self._index].all()

    @classmethod
    def concat(cls, first, *rest):
        raise NotImplementedError

    def pandas(self):
        raise NotImplementedError

class SparseArray(awkward.array.base.AwkwardArrayWithContent):
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
        out._index = awkward.util.deepcopy(out._index)
        out._content = awkward.util.deepcopy(out._content)
        out._inverse = awkward.util.deepcopy(out._inverse)
        return out

    def empty_like(self, **overrides):
        mine = {}
        mine = overrides.pop("length", self._length)
        mine = overrides.pop("default", self._default)
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.empty_like(self._content), **mine)
        else:
            return self.copy(content=self._content.empty_like(**overrides), **mine)

    def zeros_like(self, **overrides):
        mine = {}
        mine = overrides.pop("length", self._length)
        mine = overrides.pop("default", self._default)
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.zeros_like(self._content), **mine)
        else:
            return self.copy(content=self._content.zeros_like(**overrides), **mine)

    def ones_like(self, **overrides):
        mine = {}
        mine = overrides.pop("length", self._length)
        mine = overrides.pop("default", self._default)
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.ones_like(self._content), **mine)
        else:
            return self.copy(content=self._content.ones_like(**overrides), **mine)

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        if not isinstance(value, awkward.util.integer):
            raise TypeError("length must be an integer")
        if value < 0:
            raise ValueError("length must be a non-negative integer") 
        self._length = value

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, awkward.util.numpy.ndarray)
        if not issubclass(value.dtype.type, awkward.util.numpy.integer):
            raise TypeError("index must have integer dtype")
        if len(value.shape) != 1:
            raise ValueError("index must be one-dimensional")
        if (value < 0).any():
            raise ValueError("index must be a non-negative array")
        if len(value) > 0 and not (value[1:] >= value[:-1]).all():
            raise ValueError("index must be monatonically increasing")
        self._index = value
        self._inverse = None

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = awkward.util.toarray(value, awkward.util.DEFAULTTYPE)

    @property
    def default(self):
        import awkward.array.jagged

        if self._default is None:
            if isinstance(self._content, awkward.array.jagged.JaggedArray):
                return awkward.array.jagged.JaggedArray([0], [0], self._content.content)
            elif self._content.shape[1:] == ():
                return self._content.dtype.type(0)
            else:
                return awkward.util.numpy.zeros(self._content.shape[1:], dtype=self._content.dtype)

        else:
            return self._default

    @default.setter
    def default(self, value):
        self._default = value

    @property
    def type(self):
        return awkward.type.ArrayType(self._length, awkward.type.fromarray(self._content).to)
        
    def __len__(self):
        return self._length

    @property
    def shape(self):
        return (self._length,) + self._content.shape[1:]

    @property
    def dtype(self):
        return self._content.dtype

    def _valid(self):
        if len(self._index) > len(self._content):
            raise ValueError("length of index ({0}) must not be greater than the length of content ({1})".format(len(self._index), len(self._content)))

    def __iter__(self):
        self._valid()

        length = self._length
        index = self._index
        lenindex = len(self._index)
        content = self._content
        default = self.default

        i = 0
        j = awkward.util.numpy.searchsorted(index, 0, side="left")
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
        import awkward.array.union
        self._valid()

        if awkward.util.isstringslice(where):
            return self.copy(content=self._content[where])

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        if isinstance(head, awkward.util.integer):
            original_head = head
            if head < 0:
                head += self._length
            if not 0 <= head < self._length:
                raise IndexError("index {0} is out of bounds for size {1}".format(original_head, length))

            match = awkward.util.numpy.searchsorted(self._index, head, side="left")

            if self._index[match] == head:
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
                index, remainder = awkward.util.numpy.divmod(index, abs(step))
                mask[remainder != 0] = False

            index = index[mask]
            content = self._content[mask]
            if step < 0:
                index = index[::-1]
                content = content[::-1]

            return self.copy(length=length, index=index, content=content)[tail]

        elif isinstance(head, SparseArray) and len(head.shape) == 1 and issubclass(head.dtype.type, (awkward.util.numpy.bool, awkward.util.numpy.bool_)):
            head._valid()
            if self._length != head._length:
                raise IndexError("boolean index did not match indexed array along dimension 0; dimension is {0} but corresponding boolean dimension is {1}".format(self._length, head._length))

            # the new index is a cumsum (starting at zero) of the boolean values
            index = awkward.util.numpy.cumsum(head._content)
            length = index[-1]
            index[1:] = index[:-1]
            index[0] = 0

            # find my sparse elements in the mask's sparse elements
            match1 = awkward.util.numpy.searchsorted(head._index, self._index, side="left")
            match1[match1 >= len(head._index)] = len(head._index) - 1
            content = self._content[awkward.util.numpy.logical_and(head._index[match1] == self._index, head._content[match1])]

            # find the mask's sparse elements in my sparse elements
            match2 = awkward.util.numpy.searchsorted(self._index, head._index, side="left")
            index = index[awkward.util.numpy.logical_and(self._index[match2] == head._index, head._content)]

            return self.copy(length=length, index=index, content=content)

        else:
            head = awkward.util.toarray(head, awkward.util.INDEXTYPE)
            if len(head.shape) == 1 and issubclass(head.dtype.type, (awkward.util.numpy.bool, awkward.util.numpy.bool_)):
                if self._length != len(head):
                    raise IndexError("boolean index did not match indexed array along dimension 0; dimension is {0} but corresponding boolean dimension is {1}".format(self._length, len(head)))

                head = awkward.util.numpy.arange(self._length, dtype=awkward.util.INDEXTYPE)[head]

            if len(head.shape) == 1 and issubclass(head.dtype.type, awkward.util.numpy.integer):
                mask = (head < 0)
                if mask.any():
                    head[mask] += self._length
                if (head < 0).any() or (head >= self._length).any():
                    raise IndexError("indexes out of bounds for size {0}".format(self._length))
                
                match = awkward.util.numpy.searchsorted(self._index, head, side="left")
                explicit = (self._index[match] == head)

                tags = awkward.util.numpy.zeros(len(head), dtype=awkward.util.TAGTYPE)
                index = awkward.util.numpy.zeros(len(head), dtype=awkward.util.INDEXTYPE)
                tags[explicit] = 1
                index[explicit] = awkward.util.numpy.arange(awkward.util.numpy.count_nonzero(explicit))

                content = self._content[match[explicit]]
                default = awkward.util.numpy.array([self.default])
                return awkward.array.union.UnionArray(tags, index, [default, content])[tail]

            else:
                raise TypeError("cannot interpret shape {0}, dtype {1} as a fancy index or mask".format(head.shape, head.dtype))

    def _getinverse(self):
        if self._inverse is None:
            self._inverse = awkward.util.numpy.searchsorted(self._index, awkward.util.numpy.arange(self._length, dtype=awkward.util.INDEXTYPE), side="left")
            if len(self._index) > 0:
                self._inverse[self._index[-1] + 1 :] = len(self._index) - 1
        return self._inverse

    @property
    def dense(self):
        self._valid()

        if isinstance(self._content, awkward.util.numpy.ndarray):
            out = awkward.util.numpy.full(self.shape, self.default, dtype=self.dtype)
            if len(self._index) != 0:
                mask = self.boolmask(maskedwhen=True)
                out[mask] = self._content[self._inverse[mask]]
            return out

        else:
            raise NotImplementedError(type(self._content))

    def boolmask(self, maskedwhen=True):
        self._valid()

        if len(self._index) == 0:
            return awkward.util.numpy.empty(0, dtype=awkward.util.numpy.bool_)

        if maskedwhen:
            return self._index[self._getinverse()] == awkward.util.numpy.arange(self._length, dtype=awkward.util.INDEXTYPE)
        else:
            return self._index[self._getinverse()] != awkward.util.numpy.arange(self._length, dtype=awkward.util.INDEXTYPE)

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

        elif awkward.util.isstringslice(where):
            if len(where) != len(what):
                raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
            for x, y in zip(where, what):
                self._content[x] = self._invert(y)

        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        inputs = list(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], SparseArray):
                inputs[i]._valid()
                inputs[i] = inputs[i].dense   # FIXME: can do better (optimization)

        return getattr(ufunc, method)(*inputs, **kwargs)

    @classmethod
    def concat(cls, first, *rest):
        raise NotImplementedError

    def pandas(self):
        raise NotImplementedError
