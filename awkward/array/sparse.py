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

class SparseArray(awkward.array.base.AwkwardArrayWithContent):
    def __init__(self, length, coordinates, content):
        self.length = length
        self.coordinates = coordinates
        self.content = content

    @classmethod
    def fromCOO(cls, length, coordinates, content, default=0):
        content = awkward.util.toarray(content, awkward.util.DEFAULTTYPE)
        if isinstance(content, awkward.util.numpy.ndarray):
            content = awkward.util.numpy.insert(content, 0, default)
        else:
            # FIXME: maybe something with an IndexedArray of ChunkedArrays?
            raise NotImplementedError(type(content))
        return cls(length, coordinates, content)

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
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        if not isinstance(value, awkward.util.integer) or value < 0:
            raise TypeError("length must be a non-negative integer")
        self._length = value

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, awkward.util.numpy.ndarray)
        if not issubclass(value.dtype.type, awkward.util.numpy.integer):
            raise TypeError("coordinates must have integer dtype")
        if len(value.shape) != 1:
            raise TypeError("coordinates must be one-dimensional")
        if (value < 0).any():
            raise ValueError("coordinates must be a non-negative array")
        if len(value) > 0 and not (value[1:] >= value[:-1]).all():
            raise ValueError("coordinates must be monotonically increasing")
        self._coordinates = value

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = awkward.util.toarray(value, awkward.util.DEFAULTTYPE)

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
        if len(self._coordinates) >= len(self._content):
            raise ValueError("length of coordinates ({0}) must be less than the length of content ({1}); not equal because content[0] is the default value".format(len(self._coordinates), len(self._content)))

    def __iter__(self):
        self._valid()

        coords = self._coordinates
        content = self._content
        default = content[0]

        j = 0
        for i in range(self._length):
            if j >= len(coords):
                yield default
            elif coords[j] == i:
                yield content[j + 1]
                while j < len(coords) and coords[j] == i:
                    j += 1
            else:
                yield default

    def __getitem__(self, where):
        self._valid()

        if awkward.util.isstringslice(where):
            return self.copy(self._length, self._coordinates, self._content[where])

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
                raise IndexError("index {0} is out of bounds for size {1}".format(original_head, self._length))
            i = awkward.util.numpy.searchsorted(self._coordinates, head, side="left")
            if self._coordinates[i] == head:
                return self._content[(i + 1,) + tail]
            else:
                return self._content[(0,) + tail]

        if isinstance(head, slice):
            start, stop, step = head.indices(self._length)
            head = awkward.util.numpy.arange(start, stop, step)

        head = awkward.util.numpy.array(head, copy=False)
        if len(head.shape) == 1 and issubclass(head.dtype.type, (awkward.util.numpy.bool, awkward.util.numpy.bool_)):
            if self._length != len(head):
                raise IndexError("boolean index did not match indexed array along dimension 0; dimension is {0} but corresponding boolean dimension is {1}".format(self._length, len(head)))

            head = awkward.util.numpy.arange(self._length)[head]

        if len(head.shape) == 1 and issubclass(head.dtype.type, awkward.util.numpy.integer):
            mask = (head < 0)
            if mask.any():
                head[mask] += self._length
            if (head < 0).any() or (head >= self._length).any():
                raise IndexError("index is out of bounds")

            index = awkward.util.numpy.searchsorted(self._coordinates, head, side="left")
            index[index >= len(self._coordinates)] = len(self._coordinates) - 1
            missed = (self._coordinates[index] != head)
            index += 1
            index[missed] = 0
            return self._content[(index,) + tail]

        else:
            raise TypeError("cannot interpret shape {0}, dtype {1} as a fancy index or mask".format(head.shape, head.dtype))

    def __setitem__(self, where, what):
        raise NotImplementedError

    def __delitem__(self, where, what):
        raise NotImplementedError

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError

    def any(self):
        raise NotImplementedError

    def all(self):
        raise NotImplementedError

    @classmethod
    def concat(cls, first, *rest):
        raise NotImplementedError

    def pandas(self):
        raise NotImplementedError

class PaddedJaggedArray(awkward.array.base.AwkwardArrayWithContent):
    def __init__(self, starts, stops, length, content):
        self.starts = starts
        self.stops = stops
        self.length = length
        self.content = content

    @property
    def starts(self):
        return self._starts

    @starts.setter
    def starts(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, awkward.util.numpy.ndarray)
        if not issubclass(value.dtype.type, awkward.util.numpy.integer):
            raise TypeError("starts must have integer dtype")
        if len(value.shape) == 0:
            raise TypeError("starts must have at least one dimension")
        if (value < 0).any():
            raise ValueError("starts must be a non-negative array")
        self._starts = value
        self._offsets, self._counts, self._parents = None, None, None
        self._isvalid = False
        
    @property
    def stops(self):
        return self._stops

    @stops.setter
    def stops(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, awkward.util.numpy.ndarray)
        if not issubclass(value.dtype.type, awkward.util.numpy.integer):
            raise TypeError("stops must have integer dtype")
        if len(value.shape) == 0:
            raise TypeError("stops must have at least one dimension")
        if (value < 0).any():
            raise ValueError("stops must be a non-negative array")
        self._stops = value
        self._offsets, self._counts, self._parents = None, None, None
        self._isvalid = False

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        if not isinstance(value, awkward.util.integer) or value < 0:
            raise TypeError("length must be a non-negative integer")
        self._length = value

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
        self._valid()
        return self._starts.shape + (self._length,) + self._content.shape[1:]

    @property
    def type(self):
        return awkward.type.ArrayType(*(self._starts.shape + (self._length,) + awkward.type.fromarray(self._content).to))

    def _valid(self):
        import awkward.array.jagged
        if not self._isvalid:
            awkward.array.jagged.JaggedArray._validstartsstops(self._starts, self._stops)
            
            nonempty = (self._starts != self._stops)

            starts = self._starts[nonempty].reshape(-1)
            if len(starts) != 0 and starts.reshape(-1).max() >= len(self._content):
                raise ValueError("maximum start ({0}) is at or beyond the length of the content ({1})".format(starts.reshape(-1).max(), len(self._content)))

            stops = self._stops[nonempty].reshape(-1)
            if len(stops) != 0 and stops.reshape(-1).max() > len(self._content):
                raise ValueError("maximum stop ({0}) is beyond the length of the content ({1})".format(self._stops.reshape(-1).max(), len(self._content)))

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, where):
        self._valid()
        
        if awkward.util.isstringslice(where):
            return self.copy(self._length, self._coordinates, self._content[where])

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[:len(self._starts.shape)], where[len(self._starts.shape):]

        starts = self._starts[head]
        stops = self._stops[head]
        if len(starts.shape) == len(stops.shape) == 0:
            index = awkward.util.numpy.zeros(self._length, dtype=awkward.util.INDEXTYPE)
            index[: stops - starts] = awkward.util.numpy.arange(starts + 1, stops + 1)
        else:
            index = awkward.util.numpy.zeros(starts.shape + (self._length,), dtype=awkward.util.INDEXTYPE)
            HERE


        return self._content[index]
