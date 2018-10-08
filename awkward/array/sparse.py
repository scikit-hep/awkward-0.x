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
    def __init__(self, coordshape, coordinates, content, default=None):
        self.coordshape = coordshape
        self.coordinates = coordinates
        self.content = content
        self.default = default

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
    def coordshape(self):
        return self._coordshape

    @coordshape.setter
    def coordshape(self, value):
        if isinstance(value, awkward.util.integer):
            value = (value,)
        value = tuple(value)
        if not all(isinstance(x, awkward.util.integer) and x >= 0 for x in value):
            raise TypeError("coordshape must be a tuple of non-negative integers")
        if len(value) == 0:
            raise TypeError("coordshape must have at least one dimension")
        self._coordshape = value
        self._linearized = None

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, awkward.util.numpy.ndarray)
        if not issubclass(value.dtype.type, awkward.util.numpy.integer):
            raise TypeError("coordinates must have integer dtype")
        if len(value.shape) == 0:
            raise TypeError("coordinates must have at least one dimension")
        if (value < 0).any():
            raise ValueError("coordinates must be a non-negative array")
        if len(value) > 0 and not (value[1:] >= value[:-1]).all():
            raise ValueError("coordinates must be monotonically increasing")
        self._coordinates = value
        self._linearized = None

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = awkward.util.toarray(value, awkward.util.DEFAULTTYPE)

    @property
    def default(self):
        if self._default is None:
            return awkward.util.numpy.zeros(len(self._content.shape[1:]), dtype=self._content.dtype)
        else:
            return self._default

    @default.setter
    def default(self, value):
        self._default = value

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
        if self._linearized is None:
            if len(self._coordshape) != len(self._coordinates.shape) - 1:
                raise ValueError("length of coordshape ({0}) differs from coordinates dimensionality ({1}, length of coordinates.shape minus 1)".format(len(self._coordshape), len(self._coordinates.shape) - 1))
            self._linearized = (awkward.util.numpy.array(self._coordshape, dtype=awkward.util.INDEXTYPE) * self._coordinates).sum(axis=-1)

        if self._default is not None:
            if not isinstance(self._default, self._content.dtype.type):
                default = self._content.dtype.type(self._default)
            else:
                default = self._default
            if default.shape != self._content.shape[1:]:
                raise ValueError("default shape ({0}) does not match content dimensionality ({1}, which is content.shape[1:])".format(default.shape, self._content.shape[1:]))
            self._default = default

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, where):
        self._valid()

        if awkward.util.isstringslice(where):
            out = self.copy(self._starts, self._stops, self._content[where])
            out._offsets = self._offsets
            out._counts = self._counts
            out._parents = self._parents
            out._isvalid = True
            return out

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[:len(self._coordshape)], where[len(self._coordshape):]

        coordshape = awkward.util.numpy.array(self._coordshape, dtype=awkward.util.INDEXTYPE)

        if all(isinstance(x, awkward.util.integer) for x in head):
            original_head = head
            head = awkward.util.numpy.array(head, copy=True)
            mask = (head < 0)
            if mask.any():
                head[mask] -= self._coordshape[mask]
            if (head < 0).any() or (head >= self._coordshape).any():
                raise IndexError("index {0} is out of bounds for coordshape {1}".format(original_head, self._coordshape))
            



            
            coord = self._coordinates
            for i in range(len(head)):
                h = head[i]
                if h < 0:
                    h -= self._coordshape[i]
                if not 0 <= h < self._coordshape[i]:
                    raise IndexError("index {0} is out of bounds for coordshape {1}".format(head, self._coordshape))
                c = coord[(slice(None),) * (len(coord.shape) - 1) + (0,)]
                coord = coord[(slice(None),) * (len(coord.shape) - 1)]


                j = awkward.util.numpy.searchsorted(coord[0], h, side="left")
                if coord[0] == h
                    





                    
            coord = self._coordinates
            for x in head:
                y = awkward.util.numpy.searchsorted(coord[0], x, side="left")
                if coord[0, y] == x:
                    coord = coord[1:]
                    



        if isinstance(self._content, awkward.util.numpy.ndarray):
            




        
        else:
            # FIXME: other cases will require fancy ChunkedArrays and/or IndexedArrays
            raise NotImplementedError(type(self._content))

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
