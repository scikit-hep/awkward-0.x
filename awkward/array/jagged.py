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
import math
import numbers
import types

import awkward.array.base
import awkward.type
import awkward.util

def offsetsaliased(starts, stops):
    return (isinstance(starts, awkward.util.numpy.ndarray) and isinstance(stops, awkward.util.numpy.ndarray) and
            starts.base is not None and stops.base is not None and starts.base is stops.base and
            starts.ctypes.data == starts.base.ctypes.data and
            stops.ctypes.data == stops.base.ctypes.data + stops.dtype.itemsize and
            len(starts) == len(starts.base) - 1 and
            len(stops) == len(stops.base) - 1)

def counts2offsets(counts):
    offsets = awkward.util.numpy.empty(len(counts) + 1, dtype=awkward.util.INDEXTYPE)
    offsets[0] = 0
    awkward.util.numpy.cumsum(counts, out=offsets[1:])
    return offsets

def offsets2parents(offsets):
    out = awkward.util.numpy.zeros(offsets[-1], dtype=awkward.util.INDEXTYPE)
    awkward.util.numpy.add.at(out, offsets[offsets != offsets[-1]][1:], 1)
    awkward.util.numpy.cumsum(out, out=out)
    return out

def startsstops2parents(starts, stops):
    out = awkward.util.numpy.full(stops.max(), -1, dtype=awkward.util.INDEXTYPE)
    lenstarts = len(starts)
    i = 0
    while i < lenstarts:
        out[starts[i]:stops[i]] = i
        i += 1
    return out

def parents2startsstops(parents):
    # assumes that children are contiguous, but not necessarily in order or fully covering (allows empty lists)
    tmp = awkward.util.numpy.nonzero(parents[1:] != parents[:-1])[0] + 1
    changes = awkward.util.numpy.empty(len(tmp) + 2, dtype=awkward.util.INDEXTYPE)
    changes[0] = 0
    changes[-1] = len(parents)
    changes[1:-1] = tmp

    length = parents.max() + 1
    starts = awkward.util.numpy.zeros(length, dtype=awkward.util.INDEXTYPE)
    counts = awkward.util.numpy.zeros(length, dtype=awkward.util.INDEXTYPE)

    where = parents[changes[:-1]]
    real = (where >= 0)

    starts[where[real]] = (changes[:-1])[real]
    counts[where[real]] = (changes[1:] - changes[:-1])[real]

    return starts, starts + counts

def uniques2offsetsparents(uniques):
    # assumes that children are contiguous, in order, and fully covering (can't have empty lists)
    # values are ignored, apart from uniqueness
    changes = awkward.util.numpy.nonzero(uniques[1:] != uniques[:-1])[0] + 1

    offsets = awkward.util.numpy.empty(len(changes) + 2, dtype=awkward.util.INDEXTYPE)
    offsets[0] = 0
    offsets[-1] = len(uniques)
    offsets[1:-1] = changes

    parents = awkward.util.numpy.zeros(len(uniques), dtype=awkward.util.INDEXTYPE)
    parents[changes] = 1
    awkward.util.numpy.cumsum(parents, out=parents)

    return offsets, parents

class JaggedArray(awkward.array.base.AwkwardArray):
    def __init__(self, starts, stops, content):
        self.starts = starts
        self.stops = stops
        self.content = content

    @classmethod
    def fromiter(cls, iterable):
        offsets = [0]
        content = []
        for x in iterable:
            offsets.append(offsets[-1] + len(x))
            content.extend(x)
        return cls.fromoffsets(offsets, content)

    @classmethod
    def fromoffsets(cls, offsets, content):
        offsets = awkward.util.toarray(offsets, awkward.util.INDEXTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
        if len(offsets.shape) != 1 or (offsets < 0).any():
            raise ValueError("offsets must be a one-dimensional, non-negative array")
        out = cls(offsets[:-1], offsets[1:], content)
        out._offsets = offsets
        return out

    @classmethod
    def fromcounts(cls, counts, content):
        counts = awkward.util.toarray(counts, awkward.util.INDEXTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
        if (counts < 0).any():
            raise ValueError("counts must be a non-negative array")
        offsets = counts2offsets(counts.reshape(-1))
        out = cls(offsets[:-1].reshape(counts.shape), offsets[1:].reshape(counts.shape), content)
        out._offsets = offsets if len(counts.shape) == 1 else None
        out._counts = counts
        return out

    @classmethod
    def fromparents(cls, parents, content):
        parents = awkward.util.toarray(parents, awkward.util.INDEXTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
        if len(parents.shape) != 1 or len(parents) != len(content):
            raise ValueError("parents array must be one-dimensional with the same length as content")
        starts, stops = parents2startsstops(parents)
        out = cls(starts, stops, content)
        out._parents = parents
        return out

    @classmethod
    def fromuniques(cls, uniques, content):
        uniques = awkward.util.toarray(uniques, awkward.util.INDEXTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
        if len(uniques.shape) != 1 or len(uniques) != len(content):
            raise ValueError("uniques array must be one-dimensional with the same length as content")
        offsets, parents = uniques2offsetsparents(uniques)
        out = cls.fromoffsets(offsets, content)        
        out._parents = parents
        return out

    @classmethod
    def fromjagged(cls, jagged):
        jagged = jagged._tojagged(copy=False)
        return cls(jagged._starts, jagged._stops, jagged._content)

    def copy(self, starts=None, stops=None, content=None):
        out = self.__class__.__new__(self.__class__)
        out._starts  = self._starts
        out._stops   = self._stops
        out._content = self._content
        out._offsets = self._offsets
        out._counts  = self._counts
        out._parents = self._parents
        out._isvalid = self._isvalid
        if starts is not None:
            out.starts = starts
        if stops is not None:
            out.stops = stops
        if content is not None:
            out.content = content
        return out

    def deepcopy(self, starts=None, stops=None, content=None):
        out = self.copy(starts=starts, stops=stops, content=content)
        out._starts  = awkward.util.deepcopy(out._starts)
        out._stops   = awkward.util.deepcopy(out._stops)
        out._content = awkward.util.deepcopy(out._content)
        out._offsets = awkward.util.deepcopy(out._offsets)
        out._counts  = awkward.util.deepcopy(out._counts)
        out._parents = awkward.util.deepcopy(out._parents)
        return out

    def empty_like(self, **overrides):
        mine = {}
        mine["starts"] = overrides.pop("starts", self._starts)
        mine["stops"] = overrides.pop("stops", self._stops)
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.empty_like(self._content), **mine)
        else:
            return self.copy(content=self._content.empty_like(**overrides), **mine)

    def zeros_like(self, **overrides):
        mine = {}
        mine["starts"] = overrides.pop("starts", self._starts)
        mine["stops"] = overrides.pop("stops", self._stops)
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.zeros_like(self._content), **mine)
        else:
            return self.copy(content=self._content.zeros_like(**overrides), **mine)

    def ones_like(self, **overrides):
        mine = {}
        mine["starts"] = overrides.pop("starts", self._starts)
        mine["stops"] = overrides.pop("stops", self._stops)
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.ones_like(self._content), **mine)
        else:
            return self.copy(content=self._content.ones_like(**overrides), **mine)

    @property
    def starts(self):
        return self._starts

    @starts.setter
    def starts(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
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
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
        if (value < 0).any():
            raise ValueError("stops must be a non-negative array")
        self._stops = value
        self._offsets, self._counts, self._parents = None, None, None
        self._isvalid = False

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = awkward.util.toarray(value, awkward.util.CHARTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
        self._isvalid = False

    @property
    def offsets(self):
        if self._offsets is None:
            self._valid()
            if offsetsaliased(self._starts, self._stops):
                self._offsets = self._starts.base
            elif len(self._starts.shape) == 1 and awkward.util.numpy.array_equal(self._starts[1:], self._stops[:-1]):
                self._offsets = awkward.util.numpy.append(self._starts, self._stops[-1])
            else:
                raise ValueError("starts and stops are not compatible with a single offsets array")
        return self._offsets

    @offsets.setter
    def offsets(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
        if len(value.shape) != 1 or (value < 0).any():
            raise ValueError("offsets must be a one-dimensional, non-negative array")
        self._starts = value[:-1]
        self._stops = value[1:]
        self._offsets = value
        self._counts, self._parents = None, None
        self._isvalid = False

    @property
    def counts(self):
        if self._counts is None:
            self._valid()
            self._counts = self._stops - self._starts
        return self._counts

    @counts.setter
    def counts(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
        if (value < 0).any():
            raise ValueError("counts must be a non-negative array")
        offsets = counts2offsets(value.reshape(-1))
        self._starts = offsets[:-1].reshape(value.shape)
        self._stops = offsets[1:].reshape(value.shape)
        self._offsets = offsets if len(value.shape) == 1 else None
        self._counts = value
        self._parents = None
        self._isvalid = False

    @property
    def parents(self):
        if self._parents is None:
            self._valid()
            try:
                self._parents = offsets2parents(self.offsets)
            except ValueError:
                self._parents = startsstops2parents(self._starts, self._stops)
        return self._parents

    @parents.setter
    def parents(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
        if len(value) != len(content):
            raise ValueError("parents array must have the same length as content")
        self._starts, self._stops = parents2startsstops(value)
        self._offsets, self._counts = None, None
        self._parents = value

    @property
    def index(self):
        out = awkward.util.numpy.arange(len(self._content), dtype=awkward.util.INDEXTYPE)
        return self.copy(content=(out - out[self._starts[self.parents]]))

    @property
    def type(self):
        return awkward.type.ArrayType(*(self._starts.shape + (awkward.type.ArrayType(awkward.util.numpy.inf, awkward.type.fromarray(self._content).to),)))

    def __len__(self):
        self._valid()
        return len(self._starts)

    @property
    def shape(self):
        self._valid()
        return self._starts.shape

    @property
    def columns(self):
        return self._content.columns

    @property
    def allcolumns(self):
        return self._content.allcolumns

    @property
    def base(self):
        return self._content.base

    def _argfields(self, function):
        if (isinstance(function, types.FunctionType) and function.__code__.co_argcount == 1) or isinstance(self._content, awkward.util.numpy.ndarray):
            return awkward.util._argfields(function)
        else:
            return self._content._argfields(function)

    @property
    def dtype(self):
        return awkward.util.numpy.dtype(awkward.util.numpy.object)      # specifically, Numpy arrays

    @staticmethod
    def _validstartsstops(starts, stops):
        if len(starts.shape) == 0:
            raise TypeError("starts must have at least one dimension")
        if starts.shape[0] == 0:
            starts = starts.view(awkward.util.INDEXTYPE)
        if not issubclass(starts.dtype.type, awkward.util.numpy.integer):
            raise TypeError("starts must have integer dtype")

        if len(stops.shape) != len(starts.shape):
            raise TypeError("stops must have the same shape as starts")
        if stops.shape[0] == 0:
            stops = stops.view(awkward.util.INDEXTYPE)
        if not issubclass(stops.dtype.type, awkward.util.numpy.integer):
            raise TypeError("stops must have integer dtype")

        if len(starts) > len(stops):
            raise ValueError("starts must not have more elements than stops")

    def _valid(self):
        if not self._isvalid:
            self._validstartsstops(self._starts, self._stops)

            stops = self._stops[self._starts != self._stops].reshape(-1)
            if len(stops) != 0 and stops.max() > len(self._content):
                raise ValueError("maximum stop ({0}) is beyond the length of the content ({1})".format(self._stops.reshape(-1).max(), len(self._content)))

            self._isvalid = True

    def __iter__(self):
        self._valid()
        if len(self._starts.shape) != 1:
            for x in super(JaggedArray, self).__iter__():
                yield x
        else:
            stops = self._stops
            content = self._content
            for i, start in enumerate(self._starts):
                yield content[start:stops[i]]

    def __getitem__(self, where):
        self._valid()

        if awkward.util.isstringslice(where):
            out = JaggedArray(self._starts, self._stops, self._content[where])
            out._offsets = self._offsets
            out._counts = self._counts
            out._parents = self._parents
            out._isvalid = False
            return out

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        if len(self._starts.shape) == 1:
            head, tail = where[0], where[1:]
        else:
            head, tail = where[:len(self._starts.shape)], where[len(self._starts.shape):]

        if isinstance(head, JaggedArray):
            if issubclass(head._content.dtype.type, awkward.util.numpy.integer):
                if len(head.shape) == 1 and head._starts.shape != self._starts.shape:
                    raise ValueError("jagged array used as index has a different shape {0} from the jagged array it is selecting from {1}".format(head._starts.shape, self._starts.shape))

                headoffsets = counts2offsets(head.counts)
                head = head._tojagged(headoffsets[:-1], headoffsets[1:], copy=False)

                counts = head._broadcast(self.counts)._content

                indexes = awkward.util.numpy.array(head._content, copy=True)

                negatives = (indexes < 0)
                indexes[negatives] += counts[negatives]

                if not awkward.util.numpy.bitwise_and(0 <= indexes, indexes < counts).all():
                    raise IndexError("jagged array used as index contains out-of-bounds values")

                indexes += head._broadcast(self._starts)._content

                return self.copy(starts=head._starts, stops=head._stops, content=self._content[indexes])

            elif len(head.shape) == 1 and issubclass(head._content.dtype.type, (awkward.util.numpy.bool, awkward.util.numpy.bool_)):
                try:
                    offsets = self.offsets
                    thyself = self

                except ValueError:
                    offsets = counts2offsets(self.counts.reshape(-1))
                    thyself = self._tojagged(offsets[:-1], offsets[1:], copy=False)
                    thyself._starts.shape = self._starts.shape
                    thyself._stops.shape = self._stops.shape

                head = head._tojagged(thyself._starts, thyself._stops, copy=False)
                inthead = head.copy(content=head._content.view(awkward.util.numpy.uint8))
                intheadsum = inthead.sum()

                offsets = counts2offsets(intheadsum)

                return self.copy(starts=offsets[:-1].reshape(intheadsum.shape), stops=offsets[1:].reshape(intheadsum.shape), content=thyself._content[head._content])

            elif len(head.shape) == 1:
                raise TypeError("jagged index must be boolean (mask) or integer (fancy indexing)")

            else:
                # the other cases are possible, but complicated; the first sets the form
                raise NotImplementedError("jagged index content type: {0}".format(head._content.dtype))

        else:
            starts = self._starts[head]
            stops = self._stops[head]
            if len(starts.shape) == len(stops.shape) == 0:
                return self._content[starts:stops][tail]
            else:
                node = self.copy(starts=starts, stops=stops)

        while isinstance(node, JaggedArray) and len(tail) > 0:
            head, tail = tail[0], tail[1:]
            original_head = head
            if isinstance(head, (numbers.Integral, awkward.util.numpy.integer)):
                counts = node._stops - node._starts
                if head < 0:
                    head = counts + head
                if not awkward.util.numpy.bitwise_and(0 <= head, head < counts).all():
                    raise IndexError("index {0} is out of bounds for jagged min size {1}".format(original_head, counts.min()))
                node = node._content[node._starts + head]
            else:
                # the other cases are possible, but complicated; the first sets the form
                raise NotImplementedError("jagged second dimension index type: {0}".format(original_head))

        return node[tail]

    def __setitem__(self, where, what):
        if isinstance(where, awkward.util.string):
            if isinstance(what, JaggedArray):
                self._content[where] = what._tojagged(self._starts, self._stops, copy=False)._content
            else:
                self._content[where] = self._broadcast(what)._content

        elif awkward.util.isstringslice(where):
            if len(where) != len(what):
                raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
            for x, y in zip(where, what):
                if isinstance(y, JaggedArray):
                    self._content[x] = y._tojagged(self._starts, self._stops, copy=False)._content
                else:
                    self._content[x] = self._broadcast(y)._content
                
        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    @classmethod
    def zip(cls, columns1={}, *columns2, **columns3):
        import awkward.array.table
        table = awkward.array.table.Table(0, columns1, *columns2, **columns3)
        inputs = list(table._content.values())
        table._length = min(len(x) for x in inputs)

        first = None
        for i in range(len(inputs)):
            if isinstance(inputs[i], JaggedArray):
                if first is None:
                    first = inputs[i] = inputs[i]._tojagged(copy=False)
                else:
                    inputs[i] = inputs[i]._tojagged(first._starts, first._stops, copy=False)

        if first is None:
            return table

        for i in range(len(inputs)):
            if not isinstance(inputs[i], JaggedArray):
                inputs[i] = first._broadcast(inputs[i])

        newtable = awkward.array.table.Table(len(first._content), awkward.util.OrderedDict(zip(table._content, [x._content for x in inputs])))
        return cls(first._starts, first._stops, newtable)

    def _broadcast(self, data):
        data = awkward.util.toarray(data, self._content.dtype, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
        good = (self.parents >= 0)
        content = awkward.util.numpy.empty(len(self.parents), dtype=data.dtype)
        if len(data.shape) == 0:
            content[good] = data
        else:
            content[good] = data[self.parents[good]]
        out = self.copy(content=content)
        out._parents = self.parents
        return out

    def _tojagged(self, starts=None, stops=None, copy=True):
        if starts is None and stops is None:
            if copy:
                starts, stops = awkward.util.deepcopy(self._starts), awkward.util.deepcopy(self._stops)
            else:
                starts, stops = self._starts, self._stops

        elif stops is None:
            starts = awkward.util.toarray(starts, awkward.util.INDEXTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
            if len(self) != len(starts):
                raise IndexError("cannot fit JaggedArray of length {0} into starts of length {1}".format(len(self), len(starts)))

            stops = starts + self.counts

            if (stops[:-1] > starts[1:]).any():
                raise IndexError("cannot fit contents of JaggedArray into the given starts array")

        elif starts is None:
            stops = awkward.util.toarray(stops, awkward.util.INDEXTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
            if len(self) != len(stops):
                raise IndexError("cannot fit JaggedArray of length {0} into stops of length {1}".format(len(self), len(stops)))

            starts = stops - self.counts

            if (stops[:-1] > starts[1:]).any():
                raise IndexError("cannot fit contents of JaggedArray into the given stops array")

        else:
            if not awkward.util.numpy.array_equal(stops - starts, self.counts):
                raise IndexError("cannot fit contents of JaggedArray into the given starts and stops arrays")

        self._validstartsstops(starts, stops)

        if not copy and starts is self._starts and stops is self._stops:
            return self

        elif (starts is self._starts or awkward.util.numpy.array_equal(starts, self._starts)) and (stops is self._stops or awkward.util.numpy.array_equal(stops, self._stops)):
            return self.copy(starts=starts, stops=stops, content=(awkward.util.deepcopy(self._content) if copy else self._content))

        else:
            out = self.copy(starts=starts, stops=stops, content=awkward.util.numpy.empty(stops.max(), dtype=self._content.dtype))

            if offsetsaliased(self._starts, self._stops) or awkward.util.numpy.array_equal(self._starts[1:], self._stops[:-1]):
                content = self._content[self._starts[0]:self._stops[-1]]
            elif (self._starts[:-1] < self._starts[1:]).all():
                content = self._content[awkward.util.numpy.arange(len(self.parents), dtype=awkward.util.INDEXTYPE)[self.parents >= 0]]
            else:
                order = awkward.util.numpy.argsort(self.parents, kind="mergesort")
                content = self._content[order[self.parents[order] >= 0]]

            if offsetsaliased(starts, stops) or awkward.util.numpy.array_equal(starts[1:], stops[:-1]):
                out._content[starts[0]:stops[-1]] = content
            elif (starts[:-1] < starts[1:]).all():
                out._content[awkward.util.numpy.arange(len(out.parents), dtype=awkward.util.INDEXTYPE)[out.parents >= 0]] = content
            else:
                order = awkward.util.numpy.argsort(out.parents, kind="mergesort")
                out._content[order[out.parents[order] >= 0]] = content

            return out

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        import awkward.array.objects
        import awkward.array.table

        self._valid()

        if method != "__call__":
            return NotImplemented

        inputs = list(inputs)
        starts, stops = None, None
        for i in range(len(inputs)):
            if isinstance(inputs[i], (numbers.Number, awkward.util.numpy.number)):
                pass

            elif isinstance(inputs[i], JaggedArray):
                if starts is stops is None:
                    inputs[i] = inputs[i]._tojagged(copy=False)
                    starts, stops = inputs[i]._starts, inputs[i]._stops
                else:
                    inputs[i] = inputs[i]._tojagged(starts, stops, copy=False)

            elif isinstance(inputs[i], awkward.array.base.AwkwardArray):
                pass

            else:
                tmp = awkward.util.numpy.array(inputs[i], copy=False)
                if len(tmp.shape) != 0:
                    inputs[i] = tmp

        for jaggedarray in inputs:
            if isinstance(jaggedarray, JaggedArray):
                starts, stops, parents, good = jaggedarray._starts, jaggedarray._stops, None, None
                break
        else:
            assert False

        for i in range(len(inputs)):
            if isinstance(inputs[i], (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray)) and not isinstance(inputs[i], JaggedArray):
                data = awkward.util.toarray(inputs[i], inputs[i].dtype, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
                if starts.shape != data.shape:
                    raise ValueError("cannot broadcast JaggedArray of shape {0} with array of shape {1}".format(starts.shape, data.shape))

                if parents is None:
                    parents = jaggedarray.parents
                    good = (parents >= 0)

                def recurse(x):
                    if isinstance(x, awkward.array.objects.ObjectArray):
                        return x.copy(content=recurse(x.content))

                    elif isinstance(x, awkward.array.table.Table):
                        content = x.empty_like()
                        for n in x.columns:
                            content[n] = recurse(x[n])
                        return content

                    else:
                        content = awkward.util.numpy.empty(len(parents), dtype=x.dtype)
                        if len(x.shape) == 0:
                            content[good] = x
                        else:
                            content[good] = x[parents[good]]
                        return content

                content = recurse(data)

                inputs[i] = JaggedArray(starts, stops, content)

        for i in range(len(inputs)):
            if isinstance(inputs[i], JaggedArray):
                if good is None:
                    inputs[i] = inputs[i].content
                else:
                    inputs[i] = inputs[i].content[good]

        result = getattr(ufunc, method)(*inputs, **kwargs)

        if isinstance(result, tuple):
            return tuple(awkward.array.objects.Methods.maybemixin(type(x), JaggedArray)(starts, stops, x) for x in result)
        elif method == "at":
            return None
        else:
            return awkward.array.objects.Methods.maybemixin(type(result), JaggedArray)(starts, stops, result)

    @staticmethod
    def aligned(*jaggedarrays):
        if not all(isinstance(x, JaggedArray) for x in jaggedarrays):
            raise TypeError("all objects passed to JaggedArray.aligned must be JaggedArrays")

        if len(jaggedarrays) == 0:
            return True

        # empty subarrays can be represented by any start,stop as long as start == stop
        relevant, relevantstarts, relevantstops = None, None, None

        first = jaggedarrays[0]
        for next in jaggedarrays[1:]:
            if first._starts is not next._starts:
                if relevant is None:
                    relevant = (first.counts != 0)
                if relevantstarts is None:
                    relevantstarts = first._starts[relevant]
                if not awkward.util.numpy.array_equal(relevantstarts, next._starts[relevant]):
                    return False

            if first._stops is not next._stops:
                if relevant is None:
                    relevant = (first.counts != 0)
                if relevantstops is None:
                    relevantstops = first._stops[relevant]
                if not awkward.util.numpy.array_equal(relevantstops, next._stops[relevant]):
                    return False

        return True

    def argpairs(self, same=True):
        import awkward.array.table
        self._valid()
        
        counts = self.counts * (self.counts + 1) >> 1    # N * (N + 1) // 2

        offsets = counts2offsets(counts)
        indexes = awkward.util.numpy.arange(offsets[-1])
        parents = offsets2parents(offsets)

        n = self.counts[parents]
        k = indexes - offsets[parents]
        two_n_1 = (2*n + 1)
        i = awkward.util.numpy.floor((two_n_1 - awkward.util.numpy.sqrt(two_n_1*two_n_1 - 8*k)) / 2).astype(awkward.util.INDEXTYPE)

        starts_parents = self._starts[parents]

        left = starts_parents + i
        right = starts_parents + k - n*i + (i*(i + 1) >> 1)

        out = JaggedArray.fromoffsets(offsets, awkward.array.table.Table(left, right))
        out._parents = parents

        if not same:
            out = out[out["0"] != out["1"]]

        return out

    def pairs(self, same=True):
        argpairs = self.argpairs(same=same)
        left = argpairs._content["0"]
        right = argpairs._content["1"]

        out = JaggedArray.fromoffsets(argpairs.offsets, awkward.array.table.Table(self._content[left], self._content[right]))
        out._parents = argpairs._parents
        return out

    def argcross(self, other):
        import awkward.array.table
        self._valid()

        if not isinstance(other, JaggedArray):
            raise ValueError("both arrays must be JaggedArrays")
        
        if len(self) != len(other):
            raise ValueError("both JaggedArrays must have the same length")
        
        offsets = counts2offsets(self.counts * other.counts)
        indexes = awkward.util.numpy.arange(offsets[-1], dtype=awkward.util.INDEXTYPE)
        parents = offsets2parents(offsets)

        ocp = other.counts[parents]
        iop = indexes - offsets[parents]
        iop_ocp = iop // ocp

        left = self._starts[parents] + iop_ocp
        right = other._starts[parents] + iop - ocp * iop_ocp

        out = JaggedArray.fromoffsets(offsets, awkward.array.table.Table(left, right))
        out._parents = parents
        return out

    def cross(self, other):
        import awkward.array.table

        argcross = self.argcross(other)
        left, right = argcross._content._content.values()

        fields = [other._content[right]]
        if getattr(self, "_iscross", False):
            fields = [x[left] for x in self._content._content.values()] + fields
        else:
            fields = [self._content[left]] + fields

        out = JaggedArray.fromoffsets(argcross._offsets, awkward.array.table.Table(*fields))
        out._parents = argcross._parents
        out._iscross = True
        return out

    def _canuseoffset(self):
        self._valid()
        return offsetsaliased(self._starts, self._stops) or (len(self._starts.shape) == 1 and awkward.util.numpy.array_equal(self._starts[1:], self._stops[:-1]))

    def flatten(self):
        if self._canuseoffset():
            return self._content[self._starts[0]:self._stops[-1]]
        else:
            offsets = counts2offsets(self.counts.reshape(-1))
            return self._tojagged(offsets[:-1], offsets[1:], copy=False)._content

    def any(self):
        if self._canuseoffset():
            if issubclass(self._content.dtype.type, (awkward.util.numpy.bool, awkward.util.numpy.bool_)):
                content = self._content
            else:
                content = self._content != 0

            out = awkward.util.numpy.empty(self._starts.shape + content.shape[1:], dtype=content.dtype)
            nonterminal = self.offsets[self.offsets != self.offsets[-1]]
            out[:len(nonterminal)] = awkward.util.numpy.logical_or.reduceat(content[self._starts[0]:self._stops[-1]], nonterminal)
            out[self.offsets[1:] == self.offsets[:-1]] = False
            return out
            
        else:
            return self.count_nonzero() != 0

    def all(self):
        if self._canuseoffset():
            if issubclass(self._content.dtype.type, (awkward.util.numpy.bool, awkward.util.numpy.bool_)):
                content = self._content
            else:
                content = self._content != 0

            out = awkward.util.numpy.empty(self._starts.shape + content.shape[1:], dtype=content.dtype)
            nonterminal = self.offsets[self.offsets != self.offsets[-1]]
            out[:len(nonterminal)] = awkward.util.numpy.logical_and.reduceat(content[self._starts[0]:self._stops[-1]], nonterminal)
            out[self.offsets[1:] == self.offsets[:-1]] = True
            return out
            
        else:
            return self.count_nonzero() == self.count

    def count_nonzero(self):
        if issubclass(self._content.dtype.type, (awkward.util.numpy.bool, awkward.util.numpy.bool_)):
            return self.sum()
        else:
            return (self != 0).sum()

    def sum(self):
        if issubclass(self._content.dtype.type, (awkward.util.numpy.bool, awkward.util.numpy.bool_)):
            content = self._content.astype(awkward.util.numpy.int64)
        else:
            content = self._content

        if self._canuseoffset():
            out = awkward.util.numpy.empty(self._starts.shape + content.shape[1:], dtype=content.dtype)
            nonterminal = self.offsets[self.offsets != self.offsets[-1]]
            out[:len(nonterminal)] = awkward.util.numpy.add.reduceat(content[self._starts[0]:self._stops[-1]], nonterminal)
            out[self.offsets[1:] == self.offsets[:-1]] = 0
            return out

        else:
            contentsum = awkward.util.numpy.empty((len(content) + 1,) + content.shape[1:], dtype=content.dtype)
            contentsum[0] = 0
            awkward.util.numpy.cumsum(content, axis=0, out=contentsum[1:])

            nonempty = (self._starts != self._stops)

            out = awkward.util.numpy.zeros(self.shape + content.shape[1:], dtype=content.dtype)
            out[nonempty] = contentsum[self._stops[nonempty]] - contentsum[self._starts[nonempty]]
            return out

    def prod(self):
        if issubclass(self._content.dtype.type, (awkward.util.numpy.bool, awkward.util.numpy.bool_)):
            content = self._content.astype(numpy.int64)
        else:
            content = self._content

        if self._canuseoffset():
            out = awkward.util.numpy.empty(self._starts.shape + content.shape[1:], dtype=content.dtype)
            nonterminal = self.offsets[self.offsets != self.offsets[-1]]
            out[:len(nonterminal)] = awkward.util.numpy.multiply.reduceat(content[self._starts[0]:self._stops[-1]], nonterminal)
            out[self.offsets[1:] == self.offsets[:-1]] = 1
            return out

        else:
            out = awkward.util.numpy.ones(self.shape + content.shape[1:], dtype=content.dtype)
            flatout = out.reshape((-1,) + content.shape[1:])

            content = content
            flatstops = self._stops.reshape(-1)
            for i, flatstart in enumerate(self._starts.reshape(-1)):
                flatstop = flatstops[i]
                if flatstart != flatstop:
                    flatout[i] *= content[flatstart:flatstop].prod(axis=0)
            return out

    def _argminmax(self, ismin):
        if len(self._content.shape) != 1:
            raise ValueError("cannot compute arg{0} because content is not one-dimensional".format("min" if ismin else "max"))

        self._valid()
        contentmax = self._content.max()
        shiftval = awkward.util.numpy.ceil(contentmax) + 1
        if math.isnan(shiftval) or math.isinf(shiftval) or shiftval != contentmax:
            return self._minmax_general(True, ismin)

        flatstarts = self._starts.reshape(-1)
        flatstops = self._stops.reshape(-1)

        nonempty = (flatstarts != flatstops)
        nonterminal = (flatstarts < len(self._content))
        flatstarts = flatstarts[nonterminal]
        flatstops = flatstops[nonterminal]

        shift = awkward.util.numpy.zeros(self._content.shape, dtype=awkward.util.INDEXTYPE)
        shift[flatstarts] = shiftval
        awkward.util.numpy.cumsum(shift, out=shift)

        sortedindex = (self._content + shift).argsort()

        if ismin:
            flatout = sortedindex[flatstarts] - flatstarts
        else:
            flatout = sortedindex[flatstops - 1] - flatstarts

        newstarts = awkward.util.numpy.arange(len(nonempty), dtype=awkward.util.INDEXTYPE).reshape(self._starts.shape)
        newstops = awkward.util.numpy.array(newstarts)
        newstops.reshape(-1)[nonempty] += 1
        return self.copy(starts=newstarts, stops=newstops, content=flatout)

    def argmin(self):
        return self._argminmax(True)

    def argmax(self):
        return self._argminmax(False)

    def _minmax_offset(self, ismin):
        out = awkward.util.numpy.empty(self._starts.shape + self._content.shape[1:], dtype=self._content.dtype)
        nonterminal = self.offsets[self.offsets != self.offsets[-1]]

        if ismin:
            out[:len(nonterminal)] = awkward.util.numpy.minimum.reduceat(self._content[self._starts[0]:self._stops[-1]], nonterminal)
            if issubclass(self._content.dtype.type, awkward.util.numpy.floating):
                out[self.offsets[1:] == self.offsets[:-1]] = awkward.util.numpy.inf
            else:
                out[self.offsets[1:] == self.offsets[:-1]] = awkward.util.numpy.iinfo(self._content.dtype.type).max

        else:
            out[:len(nonterminal)] = awkward.util.numpy.maximum.reduceat(self._content[self._starts[0]:self._stops[-1]], nonterminal)
            if issubclass(self._content.dtype.type, awkward.util.numpy.floating):
                out[self.offsets[1:] == self.offsets[:-1]] = -awkward.util.numpy.inf
            else:
                out[self.offsets[1:] == self.offsets[:-1]] = awkward.util.numpy.iinfo(self._content.dtype.type).min

        return out

    def _minmax_general(self, isarg, ismin):
        # not a group; must iterate (in absence of modified Hillis-Steele)
        if isarg:
            if len(self._content.shape) != 1:
                raise ValueError("cannot compute arg{0} because content is not one-dimensional".format("min" if ismin else "max"))

            if ismin:
                optimum = awkward.util.numpy.argmin
            else:
                optimum = awkward.util.numpy.argmax

            out = awkward.util.numpy.empty(self._starts.shape + self._content.shape[1:], dtype=awkward.util.INDEXTYPE)

        else:
            if ismin:
                optimum = awkward.util.numpy.amin
            else:
                optimum = awkward.util.numpy.amax

            if issubclass(self._content.dtype.type, awkward.util.numpy.floating):
                if ismin:
                    out = awkward.util.numpy.full(self._starts.shape + self._content.shape[1:], awkward.util.numpy.inf, dtype=self._content.dtype)
                else:
                    out = awkward.util.numpy.full(self._starts.shape + self._content.shape[1:], -awkward.util.numpy.inf, dtype=self._content.dtype)

            elif issubclass(self._content.dtype.type, awkward.util.numpy.integer):
                if ismin:
                    out = awkward.util.numpy.full(self._starts.shape + self._content.shape[1:], awkward.util.numpy.iinfo(self._content.dtype.type).max, dtype=self._content.dtype)
                else:
                    out = awkward.util.numpy.full(self._starts.shape + self._content.shape[1:], awkward.util.numpy.iinfo(self._content.dtype.type).min, dtype=self._content.dtype)

            else:
                raise TypeError("only floating point and integer types can be minimized")

        flatout = out.reshape((-1,) + self._content.shape[1:])
        flatstarts = self._starts.reshape(-1)
        flatstops = self._stops.reshape(-1)

        content = self._content
        for i, flatstart in enumerate(flatstarts):
            flatstop = flatstops[i]
            if flatstart != flatstop:
                flatout[i] = optimum(content[flatstart:flatstop], axis=0)

        if isarg:
            newstarts = awkward.util.numpy.arange(len(flatstarts), dtype=awkward.util.INDEXTYPE).reshape(self._starts.shape)
            newstops = awkward.util.numpy.array(newstarts)
            newstops.reshape(-1)[flatstarts != flatstops] += 1
            return self.copy(starts=newstarts, stops=newstops, content=flatout)

        else:
            return out

    def min(self):
        if self._canuseoffset():
            return self._minmax_offset(True)
        else:
            return self._minmax_general(False, True)

    def max(self):
        if self._canuseoffset():
            return self._minmax_offset(False)
        else:
            return self._minmax_general(False, False)

    def pandas(self):
        import pandas

        if isinstance(self._content, awkward.util.numpy.ndarray):
            out = pandas.DataFrame(self._content)
        else:
            out = self._content.pandas()

        if isinstance(self._content, JaggedArray):
            parents = self._content._broadcast(self.parents)._content
            index = self._content._broadcast(self.index._content)._content
            out.index = pandas.MultiIndex.from_arrays([parents, index] + out.index.labels[1:])
        else:
            out.index = pandas.MultiIndex.from_arrays([self.parents, self.index._content])

        return out

class ByteJaggedArray(JaggedArray):
    def __init__(self, starts, stops, content, subdtype):
        super(ByteJaggedArray, self).__init__(starts, stops, content)
        self.subdtype = subdtype

    @classmethod
    def fromiter(cls, iterable):
        offsets = [0]
        content = []
        for x in iterable:
            offsets.append(offsets[-1] + len(x))
            content.extend(x)
        offsets = awkward.util.numpy.array(offsets, dtype=awkward.util.INDEXTYPE)
        content = awkward.util.numpy.array(content)
        offsets *= content.dtype.itemsize
        return cls(offsets[:-1], offsets[1:], content, subdtype=content.dtype)

    @classmethod
    def fromoffsets(cls, offsets, content, subdtype):
        tmp = ByteJaggedArray.__bases__[0].fromoffsets(offsets, awkward.util.numpy.array([]))
        return cls(tmp._starts, tmp._stops, content, subdtype=subdtype)

    @classmethod
    def fromcounts(cls, counts, content, subdtype):
        tmp = ByteJaggedArray.__bases__[0].fromcounts(counts, awkward.util.numpy.array([]))
        return cls(tmp._starts, tmp._stops, content, subdtype=subdtype)

    @classmethod
    def fromparents(cls, parents, content, subdtype):
        tmp = ByteJaggedArray.__bases__[0].fromparents(parents, awkward.util.numpy.array([]))
        return cls(tmp._starts, tmp._stops, content, subdtype=subdtype)

    @classmethod
    def fromuniques(cls, uniques, content, subdtype):
        tmp = ByteJaggedArray.__bases__[0].fromuniques(uniques, awkward.util.numpy.array([]))
        return cls(tmp._starts, tmp._stops, content, subdtype=subdtype)

    def copy(self, starts=None, stops=None, content=None, subdtype=None):
        out = super(ByteJaggedArray, self).copy(starts=starts, stops=stops, content=content)
        if subdtype is None:
            out._subdtype = self._subdtype
        else:
            out.subdtype = subdtype
        return out

    def deepcopy(self, starts=None, stops=None, content=None, subdtype=None):
        out = super(ByteJaggedArray, self).deepcopy(starts=starts, stops=stops, content=content)
        if subdtype is None:
            out._subdtype = self._subdtype
        else:
            out.subdtype = subdtype
        return out

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = awkward.util.toarray(value, awkward.util.CHARTYPE, awkward.util.numpy.ndarray).view(awkward.util.CHARTYPE).reshape(-1)
        self._isvalid = False

    @property
    def subdtype(self):
        return self._subdtype

    @subdtype.setter
    def subdtype(self, value):
        self._subdtype = awkward.util.numpy.dtype(value)
        self._isvalid = False

    @property
    def type(self):
        return awkward.type.ArrayType(*(self._starts.shape + (awkward.type.ArrayType(awkward.util.numpy.inf, self._subdtype),)))

    def __iter__(self):
        self._valid()
        if len(self._starts.shape) != 1:
            for x in super(JaggedArray, self).__iter__():
                yield x.view(self._subdtype)
        else:
            stops = self._stops
            content = self._content
            for i, start in enumerate(self._starts):
                yield content[start:stops[i]].view(self._subdtype)

    def _divitemsize(self, x):
        if self._subdtype.itemsize == 1:
            return x
        elif self._subdtype.itemsize == 2:
            return x >> 1
        elif self._subdtype.itemsize == 4:
            return x >> 2
        elif self._subdtype.itemsize == 8:
            return x >> 3
        else:
            return x // self._subdtype.itemsize

    def _valid(self):
        if not self._isvalid:
            super(ByteJaggedArray, self)._valid()

            if (self._divitemsize(self.counts) * self._subdtype.itemsize != self.counts).any():
                raise ValueError("not all counts are a multiple of {0}".format(self._subdtype.itemsize))

            self._isvalid = True

    def __getitem__(self, where):
        self._valid()

        if awkward.util.isstringslice(where):
            raise IndexError("cannot index ByteJaggedArray with string or sequence of strings")

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        if len(self._starts.shape) == 1:
            head, tail = where[0], where[1:]
        else:
            head, tail = where[:len(self._starts.shape)], where[len(self._starts.shape):]

        if isinstance(head, JaggedArray):
            return self._tojagged(copy=False)[where]

        else:
            starts = self._starts[head]
            stops = self._stops[head]
            if len(starts.shape) == len(stops.shape) == 0:
                return self._content[starts:stops].view(self._subdtype)[tail]
            else:
                node = self.copy(starts=starts, stops=stops)

        while isinstance(node, JaggedArray) and len(tail) > 0:
            head, tail = tail[0], tail[1:]

            if isinstance(head, (numbers.Integral, awkward.util.numpy.integer)):
                original_head = head
                counts = node._stops - node._starts
                if head < 0:
                    head = counts + head
                if not awkward.util.numpy.bitwise_and(0 <= head, head < counts).all():
                    raise IndexError("index {0} is out of bounds for jagged min size {1}".format(original_head, counts.min()))
                node = node._content[node._starts + head*self._subdtype.itemsize]
            else:
                # the other cases are possible, but complicated; the first sets the form
                raise NotImplementedError("jagged second dimension index type: {0}".format(original_head))

        return node[tail]

    def _tojagged(self, starts=None, stops=None, copy=True):
        if starts is None and stops is None:
            byteoffsets = counts2offsets(self.counts.reshape(-1))
            bytestarts, bytestops = byteoffsets[:-1].reshape(self._starts.shape), byteoffsets[1:].reshape(self._starts.shape)
            offsets = self._divitemsize(byteoffsets)
            starts, stops = offsets[:-1].reshape(self._starts.shape), offsets[1:].reshape(self._starts.shape)

        elif stops is None:
            starts = awkward.util.toarray(starts, awkward.util.INDEXTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
            if self._starts.shape != starts.shape:
                raise IndexError("cannot fit ByteJaggedArray with shape {0} into starts with shape {1}".format(self._starts.shape, starts.shape))

            bytestarts = starts * self._subdtype.itemsize
            bytestops = bytestarts + self.counts
            stops = self._divitemsize(bytestops)

            if (stops[:-1] > starts[1:]).any():
                raise IndexError("cannot fit contents of ByteJaggedArray into the given starts array")

        elif starts is None:
            stops = awkward.util.toarray(stops, awkward.util.INDEXTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
            if self._stops.shape != stops.shape:
                raise IndexError("cannot fit ByteJaggedArray with shape {0} into stops with shape {1}".format(self._stops.shape, stops.shape))

            bytestops = stops * self._subdtype.itemsize
            bytestarts = bytestops - self.counts
            starts = self._divitemsize(bytestarts)

            if (stops[:-1] > starts[1:]).any():
                raise IndexError("cannot fit contents of ByteJaggedArray into the given stops array")

        else:
            bytestarts, bytestops = starts * self._subdtype.itemsize, stops * self._subdtype.itemsize
            if not awkward.util.numpy.array_equal(bytestops - bytestarts, self.counts):
                raise IndexError("cannot fit contents of ByteJaggedArray into the given starts and stops arrays")
        
        JaggedArray._validstartsstops(bytestarts, bytestops)
        if (bytestops - bytestarts != (stops - starts) * self._subdtype.itemsize).any():
            raise ValueError("not all counts are a multiple of {0}".format(self._subdtype.itemsize))

        if awkward.util.numpy.array_equal(bytestarts, self._starts) and awkward.util.numpy.array_equal(bytestops, self._stops):
            return JaggedArray(starts, stops, content=(awkward.util.deepcopy(self._content) if copy else self._content))

        else:
            nonemptystops = bytestops[bytestarts != bytestops]
            if len(nonemptystops) == 0:
                return JaggedArray(starts, stops, awkward.util.numpy.empty(0, dtype=self._subdtype))

            bytesout = JaggedArray(bytestarts, bytestops, awkward.util.numpy.empty(nonemptystops.max(), dtype=awkward.util.CHARTYPE))

            if offsetsaliased(self._starts, self._stops) or awkward.util.numpy.array_equal(self._starts[1:], self._stops[:-1]):
                content = self._content[self._starts[0]:self._stops[-1]]
            elif (self._starts[:-1] < self._starts[1:]).all():
                content = self._content[awkward.util.numpy.arange(len(self.parents))[self.parents >= 0]]
            else:
                order = awkward.util.numpy.argsort(self.parents, kind="mergesort")
                content = self._content[order[self.parents[order] >= 0]]

            if offsetsaliased(bytestarts, bytestops) or awkward.util.numpy.array_equal(bytestarts[1:], bytestops[:-1]):
                bytesout._content[bytestarts[0]:bytestops[-1]] = content
            elif (bytestarts[:-1] < bytestarts[1:]).all():
                bytesout._content[awkward.util.numpy.arange(len(bytesout.parents), dtype=awkward.util.INDEXTYPE)[bytesout.parents >= 0]] = content
            else:
                order = awkward.util.numpy.argsort(bytesout.parents, kind="mergesort")
                bytesout._content[order[bytesout.parents[order] >= 0]] = content

            return JaggedArray(starts, stops, bytesout._content.view(self._subdtype))
