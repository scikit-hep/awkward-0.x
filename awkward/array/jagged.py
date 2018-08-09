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
import awkward.type
import awkward.util

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
        offsets = awkward.util.toarray(offsets, awkward.util.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
        if len(offsets.shape) != 1 or (offsets < 0).any():
            raise ValueError("offsets must be a one-dimensional, non-negative array")
        out = cls(offsets[:-1], offsets[1:], content)
        out._offsets = offsets
        return out

    @classmethod
    def fromcounts(cls, counts, content):
        counts = awkward.util.toarray(counts, awkward.util.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
        if (counts < 0).any():
            raise ValueError("counts must be a non-negative array")
        offsets = awkward.util.counts2offsets(counts.reshape(-1))
        out = cls(offsets[:-1].reshape(counts.shape), offsets[1:].reshape(counts.shape), content)
        out._offsets = offsets if len(counts.shape) == 1 else None
        out._counts = counts
        return out

    @classmethod
    def fromparents(cls, parents, content):
        parents = awkward.util.toarray(parents, awkward.util.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
        if len(parents.shape) != 1 or len(parents) != len(content):
            raise ValueError("parents array must be one-dimensional with the same length as content")
        starts, stops = awkward.util.parents2startsstops(parents)
        out = cls(starts, stops, content)
        out._parents = parents
        return out

    @classmethod
    def fromuniques(cls, uniques, content):
        uniques = awkward.util.toarray(uniques, awkward.util.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
        if len(uniques.shape) != 1 or len(uniques) != len(content):
            raise ValueError("uniques array must be one-dimensional with the same length as content")
        offsets, parents = awkward.util.uniques2offsetsparents(uniques)
        out = cls.fromoffsets(offsets, content)        
        out._parents = parents
        return out

    def copy(self):
        out = self.__class__.__new__(self.__class__)
        out._starts  = self._starts
        out._stops   = self._stops
        out._content = self._content
        out._offsets = self._offsets
        out._counts  = self._counts
        out._parents = self._parents
        return out

    def deepcopy(self):
        out = self.__class__.__new__(self.__class__)
        out._starts  = awkward.util.deepcopy(self._starts)
        out._stops   = awkward.util.deepcopy(self._stops)
        out._content = awkward.util.deepcopy(self._content)
        out._offsets = awkward.util.deepcopy(self._offsets)
        out._counts  = awkward.util.deepcopy(self._counts)
        out._parents = awkward.util.deepcopy(self._parents)
        return out

    @property
    def starts(self):
        return self._starts

    @starts.setter
    def starts(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
        if (value < 0).any():
            raise ValueError("starts must be a non-negative array")
        self._starts = value
        self._offsets, self._counts, self._parents = None, None, None
        
    @property
    def stops(self):
        return self._stops

    @stops.setter
    def stops(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
        if (value < 0).any():
            raise ValueError("stops must be a non-negative array")
        self._stops = value
        self._offsets, self._counts, self._parents = None, None, None

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = awkward.util.toarray(value, awkward.util.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

    @property
    def offsets(self):
        if self._offsets is None:
            self._valid()
            if awkward.util.offsetsaliased(self._starts, self._stops):
                self._offsets = self._starts.base
            elif len(self._starts.shape) == 1 and numpy.array_equal(self._starts[1:], self.stops[:-1]):
                self._offsets = numpy.append(self._starts, self.stops[-1])
            else:
                raise ValueError("starts and stops are not compatible with a single offsets array")
        return self._offsets

    @offsets.setter
    def offsets(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
        if len(value.shape) != 1 or (value < 0).any():
            raise ValueError("offsets must be a one-dimensional, non-negative array")
        self._starts = value[:-1]
        self._stops = value[1:]
        self._offsets = value
        self._counts, self._parents = None, None

    @property
    def counts(self):
        if self._counts is None:
            self._valid()
            self._counts = self._stops - self._starts
        return self._counts

    @counts.setter
    def counts(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
        if (value < 0).any():
            raise ValueError("counts must be a non-negative array")
        offsets = awkward.util.counts2offsets(value.reshape(-1))
        self._starts = offsets[:-1].reshape(value.shape)
        self._stops = offsets[1:].reshape(value.shape)
        self._offsets = offsets if len(value.shape) == 1 else None
        self._counts = value
        self._parents = None

    @property
    def parents(self):
        if self._parents is None:
            self._valid()
            self._parents = awkward.util.startsstops2parents(self._starts, self._stops)
        return self._parents

    @parents.setter
    def parents(self, value):
        value = awkward.util.toarray(value, awkward.util.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
        if len(value) != len(content):
            raise ValueError("parents array must have the same length as content")
        self._starts, self._stops = awkward.util.parents2startsstops(value)
        self._offsets, self._counts = None, None
        self._parents = value

    @property
    def type(self):
        return awkward.type.ArrayType(*(self._starts.shape + (awkward.type.ArrayType(numpy.inf, awkward.type.fromarray(self._content).to),)))

    def __len__(self):
        self._valid()
        return len(self._starts)

    @property
    def shape(self):
        self._valid()
        return self._starts.shape

    @property
    def dtype(self):
        return numpy.dtype(numpy.object)      # specifically, Numpy arrays

    def _valid(self, starts=None, stops=None):
        if starts is None:
            starts = self._starts
        if stops is None:
            stops = self._stops

        if len(starts.shape) == 0:
            raise TypeError("starts must have at least one dimension")
        if starts.shape[0] == 0:
            starts = starts.view(awkward.util.INDEXTYPE)
        if not issubclass(starts.dtype.type, numpy.integer):
            raise TypeError("starts must have integer dtype")

        if len(stops.shape) != len(starts.shape):
            raise TypeError("stops must have the same shape as starts")
        if stops.shape[0] == 0:
            stops = stops.view(awkward.util.INDEXTYPE)
        if not issubclass(stops.dtype.type, numpy.integer):
            raise TypeError("stops must have integer dtype")

        if len(starts) > len(stops):
            raise ValueError("starts must not have more elements than stops")

    def __getitem__(self, where):
        self._valid()

        if awkward.util.isstringslice(where):
            return JaggedArray(self._starts, self._stops, self._content[where])

        if where == ():
            return self
        if not isinstance(where, tuple):
            where = (where,)
        if len(self._starts.shape) == 1:
            head, tail = where[0], where[1:]
        else:
            head, tail = where[:len(self._starts.shape)], where[len(self._starts.shape):]

        starts = self._starts[head]
        stops = self._stops[head]
        
        if len(starts.shape) == len(stops.shape) == 0:
            return self.content[starts:stops][tail]

        else:
            node = JaggedArray(starts, stops, self._content)
            while isinstance(node, JaggedArray) and len(tail) > 0:
                head, tail = tail[0], tail[1:]

                if isinstance(head, (numbers.Integral, numpy.integer)):
                    original_head = head
                    counts = node._stops - node._starts
                    if head < 0:
                        head = counts + head
                    if not numpy.bitwise_and(0 <= head, head < counts).all():
                        raise IndexError("index {0} is out of bounds for jagged min size {1}".format(original_head, counts.min()))
                    node = node._content[node._starts + head]
                else:
                    # the other cases are possible, but complicated; the first sets the form
                    raise NotImplementedError("jagged second dimension index type: {0}".format(original_head))

            return node[tail]

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

    # @staticmethod
    # def broadcastable(*jaggedarrays):
    #     if not all(isinstance(x, JaggedArray) for x in jaggedarrays):
    #         raise TypeError("all objects passed to JaggedArray.broadcastable must be JaggedArrays")

    #     if len(jaggedarrays) == 0:
    #         return True

    #     # empty subarrays can be represented by any start,stop as long as start == stop
    #     relevant, relevantstarts, relevantstops = None, None, None

    #     first = jaggedarrays[0]
    #     for next in jaggedarrays[1:]:
    #         if first._starts is not next._starts:
    #             if relevant is None:
    #                 relevant = (first.counts != 0)
    #             if relevantstarts is None:
    #                 relevantstarts = first._starts[relevant]
    #             if not numpy.array_equal(relevantstarts, next._starts[relevant]):
    #                 return False

    #         if first._stops is not next._stops:
    #             if relevant is None:
    #                 relevant = (first.counts != 0)
    #             if relevantstops is None:
    #                 relevantstops = first._stops[relevant]
    #             if not numpy.array_equal(relevantstops, next._stops[relevant]):
    #                 return False

    #     return True

    # def tobroadcast(self, data):
    #     data = awkward.util.toarray(data, self._content.dtype, (numpy.ndarray, awkward.array.base.AwkwardArray))
    #     good = (self.parents >= 0)
    #     content = numpy.empty(len(self.parents), dtype=data.dtype)
    #     if len(data.shape) == 0:
    #         content[good] = data
    #     else:
    #         content[good] = data[self.parents[good]]
    #     out = JaggedArray(self._starts, self._stops, content)
    #     out._parents = self.parents
    #     return out

    def tojagged(self, starts=None, stops=None, copy=True):
        self._valid()

        if starts is None and stops is None:
            if copy:
                starts, stops = self._starts.copy(), self._stops.copy()
            else:
                starts, stops = self._starts, self._stops

        elif stops is None:
            starts = awkward.util.toarray(starts, awkward.util.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
            if len(self) != len(starts):
                raise IndexError("cannot fit JaggedArray of length {0} into starts of length {1}".format(len(self), len(starts)))

            stops = starts + self.counts

            if (stops[:-1] > starts[1:]).any():
                raise IndexError("cannot fit contents of JaggedArray into the given starts array")

        elif starts is None:
            stops = awkward.util.toarray(stops, awkward.util.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
            if len(self) != len(stops):
                raise IndexError("cannot fit JaggedArray of length {0} into stops of length {1}".format(len(self), len(stops)))

            starts = stops - self.counts

            if (stops[:-1] > starts[1:]).any():
                raise IndexError("cannot fit contents of JaggedArray into the given stops array")

        else:
            if not numpy.array_equal(stops - starts, self.counts):
                raise IndexError("cannot fit contents of JaggedArray into the given starts and stops arrays")

        if not copy and starts is self._starts and stops is self._stops:
            return self

        elif (starts is self._starts or numpy.array_equal(starts, self._starts)) and (stops is self._stops or numpy.array_equal(stops, self._stops)):
            if copy:
                return JaggedArray(starts, stops, self._content.copy())
            else:
                return JaggedArray(starts, stops, self._content)

        else:
            selfstarts, selfstops, selfcontent = self._starts, self._stops, self._content
            content = numpy.empty(stops.max(), dtype=selfcontent.dtype)
            
            lenstarts = len(starts)
            i = 0
            while i < lenstarts:
                content[starts[i]:stops[i]] = selfcontent[selfstarts[i]:selfstops[i]]
                i += 1
            
            return JaggedArray(starts, stops, content)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        self._valid()

        if method != "__call__":
            return NotImplemented

        inputs = list(inputs)
        starts, stops = None, None

        for i in range(len(inputs)):
            if isinstance(inputs[i], (numbers.Number, numpy.number)):
                pass

            elif isinstance(inputs[i], JaggedArray):
                if starts is stops is None:
                    inputs[i] = inputs[i].tojagged(copy=False)
                    starts, stops = inputs[i].starts, inputs[i].stops
                else:
                    inputs[i] = inputs[i].tojagged(starts, stops)

            else:
                inputs[i] = numpy.array(inputs[i], copy=False)

        for jaggedarray in inputs:
            if isinstance(jaggedarray, JaggedArray):
                starts, stops, parents, good = jaggedarray._starts, jaggedarray._stops, None, None
                break
        else:
            assert False

        for i in range(len(inputs)):
            if isinstance(inputs[i], numpy.ndarray):
                data = awkward.util.toarray(inputs[i], inputs[i].dtype, (numpy.ndarray, awkward.array.base.AwkwardArray))
                if parents is None:
                    parents = jaggedarray.parents
                    good = (parents >= 0)

                content = numpy.empty(len(parents), dtype=data.dtype)
                if len(data.shape) == 0:
                    content[good] = data
                else:
                    content[good] = data[parents[good]]
                inputs[i] = JaggedArray(starts, stops, content)

        for i in range(len(inputs)):
            if isinstance(inputs[i], JaggedArray):
                if good is None:
                    inputs[i] = inputs[i].content
                else:
                    inputs[i] = inputs[i].content[good]

        result = getattr(ufunc, method)(*inputs, **kwargs)

        if isinstance(result, tuple):
            return tuple(JaggedArray(starts, stops, x) for x in result)
        elif method == "at":
            return None
        else:
            return JaggedArray(starts, stops, result)

    def argcross(self, other):
        self._valid()

        if not isinstance(other, JaggedArray):
            raise ValueError("both arrays must be JaggedArrays")
        
        if len(self) != len(other):
            raise ValueError("both JaggedArrays must have the same length")
        
        offsets = awkward.util.counts2offsets(self.counts * other.counts)

        indexes = numpy.arange(offsets[-1], dtype=awkward.util.INDEXTYPE)
        parents = awkward.util.startsstops2parents(offsets[:-1], offsets[1:])

        left = numpy.empty_like(indexes)
        right = numpy.empty_like(indexes)

        left[indexes] = self._starts[parents[indexes]] + ((indexes - offsets[parents[indexes]]) // othercounts[parents[indexes]])
        right[indexes] = other._starts[parents[indexes]] + (indexes - offsets[parents[indexes]]) - othercounts[parents[indexes]] * ((indexes - offsets[parents[indexes]]) // othercounts[parents[indexes]])

        import awkward.array.table 
        out = JaggedArray.fromoffsets(offsets, awkward.array.table.Table(offsets[-1], left, right))
        out._parents = parents
        return out

    def cross(self, other):
        argcross = self.argcross(other)
        left, right = argcross._content._content.values()

        import awkward.array.table
        out = JaggedArray.fromoffsets(argcross._offsets, awkward.array.table.Table(len(argcross._content), self._content[left], other._content[right]))
        out._parents = argcross._parents
        return out

    def sum(self):
        # works because there's a group operation to undo the cumsum
        self._valid()

        contentsum = numpy.empty((len(self._content) + 1,) + self._content.shape[1:], dtype=self._content.dtype)
        contentsum[0] = 0
        awkward.util.cumsum(self._content, axis=0, out=contentsum[1:])

        nonempty = (self._starts != self._stops)

        out = numpy.zeros(self.shape + self._content.shape[1:], dtype=self._content.dtype)
        out[nonempty] = contentsum[self._stops[nonempty]] - contentsum[self._starts[nonempty]]
        return out

    def prod(self):
        # multiplication is a group, but multiplying and dividing large numbers isn't numerically stable
        self._valid()

        out = numpy.ones(self.shape + self._content.shape[1:], dtype=self._content.dtype)
        flatout = out.reshape((-1,) + self._content.shape[1:])

        content = self._content
        flatstops = self._stops.reshape(-1)
        for i, flatstart in enumerate(self._starts.reshape(-1)):
            flatstop = flatstops[i]
            if flatstart != flatstop:
                flatout[i] *= content[flatstart:flatstop].prod(axis=0)
        return out

    def min(self):
        # not a group; must iterate (in absence of modified Hillis-Steele)
        self._valid()

        if issubclass(self._content.dtype.type, numpy.floating):
            out = numpy.full(self.shape + self._content.shape[1:], numpy.inf, dtype=self._content.dtype)
        elif issubclass(self._content.dtype.type, numpy.integer):
            out = numpy.full(self.shape + self._content.shape[1:], numpy.iinfo(self._content.dtype.type).max, dtype=self._content.dtype)
        else:
            raise TypeError("only floating point and integer types can be minimized")
        flatout = out.reshape((-1,) + self._content.shape[1:])

        content = self._content
        flatstops = self._stops.reshape(-1)
        minimum = numpy.minimum
        for i, flatstart in enumerate(self._starts.reshape(-1)):
            flatstop = flatstops[i]
            if flatstart != flatstop:
                flatout[i] = minimum(flatout[i], content[flatstart:flatstop].min(axis=0))
        return out

    def max(self):
        # not a group; must iterate (in absence of modified Hillis-Steele)
        self._valid()

        if issubclass(self._content.dtype.type, numpy.floating):
            out = numpy.full(self.shape + self._content.shape[1:], -numpy.inf, dtype=self._content.dtype)
        elif issubclass(self._content.dtype.type, numpy.integer):
            out = numpy.full(self.shape + self._content.shape[1:], numpy.iinfo(self._content.dtype.type).min, dtype=self._content.dtype)
        else:
            raise TypeError("only floating point and integer types can be maximized")
        flatout = out.reshape((-1,) + self._content.shape[1:])

        content = self._content
        flatstops = self._stops.reshape(-1)
        maximum = numpy.maximum
        for i, flatstart in enumerate(self._starts.reshape(-1)):
            flatstop = flatstops[i]
            if flatstart != flatstop:
                flatout[i] = maximum(flatout[i], content[flatstart:flatstop].max(axis=0))
        return out

class ByteJaggedArray(JaggedArray):
    def __init__(self, starts, stops, content, dtype=awkward.util.CHARTYPE):
        raise NotImplementedError

# class ByteJaggedArray(JaggedArray):
#     @classmethod
#     def fromoffsets(cls, offsets, content, dtype):
#         return cls(offsets[:-1], offsets[1:], content, dtype)

#     @classmethod
#     def fromiter(cls, iterable):
#         offsets = [0]
#         content = []
#         for x in iterable:
#             offsets.append(offsets[-1] + len(x))
#             content.extend(x)
#         offsets = numpy.array(offsets, dtype=awkward.util.INDEXTYPE)
#         content = numpy.array(content)
#         offsets *= content.dtype.itemsize
#         return cls(offsets[:-1], offsets[1:], content, content.dtype)

#     def __init__(self, starts, stops, content, dtype=awkward.util.CHARTYPE):
#         super(ByteJaggedArray, self).__init__(starts, stops, content)
#         self.dtype = dtype

#     @property
#     def content(self):
#         return self._content

#     @content.setter
#     def content(self, value):
#         self._content = awkward.util.toarray(value, awkward.util.CHARTYPE, numpy.ndarray).view(awkward.util.CHARTYPE).reshape(-1)

#     @property
#     def dtype(self):
#         return self._dtype

#     @dtype.setter
#     def dtype(self, value):
#         self._dtype = numpy.dtype(value)

#     def __getitem__(self, where):
#         if awkward.util.isstringslice(where):
#             return ByteJaggedArray(self._starts, self._stops, self._content[where], self._dtype)

#         if not isinstance(where, tuple):
#             where = (where,)
#         head, tail = where[0], where[1:]

#         self._check_startsstops()
#         starts = self._starts[head]
#         stops = self._stops[head]

#         if len(starts.shape) == len(stops.shape) == 0:
#             return self._content[self._singleton((slice(starts, stops),) + tail)].view(self._dtype)
#         else:
#             return ByteJaggedArray(starts, stops, self._content[self._singleton((slice(None),) + tail)], self._dtype)

#     def tojagged(self, starts=None, stops=None, copy=True):
#         counts = self.counts

#         if starts is None and stops is None:
#             offsets = numpy.empty(len(self) + 1, dtype=awkward.util.INDEXTYPE)
#             offsets[0] = 0
#             numpy.cumsum(counts // self.dtype.itemsize, out=offsets[1:])
#             starts, stops = offsets[:-1], offsets[1:]
            
#         elif stops is None:
#             starts = awkward.util.toarray(starts, awkward.util.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
#             if len(self) != len(starts):
#                 raise IndexError("cannot fit ByteJaggedArray of length {0} into starts of length {1}".format(len(self), len(starts)))

#             stops = starts + (counts // self.dtype.itemsize)

#             if (stops[:-1] > starts[1:]).any():
#                 raise IndexError("cannot fit contents of ByteJaggedArray into the given starts array")

#         elif starts is None:
#             stops = awkward.util.toarray(stops, awkward.util.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
#             if len(self) != len(stops):
#                 raise IndexError("cannot fit ByteJaggedArray of length {0} into stops of length {1}".format(len(self), len(stops)))

#             starts = stops - (counts // self.dtype.itemsize)

#             if (stops[:-1] > starts[1:]).any():
#                 raise IndexError("cannot fit contents of ByteJaggedArray into the given stops array")

#         else:
#             if not numpy.array_equal(stops - starts, counts):
#                 raise IndexError("cannot fit contents of ByteJaggedArray into the given starts and stops arrays")

#         self._check_startsstops(starts, stops)

#         selfstarts, selfstops, selfcontent, selfdtype = self._starts, self._stops, self._content, self._dtype
#         content = numpy.empty(counts.sum() // selfdtype.itemsize, dtype=selfdtype)

#         lenstarts = len(starts)
#         i = 0
#         while i < lenstarts:
#             content[starts[i]:stops[i]] = selfcontent[selfstarts[i]:selfstops[i]].view(selfdtype)
#             i += 1

#         return JaggedArray(starts, stops, content)
