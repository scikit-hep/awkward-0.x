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
import awkward.util

class JaggedArray(awkward.array.base.AwkwardArray):
    @classmethod
    def fromoffsets(cls, offsets, content, writeable=True):
        return cls(offsets[:-1], offsets[1:], content, writeable=writeable)

    @classmethod
    def fromcounts(cls, counts, content, writeable=True):
        offsets = numpy.empty(len(counts) + 1, JaggedArray.INDEXTYPE)
        offsets[0] = 0
        numpy.cumsum(counts, offsets[1:])
        return cls(offsets[:-1], offsets[1:], content, writeable=writeable)

    @classmethod
    def fromuniques(cls, uniques, content, writeable=True):
        uniques = JaggedArray._toarray(uniques, self.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
        if len(uniques) != len(content):
            raise ValueError("uniques array must have the same length as content")
        changes = numpy.nonzero(uniques[1:] != uniques[:-1])[0] + 1
        offsets = numpy.empty(len(changes) + 2, dtype=JaggedArray.INDEXTYPE)
        offsets[0] = 0
        offsets[-1] = len(content)
        offsets[1:-1] = changes
        starts, stops = offsets[:-1], offsets[1:]
        return JaggedArray(starts, stops, content, writeable=writeable)

    @classmethod
    def fromparents(cls, parents, content, writeable=True):
        if len(parents) != len(content):
            raise ValueError("parents array must have the same length as content")

        tmp = numpy.nonzero(parents[1:] != parents[:-1])[0] + 1

        changes = numpy.empty(len(tmp) + 2, dtype=JaggedArray.INDEXTYPE)
        changes[0] = 0
        changes[-1] = len(parents)
        changes[1:-1] = tmp

        length = parents.max() + 1
        starts = numpy.zeros(length, dtype=JaggedArray.INDEXTYPE)
        counts = numpy.zeros(length, dtype=JaggedArray.INDEXTYPE)

        where = parents[changes[:-1]]
        real = (where >= 0)

        starts[where[real]] = (changes[:-1])[real]
        counts[where[real]] = (changes[1:] - changes[:-1])[real]

        return JaggedArray(starts, starts + counts, content, writeable=writeable)

    @classmethod
    def fromiterable(cls, iterable, writeable=True):
        offsets = [0]
        content = []
        for x in iterable:
            offsets.append(offsets[-1] + len(x))
            content.extend(x)
        return cls(offsets[:-1], offsets[1:], content, writeable=writeable)

    @staticmethod
    def compatible(*jaggedarrays):
        if not all(isinstance(x, JaggedArray) for x in jaggedarrays):
            raise TypeError("not all objects passed to JaggedArray.compatible are JaggedArrays")

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
                if not numpy.array_equal(relevantstarts, next._starts[relevant]):
                    return False

            if first._stops is not next._stops:
                if relevant is None:
                    relevant = (first.counts != 0)
                if relevantstops is None:
                    relevantstops = first._stops[relevant]
                if not numpy.array_equal(relevantstops, next._stops[relevant]):
                    return False

        return True

    def __init__(self, starts, stops, content, writeable=True):
        self.starts = starts
        self.stops = stops
        self.content = content
        self.writeable = writeable

    @property
    def starts(self):
        return self._starts

    @starts.setter
    def starts(self, value):
        value = self._toarray(value, self.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
        if (value < 0).any():
            raise ValueError("starts must be a non-negative array")
        self._starts = value

    @property
    def stops(self):
        return self._stops

    @stops.setter
    def stops(self, value):
        value = self._toarray(value, self.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
        if (value < 0).any():
            raise ValueError("stops must be a non-negative array")
        self._stops = value

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = self._toarray(value, self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

    @property
    def writeable(self):
        return self._writeable

    @writeable.setter
    def writeable(self, value):
        self._writeable = bool(value)

    @property
    def dtype(self):
        return numpy.dtype(object)   # specifically, subarrays

    @property
    def shape(self):
        return (len(self._starts),)

    def _offsets_is_aliased(self):
        return (isinstance(self._starts, numpy.ndarray) and isinstance(self._stops, numpy.ndarray) and
                self._starts.base is not None and self._stops.base is not None and self._starts.base is self._stops.base and
                self._starts.ctypes.data == self._starts.base.ctypes.data and
                self._stops.ctypes.data == self._stops.base.ctypes.data + self._stops.dtype.itemsize and
                len(self._starts) == len(self._starts.base) - 1 and
                len(self._stops) == len(self._stops.base) - 1)

    @property
    def offsets(self):
        if self._offsets_is_aliased():
            return self._starts.base
        elif numpy.array_equal(self._starts[1:], self.stops[:-1]):
            return numpy.append(self._starts, self.stops[-1])
        else:
            raise ValueError("starts and stops are not compatible with a single offsets array")

    @property
    def counts(self):
        return self._stops - self._starts

    @property
    def parents(self):
        out = numpy.full(len(self._content), -1, dtype=self.INDEXTYPE)
        starts, stops = self._starts, self._stops
        lenstarts = len(starts)
        i = 0
        while i < lenstarts:
            out[starts[i]:stops[i]] = i
            i += 1
        return out
        
    def __len__(self):                 # length is determined by starts
        return len(self._starts)       # data can grow by appending contents and stops before starts

    def _check_startsstops(self, starts=None, stops=None):
        if starts is None:
            starts = self._starts
        if stops is None:
            stops = self._stops

        if len(starts.shape) != 1:
            raise TypeError("starts must have 1-dimensional shape")
        if starts.shape[0] == 0:
            starts = starts.view(self.INDEXTYPE)
        if not issubclass(starts.dtype.type, numpy.integer):
            raise TypeError("starts must have integer dtype")

        if len(stops.shape) != 1:
            raise TypeError("stops must have 1-dimensional shape")
        if stops.shape[0] == 0:
            stops = stops.view(self.INDEXTYPE)
        if not issubclass(stops.dtype.type, numpy.integer):
            raise TypeError("stops must have integer dtype")

        if len(starts) > len(stops):
            raise ValueError("starts must be have as many or fewer elements as stops")

    def __getitem__(self, where):
        if self._isstring(where):
            return JaggedArray(self._starts, self._stops, self._content[where], writeable=self._writeable)

        self._check_startsstops()
        starts = self._starts[where]
        stops = self._stops[where]

        if len(starts.shape) == len(stops.shape) == 0:
            return self.content[starts:stops]
        else:
            return JaggedArray(starts, stops, self._content, writeable=self._writeable)

    def __setitem__(self, where, what):
        if self._isstring(where):
            JaggedArray(self._starts, self._stops, self._content[where], writeable=writeable)[:] = what
            return

        if not self._writeable:
            raise ValueError("assignment destination is read-only")

        self._check_startsstops()
        starts = self._starts[where]
        stops = self._stops[where]

        if len(starts.shape) == len(stops.shape) == 0:
            self._content[starts:stops] = what

        elif isinstance(what, JaggedArray):
            if len(what) != len(starts):
                raise ValueError("cannot copy JaggedArray with length {0} to JaggedArray with dimension {1}".format(len(what), len(starts)))
            for which, start, stop in awkward.util.izip(what, starts, stops):
                self._content[start:stop] = which

        elif isinstance(what, (collections.Sequence, numpy.ndarray, awkward.array.base.AwkwardArray)) and len(what) == 1:
            for start, stop in awkward.util.izip(starts, stops):
                self._content[start:stop] = what[0]

        elif isinstance(what, (collections.Sequence, numpy.ndarray, awkward.array.base.AwkwardArray)):
            if len(what) != (stops - starts).sum():
                raise ValueError("cannot copy sequence with length {0} to JaggedArray with dimension {1}".format(len(what), (stops - starts).sum()))
            this = next = 0
            for start, stop in awkward.util.izip(starts, stops):
                next += stop - start
                self._content[start:stop] = what[this:next]
                this = next

        else:
            for start, stop in awkward.util.izip(starts, stops):
                self._content[start:stop] = what

    def tojagged(self, starts=None, stops=None, copy=True, writeable=True):
        if starts is None and stops is None:
            if copy:
                starts, stops = self._starts.copy(), self._stops.copy()
            else:
                starts, stops = self._starts, self._stops

        elif stops is None:
            starts = self._toarray(starts, self.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
            if len(self) != len(starts):
                raise IndexError("cannot fit JaggedArray of length {0} into starts of length {1}".format(len(self), len(starts)))

            stops = starts + self.counts

            if (stops[:-1] > starts[1:]).any():
                raise IndexError("cannot fit contents of JaggedArray into the given starts array")

        elif starts is None:
            stops = self._toarray(stops, self.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
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
                return JaggedArray(starts, stops, self._content.copy(), writeable=writeable)
            else:
                return JaggedArray(starts, stops, self._content, writeable=writeable)

        else:
            selfstarts, selfstops, selfcontent = self._starts, self._stops, self._content
            content = numpy.empty(stops.max(), dtype=selfcontent.dtype)
            
            lenstarts = len(starts)
            i = 0
            while i < lenstarts:
                content[starts[i]:stops[i]] = selfcontent[selfstarts[i]:selfstops[i]]
                i += 1
            
            return JaggedArray(starts, stops, content, writeable=writeable)

    def makecompatible(self, data, writeable=True):
        data = self._toarray(data, self._content.dtype, (numpy.ndarray, awkward.array.base.AwkwardArray))
        parents = self.parents
        good = (parents >= 0)
        content = numpy.empty(len(parents), dtype=data.dtype)
        if len(data.shape) == 0:
            content[good] = data
        else:
            content[good] = data[parents[good]]
        return JaggedArray(self._starts, self._stops, content, writeable=writeable)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
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
                data = self._toarray(inputs[i], inputs[i].dtype, (numpy.ndarray, awkward.array.base.AwkwardArray))
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

class ByteJaggedArray(JaggedArray):
    @classmethod
    def fromoffsets(cls, offsets, content, dtype, writeable=True):
        return cls(offsets[:-1], offsets[1:], content, dtype, writeable=writeable)

    @classmethod
    def fromiterable(cls, iterable, writeable=True):
        offsets = [0]
        content = []
        for x in iterable:
            offsets.append(offsets[-1] + len(x))
            content.extend(x)
        offsets = numpy.array(offsets, dtype=ByteJaggedArray.INDEXTYPE)
        content = numpy.array(content)
        offsets *= content.dtype.itemsize
        return cls(offsets[:-1], offsets[1:], content, content.dtype, writeable=writeable)

    def __init__(self, starts, stops, content, dtype, writeable=True):
        self._writeable = writeable
        super(ByteJaggedArray, self).__init__(starts, stops, content, writeable=writeable)
        self.dtype = dtype

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = self._toarray(value, self.CHARTYPE, numpy.ndarray).view(self.CHARTYPE).reshape(-1)
        self._content.flags.writeable = self._writeable

    @property
    def writeable(self):
        return self._writeable

    @writeable.setter
    def writeable(self, value):
        self._writeable = bool(value)
        self._content.flags.writeable = self._writeable

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = numpy.dtype(value)

    def __getitem__(self, where):
        if self._isstring(where):
            return ByteJaggedArray(self._starts, self._stops, self._content[where], self._dtype, writeable=writeable)

        self._check_startsstops()
        starts = self._starts[where]
        stops = self._stops[where]

        if len(starts.shape) == len(stops.shape) == 0:
            return self._content[starts:stops].view(self._dtype)
        else:
            return ByteJaggedArray(starts, stops, self._content, self._dtype, writeable=self._writeable)

    def __setitem__(self, where, what):
        if self._isstring(where):
            ByteJaggedArray(self._starts, self._stops, self._content[where], self._dtype, writeable=writeable)[:] = what
            return

        if not self._writeable:
            raise ValueError("assignment destination is read-only")

        self._check_startsstops()
        starts = self._starts[where]
        stops = self._stops[where]

        if len(starts.shape) == len(stops.shape) == 0:
            startpos, offset = divmod(starts, self._dtype.itemsize)
            stoppos = stops // self._dtype.itemsize
            buf = numpy.frombuffer(self._content, dtype=self._dtype, count=stoppos, offset=offset)
            buf[startpos:stoppos] = what

        elif len(starts) != 0:
            if hasattr(numpy, "divmod"):
                startposes, offsets = numpy.divmod(starts, self._dtype.itemsize)
            else:
                startposes = numpy.floor_divide(starts, self._dtype.itemsize)
                offsets = numpy.remainder(starts, self._dtype.itemsize)

            stopposes = numpy.floor_divide(stops, self._dtype.itemsize)

            if isinstance(what, JaggedArray):
                if len(what) != len(startposes):
                    raise ValueError("cannot copy JaggedArray with length {0} to ByteJaggedArray with dimension {1}".format(len(what), len(startposes)))
                for which, startpos, stoppos, offset in awkward.util.izip(what, startposes, stopposes, offsets):
                    buf = numpy.frombuffer(self._content, dtype=self._dtype, count=stoppos, offset=offset)
                    buf[startpos:stoppos] = which

            elif isinstance(what, (collections.Sequence, numpy.ndarray, awkward.array.base.AwkwardArray)) and len(what) == 1:
                for startpos, stoppos, offset in awkward.util.izip(startposes, stopposes, offsets):
                    buf = numpy.frombuffer(self._content, dtype=self._dtype, count=stoppos, offset=offset)
                    buf[startpos:stoppos] = what[0]

            elif isinstance(what, (collections.Sequence, numpy.ndarray, awkward.array.base.AwkwardArray)):
                if len(what) != (stopposes - startposes).sum():
                    raise ValueError("cannot copy sequence with length {0} to ByteJaggedArray with dimension {1}".format(len(what), (stopposes - startposes).sum()))
                this = next = 0
                for startpos, stoppos, offset in awkward.util.izip(startposes, stopposes, offsets):
                    next += stoppos - startpos
                    buf = numpy.frombuffer(self._content, dtype=self._dtype, count=stoppos, offset=offset)
                    buf[startpos:stoppos] = what[this:next]
                    this = next

            else:
                for startpos, stoppos, offset in awkward.util.izip(startposes, stopposes, offsets):
                    buf = numpy.frombuffer(self._content, dtype=self._dtype, count=stoppos, offset=offset)
                    buf[startpos:stoppos] = what

    def tojagged(self, starts=None, stops=None, copy=True, writeable=True):
        counts = self.counts

        if starts is None and stops is None:
            offsets = numpy.empty(len(self) + 1, dtype=self.INDEXTYPE)
            offsets[0] = 0
            numpy.cumsum(counts // self.dtype.itemsize, out=offsets[1:])
            starts, stops = offsets[:-1], offsets[1:]
            
        elif stops is None:
            starts = self._toarray(starts, self.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
            if len(self) != len(starts):
                raise IndexError("cannot fit ByteJaggedArray of length {0} into starts of length {1}".format(len(self), len(starts)))

            stops = starts + (counts // self.dtype.itemsize)

            if (stops[:-1] > starts[1:]).any():
                raise IndexError("cannot fit contents of ByteJaggedArray into the given starts array")

        elif starts is None:
            stops = self._toarray(stops, self.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
            if len(self) != len(stops):
                raise IndexError("cannot fit ByteJaggedArray of length {0} into stops of length {1}".format(len(self), len(stops)))

            starts = stops - (counts // self.dtype.itemsize)

            if (stops[:-1] > starts[1:]).any():
                raise IndexError("cannot fit contents of ByteJaggedArray into the given stops array")

        else:
            if not numpy.array_equal(stops - starts, counts):
                raise IndexError("cannot fit contents of ByteJaggedArray into the given starts and stops arrays")

        self._check_startsstops(starts, stops)

        selfstarts, selfstops, selfcontent, selfdtype = self._starts, self._stops, self._content, self._dtype
        content = numpy.empty(counts.sum() // selfdtype.itemsize, dtype=selfdtype)

        lenstarts = len(starts)
        i = 0
        while i < lenstarts:
            content[starts[i]:stops[i]] = selfcontent[selfstarts[i]:selfstops[i]].view(selfdtype)
            i += 1

        return JaggedArray(starts, stops, content, writeable=writeable)
