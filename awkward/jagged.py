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
import itertools

import numpy

import awkward.base

class JaggedArray(awkward.base.AwkwardArray):
    INDEXTYPE = numpy.dtype(numpy.int64)

    @classmethod
    def fromoffsets(cls, offsets, content, writeable=True):
        return cls(offsets[:-1], offsets[1:], content, writeable=writeable)

    @classmethod
    def fromiterable(cls, iterable, writeable=True):
        offsets = [0]
        content = []
        for x in iterable:
            offsets.append(offsets[-1] + len(x))
            content.extend(x)
        return cls(offsets[:-1], offsets[1:], content, writeable=writeable)

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
        if not isinstance(value, awkward.base.AwkwardArray):
            value = numpy.array(value, dtype=getattr(value, "dtype", self.INDEXTYPE), copy=False)

        if not issubclass(value.dtype.type, numpy.integer):
            raise TypeError("starts must have integer dtype")
        if len(value.shape) != 1:
            raise TypeError("starts must have 1-dimensional shape")

        self._starts = value

    @property
    def stops(self):
        return self._stops

    @stops.setter
    def stops(self, value):
        if not isinstance(value, awkward.base.AwkwardArray):
            value = numpy.array(value, dtype=getattr(value, "dtype", self.INDEXTYPE), copy=False)

        if not issubclass(value.dtype.type, numpy.integer):
            raise TypeError("stops must have integer dtype")
        if len(value.shape) != 1:
            raise TypeError("stops must have 1-dimensional shape")

        self._stops = value

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        if not isinstance(value, awkward.base.AwkwardArray):
            value = numpy.array(value, copy=False)
        self._content = value

    @property
    def writeable(self):
        return self._writeable

    @writeable.setter
    def writeable(self, value):
        self._writeable = bool(value)

    @property
    def dtype(self):
        return self._content.dtype

    @property
    def shape(self):
        return self._starts.shape

    def _offsets_aliased(self):
        return (self._starts.base is not None and self._stops.base is not None and self._starts.base is self._stops.base and
                self._starts.ctypes.data == self._starts.base.ctypes.data and
                self._stops.ctypes.data == self._stops.base.ctypes.data + self._stops.dtype.itemsize and
                len(self._starts) == len(self._starts.base) - 1 and
                len(self._stops) == len(self._stops.base) - 1)

    @property
    def offsets(self):
        if self._offsets_aliased():
            return self._starts.base
        elif numpy.array_equal(self._starts[1:], self.stops[:-1]):
            return numpy.append(self._starts, self.stops[-1])
        else:
            raise ValueError("starts and stops are not compatible with a single offsets array")

    @staticmethod
    def compatible(*jaggedarrays):
        if not all(isinstance(x, JaggedArray) for x in jaggedarrays):
            raise TypeError("not all objects passed to JaggedArray.compatible are JaggedArrays")
        return all(numpy.array_equal(x._starts, jaggedarrays[0]._starts) and numpy.array_equal(x._stops, jaggedarrays[0]._stops) for x in jaggedarrays[1:])

    def __len__(self):
        return len(self._starts)

    def __getitem__(self, where):
        starts = self._starts[where]
        stops = self._stops[where]
        if len(starts.shape) == len(stops.shape) == 0:
            return self.content[starts:stops]
        else:
            return JaggedArray(starts, stops, self._content, writeable=self._writeable)

    def __setitem__(self, where, what):
        if not self._writeable:
            raise ValueError("assignment destination is read-only")

        starts = self._starts[where]
        stops = self._stops[where]

        if len(starts.shape) == len(stops.shape) == 0:
            self._content[starts:stops] = what

        elif isinstance(what, (collections.Sequence, numpy.ndarray)) and len(what) == 1:
            for start, stop in itertools.izip(starts, stops):
                self._content[start:stop] = what[0]

        elif isinstance(what, (collections.Sequence, numpy.ndarray)):
            if len(what) != (stops - starts).sum():
                raise ValueError("cannot copy sequence with size {0} to JaggedArray with dimension {1}".format(len(what), (stops - starts).sum()))
            this = next = 0
            for start, stop in itertools.izip(starts, stops):
                next += stop - start
                self._content[start:stop] = what[this:next]
                this = next

        elif isinstance(what, JaggedArray):
            if len(what) != len(starts):
                raise ValueError("cannot copy JaggedArray with size {0} to JaggedArray with dimension {1}".format(len(what), len(starts)))
            for which, start, stop in itertools.izip(what, starts, stops):
                self._content[start:stop] = which

        else:
            for start, stop in itertools.izip(starts, stops):
                self._content[start:stop] = what

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
        self._content = numpy.frombuffer(value, dtype=self.CHARTYPE)
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
        starts = self._starts[where]
        stops = self._stops[where]
        if len(starts.shape) == len(stops.shape) == 0:
            return self._content[starts:stops].view(self._dtype)
        else:
            return ByteJaggedArray(starts, stops, self._content, self._dtype, writeable=self._writeable)

    def __setitem__(self, where, what):
        if not self._writeable:
            raise ValueError("assignment destination is read-only")

        starts = self._starts[where]
        stops = self._stops[where]

        if len(starts.shape) == len(stops.shape) == 0:
            startpos, offset = divmod(starts, self._dtype.itemsize)
            stoppos = stops // self._dtype.itemsize
            buf = numpy.frombuffer(self._content, dtype=self._dtype, count=stoppos, offset=offset)
            buf[startpos:stoppos] = what

        elif len(starts) != 0:
            startposes, offsets = numpy.divmod(starts, self._dtype.itemsize)
            stopposes = numpy.floor_divide(stops, self._dtype.itemsize)

            if isinstance(what, (collections.Sequence, numpy.ndarray)) and len(what) == 1:
                for startpos, stoppos, offset in itertools.izip(startposes, stopposes, offsets):
                    buf = numpy.frombuffer(self._content, dtype=self._dtype, count=stoppos, offset=offset)
                    buf[startpos:stoppos] = what

            elif isinstance(what, (collections.Sequence, numpy.ndarray)):
                if len(what) != (stopposes - startposes).sum():
                    raise ValueError("cannot copy sequence with size {0} to ByteJaggedArray with dimension {1}".format(len(what), (stopposes - startposes).sum()))
                this = next = 0
                for startpos, stoppos, offset in itertools.izip(startposes, stopposes, offsets):
                    next += stoppos - startpos
                    buf = numpy.frombuffer(self._content, dtype=self._dtype, count=stoppos, offset=offset)
                    buf[startpos:stoppos] = what[this:next]
                    this = next

            elif isinstance(what, JaggedArray):
                if len(what) != len(startposes):
                    raise ValueError("cannot copy JaggedArray with size {0} to ByteJaggedArray with dimension {1}".format(len(what), len(startposes)))
                for which, startpos, stoppos, offset in itertools.izip(what, startposes, stopposes, offsets):
                    buf = numpy.frombuffer(self._content, dtype=self._dtype, count=stoppos, offset=offset)
                    buf[startpos:stoppos] = which

            else:
                for startpos, stoppos, offset in itertools.izip(startposes, stopposes, offsets):
                    buf = numpy.frombuffer(self._content, dtype=self._dtype, count=stoppos, offset=offset)
                    buf[startpos:stoppos] = what
