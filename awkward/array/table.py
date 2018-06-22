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

class Table(awkward.array.base.AwkwardArray):
    class Row(object):
        __slots__ = ["_table", "_index"]

        def __init__(self, table, index):
            super(Table.Row, self).__setattr__("_table", table)
            super(Table.Row, self).__setattr__("_index", index)

        def __repr__(self):
            return "<Table.Row {0}>".format(self._index)

        def __hasattr__(self, name):
            return name in self._table._content

        def __getattr__(self, name):
            return self._table._content[name][self._index]

        def __setattr__(self, name, what):
            self._table._content[name][self._index] = what

        def __contains__(self, name):
            return name in self._table._content

        def __getitem__(self, name):
            return self._table._content[name][self._index]

        def __setitem__(self, name, what):
            self._table._content[name][self._index] = what

        def __dir__(self):
            return list(self._table._content)

    def __init__(self, length, columns1={}, *columns2, **columns3):
        self.step = 1
        self.start = 0
        self.length = length
        self._content = collections.OrderedDict()

        seen = set()
        if isinstance(columns1, dict):
            for n, x in columns1.items():
                if n in seen:
                    raise ValueError("field {0} occurs more than once".format(repr(n)))
                seen.add(n)

                self[n] = x
                if len(columns2) != 0:
                    raise TypeError("only one positional argument when the first argument is a dict")

        elif isinstance(columns1, (collections.Sequence, numpy.ndarray, awkward.array.base.AwkwardArray)):
            self["f0"] = columns1
            for i, x in enumerate(columns2):
                self["f" + str(i + 1)] = x

        else:
            raise TypeError("positional arguments may be a single dict or varargs of unnamed arrays")

        for n, x in columns3.items():
            if n in seen:
                raise ValueError("field {0} occurs more than once".format(repr(n)))
            seen.add(n)

            self[n] = x

    # the "basis" is step, start, and length; stop is computed from these
    # (and is approximate when length % abs(step) != 0)

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        if not isinstance(value, (numbers.Integral, numpy.integer)) or value == 0:
            raise TypeError("step must be a non-zero integer")
        self._step = value

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        if not isinstance(value, (numbers.Integral, numpy.integer)) or value < 0:
            raise TypeError("start must be a non-negative integer")
        self._start = value

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        if not isinstance(value, (numbers.Integral, numpy.integer)) or value < 0:
            raise TypeError("length must be a non-negative integer")
        self._length = value

    @property
    def stop(self):
        out = self._start + self._step * self._length
        if out < 0:
            return None
        else:
            return out

    @stop.setter
    def stop(self, value):
        if not isinstance(value, (numbers.Integral, numpy.integer)) or value < 0:
            raise TypeError("stop must be a non-negative integer")

        # length = int(math.ceil(float(abs(value - self._start)) / abs(self._step)))
        d, m = divmod(abs(self._start, value), abs(self._step))
        self._length = d + (1 if m != 0 else 0)

    @property
    def writeable(self):
        return True

    @property
    def dtype(self):
        return numpy.dtype([(n, x.dtype) for n, x in self._content.items()])

    @property
    def shape(self):
        return (self._length,)
        
    def __len__(self):
        return self._length            # data can grow by appending fields before increasing _length

    def _check_length(self, x):
        if self._step > 0:
            lastrow = self._start + self._step*(self._length - 1)
        else:
            lastrow = self._start
        if lastrow >= len(x):
            raise ValueError("last table row index ({0}) must be smaller than all field array lengths".format(lastrow))
        return x

    def __getitem__(self, where):
        # TODO: optimized special case for step == 1 start == 0 getting integer index

        if isinstance(where, awkward.util.string):
            return self._check_length(self._content[where])[self.start:self.stop:self.step]

        elif isinstance(where, (numbers.Integral, numpy.integer)):
            if where < 0:
                normwhere = where + self._length
            else:
                normwhere = where

            if not 0 <= normwhere < self._length:
                IndexError("index {0} out of bounds for length {1}".format(where, self._length))

            return self.Row(self, self._start + self._step*normwhere)

        elif isinstance(where, slice):
            out = self.__class__(self._length, self._content)
            start, stop, step = where.indices(self._length)

            out._start = self._start + self._step*start
            out._step = self._step*step

            d, m = divmod(abs(start - stop), abs(step))
            out._length = d + (1 if m != 0 else 0)

            return out
            
        else:
            try:
                assert all(isinstance(x, awkward.util.string) for x in where)

            except (TypeError, AssertionError):
                where = numpy.array(where, copy=False)
                out = Table(0, dict((n, self._check_length(x)[self.start:self.stop:self.step][where]) for n, x in self._content.items()))
                if len(where.shape) == 1 and issubclass(where.dtype.type, numpy.integer):
                    out._length = len(where)
                elif len(where.shape) == 1 and issubclass(where.dtype.type, (numpy.bool, numpy.bool_)):
                    out._length = numpy.count_nonzero(where)
                else:
                    raise TypeError("cannot interpret shape {0}, dtype {1} as a fancy index or mask".format(where.shape, where.dtype))
                return out

            else:
                out = Table(self._length, dict((n, self._check_length(self._content[n])) for n in where))
                out._start = self._start
                out._step = self._step
                return out

    def __setitem__(self, where, what):
        if isinstance(where, awkward.util.string):
            try:
                array = self._content[where]

            except KeyError:
                if self._start != 0 or self._step != 1:
                    raise TypeError("only add new columns to the original table, not a table slice (start is {0} and step is {1})".format(self._start, self._step))
                self._content[where] = self._toarray(what, self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

            else:
                self._check_length(array)[self.start:self.stop:self.step] = what

        else:
            try:
                assert all(isinstance(x, awkward.util.string) for x in where)

            except (TypeError, AssertionError):
                for x in self._content.values():
                    self._check_length(x)[self.start:self.stop:self.step][where] = what

            else:
                if isinstance(what, (collections.Sequence, numpy.ndarray, awkward.array.base.AwkwardArray)) and len(what) == 1:
                    for n in where:
                        self._check_length(self._content[n])[self.start:self.stop:self.step] = what[0]

                elif isinstance(what, (collections.Sequence, numpy.ndarray, awkward.array.base.AwkwardArray)):
                    if len(what) != len(where):
                        raise ValueError("cannot copy seqence with size {0} to array axis with dimension {1}".format(len(what), len(where)))
                    for wht, n in zip(what, where):
                        self._check_length(self._content[n])[self.start:self.stop:self.step] = wht

                else:
                    for n in where:
                        self._check_length(self._content[n])[self.start:self.stop:self.step] = what

    def __iter__(self):
        i = self._start
        stop = self._start + self._step*self._length
        if self._step > 0:
            while i < stop:
                yield self.Row(self, i)
                i += self._step
        else:
            while i > stop:
                yield self.Row(self, i)
                i += self._step

    def __repr__(self):
        return "<Table {0} x {1} at {2:012x}>".format(self._length, len(self._content), id(self))

    def tolist(self):
        return [dict((n, self._check_length(x)[self.start:self.stop:self.step][i]) for n, x in self._content.items()) for i in range(self._length)]
