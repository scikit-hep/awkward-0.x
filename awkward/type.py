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

import numbers

import numpy

import awkward.util

class Type(object):
    def __or__(self, other):
        out = UnionType.__new__(UnionType)

        if isinstance(self, UnionType) and isinstance(other, UnionType):
            out._possibilities = self._possibilities + other._possibilities
        elif isinstance(self, UnionType):
            out._possibilities = self._possibilities + [other]
        elif isinstance(other, UnionType):
            out._possibilities = [self] + other._possibilities
        else:
            out._possibilities = [self, other]

        return out

    def __repr__(self):
        memo = set()
        labeled = []
        def find(x):
            if not isinstance(x, numpy.dtype):
                if id(x) not in memo:
                    memo.add(id(x))
                    if isinstance(x, ArrayType):
                        find(x.to)
                    elif isinstance(x, TableType):
                        for y in x._fields.values():
                            find(y)
                    elif isinstance(x, UnionType):
                        for y in x._possibilities:
                            find(y)
                else:
                    labeled.add(x)
        find(self)
        return self._repr(labeled, set())

    def _repr(self, labeled, seen):
        if id(self) in seen:
            for i, x in enumerate(labeled):
                if self is x:
                    return "T{0}".format(i)
        else:
            for i, x in enumerate(labeled):
                if self is x:
                    out = "T{0} = ".format(i)
                    break
            else:
                out = ""
            return out + self._subrepr(labeled, seen)

class ArrayType(Type):
    def __init__(self, *args):
        if len(args) == 0:
            raise ValueError("type specification missing")

        elif len(args) == 1:
            self.takes = None
            self.to = numpy.dtype(args[0])

        elif isinstance(args[0], awkward.util.string):
            self.__class__ = TableType
            self._fields = awkward.util.OrderedDict()
            if len(args) == 2:
                self[args[0]] = args[1]
            else:
                self[args[0]] = ArrayType(*args[1:])

        else:
            self.takes = args[0]
            if len(args) == 2:
                self.to = args[1]
            else:
                self.to = ArrayType(*args[1:])

    def _subrepr(self, labeled, seen):            
        HERE
        
    @property
    def takes(self):
        return self._takes

    @takes.setter
    def takes(self, value):
        if value == numpy.inf or (isinstance(value, (numbers.Integral, numpy.integer)) and value >= 0):
            self._takes = value

        else:
            raise ValueError("{0} is not allowed in type specification".format(value))

    @property
    def to(self):
        return self._to

    @to.setter
    def to(self, value):
        if isinstance(value, Type):
            self._to = value
        else:
            self._to = numpy.dtype(value)

class TableType(Type):
    def __init__(self):
        raise TypeError("TableTypes cannot be constructed directly; combine ArrayTypes that take strings with the & operator")

    def __getitem__(self, key):
        return self._fields[key]

    def __setitem__(self, key, value):
        if isinstance(value, Type):
            self._fields[key] = value
        else:
            self._fields[key] = numpy.dtype(value)

    def __delitem__(self, key):
        del self._fields[key]

    def __and__(self, other):
        out = TableType.__new__(TableType)
        out._fields = awkward.util.OrderedDict(list(self._fields.items()) + list(other._fields.items()))
        return out

class UnionType(Type):
    def __init__(self):
        raise TypeError("UnionTypes cannot be constructed directly; combine Types with the | operator")

    def __len__(self):
        return len(self._possibilities)

    def __getitem__(self, index):
        return self._possibilities[index]

    def __setitem__(self, index, value):
        if isinstance(value, Type):
            self._possibilities[index] = value
        else:
            self._possibilities[index] = numpy.dtype(value)

    def __delitem__(self, index):
        del self._possibilities[index]

    def append(self, value):
        if isinstance(value, Type):
            self._possibilities.append(value)
        else:
            self._possibilities.append(numpy.dtype(value))
