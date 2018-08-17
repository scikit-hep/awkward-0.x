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

def fromarray(array):
    if isinstance(array, numpy.ndarray):
        if array.dtype.names is None:
            out = ArrayType(*(array.shape + (array.dtype,)))
            if isinstance(array, numpy.ma.MaskedArray):
                out = OptionType(out)
            return out
        else:
            table = TableType.__new__(TableType)
            table._fields = awkward.util.OrderedDict()
            for n in array.dtype.names:
                table[n] = array.dtype[n]
            out = ArrayType(*(array.shape + (table,)))
            if isinstance(array, numpy.ma.MaskedArray):
                out = OptionType(out)
            return out
    else:
        return array.type

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

    def _labeled(self):
        seen = set()
        labeled = []
        def find(x):
            if isinstance(x, Type):
                if id(x) not in seen:
                    seen.add(id(x))
                    if isinstance(x, ArrayType):
                        find(x._to)
                    elif isinstance(x, TableType):
                        for y in x._fields.values():
                            find(y)
                    elif isinstance(x, UnionType):
                        for y in x._possibilities:
                            find(y)
                    elif isinstance(x, OptionType):
                        find(x._type)
                else:
                    labeled.append(x)
        find(self)
        return labeled

    def __repr__(self):
        return self._repr(self._labeled(), set())

    def _repr(self, labeled, seen):
        if id(self) in seen:
            for i, x in enumerate(labeled):
                if self is x:
                    return "T{0}".format(i)
        else:
            for i, x in enumerate(labeled):
                if self is x:
                    out = "T{0} := ".format(i)
                    break
            else:
                out = ""
            seen.add(id(self))
            return out + self._subrepr(labeled, seen)

    def __str__(self):
        return self._str(self._labeled(), set(), "")

    def _str(self, labeled, seen, indent):
        if id(self) in seen:
            for i, x in enumerate(labeled):
                if self is x:
                    return indent + "T{0}".format(i)
            raise AssertionError("{0} not in {1}".format(id(self), seen))
        else:
            for i, x in enumerate(labeled):
                if self is x:
                    out = indent + "T{0} := ".format(i)
                    break
            else:
                out = ""
            seen.add(id(self))
            return out + self._substr(labeled, seen, indent + (" " * len(out)))
        
    @staticmethod
    def _copy(x, seen):
        if id(x) in seen:
            return seen[id(x)]

        elif isinstance(x, ArrayType):
            seen[id(x)] = ArrayType.__new__(ArrayType)
            seen[id(x)]._takes = x.takes
            seen[id(x)]._to = Type._copy(x.to, seen)
            return seen[id(x)]

        elif isinstance(x, TableType):
            seen[id(x)] = TableType.__new__(TableType)
            seen[id(x)]._fields = awkward.util.OrderedDict()
            for n, y in x._fields.items():
                seen[id(x)]._fields[n] = Type._copy(y, seen)
            return seen[id(x)]

        elif isinstance(x, UnionType):
            seen[id(x)] = UnionType.__new__(UnionType)
            seen[id(x)]._possibilities = []
            for y in x._possibilities:
                seen[id(x)]._possibilities.append(Type._copy(y, seen))
            return seen[id(x)]

        elif isinstance(x, OptionType):
            seen[id(x)] = OptionType.__new__(OptionType)
            seen[id(x)]._type = Type._copy(x._type, seen)
            return seen[id(x)]

        else:
            return x

    @staticmethod
    def _canonical(x, seen):
        if id(x) not in seen:
            # apply string-integer commutation so that TableTypes are nested as deeply as possible
            if isinstance(x, TableType) and len(x._fields) > 0 and all(isinstance(y, ArrayType) for y in x._fields.values()):
                newtable = TableType.__new__(TableType)
                newtable._fields = awkward.util.OrderedDict()
                first = None
                for n, y in x._fields.items():
                    if first is None:
                        first = y._takes
                    elif first != y._takes:
                        break
                    newtable._fields[n] = y._to
                else:
                    x = ArrayType.__new__(ArrayType)
                    x._takes = first
                    x._to = newtable

            # apply union(X, union(Y)) == union(X, Y)
            if isinstance(x, UnionType) and any(isinstance(y, UnionType) for y in x._possibilities):
                possibilities = []
                for y in x._possibilities:
                    if isinstance(y, UnionType):
                        possibilities.extend(y._possibilities)
                    else:
                        possibilities.append(y)
                x = UnionType.__new__(UnionType)
                x._possibilities = possibilities

            # apply optional(optional(X)) == optional(X)
            if isinstance(x, OptionType) and isinstance(x._type, OptionType):
                x = x._type

            seen.add(id(x))
            if isinstance(x, ArrayType):
                x._to = Type._canonical(x._to, seen)
            elif isinstance(x, TableType):
                for n in x._fields:
                    x._fields[n] = Type._canonical(x._fields[n], seen)
            elif isinstance(x, UnionType):
                for i in range(len(x._possibilities)):
                    x._possibilities[i] = Type._canonical(x._possibilities[i], seen)
            elif isinstance(x, OptionType):
                x._type = Type._canonical(x._type, seen)

        return x

    def __eq__(self, other):
        one = Type._canonical(Type._copy(self, {}), set())
        two = Type._canonical(Type._copy(other, {}), set())
        return one._eq(two, set())

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def _finaltype(x):
        if isinstance(x, type) and issubclass(x, (numbers.Number, numpy.generic)):
            return numpy.dtype(x)
        elif isinstance(x, (awkward.util.unicode, bytes)):
            return numpy.dtype(x)
        else:
            return x

class ArrayType(Type):
    def __init__(self, *args):
        if len(args) == 0:
            raise ValueError("type specification missing")

        elif len(args) == 1:
            self.takes = None
            self.to = args[0]

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
            self._to = self._finaltype(value)

    def _subrepr(self, labeled, seen):
        if isinstance(self._to, Type):
            to = self._to._repr(labeled, seen)
            if to.startswith("ArrayType(") and to.endswith(")"):
                to = to[10:-1]
            return "ArrayType({0}, {1})".format(repr(self._takes), to)
        else:
            return "ArrayType({0}, {1})".format(repr(self._takes), repr(self._to))

    def _substr(self, labeled, seen, indent):
        takes = "[0, {0}) -> ".format(self._takes)
        if isinstance(self._to, Type):
            to = self._to._str(labeled, seen, indent + (" " * len(takes))).lstrip(" ")
        else:
            to = str(self._to)
        return takes + to

    def _eq(self, other, seen):
        if self is other:
            return True
        elif id(self) in seen:
            return False
        else:
            seen.add(id(self))
            return isinstance(other, ArrayType) and self._takes == other._takes and self._to == other._to

    def __hash__(self):
        return hash((ArrayType, self._takes, self._to))

class TableType(Type):
    def __init__(self, **fields):
        self._fields = awkward.util.OrderedDict()
        for n, x in fields.items():
            self._fields[n] = x

    def __getitem__(self, key):
        return self._fields[key]

    def __setitem__(self, key, value):
        if isinstance(value, Type):
            self._fields[key] = value
        else:
            self._fields[key] = self._finaltype(value)

    def __delitem__(self, key):
        del self._fields[key]

    def __and__(self, other):
        out = TableType.__new__(TableType)
        out._fields = awkward.util.OrderedDict(list(self._fields.items()) + list(other._fields.items()))
        return out

    def _subrepr(self, labeled, seen):
        return "TableType({0})".format(", ".join("{0}={1}".format(n, x._repr(labeled, seen) if isinstance(x, Type) else repr(x)) for n, x in self._fields.items()))

    def _substr(self, labeled, seen, indent):
        width = max(len(repr(n)) for n in self._fields.keys())
        subindent = indent + (" " * width) + "    "
        out = []
        for n, x in self._fields.items():
            if isinstance(x, Type):
                to = x._str(labeled, seen, subindent).lstrip(" ")
            else:
                to = str(x)
            out.append(("{0}{1:%ds} -> {2}" % width).format(indent, repr(n), to))
        return "\n".join(out).lstrip(" ")

    def _eq(self, other, seen):
        if self is other:
            return True
        elif id(self) in seen:
            return False
        else:
            seen.add(id(self))
            return isinstance(other, TableType) and [(n, self._fields[n]) for n in sorted(self._fields)] == [(n, other._fields[n]) for n in sorted(other._fields)]

    def __hash__(self):
        return hash((TableType, tuple((n, self._fields[n]) for n in sorted(self._fields))))

class UnionType(Type):
    def __init__(self, *possibilities):
        self._possibilities = []
        for x in possibilities:
            self.append(x)

    def __len__(self):
        return len(self._possibilities)

    def __getitem__(self, index):
        return self._possibilities[index]

    def __setitem__(self, index, value):
        if isinstance(value, Type):
            self._possibilities[index] = value
        else:
            self._possibilities[index] = self._finaltype(value)

    def __delitem__(self, index):
        del self._possibilities[index]

    def append(self, value):
        if isinstance(value, Type):
            self._possibilities.append(value)
        else:
            self._possibilities.append(self._finaltype(value))

    def _subrepr(self, labeled, seen):
        return "UnionType({0})".format(", ".join(x._repr(labeled, seen) if isinstance(x, Type) else repr(x) for x in self._possibilities))

    def _substr(self, labeled, seen, indent):
        subs = [x._str(labeled, seen, indent + " ") if isinstance(x, Type) else str(x) for x in self._possibilities]
        def lstrip(x):
            if x.startswith(indent + " "):
                return x[len(indent) + 1:]
            else:
                return x
        width = max(len(lstrip(y)) for x in subs for y in x.split("\n"))
        out = [x + " " * (width - len(lstrip(x.split("\n")[-1]))) for x in subs]
        return "(" + (" |\n" + indent + " ").join(out) + " )"

    def _eq(self, other, seen):
        if self is other:
            return True
        elif id(self) in seen:
            return False
        else:
            seen.add(id(self))
            return isinstance(other, UnionType) and set(self._possibilities) == set(other._possibilities)

    def __hash__(self):
        return hash((UnionType, tuple(self._possibilities)))

class OptionType(Type):
    def __init__(self, type):
        self.type = type

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        if isinstance(value, Type):
            self._type = value
        else:
            self._type = self._finaltype(self._type)

    def _subrepr(self, labeled, seen):
        return "OptionType({0})".format(self._type._repr(labeled, seen) if isinstance(self._type, Type) else repr(self._type))

    def _substr(self, labeled, seen, indent):
        if isinstance(self._type, Type):
            type = self._type._str(labeled, seen, indent + "  ").lstrip(" ")
        else:
            type = str(self._type)
        return "?({0})".format(type)

    def _eq(self, other, seen):
        if self is other:
            return True
        elif id(self) in seen:
            return False
        else:
            seen.add(id(self))
            return isinstance(other, OptionType) and self._type == other._type

    def __hash__(self):
        return hash((OptionType, self._type))
