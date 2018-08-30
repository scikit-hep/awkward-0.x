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

import functools
import numbers
import types

import awkward.array.base
import awkward.type
import awkward.util

class Table(awkward.array.base.AwkwardArray):
    ##################### class Row

    class Row(object):
        __slots__ = ["_table", "_index"]

        def __init__(self, table, index):
            self._table = table
            self._index = index

        def __repr__(self):
            return "<{0} {1}>".format(self._table.rowname, self._index)

        def __hasattr__(self, name):
            return name in self._table._content or "_" + name in self._table._content

        def __contains__(self, name):
            return name in self._table._content

        def __getattr__(self, name):
            if name == "tolist":
                return lambda: dict((n, self._table._try_tolist(x[self._index])) for n, x in self._table._content.items())

            content = self._table._content.get(name, None)
            if content is not None:
                return content

            content = self._table._content.get("_" + name, None)
            if content is not None:
                return content

            raise AttributeError("neither {0} nor _{1} are columns in this {2}".format(name, name, self._table.rowname))

        def __getitem__(self, where):
            if isinstance(where, awkward.util.string):
                return self._table._content[where][self._index]

            elif awkward.util.isstringslice(where):
                table = self._table.copy(content=awkward.util.OrderedDict([(n, self._table._content[n]) for n in where]))
                return table.Row(table, self._index)

            else:
                index = self._index
                if not isinstance(index, tuple):
                    index = (index,)
                if not isinstance(where, tuple):
                    where = (where,)
                return self._table.Row(table, index + where)

        def __dir__(self):
            return ["_" + x if len(x) == 0 or not x[0].isalpha() else x for x in self._table._content]

        def __iter__(self):
            i = 0
            while str(i) in self._table._content:
                yield self._table._content[str(i)]
                i += 1

        def __len__(self):
            i = 0
            while str(i) in self._table._content:
                i += 1
            return i

        def __eq__(self, other):
            if not isinstance(other, Table.Row):
                return False
            elif self._table is other._table and self._index == other._index:
                return True
            else:
                return set(self._table._content) == set(other._table._content) and all(self._table._content[n][self._index] == other._table._content[n][other._index] for n in self._table._content)

        def __ne__(self, other):
            return not self.__eq__(other)

    ##################### class Table

    def __init__(self, columns1={}, *columns2, **columns3):
        self._view = None
        self._base = None
        self.rowname = "Row"
        self._content = awkward.util.OrderedDict()

        seen = set()
        if isinstance(columns1, dict):
            for n, x in columns1.items():
                if n in seen:
                    raise ValueError("column {0} occurs more than once".format(repr(n)))
                seen.add(n)

                self[n] = x

            if len(columns2) != 0:
                raise TypeError("only one positional argument when first argument is a dict")

        else:
            self["0"] = columns1
            for i, x in enumerate(columns2):
                self[str(i + 1)] = x

        seen.update(self._content)

        for n, x in columns3.items():
            if n in seen:
                raise ValueError("column {0} occurs more than once".format(repr(n)))
            seen.add(n)

            self[n] = x

    @classmethod
    def named(cls, rowname, columns1={}, *columns2, **columns3):
        out = cls(columns1, *columns2, **columns3)
        out.rowname = rowname
        return out

    @property
    def rowname(self):
        return self._rowname

    @rowname.setter
    def rowname(self, value):
        if not isinstance(value, awkward.util.string):
            raise TypeError("rowname must be a string")
        self._rowname = value

    @classmethod
    def fromrec(cls, recarray):
        if not isinstance(recarray, awkward.util.numpy.ndarray) or recarray.dtype.names is None:
            raise TypeError("recarray must be a Numpy structured array")
        out = cls()
        for n in recarray.dtype.names:
            out[n] = recarray[n]
        return out

    def copy(self, content=None):
        out = self.__class__.__new__(self.__class__)
        out._view = self._view
        out._base = self._base
        out._rowname = self._rowname
        out._content = self._content
        if content is not None and isinstance(content, dict):
            out._content = awkward.util.OrderedDict(content.items())
        elif content is not None:
            out._content = awkward.util.OrderedDict(content)
        else:
            out._content = awkward.util.OrderedDict(self._content.items())
        return out

    def deepcopy(self, content=None):
        out = self.copy(content=content)
        index = out._index()
        if index is None:
            out._content = awkward.util.OrderedDict([(n, awkward.util.deepcopy(x)) for n, x in out._content.items()])
        else:
            out._content = awkward.util.OrderedDict([(n, awkward.util.deepcopy(x[index])) for n, x in out._content.items()])
            out._view = None
            out._base = None
        return out

    def empty_like(self, **overrides):
        out = self.__class__.__new__(self.__class__)
        out._view = None
        out._base = None
        out._rowname = self._rowname
        out._content = awkward.util.OrderedDict()
        return out

    def zeros_like(self, **overrides):
        out = self.empty_like(**overrides)
        for n, x in self._content.items():
            if isinstance(x, awkward.util.numpy.ndarray):
                out[n] = awkward.util.numpy.zeros_like(x)
            else:
                out[n] = x.zeros_like(**overrides)
        return out

    def ones_like(self, **overrides):
        out = self.empty_like(**overrides)
        for n, x in self._content.items():
            if isinstance(x, awkward.util.numpy.ndarray):
                out[n] = awkward.util.numpy.ones_like(x)
            else:
                out[n] = x.ones_like(**overrides)
        return out

    @property
    def columns(self):
        return [x for x in self._content if awkward.util.isintstring(x)] + [x for x in self._content if awkward.util.isidentifier(x)]

    @property
    def allcolumns(self):
        return list(self._content)

    @property
    def base(self):
        return self._base

    def _argfields(self, function):
        if not isinstance(function, types.FunctionType):
            raise TypeError("function (or lambda) required")

        required = function.__code__.co_varnames[:function.__code__.co_argcount]
        has_varargs = (function.__code__.co_flags & 0x04) != 0
        has_kwargs = (function.__code__.co_flags & 0x08) != 0

        args = []
        kwargs = {}

        order = self.columns

        for i, n in enumerate(required):
            if n in self._content:
                args.append(n)
            elif str(i) in self._content:
                args.append(str(i))
            else:
                args.append(order[i])

        if has_varargs:
            while str(i) in self._content:
                args.append(str(i))
                i += 1

        if has_kwargs:
            kwargs = [n for n in self._content if n not in required]

        return args, kwargs

    @property
    def dtype(self):
        return awkward.util.numpy.dtype([(n, x.dtype) for n, x in self._content.items()])

    def _length(self):
        if self._view is None:
            if len(self._content) == 0:
                return 0
            else:
                return min([len(x) for x in self._content.values()])

        elif isinstance(self._view, tuple):
            start, step, length = self._view
            return length

        else:
            return len(self._view)

    def _index(self):
        if self._view is None:
            return None

        elif isinstance(self._view, tuple):
            start, step, length = self._view
            stop = start + step*length
            if stop < 0:
                stop = None
            return slice(start, stop, step)

        else:
            return self._view

    def _newslice(self, head):
        if isinstance(head, (numbers.Integral, awkward.util.numpy.integer)):
            original_head = head

            if self._view is None:
                length = self._length()
                if head < 0:
                    head += length
                if not 0 <= head < length:
                    IndexError("index {0} out of bounds for length {1}".format(original_head, length))
                return head

            elif isinstance(self._view, tuple):
                mystart, mystep, mylength = self._view
                if head < 0:
                    head += mylength
                if not 0 <= head < mylength:
                    IndexError("index {0} out of bounds for length {1}".format(original_head, mylength))
                return mystart + mystep*head

            else:
                length = len(self._view)
                if head < 0:
                    head += length
                if not 0 <= head < length:
                    IndexError("index {0} out of bounds for length {1}".format(original_head, length))
                return self._view[head]

        elif isinstance(head, slice):
            if self._view is None or isinstance(self._view, tuple):
                start, stop, step = head.indices(self._length())
                if step == 0:
                    raise ValueError("slice step cannot be zero")
                if (step > 0 and stop - start > 0) or (step < 0 and stop - start < 0):
                    d, m = divmod(abs(start - stop), abs(step))
                    length = d + (1 if m != 0 else 0)
                else:
                    length = 0

            if self._view is None:
                return start, step, length

            elif isinstance(self._view, tuple):
                mystart, mystep, mylength = self._view
                if step > 0:
                    skip = start
                else:
                    skip = mylength - 1 - start
                return mystart + mystep*start, mystep*step, min(mylength - skip, length)

            else:
                return self._view[head]

        else:
            head = awkward.util.toarray(head, awkward.util.INDEXTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))
            if issubclass(head.dtype.type, awkward.util.numpy.integer):
                length = self._length()
                negative = (head < 0)
                head[negative] += length
                if not awkward.util.numpy.bitwise_and(0 <= head, head < length).all():
                    raise IndexError("some indexes out of bounds for length {0}".format(length))

                if self._view is None:
                    return head

                elif isinstance(self._view, tuple):
                    mystart, mystep, mylength = self._view
                    return awkward.util.numpy.arange(mystart, mystart + mystep*mylength, mystep)[head]

                else:
                    return self._view[head]

            elif issubclass(head.dtype.type, (awkward.util.numpy.bool, awkward.util.numpy.bool_)):
                length = self._length()
                if len(head) != length:
                    raise IndexError("boolean index of length {0} does not fit array of length {1}".format(len(head), length))

                if self._view is None:
                    return awkward.util.numpy.arange(length)[head]

                elif isinstance(self._view, tuple):
                    mystart, mystep, mylength = self._view
                    return awkward.util.numpy.arange(mystart, mystart + mystep*mylength, mystep)[head]

                else:
                    return self._view[head]

            else:
                raise TypeError("cannot interpret dtype {0} as a fancy index or mask".format(head.dtype))

    @property
    def shape(self):
        return (self._length(),)
        
    @property
    def type(self):
        return awkward.type.ArrayType(self._length(), functools.reduce(lambda a, b: a & b, [awkward.type.ArrayType(n, awkward.type.fromarray(x).to) for n, x in self._content.items()]))

    def __len__(self):
        return self._length()

    def _valid(self):
        return True

    def __iter__(self):
        if self._view is None:
            length = self._length()
            i = 0
            while i < length:
                yield self.Row(self, i)
                i += 1

        elif isinstance(self._view, tuple):
            mystart, mystep, mylength = self._view
            i = 0
            while i < mylength:
                yield self.Row(self, mystart + i*mystep)
                i += 1
        else:
            for i in self._view:
                yield self.Row(self, i)

    def __getitem__(self, where):
        if awkward.util.isstringslice(where):
            if isinstance(where, awkward.util.string):
                index = self._index()
                if index is None:
                    return self._content[where]
                else:
                    return self._content[where][index]
            else:
                return self.copy(content=[(n, self._content[n]) for n in where])

        if isinstance(where, tuple) and where == ():
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        newslice = self._newslice(head)

        if isinstance(newslice, (numbers.Integral, awkward.util.numpy.integer)):
            return self.Row(self, newslice)

        else:
            out = self.copy(content=self._content)
            out._view = newslice
            out._base = self._base
            return out

    def __setitem__(self, where, what):
        if self._view is not None:
                raise ValueError("new columns can only be attached to the original table, not a view (try table.base['col'] = array)")

        if isinstance(where, awkward.util.string):
            self._content[where] = awkward.util.toarray(what, awkward.util.CHARTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))

        elif awkward.util.isstringslice(where):
            if len(where) != len(what):
                raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
            for x, y in zip(where, what):
                self._content[x] = awkward.util.toarray(y, awkward.util.CHARTYPE, (awkward.util.numpy.ndarray, awkward.array.base.AwkwardArray))

        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        inputsdict = None
        for x in inputs:
            if isinstance(x, Table):
                if inputsdict is None:
                    inputsdict = awkward.util.OrderedDict([(n, []) for n in x._content])
                    table = x
                elif set(inputsdict) != set(x._content):
                    raise ValueError("Tables have different sets of columns")

        assert inputsdict is not None

        for x in inputs:
            if isinstance(x, Table):
                index = x._index()
                for n, y in x._content.items():
                    if index is None:
                        inputsdict[n].append(y)
                    else:
                        inputsdict[n].append(y[index])
            else:
                for n in inputsdict:
                    inputsdict[n].append(x)

        newcolumns = {}
        tuplelen = None
        for n, x in inputsdict.items():
            newcolumns[n] = getattr(ufunc, method)(*x, **kwargs)

            if tuplelen is None:
                if isinstance(x, tuple):
                    tuplelen = len(x)
                else:
                    tuplelen = False
            elif isinstance(x, tuple) != tuplelen:
                raise AssertionError("ufuncs return tuples of different lengths or some tuples and some non-tuples")

        assert len(newcolumns) != 0
        assert tuplelen is not None

        if awkward.util.iscomparison(ufunc):
            out = None
            for x in newcolumns.values():
                assert isinstance(x, awkward.util.numpy.ndarray)
                assert issubclass(x.dtype.type, (awkward.util.numpy.bool_, awkward.util.numpy.bool))
                if out is None:
                    out = x
                else:
                    out = awkward.util.numpy.bitwise_and(out, x, out=out)
            assert out is not None
            return out

        if method == "at":
            return None

        if tuplelen is False:
            out = table.empty_like()
            for n in inputsdict:
                out[n] = newcolumns[n]
            return out

        else:
            out = [table.empty_like() for i in range(tuplelen)]
            for n in inputsdict:
                for i in range(tuplelen):
                    out[i][n] = newcolumns[n]
            return tuple(out)

    def tolist(self):
        return list(x.tolist() for x in self)
    
    def pandas(self):
        import pandas
        return pandas.DataFrame(self._content)
