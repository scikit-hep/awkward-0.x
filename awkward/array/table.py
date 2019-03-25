#!/usr/bin/env python

# Copyright (c) 2019, IRIS-HEP
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

import re
import types
from collections import OrderedDict
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward.array.base
import awkward.type
import awkward.util

class Table(awkward.array.base.AwkwardArray):
    """
    Table
    """

    ##################### class Row

    class Row(awkward.util.NDArrayOperatorsMixin):
        """
        Table.Row
        """

        __slots__ = ["_table", "_index"]

        def __init__(self, table, index):
            self._table = table
            self._index = index

        def __repr__(self):
            if self._table.istuple:
                return "({0})".format(", ".join(str(self[n]) for n in self._table._contents))
            elif getattr(self._table, "_showdict", False):
                return "<{0} {{{1}}}>".format(self._table._rowname, ", ".join("{0}: {1}".format(repr(n), str(self[n])) for n in self._table._contents))
            else:
                return "<{0} {1}>".format(self._table._rowname, self._index)

        def __contains__(self, name):
            return name in self._table._contents

        def tolist(self):
            if self._table.istuple:
                return tuple(self[n].tolist() for n in self._table._contents)
            else:
                return dict((n, self._table._try_tolist(x[self._index])) for n, x in self._table._contents.items())

        def __len__(self):
            i = 0
            while str(i) in self._table._contents:
                i += 1
            return i

        def __iter__(self, checkiter=True):
            if checkiter:
                self._table._checkiter()
            i = 0
            while str(i) in self._table._contents:
                yield self._table._contents[str(i)]
                i += 1

        def __getitem__(self, where):
            if isinstance(where, awkward.util.string):
                try:
                    return self._table._contents[where][self._index]
                except KeyError:
                    raise ValueError("no column named {0}".format(repr(where)))

            elif self._util_isstringslice(where):
                contents = OrderedDict()
                for n in where:
                    try:
                        contents[n] = self._table._contents[n]
                    except KeyError:
                        raise ValueError("no column named {0}".format(repr(n)))
                table = self._table.copy(contents=contents)
                return table.Row(table, self._index)

            else:
                index = self._index
                if not isinstance(index, tuple):
                    index = (index,)
                if not isinstance(where, tuple):
                    where = (where,)
                return self._table.Row(table, index + where)

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            if "out" in kwargs:
                raise NotImplementedError("in-place operations not supported")

            if method != "__call__":
                return NotImplemented

            torow = not any(not isinstance(x, Table.Row) and isinstance(x, Iterable) for x in inputs)

            inputs = list(inputs)
            for i in range(len(inputs)):
                if isinstance(inputs[i], Table.Row):
                    inputs[i] = inputs[i]._table[inputs[i]._index : inputs[i]._index + 1]

            result = getattr(ufunc, method)(*inputs, **kwargs)

            if torow:
                if isinstance(result, tuple):
                    out = []
                    for x in result:
                        if isinstance(x, Table):
                            out.append(awkward.array.objects.Methods.maybemixin(type(x), self._table.Table.Row)(x, 0))
                            out[-1]._table._showdict = True
                        else:
                            out.append(x)
                    return tuple(out)
                elif method == "at":
                    return None
                else:
                    out = awkward.array.objects.Methods.maybemixin(type(result), self._table.Table.Row)(result, 0)
                    out._table._showdict = True
                    return out

            else:
                return result

        def __eq__(self, other):
            if not isinstance(other, Table.Row):
                return False
            elif self._table is other._table and self._index == other._index:
                return True
            else:
                return set(self._table._contents) == set(other._table._contents) and all(self._table._contents[n][self._index] == other._table._contents[n][other._index] for n in self._table._contents)

        def __ne__(self, other):
            return not self.__eq__(other)

        @property
        def i0(self):
            return self["0"]

        @property
        def i1(self):
            return self["1"]

        @property
        def i2(self):
            return self["2"]

        @property
        def i3(self):
            return self["3"]

        @property
        def i4(self):
            return self["4"]

        @property
        def i5(self):
            return self["5"]

        @property
        def i6(self):
            return self["6"]

        @property
        def i7(self):
            return self["7"]

        @property
        def i8(self):
            return self["8"]

        @property
        def i9(self):
            return self["9"]

    ##################### class Table

    def __init__(self, columns1={}, *columns2, **columns3):
        self._view = None
        self._base = None
        self.rowname = "Row"
        self._contents = OrderedDict()

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

        seen.update(self._contents)

        for n, x in columns3.items():
            if n in seen:
                raise ValueError("column {0} occurs more than once".format(repr(n)))
            seen.add(n)

            self[n] = x

    def tolist(self):
        return list(x.tolist() for x in self)

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
        if self.check_prop_valid:
            if not isinstance(value, awkward.util.string):
                raise TypeError("rowname must be a string")
        self._rowname = value

    @classmethod
    def fromrec(cls, recarray):
        if not isinstance(recarray, cls.numpy.ndarray) or recarray.dtype.names is None:
            raise TypeError("recarray must be a Numpy structured array")
        out = cls()
        for n in recarray.dtype.names:
            out[n] = recarray[n]
        return out

    @classmethod
    def frompairs(cls, pairs):
        out = cls()
        for n, x in pairs:
            out[n] = x
        return out

    @classmethod
    def fromview(cls, view, base):
        if view is None:
            return base

        elif isinstance(view, tuple) and len(view) == 3 and all(cls._util_isinteger(x) for x in view):
            start, step, length = view
            out = base.copy()
            out._view = int(start), int(step), int(length)
            out._base = base
            return out

        elif isinstance(view, cls.numpy.ndarray) and cls._util_isintegertype(view.dtype.type):
            out = base.copy()
            out._view = view
            out._base = base
            return out
            
        else:
            raise TypeError("view must be None, a 3-tuple of integers, or a Numpy array of integers")

    def copy(self, contents=None):
        out = self.__class__.__new__(self.__class__)
        out._view = self._view
        out._base = self._base
        out._rowname = self._rowname
        out._contents = self._contents
        if contents is not None and isinstance(contents, dict):
            out._contents = OrderedDict(contents.items())
        elif contents is not None:
            out._contents = OrderedDict(contents)
        else:
            out._contents = OrderedDict(self._contents.items())
        return out

    def deepcopy(self, contents=None):
        out = self.copy(contents=contents)
        index = out._index()
        if index is None:
            out._contents = OrderedDict([(n, self._util_deepcopy(x)) for n, x in out._contents.items()])
        else:
            out._contents = OrderedDict([(n, self._util_deepcopy(x[index])) for n, x in out._contents.items()])
            out._view = None
            out._base = None
        return out

    def empty_like(self, **overrides):
        out = self.__class__.__new__(self.__class__)
        out._view = None
        out._base = None
        out._rowname = self._rowname
        out._contents = OrderedDict()
        return out

    def zeros_like(self, **overrides):
        out = self.empty_like(**overrides)
        for n, x in self._contents.items():
            if isinstance(x, self.numpy.ndarray):
                out[n] = self.numpy.zeros_like(x)
            else:
                out[n] = x.zeros_like(**overrides)
        return out

    def ones_like(self, **overrides):
        out = self.empty_like(**overrides)
        for n, x in self._contents.items():
            if isinstance(x, self.numpy.ndarray):
                out[n] = self.numpy.ones_like(x)
            else:
                out[n] = x.ones_like(**overrides)
        return out

    def __awkward_persist__(self, ident, fill, prefix, suffix, schemasuffix, storage, compression, **kwargs):
        self._valid()
        out = {"call": ["awkward", "Table", "frompairs"],
               "args": [{"pairs": [[n, fill(x, "Table.contents", prefix, suffix, schemasuffix, storage, compression, **kwargs)] for n, x in self._contents.items()]}]}
        if isinstance(self._view, tuple):
            start, step, length = self._view
            out = {"call": ["awkward", "Table", "fromview"],
                   "args": [{"tuple": [{"json": start}, {"json": step}, {"json": length}]}, out]}

        elif isinstance(self._view, self.numpy.ndarray):
            out = {"call": ["awkward", "Table", "fromview"],
                   "args": [fill(self._view, "Table" + ".view", prefix, suffix, schemasuffix, storage, compression, **kwargs), out]}

        out["id"] = ident
        return out

    @property
    def base(self):
        return self._base

    @property
    def contents(self):
        return self._contents

    @contents.setter
    def contents(self, value):
        if self.check_prop_valid:
            if not isinstance(value, dict) or not all(isinstance(n, awkward.util.string) for n in value):
                raise TypeError("contents must be a dict from strings to arrays")
        for n in list(value):
            value[n] = self._util_toarray(value[n], self.DEFAULTTYPE)
        self._contents = value

    def _getnbytes(self, seen):
        if id(self) in seen:
            return 0
        else:
            seen.add(id(self))
            return sum(x.nbytes if isinstance(x, self.numpy.ndarray) else x._getnbytes(seen) for x in self._contents.values())

    def __len__(self):
        return self._length()

    def _gettype(self, seen):
        out = awkward.type.TableType()
        for n, x in self._contents.items():
            out[n] = awkward.type._fromarray(x, seen)
        return out

    def _length(self):
        if self._view is None:
            if len(self._contents) == 0:
                return 0
            else:
                return min([len(x) for x in self._contents.values()])

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
        if self._util_isinteger(head):
            original_head = head

            if self._view is None:
                length = self._length()
                if head < 0:
                    head += length
                if not 0 <= head < length:
                    raise IndexError("index {0} out of bounds for length {1}".format(original_head, length))
                return head

            elif isinstance(self._view, tuple):
                mystart, mystep, mylength = self._view
                if head < 0:
                    head += mylength
                if not 0 <= head < mylength:
                    raise IndexError("index {0} out of bounds for length {1}".format(original_head, mylength))
                return mystart + mystep*head

            else:
                length = len(self._view)
                if head < 0:
                    head += length
                if not 0 <= head < length:
                    raise IndexError("index {0} out of bounds for length {1}".format(original_head, length))
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
            head = self._util_toarray(head, self.INDEXTYPE)
            if self._util_isintegertype(head.dtype.type):
                length = self._length()
                negative = (head < 0)
                if negative.any():
                    head[negative] += length
                if not self.numpy.bitwise_and(0 <= head, head < length).all():
                    raise IndexError("some indexes out of bounds for length {0}".format(length))

                if self._view is None:
                    return head

                elif isinstance(self._view, tuple):
                    mystart, mystep, mylength = self._view
                    return self.numpy.arange(mystart, mystart + mystep*mylength, mystep)[head]

                else:
                    return self._view[head]

            elif issubclass(head.dtype.type, (self.numpy.bool, self.numpy.bool_)):
                length = self._length()
                if len(head) != length:
                    raise IndexError("boolean index of length {0} does not fit array of length {1}".format(len(head), length))

                if self._view is None:
                    return self.numpy.arange(length)[head]

                elif isinstance(self._view, tuple):
                    mystart, mystep, mylength = self._view
                    return self.numpy.arange(mystart, mystart + mystep*mylength, mystep)[head]

                else:
                    return self._view[head]

            else:
                raise TypeError("cannot interpret dtype {0} as a fancy index or mask".format(head.dtype))

    def _valid(self):
        if self.check_whole_valid:
            pass

    def __iter__(self, checkiter=True):
        if checkiter:
            self._checkiter()

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
        if self._util_isstringslice(where):
            if isinstance(where, awkward.util.string):
                index = self._index()
                try:
                    if index is None:
                        return self._contents[where][:self._length()]
                    else:
                        return self._contents[where][index]
                except KeyError:
                    raise ValueError("no column named {0}".format(repr(where)))
            else:
                contents = OrderedDict()
                for n in where:
                    try:
                        contents[n] = self._contents[n]
                    except KeyError:
                        raise ValueError("no column named {0}".format(repr(n)))
                return self.copy(contents=contents)

        if isinstance(where, tuple) and where == ():
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        if tail != ():
            raise NotImplementedError("multidimensional index through a Table (TODO: needed for [0, n) -> [0, m) -> \"table\" -> ...)")

        newslice = self._newslice(head)

        if self._util_isinteger(newslice):
            return self.Row(self, newslice)

        else:
            out = self.copy(contents=self._contents)
            out._view = newslice
            out._base = self
            return out

    def __setitem__(self, where, what):
        if self._view is not None:
            raise ValueError("new columns can only be attached to the original Table, not a view (try table.base['col'] = array)")

        if isinstance(where, awkward.util.string):
            try:
                len(what)
            except TypeError:
                what = self.numpy.full(len(self), what)
            self._contents[where] = self._util_toarray(what, self.DEFAULTTYPE)

        elif self._util_isstringslice(where):
            if len(where) != len(what):
                raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
            for x, y in zip(where, what):
                try:
                    len(y)
                except TypeError:
                    y = self.numpy.full(len(self), y)
                self._contents[x] = self._util_toarray(y, self.DEFAULTTYPE)

        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __delitem__(self, where):
        if self._view is not None:
            raise ValueError("columns can only be removed from the original Table, not a view (try del table.base['col'])")

        if isinstance(where, awkward.util.string):
            del self._contents[where]
        elif self._util_isstringslice(where):
            for x in where:
                del self._contents[x]
        else:
            raise TypeError("invalid index for removing column from Table: {0}".format(where))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            raise NotImplementedError("in-place operations not supported")

        if method != "__call__":
            return NotImplemented

        inputsdict = None
        for x in inputs:
            if isinstance(x, Table):
                x._valid()

                if inputsdict is None:
                    inputsdict = OrderedDict([(n, []) for n in x._contents])
                    table = x
                elif set(inputsdict) != set(x._contents):
                    raise ValueError("Tables have different sets of columns")

        assert inputsdict is not None

        for x in inputs:
            if isinstance(x, Table):
                index = x._index()
                for n, y in x._contents.items():
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

        if self._util_iscomparison(ufunc):
            out = None
            for x in newcolumns.values():
                assert isinstance(x, self.numpy.ndarray)
                assert issubclass(x.dtype.type, (self.numpy.bool_, self.numpy.bool))
                if out is None:
                    out = x
                else:
                    out = self.numpy.bitwise_and(out, x, out=out)
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

    @property
    def istuple(self):
        return self._rowname == "tuple" and list(self._contents) == [str(x) for x in range(len(self._contents))]

    def flattentuple(self):
        out = self.copy()
        out._contents = OrderedDict([(n, x.flattentuple() if isinstance(x, Table) else x) for n, x in out._contents.items()])

        if self.istuple:
            contents = OrderedDict()
            for n, x in out._contents.items():
                if isinstance(x, Table) and x.istuple:
                    if x._view is None:
                        view = slice(x._length())

                    elif isinstance(x._view, tuple):
                        start, step, length = x._view
                        stop = start + step*length
                        if stop < 0:
                            stop = None
                        view = slice(start, stop, step)

                    else:
                        view = x._view

                    for y in x._contents.values():
                        contents[str(len(contents))] = y[view]

                else:
                    contents[str(len(contents))] = x

            out._contents = contents

        return out

    def _hasjagged(self):
        return False

    def _reduce(self, ufunc, identity, dtype, regularaxis):
        out = self.Table.named({
            self.numpy.bitwise_or: "Any",
            self.numpy.bitwise_and: "All",
            None: "Count",
            self.numpy.count_nonzero: "CountNonzero",
            self.numpy.add: "Sum",
            self.numpy.multiply: "Prod",
            self.numpy.minimum: "Min",
            self.numpy.maximum: "Max"
            }[ufunc])
        out._showdict = True
        for n in self._contents:
            index = self._index()
            if index is None:
                x = self._contents[n][:self._length()]
            else:
                x = self._contents[n][index]
            out[n] = self.numpy.array([self._util_reduce(x, ufunc, identity, dtype, regularaxis)])
        return out.Row(out, 0)

    @property
    def columns(self):
        return list(self._contents)

    def astype(self, dtype):
        out = self.copy(contents={})
        for n, x in self._contents.items():
            out[n] = x.astype(dtype)
        return out

    def fillna(self, value):
        out = self.copy(contents={})
        for n, x in self._contents.items():
            out[n] = self._util_fillna(x, value)
        return out
