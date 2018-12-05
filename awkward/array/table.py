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

import types

import awkward.array.base
import awkward.type
import awkward.util

class Table(awkward.array.base.AwkwardArray):
    """
    Table
    """

    ##################### class Row

    class Row(object):
        """
        Table.Row
        """

        __slots__ = ["_table", "_index"]

        def __init__(self, table, index):
            self._table = table
            self._index = index

        def __repr__(self):
            return "<{0} {1}>".format(self._table.rowname, self._index)

        @property
        def at(self):
            return awkward.array.base.At(self)

        def __contains__(self, name):
            return name in self._table._content

        def tolist(self):
            return dict((n, self._table._try_tolist(x[self._index])) for n, x in self._table._content.items())

        def __getitem__(self, where):
            if isinstance(where, awkward.util.string):
                try:
                    return self._table._content[where][self._index]
                except KeyError:
                    raise ValueError("no column named {0}".format(repr(where)))

            elif awkward.util.isstringslice(where):
                content = awkward.util.OrderedDict()
                for n in where:
                    try:
                        content[n] = self._table._content[n]
                    except KeyError:
                        raise ValueError("no column named {0}".format(repr(n)))
                table = self._table.copy(content=content)
                return table.Row(table, self._index)

            else:
                index = self._index
                if not isinstance(index, tuple):
                    index = (index,)
                if not isinstance(where, tuple):
                    where = (where,)
                return self._table.Row(table, index + where)

        def __dir__(self):
            return ["at", "tolist"]

        def __iter__(self, checkiter=True):
            if checkiter:
                self._checkiter()
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

        elif isinstance(view, tuple) and len(view) == 3 and all(isinstance(x, awkward.util.integer) for x in view):
            start, step, length = view
            out = base.copy()
            out._view = int(start), int(step), int(length)
            out._base = base
            return out

        elif isinstance(view, awkward.util.numpy.ndarray) and issubclass(view.dtype.type, awkward.util.integer):
            out = base.copy()
            out._view = view
            out._base = base
            return out
            
        else:
            raise TypeError("view must be None, a 3-tuple of integers, or a Numpy array of integers")

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

    def __awkward_persist__(self, ident, fill, prefix, suffix, schemasuffix, storage, compression, **kwargs):
        self._valid()
        out = {"call": ["awkward", self.__class__.__name__, "frompairs"],
               "args": [{"pairs": [[n, fill(x, self.__class__.__name__ + ".content", prefix, suffix, schemasuffix, storage, compression, **kwargs)] for n, x in self._content.items()]}]}
        if isinstance(self._view, tuple):
            start, step, length = self._view
            out = {"call": ["awkward", self.__class__.__name__, "fromview"],
                   "args": [{"tuple": [{"json": start}, {"json": step}, {"json": length}]}, out]}

        elif isinstance(self._view, awkward.util.numpy.ndarray):
            out = {"call": ["awkward", self.__class__.__name__, "fromview"],
                   "args": [fill(self._view, self.__class__.__name__ + ".view", prefix, suffix, schemasuffix, storage, compression, **kwargs), out]}

        out["id"] = ident
        return out

    @property
    def base(self):
        return self._base

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        if not isinstance(value, dict) or not all(isinstance(n, awkward.util.string) for n in value):
            raise TypeError("content must be a dict from strings to arrays")
        for n in list(value):
            value[n] = awkward.util.toarray(value[n], self.DEFAULTTYPE)
        self._content = value

    def __len__(self):
        return self._length()

    def _gettype(self, seen):
        out = awkward.type.TableType()
        for n, x in self._content.items():
            out[n] = awkward.type._fromarray(x, seen)
        return out

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
        if isinstance(head, awkward.util.integer):
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
            head = awkward.util.toarray(head, self.INDEXTYPE)
            if issubclass(head.dtype.type, awkward.util.numpy.integer):
                length = self._length()
                negative = (head < 0)
                if negative.any():
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

    def _valid(self):
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
        if awkward.util.isstringslice(where):
            if isinstance(where, awkward.util.string):
                index = self._index()
                try:
                    if index is None:
                        return self._content[where][:self._length()]
                    else:
                        return self._content[where][index]
                except KeyError:
                    raise ValueError("no column named {0}".format(repr(where)))
            else:
                content = awkward.util.OrderedDict()
                for n in where:
                    try:
                        content[n] = self._content[n]
                    except KeyError:
                        raise ValueError("no column named {0}".format(repr(n)))
                return self.copy(content=content)

        if isinstance(where, tuple) and where == ():
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        if tail != ():
            raise NotImplementedError("multidimensional index through a Table (TODO: needed for [0, n) -> [0, m) -> \"table\" -> ...)")

        newslice = self._newslice(head)

        if isinstance(newslice, awkward.util.integer):
            return self.Row(self, newslice)

        else:
            out = self.copy(content=self._content)
            out._view = newslice
            out._base = self._base
            return out

    def __setitem__(self, where, what):
        if self._view is not None:
            raise ValueError("new columns can only be attached to the original Table, not a view (try table.base['col'] = array)")

        if isinstance(where, awkward.util.string):
            self._content[where] = awkward.util.toarray(what, self.DEFAULTTYPE)

        elif awkward.util.isstringslice(where):
            if len(where) != len(what):
                raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
            for x, y in zip(where, what):
                self._content[x] = awkward.util.toarray(y, self.DEFAULTTYPE)

        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __delitem__(self, where):
        if self._view is not None:
            raise ValueError("columns can only be removed from the original Table, not a view (try del table.base['col'])")

        if isinstance(where, awkward.util.string):
            del self._content[where]
        elif awkward.util.isstringslice(where):
            for x in where:
                del self._content[x]
        else:
            raise TypeError("invalid index for removing column from Table: {0}".format(where))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        inputsdict = None
        for x in inputs:
            if isinstance(x, Table):
                x._valid()

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

    def any(self):
        return any(x.any() for x in self._content.values())

    def all(self):
        return all(x.all() for x in self._content.values())

    @classmethod
    def concat(cls, first, *rest):
        raise NotImplementedError

    @classmethod
    def zip(cls, columns1={}, *columns2, **columns3):
        return cls(columns1, *columns2, **columns3)

    @property
    def columns(self):
        return [x for x in self._content if awkward.util.isintstring(x)] + [x for x in self._content if awkward.util.isidentifier(x)]

    @property
    def allcolumns(self):
        return list(self._content)

    def astype(self, dtype):
        out = self.copy(content={})
        for n, x in self._content.items():
            out[n] = x.astype(dtype)
        return out

    def pandas(self):
        import pandas
        return pandas.DataFrame(self._content)
