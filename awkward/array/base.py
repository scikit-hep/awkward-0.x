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

import types
import numbers

import numpy

import awkward
import awkward.persist
import awkward.type
import awkward.util

class AwkwardArray(awkward.util.NDArrayOperatorsMixin):
    """
    AwkwardArray: abstract base class
    """

    allow_tonumpy = True
    allow_iter = True
    check_prop_valid = True
    check_whole_valid = True

    @property
    def awkward(self):
        return awkward

    numpy = numpy
    DEFAULTTYPE = numpy.dtype(numpy.float64)
    CHARTYPE    = numpy.dtype(numpy.uint8)
    INDEXTYPE   = numpy.dtype(numpy.int64)
    TAGTYPE     = numpy.dtype(numpy.uint8)
    MASKTYPE    = numpy.dtype(numpy.bool_)
    BITMASKTYPE = numpy.dtype(numpy.uint8)
    BOOLTYPE    = numpy.dtype(numpy.bool_)

    def _checktonumpy(self):
        if not self.allow_tonumpy:
            raise RuntimeError("awkward.array.base.AwkwardArray.allow_tonumpy is False; refusing to convert to Numpy")

    def __array__(self, dtype=None):
        self._checktonumpy()

        if dtype is None:
            dtype = self.dtype

        if dtype == self.numpy.dtype(object):
            return self.numpy.array(list(self), dtype=dtype)
        else:
            return self.numpy.fromiter(self, dtype=dtype, count=len(self))

    def __getstate__(self):
        state = {}
        awkward.persist.serialize(self, state)
        return state

    def __setstate__(self, state):
        out = awkward.persist.deserialize(state)
        self.__dict__.update(out.__dict__)
        self.__class__ = out.__class__

    def _checkiter(self):
        if not self.allow_iter:
            raise RuntimeError("awkward.array.base.AwkwardArray.allow_iter is False; refusing to iterate")

    def __iter__(self, checkiter=True):
        if checkiter:
            self._checkiter()
        for i in range(len(self)):
            yield self[i]

    def __str__(self):
        if len(self) <= 6:
            return "[{0}]".format(" ".join(self._util_arraystr(x) for x in self.__iter__(checkiter=False)))

        else:
            first = self[:3]
            if isinstance(first, AwkwardArray):
                first = first.__iter__(checkiter=False)
            last = self[-3:]
            if isinstance(first, AwkwardArray):
                last = last.__iter__(checkiter=False)

            return "[{0} ... {1}]".format(" ".join(self._util_arraystr(x) for x in first), " ".join(self._util_arraystr(x) for x in last))

    def __repr__(self):
        return "<{0} {1} at 0x{2:012x}>".format(self.__class__.__name__, str(self), id(self))

    @property
    def type(self):
        return awkward.type.ArrayType(len(self), awkward.type._resolve(self._gettype({}), {}))

    @property
    def dtype(self):
        return self.type.dtype

    @property
    def shape(self):
        return self.type.shape

    def _try_tolist(self, x):
        try:
            return x.tolist()
        except AttributeError:
            return x

    def __bool__(self):
        raise ValueError("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")

    __nonzero__ = __bool__

    @property
    def size(self):
        return len(self)

    @property
    def nbytes(self):
        return self._getnbytes(set())

    def tolist(self):
        import awkward.array.table
        out = []
        for x in self:
            if isinstance(x, awkward.array.table.Table.Row):
                if x._table.istuple:
                    out.append(tuple(x[n].tolist() for n in x._table._contents))
                else:
                    out.append(dict((n, self._try_tolist(x[n])) for n in x._table._contents))
            elif isinstance(x, self.numpy.ma.core.MaskedConstant):
                out.append(None)
            else:
                out.append(self._try_tolist(x))
        return out

    def valid(self):
        try:
            self._valid()
        except:
            return False
        else:
            return True

    def any(self, regularaxis=None):
        return self._reduce(self.numpy.bitwise_or, False, self.BOOLTYPE, regularaxis)

    def all(self, regularaxis=None):
        return self._reduce(self.numpy.bitwise_and, True, self.BOOLTYPE, regularaxis)

    def count(self, regularaxis=None):
        return self._reduce(None, 0, None, regularaxis)

    def count_nonzero(self, regularaxis=None):
        return self._reduce(self.numpy.count_nonzero, 0, None, regularaxis)

    def sum(self, regularaxis=None):
        return self._reduce(self.numpy.add, 0, None, regularaxis)

    def prod(self, regularaxis=None):
        return self._reduce(self.numpy.multiply, 1, None, regularaxis)

    def min(self, regularaxis=None):
        return self._reduce(self.numpy.minimum, self.numpy.inf, None, regularaxis)

    def max(self, regularaxis=None):
        return self._reduce(self.numpy.maximum, -self.numpy.inf, None, regularaxis)

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

    @property
    def ChunkedArray(self):
        import awkward.array.chunked
        return awkward.array.chunked.ChunkedArray

    @property
    def AppendableArray(self):
        import awkward.array.chunked
        return awkward.array.chunked.AppendableArray

    @property
    def IndexedArray(self):
        import awkward.array.indexed
        return awkward.array.indexed.IndexedArray

    @property
    def SparseArray(self):
        import awkward.array.indexed
        return awkward.array.indexed.SparseArray

    @property
    def JaggedArray(self):
        import awkward.array.jagged
        return awkward.array.jagged.JaggedArray

    @property
    def MaskedArray(self):
        import awkward.array.masked
        return awkward.array.masked.MaskedArray

    @property
    def BitMaskedArray(self):
        import awkward.array.masked
        return awkward.array.masked.BitMaskedArray

    @property
    def IndexedMaskedArray(self):
        import awkward.array.masked
        return awkward.array.masked.IndexedMaskedArray

    @property
    def Methods(self):
        import awkward.array.objects
        return awkward.array.objects.Methods

    @property
    def ObjectArray(self):
        import awkward.array.objects
        return awkward.array.objects.ObjectArray

    @property
    def StringArray(self):
        import awkward.array.objects
        return awkward.array.objects.StringArray

    @property
    def Table(self):
        import awkward.array.table
        return awkward.array.table.Table

    @property
    def UnionArray(self):
        import awkward.array.union
        return awkward.array.union.UnionArray

    @property
    def VirtualArray(self):
        import awkward.array.virtual
        return awkward.array.virtual.VirtualArray

    @classmethod
    def _util_isinteger(cls, x):
        return isinstance(x, (numbers.Integral, cls.numpy.integer)) and not isinstance(x, (bool, cls.numpy.bool_, cls.numpy.bool))

    @classmethod
    def _util_isintegertype(cls, x):
        return issubclass(x, cls.numpy.integer) and not issubclass(x, (cls.numpy.bool_, cls.numpy.bool))

    @classmethod
    def _util_toarray(cls, value, defaultdtype, passthrough=None):
        import awkward.array.base
        if passthrough is None:
            passthrough = (cls.numpy.ndarray, AwkwardArray)
        if isinstance(value, passthrough):
            return value
        else:
            try:
                return cls.numpy.frombuffer(value, dtype=getattr(value, "dtype", defaultdtype)).reshape(getattr(value, "shape", -1))
            except AttributeError:
                if len(value) == 0:
                    return cls.numpy.array(value, dtype=defaultdtype, copy=False)
                else:
                    return cls.numpy.array(value, copy=False)

    @classmethod
    def _util_arraystr_draw(cls, x):
        if isinstance(x, list):
            if len(x) > 6:
                return "[" + " ".join(cls._util_arraystr_draw(y) for y in x[:3]) + " ... " + " ".join(cls._util_arraystr_draw(y) for y in x[-3:]) + "]"
            else:
                return "[" + " ".join(cls._util_arraystr_draw(y) for y in x) + "]"
        elif isinstance(x, tuple):
            return "(" + ", ".join(cls._util_arraystr_draw(y) for y in x) + ")"
        else:
            return repr(x)

    @classmethod
    def _util_arraystr(cls, array):
        if isinstance(array, cls.numpy.ndarray):
            return cls._util_arraystr_draw(array.tolist())
        elif isinstance(array, AwkwardArray):
            return str(array).replace("\n", "")
        else:
            return repr(array)

    @classmethod
    def _util_isnumpy(cls, dtype):
        if isinstance(dtype, cls.numpy.dtype):
            return True
        else:
            return dtype.isnumpy

    @classmethod
    def _util_deepcopy(cls, array):
        if array is None:
            return None
        elif isinstance(array, cls.numpy.ndarray):
            return array.copy()
        else:
            return array.deepcopy()

    @classmethod
    def _util_hasjagged(cls, array):
        return isinstance(array, AwkwardArray) and array._hasjagged()

    @classmethod
    def _util_reduce(cls, array, ufunc, identity, dtype, regularaxis):
        if isinstance(array, AwkwardArray):
            return array._reduce(ufunc, identity, dtype, regularaxis)

        elif len(array) == 0:
            if dtype is None:
                dtype = array.dtype
            return ufunc.reduce(cls.numpy.full((1,) + array.shape[1:], identity, dtype=dtype), axis=regularaxis)

        else:
            original = array
            if dtype is not None:
                array = cls.numpy.array(array, dtype=dtype, copy=False)
            if issubclass(array.dtype.type, (cls.numpy.floating, cls.numpy.complexfloating)):
                mask = cls.numpy.isnan(array)
                if mask.any():
                    if array is original:
                        array = array.copy()
                    array[mask] = identity
            return ufunc.reduce(array, axis=regularaxis)

    @classmethod
    def _util_concatenate(cls, arrays):
        if all(isinstance(x, cls.numpy.ndarray) for x in arrays):
            return cls.numpy.concatenate(arrays)
        else:
            return arrays[0].concatenate(arrays[1:])

    @classmethod
    def _util_isstringslice(cls, where):
        if isinstance(where, awkward.util.string):
            return True
        elif isinstance(where, tuple):
            return False
        elif isinstance(where, (cls.numpy.ndarray, AwkwardArray)) and issubclass(where.dtype.type, (numpy.str, numpy.str_)):
            return True
        elif isinstance(where, (cls.numpy.ndarray, AwkwardArray)) and issubclass(where.dtype.type, (numpy.object, numpy.object_)) and not issubclass(where.dtype.type, (numpy.bool, numpy.bool_)):
            return len(where) > 0 and all(isinstance(x, awkward.util.string) for x in where)
        elif isinstance(where, (cls.numpy.ndarray, AwkwardArray)):
            return False
        try:
            assert len(where) > 0 and all(isinstance(x, awkward.util.string) for x in where)
        except (TypeError, AssertionError):
            return False
        else:
            return True

    @classmethod
    def _util_iscomparison(cls, ufunc):
        return (ufunc is cls.numpy.less or
                ufunc is cls.numpy.less_equal or
                ufunc is cls.numpy.equal or
                ufunc is cls.numpy.not_equal or
                ufunc is cls.numpy.greater or
                ufunc is cls.numpy.greater_equal)

    @classmethod
    def _util_fillna(cls, array, value):
        if isinstance(array, cls.numpy.ndarray):
            data = cls.numpy.ma.getdata(array)
            mask = cls.numpy.ma.getmask(array)
            if mask is not cls.numpy.ma.nomask:
                data[mask] = value
            data[cls.numpy.isnan(data)] = value
            return data
        else:
            return array.fillna(value)

class AwkwardArrayWithContent(AwkwardArray):
    """
    AwkwardArrayWithContent: abstract base class
    """

    def __setitem__(self, where, what):
        if isinstance(where, awkward.util.string):
            self._content[where] = what

        elif self._util_isstringslice(where):
            if len(where) != len(what):
                raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
            for x, y in zip(where, what):
                self._content[x] = y

        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __delitem__(self, where):
        if isinstance(where, awkward.util.string):
            del self._content[where]
        elif self._util_isstringslice(where):
            for x in where:
                del self._content[x]
        else:
            raise TypeError("invalid index for removing column from Table: {0}".format(where))

    def _hasjagged(self):
        return self._util_hasjagged(self._content)

    @property
    def columns(self):
        if isinstance(self._content, self.numpy.ndarray):
            return []
        else:
            return self._content.columns

    def astype(self, dtype):
        return self.copy(content=self._content.astype(dtype))

    def fillna(self, value):
        return self.copy(content=self._util_fillna(self._content, value))
