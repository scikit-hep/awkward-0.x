#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-0.x/blob/master/LICENSE

import types
import numbers
import re
import keyword
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy

import awkward0
import awkward0.persist
import awkward0.type
import awkward0.util

from awkward0.util import bothmethod

class AwkwardArray(awkward0.util.NDArrayOperatorsMixin):
    """
    AwkwardArray: abstract base class
    """

    allow_tonumpy = True
    allow_iter = True
    check_prop_valid = True
    check_whole_valid = True

    @property
    def awkward(self):
        return awkward0

    @property
    def awkward0(self):
        return awkward0

    numpy = numpy
    DEFAULTTYPE = numpy.dtype(numpy.float64)
    CHARTYPE    = numpy.dtype(numpy.uint8)
    INDEXTYPE   = numpy.dtype(numpy.int64)
    TAGTYPE     = numpy.dtype(numpy.uint8)
    MASKTYPE    = numpy.dtype(numpy.bool_)
    BITMASKTYPE = numpy.dtype(numpy.uint8)
    BOOLTYPE    = numpy.dtype(numpy.bool_)

    def __init__(self, *args, **kwds):
        raise TypeError("{0} is an abstract base class; do not instantiate".format(type(self)))

    def __round__(self, n=0):
        if n == 0:
            return self.numpy.rint(self)
        else:
            factor = 10**n
            return self.numpy.rint(self * factor) / factor

    def _checktonumpy(self):
        if not self.allow_tonumpy:
            raise RuntimeError("awkward0.array.base.AwkwardArray.allow_tonumpy is False; refusing to convert to Numpy")

    def __array__(self, dtype=None):
        self._checktonumpy()

        if dtype is None:
            dtype = self.dtype

        out = self.numpy.empty(len(self), dtype=dtype)
        for i, x in enumerate(self):
            out[i] = x
        return out

    def __getstate__(self):
        state = {}
        awkward0.persist.serialize(self, state)
        return state

    def __setstate__(self, state):
        out = awkward0.persist.deserialize(state)
        self.__dict__.update(out.__dict__)
        self.__class__ = out.__class__

    def __reduce__(self):
        state = {}
        awkward0.persist.serialize(self, state)
        return (awkward0.persist.deserialize, (state,))

    def _checkiter(self):
        if not self.allow_iter:
            raise RuntimeError("awkward0.array.base.AwkwardArray.allow_iter is False; refusing to iterate")

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
        return awkward0.type.ArrayType(len(self), awkward0.type._resolve(self._gettype({}), {}))

    @property
    def layout(self):
        return awkward0.type.Layout(self)

    @property
    def dtype(self):
        return self.type.dtype

    @property
    def shape(self):
        return self.type.shape

    @property
    def ndim(self):
        return len(self.shape)

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
        out = 1
        for x in self.shape:
            out *= x
        return out

    @property
    def nbytes(self):
        return self._getnbytes(set())

    def tolist(self):
        import awkward0.array.table
        out = []
        for x in self:
            if isinstance(x, awkward0.array.table.Table.Row):
                if x._table.istuple:
                    out.append(tuple(self._try_tolist(x[n]) for n in x._table._contents))
                else:
                    out.append(dict((n, self._try_tolist(x[n])) for n in x._table._contents))
            elif isinstance(x, self.numpy.ma.core.MaskedConstant):
                out.append(None)
            else:
                out.append(self._try_tolist(x))
        return out

    def valid(self, exception=False, message=False):
        try:
            self._valid()
        except Exception as err:
            if exception:
                raise err
            elif message:
                return "{0}: {1}".format(type(err), str(err))
            else:
                return False
        else:
            if message:
                return None
            else:
                return True

    def reduce(self, ufunc, identity):
        return self._reduce(ufunc, identity, None)

    def any(self):
        return self._reduce(self.numpy.bitwise_or, False, self.BOOLTYPE)

    def all(self):
        return self._reduce(self.numpy.bitwise_and, True, self.BOOLTYPE)

    def count(self):
        return self._reduce(None, 0, None)

    def count_nonzero(self):
        return self._reduce(self.numpy.count_nonzero, 0, None)

    def sum(self):
        return self._reduce(self.numpy.add, 0, None)

    def prod(self):
        return self._reduce(self.numpy.multiply, 1, None)

    def min(self):
        return self._reduce(self.numpy.minimum, self.numpy.inf, None)

    def max(self):
        return self._reduce(self.numpy.maximum, -self.numpy.inf, None)

    def moment(self, n, weight=None):
        with self.numpy.errstate(invalid="ignore"):
            if weight is None:
                return self.numpy.true_divide((self**n).sum(), self.count())
            else:
                return self.numpy.true_divide(((self * weight)**n).sum(), (self * 0 + weight).sum())

    def mean(self, weight=None):
        with self.numpy.errstate(invalid="ignore"):
            if weight is None:
                return self.numpy.true_divide(self.sum(), self.count())
            else:
                return self.numpy.true_divide((self * weight).sum(), (self * 0 + weight).sum())

    def var(self, weight=None, ddof=0):
        with self.numpy.errstate(invalid="ignore"):
            if weight is None:
                denom = self.count()
                one = self.numpy.true_divide(self.sum(), denom)
                two = self.numpy.true_divide((self**2).sum(), denom)
            else:
                denom (self * 0 + weight).sum()
                one = self.numpy.true_divide((self * weight).sum(), denom)
                two = self.numpy.true_divide(((self * weight)**2).sum(), denom)
            if ddof != 0:
                return (two - one**2) * denom / (denom - ddof)
            else:
                return two - one**2

    def std(self, weight=None, ddof=0):
        with self.numpy.errstate(invalid="ignore"):
            return self.numpy.sqrt(self.var(weight=weight, ddof=ddof))

    def __getattr__(self, where):
        if where in dir(super(AwkwardArray, self)):
            return super(AwkwardArray, self).__getattribute__(where)
        else:
            if where in self.columns:
                try:
                    return self[where]
                except Exception as err:
                    raise AttributeError("while trying to get column {0}, an exception occurred:\n{1}: {2}".format(repr(where), type(err), str(err)))
            else:
                raise AttributeError("no column named {0}".format(repr(where)))

    def __dir__(self):
        return sorted(set(dir(super(AwkwardArray, self)) + [x for x in self.columns if self._dir_pattern.match(x) and not keyword.iskeyword(x)]))
    _dir_pattern = re.compile(r"^[a-zA-Z_]\w*$")

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
        import awkward0.array.chunked
        return awkward0.array.chunked.ChunkedArray

    @property
    def AppendableArray(self):
        import awkward0.array.chunked
        return awkward0.array.chunked.AppendableArray

    @property
    def IndexedArray(self):
        import awkward0.array.indexed
        return awkward0.array.indexed.IndexedArray

    @property
    def SparseArray(self):
        import awkward0.array.indexed
        return awkward0.array.indexed.SparseArray

    @property
    def JaggedArray(self):
        import awkward0.array.jagged
        return awkward0.array.jagged.JaggedArray

    @property
    def MaskedArray(self):
        import awkward0.array.masked
        return awkward0.array.masked.MaskedArray

    @property
    def BitMaskedArray(self):
        import awkward0.array.masked
        return awkward0.array.masked.BitMaskedArray

    @property
    def IndexedMaskedArray(self):
        import awkward0.array.masked
        return awkward0.array.masked.IndexedMaskedArray

    @property
    def Methods(self):
        import awkward0.array.objects
        return awkward0.array.objects.Methods

    @property
    def ObjectArray(self):
        import awkward0.array.objects
        return awkward0.array.objects.ObjectArray

    @property
    def StringArray(self):
        import awkward0.array.objects
        return awkward0.array.objects.StringArray

    @property
    def Table(self):
        import awkward0.array.table
        return awkward0.array.table.Table

    @property
    def UnionArray(self):
        import awkward0.array.union
        return awkward0.array.union.UnionArray

    @property
    def VirtualArray(self):
        import awkward0.array.virtual
        return awkward0.array.virtual.VirtualArray

    @classmethod
    def _util_isinteger(cls, x):
        return isinstance(x, (numbers.Integral, cls.numpy.integer)) and not isinstance(x, (bool, cls.numpy.bool_, cls.numpy.bool))

    @classmethod
    def _util_isintegertype(cls, x):
        return issubclass(x, cls.numpy.integer) and not issubclass(x, (cls.numpy.bool_, cls.numpy.bool))

    @classmethod
    def _util_toarray(cls, value, defaultdtype, passthrough=None):
        import awkward0.array.base
        if passthrough is None:
            passthrough = (cls.numpy.ndarray, AwkwardArray)
        if isinstance(value, passthrough):
            return value
        else:
            try:
                return cls.numpy.frombuffer(value, dtype=getattr(value, "dtype", defaultdtype)).reshape(getattr(value, "shape", -1))
            except (AttributeError, TypeError):
                if len(value) == 0:
                    return cls.numpy.array(value, dtype=defaultdtype, copy=False)
                else:
                    return cls.numpy.array(value, copy=False)

    @classmethod
    def _util_arraystr_draw(cls, x):
        if isinstance(x, tuple):
            return "(" + ", ".join(cls._util_arraystr_draw(y) for y in x) + ")"
        elif isinstance(x, Iterable):
            if len(x) > 6:
                return "[" + " ".join(cls._util_arraystr_draw(y) for y in x[:3]) + " ... " + " ".join(cls._util_arraystr_draw(y) for y in x[-3:]) + "]"
            else:
                return "[" + " ".join(cls._util_arraystr_draw(y) for y in x) + "]"
        else:
            return repr(x)

    @classmethod
    def _util_arraystr(cls, array):
        if isinstance(array, cls.numpy.ndarray):
            return cls._util_arraystr_draw(array)
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
    def _util_counts(cls, array):
        if isinstance(array, AwkwardArray):
            return array.counts
        elif len(array.shape) == 1:
            return cls.numpy.full(array.shape[0], -1, dtype=cls.INDEXTYPE)
        else:
            return cls.numpy.full(array.shape[0], array.shape[1], dtype=cls.INDEXTYPE)

    @classmethod
    def _util_boolmask(cls, array, maskedwhen):
        if isinstance(array, AwkwardArray):
            return array.boolmask(maskedwhen=maskedwhen)
        elif isinstance(array, cls.numpy.ma.MaskedArray) and array.mask is not False:
            if maskedwhen:
                return array.mask
            else:
                return ~array.mask
        else:
            if maskedwhen:
                return cls.numpy.zeros(len(array), dtype=cls.MASKTYPE)
            else:
                return cls.numpy.ones(len(array), dtype=cls.MASKTYPE)

    @property
    def ismasked(self):
        return self.boolmask(maskedwhen=True)

    @property
    def isunmasked(self):
        return self.boolmask(maskedwhen=False)

    @classmethod
    def _util_flattentuple(cls, array):
        if isinstance(array, AwkwardArray):
            return array.flattentuple()
        else:
            return array

    @classmethod
    def _util_flatten(cls, array, axis):
        if isinstance(array, AwkwardArray):
            return array.flatten(axis=axis)
        else:
            axis = min(axis, len(array.shape) - 1)
            return array.reshape(array.shape[:axis] + (-1,) + array.shape[axis + 2:])

    @classmethod
    def _util_pad(cls, array, length, maskedwhen, clip, axis):
        if isinstance(array, AwkwardArray):
            return array.pad(length, maskedwhen=maskedwhen, clip=clip, axis=axis)

        elif len(array.shape) == 1:
            raise ValueError("pad cannot be applied to scalars")

        elif length == 0 and clip:
            if isinstance(maskedwhen, cls.numpy.ma.core.MaskedConstant):
                return cls.JaggedArray.fget(None).fromoffsets([0], cls.numpy.ma.array([]))
            else:
                return cls.JaggedArray.fget(None).fromoffsets([0], cls.MaskedArray.fget(None)([], []))

        elif array.shape[1] > length and clip:
            offsets = cls.numpy.arange(0, length*(len(array) + 1), length, dtype=cls.INDEXTYPE)
            content = array[(slice(None), slice(length)) + array.shape[2:]]
            if isinstance(maskedwhen, cls.numpy.ma.core.MaskedConstant):
                return cls.JaggedArray.fget(None).fromoffsets(offsets, cls.numpy.ma.array(content.reshape((-1,) + array.shape[2:])))
            else:
                return cls.JaggedArray.fget(None).fromoffsets(offsets, cls.MaskedArray.fget(None).fromcontent(content.reshape((-1,) + array.shape[2:]), maskedwhen=maskedwhen))

        elif array.shape[1] >= length:
            offsets = cls.numpy.arange(0, array.shape[1]*(len(array) + 1), array.shape[1], dtype=cls.INDEXTYPE)
            if isinstance(maskedwhen, cls.numpy.ma.core.MaskedConstant):
                return cls.JaggedArray.fget(None).fromoffsets(offsets, cls.numpy.ma.array(array.reshape((-1,) + array.shape[2:])))
            else:
                return cls.JaggedArray.fget(None).fromoffsets(offsets, cls.MaskedArray.fget(None).fromcontent(array.reshape((-1,) + array.shape[2:]), maskedwhen=maskedwhen))

        else:
            offsets = cls.numpy.arange(0, length*(len(array) + 1), length, dtype=cls.INDEXTYPE)
            content = cls.numpy.empty(array.shape[:1] + (length,) + array.shape[2:], dtype=array.dtype)
            content[:, :array.shape[1]] = array
            if isinstance(maskedwhen, cls.numpy.ma.core.MaskedConstant):
                mask = cls.numpy.ones(len(array), dtype=cls.MASKTYPE)
                return cls.JaggedArray.fget(None).fromoffsets(offsets, cls.numpy.ma.array(content.reshape((-1,) + array.shape[2:]), mask=mask))
            if maskedwhen:
                mask = cls.numpy.ones((len(array), length), dtype=cls.MASKTYPE)
            else:
                mask = cls.numpy.zeros((len(array), length), dtype=cls.MASKTYPE)
            mask[:, :array.shape[1]] = not maskedwhen
            return cls.JaggedArray.fget(None).fromoffsets(offsets, cls.MaskedArray.fget(None)(mask.reshape((-1,) + array.shape[2:]), content.reshape((-1,) + array.shape[2:]), maskedwhen=maskedwhen))

    @classmethod
    def _util_regular(cls, array):
        if isinstance(array, AwkwardArray):
            return array.regular()
        else:
            return array

    @classmethod
    def _util_reduce(cls, array, ufunc, identity, dtype):
        if isinstance(array, AwkwardArray):
            return array._reduce(ufunc, identity, dtype)

        elif len(array) == 0:
            if dtype is None:
                dtype = array.dtype
            return ufunc.reduce(cls.numpy.full((1,) + array.shape[1:], identity, dtype=dtype), axis=-1)

        else:
            original = array
            if dtype is not None:
                array = cls.numpy.array(array, dtype=dtype, copy=False)
            if issubclass(array.dtype.type, (cls.numpy.floating, cls.numpy.complexfloating)):
                mask = cls.numpy.isnan(array)
                if mask.any():
                    if array is original or not array.flags.owndata:
                        array = array.copy()
                    array[mask] = identity
            return ufunc.reduce(array, axis=None)

    @classmethod
    def _util_concatenate(cls, arrays):
        if all(isinstance(x, cls.numpy.ndarray) for x in arrays):
            return cls.numpy.concatenate(arrays)
        else:
            return arrays[0].concatenate(arrays[1:])

    @bothmethod
    def concatenate(isclassmethod, cls_or_self, arrays, axis=0):
        if len(arrays) < 1:
            raise ValueError("at least one array needed to concatenate")

        if isclassmethod:
            cls = cls_or_self
        else:
            self = cls_or_self
            cls = type(self)
            arrays = (self,) + tuple(arrays)

        def resolve(t):
            for b in t.__bases__:
                if issubclass(t, AwkwardArray):
                    return resolve(b)
            else:
                return t

        if all(type(x) == cls.numpy.ndarray for x in arrays):
            return cls.numpy.concatenate(arrays, axis=axis)

        if not all(resolve(type(x)) == resolve(type(arrays[0])) for x in arrays):
            if axis == 0:
                tags = cls.numpy.concatenate([cls.numpy.full(len(x), i, dtype=cls.TAGTYPE) for i, x in enumerate(arrays)])
                return cls.UnionArray.fget(None).fromtags(tags, arrays)
            else:
                raise NotImplementedError("axis > 0 for different types")

        for x in arrays:
            x.valid()

        if axis == 0:
            return type(arrays[0])._concatenate_axis0(arrays)
        elif axis == 1:
            return type(arrays[0])._concatenate_axis1(arrays)
        else:
            raise NotImplementedError("axis > 1")

    @classmethod
    def _concatenate_axis0(cls, arrays):
        raise NotImplementedError("{0}.concatenate with axis=0 not implemented".format(cls.__name__))

    @classmethod
    def _concatenate_axis1(cls, arrays):
        raise NotImplementedError("{0}.concatenate with axis=1 not implemented".format(cls.__name__))

    @classmethod
    def _util_isstringslice(cls, where):
        if isinstance(where, awkward0.util.string):
            return True
        elif isinstance(where, bytes):
            raise TypeError("column selection must be str, not bytes, in Python 3")
        elif isinstance(where, tuple):
            return False
        elif isinstance(where, (cls.numpy.ndarray, AwkwardArray)) and issubclass(where.dtype.type, (numpy.str, numpy.str_)):
            return True
        elif isinstance(where, (cls.numpy.ndarray, AwkwardArray)) and issubclass(where.dtype.type, (numpy.object, numpy.object_)) and not issubclass(where.dtype.type, (numpy.bool, numpy.bool_)):
            return len(where) > 0 and all(isinstance(x, awkward0.util.string) for x in where)
        elif isinstance(where, (cls.numpy.ndarray, AwkwardArray)):
            return False
        try:
            assert len(where) > 0 and all(isinstance(x, awkward0.util.string) for x in where)
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

    @classmethod
    def _util_columns_descend(cls, array, seen):
        if isinstance(array, cls.numpy.ndarray):
            if array.dtype.fields is None:
                return []
            else:
                return list(array.dtype.fields)
        else:
            return array._util_columns(seen)

    @property
    def columns(self):
        return self._util_columns(set())

    @classmethod
    def _util_rowname_descend(cls, array, seen):
        if isinstance(array, cls.numpy.ndarray):
            if array.dtype.fields is None:
                raise TypeError("not a Table, so there is no rowname")
            else:
                return None
        else:
            return array._util_rowname(seen)

    @property
    def rowname(self):
        return self._util_rowname(set())

    @property
    def istuple(self):
        columns = self.columns
        return self.rowname == "tuple" and columns == [str(x) for x in range(len(columns))]

    def unzip(self):
        return tuple(self[column_name] for column_name in self._util_columns(set()))

class AwkwardArrayWithContent(AwkwardArray):
    """
    AwkwardArrayWithContent: abstract base class
    """

    def __setitem__(self, where, what):
        if isinstance(where, awkward0.util.string):
            self._content[where] = what

        elif self._util_isstringslice(where):
            what = what.unzip()
            if len(where) != len(what):
                raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
            for x, y in zip(where, what):
                self._content[x] = y

        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __delitem__(self, where):
        if isinstance(where, awkward0.util.string):
            del self._content[where]
        elif self._util_isstringslice(where):
            for x in where:
                del self._content[x]
        else:
            raise TypeError("invalid index for removing column from Table: {0}".format(where))

    def _hasjagged(self):
        return self._util_hasjagged(self._content)

    def _util_columns(self, seen):
        if id(self) in seen:
            return []
        seen.add(id(self))
        return self._util_columns_descend(self._content, seen)

    def _util_rowname(self, seen):
        if id(self) in seen:
            raise TypeError("not a Table, so there is no rowname")
        seen.add(id(self))
        return self._util_rowname_descend(self._content, seen)

    def astype(self, dtype):
        return self.copy(content=self._content.astype(dtype))

    def fillna(self, value):
        return self.copy(content=self._util_fillna(self._content, value))
