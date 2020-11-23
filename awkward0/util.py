#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-0.x/blob/master/LICENSE

import itertools
import importlib
import sys
import os
from collections import OrderedDict
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from functools import wraps

import numpy

if sys.version_info[0] <= 2:
    izip = itertools.izip
    string = basestring
    unicode = unicode
else:
    izip = zip
    string = str
    unicode = str

frombuffer = numpy.frombuffer

def toarray(value, defaultdtype, passthrough=None):
    import awkward0.array.base
    return awkward0.array.base.AwkwardArray._util_toarray(value, defaultdtype, passthrough=passthrough)

def awkwardlib(awkwardlib):
    import awkward0
    return awkward0

class bothmethod(object):
    def __init__(self, fcn):
        self.fcn = fcn
    def __get__(self, ins, typ):
        if ins is None:
            return lambda *args, **kwargs: self.fcn(True, typ, *args, **kwargs)
        else:
            return lambda *args, **kwargs: self.fcn(False, ins, *args, **kwargs)

# numpy on windows has some strange behavior with dtypes of certain functions
# requiring us to cast down to int32 for (at least): ufunc.reduceat, repeat
def windows_safe(array):
    if os.name == "nt":
        return array.astype(numpy.int32)
    return array

################################################################ wrappers (used to be in uproot-methods)

def _normalize_arrays(cls, arrays):
    length = None
    for i in range(len(arrays)):
        if isinstance(arrays[i], Iterable):
            if length is None:
                length = len(arrays[i])
                break
    if length is None:
        raise TypeError("cannot construct an array if all arguments are scalar")

    arrays = list(arrays)
    jaggedtype = [cls.awkward0.JaggedArray] * len(arrays)
    starts, stops = None, None
    for i in range(len(arrays)):
        if starts is None and isinstance(arrays[i], cls.awkward0.JaggedArray):
            starts, stops = arrays[i].starts, arrays[i].stops

        if isinstance(arrays[i], cls.awkward0.JaggedArray):
            jaggedtype[i] = type(arrays[i])

        if not isinstance(arrays[i], Iterable):
            arrays[i] = cls.awkward0.numpy.full(length, arrays[i])

        arrays[i] = cls.awkward0.util.toarray(arrays[i], cls.awkward0.numpy.float64)

    if starts is None:
        return arrays

    for i in range(len(arrays)):
        if not isinstance(arrays[i], cls.awkward0.JaggedArray) or not (cls.awkward0.numpy.array_equal(starts, arrays[i].starts) and cls.awkward0.numpy.array_equal(stops, arrays[i].stops)):
            content = cls.awkward0.numpy.zeros(stops.max(), dtype=cls.awkward0.numpy.float64)
            arrays[i] = jaggedtype[i](starts, stops, content) + arrays[i]    # invoke jagged broadcasting to align arrays

    return arrays

def unwrap_jagged(cls, awkcls, arrays):
    if not isinstance(arrays[0], cls.awkward0.JaggedArray):
        return lambda x: x, arrays

    counts = arrays[0].counts.reshape(-1)
    offsets = awkcls.counts2offsets(counts)
    starts, stops = offsets[:-1], offsets[1:]
    starts = starts.reshape(arrays[0].starts.shape[:-1] + (-1,))
    stops = stops.reshape(arrays[0].stops.shape[:-1] + (-1,))
    wrap, arrays = unwrap_jagged(cls, awkcls, [x.flatten() for x in arrays])
    return lambda x: awkcls(starts, stops, wrap(x)), arrays

def wrapjaggedmethod(awkcls):
    def wrapjagged_decorator(func):
        @wraps(func)
        def func_wrapper(cls, *arrays):
            wrap, arrays = unwrap_jagged(cls, awkcls, _normalize_arrays(cls, arrays))
            return wrap(func(cls, *arrays))
        return func_wrapper
    return wrapjagged_decorator

################################################################ array helpers

try:
    NDArrayOperatorsMixin = numpy.lib.mixins.NDArrayOperatorsMixin

except AttributeError:
    from numpy.core import umath as um

    def _disables_array_ufunc(obj):
        """True when __array_ufunc__ is set to None."""
        try:
            return obj.__array_ufunc__ is None
        except AttributeError:
            return False

    def _binary_method(ufunc, name):
        """Implement a forward binary method with a ufunc, e.g., __add__."""
        def func(self, other):
            if _disables_array_ufunc(other):
                return NotImplemented
            return ufunc(self, other)
        func.__name__ = '__{}__'.format(name)
        return func

    def _reflected_binary_method(ufunc, name):
        """Implement a reflected binary method with a ufunc, e.g., __radd__."""
        def func(self, other):
            if _disables_array_ufunc(other):
                return NotImplemented
            return ufunc(other, self)
        func.__name__ = '__r{}__'.format(name)
        return func

    def _inplace_binary_method(ufunc, name):
        """Implement an in-place binary method with a ufunc, e.g., __iadd__."""
        def func(self, other):
            return ufunc(self, other, out=(self,))
        func.__name__ = '__i{}__'.format(name)
        return func

    def _numeric_methods(ufunc, name):
        """Implement forward, reflected and inplace binary methods with a ufunc."""
        return (_binary_method(ufunc, name),
                _reflected_binary_method(ufunc, name),
                _inplace_binary_method(ufunc, name))

    def _unary_method(ufunc, name):
        """Implement a unary special method with a ufunc."""
        def func(self):
            return ufunc(self)
        func.__name__ = '__{}__'.format(name)
        return func

    class NDArrayOperatorsMixin(object):
        __lt__ = _binary_method(um.less, 'lt')
        __le__ = _binary_method(um.less_equal, 'le')
        __eq__ = _binary_method(um.equal, 'eq')
        __ne__ = _binary_method(um.not_equal, 'ne')
        __gt__ = _binary_method(um.greater, 'gt')
        __ge__ = _binary_method(um.greater_equal, 'ge')

        # numeric methods
        __add__, __radd__, __iadd__ = _numeric_methods(um.add, 'add')
        __sub__, __rsub__, __isub__ = _numeric_methods(um.subtract, 'sub')
        __mul__, __rmul__, __imul__ = _numeric_methods(um.multiply, 'mul')
        if sys.version_info.major < 3:
            # Python 3 uses only __truediv__ and __floordiv__
            __div__, __rdiv__, __idiv__ = _numeric_methods(um.divide, 'div')
        __truediv__, __rtruediv__, __itruediv__ = _numeric_methods(
            um.true_divide, 'truediv')
        __floordiv__, __rfloordiv__, __ifloordiv__ = _numeric_methods(
            um.floor_divide, 'floordiv')
        __mod__, __rmod__, __imod__ = _numeric_methods(um.remainder, 'mod')
        if hasattr(um, "divmod"):
            __divmod__ = _binary_method(um.divmod, 'divmod')
            __rdivmod__ = _reflected_binary_method(um.divmod, 'divmod')
        # __idivmod__ does not exist
        # TODO: handle the optional third argument for __pow__?
        __pow__, __rpow__, __ipow__ = _numeric_methods(um.power, 'pow')
        __lshift__, __rlshift__, __ilshift__ = _numeric_methods(
            um.left_shift, 'lshift')
        __rshift__, __rrshift__, __irshift__ = _numeric_methods(
            um.right_shift, 'rshift')
        __and__, __rand__, __iand__ = _numeric_methods(um.bitwise_and, 'and')
        __xor__, __rxor__, __ixor__ = _numeric_methods(um.bitwise_xor, 'xor')
        __or__, __ror__, __ior__ = _numeric_methods(um.bitwise_or, 'or')

        # unary methods
        __neg__ = _unary_method(um.negative, 'neg')
        if hasattr(um, "positive"):
            __pos__ = _unary_method(um.positive, 'pos')
            __abs__ = _unary_method(um.absolute, 'abs')
            __invert__ = _unary_method(um.invert, 'invert')

################################################################ conversion of arrays to Pandas

def topandas(array, flatten=False):
    import pandas
    import awkward0.array.base

    if isinstance(array, awkward0.array.base.AwkwardArray):
        if flatten:
            return topandas_flatten(array)
        else:
            out = array._topandas({})
            if len(out.columns) == 0:
                return pandas.Series(out)
            else:
                return pandas.DataFrame({n: out[n] for n in out.columns}, columns=out.columns)
    else:
        out = numpy.array(array, copy=False)
        if out.dtype.fields is None:
            return pandas.Series(out)
        else:
            return pandas.DataFrame(out)

def topandas_flatten(array):
    import numpy
    import pandas

    import awkward0.array.base
    import awkward0.array.chunked
    import awkward0.array.jagged
    import awkward0.array.objects
    import awkward0.array.table
    import awkward0.array.virtual
    import awkward0.type

    if isinstance(array, awkward0.array.base.AwkwardArray):
        numpy = array.numpy
        JaggedArray = array.JaggedArray
        Table = array.Table
    else:
        JaggedArray = awkward0.array.jagged.JaggedArray
        Table = awkward0.array.table.Table

    def unwrap(a):
        if isinstance(a, awkward0.array.chunked.ChunkedArray):
            chunks = [unwrap(x) for x in a.chunks]
            if any(isinstance(x, awkward0.array.jagged.JaggedArray) for x in chunks):
                return awkward0.array.jagged.JaggedArray.concatenate(chunks)
            else:
                return numpy.concatenate([x.regular() for x in chunks])
        elif isinstance(a, awkward0.array.virtual.VirtualArray):
            return a.array
        else:
            return a

    globalindex = [None]
    localindex = []
    columns = []
    def recurse(array, tpe, cols, seriously):
        if isinstance(tpe, awkward0.type.TableType):
            starts, stops = None, None
            out, deferred, unflattened = None, {}, None

            for n in tpe.columns:
                if not isinstance(n, str):
                    raise ValueError("column names must be strings")

                arrayn = unwrap(array[n])
                tpen = tpe[n]
                colsn = cols + (n,) if seriously else cols

                if isinstance(arrayn, awkward0.array.objects.ObjectArray) and not isinstance(arrayn, awkward0.array.objects.StringArray):
                    arrayn = arrayn.content
                if not isinstance(tpen, (numpy.dtype, str, bytes, awkward0.type.Type)):
                    tpen = awkward0.type.fromarray(arrayn).to
                if isinstance(tpen, numpy.dtype):
                    columns.append(colsn)
                    tmp = arrayn

                elif isinstance(tpen, type) and issubclass(tpen, (str, bytes)):
                    columns.append(colsn)
                    tmp = arrayn

                elif isinstance(tpen, awkward0.type.ArrayType) and tpen.takes == numpy.inf:
                    tmp = JaggedArray(arrayn.starts, arrayn.stops, recurse(arrayn.content, tpen.to, colsn, True))

                elif isinstance(tpen, awkward0.type.TableType):
                    tmp = recurse(arrayn, tpen, colsn, True)

                elif isinstance(tpen, awkward0.type.OptionType) and isinstance(arrayn.content, numpy.ndarray):
                    columns.append(colsn)
                    tmp = numpy.ma.MaskedArray(arrayn.content, arrayn.boolmask(maskedwhen=True))

                else:
                    raise ValueError("this array has unflattenable substructure:\n\n{0}".format(str(tpen)))

                if isinstance(tmp, awkward0.array.jagged.JaggedArray):
                    if isinstance(tmp.content, awkward0.array.jagged.JaggedArray):
                        unflattened = tmp
                        tmp = tmp.flatten(axis=1)

                    if starts is None:
                        starts, stops = tmp.starts, tmp.stops
                    elif not numpy.array_equal(starts, tmp.starts) or not numpy.array_equal(stops, tmp.stops):
                        raise ValueError("this array has more than one jagged array structure")
                    if out is None:
                        out = JaggedArray(starts, stops, Table({n: tmp.content}))
                    else:
                        out[n] = tmp

                else:
                    deferred[n] = tmp

            if out is None:
                out = Table()

            for n, x in deferred.items():
                out[n] = x

            m = ""
            while m in tpe.columns:
                m = m + " "
            out[m] = numpy.arange(len(out))
            globalindex[0] = out[m].flatten()

            for n in tpe.columns:
                arrayn = unwrap(array[n])
                if isinstance(arrayn, awkward0.array.jagged.JaggedArray):
                    if unflattened is None:
                        localindex.insert(0, out[n].localindex.flatten())
                    else:
                        oldloc = unflattened.content.localindex
                        tab = JaggedArray(oldloc.starts, oldloc.stops, Table({"oldloc": oldloc.content}))
                        tab["newloc"] = arrayn.localindex.flatten()
                        localindex.insert(0, tab["newloc"].flatten())
                    break

            return out[tpe.columns]

        else:
            return recurse(Table({"": array}), awkward0.type.TableType(**{"": tpe}), cols, False)[""]

    tmp = recurse(array, awkward0.type.fromarray(array).to, (), True)
    if isinstance(tmp, awkward0.array.jagged.JaggedArray):
        tmp = tmp.flatten()

    deepest = max(len(x) for x in columns)

    out = {}
    for i, col in enumerate(columns):
        x = tmp
        for c in col:
            x = x[c]
        columns[i] = col + ("",) * (deepest - len(col))
        out[columns[i]] = x

    index = globalindex + localindex
    if len(index) == 1:
        index = pandas.Index(index[0])
    else:
        index = pandas.MultiIndex.from_arrays(index)

    if len(columns) == 1 and deepest == 0:
        return pandas.Series(out[()], index=index)
    else:
        return pandas.DataFrame(data=out, index=index, columns=pandas.MultiIndex.from_tuples(columns))
