#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import itertools
import importlib
import sys
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
    import awkward.array.base
    return awkward.array.base.AwkwardArray._util_toarray(value, defaultdtype, passthrough=passthrough)

def awkwardlib(awkwardlib):
    if awkwardlib is None:
        import awkward
        return awkward
    elif isinstance(awkwardlib, str):
        return importlib.import_module(awkwardlib)
    else:
        return awkwardlib

class bothmethod(object):
    def __init__(self, fcn):
        self.fcn = fcn
    def __get__(self, ins, typ):
        if ins is None:
            return lambda *args, **kwargs: self.fcn(True, typ, *args, **kwargs)
        else:
            return lambda *args, **kwargs: self.fcn(False, ins, *args, **kwargs)

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
    jaggedtype = [cls.awkward.JaggedArray] * len(arrays)
    starts, stops = None, None
    for i in range(len(arrays)):
        if starts is None and isinstance(arrays[i], cls.awkward.JaggedArray):
            starts, stops = arrays[i].starts, arrays[i].stops

        if isinstance(arrays[i], cls.awkward.JaggedArray):
            jaggedtype[i] = type(arrays[i])

        if not isinstance(arrays[i], Iterable):
            arrays[i] = cls.awkward.numpy.full(length, arrays[i])

        arrays[i] = cls.awkward.util.toarray(arrays[i], cls.awkward.numpy.float64)

    if starts is None:
        return arrays

    for i in range(len(arrays)):
        if not isinstance(arrays[i], cls.awkward.JaggedArray) or not (cls.awkward.numpy.array_equal(starts, arrays[i].starts) and cls.awkward.numpy.array_equal(stops, arrays[i].stops)):
            content = cls.awkward.numpy.zeros(stops.max(), dtype=cls.awkward.numpy.float64)
            arrays[i] = jaggedtype[i](starts, stops, content) + arrays[i]    # invoke jagged broadcasting to align arrays

    return arrays

def unwrap_jagged(cls, awkcls, arrays):
    if not isinstance(arrays[0], cls.awkward.JaggedArray):
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
