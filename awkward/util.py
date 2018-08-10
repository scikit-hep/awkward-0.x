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

import itertools
import sys
try:
    import collections
    OrderedDict = collections.OrderedDict
except ImportError:
    # simple OrderedDict implementation for Python 2.6
    class OrderedDict(dict):
        def __init__(self, items=(), **kwds):
            items = list(items)
            self._order = [k for k, v in items] + [k for k, v in kwds.items()]
            super(OrderedDict, self).__init__(items)
        def keys(self):
            return self._order
        def values(self):
            return [self[k] for k in self._order]
        def items(self):
            return [(k, self[k]) for k in self._order]
        def __setitem__(self, name, value):
            if name not in self._order:
                self._order.append(name)
            super(OrderedDict, self).__setitem__(name, value)
        def __delitem__(self, name):
            if name in self._order:
                self._order.remove(name)
            super(OrderedDict, self).__delitem__(name)
        def __repr__(self):
            return "OrderedDict([{0}])".format(", ".join("({0}, {1})".format(repr(k), repr(v)) for k, v in self.items()))

if sys.version_info[0] <= 2:
    izip = itertools.izip
    string = basestring
else:
    izip = zip
    string = str

def isstringslice(where):
    if isinstance(where, string):
        return True
    elif isinstance(where, tuple):
        return False
    try:
        assert all(isinstance(x, string) for x in where)
    except (TypeError, AssertionError):
        return False
    else:
        return True

################################################################ array operations

import numpy

CHARTYPE = numpy.dtype(numpy.uint8)
INDEXTYPE = numpy.dtype(numpy.int64)
MASKTYPE = numpy.dtype(numpy.bool_)
BITMASKTYPE = numpy.dtype(numpy.uint8)

cumsum = numpy.cumsum
cumprod = numpy.cumprod
nonzero = numpy.nonzero
arange = numpy.arange

def toarray(value, defaultdtype, passthrough):
    if isinstance(value, passthrough):
        return value
    else:
        try:
            return numpy.frombuffer(value, dtype=getattr(value, "dtype", defaultdtype)).reshape(getattr(value, "shape", -1))
        except AttributeError:
            return numpy.array(value, copy=False)

def deepcopy(array):
    if array is None:
        return None
    elif isinstance(array, numpy.ndarray):
        return array.copy()
    else:
        return array.deepcopy()

def offsetsaliased(starts, stops):
    return (isinstance(starts, numpy.ndarray) and isinstance(stops, numpy.ndarray) and
            starts.base is not None and stops.base is not None and starts.base is stops.base and
            starts.ctypes.data == starts.base.ctypes.data and
            stops.ctypes.data == stops.base.ctypes.data + stops.dtype.itemsize and
            len(starts) == len(starts.base) - 1 and
            len(stops) == len(stops.base) - 1)

def counts2offsets(counts):
    offsets = numpy.empty(len(counts) + 1, dtype=INDEXTYPE)
    offsets[0] = 0
    cumsum(counts, out=offsets[1:])
    return offsets

def offsets2parents(offsets):
    out = numpy.zeros(offsets[-1], dtype=INDEXTYPE)
    numpy.add.at(out, offsets[offsets != offsets[-1]][1:], 1)
    cumsum(out, out=out)
    return out

def startsstops2parents(starts, stops):
    out = numpy.full(stops.max(), -1, dtype=INDEXTYPE)
    lenstarts = len(starts)
    i = 0
    while i < lenstarts:
        out[starts[i]:stops[i]] = i
        i += 1
    return out

def parents2startsstops(parents):
    # assumes that children are contiguous, but not necessarily in order or fully covering (allows empty lists)
    tmp = nonzero(parents[1:] != parents[:-1])[0] + 1
    changes = numpy.empty(len(tmp) + 2, dtype=INDEXTYPE)
    changes[0] = 0
    changes[-1] = len(parents)
    changes[1:-1] = tmp

    length = parents.max() + 1
    starts = numpy.zeros(length, dtype=INDEXTYPE)
    counts = numpy.zeros(length, dtype=INDEXTYPE)

    where = parents[changes[:-1]]
    real = (where >= 0)

    starts[where[real]] = (changes[:-1])[real]
    counts[where[real]] = (changes[1:] - changes[:-1])[real]

    return starts, starts + counts

def uniques2offsetsparents(uniques):
    # assumes that children are contiguous, in order, and fully covering (can't have empty lists)
    # values are ignored, apart from uniqueness
    changes = nonzero(uniques[1:] != uniques[:-1])[0] + 1

    offsets = numpy.empty(len(changes) + 2, dtype=INDEXTYPE)
    offsets[0] = 0
    offsets[-1] = len(uniques)
    offsets[1:-1] = changes

    parents = numpy.zeros(len(uniques), dtype=INDEXTYPE)
    parents[changes] = 1
    cumsum(parents, out=parents)

    return offsets, parents

################################################################ ufunc-to-Python operations

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
