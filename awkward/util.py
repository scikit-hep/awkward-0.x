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

import ast
import itertools
import re
import sys
import types
from collections import OrderedDict
    
if sys.version_info[0] <= 2:
    izip = itertools.izip
    string = basestring
    unicode = unicode
else:
    izip = zip
    string = str
    unicode = str

def isidentifier(x):
    if not isinstance(x, string):
        return False

    if sys.version_info[0] <= 2:
        try:
            node = ast.parse(x)
        except SyntaxError:
            return False
        else:
            return isinstance(node, ast.Module) and len(node.body) == 1 and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Name) and node.body[0].value.id == x

    else:
        return x.isidentifier()

def isintstring(x):
    return isinstance(x, string) and isintstring._pattern.match(x) is not None
isintstring._pattern = re.compile("^(0|[1-9]+[0-9]*)$")

def is_intstring(x):
    return isinstance(x, string) and is_intstring._pattern.match(x) is not None
is_intstring._pattern = re.compile("^_(0|[1-9]+[0-9]*)$")

################################################################ array helpers

import numpy   # all access to Numpy passes through here

CHARTYPE = numpy.dtype(numpy.uint8)
INDEXTYPE = numpy.dtype(numpy.int64)
MASKTYPE = numpy.dtype(numpy.bool_)
BITMASKTYPE = numpy.dtype(numpy.uint8)

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

def iscomparison(ufunc):
    return (ufunc is numpy.less or
            ufunc is numpy.less_equal or
            ufunc is numpy.equal or
            ufunc is numpy.not_equal or
            ufunc is numpy.greater or
            ufunc is numpy.greater_equal)

def _argfields(function):
    if not isinstance(function, types.FunctionType):
        raise TypeError("apply method requires a function (or lambda)")

    if function.__code__.co_argcount != 1:
        raise TypeError("apply method requires a one-argument function (or lambda) when applied to non-Tables")

    return None, None

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
