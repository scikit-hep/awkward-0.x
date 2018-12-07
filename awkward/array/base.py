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

import awkward.persist
import awkward.type
import awkward.util

class At(object):
    def __init__(self, array):
        self._array = array

    def __repr__(self):
        return "<at accessor for {0}".format(repr(self._array)[1:])

    def __getattr__(self, where):
        return self._array[where]

    def __call__(self, where):
        return self._array[str(where)]

class AwkwardArray(awkward.util.NDArrayOperatorsMixin):
    """
    AwkwardArray: abstract base class
    """

    @property
    def at(self):
        return At(self)

    allow_tonumpy = True
    allow_iter = True
    # TODO for 1.0: add check_prop_valid and check_whole_valid parameters

    DEFAULTTYPE = awkward.util.DEFAULTTYPE
    CHARTYPE    = awkward.util.CHARTYPE
    INDEXTYPE   = awkward.util.INDEXTYPE
    TAGTYPE     = awkward.util.TAGTYPE
    MASKTYPE    = awkward.util.MASKTYPE
    BITMASKTYPE = awkward.util.BITMASKTYPE
    BOOLTYPE    = awkward.util.BOOLTYPE

    def _checktonumpy(self):
        if not self.allow_tonumpy:
            raise RuntimeError("awkward.array.base.AwkwardArray.allow_tonumpy is False; refusing to convert to Numpy")

    def __array__(self, dtype=None):
        self._checktonumpy()

        if dtype is None:
            dtype = self.dtype
        
        if dtype == awkward.util.numpy.dtype(object):
            return awkward.util.numpy.array(list(self), dtype=dtype)
        else:
            return awkward.util.numpy.fromiter(self, dtype=dtype, count=len(self))

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
            return "[{0}]".format(" ".join(awkward.util.array_str(x) for x in self.__iter__(checkiter=False)))
        else:
            return "[{0} ... {1}]".format(" ".join(awkward.util.array_str(x) for x in self[:3].__iter__(checkiter=False)), " ".join(awkward.util.array_str(x) for x in self[-3:].__iter__(checkiter=False)))

    def __repr__(self):
        return "<{0} {1} at {2:012x}>".format(self.__class__.__name__, str(self), id(self))

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

    def tolist(self):
        import awkward.array.table
        out = []
        for x in self:
            if isinstance(x, awkward.array.table.Table.Row):
                out.append(dict((n, self._try_tolist(x[n])) for n in x._table._content))
            elif isinstance(x, awkward.util.numpy.ma.core.MaskedConstant):
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

    def _argfields(self, function):
        if not isinstance(function, types.FunctionType):
            raise TypeError("function (or lambda) required")

        if (isinstance(function, types.FunctionType) and function.__code__.co_argcount == 1) or isinstance(self._content, awkward.util.numpy.ndarray):
            return None, None

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

    def apply(self, function):
        args, kwargs = self._argfields(function)
        if args is None and kwargs is None:
            return function(self)
        else:
            args = tuple(self[n] for n in args)
            kwargs = dict((n, self[n]) for n in kwargs)
            return function(*args, **kwargs)

    def filter(self, function):
        args, kwargs = self._argfields(function)
        if args is None and kwargs is None:
            return self[function(self)]
        else:
            args = tuple(self[n] for n in args)
            kwargs = dict((n, self[n]) for n in kwargs)
            return self[function(*args, **kwargs)]

    def maxby(self, function):
        args, kwargs = self._argfields(function)
        if args is None and kwargs is None:
            return self[function(self).argmax()]
        else:
            args = tuple(self[n] for n in args)
            kwargs = dict((n, self[n]) for n in kwargs)
            return self[function(*args, **kwargs).argmax()]

    def minby(self, function):
        args, kwargs = self._argfields(function)
        if args is None and kwargs is None:
            return self[function(self).argmin()]
        else:
            args = tuple(self[n] for n in args)
            kwargs = dict((n, self[n]) for n in kwargs)
            return self[function(*args, **kwargs).argmin()]

class AwkwardArrayWithContent(AwkwardArray):
    """
    AwkwardArrayWithContent: abstract base class
    """

    def __setitem__(self, where, what):
        if isinstance(where, awkward.util.string):
            self._content[where] = what

        elif awkward.util.isstringslice(where):
            if len(where) != len(what):
                raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
            for x, y in zip(where, what):
                self._content[x] = y

        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __delitem__(self, where):
        if isinstance(where, awkward.util.string):
            del self._content[where]
        elif awkward.util.isstringslice(where):
            for x in where:
                del self._content[x]
        else:
            raise TypeError("invalid index for removing column from Table: {0}".format(where))

    @property
    def base(self):
        if isinstance(self._content, awkward.util.numpy.ndarray):
            raise TypeError("array has no Table, and hence no base")
        return self._content.base

    @property
    def columns(self):
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return []
        else:
            return self._content.columns

    @property
    def allcolumns(self):
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return []
        else:
            return self._content.allcolumns

    def astype(self, dtype):
        return self.copy(content=self._content.astype(dtype))
