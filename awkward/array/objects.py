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

import codecs
import importlib

import awkward.array.base
import awkward.persist
import awkward.type
import awkward.util

class Methods(object):
    """
    Methods: abstract mix-in
    """

    @staticmethod
    def mixin(methods, awkwardtype):
        assert not issubclass(methods, awkward.array.base.AwkwardArray)
        assert not issubclass(awkwardtype, Methods)
        return type(awkwardtype.__name__ + "Methods", (methods, awkwardtype), {})

    @staticmethod
    def maybemixin(sample, awkwardtype):
        if issubclass(sample, Methods):
            assert issubclass(sample, awkward.array.base.AwkwardArray)
            allbases = tuple(x for x in sample.__bases__ if not issubclass(x, awkward.array.base.AwkwardArray)) + (awkwardtype,)
            return type(awkwardtype.__name__ + "Methods", allbases, {})
        else:
            return awkwardtype

class ObjectArray(awkward.array.base.AwkwardArrayWithContent):
    """
    ObjectArray
    """

    def __init__(self, content, generator, args=(), kwargs={}):
        self.content = content
        self.generator = generator
        self.args = args
        self.kwargs = kwargs

    def copy(self, content=None, generator=None, args=None, kwargs=None):
        out = self.__class__.__new__(self.__class__)
        out._content = self._content
        out._generator = self._generator
        out._args = self._args
        out._kwargs = self._kwargs
        if content is not None:
            out.content = content
        if generator is not None:
            out.generator = generator
        if args is not None:
            out.args = args
        if kwargs is not None:
            out.kwargs = kwargs
        return out

    def deepcopy(self, content=None, generator=None, args=None, kwargs=None):
        out = self.copy(content=content, generator=generator, args=args, kwargs=kwargs)
        out._content = awkward.util.deepcopy(out._content)
        return out

    def empty_like(self, **overrides):
        mine = {}
        mine["generator"] = overrides.pop("generator", self._generator)
        mine["args"] = overrides.pop("args", self._args)
        mine["kwargs"] = overrides.pop("kwargs", self._kwargs)
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.empty_like(self._content), **mine)
        else:
            return self.copy(content=self._content.empty_like(**overrides), **mine)

    def zeros_like(self, **overrides):
        mine = {}
        mine["generator"] = overrides.pop("generator", self._generator)
        mine["args"] = overrides.pop("args", self._args)
        mine["kwargs"] = overrides.pop("kwargs", self._kwargs)
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.zeros_like(self._content), **mine)
        else:
            return self.copy(content=self._content.zeros_like(**overrides), **mine)

    def ones_like(self, **overrides):
        mine = {}
        mine["generator"] = overrides.pop("generator", self._generator)
        mine["args"] = overrides.pop("args", self._args)
        mine["kwargs"] = overrides.pop("kwargs", self._kwargs)
        if isinstance(self._content, awkward.util.numpy.ndarray):
            return self.copy(content=awkward.util.numpy.ones_like(self._content), **mine)
        else:
            return self.copy(content=self._content.ones_like(**overrides), **mine)

    def __awkward_persist__(self, ident, fill, prefix, suffix, schemasuffix, storage, compression, **kwargs):
        self._valid()
        return {"id": ident,
                "call": ["awkward", self.__class__.__name__],
                "args": [fill(self._content, self.__class__.__name__ + ".content", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                         fill(self._generator, self.__class__.__name__ + ".generator", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                         {"tuple": [fill(x, self.__class__.__name__ + ".args", prefix, suffix, schemasuffix, storage, compression, **kwargs) for x in self._args]},
                         {"dict": {n: fill(x, self.__class__.__name__ + ".kwargs", prefix, suffix, schemasuffix, storage, compression, **kwargs) for n, x in self._kwargs.items()}}]}

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = awkward.util.toarray(value, self.DEFAULTTYPE)

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, value):
        if not callable(value):
            raise TypeError("generator must be a callable (of one argument: the array slice)")
        self._generator = value

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, value):
        if not isinstance(value, tuple):
            value = (value,)
        self._args = value

    @property
    def kwargs(self):
        return self._kwargs

    @kwargs.setter
    def kwargs(self, value):
        if not isinstance(value, dict):
            raise TypeError("kwargs must be a dict")
        self._kwargs = dict(value)

    def __len__(self):
        return len(self._content)

    def _gettype(self, seen):
        return self._generator

    def _valid(self):
        pass
        
    def __iter__(self, checkiter=True):
        if checkiter:
            self._checkiter()
        for x in self._content:
            yield self.generator(x, *self._args, **self._kwargs)

    def __getitem__(self, where):
        if awkward.util.isstringslice(where):
            return self._content[where]

        if isinstance(where, tuple) and where == ():
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        content = self._content[head]
        if isinstance(head, awkward.util.integer):
            if isinstance(tail, tuple) and tail == ():
                return self.generator(content, *self._args, **self._kwargs)
            else:
                return self.generator(content, *self._args, **self._kwargs)[tail]

        elif isinstance(tail, tuple) and tail == ():
            return self.copy(content=content)

        else:
            return [x[tail] for x in content]   # FIXME: in self.copy(content=content), right?

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        contents = []
        for x in inputs:
            if isinstance(x, ObjectArray):
                x._valid()
                contents.append(x._content)
            else:
                contents.append(x)

        result = getattr(ufunc, method)(*contents, **kwargs)

        if awkward.util.iscomparison(ufunc):
            return result
        else:
            return self.copy(content=result)

    def any(self):
        return any(x for x in self)

    def all(self):
        return all(x for x in self)

    @classmethod
    def concat(cls, first, *rest):
        raise NotImplementedError

    def pandas(self):
        raise NotImplementedError

####################################################################### strings

class StringMethods(object):
    """
    StringMethods
    """

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        import awkward.array.jagged

        if method != "__call__":
            raise NotImplemented

        if ufunc is awkward.util.numpy.equal or ufunc is awkward.util.numpy.not_equal:
            if len(inputs) < 2:
                raise ValueError("invalid number of arguments")
            left, right = inputs[0], inputs[1]

            if isinstance(left, (str, bytes)):
                left = StringArray.fromstr(len(right), left)
            elif isinstance(left, awkward.util.numpy.ndarray) and (left.dtype.kind == "U" or left.dtype.kind == "S"):
                left = StringArray.fromnumpy(left)
            elif isinstance(left, awkward.util.numpy.ndarray) and left.dtype == awkward.util.numpy.dtype(object):
                left = StringArray.fromiter(left)
            elif not isinstance(left, StringMethods):
                return awkward.util.numpy.zeros(len(right), dtype=self.BOOLTYPE)

            if isinstance(right, (str, bytes)):
                right = StringArray.fromstr(len(left), right)
            elif isinstance(right, awkward.util.numpy.ndarray) and (right.dtype.kind == "U" or right.dtype.kind == "S"):
                right = StringArray.fromnumpy(right)
            elif isinstance(right, awkward.util.numpy.ndarray) and right.dtype == awkward.util.numpy.dtype(object):
                right = StringArray.fromiter(right)
            elif not isinstance(right, StringMethods):
                return awkward.util.numpy.zeros(len(left), dtype=self.BOOLTYPE)

            left = awkward.array.jagged.JaggedArray(left.starts, left.stops, left.content)
            right = awkward.array.jagged.JaggedArray(right.starts, right.stops, right.content)

            maybeequal = (left.counts == right.counts)

            leftmask = left[maybeequal]
            rightmask = right[maybeequal]

            reallyequal = (leftmask == rightmask).count_nonzero() == leftmask.counts

            out = awkward.util.numpy.zeros(len(left), dtype=self.BOOLTYPE)
            out[maybeequal] = reallyequal

            if ufunc is awkward.util.numpy.equal:
                return out
            else:
                return awkward.util.numpy.logical_not(out)

        else:
            return super(StringMethods, self).__array_ufunc__(ufunc, method, *inputs, **kwargs)

def tostring(x, decoder):
    if decoder is None:
        return x.tostring()
    else:
        return decoder(x, errors="replace")[0]

class StringArray(StringMethods, ObjectArray):
    """
    StringArray
    """

    def __init__(self, starts, stops, content, encoding="utf-8"):
        import awkward.array.jagged
        self._content = awkward.array.jagged.JaggedArray(starts, stops, content)
        self._generator = tostring
        self._kwargs = {}
        self.encoding = encoding

    @classmethod
    def fromstr(cls, length, string, encoding="utf-8"):   # FIXME: infer encoding from string
        if encoding is not None:
            encoder = codecs.getencoder(encoding)
            string = encoder(string)[0]
        content = awkward.util.numpy.empty(length * len(string), dtype=cls.CHARTYPE)
        for i, x in string:
            content[0::length] = ord(x)                   # FIXME: use numpy.tile!
        counts = awkward.util.numpy.empty(length, dtype=cls.INDEXTYPE)
        counts[:] = length
        return cls.fromcounts(counts, content, encoding)

    @classmethod
    def fromnumpy(cls, array):
        import awkward.array.jagged

        if array.dtype.kind == "S":
            encoding = None
        elif array.dtype.kind == "U":
            encoding = "utf-32le"
        else:
            raise TypeError("not a string array")

        starts = awkward.util.numpy.arange(                   0,  len(array)      * array.dtype.itemsize, array.dtype.itemsize)
        stops  = awkward.util.numpy.arange(array.dtype.itemsize, (len(array) + 1) * array.dtype.itemsize, array.dtype.itemsize)
        content = array.view(cls.CHARTYPE)

        shorter = awkward.util.numpy.ones(len(array), dtype=cls.BOOLTYPE)
        if array.dtype.kind == "S":
            for checkat in range(array.dtype.itemsize - 1, -1, -1):
                shorter &= (content[checkat::array.dtype.itemsize] == 0)
                stops[shorter] -= 1
                if not shorter.any():
                    break

        elif array.dtype.kind == "U":
            content2 = content.view(awkward.util.numpy.uint32)
            itemsize2 = array.dtype.itemsize >> 2                 # itemsize // 4
            for checkat in range(itemsize2 - 1, -1, -1):
                shorter &= (content2[checkat::itemsize2] == 0)    # all four bytes are zero
                stops[shorter] -= 4
                if not shorter.any():
                    break

        out = cls.__new__(cls)
        out._content = awkward.array.jagged.JaggedArray(starts, stops, content)
        out._generator = tostring
        out._kwargs = {}
        out.encoding = encoding
        return out
        
    @classmethod
    def fromiter(cls, iterable, encoding="utf-8"):
        if encoding is None:
            encoded = iterable
        else:
            encoder = codecs.getencoder(encoding)
            encoded = [encoder(x)[0] for x in iterable]
        counts = [len(x) for x in encoded]
        content = awkward.util.numpy.empty(sum(counts), dtype=cls.CHARTYPE)
        i = 0
        for x in encoded:
            content[i : i + len(x)] = awkward.util.numpy.frombuffer(x, dtype=cls.CHARTYPE)
            i += len(x)
        return cls.fromcounts(counts, content, encoding)

    @classmethod
    def fromoffsets(cls, offsets, content, encoding="utf-8"):
        import awkward.array.jagged
        out = cls.__new__(cls)
        out._content = awkward.array.jagged.JaggedArray.fromoffsets(offsets, content)
        out._generator = tostring
        out._kwargs = {}
        out.encoding = encoding
        return out

    @classmethod
    def fromcounts(cls, counts, content, encoding="utf-8"):
        import awkward.array.jagged
        out = cls.__new__(cls)
        out._content = awkward.array.jagged.JaggedArray.fromcounts(counts, content)
        out._generator = tostring
        out._kwargs = {}
        out.encoding = encoding
        return out

    @classmethod
    def fromparents(cls, parents, content, encoding="utf-8"):
        import awkward.array.jagged
        out = cls.__new__(cls)
        out._content = awkward.array.jagged.JaggedArray.fromparents(parents, content)
        out._generator = tostring
        out._kwargs = {}
        out.encoding = encoding
        return out

    @classmethod
    def fromuniques(cls, uniques, content, encoding="utf-8"):
        import awkward.array.jagged
        out = cls.__new__(cls)
        out._content = awkward.array.jagged.JaggedArray.fromuniques(uniques, content)
        out._generator = tostring
        out._kwargs = {}
        out.encoding = encoding
        return out

    @classmethod
    def fromjagged(cls, jagged, encoding="utf-8"):
        if jagged.content.type.to != cls.CHARTYPE:
            raise TypeError("jagged array must have CHARTYPE ({0})".format(str(cls.CHARTYPE)))
        out = cls.__new__(cls)
        out._content = jagged
        out._generator = tostring
        out._kwargs = {}
        out.encoding = encoding
        return out

    def copy(self, starts=None, stops=None, content=None, encoding=None):
        import awkward.array.jagged
        out = self.__class__.__new__(self.__class__)
        out._content = awkward.array.jagged.JaggedArray(self.starts, self.stops, self.content)
        out._generator = self._generator
        out._args = self._args
        out._kwargs = self._kwargs
        out._encoding = self._encoding
        if starts is not None:
            out.starts = starts
        if stops is not None:
            out.stops = stops
        if content is not None:
            out.content = content
        if encoding is not None:
            out.encoding = encoding
        return out

    def deepcopy(self, starts=None, stops=None, content=None, encoding=None):
        out = self.copy(starts=starts, stops=stops, content=content, encoding=encoding)
        out._content._starts = awkward.util.deepcopy(out._content._starts)
        out._content._stops = awkward.util.deepcopy(out._content._stops)
        out._content._content = awkward.util.deepcopy(out._content._content)
        return out

    def empty_like(self, **overrides):
        mine = {}
        mine["encoding"] = overrides.pop("encoding", self._encoding)
        jagged = self._content.empty_like(**overrides)
        return self.copy(jagged.starts, jagged.stops, jagged.content, **mine)

    def zeros_like(self, **overrides):
        mine = {}
        mine["encoding"] = overrides.pop("encoding", self._encoding)
        jagged = self._content.zeros_like(**overrides)
        return self.copy(jagged.starts, jagged.stops, jagged.content, **mine)

    def ones_like(self, **overrides):
        mine = {}
        mine["encoding"] = overrides.pop("encoding", self._encoding)
        jagged = self._content.ones_like(**overrides)
        return self.copy(jagged.starts, jagged.stops, jagged.content, **mine)

    def __awkward_persist__(self, ident, fill, prefix, suffix, schemasuffix, storage, compression, **kwargs):
        import awkward.array.jagged
        self._valid()
        n = self.__class__.__name__
        if awkward.array.jagged.offsetsaliased(self.starts, self.stops) and len(self.starts) > 0 and self.starts[0] == 0:
            return {"id": ident,
                    "call": ["awkward", n, "fromcounts"],
                    "args": [fill(self.counts, n + ".counts", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                             fill(self.content, n + ".content", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                             self._encoding]}
        else:
            return {"id": ident,
                    "call": ["awkward", n],
                    "args": [fill(self.starts, n + ".starts", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                             fill(self.stops, n + ".stops", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                             fill(self.content, n + ".content", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                             self._encoding]}

    @property
    def starts(self):
        return self._content.starts

    @starts.setter
    def starts(self, value):
        self._content.starts = value

    @property
    def stops(self):
        return self._content.stops

    @stops.setter
    def stops(self, value):
        self._content.stops = value

    @property
    def content(self):
        return self._content.content

    @content.setter
    def content(self, value):
        self._content.content = value

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return {}

    @property
    def encoding(self):
        return self._encoding

    @encoding.setter
    def encoding(self, value):
        if value is None:
            decodefcn = None
        else:
            decodefcn = codecs.getdecoder(value)
        self._encoding = value
        self._args = (decodefcn,)

    @property
    def offsets(self):
        return self._content.offsets

    @property
    def counts(self):
        return self._content.counts

    @property
    def parents(self):
        return self._content.parents

    @property
    def index(self):
        return self._content.index

    def _gettype(self, seen):
        if self._encoding is None:
            return bytes
        else:
            return str

    def __getitem__(self, where):
        if awkward.util.isstringslice(where):
            raise IndexError("cannot index StringArray with string or sequence of strings")

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        if isinstance(head, awkward.util.integer):
            return super(StringArray, self).__getitem__(where)

        elif tail == ():
            out = self._content[where]
            return self.__class__(out.starts, out.stops, out.content, self.encoding)

        else:
            out = self._content[where]
            return self.__class__(out.starts, out.stops, out.content, self.encoding)
