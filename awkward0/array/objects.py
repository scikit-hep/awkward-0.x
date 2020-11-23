#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-0.x/blob/master/LICENSE

import codecs
import importlib

import awkward0.array.base
import awkward0.persist
import awkward0.type
import awkward0.util

class Methods(object):
    """
    Methods: abstract mix-in
    """

    @staticmethod
    def mixin(methods, awkwardtype):
        assert not issubclass(methods, awkward0.array.base.AwkwardArray)
        assert not issubclass(awkwardtype, Methods)
        return type(awkwardtype.__name__ + "Methods", (methods, awkwardtype), {})

    @staticmethod
    def maybemixin(sample, awkwardtype):
        if issubclass(sample, Methods):
            assert issubclass(sample, awkward0.array.base.AwkwardArray)
            allbases = tuple(x for x in sample.__bases__ if not issubclass(x, awkward0.array.base.AwkwardArray)) + (awkwardtype,)
            return type(awkwardtype.__name__ + "Methods", allbases, {})
        else:
            return awkwardtype

class ObjectArray(awkward0.array.base.AwkwardArrayWithContent):
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
        out._content = self._util_deepcopy(out._content)
        return out

    def empty_like(self, **overrides):
        mine = {}
        mine["generator"] = overrides.pop("generator", self._generator)
        mine["args"] = overrides.pop("args", self._args)
        mine["kwargs"] = overrides.pop("kwargs", self._kwargs)
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.empty_like(self._content), **mine)
        else:
            return self.copy(content=self._content.empty_like(**overrides), **mine)

    def zeros_like(self, **overrides):
        mine = {}
        mine["generator"] = overrides.pop("generator", self._generator)
        mine["args"] = overrides.pop("args", self._args)
        mine["kwargs"] = overrides.pop("kwargs", self._kwargs)
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.zeros_like(self._content), **mine)
        else:
            return self.copy(content=self._content.zeros_like(**overrides), **mine)

    def ones_like(self, **overrides):
        mine = {}
        mine["generator"] = overrides.pop("generator", self._generator)
        mine["args"] = overrides.pop("args", self._args)
        mine["kwargs"] = overrides.pop("kwargs", self._kwargs)
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.ones_like(self._content), **mine)
        else:
            return self.copy(content=self._content.ones_like(**overrides), **mine)

    def __awkward_serialize__(self, serializer):
        self._valid()
        return serializer.encode_call(
            ["awkward0", "ObjectArray"],
            serializer(self._content, "ObjectArray.content"),
            serializer(self._generator, "ObjectArray.generator"),
            {"tuple": [
                serializer(x, "ObjectArray.args") for x in self._args
            ]},
            {"dict": {
                n: serializer(x, "ObjectArray.kwargs")
                for n, x in self._kwargs.items()
            }},
        )

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = self._util_toarray(value, self.DEFAULTTYPE)

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, value):
        if self.check_prop_valid:
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
        if self.check_prop_valid:
            if not isinstance(value, dict):
                raise TypeError("kwargs must be a dict")
        self._kwargs = dict(value)

    def _getnbytes(self, seen):
        if id(self) in seen:
            return 0
        else:
            seen.add(id(self))
            return (self._content.nbytes if isinstance(self._content, self.numpy.ndarray) else self._content._getnbytes(seen))

    def __len__(self):
        return len(self._content)

    def _gettype(self, seen):
        return self._generator

    def _util_layout(self, position, seen, lookup):
        awkward0.type.LayoutNode(self._content, position + (0,), seen, lookup)
        return (awkward0.type.LayoutArg("content", position + (0,)),
                awkward0.type.LayoutArg("generator", self._generator),
                awkward0.type.LayoutArg("args", self._args),
                awkward0.type.LayoutArg("kwargs", self._kwargs))

    def _valid(self):
        if self.check_whole_valid:
            pass

    def __iter__(self, checkiter=True):
        if checkiter:
            self._checkiter()
        for x in self._content:
            yield self.generator(x, *self._args, **self._kwargs)

    def __getitem__(self, where):
        if self._util_isstringslice(where):
            return self._content[where]

        if isinstance(where, tuple) and where == ():
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        content = self._content[head]
        if self._util_isinteger(head):
            if isinstance(tail, tuple) and tail == ():
                return self.generator(content, *self._args, **self._kwargs)
            else:
                return self.generator(content, *self._args, **self._kwargs)[tail]

        elif isinstance(tail, tuple) and tail == ():
            return self.copy(content=content)

        else:
            return [x[tail] for x in content]   # FIXME: in self.copy(content=content), right?

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            raise NotImplementedError("in-place operations not supported")

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

        if self._util_iscomparison(ufunc):
            return result
        else:
            return self.copy(content=result)

    def _hasjagged(self):
        return False

    @property
    def counts(self):
        return self._util_counts(self._content)

    def boolmask(self, maskedwhen=True):
        return self._util_boolmask(self._content, maskedwhen)

    def choose(self, n):
        raise TypeError("cannot call choose on ObjectArray")

    def argchoose(self, n):
        raise TypeError("cannot call argchoose on ObjectArray")

    def distincts(self, nested=False):
        raise TypeError("cannot call distincts on ObjectArray")

    def argdistincts(self, nested=False):
        raise TypeError("cannot call argdistincts on ObjectArray")

    def pairs(self, nested=False):
        raise TypeError("cannot call pairs on ObjectArray")

    def argpairs(self, nested=False):
        raise TypeError("cannot call argpairs on ObjectArray")

    def cross(self, other, nested=False):
        raise TypeError("cannot call cross on ObjectArray")

    def argcross(self, other, nested=False):
        raise TypeError("cannot call argcross on ObjectArray")

    def flattentuple(self):
        return self.copy(content=self._util_flattentuple(self._content))

    def flatten(self, axis=0):
        return self.copy(content=self._util_flatten(self._content, axis))

    def pad(self, length, maskedwhen=True, clip=False, axis=0):
        return self.copy(content=self._util_pad(self._content, length, maskedwhen, clip, axis))

    def regular(self):
        return self.numpy.array(self)

    def _reduce(self, ufunc, identity, dtype):
        raise TypeError("cannot call reducer on ObjectArray")

    def _prepare(self, ufunc, identity, dtype):
        raise TypeError("cannot call reducer on ObjectArray")

    def argmin(self):
        raise TypeError("cannot call argmin on ObjectArray")

    def argmax(self):
        raise TypeError("cannot call argmax on ObjectArray")

    @classmethod
    def _concatenate_axis0(cls, arrays):
        out = arrays[0].copy(content=[])
        out._content = arrays[0]._content.__class__.concatenate([a._content for a in arrays])
        return out

    _topandas_name = "ObjectSeries"

    def _topandas(self, seen):
        import awkward0.pandas
        if id(self) in seen:
            return seen[id(self)]
        else:
            out = seen[id(self)] = self.copy()
            out.__class__ = awkward0.pandas.mixin(type(self))
            if isinstance(self._content, awkward0.array.base.AwkwardArray):
                out._content = out._content._topandas(seen)
            return out

####################################################################### strings

class StringMethods(object):
    """
    StringMethods
    """

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            raise NotImplementedError("in-place operations not supported")

        if method != "__call__":
            raise NotImplemented

        if ufunc is self.numpy.equal or ufunc is self.numpy.not_equal:
            if len(inputs) < 2:
                raise ValueError("invalid number of arguments")
            left, right = inputs[0], inputs[1]

            if isinstance(left, (str, bytes)):
                left = self.StringArray.fromstr(len(right), left)
            elif isinstance(left, self.numpy.ndarray) and (left.dtype.kind == "U" or left.dtype.kind == "S"):
                left = self.StringArray.fromnumpy(left)
            elif isinstance(left, self.numpy.ndarray) and left.dtype == self.numpy.dtype(object):
                left = self.StringArray.fromiter(left)
            elif not isinstance(left, StringMethods):
                return self.numpy.zeros(len(right), dtype=self.BOOLTYPE)

            if isinstance(right, (str, bytes)):
                right = self.StringArray.fromstr(len(left), right)
            elif isinstance(right, self.numpy.ndarray) and (right.dtype.kind == "U" or right.dtype.kind == "S"):
                right = self.StringArray.fromnumpy(right)
            elif isinstance(right, self.numpy.ndarray) and right.dtype == self.numpy.dtype(object):
                right = self.StringArray.fromiter(right)
            elif not isinstance(right, StringMethods):
                return self.numpy.zeros(len(left), dtype=self.BOOLTYPE)

            left = self.JaggedArray(left.starts, left.stops, left.content)
            right = self.JaggedArray(right.starts, right.stops, right.content)

            maybeequal = (left.counts == right.counts)

            leftmask = left[maybeequal]
            rightmask = right[maybeequal]

            reallyequal = (leftmask == rightmask).count_nonzero() == leftmask.counts

            out = self.numpy.zeros(len(left), dtype=self.BOOLTYPE)
            out[maybeequal] = reallyequal

            if ufunc is self.numpy.equal:
                return out
            else:
                return self.numpy.logical_not(out)

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
        self._content = self.JaggedArray(starts, stops, content)
        self._generator = tostring
        self._kwargs = {}
        self.encoding = encoding

    @classmethod
    def fromstr(cls, length, string, encoding="utf-8"):   # FIXME: infer encoding from string
        if encoding is not None:
            encoder = codecs.getencoder(encoding)
            string = encoder(string)[0]
        content = cls.numpy.empty(length * len(string), dtype=cls.CHARTYPE)
        for i, x in string:
            content[0::length] = ord(x)                   # FIXME: use numpy.tile!
        counts = cls.numpy.empty(length, dtype=cls.INDEXTYPE)
        counts[:] = length
        return cls.fromcounts(counts, content, encoding)

    @classmethod
    def fromnumpy(cls, array):
        if array.dtype.kind == "S":
            encoding = None
        elif array.dtype.kind == "U":
            encoding = "utf-32le"
        else:
            raise TypeError("not a string array")

        starts = cls.numpy.arange(                   0,  len(array)      * array.dtype.itemsize, array.dtype.itemsize)
        stops  = cls.numpy.arange(array.dtype.itemsize, (len(array) + 1) * array.dtype.itemsize, array.dtype.itemsize)
        content = array.view(cls.CHARTYPE)

        shorter = cls.numpy.ones(len(array), dtype=cls.BOOLTYPE)
        if array.dtype.kind == "S":
            for checkat in range(array.dtype.itemsize - 1, -1, -1):
                shorter &= (content[checkat::array.dtype.itemsize] == 0)
                stops[shorter] -= 1
                if not shorter.any():
                    break

        elif array.dtype.kind == "U":
            content2 = content.view(cls.numpy.uint32)
            itemsize2 = array.dtype.itemsize >> 2                 # itemsize // 4
            for checkat in range(itemsize2 - 1, -1, -1):
                shorter &= (content2[checkat::itemsize2] == 0)    # all four bytes are zero
                stops[shorter] -= 4
                if not shorter.any():
                    break

        out = cls.__new__(cls)
        out._content = cls.JaggedArray.fget(None)(starts, stops, content)
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
        content = cls.numpy.empty(sum(counts), dtype=cls.CHARTYPE)
        i = 0
        for x in encoded:
            content[i : i + len(x)] = cls.numpy.frombuffer(x, dtype=cls.CHARTYPE)
            i += len(x)
        return cls.fromcounts(counts, content, encoding)

    @classmethod
    def fromoffsets(cls, offsets, content, encoding="utf-8"):
        out = cls.__new__(cls)
        out._content = cls.JaggedArray.fget(None).fromoffsets(offsets, content)
        out._generator = tostring
        out._kwargs = {}
        out.encoding = encoding
        return out

    @classmethod
    def fromcounts(cls, counts, content, encoding="utf-8"):
        out = cls.__new__(cls)
        out._content = cls.JaggedArray.fget(None).fromcounts(counts, content)
        out._generator = tostring
        out._kwargs = {}
        out.encoding = encoding
        return out

    @classmethod
    def fromparents(cls, parents, content, encoding="utf-8"):
        out = cls.__new__(cls)
        out._content = cls.JaggedArray.fget(None).fromparents(parents, content)
        out._generator = tostring
        out._kwargs = {}
        out.encoding = encoding
        return out

    @classmethod
    def fromuniques(cls, uniques, content, encoding="utf-8"):
        out = cls.__new__(cls)
        out._content = cls.JaggedArray.fget(None).fromuniques(uniques, content)
        out._generator = tostring
        out._kwargs = {}
        out.encoding = encoding
        return out

    @classmethod
    def fromjagged(cls, jagged, encoding="utf-8"):
        if awkward0.type.fromarray(jagged.content).to != cls.CHARTYPE:
            raise TypeError("jagged array must have CHARTYPE ({0})".format(str(cls.CHARTYPE)))
        out = cls.__new__(cls)
        out._content = jagged
        out._generator = tostring
        out._kwargs = {}
        out.encoding = encoding
        return out

    def copy(self, starts=None, stops=None, content=None, encoding=None):
        out = self.__class__.__new__(self.__class__)
        out._content = self.JaggedArray(self.starts, self.stops, self.content)
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
        out._content._starts = self._util_deepcopy(out._content._starts)
        out._content._stops = self._util_deepcopy(out._content._stops)
        out._content._content = self._util_deepcopy(out._content._content)
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

    def __awkward_serialize__(self, serializer):
        self._valid()
        if self._content.offsetsaliased(self.starts, self.stops) and len(self.starts) > 0 and self.starts[0] == 0:
            return serializer.encode_call(
                ["awkward0", "StringArray", "fromcounts"],
                serializer(self.counts, "StringArray.counts"),
                serializer(self.content, "StringArray.content"),
                serializer(self._encoding),
            )
        else:
            return serializer.encode_call(
                ["awkward0", "StringArray"],
                serializer(self.starts, "StringArray.starts"),
                serializer(self.stops, "StringArray.stops"),
                serializer(self.content, "StringArray.content"),
                serializer(self._encoding),
            )

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
        if self._util_isstringslice(where):
            raise IndexError("cannot index StringArray with string or sequence of strings")

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        if self._util_isinteger(head):
            return super(StringArray, self).__getitem__(where)

        elif tail == ():
            out = self._content[where]
            return self.__class__(out.starts, out.stops, out.content, self.encoding)

        else:
            out = self._content[where]
            return self.__class__(out.starts, out.stops, out.content, self.encoding)

    def regular(self):
        self._valid()
        return self.numpy.array(self)

    @property
    def iscompact(self):
        return self._content.iscompact

    def compact(self):
        return self.fromjagged(self._content.compact(), self.encoding)

    def flatten(self, axis=0):
        import awkward0.array.jagged
        content = self._util_flatten(self._content, axis)
        if isinstance(content, awkward0.array.jagged.JaggedArray):
            return self.fromjagged(content, self._encoding)
        else:
            return self.fromjagged(self.JaggedArray.fromcounts([len(content)], content))

    def pad(self, length, maskedwhen=None, clip=False, axis=0):
        if not self._util_isinteger(axis) or axis < 0:
            raise TypeError("axis must be a non-negative integer (can't count from the end)")

        if axis > 0:
            raise ValueError("axis too deep for StringArray")

        if maskedwhen is None:
            maskedwhen = ord(b" ")
        elif not isinstance(maskedwhen, bytes) or not len(maskedwhen) == 1:
            raise TypeError("to pad a StringArray, set maskedwhen to a one-character bytestring, such as b' '")
        else:
            maskedwhen = ord(maskedwhen)
        import awkward0.array.jagged
        import awkward0.array.masked
        padded = self._util_pad(self._content, length, True, clip)
        assert isinstance(padded, awkward0.array.jagged.JaggedArray)
        assert isinstance(padded.content, awkward0.array.masked.MaskedArray)
        if padded.content.content is self._content.content:
            chars = padded.content.content.copy()
        else:
            chars = padded.content.content
        chars[padded.content.mask] = maskedwhen
        padded.content = chars
        return self.fromjagged(padded, self._encoding)

    # @awkward0.util.bothmethod
    # def concatenate(isclassmethod, cls_or_self, arrays, axis=0):
    #     if isclassmethod:
    #         cls = cls_or_self
    #         if not all(isinstance(x, StringArray) for x in arrays):
    #             raise TypeError("cannot concatenate non-StringArrays with StringArray.concatenate")
    #     else:
    #         self = cls_or_self
    #         cls = self.__class__
    #         if not isinstance(self, StringArray) or not all(isinstance(x, StringArray) for x in arrays):
    #             raise TypeError("cannot concatenate non-StringArrays with StringArrays.concatenate")
    #         arrays = (self,) + tuple(arrays)
    #
    #     jagged = self.JaggedArray.concatenate([x._content for x in arrays], axis=axis)
    #     return self.fromjagged(jagged, self.encoding)

    @classmethod
    def _concatenate_axis0(cls, arrays):
        assert all(isinstance(x, StringArray) for x in arrays)
        return cls.fromjagged(cls.JaggedArray.fget(None)._concatenate_axis0([x._content for x in arrays]), encoding=arrays[0]._encoding)

    @classmethod
    def _concatenate_axis1(cls, arrays):
        assert all(isinstance(x, StringArray) for x in arrays)
        tmp = cls.JaggedArray.fget(None)._concatenate_axis1([x._content for x in arrays])
        tmp._content = tmp._content.astype(cls.CHARTYPE)
        return cls.fromjagged(tmp, encoding=arrays[0]._encoding)

    def fillna(self, value):
        return self

    _topandas_name = "StringSeries"

    def _topandas(self, seen):
        import awkward0.pandas
        if id(self) in seen:
            return seen[id(self)]
        else:
            out = seen[id(self)] = self.copy()
            out.__class__ = awkward0.pandas.mixin(type(self))
            if isinstance(self._content, awkward0.array.base.AwkwardArray):
                out._content = out._content._topandas(seen)
            return out
