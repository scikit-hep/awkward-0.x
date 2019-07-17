#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import operator
import math
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy
import numba
import numba.typing.arraydecl

import awkward.array.base
import awkward.array.jagged
from .base import NumbaMethods
from .base import AwkwardArrayType
from .base import clsrepr
from .base import ISADVANCED
from .base import NOTADVANCED
from .base import sliceval2
from .base import sliceval3

######################################################################## Numba-accelerated interface

class JaggedArrayNumba(NumbaMethods, awkward.array.jagged.JaggedArray):
    # @classmethod
    # def offsetsaliased(cls, starts, stops):
    ### base implementation is fine and don't need in Numba

    @classmethod
    def counts2offsets(cls, counts):
        return cls._counts2offsets(counts)

    @staticmethod
    @numba.njit
    def _counts2offsets(counts):
        offsets = numpy.empty(len(counts) + 1, dtype=numpy.int64)
        offsets[0] = 0
        for i in range(len(counts)):
            offsets[i + 1] = offsets[i] + counts[i]
        return offsets

    @classmethod
    def offsets2parents(cls, offsets):
        return cls._offsets2parents(offsets)

    @staticmethod
    @numba.njit
    def _offsets2parents(offsets):
        if len(offsets) == 0:
            raise ValueError("offsets must have at least one element")
        parents = numpy.empty(offsets[-1], dtype=numpy.int64)
        j = 0
        k = -1
        for i in offsets:
            while j < i:
                parents[j] = k
                j += 1
            k += 1
        return parents

    @classmethod
    def startsstops2parents(cls, starts, stops):
        return cls._startsstops2parents(starts, stops)

    @staticmethod
    @numba.njit
    def _startsstops2parents(starts, stops):
        out = numpy.full(stops.max(), -1, numpy.int64)
        for i in range(len(starts)):
            out[starts[i]:stops[i]] = i
        return out

    @classmethod
    def parents2startsstops(cls, parents, length=None):
        if length is None:
            length = parents.max() + 1
        return cls._parents2startsstops(parents, length)

    @staticmethod
    @numba.njit
    def _parents2startsstops(parents, length):
        starts = numpy.zeros(length, numpy.int64)
        stops = numpy.zeros(length, numpy.int64)

        last = -1
        for k in range(len(parents)):
            this = parents[k]
            if last != this:
                if last >= 0 and last < length:
                    stops[last] = k
                if this >= 0 and this < length:
                    starts[this] = k
            last = this

        if last != -1:
            stops[last] = len(parents)

        return starts, stops

    # @classmethod
    # def uniques2offsetsparents(cls, uniques):
    ### base implementation is fine and don't need in Numba

    # def __init__(self, starts, stops, content):
    ### base implementation is fine and already exposed in Numba

    @classmethod
    def fromiter(cls, iterable):
        import awkward.numba
        return awkward.numba.fromiter(iterable)

    # @classmethod
    # def fromoffsets(cls, offsets, content):
    ### base implementation is fine and don't need in Numba

    # @classmethod
    # def fromcounts(cls, counts, content):
    ### base implementation is fine and don't need in Numba

    # @classmethod
    # def fromparents(cls, parents, content, length=None):
    ### base implementation is fine and don't need in Numba

    # @classmethod
    # def fromuniques(cls, uniques, content):
    ### base implementation is fine and don't need in Numba

    # @classmethod
    # def fromindex(cls, index, content, validate=True):
    ### base implementation is fine and don't need in Numba

    # @classmethod
    # def fromjagged(cls, jagged):
    ### base implementation is fine and don't need in Numba

    # @classmethod
    # def fromregular(cls, regular):
    ### base implementation is fine and don't need in Numba

    # @classmethod
    # def fromfolding(cls, content, size):
    ### base implementation is fine and don't need in Numba

    # def copy(self, starts=None, stops=None, content=None):
    ### base implementation is fine and don't need in Numba

    # def deepcopy(self, starts=None, stops=None, content=None):
    ### base implementation is fine and don't need in Numba

    # def empty_like(self, **overrides):
    ### base implementation is fine and don't need in Numba

    # def zeros_like(self, **overrides):
    ### base implementation is fine and don't need in Numba

    # def ones_like(self, **overrides):
    ### base implementation is fine and don't need in Numba

    # def __awkward_persist__(self, ident, fill, prefix, suffix, schemasuffix, storage, compression, **kwargs):
    ### base implementation is fine and don't need in Numba

    # @property
    # def starts(self):
    ### base implementation is fine and already exposed in Numba

    # @starts.setter
    # def starts(self, value):
    ### base implementation is fine and don't need in Numba
        
    # @property
    # def stops(self):
    ### base implementation is fine and already exposed in Numba

    # @stops.setter
    # def stops(self, value):
    ### base implementation is fine and don't need in Numba

    # @property
    # def content(self):
    ### base implementation is fine and already exposed in Numba

    # @content.setter
    # def content(self, value):
    ### base implementation is fine and don't need in Numba

    # @property
    # def offsets(self):
    ### base implementation is fine and already exposed in Numba

    # @offsets.setter
    # def offsets(self, value):
    ### base implementation is fine and don't need in Numba

    # @property
    # def counts(self):
    ### base implementation is fine and already exposed in Numba

    # @counts.setter
    # def counts(self, value):
    ### base implementation is fine and don't need in Numba

    # @property
    # def parents(self):
    ### base implementation is fine and already exposed in Numba

    # @parents.setter
    # def parents(self, value):
    ### base implementation is fine and don't need in Numba

    # @property
    # def index(self):
    ### base implementation is fine and already exposed in Numba

    # def __len__(self):
    ### base implementation is fine and already exposed in Numba

    # def _gettype(self, seen):
    ### base implementation is fine and don't need in Numba

    def _valid(self):
        pass             # do validation in place from now on

    # @staticmethod
    # def _validstartsstops(starts, stops):
    ### base implementation is fine and don't need in Numba

    # def __iter__(self, checkiter=True):
    ### base implementation is fine and already exposed in Numba

    def __getitem__(self, where):
        if not isinstance(where, tuple):
            where = (where,)
        if len(where) == 0:
            return self

        newwhere = ()
        for x in where:
            if isinstance(x, Iterable) and not isinstance(x, (numpy.ndarray, awkward.array.base.AwkwardArray)):
                newwhere = newwhere + (numpy.array(x),)
            else:
                newwhere = newwhere + (x,)

        if len(newwhere) == 1:
            newwhere = newwhere[0]
            
        return self._getitem_impl(newwhere)

    @numba.njit
    def _getitem_impl(self, newwhere):
        return self[newwhere]

    # def __setitem__(self, where, what):

    @numba.generated_jit(nopython=True)
    def tojagged(self, data):
        assert not isinstance(data, JaggedArrayType)

        if isinstance(data, AwkwardArrayType):
            def impl(self, data):
                if len(self.starts) != len(data):
                    raise ValueError("cannot broadcast AwkwardArray to match JaggedArray with a different length")
                if len(self.starts.shape) != 1:
                    raise ValueError("cannot broadcast AwkwardArray to match JaggedArray that has len(starts.shape) != 1; call jagged.structure1d() first")
                index = numpy.empty(len(self.content), numpy.int64)
                for i in range(len(self.starts)):
                    index[self.starts[i]:self.stops[i]] = i
                return _JaggedArray_new(self, self.starts, self.stops, data[index], self.iscompact)
            return impl

        elif isinstance(data, numba.types.Array):
            def impl(self, data):
                if self.starts.shape != data.shape:
                    raise ValueError("cannot broadcast Numpy array to match a JaggedArray with a different length (or more generally, starts.shape)")
                content = numpy.empty(len(self.content), data.dtype)
                flatstarts = self.starts.reshape(-1)
                flatstops = self.stops.reshape(-1)
                flatdata = data.reshape(-1)
                for i in range(len(flatstarts)):
                    content[flatstarts[i]:flatstops[i]] = flatdata[i]
                return _JaggedArray_new(self, self.starts, self.stops, content, self.iscompact)
            return impl

        else:
            def impl(self, data):
                content = numpy.full(len(self.content), data)
                return _JaggedArray_new(self, self.starts, self.stops, content, self.iscompact)
            return impl

    # def _tojagged(self, starts=None, stops=None, copy=True):

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            raise NotImplementedError("in-place operations not supported")
        if method != "__call__":
            return NotImplemented

        first = None
        inputs = list(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], awkward.array.jagged.JaggedArray):
                inputs[i] = inputs[i].compact()
                shift = inputs[i].starts[0]
                if shift != 0:
                    inputs[i] = inputs[i].copy(inputs[i].starts - shift, inputs[i].stops - shift, inputs[i].content[shift:])
                if first is None:
                    first = inputs[i]
                elif first.starts[0] != inputs[i].starts[0] or not numpy.array_equal(first.stops, inputs[i].stops):
                    raise ValueError("JaggedArrays in Numpy ufunc have incompatible structure")

        assert first is not None

        for i in range(len(inputs)):
            if isinstance(inputs[i], awkward.array.jagged.JaggedArray):
                pass
            elif isinstance(inputs[i], awkward.array.base.AwkwardArray):
                inputs[i] = first.tojagged(inputs[i])
            elif isinstance(inputs[i], Iterable):
                inputs[i] = first.tojagged(numpy.array(inputs[i], copy=False))
            else:
                inputs[i] = first.tojagged(inputs[i])

        for i in range(len(inputs)):
            inputs[i] = inputs[i]._content

        result = getattr(ufunc, method)(*inputs, **kwargs)

        if isinstance(result, tuple):
            return tuple(self.Methods.maybemixin(type(x), self.JaggedArray)(first.starts, first.stops, x) if isinstance(x, (numpy.ndarray, awkward.array.base.AwkwardArray)) else x for x in result)
        else:
            return self.Methods.maybemixin(type(result), self.JaggedArray)(first.starts, first.stops, result)

    @numba.njit
    def regular(self):
        return self.regular()

    ### FIXME: this whole section can't be done until we have Tables

    # def _argpairs(self):

    # def _argdistincts(self, absolute):

    # def argdistincts(self, nested=False):

    # def distincts(self, nested=False):

    # def argpairs(self, nested=False):

    # def pairs(self, nested=False):

    # def _argcross(self, other):

    # def argcross(self, other, nested=False):

    # def cross(self, other, nested=False):

    # def _canuseoffset(self):
    ### base implementation is fine and don't need in Numba

    # @property
    # def iscompact(self):
    ### base implementation is fine and already exposed in Numba

    @numba.njit
    def compact(self):
        return self.compact()

    def flatten(self, axis=0):
        if not self._util_isinteger(axis) or axis < 0:
            raise TypeError("axis must be a non-negative integer (can't count from the end)")
        if axis > 0:
            if isinstance(self._content, JaggedArray):
                counts = self.JaggedArray.fromcounts(self.counts, self._content.counts).sum()
                return self.JaggedArray.fromcounts(counts, self._content.flatten(axis=axis - 1))

        if len(self) == 0:
            return self._content[0:0]
        elif self.iscompact:
            return self._content[self._starts[0]:self.stops[-1]]  # no underscore in stops
        else:
            out = self.compact()
            return out._content[out._starts[0]:out.stops[0]]      # no underscore in stops

    # def structure1d(self, levellimit=None):
    ### base implementation is fine and can't(?) be exposed in Numba (type manipulation is hard!)

    # def _hasjagged(self):
    ### base implementation is fine

    # def _reduce(self, ufunc, identity, dtype):

    @numba.njit
    def argmin(self):
        return self.argmin()

    @numba.njit
    def argmax(self):
        return self.argmax()

    def _argminmax(self, ismin):
        raise RuntimeError("helper function not needed in JaggedArrayNumba")

    def _argminmax_general(self, ismin):
        raise RuntimeError("helper function not needed in JaggedArrayNumba")

    # def _concatenate_axis0(isclassmethod, cls_or_self, arrays):
    #     if isinstance(arrays, (numpy.ndarray, awkward.array.base.AwkwardArray)):
    #         arrays = (arrays,)
    #     else:
    #         arrays = tuple(arrays)
    #     if isclassmethod:
    #         cls = cls_or_self
    #         if not all(isinstance(x, awkward.array.jagged.JaggedArray) for x in arrays):
    #             raise TypeError("cannot concatenate non-JaggedArrays with JaggedArray.concatenate")
    #     else:
    #         self = cls_or_self
    #         cls = self.__class__
    #         if not isinstance(self, awkward.array.jagged.JaggedArray) or not all(isinstance(x, awkward.array.jagged.JaggedArray) for x in arrays):
    #             raise TypeError("cannot concatenate non-JaggedArrays with JaggedArray.concatenate")
    #         arrays = (self,) + arrays
    #     if len(arrays) == 0:
    #         raise TypeError("concatenate requires at least one array")
    #     return _JaggedArray_concatenate_njit(arrays, axis)
    ### FIXME: left unfinished

    # @awkward.util.bothmethod
    # def zip(isclassmethod, cls_or_self, columns1={}, *columns2, **columns3):
    ### FIXME: can't do this one until we have Tables

    # def pad(self, length, maskedwhen=True, clip=False):

######################################################################## register types in Numba

@numba.extending.typeof_impl.register(awkward.array.jagged.JaggedArray)
def _JaggedArray_typeof(val, c):
    return JaggedArrayType(numba.typeof(val.starts), numba.typeof(val.stops), numba.typeof(val.content), special=type(val))

class JaggedArrayType(AwkwardArrayType):
    def __init__(self, startstype, stopstype, contenttype, special=awkward.array.jagged.JaggedArray):
        super(JaggedArrayType, self).__init__(name="JaggedArrayType({0}, {1}, {2}{3})".format(startstype.name, stopstype.name, contenttype.name, "" if special is awkward.array.jagged.JaggedArray else clsrepr(special)))
        if startstype.ndim != stopstype.ndim:
            raise ValueError("JaggedArray.starts must have the same number of dimensions as JaggedArray.stops")
        if startstype.ndim == 0:
            raise ValueError("JaggedArray.starts and JaggedArray.stops must have at least one dimension")
        self.startstype = startstype
        self.stopstype = stopstype
        self.contenttype = contenttype
        self.special = special

    def getitem(self, wheretype):
        if self.startstype.ndim > 1 and not any(isinstance(x, (numba.types.Array, JaggedArrayType)) for x in wheretype.types[:self.startstype.ndim]):
            headtype = numba.types.Tuple(wheretype.types[:self.startstype.ndim])
            tailtype = numba.types.Tuple(wheretype.types[self.startstype.ndim:])

            outstartstype = numba.typing.arraydecl.get_array_index_type(self.startstype, headtype).result
            outstopstype = numba.typing.arraydecl.get_array_index_type(self.stopstype, headtype).result
            if isinstance(self.contenttype, JaggedArrayType):
                outcontenttype = self.contenttype.getitem(tailtype)
            else:
                outcontenttype = numba.typing.arraydecl.get_array_index_type(self.contenttype, tailtype).result

            assert isinstance(outstartstype, numba.types.Array) == isinstance(outstopstype, numba.types.Array)
            if isinstance(outstartstype, numba.types.Array):
                return JaggedArrayType(outstartstype, outstopstype, outcontenttype, special=self.special)
            else:
                return outcontenttype

        else:
            headtype = wheretype.types[0]
            tailtype = numba.types.Tuple(wheretype.types[1:])

            if isinstance(headtype, JaggedArrayType) and len(tailtype.types) == 0:
                return _JaggedArray_typer_getitem_jagged(self, headtype)

            else:
                fake = _JaggedArray_typer_getitem(JaggedArrayType(JaggedArrayNumba.NUMBA_INDEXTYPE[:], JaggedArrayNumba.NUMBA_INDEXTYPE[:], self), wheretype, NOTADVANCED)
                if isinstance(fake, numba.types.Array):
                    return fake.dtype
                else:
                    return fake.contenttype

    @property
    def len_impl(self):
        return _JaggedArray_lower_len

    @property
    def getitem_impl(self):
        return lambda context, builder, sig, args: _JaggedArray_lower_getitem_integer(context, builder, sig, args, checkvalid=False)

def _JaggedArray_typer_getitem_jagged(arraytype, headtype):
    if isinstance(headtype.contenttype, JaggedArrayType):
        if not isinstance(arraytype.contenttype, JaggedArrayType):
            raise TypeError("index (in square brackets) is more deeply jagged than array (before square brackets)")
        contenttype = _JaggedArray_typer_getitem_jagged(arraytype.contenttype, headtype.contenttype)
    elif isinstance(headtype.contenttype, numba.types.Array) and headtype.contenttype.ndim == 1 and isinstance(headtype.contenttype.dtype, (numba.types.Boolean, numba.types.Integer)):
        contenttype = arraytype.contenttype
    else:
        raise TypeError("jagged indexing must be boolean or integers with 1-dimensional content")

    return JaggedArrayType(arraytype.startstype, arraytype.stopstype, contenttype, special=arraytype.special)

def _JaggedArray_typer_getitem(arraytype, wheretype, advancedtype):
    if len(wheretype.types) == 0:
        return arraytype

    if arraytype.startstype.ndim != 1 or arraytype.stopstype.ndim != 1:
        raise NotImplementedError("multidimensional starts and stops not supported; call jagged.structure1d() first")

    isarray = (isinstance(wheretype.types[0], numba.types.Array) and wheretype.types[0].ndim == 1)

    contenttype = _JaggedArray_typer_getitem(arraytype.contenttype, numba.types.Tuple(wheretype.types[1:]), ISADVANCED if isarray else advancedtype)

    if isinstance(wheretype.types[0], numba.types.Integer) or (isarray and advancedtype == ISADVANCED):
        return contenttype
    elif isinstance(wheretype.types[0], numba.types.SliceType) or (isarray and advancedtype == NOTADVANCED):
        return JaggedArrayType(arraytype.startstype, arraytype.stopstype, contenttype, special=arraytype.special)
    elif isinstance(wheretype.types[0], JaggedArrayType):
        raise TypeError("cannot use jagged indexing in a tuple")
    else:
        raise TypeError("cannot be used for indexing: {0}".format(wheretype.types[0]))

def _JaggedArray_getitem_next(array, where):
    pass

@numba.extending.type_callable(_JaggedArray_getitem_next)
def _JaggedArray_type_getitem_next(context):
    return _JaggedArray_typer_getitem

######################################################################## model and boxing

@numba.extending.register_model(JaggedArrayType)
class JaggedArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("starts", fe_type.startstype),
                   ("stops", fe_type.stopstype),
                   ("content", fe_type.contenttype),
                   ("iscompact", JaggedArrayNumba.NUMBA_BOOLTYPE)]
        super(JaggedArrayModel, self).__init__(dmm, fe_type, members)

numba.extending.make_attribute_wrapper(JaggedArrayType, "starts", "starts")
numba.extending.make_attribute_wrapper(JaggedArrayType, "stops", "stops")
numba.extending.make_attribute_wrapper(JaggedArrayType, "content", "content")
numba.extending.make_attribute_wrapper(JaggedArrayType, "iscompact", "iscompact")

@numba.extending.unbox(JaggedArrayType)
def _JaggedArray_unbox(typ, obj, c):
    starts_obj = c.pyapi.object_getattr_string(obj, "starts")
    stops_obj = c.pyapi.object_getattr_string(obj, "stops")
    content_obj = c.pyapi.object_getattr_string(obj, "content")
    iscompact_obj = c.pyapi.object_getattr_string(obj, "iscompact")

    array = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    array.starts = c.pyapi.to_native_value(typ.startstype, starts_obj).value
    array.stops = c.pyapi.to_native_value(typ.stopstype, stops_obj).value
    array.content = c.pyapi.to_native_value(typ.contenttype, content_obj).value
    array.iscompact = c.pyapi.to_native_value(JaggedArrayNumba.NUMBA_BOOLTYPE, iscompact_obj).value

    c.pyapi.decref(starts_obj)
    c.pyapi.decref(stops_obj)
    c.pyapi.decref(content_obj)
    c.pyapi.decref(iscompact_obj)

    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(array._getvalue(), is_error)

@numba.extending.box(JaggedArrayType)
def _JaggedArray_box(typ, val, c):
    array = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    starts_obj = c.pyapi.from_native_value(typ.startstype, array.starts, c.env_manager)
    stops_obj = c.pyapi.from_native_value(typ.stopstype, array.stops, c.env_manager)
    content_obj = c.pyapi.from_native_value(typ.contenttype, array.content, c.env_manager)

    cls = c.pyapi.unserialize(c.pyapi.serialize_object(typ.special))
    out = c.pyapi.call_function_objargs(cls, (starts_obj, stops_obj, content_obj))

    c.pyapi.decref(starts_obj)
    c.pyapi.decref(stops_obj)
    c.pyapi.decref(content_obj)

    return out

@numba.extending.type_callable(awkward.array.jagged.JaggedArray)
def _JaggedArray_type_init(context):
    def typer(startstype, stopstype, contenttype):
        if isinstance(startstype, numba.types.Array) and isinstance(stopstype, numba.types.Array) and isinstance(contenttype, (numba.types.Array, AwkwardArrayType)):
            return JaggedArrayType(startstype, stopstype, contenttype, special=awkward.array.jagged.JaggedArray)
    return typer

@numba.extending.type_callable(JaggedArrayNumba)
def _JaggedArray_type_init(context):
    def typer(startstype, stopstype, contenttype):
        if isinstance(startstype, numba.types.Array) and isinstance(stopstype, numba.types.Array) and isinstance(contenttype, (numba.types.Array, AwkwardArrayType)):
            return JaggedArrayType(startstype, stopstype, contenttype, special=JaggedArrayNumba)
    return typer

@numba.extending.lower_builtin(awkward.array.jagged.JaggedArray, numba.types.Array, numba.types.Array, numba.types.Array)
@numba.extending.lower_builtin(awkward.array.jagged.JaggedArray, numba.types.Array, numba.types.Array, AwkwardArrayType)
@numba.extending.lower_builtin(JaggedArrayNumba, numba.types.Array, numba.types.Array, numba.types.Array)
@numba.extending.lower_builtin(JaggedArrayNumba, numba.types.Array, numba.types.Array, AwkwardArrayType)
def _JaggedArray_init_array(context, builder, sig, args):
    startstype, stopstype, contenttype = sig.args
    startsval, stopsval, contentval = args

    if context.enable_nrt:
        context.nrt.incref(builder, startstype, startsval)
        context.nrt.incref(builder, stopstype, stopsval)
        context.nrt.incref(builder, contenttype, contentval)

    array = numba.cgutils.create_struct_proxy(sig.return_type)(context, builder)
    array.starts = startsval
    array.stops = stopsval
    array.content = contentval
    array.iscompact = context.get_constant(JaggedArrayNumba.NUMBA_BOOLTYPE, False)   # unless you reproduce that logic here or call out to Python
    return array._getvalue()

def _JaggedArray_new(array, starts, stops, content, iscompact):
    pass

@numba.extending.type_callable(_JaggedArray_new)
def _JaggedArray_type_new(context):
    def typer(arraytype, startstype, stopstype, contenttype, iscompacttype):
        return JaggedArrayType(startstype, stopstype, contenttype, special=arraytype.special)
    return typer

@numba.extending.lower_builtin(_JaggedArray_new, JaggedArrayType, numba.types.Array, numba.types.Array, numba.types.Array, numba.types.Boolean)
@numba.extending.lower_builtin(_JaggedArray_new, JaggedArrayType, numba.types.Array, numba.types.Array, AwkwardArrayType, numba.types.Boolean)
def _JaggedArray_lower_new(context, builder, sig, args):
    arraytype, startstype, stopstype, contenttype, iscompacttype = sig.args
    arrayval, startsval, stopsval, contentval, iscompactval = args

    if context.enable_nrt:
        context.nrt.incref(builder, startstype, startsval)
        context.nrt.incref(builder, stopstype, stopsval)
        context.nrt.incref(builder, contenttype, contentval)

    array = numba.cgutils.create_struct_proxy(sig.return_type)(context, builder)
    array.starts = startsval
    array.stops = stopsval
    array.content = contentval
    array.iscompact = iscompactval
    return array._getvalue()

######################################################################## utilities

def _check_startstop_contentlen(context, builder, starttype, startval, stoptype, stopval, contenttype, contentval):
    if isinstance(contenttype, numba.types.Array):
        contentlen = numba.targets.arrayobj.array_len(context, builder, numba.types.intp(contenttype), (contentval,))
    else:
        contentlen = _JaggedArray_lower_len(context, builder, numba.types.intp(contenttype), (contentval,))

    with builder.if_then(builder.or_(builder.or_(builder.icmp_signed("<", startval, context.get_constant(starttype, 0)),
                                                 builder.icmp_signed("<", stopval, context.get_constant(stoptype, 0))),
                                     builder.or_(builder.icmp_signed(">=", startval, contentlen),
                                                 builder.icmp_signed(">", stopval, contentlen))),
                         likely=False):
        context.call_conv.return_user_exc(builder, ValueError, ("JaggedArray.starts or JaggedArray.stops is beyond the range of JaggedArray.content",))

######################################################################## lowered len

@numba.extending.lower_builtin(len, JaggedArrayType)
def _JaggedArray_lower_len(context, builder, sig, args):
    arraytype, = sig.args
    arrayval, = args

    array = numba.cgutils.create_struct_proxy(arraytype)(context, builder, value=arrayval)

    return numba.targets.arrayobj.array_len(context, builder, numba.types.intp(arraytype.startstype), (array.starts,))

######################################################################## lowered getitem

@numba.extending.lower_builtin(operator.getitem, JaggedArrayType, numba.types.Integer)
def _JaggedArray_lower_getitem_integer(context, builder, sig, args, checkvalid=True):
    arraytype, wheretype = sig.args
    arrayval, whereval = args

    array = numba.cgutils.create_struct_proxy(arraytype)(context, builder, value=arrayval)

    startstype = arraytype.startstype
    stopstype = arraytype.stopstype
    contenttype = arraytype.contenttype

    if startstype.ndim == 1:
        start = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, startstype.dtype(startstype, wheretype), (array.starts, whereval))
        stop = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, stopstype.dtype(stopstype, wheretype), (array.stops, whereval))

        if checkvalid:
            _check_startstop_contentlen(context, builder, startstype.dtype, start, stopstype.dtype, stop, contenttype, array.content)

        if isinstance(contenttype, numba.types.Array):
            return numba.targets.arrayobj.getitem_arraynd_intp(context, builder, contenttype(contenttype, numba.types.slice2_type), (array.content, sliceval2(context, builder, start, stop)))
        else:
            return _JaggedArray_lower_getitem_slice(context, builder, contenttype(contenttype, numba.types.slice2_type), (array.content, sliceval2(context, builder, start, stop)))

    else:
        outstartstype = numba.types.Array(startstype.dtype, startstype.ndim - 1, startstype.layout)
        outstopstype = numba.types.Array(stopstype.dtype, stopstype.ndim - 1, stopstype.layout)

        starts = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, outstartstype(startstype, wheretype), (array.starts, whereval))
        stops = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, outstopstype(stopstype, wheretype), (array.stops, whereval))

        outtype = JaggedArrayType(outstartstype, outstopstype, contenttype, special=arraytype.special)
        return _JaggedArray_lower_new(context, builder, outtype(arraytype, outstartstype, outstopstype, contenttype, JaggedArrayNumba.NUMBA_BOOLTYPE), (arrayval, starts, stops, array.content, array.iscompact))

@numba.extending.lower_builtin(operator.getitem, JaggedArrayType, numba.types.SliceType)
def _JaggedArray_lower_getitem_slice(context, builder, sig, args):
    arraytype, wheretype = sig.args
    arrayval, whereval = args

    array = numba.cgutils.create_struct_proxy(arraytype)(context, builder, value=arrayval)

    startstype = arraytype.startstype
    starts = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, startstype(startstype, wheretype), (array.starts, whereval))

    stopstype = arraytype.stopstype
    stops = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, stopstype(stopstype, wheretype), (array.stops, whereval))

    slice = context.make_helper(builder, wheretype, value=whereval)
    iscompact = builder.and_(array.iscompact,
                             builder.or_(builder.icmp_signed("==", slice.step, context.get_constant(numba.types.intp, 1)),
                                         builder.icmp_signed("==", slice.step, context.get_constant(numba.types.intp, numba.types.intp.maxval))))

    contenttype = arraytype.contenttype
    return _JaggedArray_lower_new(context, builder, arraytype(arraytype, startstype, stopstype, contenttype, JaggedArrayNumba.NUMBA_BOOLTYPE), (arrayval, starts, stops, array.content, iscompact))

@numba.extending.lower_builtin(operator.getitem, JaggedArrayType, numba.types.Array)
def _JaggedArray_lower_getitem_array(context, builder, sig, args):
    arraytype, wheretype = sig.args
    arrayval, whereval = args

    array = numba.cgutils.create_struct_proxy(arraytype)(context, builder, value=arrayval)

    startstype = arraytype.startstype
    starts = numba.targets.arrayobj.fancy_getitem_array(context, builder, startstype(startstype, wheretype), (array.starts, whereval))

    stopstype = arraytype.stopstype
    stops = numba.targets.arrayobj.fancy_getitem_array(context, builder, stopstype(stopstype, wheretype), (array.stops, whereval))

    contenttype = arraytype.contenttype
    return _JaggedArray_lower_new(context, builder, arraytype(arraytype, startstype, stopstype, contenttype, JaggedArrayNumba.NUMBA_BOOLTYPE), (arrayval, starts, stops, array.content, context.get_constant(JaggedArrayNumba.NUMBA_BOOLTYPE, False)))

@numba.extending.lower_builtin(operator.getitem, JaggedArrayType, JaggedArrayType)
def _JaggedArray_lower_getitem_jaggedarray(context, builder, sig, args):
    arraytype, wheretype = sig.args

    if isinstance(wheretype, JaggedArrayType) and isinstance(wheretype.contenttype, JaggedArrayType):
        def getitem(array, where):
            return _JaggedArray_new(array, array.starts, array.stops, array.content[where.content], True)

    elif isinstance(wheretype, JaggedArrayType) and isinstance(wheretype.contenttype.dtype, numba.types.Boolean) and isinstance(arraytype.contenttype, numba.types.Array):
        def getitem(array, where):
            if len(array) != len(where):
                raise IndexError("jagged index must have the same (outer) length as the JaggedArray it indexes")
            offsets = numpy.empty(len(array.starts) + 1, numpy.int64)
            offsets[0] = 0
            content = numpy.empty(where.content.astype(numpy.int64).sum(), array.content.dtype)
            k = 0
            for i in range(len(array.starts)):
                length = array.stops[i] - array.starts[i]
                wherei = where[i]
                if len(wherei) > length:
                    raise IndexError("jagged index is out of bounds in JaggedArray")

                for j in range(len(wherei)):
                    if wherei[j]:
                        content[k] = array.content[array.starts[i] + j]
                        k += 1
                offsets[i + 1] = k

            starts = offsets[:-1]
            stops = offsets[1:]
            return _JaggedArray_new(array, starts, stops, content, True)

    elif isinstance(wheretype, JaggedArrayType) and isinstance(wheretype.contenttype.dtype, numba.types.Boolean):
        def getitem(array, where):
            if len(array) != len(where):
                raise IndexError("jagged index must have the same (outer) length as the JaggedArray it indexes")
            offsets = numpy.empty(len(array.starts) + 1, numpy.int64)
            offsets[0] = 0
            index = numpy.empty(len(where.content), numpy.int64)
            k = 0
            for i in range(len(array.starts)):
                length = array.stops[i] - array.starts[i]
                wherei = where[i]
                if len(wherei) > length:
                    raise IndexError("jagged index is out of bounds in JaggedArray")

                for j in range(len(wherei)):
                    if wherei[j]:
                        index[k] = array.starts[i] + j
                        k += 1
                offsets[i + 1] = k

            starts = offsets[:-1]
            stops = offsets[1:]
            return _JaggedArray_new(array, starts, stops, array.content[index[:k]], True)

    elif isinstance(wheretype, JaggedArrayType) and isinstance(wheretype.contenttype.dtype, numba.types.Integer) and isinstance(arraytype.contenttype, numba.types.Array):
        def getitem(array, where):
            if len(array) != len(where):
                raise IndexError("jagged index must have the same (outer) length as the JaggedArray it indexes")
            offsets = numpy.empty(len(array.starts) + 1, numpy.int64)
            offsets[0] = 0
            content = numpy.empty(len(where.content), array.content.dtype)
            k = 0
            for i in range(len(array.starts)):
                length = array.stops[i] - array.starts[i]
                wherei = where[i]

                for j in range(len(wherei)):
                    norm = wherei[j]
                    if norm < 0:
                        norm += length
                    if norm < 0 or norm >= length:
                        raise IndexError("jagged index is out of bounds in JaggedArray")
                    content[k] = array.content[array.starts[i] + norm]
                    k += 1
                offsets[i + 1] = k

            starts = offsets[:-1]
            stops = offsets[1:]
            return _JaggedArray_new(array, starts, stops, content, True)

    elif isinstance(wheretype, JaggedArrayType) and isinstance(wheretype.contenttype.dtype, numba.types.Integer):
        def getitem(array, where):
            if len(array) != len(where):
                raise IndexError("jagged index must have the same (outer) length as the JaggedArray it indexes")
            offsets = numpy.empty(len(array.starts) + 1, numpy.int64)
            offsets[0] = 0
            index = numpy.empty(len(where.content), numpy.int64)
            k = 0
            for i in range(len(array.starts)):
                length = array.stops[i] - array.starts[i]
                wherei = where[i]

                for j in range(len(wherei)):
                    norm = wherei[j]
                    if norm < 0:
                        norm += length
                    if norm < 0 or norm >= length:
                        raise IndexError("jagged index is out of bounds in JaggedArray")
                    index[k] = array.starts[i] + norm
                    k += 1
                offsets[i + 1] = k

            starts = offsets[:-1]
            stops = offsets[1:]
            return _JaggedArray_new(array, starts, stops, array.content[index], True)

    else:
        raise AssertionError(where)

    return context.compile_internal(builder, getitem, sig, args)

@numba.extending.lower_builtin(operator.getitem, JaggedArrayType, numba.types.BaseTuple)
def _JaggedArray_lower_getitem_enter(context, builder, sig, args):
    arraytype, wheretype = sig.args
    arrayval, whereval = args

    if len(wheretype.types) == 1:
        if isinstance(wheretype.types[0], numba.types.Integer):
            getitem = _JaggedArray_lower_getitem_integer
        elif isinstance(wheretype.types[0], numba.types.SliceType):
            getitem = _JaggedArray_lower_getitem_slice
        elif isinstance(wheretype.types[0], numba.types.Array):
            getitem = _JaggedArray_lower_getitem_array
        return getitem(context, builder, sig.return_type(arraytype, wheretype.types[0]), (arrayval, builder.extract_value(whereval, 0)))

    if any(isinstance(x, numba.types.Array) for x in wheretype.types):
        arraylen = numba.cgutils.alloca_once_value(builder, context.get_constant(JaggedArrayNumba.NUMBA_INDEXTYPE, 0))
        for i, whereitemtype in enumerate(wheretype.types):
            if isinstance(whereitemtype, numba.types.Array):
                if isinstance(whereitemtype.dtype, numba.types.Boolean):
                    enter_arraylen = lambda whereitem, arraylen: max(arraylen, whereitem.astype(numpy.int64).sum())
                else:
                    enter_arraylen = lambda whereitem, arraylen: max(arraylen, len(whereitem))

                whereitemval = builder.extract_value(whereval, i)
                arraylenval = context.compile_internal(builder, enter_arraylen, JaggedArrayNumba.NUMBA_INDEXTYPE(whereitemtype, JaggedArrayNumba.NUMBA_INDEXTYPE), (whereitemval, builder.load(arraylen)))
                builder.store(arraylenval, arraylen)

        arraylenval = builder.load(arraylen)
        newwheretype = []
        newwherevals = []
        for i, old in enumerate(wheretype.types):
            if isinstance(old, numba.types.Array) and isinstance(old.dtype, numba.types.Boolean):
                toadvanced = lambda whereitem, arraylen: numpy.where(whereitem)[0]
            elif isinstance(old, numba.types.Array):
                toadvanced = lambda whereitem, arraylen: numpy.full(arraylen, whereitem[0], numpy.int64) if len(whereitem) == 1 else whereitem
            elif isinstance(old, numba.types.Integer):
                toadvanced = lambda whereitem, arraylen: numpy.full(arraylen, whereitem, numpy.int64)
            else:
                toadvanced = None

            whereitemval = builder.extract_value(whereval, i)
            if toadvanced is None:
                newwheretype.append(old)
                newwherevals.append(whereitemval)
            else:
                new = numba.types.Array(JaggedArrayNumba.NUMBA_INDEXTYPE, 1, "C") if isinstance(old, (numba.types.Array, numba.types.Integer)) else old
                newwheretype.append(new)
                newwherevals.append(context.compile_internal(builder, toadvanced, new(old, JaggedArrayNumba.NUMBA_INDEXTYPE), (whereitemval, arraylenval)))

        wheretype = numba.types.Tuple(tuple(newwheretype))
        whereval = context.make_tuple(builder, wheretype, tuple(newwherevals))

    def fake1(array, where):
        return _JaggedArray_getitem_next(awkward.array.jagged.JaggedArray(numpy.array([0], numpy.int64), numpy.array([len(array)], numpy.int64), array), where, None)[0]

    def fake2(array, where):
        out = _JaggedArray_getitem_next(awkward.array.jagged.JaggedArray(numpy.array([0], numpy.int64), numpy.array([len(array)], numpy.int64), array), where, None)
        return out.content[out.starts[0]:out.stops[-1]]

    fake = fake1 if all(isinstance(x, numba.types.Integer) for x in wheretype.types) else fake2
    return context.compile_internal(builder, fake, sig.return_type(arraytype, wheretype), (arrayval, whereval))

@numba.generated_jit(nopython=True)
def _JaggedArray_getitem_enter_toadvanced(whereitem, arraylen):
    if isinstance(whereitem, numba.types.Array) and isinstance(whereitem.dtype, numba.types.Boolean):
        return lambda whereitem, arraylen: numpy.nonzero(whereitem)[0]
    elif isinstance(whereitem, numba.types.Array):
        return lambda whereitem, arraylen: numpy.full(arraylen, whereitem[0], numpy.int64) if len(whereitem) == 1 else whereitem
    elif isinstance(whereitem, numba.types.Integer):
        return lambda whereitem, arraylen: numpy.full(arraylen, whereitem, numpy.int64)
    else:
        return lambda whereitem, arraylen: whereitem

@numba.extending.lower_builtin(_JaggedArray_getitem_next, numba.types.Array, numba.types.BaseTuple, numba.types.NoneType)
@numba.extending.lower_builtin(_JaggedArray_getitem_next, JaggedArrayType, numba.types.BaseTuple, numba.types.NoneType)
@numba.extending.lower_builtin(_JaggedArray_getitem_next, numba.types.Array, numba.types.BaseTuple, numba.types.Array)
@numba.extending.lower_builtin(_JaggedArray_getitem_next, JaggedArrayType, numba.types.BaseTuple, numba.types.Array)
def _JaggedArray_lower_getitem_next(context, builder, sig, args):
    arraytype, wheretype, advancedtype = sig.args
    arrayval, whereval, advancedval = args

    if len(wheretype.types) == 0:
        if context.enable_nrt:
            context.nrt.incref(builder, arraytype, arrayval)
        return arrayval

    headtype = wheretype.types[0]
    tailtype = numba.types.Tuple(wheretype.types[1:])
    headval = numba.targets.tupleobj.static_getitem_tuple(context, builder, headtype(wheretype, JaggedArrayNumba.NUMBA_INDEXTYPE), (whereval, 0))
    tailval = numba.targets.tupleobj.static_getitem_tuple(context, builder, tailtype(wheretype, numba.types.slice2_type), (whereval, slice(1, None)))

    if isinstance(headtype, numba.types.Integer):
        if isinstance(arraytype.contenttype, numba.types.Array):
            def getitem(array, head, tail, advanced):
                content = numpy.empty(len(array.starts), array.content.dtype)
                for i in range(len(array.starts)):
                    norm = head
                    if norm < 0:
                        norm += array.stops[i] - array.starts[i]
                    j = array.starts[i] + norm
                    if j >= array.stops[i]:
                        raise ValueError("integer index is beyond the range of one of the JaggedArray starts/stops pairs")
                    content[i] = array.content[j]
                return _JaggedArray_getitem_next(content, tail, advanced)

        else:
            def getitem(array, head, tail, advanced):
                index = numpy.empty(len(array.starts), numpy.int64)
                for i in range(len(array.starts)):
                    norm = head
                    if norm < 0:
                        norm += array.stops[i] - array.starts[i]
                    j = array.starts[i] + norm
                    if j >= array.stops[i]:
                        raise ValueError("integer index is beyond the range of one of the JaggedArray starts/stops pairs")
                    index[i] = j
                return _JaggedArray_getitem_next(array.content[index], tail, advanced)

    elif isinstance(headtype, numba.types.SliceType) and headtype.members == 2 and advancedtype == NOTADVANCED and not any(isinstance(x, numba.types.Array) for x in tailtype):
        intp_maxval = numba.types.intp.maxval

        def getitem(array, head, tail, advanced):
            if (head.start == 0 or head.start == intp_maxval) and head.stop == intp_maxval:
                next = _JaggedArray_getitem_next(array.content, tail, advanced)
                return _JaggedArray_new(array, array.starts, array.stops, next, array.iscompact)

            starts = numpy.empty(len(array.starts), numpy.int64)
            stops = numpy.empty(len(array.starts), numpy.int64)
            for i in range(len(array.starts)):
                length = array.stops[i] - array.starts[i]
                a = head.start
                b = head.stop

                if a == intp_maxval:
                    a = 0
                elif a < 0:
                    a += length
                if b == intp_maxval:
                    b = length
                elif b < 0:
                    b += length

                if b <= a:
                    a = 0
                    b = 0
                if a < 0:
                    a = 0
                elif a > length:
                    a = length
                if b < 0:
                    b = 0
                elif b > length:
                    b = length

                starts[i] = array.starts[i] + a
                stops[i] = array.starts[i] + b

            next = _JaggedArray_getitem_next(array.content, tail, advanced)
            return _JaggedArray_new(array, starts, stops, next, False)

    elif isinstance(headtype, numba.types.SliceType):
        intp_maxval = numba.types.intp.maxval

        def getitem(array, head, tail, advanced):
            if head.step == 0:
                raise ValueError("slice step cannot be zero")

            offsets = numpy.empty(len(array.starts) + 1, numpy.int64)
            offsets[0] = 0
            index = numpy.empty(len(array.content), numpy.int64)
            k = 0
            for i in range(len(array.starts)):
                length = array.stops[i] - array.starts[i]
                a = head.start
                b = head.stop
                c = head.step
                if c == intp_maxval:
                    c = 1

                if a == intp_maxval and c > 0:
                    a = 0
                elif a == intp_maxval:
                    a = length - 1
                elif a < 0:
                    a += length

                if b == intp_maxval and c > 0:
                    b = length
                elif b == intp_maxval:
                    b = -1
                elif b < 0:
                    b += length

                if c > 0:
                    if b <= a:
                        a = 0
                        b = 0
                    if a < 0:
                        a = 0
                    elif a > length:
                        a = length
                    if b < 0:
                        b = 0
                    elif b > length:
                        b = length
                else:
                    if a <= b:
                        a = 0
                        b = 0
                    if a < -1:
                        a = -1
                    elif a >= length:
                        a = length - 1
                    if b < -1:
                        b = -1
                    elif b >= length:
                        b = length - 1

                for j in range(a, b, c):
                    index[k] = array.starts[i] + j
                    k += 1
                offsets[i + 1] = k

            starts = offsets[:-1]
            stops = offsets[1:]
            next = _JaggedArray_getitem_next(array.content[index[:k]], tail, _spread_advanced(starts, stops, advanced))
            return _JaggedArray_new(array, starts, stops, next, True)

    elif isinstance(headtype, numba.types.Array):
        if advancedtype == NOTADVANCED:
            def getitem(array, head, tail, advanced):
                offsets = numpy.empty(len(array.starts) + 1, numpy.int64)
                offsets[0] = 0
                index = numpy.empty(len(head)*len(array.starts), numpy.int64)
                nextadvanced = numpy.empty(len(index), numpy.int64)

                k = 0
                for i in range(len(array.starts)):
                    length = array.stops[i] - array.starts[i]

                    for j in range(len(head)):
                        norm = head[j]
                        if norm < 0:
                            norm += length
                        if norm < 0 or norm >= length:
                            raise IndexError("advanced index is out of bounds in JaggedArray")
                        index[k] = array.starts[i] + norm
                        nextadvanced[k] = j
                        k += 1
                    offsets[i + 1] = k

                starts = offsets[:-1]
                stops = offsets[1:]
                next = _JaggedArray_getitem_next(array.content[index], tail, nextadvanced)
                return _JaggedArray_new(array, starts, stops, next, True)

        else:
            def getitem(array, head, tail, advanced):
                index = numpy.empty(len(array.starts), numpy.int64)
                nextadvanced = numpy.empty(len(index), numpy.int64)

                for i in range(len(advanced)):
                    length = array.stops[i] - array.starts[i]
                    if advanced[i] >= len(head):
                        raise IndexError("advanced index lengths do not match")
                    norm = head[advanced[i]]
                    if norm < 0:
                        norm += length
                    if norm < 0 or norm >= length:
                        raise IndexError("advanced index is out of bounds in JaggedArray")
                    index[i] = array.starts[i] + norm
                    nextadvanced[i] = i

                next = _JaggedArray_getitem_next(array.content[index], tail, nextadvanced)
                return next

    else:
        raise AssertionError(head)

    sig = sig.return_type(arraytype, headtype, tailtype, advancedtype)
    args = (arrayval, headval, tailval, advancedval)
    return context.compile_internal(builder, getitem, sig, args)

@numba.generated_jit(nopython=True)
def _spread_advanced(starts, stops, advanced):
    if isinstance(advanced, numba.types.NoneType):
        return lambda starts, stops, advanced: advanced
    else:
        def impl(starts, stops, advanced):
            counts = stops - starts
            nextadvanced = numpy.empty(counts.sum(), numpy.int64)
            k = 0
            for i in range(len(counts)):
                length = counts[i]
                nextadvanced[k : k + counts[i]] = advanced[i]
                k += length
            return nextadvanced
        return impl

######################################################################## overloading ufuncs

### See numba.typing.npydecl for typing, and then maybe lower as usual?

### @numba.extending.lower_builtin(numpy.add, JaggedArrayType, JaggedArrayType)
### @numba.extending.lower_builtin(numpy.add, JaggedArrayType, numba.types.Array)
### @numba.extending.lower_builtin(numpy.add, numba.types.Array, JaggedArrayType)
### ???






######################################################################## other lowered methods in Numba, including reducers

@numba.typing.templates.infer_getattr
class _JaggedArrayType_type_methods(numba.typing.templates.AttributeTemplate):
    key = JaggedArrayType

    def resolve_reducer(self, arraytype, args, kwargs, endtype):
        if len(args) == 0 and len(kwargs) == 0:
            if isinstance(arraytype, JaggedArrayType) and isinstance(arraytype.contenttype, JaggedArrayType):
                contenttype = self.resolve_reducer(arraytype.contenttype, args, kwargs, endtype)
                return JaggedArrayType(arraytype.startstype, arraytype.stopstype, contenttype.return_type, special=JaggedArrayNumba)()
            elif isinstance(arraytype, JaggedArrayType) and isinstance(arraytype.contenttype, numba.types.Array):
                if endtype is None:
                    endtype = arraytype.contenttype.dtype
                return numba.types.Array(endtype, 1, arraytype.contenttype.layout)()

    @numba.typing.templates.bound_function("any")
    def resolve_any(self, arraytype, args, kwargs):
        return self.resolve_reducer(arraytype, args, kwargs, JaggedArrayNumba.NUMBA_BOOLTYPE)

    @numba.typing.templates.bound_function("all")
    def resolve_all(self, arraytype, args, kwargs):
        return self.resolve_reducer(arraytype, args, kwargs, JaggedArrayNumba.NUMBA_BOOLTYPE)

    @numba.typing.templates.bound_function("count")
    def resolve_count(self, arraytype, args, kwargs):
        return self.resolve_reducer(arraytype, args, kwargs, JaggedArrayNumba.NUMBA_INDEXTYPE)

    @numba.typing.templates.bound_function("count_nonzero")
    def resolve_count_nonzero(self, arraytype, args, kwargs):
        return self.resolve_reducer(arraytype, args, kwargs, JaggedArrayNumba.NUMBA_INDEXTYPE)

    @numba.typing.templates.bound_function("sum")
    def resolve_sum(self, arraytype, args, kwargs):
        return self.resolve_reducer(arraytype, args, kwargs, None)

    @numba.typing.templates.bound_function("prod")
    def resolve_prod(self, arraytype, args, kwargs):
        return self.resolve_reducer(arraytype, args, kwargs, None)

    @numba.typing.templates.bound_function("min")
    def resolve_min(self, arraytype, args, kwargs):
        return self.resolve_reducer(arraytype, args, kwargs, None)

    @numba.typing.templates.bound_function("max")
    def resolve_max(self, arraytype, args, kwargs):
        return self.resolve_reducer(arraytype, args, kwargs, None)

#     @numba.typing.templates.bound_function("structure1d")
#     def resolve_structure1d(self, arraytype, args, kwargs):
#         if len(args) == 0:
#             return arraytype()
#         elif len(args) == 1 and isinstance(args[0], numba.types.NoneType):
#             return arraytype(args[0])
#         elif len(args) == 1 and isinstance(args[0], numba.types.Integer):
#             return arraytype(args[0])

# @numba.extending.lower_builtin("structure1d", JaggedArrayType)
# def _JaggedArray_lower_structure1d_1(context, builder, sig, args):
#     arraytype, = sig.args
#     arrayval, = args
#     return _JaggedArray_lower_structure1d_3(context, builder, sig.return_type(arraytype, JaggedArrayNumba.NUMBA_INDEXTYPE), (arrayval, context.get_constant(JaggedArrayNumba.NUMBA_INDEXTYPE, -1),))

def _JaggedArray_lower_reduce_descend(which, context, builder, sig, args):
    arraytype, = sig.args
    arrayval, = args
    array = numba.cgutils.create_struct_proxy(arraytype)(context, builder, value=arrayval)
    content = which(context, builder, sig.return_type.contenttype(arraytype.contenttype), (array.content,))
    return _JaggedArray_lower_new(context, builder, sig.return_type(arraytype, arraytype.startstype, arraytype.stopstype, sig.return_type.contenttype, JaggedArrayNumba.NUMBA_BOOLTYPE), (arrayval, array.starts, array.stops, content, array.iscompact))

@numba.extending.lower_builtin("any", JaggedArrayType)
def _JaggedArray_lower_any(context, builder, sig, args):
    if isinstance(sig.args[0].contenttype, JaggedArrayType):
        return _JaggedArray_lower_reduce_descend(_JaggedArray_lower_any, context, builder, sig, args)
    def run(array):
        out = numpy.empty(array.starts.shape, numpy.bool_)
        flatout = out.reshape(-1)
        flatstarts = array.starts.reshape(-1)
        flatstops = array.stops.reshape(-1)
        for i in range(len(flatstarts)):
            flatout[i] = False
            for j in range(flatstarts[i], flatstops[i]):
                if not math.isnan(array.content[j]) and array.content[j] != 0:
                    flatout[i] = True
                    break
        return out
    return context.compile_internal(builder, run, sig, args)

@numba.extending.lower_builtin("all", JaggedArrayType)
def _JaggedArray_lower_all(context, builder, sig, args):
    if isinstance(sig.args[0].contenttype, JaggedArrayType):
        return _JaggedArray_lower_reduce_descend(_JaggedArray_lower_all, context, builder, sig, args)
    def run(array):
        out = numpy.empty(array.starts.shape, numpy.bool_)
        flatout = out.reshape(-1)
        flatstarts = array.starts.reshape(-1)
        flatstops = array.stops.reshape(-1)
        for i in range(len(flatstarts)):
            flatout[i] = True
            for j in range(flatstarts[i], flatstops[i]):
                if not math.isnan(array.content[j]) and array.content[j] == 0:
                    flatout[i] = False
                    break
        return out
    return context.compile_internal(builder, run, sig, args)

@numba.extending.lower_builtin("count", JaggedArrayType)
def _JaggedArray_lower_count(context, builder, sig, args):
    if isinstance(sig.args[0].contenttype, JaggedArrayType):
        return _JaggedArray_lower_reduce_descend(_JaggedArray_lower_count, context, builder, sig, args)
    def run(array):
        out = numpy.empty(array.starts.shape, numpy.int64)
        flatout = out.reshape(-1)
        flatstarts = array.starts.reshape(-1)
        flatstops = array.stops.reshape(-1)
        for i in range(len(flatstarts)):
            flatout[i] = 0
            for j in range(flatstarts[i], flatstops[i]):
                if not math.isnan(array.content[j]):
                    flatout[i] += 1
        return out
    return context.compile_internal(builder, run, sig, args)

@numba.extending.lower_builtin("count_nonzero", JaggedArrayType)
def _JaggedArray_lower_count_nonzero(context, builder, sig, args):
    if isinstance(sig.args[0].contenttype, JaggedArrayType):
        return _JaggedArray_lower_reduce_descend(_JaggedArray_lower_count_nonzero, context, builder, sig, args)
    def run(array):
        out = numpy.empty(array.starts.shape, numpy.int64)
        flatout = out.reshape(-1)
        flatstarts = array.starts.reshape(-1)
        flatstops = array.stops.reshape(-1)
        for i in range(len(flatstarts)):
            flatout[i] = 0
            for j in range(flatstarts[i], flatstops[i]):
                if not math.isnan(array.content[j]) and array.content[j] != 0:
                    flatout[i] += 1
        return out
    return context.compile_internal(builder, run, sig, args)

@numba.extending.lower_builtin("sum", JaggedArrayType)
def _JaggedArray_lower_sum(context, builder, sig, args):
    if isinstance(sig.args[0].contenttype, JaggedArrayType):
        return _JaggedArray_lower_reduce_descend(_JaggedArray_lower_sum, context, builder, sig, args)
    def run(array):
        out = numpy.empty(array.starts.shape, array.content.dtype)
        flatout = out.reshape(-1)
        flatstarts = array.starts.reshape(-1)
        flatstops = array.stops.reshape(-1)
        for i in range(len(flatstarts)):
            flatout[i] = 0
            for j in range(flatstarts[i], flatstops[i]):
                if not math.isnan(array.content[j]):
                    flatout[i] += array.content[j]
        return out
    return context.compile_internal(builder, run, sig, args)

@numba.extending.lower_builtin("prod", JaggedArrayType)
def _JaggedArray_lower_prod(context, builder, sig, args):
    if isinstance(sig.args[0].contenttype, JaggedArrayType):
        return _JaggedArray_lower_reduce_descend(_JaggedArray_lower_prod, context, builder, sig, args)
    def run(array):
        out = numpy.empty(array.starts.shape, array.content.dtype)
        flatout = out.reshape(-1)
        flatstarts = array.starts.reshape(-1)
        flatstops = array.stops.reshape(-1)
        for i in range(len(flatstarts)):
            flatout[i] = 1
            for j in range(flatstarts[i], flatstops[i]):
                if not math.isnan(array.content[j]):
                    flatout[i] *= array.content[j]
        return out
    return context.compile_internal(builder, run, sig, args)

@numba.extending.lower_builtin("min", JaggedArrayType)
def _JaggedArray_lower_min(context, builder, sig, args):
    if isinstance(sig.args[0].contenttype, JaggedArrayType):
        return _JaggedArray_lower_reduce_descend(_JaggedArray_lower_min, context, builder, sig, args)
    def run(array, identity):
        out = numpy.empty(array.starts.shape, array.content.dtype)
        flatout = out.reshape(-1)
        flatstarts = array.starts.reshape(-1)
        flatstops = array.stops.reshape(-1)
        for i in range(len(flatstarts)):
            flatout[i] = identity
            for j in range(flatstarts[i], flatstops[i]):
                if not math.isnan(array.content[j]) and array.content[j] < flatout[i]:
                    flatout[i] = array.content[j]
        return out
    datatype = sig.args[0].contenttype.dtype
    if isinstance(datatype, numba.types.Boolean):
        identity = True
    elif isinstance(datatype, numba.types.Integer):
        identity = datatype.maxval
    else:
        identity = numpy.inf
        datatype = numba.types.float64
    return context.compile_internal(builder, run, sig.return_type(sig.args[0], datatype), (args[0], context.get_constant(datatype, identity)))

@numba.extending.lower_builtin("max", JaggedArrayType)
def _JaggedArray_lower_max(context, builder, sig, args):
    if isinstance(sig.args[0].contenttype, JaggedArrayType):
        return _JaggedArray_lower_reduce_descend(_JaggedArray_lower_max, context, builder, sig, args)
    def run(array, identity):
        out = numpy.empty(array.starts.shape, array.content.dtype)
        flatout = out.reshape(-1)
        flatstarts = array.starts.reshape(-1)
        flatstops = array.stops.reshape(-1)
        for i in range(len(flatstarts)):
            flatout[i] = identity
            for j in range(flatstarts[i], flatstops[i]):
                if not math.isnan(array.content[j]) and array.content[j] > flatout[i]:
                    flatout[i] = array.content[j]
        return out
    datatype = sig.args[0].contenttype.dtype
    if isinstance(datatype, numba.types.Boolean):
        identity = False
    elif isinstance(datatype, numba.types.Integer):
        identity = datatype.minval
    else:
        identity = -numpy.inf
        datatype = numba.types.float64
    return context.compile_internal(builder, run, sig.return_type(sig.args[0], datatype), (args[0], context.get_constant(datatype, identity)))

@numba.extending.overload_attribute(JaggedArrayType, "offsets")
def _JaggedArray_offsets(arraytype):
    if arraytype.startstype.ndim == 1:
        def impl(array):
            offsets = numpy.empty(len(array.starts) + 1, numpy.int64)
            if len(array.starts) == 0:
                offsets[0] = 0
                return offsets
            offsets = array.starts[0]
            for i in range(1, len(array.starts)):
                if array.starts[i + 1] != array.stops[i]:
                    raise ValueError("starts and stops are not compatible with a single offsets array; call jagged.compact() first")
                offsets[i] = array.stops[i]
            return offsets
        return impl
    else:
        raise TypeError("len(starts.shape) must be 1 to compute offsets; call jagged.structure1d() first")

@numba.extending.overload_attribute(JaggedArrayType, "counts")
def _JaggedArray_counts(arraytype):
    def impl(array):
        return array.stops - array.starts
    return impl

@numba.extending.overload_attribute(JaggedArrayType, "parents")
def _JaggedArray_parents(arraytype):
    def impl(array):
        out = numpy.full(array.stops.max(), -1, numpy.int64)
        for i in range(len(array.starts)):
            out[array.starts[i]:array.stops[i]] = i
        return out
    return impl

@numba.extending.overload_attribute(JaggedArrayType, "index")
def _JaggedArray_index(arraytype):
    def impl(array):
        out = numpy.full(array.stops.max(), -1, numpy.int64)
        for i in range(len(array.starts)):
            for j in range(array.starts[i], array.stops[i]):
                out[j] = j - array.starts[i]
        return JaggedArray(array.starts, array.stops, out, array.iscompact)

@numba.extending.overload_method(JaggedArrayType, "regular")
def _JaggedArray_regular(arraytype):
    if not isinstance(arraytype.contenttype, numba.types.Array):
        raise TypeError("JaggedArray.content must be a Numpy array to use jagged.regular()")
    def impl(array):
        count = -1
        for i in range(len(array.starts)):
            if count == -1:
                count = array.stops[i] - array.starts[i]
            elif count != array.stops[i] - array.starts[i]:
                raise ValueError("JaggedArray is not regular: different elements have different counts")
        return array.content.reshape(array.starts.shape + (count,))
    return impl

@numba.extending.overload_method(JaggedArrayType, "compact")
def _JaggedArray_compact(arraytype):
    if isinstance(arraytype, JaggedArrayType) and isinstance(arraytype.contenttype, AwkwardArrayType):
        def impl(array):
            if array.iscompact:
                return array
            if len(array.starts) == 0:
                return _JaggedArray_new(array, array.starts, array.stops[0:0], array.content[0:0], True)

            if array.starts.shape != array.stops.shape:
                raise ValueError("JaggedArray.starts must have the same shape as JaggedArray.stops")
            flatstarts = array.starts.reshape(-1)
            flatstops = array.stops.reshape(-1)

            offsets = numpy.empty(len(flatstarts) + 1, flatstarts.dtype)
            offsets[0] = 0
            for i in range(len(flatstarts)):
                count = flatstops[i] - flatstarts[i]
                if count < 0:
                    raise ValueError("JaggedArray.stops[i] must be greater than or equal to JaggedArray.starts[i] for all i")
                offsets[i + 1] = offsets[i] + count

            index = numpy.empty(offsets[-1], numpy.int64)
            k = 0
            for i in range(len(flatstarts)):
                for j in range(flatstarts[i], flatstops[i]):
                    index[k] = j
                    k += 1

            starts = offsets[:-1].reshape(array.starts.shape)
            stops = offsets[1:].reshape(array.starts.shape)    # intentional
            content = array.content[index]
            return _JaggedArray_new(array, starts, stops, content, True)

        return impl

    elif isinstance(arraytype, JaggedArrayType) and isinstance(arraytype.contenttype, numba.types.Array):
        def impl(array):
            if array.iscompact:
                return array
            if len(array.starts) == 0:
                return _JaggedArray_new(array, array.starts, array.stops[0:0], array.content[0:0], True)

            if array.starts.shape != array.stops.shape:
                raise ValueError("JaggedArray.starts must have the same shape as JaggedArray.stops")
            flatstarts = array.starts.ravel()
            flatstops = array.stops.ravel()

            offsets = numpy.empty(len(flatstarts) + 1, flatstarts.dtype)
            offsets[0] = 0
            for i in range(len(flatstarts)):
                count = flatstops[i] - flatstarts[i]
                if count < 0:
                    raise ValueError("JaggedArray.stops[i] must be greater than or equal to JaggedArray.starts[i] for all i")
                offsets[i + 1] = offsets[i] + count

            content = numpy.empty(offsets[-1], array.content.dtype)
            k = 0
            for i in range(len(flatstarts)):
                for j in range(flatstarts[i], flatstops[i]):
                    content[k] = array.content[j]
                    k += 1

            starts = offsets[:-1].reshape(array.starts.shape)
            stops = offsets[1:].reshape(array.starts.shape)    # intentional
            return _JaggedArray_new(array, starts, stops, content, True)

        return impl

@numba.extending.overload_method(JaggedArrayType, "flatten")
def _JaggedArray_flatten(arraytype):
    if isinstance(arraytype, JaggedArrayType):
        def impl(array):
            if len(array.starts) == 0:
                return array.content[0:0]
            else:
                a = array.compact()
                return a.content[a.starts[0]:a.stops[-1]]
        return impl

@numba.extending.overload_method(JaggedArrayType, "argmin")
def _JaggedArray_argmin(arraytype):
    if isinstance(arraytype, JaggedArrayType) and isinstance(arraytype.contenttype, AwkwardArrayType):
        def impl(array):
            return _JaggedArray_new(array, array.starts, array.stops, _JaggedArray_argmin(array.content), array.iscompact)
        return impl
    elif isinstance(arraytype, JaggedArrayType):
        def impl(array):
            return _JaggedArray_argminmax(array, True)
        return impl

@numba.extending.overload_method(JaggedArrayType, "argmax")
def _JaggedArray_argmax(arraytype):
    if isinstance(arraytype, JaggedArrayType) and isinstance(arraytype.contenttype, AwkwardArrayType):
        def impl(array):
            return _JaggedArray_new(array, array.starts, array.stops, _JaggedArray_argmax(array.content), array.iscompact)
        return impl
    elif isinstance(arraytype, JaggedArrayType):
        def impl(array):
            return _JaggedArray_argminmax(array, False)
        return impl
    
@numba.njit
def _JaggedArray_argminmax(array, ismin):
    if len(array.content.shape) != 1:
        raise NotImplementedError("content is not one-dimensional")

    flatstarts = array.starts.reshape(-1)
    flatstops = array.stops.reshape(-1)

    offsets = numpy.empty(len(flatstarts) + 1, numpy.int64)
    offsets[0] = 0
    for i in range(len(flatstarts)):
        if flatstarts[i] == flatstops[i]:
            offsets[i + 1] = offsets[i]
        else:
            offsets[i + 1] = offsets[i] + 1

    starts = offsets[:-1].reshape(array.starts.shape)
    stops = offsets[1:].reshape(array.stops.shape)

    output = numpy.empty(offsets[-1], dtype=numpy.int64)

    if ismin:
        k = 0
        for i in range(len(flatstarts)):
            if flatstops[i] != flatstarts[i]:
                best = array.content[flatstarts[i]]
                bestj = 0
                for j in range(flatstarts[i] + 1, flatstops[i]):
                    if array.content[j] < best:
                        best = array.content[j]
                        bestj = j - flatstarts[i]
                output[k] = bestj
                k += 1

    else:
        k = 0
        for i in range(len(flatstarts)):
            if flatstops[i] != flatstarts[i]:
                best = array.content[flatstarts[i]]
                bestj = 0
                for j in range(flatstarts[i] + 1, flatstops[i]):
                    if array.content[j] > best:
                        best = array.content[j]
                        bestj = j - flatstarts[i]
                output[k] = bestj
                k += 1

    return _JaggedArray_new(array, starts, stops, output, True)

@numba.njit
def _JaggedArray_concatenate_njit(arrays, axis):
    return _JaggedArray_concatenate(arrays, axis)

def _JaggedArray_concatenate(arrays, axis):
    pass

@numba.extending.type_callable(_JaggedArray_concatenate)
def _JaggedArray_concatenate(context):
    return None

@numba.extending.lower_builtin(_JaggedArray_concatenate, numba.types.Tuple, numba.types.Integer)
def _JaggedArray_lower_concatenate(context, builder, sig, args):
    raise NotImplementedError
