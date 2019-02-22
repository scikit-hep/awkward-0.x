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

import operator

import numpy
import numba
import numba.typing.arraydecl

import awkward.array.jagged
from .base import NumbaMethods
from .base import AwkwardArrayType
from .base import clsrepr
from .base import ISADVANCED
from .base import NOTADVANCED
from .base import sliceval2
from .base import sliceval3
from .base import inline

######################################################################## optimized functions (hidden)

@numba.njit
def _offsets2parents_fill(offsets, parents):
    j = 0
    k = -1
    for i in offsets:
        while j < i:
            parents[j] = k
            j += 1
        k += 1

@numba.njit
def _argminmax_fillmin(starts, stops, content, output):
    k = 0
    for i in range(len(starts)):
        if stops[i] != starts[i]:
            best = content[starts[i]]
            bestj = 0
            for j in range(starts[i] + 1, stops[i]):
                if content[j] < best:
                    best = content[j]
                    bestj = j - starts[i]
            output[k] = bestj
            k += 1

@numba.njit
def _argminmax_fillmax(starts, stops, content, output):
    k = 0
    for i in range(len(starts)):
        if stops[i] != starts[i]:
            best = content[starts[i]]
            bestj = 0
            for j in range(starts[i] + 1, stops[i]):
                if content[j] > best:
                    best = content[j]
                    bestj = j - starts[i]
            output[k] = bestj
            k += 1

######################################################################## Numba-accelerated interface

class JaggedArrayNumba(NumbaMethods, awkward.array.jagged.JaggedArray):
    @classmethod
    def offsets2parents(cls, offsets):
        if len(offsets) == 0:
            raise ValueError("offsets must have at least one element")
        parents = cls.numpy.empty(offsets[-1], dtype=cls.JaggedArray.fget(None).INDEXTYPE)
        _offsets2parents_fill(offsets, parents)
        return parents

    def _argminmax(self, ismin):
        if len(self._starts) == len(self._stops) == 0:
            return self.copy()

        if len(self._content.shape) != 1:
            raise ValueError("cannot compute arg{0} because content is not one-dimensional".format("min" if ismin else "max"))

        # subarray with counts > 0 --> counts = 1
        counts = (self.counts != 0).astype(self.INDEXTYPE).reshape(-1)
        # offsets for these 0 or 1 counts (involves a cumsum)
        offsets = self.counts2offsets(counts)
        # starts and stops derived from offsets and reshaped to original starts and stops (see specification)
        starts, stops = offsets[:-1], offsets[1:]

        starts = starts.reshape(self._starts.shape[:-1] + (-1,))
        stops = stops.reshape(self._stops.shape[:-1] + (-1,))

        # content to fit the new offsets
        content = awkward.util.numpy.empty(offsets[-1], dtype=self.INDEXTYPE)

        # fill the new content
        if ismin:
            _argminmax_fillmin(self._starts.reshape(-1), self._stops.reshape(-1), self._content, content)
        else:
            _argminmax_fillmax(self._starts.reshape(-1), self._stops.reshape(-1), self._content, content)

        return self.copy(starts=starts, stops=stops, content=content)

    def _argminmax_general(self, ismin):
        raise RuntimeError("helper function not needed in JaggedArrayNumba")

######################################################################## register types in Numba

class JaggedArrayType(AwkwardArrayType):
    def __init__(self, startstype, stopstype, contenttype, special=awkward.array.jagged.JaggedArray):
        super(JaggedArrayType, self).__init__(name="JaggedArrayType({0}, {1}, {2}{3})".format(startstype.name, stopstype.name, contenttype.name, "" if special is awkward.array.jagged.JaggedArray else clsrepr(special)))
        if startstype.ndim != stopstype.ndim:
            raise ValueError("JaggedArray.starts must have the same number of dimensions as JaggedArray.stops")
        self.startstype = startstype
        self.stopstype = stopstype
        self.contenttype = contenttype
        self.special = special

    def getitem(self, wheretype):
        headtype = wheretype.types[0]
        tailtype = numba.types.Tuple(wheretype.types[1:])
        if isinstance(headtype, numba.types.Integer) and isinstance(self.contenttype, numba.types.Array):
            return numba.typing.arraydecl.get_array_index_type(self.contenttype, tailtype).result

        fake = _JaggedArray_typer_getitem(JaggedArrayType(numba.types.int64[:], numba.types.int64[:], self), wheretype, NOTADVANCED)

        if isinstance(fake, numba.types.Array):
            return fake.dtype
        else:
            return fake.contenttype

def _JaggedArray_typer_getitem(arraytype, wheretype, advancedtype):
    if len(wheretype.types) == 0:
        return arraytype

    isarray = (isinstance(wheretype.types[0], numba.types.Array) and wheretype.types[0].ndim == 1)

    contenttype = _JaggedArray_typer_getitem(arraytype.contenttype, numba.types.Tuple(wheretype.types[1:]), ISADVANCED if isarray else advancedtype)

    if isinstance(wheretype.types[0], numba.types.Integer) or (advancedtype == ISADVANCED and isarray):
        return contenttype
    elif isinstance(wheretype.types[0], numba.types.SliceType) or (advancedtype == NOTADVANCED and isarray):
        return JaggedArrayType(arraytype.startstype, arraytype.stopstype, contenttype, special=arraytype.special)
    else:
        raise TypeError("cannot be used for indexing: {0}".format(wheretype))

def _JaggedArray_getitem_next(array, where):
    pass

@numba.extending.type_callable(_JaggedArray_getitem_next)
def _JaggedArray_type_getitem_next(context):
    return _JaggedArray_typer_getitem

@numba.extending.typeof_impl.register(awkward.array.jagged.JaggedArray)
def _JaggedArray_typeof(val, c):
    return JaggedArrayType(numba.typeof(val.starts), numba.typeof(val.stops), numba.typeof(val.content), special=type(val))

######################################################################## model and boxing

@numba.extending.register_model(JaggedArrayType)
class JaggedArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("starts", fe_type.startstype),
                   ("stops", fe_type.stopstype),
                   ("content", fe_type.contenttype),
                   ("iscompact", numba.types.boolean)]
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
    array.iscompact = c.pyapi.to_native_value(numba.types.boolean, iscompact_obj).value

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
    array.iscompact = context.get_constant(numba.types.boolean, False)   # unless you reproduce that logic here or call out to Python
    return array._getvalue()

def _JaggedArray_new(array, starts, stops, content, iscompact):
    pass

@numba.extending.type_callable(_JaggedArray_new)
def _JaggedArray_type_new(context):
    def typer(arraytype, startstype, stopstype, contenttype, iscompacttype):
        return arraytype
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
def _JaggedArray_lower_getitem_integer(context, builder, sig, args):
    arraytype, wheretype = sig.args
    arrayval, whereval = args

    array = numba.cgutils.create_struct_proxy(arraytype)(context, builder, value=arrayval)

    startstype = arraytype.startstype
    start = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, startstype.dtype(startstype, wheretype), (array.starts, whereval))

    stopstype = arraytype.stopstype
    stop = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, stopstype.dtype(stopstype, wheretype), (array.stops, whereval))

    contenttype = arraytype.contenttype
    _check_startstop_contentlen(context, builder, startstype.dtype, start, stopstype.dtype, stop, contenttype, array.content)

    if isinstance(contenttype, numba.types.Array):
        return numba.targets.arrayobj.getitem_arraynd_intp(context, builder, contenttype(contenttype, numba.types.slice2_type), (array.content, sliceval2(context, builder, start, stop)))
    else:
        return _JaggedArray_lower_getitem_slice(context, builder, contenttype(contenttype, numba.types.slice2_type), (array.content, sliceval2(context, builder, start, stop)))

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
    return _JaggedArray_lower_new(context, builder, arraytype(arraytype, startstype, stopstype, contenttype, numba.types.boolean), (arrayval, starts, stops, array.content, iscompact))

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
    return _JaggedArray_lower_new(context, builder, arraytype(arraytype, startstype, stopstype, contenttype, numba.types.boolean), (arrayval, starts, stops, array.content, context.get_constant(numba.types.boolean, False)))

@numba.extending.lower_builtin(operator.getitem, JaggedArrayType, numba.types.BaseTuple)
def _JaggedArray_lower_getitem_enter(context, builder, sig, args):
    arraytype, wheretype = sig.args
    arrayval, whereval = args

    if any(isinstance(x, numba.types.Array) for x in wheretype.types):
        arraylen = numba.cgutils.alloca_once_value(builder, context.get_constant(numba.types.int64, 0))
        for i, whereitemtype in enumerate(wheretype.types):
            if isinstance(whereitemtype, numba.types.Array):
                if isinstance(whereitemtype.dtype, numba.types.Boolean):
                    enter_arraylen = lambda whereitem, arraylen: max(arraylen, numpy.count_nonzero(whereitem))
                else:
                    enter_arraylen = lambda whereitem, arraylen: max(arraylen, len(whereitem))

                whereitemval = builder.extract_value(whereval, i)
                arraylenval = context.compile_internal(builder, enter_arraylen, numba.types.int64(whereitemtype, numba.types.int64), (whereitemval, builder.load(arraylen)))
                builder.store(arraylenval, arraylen)

        arraylenval = builder.load(arraylen)
        newwheretype = numba.types.Tuple(tuple(_JaggedArray_getitem_enter_toadvanced_typer(x) for x in wheretype))
        newwherevals = []
        for i, (old, new) in enumerate(zip(wheretype.types, newwheretype.types)):
            whereitemval = builder.extract_value(whereval, i)
            newwherevals.append(inline(context, builder, _JaggedArray_getitem_enter_toadvanced, new(old, numba.types.int64), (whereitemval, arraylenval)))
        newwhereval = context.make_tuple(builder, newwheretype, tuple(newwherevals))
    else:
        newwheretype = wheretype
        newwhereval = whereval

    def fake1(array, newwhere):
        return _JaggedArray_getitem_next(awkward.array.jagged.JaggedArray(numpy.array([0], numpy.int64), numpy.array([len(array)], numpy.int64), array), newwhere, None)[0]

    def fake2(array, newwhere):
        out = _JaggedArray_getitem_next(awkward.array.jagged.JaggedArray(numpy.array([0], numpy.int64), numpy.array([len(array)], numpy.int64), array), newwhere, None)
        return out.content[out.starts[0]:out.stops[-1]]

    fake = fake1 if all(isinstance(x, numba.types.Integer) for x in newwheretype.types) else fake2
    return context.compile_internal(builder, fake, sig.return_type(arraytype, newwheretype), (arrayval, newwhereval))

def _JaggedArray_getitem_enter_toadvanced_typer(whereitemtype):
    if isinstance(whereitemtype, (numba.types.Array, numba.types.Integer)):
        return numba.types.Array(numba.types.int64, 1, "C")
    else:
        return whereitemtype
    
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
    headval = numba.targets.tupleobj.static_getitem_tuple(context, builder, headtype(wheretype, numba.types.int64), (whereval, 0))
    tailval = numba.targets.tupleobj.static_getitem_tuple(context, builder, tailtype(wheretype, numba.types.slice2_type), (whereval, slice(1, None)))

    sig = sig.return_type(arraytype, headtype, tailtype, advancedtype)
    args = (arrayval, headval, tailval, advancedval)
    return inline(context, builder, _JaggedArray_bytype_getitem_next, sig, args)

@numba.generated_jit(nopython=True)
def _JaggedArray_bytype_getitem_next(array, head, tail, advanced):
    intp_maxval = numba.types.intp.maxval

    if isinstance(head, numba.types.Integer):
        if isinstance(array.contenttype, numba.types.Array):
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
                return _JaggedArray_getitem_next(array.content, tail, advanced)

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

    elif isinstance(head, numba.types.SliceType):
        def getitem(array, head, tail, advanced):
            if head.step == 0:
                raise ValueError("slice step cannot be zero")

            offsets = numpy.empty(len(array.starts) + 1, numpy.int64)
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
            return _JaggedArray_lower_new(array, starts, stops, next, True)

    elif isinstance(head, numba.types.Array) and isinstance(head.dtype, numba.types.Integer) and head.ndim == 1:
        if advanced == NOTADVANCED:
            def getitem(array, head, tail, advanced):
                offsets = numpy.empty(len(array.starts) + 1, numpy.int64)
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
                return _JaggedArray_lower_new(array, starts, stops, next, True)

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

                next = getitem_next(array.content[index], tail, nextadvanced)
                return next

    else:
        raise AssertionError(head)

    return getitem

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
                nextadvanced[k : k + counts[i]] = advanced[i]
                k += length
            return nextadvanced
        return impl

######################################################################## other lowered methods in Numba

@numba.typing.templates.infer_getattr
class _JaggedArrayType_type_methods(numba.typing.templates.AttributeTemplate):
    key = JaggedArrayType

    # @numba.typing.templates.bound_function("name-of-method")
    # def resolve_name-of-method(self, arraytype, args, kwargs):
    #     standard typer...

@numba.extending.overload_method(JaggedArrayType, "compact")
def _JaggedArray_compact(arraytype):
    if isinstance(arraytype, JaggedArrayType) and isinstance(arraytype.contenttype, AwkwardArrayType):
        def impl(array):
            if array.iscompact:
                return array
            if len(array.starts) == 0:
                tmp1 = array
                tmp2 = array.starts
                tmp3 = array.stops[0:0]
                tmp4 = array.content[0:0]
                tmp5 = True
                return _JaggedArray_new(tmp1, tmp2, tmp3, tmp4, tmp5)
                # return _JaggedArray_new(array, array.starts, array.stops[0:0], array.content[0:0], True)

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
