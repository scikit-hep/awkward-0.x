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

from awkward.array.base import AwkwardArray
from awkward.array.chunked import ChunkedArray, AppendableArray
from awkward.array.indexed import IndexedArray, SparseArray
from awkward.array.jagged import JaggedArray
from awkward.array.masked import MaskedArray, BitMaskedArray, IndexedMaskedArray
from awkward.array.objects import Methods, ObjectArray, StringArray
from awkward.array.table import Table
from awkward.array.union import UnionArray
from awkward.array.virtual import VirtualArray

######################################################################## utilities

def sliceval2(context, builder, start, stop):
    out = context.make_helper(builder, numba.types.slice2_type)
    out.start = start
    out.stop = stop
    out.step = context.get_constant(numba.types.intp, 1)
    return out._getvalue()

def sliceval3(context, builder, start, stop, step):
    out = context.make_helper(builder, numba.types.slice3_type)
    out.start = start
    out.stop = stop
    out.step = step
    return out._getvalue()

def check_startstop_contentlen(context, builder, starttype, startval, stoptype, stopval, contenttype, contentval):
    if isinstance(contenttype, numba.types.Array):
        contentlen = numba.targets.arrayobj.array_len(context, builder, numba.types.intp(contenttype), (contentval,))
    else:
        contentlen = JaggedArray_lower_len(context, builder, numba.types.intp(contenttype), (contentval,))

    with builder.if_then(builder.or_(builder.or_(builder.icmp_signed("<", startval, context.get_constant(starttype, 0)),
                                                 builder.icmp_signed("<", stopval, context.get_constant(stoptype, 0))),
                                     builder.or_(builder.icmp_signed(">=", startval, contentlen),
                                                 builder.icmp_signed(">", stopval, contentlen))),
                         likely=False):
        context.call_conv.return_user_exc(builder, ValueError, ("JaggedArray.starts or JaggedArray.stops is beyond the range of JaggedArray.content",))

######################################################################## AwkwardArrayType

class AwkwardArrayType(numba.types.Type):
    pass

@numba.typing.templates.infer_global(len)
class AwkwardArrayType_type_len(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            arraytype, = args
            if isinstance(arraytype, AwkwardArrayType):
                return numba.typing.templates.signature(numba.types.intp, arraytype)

@numba.typing.templates.infer_global(operator.getitem)
class AwkwardArrayType_type_getitem(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0:
            arraytype, wheretype = args
            if isinstance(arraytype, AwkwardArrayType):
                return numba.typing.templates.signature(arraytype.getitem(wheretype, False), *args, **kwargs)

def getitem(arraytype, wheretype, advanced):
    if isinstance(arraytype, AwkwardArrayType):
        return arraytype.getitem(wheretype, advanced)
    if isinstance(arraytype, numba.types.Array):
        return numba.typing.arraydecl.get_array_index_type(arraytype, wheretype).result

def specialrepr(x):
    if x is ChunkedArray or x is AppendableArray or x is IndexedArray or x is SparseArray or x is JaggedArray or x is MaskedArray or x is BitMaskedArray or x is IndexedMaskedArray or x is Methods or x is ObjectArray or x is StringArray or x is Table or x is UnionArray or x is VirtualArray:
        return x.__name__
    elif x.__bases__ == (object,):
        return x.__module__ + "." + x.__name__
    else:
        return x.__module__ + "." + x.__name__ + "<" + ", ".join(specialrepr(y) for y in x.__bases__) + ">"

######################################################################## JaggedArrayType

class JaggedArrayType(AwkwardArrayType):
    def __init__(self, startstype, stopstype, contenttype, specialization=JaggedArray):
        if startstype.ndim != stopstype.ndim:
            raise ValueError("len(JaggedArray.starts.shape) must be equal to len(JaggedArray.stops.shape)")
        super(JaggedArrayType, self).__init__(name="{0}({1}, {2}, {3})".format(specialrepr(specialization), startstype.name, stopstype.name, contenttype.name))
        self.startstype = startstype
        self.stopstype = stopstype
        self.contenttype = contenttype
        self.specialization = specialization

    def getitem(self, wheretype, advanced):
        if not isinstance(wheretype, numba.types.BaseTuple):
            wheretype = numba.types.Tuple((wheretype,))
        if len(wheretype) == 0:
            return self
        head, tail = numba.types.Tuple(wheretype.types[:self.startstype.ndim]), numba.types.Tuple(wheretype.types[self.startstype.ndim:])

        startstype = getitem(self.startstype, head, advanced)
        stopstype = getitem(self.stopstype, head, advanced)
        if startstype is None or stopstype is None:
            return None

        headarray = any(isinstance(x, numba.types.Array) for x in head.types)
        finaltailarray = isinstance(self.contenttype, numba.types.Array) and any(isinstance(x, numba.types.Array) for x in tail.types)

        contenttype = getitem(self.contenttype, tail, advanced or headarray)
        if contenttype is None:
            return None

        assert isinstance(startstype, numba.types.Array) == isinstance(stopstype, numba.types.Array)
        if isinstance(startstype, numba.types.Array) and not (advanced and headarray) and not (advanced and finaltailarray):
            return JaggedArrayType(startstype, stopstype, contenttype, specialization=self.specialization)
        else:
            return contenttype

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
def JaggedArray_unbox(typ, obj, c):
    starts_obj = c.pyapi.object_getattr_string(obj, "starts")
    stops_obj = c.pyapi.object_getattr_string(obj, "stops")
    content_obj = c.pyapi.object_getattr_string(obj, "content")
    iscompact_obj = c.pyapi.object_getattr_string(obj, "iscompact")

    jaggedarray = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    jaggedarray.starts = c.pyapi.to_native_value(typ.startstype, starts_obj).value
    jaggedarray.stops = c.pyapi.to_native_value(typ.stopstype, stops_obj).value
    jaggedarray.content = c.pyapi.to_native_value(typ.contenttype, content_obj).value
    jaggedarray.iscompact = c.pyapi.to_native_value(numba.types.boolean, iscompact_obj).value

    c.pyapi.decref(starts_obj)
    c.pyapi.decref(stops_obj)
    c.pyapi.decref(content_obj)
    c.pyapi.decref(iscompact_obj)

    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(jaggedarray._getvalue(), is_error)

@numba.extending.box(JaggedArrayType)
def JaggedArray_box(typ, val, c):
    jaggedarray = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    starts_obj = c.pyapi.from_native_value(typ.startstype, jaggedarray.starts, c.env_manager)
    stops_obj = c.pyapi.from_native_value(typ.stopstype, jaggedarray.stops, c.env_manager)
    content_obj = c.pyapi.from_native_value(typ.contenttype, jaggedarray.content, c.env_manager)

    cls = c.pyapi.unserialize(c.pyapi.serialize_object(typ.specialization))
    out = c.pyapi.call_function_objargs(cls, (starts_obj, stops_obj, content_obj))

    c.pyapi.decref(starts_obj)
    c.pyapi.decref(stops_obj)
    c.pyapi.decref(content_obj)
    return out

@numba.extending.type_callable(JaggedArray)
def JaggedArray_type_init(context):
    def typer(startstype, stopstype, contenttype):
        if isinstance(startstype, numba.types.Array) and isinstance(stopstype, numba.types.Array) and isinstance(contenttype, (numba.types.Array, AwkwardArrayType)):
            return JaggedArrayType(startstype, stopstype, contenttype)
    return typer

@numba.extending.lower_builtin(JaggedArray, numba.types.Array, numba.types.Array, numba.types.Array)
@numba.extending.lower_builtin(JaggedArray, numba.types.Array, numba.types.Array, AwkwardArrayType)
def JaggedArray_init_array(context, builder, sig, args):
    startstype, stopstype, contenttype = sig.args
    starts, stops, content = args

    if context.enable_nrt:
        context.nrt.incref(builder, startstype, starts)
        context.nrt.incref(builder, stopstype, stops)
        context.nrt.incref(builder, contenttype, content)

    jaggedarray = numba.cgutils.create_struct_proxy(sig.return_type)(context, builder)
    jaggedarray.starts = starts
    jaggedarray.stops = stops
    jaggedarray.content = content
    jaggedarray.iscompact = context.get_constant(numba.types.boolean, False)
    return jaggedarray._getvalue()

######################################################################## JaggedArray len and getitem

@numba.extending.lower_builtin(len, JaggedArrayType)
def JaggedArray_lower_len(context, builder, sig, args):
    jaggedarraytype, = sig.args
    jaggedarrayval, = args

    jaggedarray = numba.cgutils.create_struct_proxy(jaggedarraytype)(context, builder, value=jaggedarrayval)

    startstype = jaggedarraytype.startstype
    return numba.targets.arrayobj.array_len(context, builder, numba.types.intp(startstype), (jaggedarray.starts,))

@numba.extending.lower_builtin(operator.getitem, JaggedArrayType, numba.types.Integer)
def JaggedArray_lower_getitem_integer(context, builder, sig, args):
    jaggedarraytype, wheretype = sig.args
    jaggedarrayval, whereval = args

    jaggedarray = numba.cgutils.create_struct_proxy(jaggedarraytype)(context, builder, value=jaggedarrayval)

    startstype = jaggedarraytype.startstype
    start = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, startstype.dtype(startstype, wheretype), (jaggedarray.starts, whereval))

    stopstype = jaggedarraytype.stopstype
    stop = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, stopstype.dtype(stopstype, wheretype), (jaggedarray.stops, whereval))

    contenttype = jaggedarraytype.contenttype
    check_startstop_contentlen(context, builder, startstype.dtype, start, stopstype.dtype, stop, contenttype, jaggedarray.content)

    if isinstance(contenttype, numba.types.Array):
        return numba.targets.arrayobj.getitem_arraynd_intp(context, builder, contenttype(contenttype, numba.types.slice2_type), (jaggedarray.content, sliceval2(context, builder, start, stop)))
    else:
        return JaggedArray_lower_getitem_slice(context, builder, contenttype(contenttype, numba.types.slice2_type), (jaggedarray.content, sliceval2(context, builder, start, stop)))

@numba.extending.lower_builtin(operator.getitem, JaggedArrayType, numba.types.SliceType)
def JaggedArray_lower_getitem_slice(context, builder, sig, args):
    jaggedarraytype, wheretype = sig.args
    jaggedarrayval, whereval = args

    jaggedarray = numba.cgutils.create_struct_proxy(jaggedarraytype)(context, builder, value=jaggedarrayval)

    startstype = jaggedarraytype.startstype
    starts = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, startstype(startstype, wheretype), (jaggedarray.starts, whereval))

    stopstype = jaggedarraytype.stopstype
    stops = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, stopstype(stopstype, wheretype), (jaggedarray.stops, whereval))

    contenttype = jaggedarraytype.contenttype
    return JaggedArray_lower_copy(context, builder, jaggedarraytype(jaggedarraytype, startstype, stopstype, contenttype, numba.types.boolean), (jaggedarrayval, starts, stops, jaggedarray.content, jaggedarray.iscompact))

@numba.extending.lower_builtin(operator.getitem, JaggedArrayType, numba.types.Array)
def JaggedArray_lower_getitem_array(context, builder, sig, args):
    jaggedarraytype, wheretype = sig.args
    jaggedarrayval, whereval = args

    jaggedarray = numba.cgutils.create_struct_proxy(jaggedarraytype)(context, builder, value=jaggedarrayval)

    startstype = jaggedarraytype.startstype
    starts = numba.targets.arrayobj.fancy_getitem_array(context, builder, startstype(startstype, wheretype), (jaggedarray.starts, whereval))

    stopstype = jaggedarraytype.stopstype
    stops = numba.targets.arrayobj.fancy_getitem_array(context, builder, stopstype(stopstype, wheretype), (jaggedarray.stops, whereval))

    contenttype = jaggedarraytype.contenttype
    return JaggedArray_lower_copy(context, builder, jaggedarraytype(jaggedarraytype, startstype, stopstype, contenttype, numba.types.boolean), (jaggedarrayval, starts, stops, jaggedarray.content, context.get_constant(numba.types.boolean, False)))

# def JaggedArray_getitem_intarray(jaggedarray, where):
#     starts = jaggedarray.starts[where]
#     stops = jaggedarray.stops[where]
#     return jaggedarray.copy(starts, stops, jaggedarray.content, False)

# def JaggedArray_lower_getitem(context, builder, sig, args):
#     if isinstance(sig.args[1], numba.types.Integer):
#         getitem = JaggedArray_getitem_integer
#     elif isinstance(sig.args[1], numba.types.SliceType):
#         getitem = JaggedArray_getitem_slice
#     elif isinstance(sig.args[1], numba.types.Array) and sig.args[1].ndim == 1 and isinstance(sig.args[1].dtype, numba.types.Boolean):
#         getitem = JaggedArray_getitem_boolarray
#     elif isinstance(sig.args[1], numba.types.Array) and sig.args[1].ndim == 1 and isinstance(sig.args[1].dtype, numba.types.Integer):
#         getitem = JaggedArray_getitem_intarray
#     else:
#         raise AssertionError(sig.args[1])
#     if sig.args not in getitem.overloads:
#         getitem.compile(sig)
#     cres = getitem.overloads[sig.args]
#     return cres.target_context.get_function(cres.entry_point, cres.signature)._imp(context, builder, sig, args, loc=None)

def JaggedArray_lower_getitem_tuple_next(context, builder, nexttype, nextval, tailtype, tailval):
    return nextval

@numba.extending.lower_builtin(operator.getitem, JaggedArrayType, numba.types.BaseTuple)
def JaggedArray_lower_getitem_tuple_entry(context, builder, sig, args):
    jaggedarraytype, wheretype = sig.args
    jaggedarrayval, whereval = args

    if len(wheretype.types) == 0:
        return jaggedarrayval

    jaggedarray = numba.cgutils.create_struct_proxy(jaggedarraytype)(context, builder, value=jaggedarrayval)

    headtype = numba.types.Tuple(wheretype.types[:jaggedarraytype.startstype.ndim])
    tailtype = numba.types.Tuple(wheretype.types[jaggedarraytype.startstype.ndim:])
    headval = numba.targets.tupleobj.static_getitem_tuple(context, builder, headtype(whereval, numba.types.slice2_type), (whereval, slice(None, jaggedarraytype.startstype.ndim)))
    tailval = numba.targets.tupleobj.static_getitem_tuple(context, builder, tailtype(whereval, numba.types.slice2_type), (whereval, slice(jaggedarraytype.startstype.ndim, None)))

    startstype = numba.typing.arraydecl.get_array_index_type(jaggedarraytype.startstype, headtype).result
    stopstype = numba.typing.arraydecl.get_array_index_type(jaggedarraytype.stopstype, headtype).result
    starts = numba.targets.arrayobj.getitem_array_tuple(context, builder, startstype(jaggedarraytype.startstype, headtype), (jaggedarray.starts, headval))
    stops = numba.targets.arrayobj.getitem_array_tuple(context, builder, stopstype(jaggedarraytype.stopstype, headtype), (jaggedarray.stops, headval))

    contenttype = jaggedarraytype.contenttype

    assert isinstance(startstype, numba.types.Integer) == isinstance(stopstype, numba.types.Integer)
    if isinstance(startstype, numba.types.Integer):
        check_startstop_contentlen(context, builder, startstype, starts, stopstype, stops, contenttype, jaggedarray.content)

        if isinstance(contenttype, numba.types.Array):
            nexttype = contenttype
            nextval = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, nexttype(contenttype, numba.types.slice2_type), (jaggedarray.content, sliceval2(context, builder, starts, stops)))
            if len(tailtype.types) == 0:
                return nextval
            else:
                rettype = numba.typing.arraydecl.get_array_index_type(nexttype, tailtype).result
                return numba.targets.arrayobj.getitem_array_tuple(context, builder, rettype(nexttype, tailtype), (nextval, tailval))

        else:
            nexttype = contenttype
            nextval = JaggedArray_lower_getitem_slice(context, builder, contenttype(contenttype, numba.types.slice2_type), (jaggedarray.content, sliceval2(context, builder, starts, stops)))
 
    else:
        nexttype = JaggedArrayType(startstype, stopstype, contenttype, specialization=jaggedarraytype.specialization)
        nextval = JaggedArray_lower_copy(context, builder, nexttype(jaggedarraytype, startstype, stopstype, contenttype, numba.types.boolean), (jaggedarrayval, starts, stops, jaggedarray.content, context.get_constant(numba.types.boolean, False)))

    return JaggedArray_lower_getitem_tuple_next(context, builder, nexttype, nextval, tailtype, tailval)

######################################################################## JaggedArray_methods

@numba.typing.templates.infer_getattr
class JaggedArrayType_type_methods(numba.typing.templates.AttributeTemplate):
    key = JaggedArrayType

    @numba.typing.templates.bound_function("copy")
    def resolve_copy(self, jaggedarraytype, args, kwargs):
        if len(args) == 4 and len(kwargs) == 0:
            startstype, stopstype, contenttype, iscompacttype = args
            if isinstance(startstype, numba.types.Array) and isinstance(stopstype, numba.types.Array) and isinstance(contenttype, (numba.types.Array, AwkwardArrayType)) and isinstance(iscompacttype, numba.types.Boolean):
                return jaggedarraytype(startstype, stopstype, contenttype, iscompacttype)

@numba.extending.lower_builtin("copy", JaggedArrayType, numba.types.Array, numba.types.Array, numba.types.Array, numba.types.boolean)
@numba.extending.lower_builtin("copy", JaggedArrayType, numba.types.Array, numba.types.Array, AwkwardArrayType, numba.types.boolean)
def JaggedArray_lower_copy(context, builder, sig, args):
    jaggedarraytype, startstype, stopstype, contenttype, iscompacttype = sig.args
    jaggedarray, starts, stops, content, iscompact = args

    if context.enable_nrt:
        context.nrt.incref(builder, startstype, starts)
        context.nrt.incref(builder, stopstype, stops)
        context.nrt.incref(builder, contenttype, content)

    jaggedarray = numba.cgutils.create_struct_proxy(sig.return_type)(context, builder)
    jaggedarray.starts = starts
    jaggedarray.stops = stops
    jaggedarray.content = content
    jaggedarray.iscompact = iscompact
    return jaggedarray._getvalue()

@numba.extending.overload_method(JaggedArrayType, "compact")
def JaggedArray_compact(jaggedarraytype):
    if isinstance(jaggedarraytype, JaggedArrayType):
        def impl(jaggedarray):
            if jaggedarray.iscompact:
                return jaggedarray
            if len(jaggedarray.starts) == 0:
                return jaggedarray.copy(jaggedarray.starts, jaggedarray.starts, jaggedarray.content[0:0], True)

            if jaggedarray.starts.shape != jaggedarray.stops.shape:
                raise ValueError("JaggedArray.starts.shape must be equal to JaggedArray.stops.shape")
            flatstarts = jaggedarray.starts.ravel()
            flatstops = jaggedarray.stops.ravel()

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

            starts = offsets[:-1].reshape(jaggedarray.starts.shape)
            stops = offsets[1:].reshape(jaggedarray.stops.shape)
            content = jaggedarray.content[index]
            return jaggedarray.copy(starts, stops, content, True)

    return impl

@numba.extending.overload_method(JaggedArrayType, "flatten")
def JaggedArray_flatten(jaggedarraytype):
    if isinstance(jaggedarraytype, JaggedArrayType):
        def impl(jaggedarray):
            return jaggedarray.compact().content
    return impl

######################################################################## AwkwardArray_typeof

@numba.extending.typeof_impl.register(JaggedArray)
def AwkwardArray_typeof(val, c):
    if isinstance(val, JaggedArray):
        return JaggedArrayType(numba.typeof(val.starts), numba.typeof(val.stops), numba.typeof(val.content), type(val))
    else:
        raise NotImplementedError(type(val))
