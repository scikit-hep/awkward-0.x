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

from awkward.array.base import AwkwardArray
from awkward.array.chunked import ChunkedArray, AppendableArray
from awkward.array.indexed import IndexedArray, SparseArray
from awkward.array.jagged import JaggedArray
from awkward.array.masked import MaskedArray, BitMaskedArray, IndexedMaskedArray
from awkward.array.objects import Methods, ObjectArray, StringArray
from awkward.array.table import Table
from awkward.array.union import UnionArray
from awkward.array.virtual import VirtualArray

######################################################################## AwkwardArrayType

class AwkwardArrayType(numba.types.Type):
    pass

@numba.typing.templates.infer_global(operator.getitem)
class AwkwardArrayType_type_getitem(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0:
            arraytype, indextype = args
            if isinstance(arraytype, AwkwardArrayType):
                return numba.typing.templates.signature(arraytype.getitem(indextype), *args, **kwargs)

######################################################################## JaggedArrayType

class JaggedArrayType(AwkwardArrayType):
    def __init__(self, startstype, stopstype, contenttype, specialization=JaggedArray):
        super(JaggedArrayType, self).__init__(name="JaggedArrayType{0}({1}, {2}, {3})".format("" if specialization is JaggedArray else repr(abs(hash(specialization))), startstype.name, stopstype.name, contenttype.name))
        self.startstype = startstype
        self.stopstype = stopstype
        self.contenttype = contenttype
        self.specialization = specialization
        
    def getitem(self, indextype):
        if isinstance(indextype, numba.types.Integer):
            return self.contenttype
        if isinstance(indextype, numba.types.SliceType):
            return self

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

######################################################################## JaggedArray_getitem

@numba.extending.overload(len)
def JaggedArray_len(jaggedarraytype):
    if isinstance(jaggedarraytype, JaggedArrayType):
        def impl(jaggedarray):
            return len(jaggedarray.starts)
        return impl

@numba.njit
def JaggedArray_getitem_integer(jaggedarray, where):
    start = jaggedarray.starts[where]
    stop = jaggedarray.stops[where]
    if start >= len(jaggedarray.content) or stop > len(jaggedarray.content):
        raise ValueError("JaggedArray.starts or JaggedArray.stops extends beyond JaggedArray.content")
    return jaggedarray.content[start:stop]

@numba.njit
def JaggedArray_getitem_slice(jaggedarray, where):
    starts = jaggedarray.starts[where]
    stops = jaggedarray.stops[where]
    return jaggedarray.copy(starts, stops, jaggedarray.content, jaggedarray.iscompact)

@numba.extending.lower_builtin(operator.getitem, JaggedArrayType, numba.types.Integer)
@numba.extending.lower_builtin(operator.getitem, JaggedArrayType, numba.types.SliceType)
def JaggedArray_lower_getitem(context, builder, sig, args):
    if isinstance(sig.args[1], numba.types.Integer):
        getitem = JaggedArray_getitem_integer
    elif isinstance(sig.args[1], numba.types.SliceType):
        getitem = JaggedArray_getitem_slice
    else:
        raise AssertionError(sig.args[1])
    if sig.args not in getitem.overloads:
        getitem.compile(sig)
    cres = getitem.overloads[sig.args]
    return cres.target_context.get_function(cres.entry_point, cres.signature)._imp(context, builder, sig, args, loc=None)

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

    jaggedarray = numba.cgutils.create_struct_proxy(jaggedarraytype)(context, builder)
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
