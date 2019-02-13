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

import numba

from awkward.array.chunked import ChunkedArray, AppendableArray
from awkward.array.indexed import IndexedArray, SparseArray
from awkward.array.jagged import JaggedArray
from awkward.array.masked import MaskedArray, BitMaskedArray, IndexedMaskedArray
from awkward.array.objects import Methods, ObjectArray, StringArray
from awkward.array.table import Table
from awkward.array.union import UnionArray
from awkward.array.virtual import VirtualArray

class AwkwardType(numba.types.Type):
    pass

@numba.typing.templates.infer_global(operator.getitem)
class type_getitem(numba.typing.templates.AbstractTemplate):
    def generic(self, args, kwargs):
        if len(args) == 2 and len(kwargs) == 0:
            arraytype, indextype = args
            if isinstance(arraytype, AwkwardType):
                return numba.typing.templates.signature(arraytype.getitem(indextype), *args, **kwargs)

class JaggedArrayType(AwkwardType):
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

@numba.extending.typeof_impl.register(JaggedArray)
def JaggedArray_typeof(val, c):
    return JaggedArrayType(numba.typeof(val.starts), numba.typeof(val.stops), numba.typeof(val.content), type(val))

@numba.extending.register_model(JaggedArrayType)
class JaggedArrayModel(numba.datamodel.models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("starts", fe_type.startstype),
                   ("stops", fe_type.stopstype),
                   ("content", fe_type.contenttype)]
        super(JaggedArrayModel, self).__init__(dmm, fe_type, members)

numba.extending.make_attribute_wrapper(JaggedArrayType, "starts", "starts")
numba.extending.make_attribute_wrapper(JaggedArrayType, "stops", "stops")
numba.extending.make_attribute_wrapper(JaggedArrayType, "content", "content")

@numba.extending.type_callable(JaggedArray)
def JaggedArray_init_type(context):
    def typer(startstype, stopstype, contenttype):
        if isinstance(startstype, numba.types.Array) and isinstance(stopstype, numba.types.Array) and isinstance(contenttype, (numba.types.Array, AwkwardType)):
            return JaggedArrayType(startstype, stopstype, contenttype)
    return typer

@numba.extending.lower_builtin(JaggedArray, numba.types.Array, numba.types.Array, numba.types.Array)
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
    return jaggedarray._getvalue()

    # startsobj = numba.targets.arrayobj.make_array(startstype)(context, builder, value=startsval)

    # print("data", startsobj.data)
    # print("shape", startsobj.shape)
    # print("strides", startsobj.strides)
    # print("itemsize", startsobj.itemsize)
    # print("meminfo", startsobj.meminfo)
    # print("parent", startsobj.parent)

    # starts = numba.targets.arrayobj.make_array(startstype)(context, builder)
    # # numba.targets.arrayobj.populate_array(starts,
    # #                                       data=startsobj.data,
    # #                                       shape=startsobj.shape,
    # #                                       strides=startsobj.strides,
    # #                                       itemsize=startsobj.itemsize,
    # #                                       meminfo=startsobj.meminfo,
    # #                                       parent=startsobj.parent)

    # stopsobj = numba.targets.arrayobj.make_array(stopstype)(context, builder, value=stopsval)
    # stops = numba.targets.arrayobj.make_array(stopstype)(context, builder)
    # # numba.targets.arrayobj.populate_array(stops,
    # #                                       data=stopsobj.data,
    # #                                       shape=stopsobj.shape,
    # #                                       strides=stopsobj.strides,
    # #                                       itemsize=stopsobj.itemsize,
    # #                                       meminfo=stopsobj.meminfo,
    # #                                       parent=stopsobj.parent)

    # contentobj = numba.targets.arrayobj.make_array(contenttype)(context, builder, value=contentval)
    # content = numba.targets.arrayobj.make_array(contenttype)(context, builder)
    # # numba.targets.arrayobj.populate_array(content,
    # #                                       data=contentobj.data,
    # #                                       shape=contentobj.shape,
    # #                                       strides=contentobj.strides,
    # #                                       itemsize=contentobj.itemsize,
    # #                                       meminfo=contentobj.meminfo,
    # #                                       parent=contentobj.parent)

    # jaggedarray = numba.cgutils.create_struct_proxy(sig.return_type)(context, builder)
    # jaggedarray.starts = starts._getvalue()
    # jaggedarray.stops = stops._getvalue()
    # jaggedarray.content = content._getvalue()
    # return jaggedarray._getvalue()

@numba.extending.lower_builtin(operator.getitem, JaggedArrayType, numba.types.Integer)
def JaggedArray_getitem_lower_integer(context, builder, sig, args):
    print("JaggedArray_getitem_lower_integer", sig)

    @numba.njit
    def JaggedArray_getitem_integer(array, where):
        start = array.starts[where]
        stop = array.stops[where]
        return array.content[start:stop]

    if sig.args not in JaggedArray_getitem_integer.overloads:
        print("JaggedArray_getitem_lower_integer compile")
        JaggedArray_getitem_integer.compile(sig)
    cres = JaggedArray_getitem_integer.overloads[sig.args]
    out = cres.target_context.get_function(cres.entry_point, cres.signature)._imp(context, builder, sig, args, loc=None)
    print("JaggedArray_getitem_lower_integer out")
    return out

@numba.extending.lower_builtin(operator.getitem, JaggedArrayType, numba.types.SliceType)
def JaggedArray_getitem_lower_slice(context, builder, sig, args):
    print("JaggedArray_getitem_lower_slice", sig)

    @numba.njit
    def JaggedArray_getitem_slice(array, where):
        starts = array.starts[where]
        stops = array.stops[where]
        return JaggedArray(starts, stops, array.content)

    if sig.args not in JaggedArray_getitem_slice.overloads:
        print("JaggedArray_getitem_lower_slice compile")
        JaggedArray_getitem_slice.compile(sig)
    cres = JaggedArray_getitem_slice.overloads[sig.args]
    out = cres.target_context.get_function(cres.entry_point, cres.signature)._imp(context, builder, sig, args, loc=None)
    print("JaggedArray_getitem_lower_slice out")
    return out

@numba.extending.unbox(JaggedArrayType)
def JaggedArray_unbox(typ, obj, c):
    starts_obj = c.pyapi.object_getattr_string(obj, "starts")
    stops_obj = c.pyapi.object_getattr_string(obj, "stops")
    content_obj = c.pyapi.object_getattr_string(obj, "content")

    jaggedarray = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    jaggedarray.starts = c.pyapi.to_native_value(typ.startstype, starts_obj).value
    jaggedarray.stops = c.pyapi.to_native_value(typ.stopstype, stops_obj).value
    jaggedarray.content = c.pyapi.to_native_value(typ.contenttype, content_obj).value

    c.pyapi.decref(starts_obj)
    c.pyapi.decref(stops_obj)
    c.pyapi.decref(content_obj)

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

    # c.pyapi.incref(out)

    return out
