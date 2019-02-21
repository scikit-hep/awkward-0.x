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
import numba.typing.arraydecl

import awkward.array.jagged
from .base import NumbaMethods
from .base import AwkwardArrayType
from .base import clsrepr
from .base import ISADVANCED
from .base import NOTADVANCED
from .base import sliceval2
from .base import sliceval3

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

    def getitem(self, where):
        head = numba.types.Tuple(where.types[0])
        if isinstance(head, numba.types.Integer) and isinstance(self.content, numba.types.Array):
            return numba.typing.arraydecl.get_array_index_type(self.content, tail).result

        fake = _JaggedArray_getitem_typer(JaggedArrayType(numba.types.int64[:], numba.types.int64[:], self), where, NOTADVANCED)
        if isinstance(fake, numba.types.Array):
            return fake.dtype
        else:
            return fake.content

def _JaggedArray_getitem_typer(arraytype, wheretype, advancedtype):
    if len(wheretype.types) == 0:
        return arraytype

    assert arraytype.startstype.ndim == arraytype.stopstype.ndim == 1

    isarray = (isinstance(wheretype.types[0], numba.types.Array) and wheretype.types[0].ndim == 1)

    contenttype = _JaggedArray_getitem_typer(arraytype.contenttype, numba.types.Tuple(wheretype.types[1:]), ISADVANCED if isarray else advancedtype)

    if isinstance(wheretype.types[0], numba.types.Integer) or (advancedtype == ISADVANCED and isarray):
        return contenttype
    elif isinstance(wheretype.types[0], numba.types.SliceType) or (advancedtype == NOTADVANCED and isarray):
        return JaggedArrayType(arraytype.startstype, arraytype.stopstype, contenttype, special=arraytype.special)
    else:
        raise TypeError("cannot be used for indexing: {0}".format(wheretype))

def _JaggedArray_getitem_next(array, where):
    pass

@numba.extending.type_callable(_JaggedArray_getitem_next)
def _JaggedArray_getitem_next_typer(context):
    return _JaggedArray_getitem_typer

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
    array.starts = c.pyapi_to_native_value(typ.startstype, starts_obj).value
    array.stops = c.pyapi_to_native_value(typ.stopstype, stops_obj).value
    array.content = c.pyapi_to_native_value(typ.contenttype, content_obj).value
    array.iscompact = c.pyapi_to_native_value(numba.types.boolean, iscompact_obj).value

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

######################################################################## lower methods in Numba

@numba.extending.lower_builtin(len, JaggedArrayType)
def _JaggedArray_lower_len(context, builder, sig, args):
    arraytype, = sig.args
    arrayval, = args

    array = numba.cgutils.create_struct_proxy(arraytype)(context, builder, value=arrayval)

    return numba.targets.arrayobj.array_len(context, builder, numba.types.intp(arraytype.startstype), (array.starts,))

@numba.extending.lower_builtin(operator.getitem, JaggedArrayType, numba.types.Integer)
def _JaggedArray_getitem_lower_integer(context, builder, sig, args):
    arraytype, wheretype = sig.args
    arrayval, whereval = args

    array = numba.cgutils.create_struct_proxy(arraytype)(context, builder, value=arrayval)

    startstype = arraytype.startstype
    start = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, startstype.dtype(startstype, wheretype), (array.starts, whereval))

    stopstype = arraytype.stopstype
    stop = numba.targets.arrayobj.getitem_arraynd_intp(context, builder, stopstype.dtype(stopstype, wheretype), (array.stops, whereval))

    contenttype = arraytype.contenttype
    _check_startstop_contentlen(context, builder, startstype.dtype, start, stopstype.dtype, stop, contenttype, array.content)

    HERE
