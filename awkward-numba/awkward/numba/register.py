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
        
class JaggedArrayType(AwkwardType):
    def __init__(self, startstype, stopstype, contenttype):
        super(JaggedArrayType, self).__init__(name="JaggedArrayType({0}, {1}, {2})".format(startstype.name, stopstype.name, contenttype.name))
        self.startstype = startstype
        self.stopstype = stopstype
        self.contenttype = contenttype

@numba.extending.typeof_impl.register(JaggedArray)
def typeof_JaggedArray(val, c):
    return JaggedArrayType(numba.typeof(val.starts), numba.typeof(val.stops), numba.typeof(val.content))

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

@numba.extending.unbox(JaggedArrayType)
def unbox_JaggedArray(typ, obj, c):
    starts_obj = c.pyapi.object_getattr_string(obj, "starts")
    stops_obj = c.pyapi.object_getattr_string(obj, "stops")
    content_obj = c.pyapi.object_getattr_string(obj, "content")

    jaggedarray = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    jaggedarray.starts = c.pyapi.to_native_value(typ.startstype, starts_obj).value
    jaggedarray.stops = c.pyapi.to_native_value(typ.stopstype, stops_obj).value
    jaggedarray.content = c.pyapi.to_native_value(typ.contenttype, content_obj).value

    is_error = numba.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.extending.NativeValue(jaggedarray._getvalue(), is_error)
