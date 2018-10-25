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

import awkward.array.chunked
import awkward.array.jagged
import awkward.array.masked
import awkward.array.table
import awkward.util

ARROWMASKTYPE = awkward.util.numpy.uint8
ARROWINDEXTYPE = awkward.util.numpy.int32

def view(obj):
    import pyarrow

    def recurse(tpe, buffers):
        if isinstance(tpe, pyarrow.lib.StructType):
            pairs = []
            for i in range(tpe.num_children - 1, -1, -1):
                pairs.insert(0, (tpe[i].name, recurse(tpe[i].type, buffers)))
            mask = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROWMASKTYPE)
            return awkward.array.masked.BitMaskedArray(mask, awkward.array.table.Table.frompairs(pairs), maskedwhen=False, lsborder=True)

        elif isinstance(tpe, pyarrow.lib.ListType):
            content = recurse(tpe.value_type, buffers)
            offsets = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROWINDEXTYPE)
            mask = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROWMASKTYPE)
            return awkward.array.masked.BitMaskedArray(mask, awkward.array.jagged.JaggedArray.fromoffsets(offsets, content), maskedwhen=False, lsborder=True)

        elif isinstance(tpe, pyarrow.lib.DataType):
            content = awkward.util.numpy.frombuffer(buffers.pop(), dtype=tpe.to_pandas_dtype())
            mask = awkward.util.numpy.frombuffer(buffers.pop(), dtype=ARROWMASKTYPE)
            return awkward.array.masked.BitMaskedArray(mask, content, maskedwhen=False, lsborder=True)

        else:
            raise NotImplementedError(repr(tpe))

    if isinstance(obj, pyarrow.lib.Array):
        buffers = obj.buffers()
        out = recurse(obj.type, buffers)
        assert len(buffers) == 0
        return out

    elif isinstance(obj, pyarrow.lib.ChunkedArray):
        chunks = [x for x in obj.chunks if len(x) > 0]
        return awkward.array.chunked.ChunkedArray([view(x) for x in chunks], counts=[len(x) for x in chunks])

    else:
        raise NotImplementedError(type(obj))
