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

import numpy

from awkward.numba.array.chunked import ChunkedArrayNumba as ChunkedArray
from awkward.numba.array.chunked import AppendableArrayNumba as AppendableArray
from awkward.numba.array.indexed import IndexedArrayNumba as IndexedArray
from awkward.numba.array.indexed import SparseArrayNumba as SparseArray
from awkward.numba.array.jagged import JaggedArrayNumba as JaggedArray
from awkward.numba.array.masked import MaskedArrayNumba as MaskedArray
from awkward.numba.array.masked import BitMaskedArrayNumba as BitMaskedArray
from awkward.numba.array.masked import IndexedMaskedArrayNumba as IndexedMaskedArray
from awkward.numba.array.objects import MethodsNumba as Methods
from awkward.numba.array.objects import ObjectArrayNumba as ObjectArray
from awkward.numba.array.objects import StringArrayNumba as StringArray
from awkward.numba.array.table import TableNumba as Table
from awkward.numba.array.union import UnionArrayNumba as UnionArray
from awkward.numba.array.virtual import VirtualArrayNumba as VirtualArray

# convenient access to the version number
from awkward.version import __version__

__all__ = ["ChunkedArray", "AppendableArray", "IndexedArray", "SparseArray", "JaggedArray", "MaskedArray", "BitMaskedArray", "IndexedMaskedArray", "Methods", "ObjectArray", "Table", "UnionArray", "VirtualArray", "StringArray", "__version__"]
