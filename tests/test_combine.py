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

import unittest

import numpy

from awkward import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    ################### old tests

    # def test_ChunkedArray_ChunkedArray(self):
    #     pass

    # def test_ChunkedArray_PartitionedArray(self):
    #     pass

    # def test_ChunkedArray_AppendableArray(self):
    #     pass

    # def test_ChunkedArray_IndexedArray(self):
    #     pass

    # def test_ChunkedArray_ByteIndexedArray(self):
    #     pass

    # def test_ChunkedArray_IndexedMaskedArray(self):
    #     pass

    # def test_ChunkedArray_UnionArray(self):
    #     pass

    # def test_ChunkedArray_JaggedArray(self):
    #     pass

    # def test_ChunkedArray_ByteJaggedArray(self):
    #     pass

    # def test_ChunkedArray_MaskedArray(self):
    #     pass

    # def test_ChunkedArray_BitMaskedArray(self):
    #     pass

    # def test_ChunkedArray_SparseArray(self):
    #     pass

    # def test_ChunkedArray_Table(self):
    #     pass

    # def test_ChunkedArray_VirtualArray(self):
    #     pass

    # def test_ChunkedArray_VirtualObjectArray(self):
    #     pass

    # def test_PartitionedArray_ChunkedArray(self):
    #     pass

    # def test_PartitionedArray_PartitionedArray(self):
    #     pass

    # def test_PartitionedArray_AppendableArray(self):
    #     pass

    # def test_PartitionedArray_IndexedArray(self):
    #     pass

    # def test_PartitionedArray_ByteIndexedArray(self):
    #     pass

    # def test_PartitionedArray_IndexedMaskedArray(self):
    #     pass

    # def test_PartitionedArray_UnionArray(self):
    #     pass

    # def test_PartitionedArray_JaggedArray(self):
    #     pass

    # def test_PartitionedArray_ByteJaggedArray(self):
    #     pass

    # def test_PartitionedArray_MaskedArray(self):
    #     pass

    # def test_PartitionedArray_BitMaskedArray(self):
    #     pass

    # def test_PartitionedArray_SparseArray(self):
    #     pass

    # def test_PartitionedArray_Table(self):
    #     pass

    # def test_PartitionedArray_VirtualArray(self):
    #     pass

    # def test_PartitionedArray_VirtualObjectArray(self):
    #     pass
    
    # def test_AppendableArray_ChunkedArray(self):
    #     pass

    # def test_AppendableArray_PartitionedArray(self):
    #     pass

    # def test_AppendableArray_AppendableArray(self):
    #     pass

    # def test_AppendableArray_IndexedArray(self):
    #     pass

    # def test_AppendableArray_ByteIndexedArray(self):
    #     pass

    # def test_AppendableArray_IndexedMaskedArray(self):
    #     pass

    # def test_AppendableArray_UnionArray(self):
    #     pass

    # def test_AppendableArray_JaggedArray(self):
    #     pass

    # def test_AppendableArray_ByteJaggedArray(self):
    #     pass

    # def test_AppendableArray_MaskedArray(self):
    #     pass

    # def test_AppendableArray_BitMaskedArray(self):
    #     pass

    # def test_AppendableArray_SparseArray(self):
    #     pass

    # def test_AppendableArray_Table(self):
    #     pass

    # def test_AppendableArray_VirtualArray(self):
    #     pass

    # def test_AppendableArray_VirtualObjectArray(self):
    #     pass
    
    # def test_IndexedArray_ChunkedArray(self):
    #     pass

    # def test_IndexedArray_PartitionedArray(self):
    #     pass

    # def test_IndexedArray_AppendableArray(self):
    #     pass

    # def test_IndexedArray_IndexedArray(self):
    #     pass

    # def test_IndexedArray_ByteIndexedArray(self):
    #     pass

    # def test_IndexedArray_IndexedMaskedArray(self):
    #     pass

    # def test_IndexedArray_UnionArray(self):
    #     pass

    # def test_IndexedArray_JaggedArray(self):
    #     pass

    # def test_IndexedArray_ByteJaggedArray(self):
    #     pass

    # def test_IndexedArray_MaskedArray(self):
    #     pass

    # def test_IndexedArray_BitMaskedArray(self):
    #     pass

    # def test_IndexedArray_SparseArray(self):
    #     pass

    # def test_IndexedArray_Table(self):
    #     pass

    # def test_IndexedArray_VirtualArray(self):
    #     pass

    # def test_IndexedArray_VirtualObjectArray(self):
    #     pass
    
    # def test_ByteIndexedArray_ChunkedArray(self):
    #     pass

    # def test_ByteIndexedArray_PartitionedArray(self):
    #     pass

    # def test_ByteIndexedArray_AppendableArray(self):
    #     pass

    # def test_ByteIndexedArray_IndexedArray(self):
    #     pass

    # def test_ByteIndexedArray_ByteIndexedArray(self):
    #     pass

    # def test_ByteIndexedArray_IndexedMaskedArray(self):
    #     pass

    # def test_ByteIndexedArray_UnionArray(self):
    #     pass

    # def test_ByteIndexedArray_JaggedArray(self):
    #     pass

    # def test_ByteIndexedArray_ByteJaggedArray(self):
    #     pass

    # def test_ByteIndexedArray_MaskedArray(self):
    #     pass

    # def test_ByteIndexedArray_BitMaskedArray(self):
    #     pass

    # def test_ByteIndexedArray_SparseArray(self):
    #     pass

    # def test_ByteIndexedArray_Table(self):
    #     pass

    # def test_ByteIndexedArray_VirtualArray(self):
    #     pass

    # def test_ByteIndexedArray_VirtualObjectArray(self):
    #     pass
    
    # def test_IndexedMaskedArray_ChunkedArray(self):
    #     pass

    # def test_IndexedMaskedArray_PartitionedArray(self):
    #     pass

    # def test_IndexedMaskedArray_AppendableArray(self):
    #     pass

    # def test_IndexedMaskedArray_IndexedArray(self):
    #     pass

    # def test_IndexedMaskedArray_ByteIndexedArray(self):
    #     pass

    # def test_IndexedMaskedArray_IndexedMaskedArray(self):
    #     pass

    # def test_IndexedMaskedArray_UnionArray(self):
    #     pass

    # def test_IndexedMaskedArray_JaggedArray(self):
    #     pass

    # def test_IndexedMaskedArray_ByteJaggedArray(self):
    #     pass

    # def test_IndexedMaskedArray_MaskedArray(self):
    #     pass

    # def test_IndexedMaskedArray_BitMaskedArray(self):
    #     pass

    # def test_IndexedMaskedArray_SparseArray(self):
    #     pass

    # def test_IndexedMaskedArray_Table(self):
    #     pass

    # def test_IndexedMaskedArray_VirtualArray(self):
    #     pass

    # def test_IndexedMaskedArray_VirtualObjectArray(self):
    #     pass

    # def test_UnionArray_ChunkedArray(self):
    #     pass

    # def test_UnionArray_PartitionedArray(self):
    #     pass

    # def test_UnionArray_AppendableArray(self):
    #     pass

    # def test_UnionArray_IndexedArray(self):
    #     pass

    # def test_UnionArray_ByteIndexedArray(self):
    #     pass

    # def test_UnionArray_IndexedMaskedArray(self):
    #     pass

    # def test_UnionArray_UnionArray(self):
    #     pass

    # def test_UnionArray_JaggedArray(self):
    #     pass

    # def test_UnionArray_ByteJaggedArray(self):
    #     pass

    # def test_UnionArray_MaskedArray(self):
    #     pass

    # def test_UnionArray_BitMaskedArray(self):
    #     pass

    # def test_UnionArray_SparseArray(self):
    #     pass

    # def test_UnionArray_Table(self):
    #     pass

    # def test_UnionArray_VirtualArray(self):
    #     pass

    # def test_UnionArray_VirtualObjectArray(self):
    #     pass
    
    # def test_JaggedArray_ChunkedArray(self):
    #     a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], ChunkedArray([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7, 8.8, 9.9]]))
    #     assert a.tolist() == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]
    #     assert a[2].tolist() == [3.3, 4.4]
    #     assert a[2:4].tolist() == [[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]

    # def test_JaggedArray_PartitionedArray(self):
    #     a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], PartitionedArray([0, 4, 10], [[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7, 8.8, 9.9]]))
    #     assert a.tolist() == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]
    #     assert a[2].tolist() == [3.3, 4.4]
    #     assert a[2:4].tolist() == [[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]

    # def test_JaggedArray_AppendableArray(self):
    #     pass

    # def test_JaggedArray_IndexedArray(self):
    #     a = JaggedArray([0, 3, 3], [3, 3, 5], IndexedArray([9, 8, 7, 4, 4], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    #     assert a.tolist() == [[9.9, 8.8, 7.7], [], [4.4, 4.4]]

    # def test_JaggedArray_ByteIndexedArray(self):
    #     pass

    # def test_JaggedArray_IndexedMaskedArray(self):
    #     pass

    # def test_JaggedArray_UnionArray(self):
    #     pass

    # def test_JaggedArray_JaggedArray(self):
    #     a = JaggedArray([0, 2, 2], [2, 2, 4], JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    #     assert a.tolist() == [[[0.0, 1.1, 2.2], []], [], [[3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]]

    # def test_JaggedArray_ByteJaggedArray(self):
    #     pass

    # def test_JaggedArray_MaskedArray(self):
    #     a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], MaskedArray([False, False, True, True, False, False, False, False, True, False], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    #     assert a.tolist() == [[0.0, 1.1, None], [], [None, 4.4], [5.5, 6.6, 7.7, None, 9.9]]

    # def test_JaggedArray_BitMaskedArray(self):
    #     pass

    # def test_JaggedArray_SparseArray(self):
    #     pass

    # def test_JaggedArray_Table(self):
    #     a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], Table(10, one=[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], two=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900]))
    #     assert a.tolist() == [[{"one": 0.0, "two": 0}, {"one": 1.1, "two": 100}, {"one": 2.2, "two": 200}], [], [{"one": 3.3, "two": 300}, {"one": 4.4, "two": 400}], [{"one": 5.5, "two": 500}, {"one": 6.6, "two": 600}, {"one": 7.7, "two": 700}, {"one": 8.8, "two": 800}, {"one": 9.9, "two": 900}]]
    #     assert a["one"].tolist() == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]
    #     assert a["two"].tolist() == [[0, 100, 200], [], [300, 400], [500, 600, 700, 800, 900]]

    # def test_JaggedArray_VirtualArray(self):
    #     a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], VirtualArray(lambda: [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    #     assert a.tolist() == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5, 6.6, 7.7, 8.8, 9.9]]

    # def test_JaggedArray_VirtualObjectArray(self):
    #     a = JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], VirtualObjectArray(str, [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    #     assert a.tolist() == [["0.0", "1.1", "2.2"], [], ["3.3", "4.4"], ["5.5", "6.6", "7.7", "8.8", "9.9"]]
    
    # def test_ByteJaggedArray_ChunkedArray(self):
    #     pass

    # def test_ByteJaggedArray_PartitionedArray(self):
    #     pass

    # def test_ByteJaggedArray_AppendableArray(self):
    #     pass

    # def test_ByteJaggedArray_IndexedArray(self):
    #     pass

    # def test_ByteJaggedArray_ByteIndexedArray(self):
    #     pass

    # def test_ByteJaggedArray_IndexedMaskedArray(self):
    #     pass

    # def test_ByteJaggedArray_UnionArray(self):
    #     pass

    # def test_ByteJaggedArray_JaggedArray(self):
    #     pass

    # def test_ByteJaggedArray_ByteJaggedArray(self):
    #     pass

    # def test_ByteJaggedArray_MaskedArray(self):
    #     pass

    # def test_ByteJaggedArray_BitMaskedArray(self):
    #     pass

    # def test_ByteJaggedArray_SparseArray(self):
    #     pass

    # def test_ByteJaggedArray_Table(self):
    #     pass

    # def test_ByteJaggedArray_VirtualArray(self):
    #     pass

    # def test_ByteJaggedArray_VirtualObjectArray(self):
    #     pass
    
    # def test_MaskedArray_ChunkedArray(self):
    #     pass

    # def test_MaskedArray_PartitionedArray(self):
    #     pass

    # def test_MaskedArray_AppendableArray(self):
    #     pass

    # def test_MaskedArray_IndexedArray(self):
    #     pass

    # def test_MaskedArray_ByteIndexedArray(self):
    #     pass

    # def test_MaskedArray_IndexedMaskedArray(self):
    #     pass

    # def test_MaskedArray_UnionArray(self):
    #     pass

    # def test_MaskedArray_JaggedArray(self):
    #     a = MaskedArray([True, False, False, True], JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    #     assert a.tolist() == [None, [], [3.3, 4.4], None]

    # def test_MaskedArray_ByteJaggedArray(self):
    #     pass

    # def test_MaskedArray_MaskedArray(self):
    #     a = MaskedArray([False, False, False, False, False, True, True, True, True, True], MaskedArray([False, True, False, True, False, True, False, True, False, True], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    #     assert a.tolist() == [0.0, None, 2.2, None, 4.4, None, None, None, None, None]

    # def test_MaskedArray_BitMaskedArray(self):
    #     pass

    # def test_MaskedArray_SparseArray(self):
    #     pass

    # def test_MaskedArray_Table(self):
    #     a = MaskedArray([False, True, False, True, False, True, False, True, False, True], Table(10, one=[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], two=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900]))
    #     assert a[2]["one"], 2 == 2
    #     assert a[2]["two"] == 200
    #     assert a.tolist() == [{"one": 0.0, "two": 0}, None, {"one": 2.2, "two": 200}, None, {"one": 4.4, "two": 400}, None, {"one": 6.6, "two": 600}, None, {"one": 8.8, "two": 800}, None]

    # def test_MaskedArray_VirtualArray(self):
    #     pass

    # def test_MaskedArray_VirtualObjectArray(self):
    #     pass
    
    # def test_BitMaskedArray_ChunkedArray(self):
    #     pass

    # def test_BitMaskedArray_PartitionedArray(self):
    #     pass

    # def test_BitMaskedArray_AppendableArray(self):
    #     pass

    # def test_BitMaskedArray_IndexedArray(self):
    #     pass

    # def test_BitMaskedArray_ByteIndexedArray(self):
    #     pass

    # def test_BitMaskedArray_IndexedMaskedArray(self):
    #     pass

    # def test_BitMaskedArray_UnionArray(self):
    #     pass

    # def test_BitMaskedArray_JaggedArray(self):
    #     pass

    # def test_BitMaskedArray_ByteJaggedArray(self):
    #     pass

    # def test_BitMaskedArray_MaskedArray(self):
    #     pass

    # def test_BitMaskedArray_BitMaskedArray(self):
    #     pass

    # def test_BitMaskedArray_SparseArray(self):
    #     pass

    # def test_BitMaskedArray_Table(self):
    #     pass

    # def test_BitMaskedArray_VirtualArray(self):
    #     pass

    # def test_BitMaskedArray_VirtualObjectArray(self):
    #     pass
    
    # def test_SparseArray_ChunkedArray(self):
    #     pass

    # def test_SparseArray_PartitionedArray(self):
    #     pass

    # def test_SparseArray_AppendableArray(self):
    #     pass

    # def test_SparseArray_IndexedArray(self):
    #     pass

    # def test_SparseArray_ByteIndexedArray(self):
    #     pass

    # def test_SparseArray_IndexedMaskedArray(self):
    #     pass

    # def test_SparseArray_UnionArray(self):
    #     pass

    # def test_SparseArray_JaggedArray(self):
    #     pass

    # def test_SparseArray_ByteJaggedArray(self):
    #     pass

    # def test_SparseArray_MaskedArray(self):
    #     pass

    # def test_SparseArray_BitMaskedArray(self):
    #     pass

    # def test_SparseArray_SparseArray(self):
    #     pass

    # def test_SparseArray_Table(self):
    #     pass

    # def test_SparseArray_VirtualArray(self):
    #     pass

    # def test_SparseArray_VirtualObjectArray(self):
    #     pass
    
    # def test_Table_ChunkedArray(self):
    #     a = Table(10, one=ChunkedArray([[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7, 8.8, 9.9]]), two=ChunkedArray([[0, 100, 200, 300, 400, 500, 600], [700, 800, 900]]))
    #     assert a.tolist() == [{"one": 0.0, "two": 0}, {"one": 1.1, "two": 100}, {"one": 2.2, "two": 200}, {"one": 3.3, "two": 300}, {"one": 4.4, "two": 400}, {"one": 5.5, "two": 500}, {"one": 6.6, "two": 600}, {"one": 7.7, "two": 700}, {"one": 8.8, "two": 800}, {"one": 9.9, "two": 900}]

    # def test_Table_PartitionedArray(self):
    #     a = Table(10, one=PartitionedArray([0, 4, 10], [[0.0, 1.1, 2.2, 3.3], [4.4, 5.5, 6.6, 7.7, 8.8, 9.9]]), two=PartitionedArray([0, 7, 10], [[0, 100, 200, 300, 400, 500, 600], [700, 800, 900]]))
    #     assert a.tolist() == [{"one": 0.0, "two": 0}, {"one": 1.1, "two": 100}, {"one": 2.2, "two": 200}, {"one": 3.3, "two": 300}, {"one": 4.4, "two": 400}, {"one": 5.5, "two": 500}, {"one": 6.6, "two": 600}, {"one": 7.7, "two": 700}, {"one": 8.8, "two": 800}, {"one": 9.9, "two": 900}]

    # def test_Table_AppendableArray(self):
    #     pass

    # def test_Table_IndexedArray(self):
    #     a = Table(5, one=IndexedArray([8, 6, 4, 2, 0], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]), two=IndexedArray([0, 1, 2, 2, 1], [0, 100, 200]))
    #     assert a.tolist() == [{"one": 8.8, "two": 0}, {"one": 6.6, "two": 100}, {"one": 4.4, "two": 200}, {"one": 2.2, "two": 200}, {"one": 0.0, "two": 100}]

    # def test_Table_ByteIndexedArray(self):
    #     pass

    # def test_Table_IndexedMaskedArray(self):
    #     pass

    # def test_Table_UnionArray(self):
    #     pass

    # def test_Table_JaggedArray(self):
    #     a = Table(4, one=JaggedArray([0, 3, 3, 5], [3, 3, 5, 10], [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]), two=[0, 100, 200, 300])
    #     assert a.tolist() == [{"one": [0.0, 1.1, 2.2], "two": 0}, {"one": [], "two": 100}, {"one": [3.3, 4.4], "two": 200}, {"one": [5.5, 6.6, 7.7, 8.8, 9.9], "two": 300}]

    # def test_Table_ByteJaggedArray(self):
    #     pass

    # def test_Table_MaskedArray(self):
    #     pass

    # def test_Table_BitMaskedArray(self):
    #     pass

    # def test_Table_SparseArray(self):
    #     pass

    # def test_Table_Table(self):
    #     a = Table(5, one=[0.0, 1.1, 2.2, 3.3, 4.4], two=Table(5, x=[0, 100, 200, 300, 400], y=[False, True, False, True, False]))
    #     assert a.tolist() == [{"one": 0.0, "two": {"x": 0, "y": False}}, {"one": 1.1, "two": {"x": 100, "y": True}}, {"one": 2.2, "two": {"x": 200, "y": False}}, {"one": 3.3, "two": {"x": 300, "y": True}}, {"one": 4.4, "two": {"x": 400, "y": False}}]

    # def test_Table_VirtualArray(self):
    #     pass

    # def test_Table_VirtualObjectArray(self):
    #     pass
    
    # def test_VirtualArray_ChunkedArray(self):
    #     pass

    # def test_VirtualArray_PartitionedArray(self):
    #     pass

    # def test_VirtualArray_AppendableArray(self):
    #     pass

    # def test_VirtualArray_IndexedArray(self):
    #     pass

    # def test_VirtualArray_ByteIndexedArray(self):
    #     pass

    # def test_VirtualArray_IndexedMaskedArray(self):
    #     pass

    # def test_VirtualArray_UnionArray(self):
    #     pass

    # def test_VirtualArray_JaggedArray(self):
    #     pass

    # def test_VirtualArray_ByteJaggedArray(self):
    #     pass

    # def test_VirtualArray_MaskedArray(self):
    #     pass

    # def test_VirtualArray_BitMaskedArray(self):
    #     pass

    # def test_VirtualArray_SparseArray(self):
    #     pass

    # def test_VirtualArray_Table(self):
    #     pass

    # def test_VirtualArray_VirtualArray(self):
    #     pass

    # def test_VirtualArray_VirtualObjectArray(self):
    #     pass
    
    # def test_VirtualObjectArray_ChunkedArray(self):
    #     pass

    # def test_VirtualObjectArray_PartitionedArray(self):
    #     pass

    # def test_VirtualObjectArray_AppendableArray(self):
    #     pass

    # def test_VirtualObjectArray_IndexedArray(self):
    #     pass

    # def test_VirtualObjectArray_ByteIndexedArray(self):
    #     pass

    # def test_VirtualObjectArray_IndexedMaskedArray(self):
    #     pass

    # def test_VirtualObjectArray_UnionArray(self):
    #     pass

    # def test_VirtualObjectArray_JaggedArray(self):
    #     pass

    # def test_VirtualObjectArray_ByteJaggedArray(self):
    #     pass

    # def test_VirtualObjectArray_MaskedArray(self):
    #     pass

    # def test_VirtualObjectArray_BitMaskedArray(self):
    #     pass

    # def test_VirtualObjectArray_SparseArray(self):
    #     pass

    # def test_VirtualObjectArray_Table(self):
    #     pass

    # def test_VirtualObjectArray_VirtualArray(self):
    #     pass

    # def test_VirtualObjectArray_VirtualObjectArray(self):
    #     pass
