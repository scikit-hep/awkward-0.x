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
import awkward

def spread_advanced(starts, stops, advanced):
    if advanced is None:
        return advanced
    else:
        assert len(starts) == len(stops) == len(advanced)
        counts = stops - starts
        nextadvanced = numpy.full(counts.sum(), 999, int)
        k = 0
        for i in range(len(counts)):
            length = counts[i]
            nextadvanced[k : k + length] = advanced[i]
            k += length
        return nextadvanced

def getitem_integer(array, head, tail, advanced):
    index = numpy.full(len(array.starts), 999, int)
    for i in range(len(array.starts)):
        j = array.starts[i] + head
        if j >= array.stops[i]:
            raise ValueError("integer index is beyond the range of one of the JaggedArray.starts-JaggedArray.stops pairs")
        index[i] = j

    next = getitem_next(array.content[index], tail, advanced)    
    return next

def getitem_slice2(array, head, tail, advanced):
    if (head.start is None or head.start == 0) and head.stop is None:
        next = getitem_next(array.content, tail, advanced)
        return awkward.JaggedArray(array.starts, array.stops, next)

    starts = numpy.full(len(array.starts), 999, int)
    stops = numpy.full(len(array.starts), 999, int)
    for i in range(len(array.starts)):
        length = array.stops[i] - array.starts[i]
        a, b = head.start, head.stop

        if a is None:
            a = 0
        elif a < 0:
            a += length
        if b is None:
            b = length
        elif b < 0:
            b += length

        if b <= a:
            a, b = 0, 0
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

    next = getitem_next(array.content, tail, advanced)
    return awkward.JaggedArray(starts, stops, next)

def getitem_slice3(array, head, tail, advanced):
    if head.step == 0:
        raise ValueError

    offsets = numpy.full(len(array.starts) + 1, 999, int)
    offsets[0] = 0
    index = numpy.full(len(array.content), 999, int)  # too big, but okay
    k = 0
    for i in range(len(array.starts)):
        length = array.stops[i] - array.starts[i]
        a, b, c = head.start, head.stop, head.step
        if c is None:
            c = 1

        if a is None and c > 0:
            a = 0
        elif a is None:
            a = length - 1
        elif a < 0:
            a += length

        if b is None and c > 0:
            b = length
        elif b is None:
            b = -1
        elif b < 0:
            b += length

        if c > 0:
            if b <= a:
                a, b = 0, 0
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
                a, b = 0, 0
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
    next = getitem_next(array.content[index[:k]], tail, spread_advanced(starts, stops, advanced))
    return awkward.JaggedArray(starts, stops, next)

def getitem_intarray_none(array, head, tail, advanced):
    offsets = numpy.full(len(array.starts) + 1, 999, int)
    offsets[0] = 0
    index = numpy.full(len(head)*len(array.starts), 999, int)
    nextadvanced = numpy.full(len(index), 999, int)

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
    next = getitem_next(array.content[index], tail, nextadvanced)
    return awkward.JaggedArray(starts, stops, next)

def getitem_intarray_some(array, head, tail, advanced):
    index = numpy.full(len(array.starts), 999, int)
    nextadvanced = numpy.full(len(index), 999, int)

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

def getitem_next(array, where, advanced):
    if len(where) == 0:
        return array
    if isinstance(array, numpy.ndarray):
        return array[where]
    
    head = where[0]
    tail = where[1:]
    if isinstance(head, int):
        return getitem_integer(array, head, tail, advanced)

    elif isinstance(head, slice) and (head.step is None or head.step == 1) and advanced is None and not any(isinstance(x, numpy.ndarray) for x in where):
        return getitem_slice2(array, head, tail, advanced)

    elif isinstance(head, slice):
        return getitem_slice3(array, head, tail, advanced)

    elif isinstance(head, numpy.ndarray) and issubclass(head.dtype.type, numpy.integer):
        if advanced is None:
            return getitem_intarray_none(array, head, tail, advanced)
        else:
            return getitem_intarray_some(array, head, tail, advanced)

    else:
        raise NotImplementedError(head)

def getitem_enter(array, where):
    if len(where) == 0:
        return array

    arraylen = 0
    for x in where:
        if isinstance(x, numpy.ndarray) and len(x.shape) == 1:
            if issubclass(x.dtype.type, (numpy.bool_, numpy.bool)):
                arraylen = max(arraylen, numpy.count_nonzero(x))
            else:
                arraylen = max(arraylen, len(x))

    newwhere = []
    for x in where:
        if isinstance(x, numpy.ndarray) and len(x.shape) == 1 and issubclass(x.dtype.type, (numpy.bool, numpy.bool_)):
            newwhere.append(numpy.nonzero(x)[0])
        elif isinstance(x, int) and arraylen != 0:
            newwhere.append(numpy.full(arraylen, x, int))
        elif isinstance(x, numpy.ndarray) and x.shape == (1,):
            newwhere.append(numpy.full(arraylen, x, int))
        else:
            newwhere.append(x)

    fake = getitem_next(awkward.JaggedArray([0], [len(array)], array), newwhere, None)
    if isinstance(fake, numpy.ndarray):
        return fake[0]
    else:
        return fake.content[fake.starts[0]:fake.stops[-1]]

slices = [2, slice(None), slice(2, 4), slice(1, None, 2), slice(None, None, -1), numpy.array([2, 0, 0]), numpy.array([3, 1, 2]), numpy.array([True, False, True, True]), numpy.array([True, True, True, False])]

a = numpy.arange(4**4).reshape(4, 4, 4, 4)
a2 = awkward.fromiter(a)

for x in slices:
    print(x)
    assert a[x,].tolist() == getitem_enter(a2, (x,)).tolist()
    for y in slices:
        print(x, y)
        assert a[x, y].tolist() == getitem_enter(a2, (x, y)).tolist()
        for z in slices:
            print(x, y, z)
            assert a[x, y, z].tolist() == getitem_enter(a2, (x, y, z)).tolist()

tmp = a2.content.content.content.tolist()
tmp.insert(4, 999)
a2.content.content.stops = a2.content.content.stops.copy()
a2.content.content.starts[1:] += 1
a2.content.content.stops[1:] += 1
a2.content.content.content = tmp

for x in slices:
    print(x)
    assert a[x,].tolist() == getitem_enter(a2, (x,)).tolist()
    for y in slices:
        print(x, y)
        assert a[x, y].tolist() == getitem_enter(a2, (x, y)).tolist()
        for z in slices:
            print(x, y, z)
            assert a[x, y, z].tolist() == getitem_enter(a2, (x, y, z)).tolist()

a = numpy.arange(4**3).reshape(4, 4, 4)
a2 = awkward.fromiter(a)
for x in slices:
    print(x)
    assert a[x,].tolist() == getitem_enter(a2, (x,)).tolist()
    for y in slices:
        print(x, y)
        assert a[x, y].tolist() == getitem_enter(a2, (x, y)).tolist()
        for z in slices:
            print(x, y, z)
            assert a[x, y, z].tolist() == getitem_enter(a2, (x, y, z)).tolist()
