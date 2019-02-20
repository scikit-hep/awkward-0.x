import numpy, awkward

a = numpy.arange(4**4).reshape(4, 4, 4, 4)
a2 = awkward.fromiter(a)

slices = [2, slice(None), slice(2, 4), slice(1, None, 2), slice(None, None, -1), numpy.array([2, 0, 0]), numpy.array([3, 1, 2]), numpy.array([True, False, True, True]), numpy.array([True, True, True, False])]

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

def getitem_slice3(array, head, tail, advanced):
    if head.step == 0:
        raise ValueError
    starts = numpy.full(len(array.starts), 999, int)
    stops = numpy.full(len(array.stops), 999, int)
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

        starts[i] = k
        for j in range(a, b, c):
            index[k] = array.starts[i] + j
            k += 1
        stops[i] = k

    next = getitem_next(array.content[index[:k]], tail, spread_advanced(starts, stops, advanced))
    return awkward.JaggedArray(starts, stops, next)

def getitem_intarray_none(array, head, tail, advanced):
    starts = numpy.full(len(array.starts), 999, int)
    stops = numpy.full(len(array.stops), 999, int)
    index = numpy.full(len(head)*len(array.starts), 999, int)
    nextadvanced = numpy.full(len(index), 999, int)

    k = 0
    for i in range(len(array.starts)):
        length = array.stops[i] - array.starts[i]

        starts[i] = k
        for j in range(len(head)):
            norm = head[j]
            if norm < 0:
                norm += length
            if norm < 0 or norm >= length:
                raise IndexError("advanced index is out of bounds in JaggedArray")
            index[k] = array.starts[i] + norm
            nextadvanced[k] = j
            k += 1
        stops[i] = k

    next = getitem_next(array.content[index], tail, nextadvanced)
    return awkward.JaggedArray(starts, stops, next)

def getitem_intarray_some(array, head, tail, advanced):
    index = numpy.full(len(array.starts), 999, int)
    nextadvanced = numpy.full(len(index), 999, int)

    for i in range(len(advanced)):
        length = array.stops[i] - array.starts[i]
        if advanced[i] >= len(head):
            raise IndexError("advanced index lengths do not match")
        normj = head[advanced[i]]
        if normj < 0:
            normj += length
        if normj < 0 or normj >= length:
            raise IndexError("advanced index is out of bounds in JaggedArray")
        index[i] = array.starts[i] + normj
        nextadvanced[i] = i

    next = getitem_next(array.content[index], tail, nextadvanced)
    return next

def getitem_next(array, slices, advanced):
    if len(slices) == 0:
        return array
    if isinstance(array, numpy.ndarray):
        return array[slices]
    
    head = slices[0]
    tail = slices[1:]
    if isinstance(head, int):
        return getitem_integer(array, head, tail, advanced)

    elif isinstance(head, slice):
        return getitem_slice3(array, head, tail, advanced)

    elif isinstance(head, numpy.ndarray) and issubclass(head.dtype.type, numpy.integer):
        if advanced is None:
            return getitem_intarray_none(array, head, tail, advanced)
        else:
            return getitem_intarray_some(array, head, tail, advanced)

    else:
        raise NotImplementedError(head)

def getitem_enter(array, slices):
    if len(slices) == 0:
        return array

    arraylen = 0
    for x in slices:
        if isinstance(x, numpy.ndarray) and len(x.shape) == 1:
            if issubclass(x.dtype.type, (numpy.bool_, numpy.bool)):
                arraylen = max(arraylen, numpy.count_nonzero(x))
            else:
                arraylen = max(arraylen, len(x))

    newslices = []
    for x in slices:
        if isinstance(x, numpy.ndarray) and len(x.shape) == 1 and issubclass(x.dtype.type, (numpy.bool, numpy.bool_)):
            newslices.append(numpy.nonzero(x)[0])
        elif isinstance(x, int) and arraylen != 0:
            newslices.append(numpy.full(arraylen, x, int))
        elif isinstance(x, numpy.ndarray) and x.shape == (1,):
            newslices.append(numpy.full(arraylen, x, int))
        else:
            newslices.append(x)

    fake = getitem_next(awkward.JaggedArray([0], [len(array)], array), newslices, None)
    if isinstance(fake, numpy.ndarray):
        return fake[0]
    else:
        return fake.content[fake.starts[0]:fake.stops[-1]]

def check(left, right):
    if left.tolist() != right.tolist():
        print(left.tolist())
        print(right.tolist())
        raise AssertionError

for x in slices:
    print(x)
    check(a[x,], getitem_enter(a2, (x,)))
    for y in slices:
        print(x, y)
        check(a[x, y], getitem_enter(a2, (x, y)))
        for z in slices:
            print(x, y, z)
            check(a[x, y, z], getitem_enter(a2, (x, y, z)))
