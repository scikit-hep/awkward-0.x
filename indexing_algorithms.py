import numpy, awkward

a = numpy.arange(4**3).reshape(4, 4, 4)
a2 = awkward.fromiter(a)

slices = [2, slice(None), slice(2, 4), slice(None, None, -1)]  # , numpy.array([2, 0, 0]), numpy.array([True, False, True, True])]

def getitem_integer(array, head, tail):
    if isinstance(array, numpy.ndarray):
        return array[(head,) + tail]

    index = numpy.empty(len(array.starts), int)
    for i in range(len(array.starts)):
        j = array.starts[i] + head
        if j >= array.stops[i]:
            raise ValueError("integer index is beyond the range of one of the JaggedArray.starts-JaggedArray.stops pairs")
        index[i] = j

    next = getitem_next(array.content[index], tail)    
    return next

def getitem_slice3(array, head, tail):
    if isinstance(array, numpy.ndarray):
        return array[(head,) + tail]

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

    next = getitem_next(array.content[index[:k]], tail)
    return awkward.JaggedArray(starts, stops, next)

def getitem_next(array, slices):
    if len(slices) == 0:
        return array
    
    head = slices[0]
    tail = slices[1:]
    if isinstance(head, int):
        return getitem_integer(array, head, tail)

    elif isinstance(head, slice):
        return getitem_slice3(array, head, tail)

    else:
        raise NotImplementedError(head)

def getitem_enter(array, slices):
    if len(slices) == 0:
        return array
    if isinstance(array, numpy.ndarray):
        return array[slices]

    head = slices[0]
    tail = slices[1:]
    if isinstance(head, int):
        return getitem_enter(array.content[array.starts[head]:array.stops[head]], tail)

    elif isinstance(head, slice):
        return getitem_next(awkward.JaggedArray(array.starts[head], array.stops[head], array.content), tail)

    else:
        raise NotImplementedError(head)

def check(slices, left, right):
    print(slices)
    if left.tolist() != right.tolist():
        print(left.tolist())
        print(right.tolist())
        raise AssertionError

for x in slices:
    check((x,), a[x,], getitem_enter(a2, (x,)))
    for y in slices:
        check((x, y), a[x, y], getitem_enter(a2, (x, y)))
        for z in slices:
            check((x, y, z), a[x, y, z], getitem_enter(a2, (x, y, z)))
