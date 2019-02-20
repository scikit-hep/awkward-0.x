import numpy, awkward

a = numpy.arange(4**4).reshape(4, 4, 4, 4)
a2 = awkward.fromiter(a)

slices = [2]  # , slice(None), slice(2, 4), slice(None, None, -1), numpy.array([2, 0, 0]), numpy.array([True, False, True, True])]

def getitem_next(array, slices):
    if len(slices) == 0:
        return array

    head = slices[0]
    tail = slices[1:]
    if isinstance(head, int):
        return getitem_next(array.content[array.starts[head]:array.stops[head]], tail)
    else:
        raise NotImplementedError(head)
        
def getitem_enter(array, slices):
    if len(slices) == 0:
        return array

    head = slices[0]
    tail = slices[1:]
    if isinstance(head, int):
        return getitem_next(array.content[array.starts[head]:array.stops[head]], tail)
    else:
        raise NotImplementedError(head)

def check(slices, left, right):
    if left.tolist() != right.tolist():
        print(slices)
        print(left.tolist())
        print(right.tolist())
        raise AssertionError

for x in slices:
    check(slices, a[x,], getitem_enter(a2, (x,)))
    for y in slices:
        check(slices, a[x, y], getitem_enter(a2, (x, y)))
        for z in slices:
            check(slices, a[x, y, z], getitem_enter(a2, (x, y, z)))
