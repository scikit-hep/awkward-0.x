#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-0.x/blob/master/LICENSE

import functools
import math
import numbers
import operator
from collections import OrderedDict
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward0.array.base
import awkward0.persist
import awkward0.type
import awkward0.util

class JaggedArray(awkward0.array.base.AwkwardArrayWithContent):
    """
    JaggedArray
    """

    @classmethod
    def offsetsaliased(cls, starts, stops):
        if not hasattr(starts, "base") or starts.base is None:
            starts_base = None
        else:
            starts_base = cls._util_toarray(starts.base, cls.INDEXTYPE)
        if not hasattr(stops, "base") or stops.base is None:
            stops_base = None
        else:
            stops_base = cls._util_toarray(stops.base, cls.INDEXTYPE)
        return (isinstance(starts, cls.numpy.ndarray) and isinstance(stops, cls.numpy.ndarray) and
                starts_base is not None and stops_base is not None and starts_base is stops_base and
                starts.ctypes.data == starts_base.ctypes.data and
                stops.ctypes.data == stops_base.ctypes.data + stops.dtype.itemsize and
                len(starts) == len(starts_base) - 1 and
                len(stops) == len(stops_base) - 1)

    @classmethod
    def counts2offsets(cls, counts):
        offsets = cls.numpy.empty(len(counts) + 1, dtype=cls.JaggedArray.fget(None).INDEXTYPE)
        offsets[0] = 0
        cls.numpy.cumsum(counts, out=offsets[1:])
        return offsets

    @classmethod
    def offsets2parents(cls, offsets):
        dtype = cls.JaggedArray.fget(None).INDEXTYPE
        counts = cls.numpy.diff(cls.numpy.r_[dtype.type(0), offsets])  # numpy >= 1.16: diff prepend
        indices = cls.numpy.arange(-1, len(offsets) - 1, dtype=dtype)
        return cls.numpy.repeat(indices, awkward0.util.windows_safe(counts))

    @classmethod
    def startsstops2parents(cls, starts, stops):
        assert starts.shape == stops.shape
        starts = starts.reshape(-1)  # flatten in case multi-d jagged
        stops = stops.reshape(-1)
        dtype = cls.JaggedArray.fget(None).INDEXTYPE
        out = cls.numpy.full(stops.max(), -1, dtype=dtype)
        indices = cls.numpy.arange(len(starts), dtype=dtype)
        counts = stops - starts
        parents = cls.numpy.repeat(indices, awkward0.util.windows_safe(counts))
        contiguous_offsets = cls.JaggedArray.fget(None).counts2offsets(counts)
        content_indices = cls.numpy.arange(contiguous_offsets[-1])
        content_indices -= cls.numpy.repeat(contiguous_offsets[:-1], awkward0.util.windows_safe(counts))
        content_indices += starts[parents]
        cls.numpy.put(out, awkward0.util.windows_safe(content_indices), parents)
        return out

    @classmethod
    def parents2startsstops(cls, parents, length=None):
        # FIXME for 1.0: use length to add empty lists at the end of the jagged array or truncate
        # assumes that children are contiguous, but not necessarily in order or fully covering (allows empty lists)
        tmp = cls.numpy.nonzero(parents[1:] != parents[:-1])[0] + 1
        changes = cls.numpy.empty(len(tmp) + 2, dtype=cls.JaggedArray.fget(None).INDEXTYPE)
        changes[0] = 0
        changes[-1] = len(parents)
        changes[1:-1] = tmp

        length = parents.max() + 1 if len(parents) > 0 else 0
        starts = cls.numpy.zeros(length, dtype=cls.JaggedArray.fget(None).INDEXTYPE)
        counts = cls.numpy.zeros(length, dtype=cls.JaggedArray.fget(None).INDEXTYPE)

        where = parents[changes[:-1]]
        real = (where >= 0)

        starts[where[real]] = (changes[:-1])[real]
        counts[where[real]] = (changes[1:] - changes[:-1])[real]

        return starts, starts + counts

    @classmethod
    def uniques2offsetsparents(cls, uniques):
        # assumes that children are contiguous, in order, and fully covering (can't have empty lists)
        # values are ignored, apart from uniqueness
        changes = cls.numpy.nonzero(uniques[1:] != uniques[:-1])[0] + 1

        offsets = cls.numpy.empty(len(changes) + 2, dtype=cls.JaggedArray.fget(None).INDEXTYPE)
        offsets[0] = 0
        offsets[-1] = len(uniques)
        offsets[1:-1] = changes

        parents = cls.numpy.zeros(len(uniques), dtype=cls.JaggedArray.fget(None).INDEXTYPE)
        parents[changes] = 1
        cls.numpy.cumsum(parents, out=parents)

        return offsets, parents

    def __init__(self, starts, stops, content):
        if self.offsetsaliased(starts, stops):
            self.content = content
            self._starts, self._stops = starts, stops
            self._offsets = starts.base
            self._counts, self._parents = None, None
            self._isvalid = False

            if not self._util_isintegertype(self._offsets.dtype.type):
                raise TypeError("offsets must have integer dtype")
            if len(self._offsets.shape) != 1:
                raise ValueError("offsets must be a one-dimensional array")
            if len(self._offsets) == 0:
                raise ValueError("offsets must be a non-empty array")
            if (self._offsets < 0).any():
                raise ValueError("offsets must be a non-negative array")

        else:
            self.starts = starts
            self.stops = stops
            self.content = content

    @classmethod
    def fromiter(cls, iterable):
        import awkward0.generate
        if len(iterable) == 0:
            return cls.JaggedArray.fget(None)([], [], [])
        else:
            return awkward0.generate.fromiter(iterable)

    @classmethod
    def fromoffsets(cls, offsets, content):
        offsets = cls._util_toarray(offsets, cls.INDEXTYPE, cls.numpy.ndarray)
        if hasattr(offsets.base, 'base') and type(offsets.base.base).__module__ == "pyarrow.lib" and type(offsets.base.base).__name__ == "Buffer":
            # special exception to prevent copy in awkward0.fromarrow
            pass
        elif offsets.base is not None:
            # We rely on the starts,stops slices to be views.
            # If offsets is already a view, the base will not be offsets but its underlying base.
            # Make a copy to prevent that
            offsets = offsets.copy()
        return cls(offsets[:-1], offsets[1:], content)

    @classmethod
    def fromcounts(cls, counts, content):
        counts = cls._util_toarray(counts, cls.INDEXTYPE, cls.numpy.ndarray)
        if not cls._util_isintegertype(counts.dtype.type):
            raise TypeError("counts must have integer dtype")
        if (counts < 0).any():
            raise ValueError("counts must be a non-negative array")
        offsets = cls.counts2offsets(counts.reshape(-1))
        out = cls(offsets[:-1].reshape(counts.shape), offsets[1:].reshape(counts.shape), content)
        out._offsets = offsets if len(counts.shape) == 1 else None
        out._counts = counts
        return out

    @classmethod
    def fromparents(cls, parents, content, length=None):
        parents = cls._util_toarray(parents, cls.INDEXTYPE, cls.numpy.ndarray)
        if not cls._util_isintegertype(parents.dtype.type):
            raise TypeError("parents must have integer dtype")
        if len(parents.shape) != 1 or len(parents) != len(content):
            raise ValueError("parents array must be one-dimensional with the same length as content")
        starts, stops = cls.parents2startsstops(parents, length=length)
        out = cls(starts, stops, content)
        out._parents = parents
        return out

    @classmethod
    def fromuniques(cls, uniques, content):
        uniques = cls._util_toarray(uniques, cls.INDEXTYPE, cls.numpy.ndarray)
        if not cls._util_isintegertype(uniques.dtype.type):
            raise TypeError("uniques must have integer dtype")
        if len(uniques.shape) != 1 or len(uniques) != len(content):
            raise ValueError("uniques array must be one-dimensional with the same length as content")
        offsets, parents = cls.uniques2offsetsparents(uniques)
        out = cls.fromoffsets(offsets, content)
        out._parents = parents
        return out

    @classmethod
    def fromlocalindex(cls, index, content, validate=True):
        index = cls._util_toarray(index, cls.INDEXTYPE, (cls.numpy.ndarray, JaggedArray))
        original_counts = None
        if isinstance(index, JaggedArray):
            if validate:
                original_counts = index.counts
            index = index.flatten()

        if not cls._util_isintegertype(index.dtype.type):
            raise TypeError("localindex must have integer dtype")
        if len(index.shape) != 1 or len(index) != len(content):
            raise ValueError("localindex array must be one-dimensional with the same length as content")

        if validate:
            if not ((index[1:] - index[:-1])[(index != 0)[1:]] == 1).all():
                raise ValueError("every localindex that is not zero must be one greater than the previous")

        starts = cls.numpy.nonzero(index == 0)[0]
        offsets = cls.numpy.empty(len(starts) + 1, dtype=cls.INDEXTYPE)
        offsets[:-1] = starts
        offsets[-1] = len(index)
        if original_counts is not None:
            if not cls.numpy.array_equal(offsets[1:] - starts, original_counts):
                raise ValueError("jagged structure of index does not match jagged structure derived from localindex")

        return cls.fromoffsets(offsets, content)

    @classmethod
    def fromjagged(cls, jagged):
        return cls(jagged._starts, jagged._stops, jagged._content)

    @classmethod
    def fromregular(cls, regular):
        regular = cls._util_toarray(regular, cls.DEFAULTTYPE, cls.numpy.ndarray)
        shape = regular.shape
        if len(shape) <= 1:
            raise ValueError("regular array must have more than one dimension")
        out = regular.reshape(-1)
        for x in shape[:0:-1]:
            out = cls.fromfolding(out, x)
        return out

    @classmethod
    def fromfolding(cls, content, size):
        content = cls._util_toarray(content, cls.DEFAULTTYPE)
        quotient = -(-len(content) // size)
        offsets = cls.numpy.arange(0, quotient * size + 1, size, dtype=cls.INDEXTYPE)
        if len(offsets) > 0:
            offsets[-1] = len(content)
        return cls.fromoffsets(offsets, content)

    def copy(self, starts=None, stops=None, content=None):
        out = self.__class__.__new__(self.__class__)
        out._starts  = self._starts
        out._stops   = self._stops
        out._content = self._content
        out._offsets = self._offsets
        out._counts  = self._counts
        out._parents = self._parents
        out._isvalid = self._isvalid
        if starts is not None:
            out.starts = starts
        if stops is not None:
            out.stops = stops
        if content is not None:
            out.content = content
        return out

    def deepcopy(self, starts=None, stops=None, content=None):
        out = self.copy(starts=starts, stops=stops, content=content)
        out._starts  = self._util_deepcopy(out._starts)
        out._stops   = self._util_deepcopy(out._stops)
        out._content = self._util_deepcopy(out._content)
        out._offsets = self._util_deepcopy(out._offsets)
        out._counts  = self._util_deepcopy(out._counts)
        out._parents = self._util_deepcopy(out._parents)
        return out

    def empty_like(self, **overrides):
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.empty_like(self._content))
        else:
            return self.copy(content=self._content.empty_like(**overrides))

    def zeros_like(self, **overrides):
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.zeros_like(self._content))
        else:
            return self.copy(content=self._content.zeros_like(**overrides))

    def ones_like(self, **overrides):
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.ones_like(self._content))
        else:
            return self.copy(content=self._content.ones_like(**overrides))

    def __awkward_serialize__(self, serializer):
        if self._canuseoffset() and len(self) > 0:
            return serializer.encode_call(
                ["awkward0", "JaggedArray", "fromcounts"],
                serializer(self.counts, "JaggedArray.counts"),
                serializer(self._content[self._starts[0]:self._stops[-1]], "JaggedArray.content"),
            )
        else:
            return serializer.encode_call(
                ["awkward0", "JaggedArray"],
                serializer(self._starts, "JaggedArray.starts"),
                serializer(self._stops, "JaggedArray.stops"),
                serializer(self._content, "JaggedArray.content"),
            )

    @property
    def starts(self):
        return self._starts

    @starts.setter
    def starts(self, value):
        value = self._util_toarray(value, self.INDEXTYPE, self.numpy.ndarray)
        if self.check_prop_valid:
            if not self._util_isintegertype(value.dtype.type):
                raise TypeError("starts must have integer dtype")
            if len(value.shape) == 0:
                raise ValueError("starts must have at least one dimension")
            if (value < 0).any():
                raise ValueError("starts must be a non-negative array")
        self._starts = value
        self._offsets, self._counts, self._parents = None, None, None
        self._isvalid = False

    @property
    def stops(self):
        if len(self._stops) == len(self._starts):
            return self._stops
        else:
            return self._stops[:len(self._starts)]

    @stops.setter
    def stops(self, value):
        value = self._util_toarray(value, self.INDEXTYPE, self.numpy.ndarray)
        if self.check_prop_valid:
            if not self._util_isintegertype(value.dtype.type):
                raise TypeError("stops must have integer dtype")
            if len(value.shape) == 0:
                raise ValueError("stops must have at least one dimension")
            if (value < 0).any():
                raise ValueError("stops must be a non-negative array")
        self._stops = value
        self._offsets, self._counts, self._parents = None, None, None
        self._isvalid = False

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = self._util_toarray(value, self.DEFAULTTYPE)
        self._isvalid = False

    @property
    def offsets(self):
        if self._offsets is None:
            self._valid()
            if self.offsetsaliased(self._starts, self._stops):
                self._offsets = self._starts.base
            elif len(self._starts.shape) == 1 and self.numpy.array_equal(self._starts[1:], self._stops[:-1]):
                if len(self._stops) == 0:
                    return self.numpy.array([0], dtype=self.INDEXTYPE)
                else:
                    self._offsets = self.numpy.append(self._starts, self._stops[-1])
            else:
                raise ValueError("starts and stops are not compatible with a single offsets array")
        return self._offsets

    @offsets.setter
    def offsets(self, value):
        value = self._util_toarray(value, self.INDEXTYPE, self.numpy.ndarray)
        if self.check_prop_valid:
            if not self._util_isintegertype(value.dtype.type):
                raise TypeError("offsets must have integer dtype")
            if len(value.shape) != 1 or (value < 0).any():
                raise ValueError("offsets must be a one-dimensional, non-negative array")
            if len(value) == 0:
                raise ValueError("offsets must be non-empty")
        self._starts = value[:-1]
        self._stops = value[1:]
        self._offsets = value
        self._counts, self._parents = None, None
        self._isvalid = False

    @property
    def counts(self):
        if self._counts is None:
            self._valid()
            self._counts = self.stops - self._starts
        return self._counts

    @counts.setter
    def counts(self, value):
        value = self._util_toarray(value, self.INDEXTYPE, self.numpy.ndarray)
        if self.check_prop_valid:
            if not self._util_isintegertype(value.dtype.type):
                raise TypeError("counts must have integer dtype")
            if len(value.shape) == 0:
                raise ValueError("counts must have at least one dimension")
            if (value < 0).any():
                raise ValueError("counts must be a non-negative array")
        offsets = self.counts2offsets(value.reshape(-1))
        self._starts = offsets[:-1].reshape(value.shape)
        self._stops = offsets[1:].reshape(value.shape)
        self._offsets = offsets if len(value.shape) == 1 else None
        self._counts = value
        self._parents = None
        self._isvalid = False

    @property
    def parents(self):
        if self._parents is None:
            self._valid()
            if self._canuseoffset():
                self._parents = self.offsets2parents(self.offsets)
            else:
                self._parents = self.startsstops2parents(self._starts, self._stops)
        return self._parents

    @parents.setter
    def parents(self, value):
        value = self._util_toarray(value, self.INDEXTYPE, self.numpy.ndarray)
        if self.check_prop_valid:
            if not self._util_isintegertype(value.dtype.type):
                raise TypeError("parents must have integer dtype")
            if len(value.shape) == 0:
                raise ValueError("parents must have at least one dimension")
        self._starts, self._stops = self.parents2startsstops(value)
        self._offsets, self._counts = None, None
        self._parents = value

    @property
    def localindex(self):
        if len(self) == 0:
            return self.JaggedArray([], [], self.numpy.array([], dtype=self.INDEXTYPE))
        elif self._canuseoffset():
            out = self.numpy.arange(self.offsets[0], self.offsets[-1], dtype=self.INDEXTYPE)
            out -= self.offsets[self.parents[self.parents >= 0]]
            return self.JaggedArray.fromoffsets(self.offsets - self.offsets[0], out)
        else:
            counts = self.counts.reshape(-1)  # flatten in case multi-d jagged
            offsets = self.counts2offsets(counts)
            out = self.numpy.arange(offsets[-1], dtype=self.INDEXTYPE)
            out -= self.numpy.repeat(offsets[:-1], awkward0.util.windows_safe(counts))
            return self.JaggedArray(offsets[:-1].reshape(self.shape), offsets[1:].reshape(self.shape), out)

    def _getnbytes(self, seen):
        if id(self) in seen:
            return 0
        else:
            seen.add(id(self))
            if self.offsetsaliased(self._starts, self._stops):
                return self._starts.base.nbytes + (self._content.nbytes if isinstance(self._content, self.numpy.ndarray) else self._content._getnbytes(seen))
            else:
                return self._starts.nbytes + self._stops.nbytes + (self._content.nbytes if isinstance(self._content, self.numpy.ndarray) else self._content._getnbytes(seen))

    def __len__(self):
        return len(self._starts)

    def _gettype(self, seen):
        return awkward0.type.ArrayType(*(self._starts.shape[1:] + (self.numpy.inf, awkward0.type._fromarray(self._content, seen))))

    def _util_layout(self, position, seen, lookup):
        awkward0.type.LayoutNode(self._starts, position + (0,), seen, lookup)
        awkward0.type.LayoutNode(self._stops, position + (1,), seen, lookup)
        awkward0.type.LayoutNode(self._content, position + (2,), seen, lookup)
        return (awkward0.type.LayoutArg("starts", position + (0,)),
                awkward0.type.LayoutArg("stops", position + (1,)),
                awkward0.type.LayoutArg("content", position + (2,)))

    def _valid(self):
        if not self._isvalid:
            if self.offsetsaliased(self._starts, self._stops):
                self._offsets = self._starts.base
                if self.check_whole_valid:
                    if not (self._offsets[1:] >= self._offsets[:-1]).all():
                        raise ValueError("offsets must be monatonically increasing")
                    if self._offsets.max() > len(self._content):
                        raise ValueError("maximum offset {0} is beyond the length of the content ({1})".format(self._offsets.max(), len(self._content)))

            else:
                if self.check_whole_valid:
                    self._validstartsstops(self._starts, self._stops)
                nonempty = (self._starts != self._stops)
                starts = self._starts[nonempty].reshape(-1)
                if self.check_whole_valid:
                    if len(starts) != 0 and starts.reshape(-1).max() >= len(self._content):
                        raise ValueError("maximum start ({0}) is at or beyond the length of the content ({1})".format(starts.reshape(-1).max(), len(self._content)))
                stops = self._stops[nonempty].reshape(-1)
                if self.check_whole_valid:
                    if len(stops) != 0 and stops.reshape(-1).max() > len(self._content):
                        raise ValueError("maximum stop ({0}) is beyond the length of the content ({1})".format(self._stops.reshape(-1).max(), len(self._content)))

            self._isvalid = True

    @classmethod
    def _validstartsstops(cls, starts, stops):
        if cls.check_whole_valid:
            if len(starts) > len(stops):
                raise ValueError("starts must have the same (or shorter) length than stops")
            if starts.shape[1:] != stops.shape[1:]:
                raise ValueError("starts and stops must have the same dimensionality (shape[1:])")
            if not (stops[:len(starts)] >= starts).all():
                raise ValueError("stops must be greater than or equal to starts")

    def __iter__(self, checkiter=True):
        if checkiter:
            self._checkiter()
        self._valid()
        if len(self._starts.shape) != 1:
            for x in super(JaggedArray, self).__iter__(checkiter=checkiter):
                yield x
        else:
            stops = self._stops
            content = self._content
            for i, start in enumerate(self._starts):
                yield content[start:stops[i]]

    def __getitem__(self, where):
        self._valid()

        if self._util_isstringslice(where):
            content = self._content[where]
            cls = awkward0.array.objects.Methods.maybemixin(type(content), self.JaggedArray)
            out = cls.__new__(cls)
            out.__dict__.update(self.__dict__)
            out._content = content
            out.__doc__ = content.__doc__
            return out

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[:len(self._starts.shape)], where[len(self._starts.shape):]

        if len(head) == 1 and isinstance(head[0], JaggedArray):
            head = head[0]

            if isinstance(self._content, JaggedArray) and isinstance(head._content, JaggedArray):
                return self.copy(content=self._content[head._content])

            elif self._util_isintegertype(head._content.dtype.type):
                if len(head.shape) == 1 and head._starts.shape != self._starts.shape:
                    raise ValueError("jagged array used as index has a different shape {0} from the jagged array it is selecting from {1}".format(head._starts.shape, self._starts.shape))

                headoffsets = self.counts2offsets(head.counts)
                head = head._tojagged(headoffsets[:-1], headoffsets[1:], copy=False)

                counts = head.tojagged(self.counts)._content

                indexes = self.numpy.array(head._content[:headoffsets[-1]], dtype=self.INDEXTYPE, copy=True)

                negatives = (indexes < 0)
                indexes[negatives] += counts[negatives]

                if not self.numpy.bitwise_and(0 <= indexes, indexes < counts).all():
                    raise IndexError("jagged array used as index contains out-of-bounds values")

                indexes += head.tojagged(self._starts)._content

                return self.copy(starts=head._starts, stops=head._stops, content=self._content[indexes])

            elif len(head.shape) == 1 and issubclass(head._content.dtype.type, (self.numpy.bool, self.numpy.bool_)):
                try:
                    offsets = self.offsets
                    thyself = self

                except ValueError:
                    offsets = self.counts2offsets(self.counts.reshape(-1))
                    thyself = self._tojagged(offsets[:-1], offsets[1:], copy=False)
                    thyself._starts.shape = self._starts.shape
                    thyself._stops.shape = self._stops.shape

                head = head._tojagged(thyself._starts, thyself._stops, copy=False)
                inthead = head.copy(content=head._content.astype(self.INDEXTYPE))
                intheadsum = inthead.sum()

                offsets = self.counts2offsets(intheadsum)

                headcontent = self.numpy.array(head._content, dtype=self.BOOLTYPE)

                headcontent_indices_to_ignore = self.numpy.resize(head.parents < 0, headcontent.shape)
                headcontent_indices_to_ignore[len(head.parents):] = True
                headcontent[headcontent_indices_to_ignore] = False

                original_headcontent_length = len(headcontent)
                headcontent = self.numpy.resize(headcontent, thyself._content.shape)
                headcontent[original_headcontent_length:] = False

                return self.copy(starts=offsets[:-1].reshape(intheadsum.shape), stops=offsets[1:].reshape(intheadsum.shape), content=thyself._content[headcontent])

            elif head.shape == self.shape and issubclass(head._content.dtype.type, (self.numpy.bool, self.numpy.bool_)):
                index = self.localindex + self.starts
                flatindex = index.flatten()[head.flatten()]
                return type(self).fromcounts(head.sum(), self._content[flatindex])

            else:
                raise TypeError("jagged index must be boolean (mask) or integer (fancy indexing)")

        else:
            starts = self._starts[head]
            stops = self._stops[head]
            if len(starts.shape) == len(stops.shape) == 0:
                return self._content[starts:stops][tail]
            else:
                node = self.copy(starts=starts, stops=stops)

        head = head[-1]

        nslices = 0
        while isinstance(node, JaggedArray) and len(tail) > 0:
            wasslice = isinstance(head, slice)
            head, tail = tail[0], tail[1:]

            original_head = head
            if self._util_isinteger(head):
                stack = []
                for x in range(nslices):
                    stack.insert(0, node.counts)
                    node = node.flatten()

                if isinstance(node, JaggedArray):
                    counts = node.stops - node._starts
                    if head < 0:
                        head = counts + head
                    if not self.numpy.bitwise_and(0 <= head, head < counts).all():
                        raise IndexError("index {0} is out of bounds for jagged min size {1}".format(original_head, counts.min()))
                    node = node._content[node._starts + head]
                else:
                    node = node[:, head]

                for oldcounts in stack:
                    node = type(self).fromcounts(oldcounts, node)

            elif isinstance(head, slice):
                nslices += 1
                if nslices >= 2:
                    raise NotImplementedError("this implementation cannot slice a JaggedArray in more than two dimensions")

                # If we sliced down to an empty jagged array, take a shortcut
                if len(node) == 0:
                    return node

                counts = node.stops - node._starts
                step = 1 if head.step is None else head.step

                if step == 0:
                    raise ValueError("slice step cannot be zero")

                elif step > 0:
                    if head.start is None:
                        starts = self.numpy.zeros(counts.shape, dtype=self.INDEXTYPE)
                    elif head.start >= 0:
                        starts = self.numpy.minimum(counts, head.start)
                    else:
                        starts = self.numpy.maximum(0, self.numpy.minimum(counts, counts + head.start))

                    if head.stop is None:
                        stops = counts
                    elif head.stop >= 0:
                        stops = self.numpy.minimum(counts, head.stop)
                    else:
                        stops = self.numpy.maximum(0, self.numpy.minimum(counts, counts + head.stop))

                    stops = self.numpy.maximum(starts, stops)

                    start = starts.min()
                    stop = stops.max()
                    indexes = self.numpy.empty((len(node), abs(stop - start)), dtype=self.INDEXTYPE)
                    indexes[:, :] = self.numpy.arange(start, stop)

                    mask = indexes >= starts.reshape((len(node), 1))
                    self.numpy.bitwise_and(mask, indexes < stops.reshape((len(node), 1)), out=mask)
                    if step != 1:
                        self.numpy.bitwise_and(mask, self.numpy.remainder(indexes - starts.reshape((len(node), 1)), step) == 0, out=mask)

                else:
                    if head.start is None:
                        starts = counts - 1
                    elif head.start >= 0:
                        starts = self.numpy.minimum(counts - 1, head.start)
                    else:
                        starts = self.numpy.maximum(-1, self.numpy.minimum(counts - 1, counts + head.start))

                    if head.stop is None:
                        stops = self.numpy.full(counts.shape, -1, dtype=self.INDEXTYPE)
                    elif head.stop >= 0:
                        stops = self.numpy.minimum(counts - 1, head.stop)
                    else:
                        stops = self.numpy.maximum(-1, self.numpy.minimum(counts - 1, counts + head.stop))

                    stops = self.numpy.minimum(starts, stops)

                    start = starts.max()
                    stop = stops.min()
                    indexes = self.numpy.empty((len(node), abs(stop - start)), dtype=self.INDEXTYPE)
                    indexes[:, :] = self.numpy.arange(start, stop, -1)

                    mask = indexes <= starts.reshape((len(node), 1))
                    self.numpy.bitwise_and(mask, indexes > stops.reshape((len(node), 1)), out=mask)
                    if step != -1:
                        self.numpy.bitwise_and(mask, self.numpy.remainder(indexes - starts.reshape((len(node), 1)), step) == 0, out=mask)

                newcounts = self.numpy.count_nonzero(mask, axis=1)
                newoffsets = self.counts2offsets(newcounts.reshape(-1))
                newcontent = node._content[(indexes + node._starts.reshape((len(node), 1)))[mask]]

                node = node.copy(starts=newoffsets[:-1], stops=newoffsets[1:], content=newcontent)

            else:
                head = self.numpy.array(head, copy=False)
                if len(head.shape) == 1 and self._util_isintegertype(head.dtype.type):
                    if wasslice:
                        stack = []
                        for x in range(nslices):
                            stack.insert(0, node.counts)
                            node = node.flatten()

                        index = self.numpy.tile(head, len(node))
                        mask = (index < 0)
                        if mask.any():
                            pluscounts = (index.reshape(-1, len(head)) + node.counts.reshape(-1, 1)).reshape(-1)
                            index[mask] = pluscounts[mask]
                        if (index < 0).any() or (index.reshape(-1, len(head)) >= node.counts.reshape(-1, 1)).any():
                            raise IndexError("index in jagged subdimension is out of bounds")
                        index = (index.reshape(-1, len(head)) + node._starts.reshape(-1, 1)).reshape(-1)
                        node = node._content[index]
                        if isinstance(node, JaggedArray):
                            node._starts = node._starts.reshape(-1, len(head))
                            node._stops = node._stops.reshape(-1, len(head))
                        elif isinstance(node, self.numpy.ndarray):
                            node = node.reshape(-1, len(head))
                        else:
                            raise NotImplementedError

                        for oldcounts in stack:
                            node = type(self).fromcounts(oldcounts, node)

                    else:
                        if len(node) != len(head):
                            raise IndexError("shape mismatch: indexing arrays could not be broadcast together with shapes {0} {1}".format(len(node), len(head)))
                        index = head.copy() if head is original_head else head
                        mask = (index < 0)
                        if mask.any():
                            index[mask] += head.counts
                        if (index < 0).any() or (index >= node.counts).any():
                            raise IndexError("index in jagged subdimension is out of bounds")
                        index += node._starts
                        node = node._content[index]

                elif len(head.shape) == 1 and issubclass(head.dtype.type, (self.numpy.bool, self.numpy.bool_)):
                    if wasslice:
                        stack = []
                        for x in range(nslices):
                            stack.insert(0, node.counts)
                            node = node.flatten()

                        if len(node) != 0 and not (node.counts == len(head)).all():
                            raise IndexError("jagged subdimension is not regular and cannot match boolean shape {0}".format(len(head)))
                        head = self.numpy.nonzero(head)[0]
                        index = self.numpy.tile(head, len(node))
                        index = (index.reshape(-1, len(node)) + node._starts.reshape(-1, 1)).reshape(-1)
                        node = node._content[index]
                        if isinstance(node, JaggedArray):
                            node._starts = node._starts.reshape(-1, len(head))
                            node._stops = node._stops.reshape(-1, len(head))
                        elif isinstance(node, self.numpy.ndarray):
                            node = node.reshape(-1, len(head))
                        else:
                            raise NotImplementedError

                        for oldcounts in stack:
                            node = type(self).fromcounts(oldcounts, node)

                    else:
                        index = self.numpy.nonzero(head)[0]
                        if len(node) != len(index):
                            raise IndexError("shape mismatch: indexing arrays could not be broadcast together with shapes {0} {1}".format(len(node), len(index)))
                        index += node._starts
                        node = node._content[index]

                else:
                    raise TypeError("cannot interpret shape {0}, dtype {1} as a fancy index or mask".format(head.shape, head.dtype))

            if isinstance(node, self.numpy.ndarray) and len(node.shape) < sum(0 if isinstance(x, slice) else 1 for x in tail):
                raise IndexError("IndexError: too many indices for array")

        return node[tail]

    def __setitem__(self, where, what):
        if isinstance(where, awkward0.util.string):
            self._content[where] = self.tojagged(what)._content

        elif self._util_isstringslice(where):
            what = what.unzip()
            if len(where) != len(what):
                raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
            for x, y in zip(where, what):
                self._content[x] = self.tojagged(y)._content

        elif isinstance(where, JaggedArray):
            if isinstance(what, JaggedArray):
                what = what.flatten()

            if len(where.shape) == 1:
                if where._starts.shape != self._starts.shape:
                    raise ValueError("jagged array used as index has a different shape {0} from the jagged array it is selecting from {1}".format(where._starts.shape, self._starts.shape))
            elif where.shape != self.shape:
                raise ValueError("jagged array used as index has a different shape {0} from the jagged array it is selecting from {1}".format(where._starts.shape, self._starts.shape))

            if self._util_isintegertype(where._content.dtype.type):

                whereoffsets = self.counts2offsets(where.counts)
                where = where._tojagged(whereoffsets[:-1], whereoffsets[1:], copy=False)

                counts = where.tojagged(self.counts)._content

                indexes = self.numpy.array(where._content[:whereoffsets[-1]], copy=True)

                negatives = (indexes < 0)
                indexes[negatives] += counts[negatives]

                if not self.numpy.bitwise_and(0 <= indexes, indexes < counts).all():
                    raise IndexError("jagged array used as index contains out-of-bounds values")

                indexes += where.tojagged(self._starts)._content

                self._content[indexes] = what

            elif issubclass(where._content.dtype.type, (self.numpy.bool, self.numpy.bool_)):
                index = self.localindex + self.starts
                flatindex = index.flatten()[where.flatten()]
                self._content[flatindex] = what

            else:
                raise TypeError("jagged index must be boolean (mask) or integer (fancy indexing)")

        else:
            raise TypeError("invalid index for assigning to JaggedArray: {0}".format(where))

    def tojagged(self, data):
        if isinstance(data, JaggedArray):
            selfcounts = self.stops - self._starts
            datacounts = data.stops - data._starts
            if not self.numpy.array_equal(selfcounts, datacounts):
                raise ValueError("cannot broadcast JaggedArray to match JaggedArray with a different counts")
            if len(self._starts) == 0:
                return self.copy(content=data._content)

            tmp = self.compact()
            tmpparents = self.offsets2parents(tmp.offsets)

            index = self.JaggedArray(tmp._starts, tmp._stops, (self.numpy.arange(tmp._stops[-1], dtype=self.INDEXTYPE) - tmp._starts[tmpparents]))

            data = data.compact()
            return self.copy(content=data._content[self.IndexedArray.invert((index + self._starts)._content)])

        elif isinstance(data, awkward0.array.base.AwkwardArray):
            if len(self._starts) != len(data):
                raise ValueError("cannot broadcast AwkwardArray to match JaggedArray with a different length")
            if len(self._starts) == 0:
                return self.copy(content=data)
            out = self.copy(content=data[self.parents])
            out._parents = self.parents
            return out

        elif isinstance(data, self.numpy.ndarray):
            content = self.numpy.empty(len(self.parents), dtype=data.dtype)
            if len(data.shape) == 0 or (len(data.shape) == 1 and data.shape[0] == 1):
                content[:] = data
            else:
                good = (self.parents >= 0)
                content[good] = data[self.parents[good]]
            out = self.copy(content=content)
            out._parents = self.parents
            return out

        elif isinstance(data, Iterable):
            return self.tojagged(self.numpy.array(data))

        else:
            return self.tojagged(self.numpy.array([data]))

    def _tojagged(self, starts=None, stops=None, copy=True):
        if starts is None and stops is None:
            if copy:
                starts, stops = self._util_deepcopy(self._starts), self._util_deepcopy(self._stops)
            else:
                starts, stops = self._starts, self._stops

        elif stops is None:
            starts = self._util_toarray(starts, self.INDEXTYPE)
            if len(self) != len(starts):
                raise ValueError("cannot fit JaggedArray of length {0} into starts of length {1}".format(len(self), len(starts)))

            stops = starts + self.counts

            if (stops[:-1] > starts[1:]).any():
                raise ValueError("cannot fit contents of JaggedArray into the given starts array")

        elif starts is None:
            stops = self._util_toarray(stops, self.INDEXTYPE)
            if len(self) != len(stops):
                raise ValueError("cannot fit JaggedArray of length {0} into stops of length {1}".format(len(self), len(stops)))

            starts = stops - self.counts

            if (stops[:-1] > starts[1:]).any():
                raise ValueError("cannot fit contents of JaggedArray into the given stops array")

        else:
            if not self.numpy.array_equal(stops - starts, self.counts):
                raise ValueError("cannot fit contents of JaggedArray into the given starts and stops arrays")

        self._validstartsstops(starts, stops)

        if not copy and starts is self._starts and stops is self._stops:
            return self

        elif (starts is self._starts or self.numpy.array_equal(starts, self._starts)) and (stops is self._stops or self.numpy.array_equal(stops, self._stops)):
            return self.copy(starts=starts, stops=stops, content=(self._util_deepcopy(self._content) if copy else self._content))

        else:
            if self.offsetsaliased(starts, stops):
                parents = self.offsets2parents(starts.base)
            elif len(starts.shape) == 1 and self.numpy.array_equal(starts[1:], stops[:-1]):
                if len(stops) == 0:
                    offsets = self.numpy.array([0], dtype=self.INDEXTYPE)
                else:
                    offsets = self.numpy.append(starts, stops[-1])
                parents = self.offsets2parents(offsets)
            else:
                parents = self.startsstops2parents(starts, stops)

            good = (parents >= 0)
            increase = self.numpy.arange(len(parents), dtype=self.INDEXTYPE)
            increase[good] -= increase[starts[parents[good]]]
            index = self._starts[parents]
            index += increase
            index *= good
            out = self.copy(starts=starts, stops=stops, content=self._content[index])
            out._parents = parents
            return out

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        import awkward0.array.objects
        import awkward0.array.table

        if "out" in kwargs:
            raise NotImplementedError("in-place operations not supported")

        if method != "__call__":
            return NotImplemented

        starts, stops = None, None
        for i in range(len(inputs)):
            if isinstance(inputs[i], JaggedArray):
                try:
                    offsets = inputs[i].offsets   # calls _valid()
                except ValueError:
                    counts = inputs[i].counts
                    offsets = self.counts2offsets(counts.reshape(-1))
                    starts, stops = offsets[:-1], offsets[1:]
                    starts = starts.reshape(counts.shape)
                    stops = stops.reshape(counts.shape)
                else:
                    starts, stops = offsets[:-1], offsets[1:]

        assert starts is not None and stops is not None

        inputs = list(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], JaggedArray):
                inputs[i] = inputs[i]._tojagged(starts, stops, copy=False)

            elif isinstance(inputs[i], (self.numpy.ndarray, awkward0.array.base.AwkwardArray)):
                pass

            else:
                try:
                    for first in inputs[i]:
                        break
                except TypeError:
                    pass
                else:
                    if "first" not in locals() or isinstance(first, (numbers.Number, self.numpy.bool_, self.numpy.bool, self.numpy.number)):
                        inputs[i] = self.numpy.array(inputs[i], copy=False)
                    else:
                        inputs[i] = self.JaggedArray.fromiter(inputs[i])

        for jaggedarray in inputs:
            if isinstance(jaggedarray, JaggedArray):
                starts, stops, parents, good = jaggedarray._starts, jaggedarray._stops, None, None
                break
        else:
            assert False

        for i in range(len(inputs)):
            if isinstance(inputs[i], (self.numpy.ndarray, awkward0.array.base.AwkwardArray)) and not isinstance(inputs[i], JaggedArray):
                data = self._util_toarray(inputs[i], inputs[i].dtype)
                if starts.shape != data.shape:
                    raise ValueError("cannot broadcast JaggedArray of shape {0} with array of shape {1}".format(starts.shape, data.shape))

                if parents is None:
                    parents = jaggedarray.parents
                    if self._canuseoffset() and len(jaggedarray.starts) > 0 and jaggedarray.starts[0] == 0:
                        good = None
                    else:
                        good = (parents >= 0)

                def recurse(x):
                    if isinstance(x, awkward0.array.objects.ObjectArray):
                        return x.copy(content=recurse(x.content))

                    elif isinstance(x, awkward0.array.table.Table):
                        content = x.empty_like()
                        for n in x.columns:
                            content[n] = recurse(x[n])
                        return content

                    elif good is None:
                        if len(x.shape) == 0:
                            content = self.numpy.full(len(parents), x, dtype=x.dtype)
                        else:
                            content = x.reshape(-1)[parents]
                        return content

                    else:
                        content = self.numpy.empty(len(parents), dtype=x.dtype)
                        if len(x.shape) == 0:
                            content[good] = x
                        else:
                            content[good] = x.reshape(-1)[parents[good]]
                        return content

                content = recurse(data)

                inputs[i] = self.JaggedArray(starts, stops, content)

        for i in range(len(inputs)):
            if isinstance(inputs[i], JaggedArray):
                inputs[i] = inputs[i].flatten()

        result = getattr(ufunc, method)(*inputs, **kwargs)

        counts = stops - starts
        if isinstance(result, tuple):
            return tuple(self.Methods.maybemixin(type(x), self.JaggedArray).fromcounts(counts, x) if isinstance(x, (self.numpy.ndarray, awkward0.array.base.AwkwardArray)) else x for x in result)
        elif method == "at":
            return None
        else:
            return self.Methods.maybemixin(type(result), self.JaggedArray).fromcounts(counts, result)

    def regular(self):
        self._valid()

        if len(self) > 0 and not (self.counts.reshape(-1)[0] == self.counts).all():
            raise ValueError("jagged array is not regular: different elements have different counts")
        count = self.counts.reshape(-1)[0]
        content = self._util_regular(self._content)

        if self._canuseoffset():
            out = content[self._starts[0]:self._stops[-1]]
            return out.reshape(self._starts.shape + (count,) + content.shape[1:])

        else:
            indexes = self.numpy.repeat(self._starts, awkward0.util.windows_safe(count)).reshape(self._starts.shape + (count,))
            indexes += self.numpy.arange(count)
            return content[indexes]

    def _argpairs(self):
        self._valid()

        counts = self.counts * (self.counts + 1) >> 1    # N * (N + 1) // 2

        offsets = self.counts2offsets(counts)
        indexes = self.numpy.arange(offsets[-1])
        parents = self.offsets2parents(offsets)

        n = self.counts[parents]
        k = indexes - offsets[parents]
        two_n_1 = (2*n + 1)
        i = self.numpy.floor((two_n_1 - self.numpy.sqrt(two_n_1*two_n_1 - 8*k)) / 2).astype(self.INDEXTYPE)

        starts_parents = self._starts[parents]

        left = starts_parents + i
        right = starts_parents + k - n*i + (i*(i + 1) >> 1)

        out = self.JaggedArray.fromoffsets(offsets, self.Table.named("tuple", left, right))
        out._parents = parents
        return out

    def _argdistincts(self, absolute):
        counts_comb = self.counts*(self.counts - 1) // 2
        offsets_comb = self.counts2offsets(counts_comb)
        parents_comb = self.offsets2parents(offsets_comb)
        local_indices = self.numpy.arange(offsets_comb[-1]) - offsets_comb[parents_comb]

        ### Consider the double-loop:
        #   for i in range(n):
        #       for j in range(i + 1, n):
        #           pairs.append((i, j))
        # At the beginning the i-th iteration of the outer loop,
        #   len(pairs) = L = (n - 1) + (n - 2) + ... + (n - i)
        #                  = n*i - i*(i + 1)/2
        #   => -i^2 + (2*n - 1)*i - 2*L = 0
        # So the quadratic formula gives i as a function of L at that point:
        #   i = [(2*n - 1) - sqrt((2*n - 1)^2 - 4*2*L)] / 2
        # Since i(L) is monotone increasing, and won't reach i+1 until
        # the next outer loop, the floor gives the i value inside the outer loop.
        # To find j, we can subtract L from our local indices to get the index
        # from zero of the inner loop, then add i + 1.

        n = self.counts[parents_comb]
        b = 2*n - 1
        i = self.numpy.floor((b - self.numpy.sqrt(b*b - 8*local_indices)) / 2).astype(counts_comb.dtype)
        j = local_indices + i*(i - b + 2) // 2 + 1

        if absolute:
            starts_parents = self._starts[parents_comb]
            i += starts_parents
            j += starts_parents

        out = self.JaggedArray.fromoffsets(offsets_comb, self.Table.named("tuple", i, j))
        out._parents = parents_comb
        return out

    def _argchoose(self, n, absolute):
        np = self.numpy

        def root2(a):
            return np.floor((1+np.sqrt(8*a+1))/2).astype(self.INDEXTYPE)

        def root3(a):
            out = 2*np.ones(a.shape, dtype=self.INDEXTYPE)
            mask = a > 0
            rad = np.power(np.sqrt(3)*np.sqrt(243*a[mask]**2 - 1) + 27*a[mask], 1./3)
            # 1e-12 to correct rounding error (good to 1000 choose 3)
            out[mask] = np.floor(np.power(3, -2./3)*rad + np.power(3, -1./3)/rad + 1 + 1e-12)
            return out

        def root4(a):
            # good to (at least) 100 choose 4
            return np.floor((np.sqrt(4*np.sqrt(24*a + 1) + 5) + 3)/2).astype(self.INDEXTYPE)

        if n > 5:
            # Need to solve general polynomial \prod_{i<=k} (n-i) / k! = a
            # Not sure if possible, is the Galois group solvable for this class of polynomial?
            # The first level is a freebie thanks to np.repeat()
            # I think the next levels should be too, but could not quite make it work...
            raise NotImplementedError
        elif n == 5:
            counts = self.counts*(self.counts - 1)*(self.counts - 2)*(self.counts - 3)*(self.counts - 4)//120
        elif n == 4:
            counts = self.counts*(self.counts - 1)*(self.counts - 2)*(self.counts - 3)//24
        elif n == 3:
            counts = self.counts*(self.counts - 1)*(self.counts - 2)//6
        elif n == 2:
            counts = self.counts*(self.counts - 1)//2
        elif n <= 1:
            raise ValueError("Choosing 0 or 1 items is trivial")

        local = self.localindex.content
        offsets = self.JaggedArray.counts2offsets(counts)
        indices = np.arange(offsets[-1])
        parents = self.JaggedArray.offsets2parents(offsets)

        if n == 5:
            k5 = indices - offsets[parents]
            i5 = np.repeat(local, awkward0.util.windows_safe(local*(local - 1)*(local - 2)*(local - 3)//24))
            k4 = k5 - i5*(i5 - 1)*(i5 - 2)*(i5 - 3)*(i5 - 4)//120
            i4 = root4(k4)
            k3 = k4 - i4*(i4 - 1)*(i4 - 2)*(i4 - 3)//24
            i3 = root3(k3)
            k2 = k3 - i3*(i3 - 1)*(i3 - 2)//6
            i2 = root2(k2)
            k1 = k2 - i2*(i2 - 1)//2
            i1 = k1
            if absolute:
                starts_parents = self.starts[parents]
                for idx in [i1, i2, i3, i4, i5]:
                    idx += starts_parents
            out = self.JaggedArray.fromoffsets(offsets, self.Table.named("tuple", i1, i2, i3, i4, i5))
        elif n == 4:
            k4 = indices - offsets[parents]
            i4 = np.repeat(local, awkward0.util.windows_safe(local*(local - 1)*(local - 2)//6))
            k3 = k4 - i4*(i4 - 1)*(i4 - 2)*(i4 - 3)//24
            i3 = root3(k3)
            k2 = k3 - i3*(i3 - 1)*(i3 - 2)//6
            i2 = root2(k2)
            k1 = k2 - i2*(i2 - 1)//2
            i1 = k1
            if absolute:
                starts_parents = self.starts[parents]
                for idx in [i1, i2, i3, i4]:
                    idx += starts_parents
            out = self.JaggedArray.fromoffsets(offsets, self.Table.named("tuple", i1, i2, i3, i4))
        elif n == 3:
            k3 = indices - offsets[parents]
            i3 = np.repeat(local, awkward0.util.windows_safe(local*(local - 1)//2))
            k2 = k3 - i3*(i3 - 1)*(i3 - 2)//6
            i2 = root2(k2)
            k1 = k2 - i2*(i2 - 1)//2
            i1 = k1
            if absolute:
                starts_parents = self.starts[parents]
                for idx in [i1, i2, i3]:
                    idx += starts_parents
            out = self.JaggedArray.fromoffsets(offsets, self.Table.named("tuple", i1, i2, i3))
        elif n == 2:
            k2 = indices - offsets[parents]
            i2 = np.repeat(local, awkward0.util.windows_safe(local))
            k1 = k2 - i2*(i2 - 1)//2
            i1 = k1
            if absolute:
                starts_parents = self.starts[parents]
                for idx in [i1, i2]:
                    idx += starts_parents
            out = self.JaggedArray.fromoffsets(offsets, self.Table.named("tuple", i1, i2))

        return out

    def argchoose(self, n):
        """
        Return all unique combinations (up to permutation) of n elements,
        taken without replacement from the indices of the jagged dimension.
        Combinations are ordered such that those of the first elements precede later ones,
        e.g. for n=2: (0,1), (0,2), (1,2), (0,3), (1,3), (2,3), (0,4), ...
        """
        return self._argchoose(n, absolute=False)

    def choose(self, n):
        """
        Return all unique combinations (up to permutation) of n elements,
        taken without replacement from the contents of the jagged dimension.
        Combinations are ordered such that those of the first elements precede later ones,
        e.g. for n=2: (0,1), (0,2), (1,2), (0,3), (1,3), (2,3), (0,4), ...
        """
        args = self._argchoose(n, absolute=True)
        columns = (self.content[args.content[c]] for c in args.columns)
        out = self.JaggedArray.fromoffsets(args.offsets, self.Table.named("tuple", *columns).flattentuple())
        return out

    def argdistincts(self, nested=False):
        """
        Return all unique combinations (up to permutation) of two elements,
        taken without replacement from the indices of the jagged dimension.
        Combinations are ordered lexicographically.
        nested: Return a doubly-jagged array where the first jagged dimension
        matches the shape of this array
        """
        out = self._argdistincts(absolute=False)

        if nested:
            out = self.JaggedArray.fromcounts(self.numpy.maximum(0, self.counts - 1), self.JaggedArray.fromcounts(self.localindex[:, :0:-1].flatten(), out._content))

        return out

    def distincts(self, nested=False):
        """
        Return all unique combinations (up to permutation) of two elements,
        taken without replacement from the contents of the jagged dimension.
        Combinations are ordered lexicographically.
        nested: Return a doubly-jagged array where the first jagged dimension
        matches the shape of this array
        """
        argpairs = self._argdistincts(absolute=True)
        left = argpairs._content["0"]
        right = argpairs._content["1"]

        out = self.JaggedArray.fromoffsets(argpairs.offsets, self.Table.named("tuple", self._content[left], self._content[right]).flattentuple())
        out._parents = argpairs._parents

        if nested:
            out = self.JaggedArray.fromcounts(self.numpy.maximum(0, self.counts - 1), self.JaggedArray.fromcounts(self.localindex[:, :0:-1].flatten(), out._content))

        return out

    def argpairs(self, nested=False):
        out = self._argpairs()
        out["0"] = out["0"] - self._starts
        out["1"] = out["1"] - self._starts

        if nested:
            out = self.JaggedArray.fromcounts(self.counts, self.JaggedArray.fromcounts((self.localindex[:, ::-1] + 1).flatten(), out._content))

        return out

    def pairs(self, nested=False):
        argpairs = self._argpairs()
        left = argpairs._content["0"]
        right = argpairs._content["1"]

        out = self.JaggedArray.fromoffsets(argpairs.offsets, self.Table.named("tuple", self._content[left], self._content[right]).flattentuple())
        out._parents = argpairs._parents

        if nested:
            out = self.JaggedArray.fromcounts(self.counts, self.JaggedArray.fromcounts((self.localindex[:, ::-1] + 1).flatten(), out._content))

        return out

    def _argcross(self, other):
        self._valid()

        if not isinstance(other, JaggedArray):
            raise TypeError("both arrays must be JaggedArrays")

        if len(self) != len(other):
            raise ValueError("both JaggedArrays must have the same length")

        offsets = self.counts2offsets(self.counts * other.counts)
        indexes = self.numpy.arange(offsets[-1], dtype=self.INDEXTYPE)
        parents = self.offsets2parents(offsets)

        ocp = other.counts[parents]
        iop = indexes - offsets[parents]
        iop_ocp = iop // ocp

        left = self._starts[parents] + iop_ocp
        right = other._starts[parents] + iop - ocp * iop_ocp

        out = self.JaggedArray.fromoffsets(offsets, self.Table.named("tuple", left, right))
        out._offsets = offsets
        out._parents = parents
        return out

    def argcross(self, other, nested=False):
        if isinstance(other, self.VirtualArray):
            other = other.array
        out = self._argcross(other)
        out["0"] = out["0"] - self._starts
        out["1"] = out["1"] - other._starts

        if nested:
            out = self.JaggedArray.fromcounts(self.counts, self.JaggedArray.fromcounts(self.tojagged(other.counts).flatten(), out._content))

        return out

    def cross(self, other, nested=False):
        if isinstance(other, self.VirtualArray):
            other = other.array
        if hasattr(self, "_nestedcross"):
            thyself = self._nestedcross
        else:
            thyself = self

        argcross = thyself._argcross(other)
        left, right = argcross._content._contents.values()

        out = self.JaggedArray.fromoffsets(argcross._offsets, self.Table.named("tuple", thyself._content[left], other._content[right]).flattentuple())
        out._parents = argcross._parents
        out._iscross = True

        if nested:
            old = out
            out = self.JaggedArray.fromcounts(thyself.counts, self.JaggedArray.fromcounts(thyself.tojagged(other.counts).flatten(), out._content))
            out._nestedcross = old

        if hasattr(self, "_nestedcross"):
            counts = out.counts.copy()
            mask = (self.counts != 0)
            counts[mask] //= self.counts[mask]
            old = out
            out = self.JaggedArray.fromcounts(self.counts, self.JaggedArray.fromcounts(self.tojagged(counts).flatten(), out._content))
            out._nestedcross = old

        return out

    def _canuseoffset(self):
        self._valid()
        return self.offsetsaliased(self._starts, self._stops) or (len(self._starts.shape) == 1 and self.numpy.array_equal(self._starts[1:], self.stops[:-1]))

    @property
    def iscompact(self):
        if len(self._starts) == 0:
            return True
        else:
            flatstarts = self._starts.reshape(-1)
            flatstops = self.stops.reshape(-1)   # no underscore!
            if not self.offsetsaliased(self._starts, self._stops) and not self.numpy.array_equal(flatstarts[1:], flatstops[:-1]):
                return False
            if not self._isvalid and not (flatstops >= flatstarts).all():
                raise ValueError("offsets must be monatonically increasing")
            return True

    def compact(self):
        if self.iscompact:
            return self
        else:
            offsets = self.counts2offsets(self.counts.reshape(-1))
            if len(self._starts.shape) == 1:
                tmp = self
            else:                                                # no underscore!
                tmp = self.JaggedArray(self._starts.reshape(-1), self.stops.reshape(-1), self._content)
            out = tmp._tojagged(offsets[:-1], offsets[1:], copy=False)
            out.starts.shape = self._starts.shape
            out.stops.shape = self._starts.shape  # intentional: self._stops can too long
            return out

    def flattentuple(self):
        return self.copy(content=self._util_flattentuple(self._content))

    def flatten(self, axis=0):
        if not self._util_isinteger(axis) or axis < 0:
            raise TypeError("axis must be a non-negative integer (can't count from the end)")

        if axis > 1:
            return self.copy(content=self._util_flatten(self._content, axis - 1))

        elif axis == 1:
            self = self.compact()
            innercounts = self._util_counts(self._content)
            if (innercounts < 0).any():
                raise ValueError("cannot flatten an array containing scalar values")
            counts = self.JaggedArray.fromcounts(self.counts, innercounts).sum()
            return type(self).fromcounts(counts, self._util_flatten(self._content, axis - 1))

        else:
            if len(self) == 0:
                return self._content[0:0]
            elif self._canuseoffset():
                return self._content[self._starts[0]:self._stops[-1]]
            else:
                offsets = self.counts2offsets(self.counts.reshape(-1))
                if len(self._starts.shape) == 1:
                    out = self
                else:
                    out = self.JaggedArray(self._starts.reshape(-1), self._stops.reshape(-1), self._content)
                return out._tojagged(offsets[:-1], offsets[1:], copy=False)._content

    def structure1d(self, levellimit=None):
        if len(self._starts) > len(self._stops):
            raise ValueError("starts must have the same (or shorter) length than stops")
        if self._starts.shape[1:] != self._stops.shape[1:]:
            raise ValueError("starts and stops must have the same dimensionality (shape[1:])")

        if levellimit is None:
            levellimit = -1

        out = self
        if levellimit != 0 and isinstance(self._content, awkward0.array.base.AwkwardArray):
            out = self.copy(content=self._content.structure1d(levellimit - 1))

        while len(out._starts.shape) > 1:
            last = out._starts.shape[-1]
            length = functools.reduce(operator.mul, out._starts.shape[:-1], 1)
            offsets = self.numpy.arange(0, (length + 1)*last, last)
            outerstarts = offsets[:-1].reshape(out._starts.shape[:-2] + (-1,))
            outerstops = offsets[1:].reshape(out._stops.shape[:-2] + (-1,))
            innerstarts = out._starts.reshape(-1)
            innerstops = out._stops.reshape(-1)
            out = self.copy(starts=outerstarts, stops=outerstops, content=self.copy(starts=innerstarts, stops=innerstops, content=out._content))

        return out

    def _hasjagged(self):
        return True

    def _reduce(self, ufunc, identity, dtype):
        import awkward0.array.table
        self._valid()

        if self._util_hasjagged(self._content):
            return self.copy(content=self._content._reduce(ufunc, identity, dtype))

        elif isinstance(self._content, awkward0.array.table.Table):
            out = self._content.copy(contents=[])
            for n, x in self._content._contents.items():
                out[n] = self.copy(content=x)._reduce(ufunc, identity, dtype)
            return out

        elif isinstance(self._content, awkward0.array.base.AwkwardArray):
            thyself = self.copy(content=self._content._prepare(ufunc, identity, dtype))

        elif len(self._content.shape) > 1:
            if ufunc is None:
                ufunc = self.numpy.add
                if issubclass(self._content.dtype.type, (self.numpy.floating, self.numpy.complexfloating)):
                    content = 1 - self.numpy.isnan(self._content).astype(self.INDEXTYPE)
                else:
                    content = self.numpy.ones(self._content.shape, dtype=self.INDEXTYPE)

            elif ufunc is self.numpy.count_nonzero:
                ufunc = self.numpy.add
                content = 1 - (self._content == 0).astype(self.INDEXTYPE)

            else:
                content = self._content

            if dtype is not None:
                content = content.astype(dtype)

            return self.copy(content=ufunc.reduce(content, axis=-1))

        else:
            thyself = self.copy()

        if isinstance(thyself._content, awkward0.array.table.Table):
            out = thyself._content.copy(contents=[])
            for n, x in thyself._content._contents.items():
                out[n] = thyself.copy(content=x)._reduce(ufunc, identity, dtype)
            return out

        if len(thyself._starts.shape) > 1:
            thyself._starts = thyself._starts.reshape(-1)
            thyself._stops = thyself._stops.reshape(-1)

        if not thyself._canuseoffset():
            offsets = self.counts2offsets(thyself.counts)
            thyself = thyself._tojagged(offsets[:-1], offsets[1:], copy=False)

        content = thyself._content
        if ufunc is None:
            ufunc = self.numpy.add
            if issubclass(content.dtype.type, (self.numpy.floating, self.numpy.complexfloating)):
                content = 1 - self.numpy.isnan(content).astype(self.INDEXTYPE)
            else:
                content = self.numpy.ones(content.shape, dtype=self.INDEXTYPE)

        elif ufunc is self.numpy.count_nonzero:
            ufunc = self.numpy.add
            content = 1 - (content == 0).astype(self.INDEXTYPE)

        elif issubclass(content.dtype.type, (self.numpy.floating, self.numpy.complexfloating)):
            mask = self.numpy.isnan(content)
            if mask.any():
                content = content.copy()
                content[mask] = identity

        if dtype is None and issubclass(content.dtype.type, (self.numpy.bool_, self.numpy.bool)):
            dtype = self.numpy.dtype(type(identity))
        if dtype is None:
            dtype = content.dtype
        else:
            content = content.astype(dtype)

        if identity == self.numpy.inf:
            if issubclass(dtype.type, (self.numpy.bool_, self.numpy.bool)):
                identity = True
            elif self._util_isintegertype(dtype.type):
                identity = self.numpy.iinfo(dtype.type).max

        elif identity == -self.numpy.inf:
            if issubclass(dtype.type, (self.numpy.bool_, self.numpy.bool)):
                identity = False
            elif self._util_isintegertype(dtype.type):
                identity = self.numpy.iinfo(dtype.type).min

        out = self.numpy.empty(thyself._starts.shape[:1], dtype=dtype)

        if len(out) != 0:
            nonterminal = thyself.offsets[thyself.offsets < len(content)]

            for axis in range(1, len(content.shape)):
                content = ufunc.reduce(content, axis=axis)

            out = ufunc.reduceat(content, awkward0.util.windows_safe(nonterminal))[:len(out)]
            if len(out) < len(thyself):
                tmp = self.numpy.empty((len(thyself),) + out.shape[1:], dtype=out.dtype)
                tmp[:len(out)] = out
                out = tmp

            out[thyself.starts == thyself.stops] = identity

        return out.reshape(self._starts.shape)

    def argmin(self):
        self._valid()
        if self._util_hasjagged(self._content):
            return self.copy(content=self._content.argmin())
        else:
            return self._argminmax(True)

    def argmax(self):
        self._valid()
        if self._util_hasjagged(self._content):
            return self.copy(content=self._content.argmax())
        else:
            return self._argminmax(False)

    def _argminmax(self, ismin):
        if len(self._starts) == len(self._stops) == 0:
            return self.copy(content=self.numpy.array([], dtype=self.INDEXTYPE))

        if len(self._content.shape) != 1:
            raise ValueError("cannot compute arg{0} because content is not one-dimensional".format("min" if ismin else "max"))

        if len(self._content) == 0:
            return self.copy(content=self.numpy.array([], dtype=self.INDEXTYPE))

        if ismin:
            out = self.localindex[self.min() == self]
        else:
            out = self.localindex[self.max() == self]

        # workaround for lack of general out[...,:1] support
        nonempty = out.counts > 0
        if self.offsetsaliased(out._starts, out._stops):
            out.stops = out.stops.copy()
        out.stops[nonempty] = out.starts[nonempty] + 1
        return out

    def argsort(self, ascending=False):
        self._valid()
        if self._util_hasjagged(self._content):
            return self.copy(content=self._content.argsort(ascending))
        else:
            return self._argsort(ascending)

    def _argsort(self, ascending=False):
        reducer = self.JaggedArray.min if ascending else self.JaggedArray.max
        localindex = self.localindex
        out = localindex.empty_like()
        next_start = self.numpy.zeros_like(out.starts)
        tmp = self.copy()
        while tmp.content.size > 0:
            best = reducer(tmp) == tmp
            if self.numpy.isnan(tmp.content).all():
                # put NaN last always
                best = self.numpy.isnan(tmp)
            argbest = localindex[best]
            idx = out.starts + next_start + argbest.localindex
            out._content[idx.flatten()] = argbest.content
            next_start += argbest.counts
            tmp = tmp[~best]
            localindex = localindex[~best]

        # If masked entries were present, they would be dropped by the __getitem__
        # So we need to trim the size of the output array correspondingly
        out.stops = out.starts + next_start
        return out

    @classmethod
    def _concatenate_axis0(cls, arrays):
        starts = cls.numpy.concatenate([x._starts for x in arrays])
        stops = cls.numpy.concatenate([x._stops for x in arrays])
        content = cls._util_concatenate([x._content for x in arrays])

        startsi = 0
        contenti = 0
        for i, array in enumerate(arrays):
            if i != 0:
                startsstart, startsstop = startsi, startsi + len(array._starts)
                starts[startsstart:startsstop] += contenti
                stops[startsstart:startsstop] += contenti
            startsi += len(array._starts)
            contenti += len(array._content)

        return cls(starts, stops, content)

    @classmethod
    def _concatenate_axis1(cls, arrays):
        import awkward0.array.table

        if len(arrays) == 0:
            raise ValueError("at least one array must be provided")   # this can only happen in the classmethod case
        if any(len(a) != len(arrays[0]) for a in arrays):
            raise ValueError("cannot concatenate JaggedArrays of different lengths with axis=1")
        if any(len(a.starts.shape) > 1 for a in arrays):
            raise NotImplementedError

        np = cls.numpy

        flatarrays = [a.flatten() for a in arrays]
        n_arrays = len(arrays)

        if n_arrays > 0 and all(isinstance(a, awkward0.array.table.Table) and set(cls._util_columns_descend(a, set())) == set(cls._util_columns_descend(flatarrays[0], set())) for a in flatarrays):
            results = {}
            for n in flatarrays[0].columns:
                results[n] = cls._concatenate_axis1([a[n] for a in arrays])
            return cls.zip(results)

        # the first step is to get the starts and stops for the stacked structure
        counts = np.vstack([a.counts for a in arrays])
        flat_counts = counts.T.flatten()
        offsets = cls.counts2offsets(flat_counts)
        starts, stops = offsets[:-1], offsets[1:]

        n_content = sum([len(a) for a in flatarrays])
        content_type = type(flatarrays[0])

        # get masks for each of the arrays so we can fill the stacked content array at the right indices
        def get_mask(array_index):
            working_array = np.zeros(n_content + 1, dtype=cls.INDEXTYPE)
            starts_i = starts[i::n_arrays]
            stops_i = stops[i::n_arrays]
            not_empty = starts_i != stops_i
            working_array[starts_i[not_empty]] += 1
            working_array[stops_i[not_empty]] -= 1
            mask = np.array(np.cumsum(working_array)[:-1], dtype=cls.MASKTYPE)
            return mask

        # find most general type with a tentative sum which implements the right type-promotion,
        # except for booleans which would get promoted to integers when summing
        def get_dtype(arrays):
            ick = [x[0] for x in arrays if len(x) != 0]
            if len(ick) == 0:
                dtype = np.dtype(arrays[0].dtype)
            else:
                dtype = np.dtype(np.array(ick).dtype, False)
            allbools = not np.any([a.dtype != np.dtype(bool) for a in arrays])
            dtype = np.dtype(bool) if allbools else dtype
            return dtype

        if content_type == np.ndarray:
            content = np.zeros(n_content, dtype=get_dtype(flatarrays))
            for i in range(n_arrays):
                content[get_mask(i)] = flatarrays[i]

        elif issubclass(type(flatarrays[0]), awkward0.array.objects.ObjectArray) and isinstance(flatarrays[0]._content, awkward0.array.table.Table):
            tablecontent = OrderedDict()
            tables = [a._content for a in flatarrays]

            # make sure all tables have the same columns
            for i in range(len(tables) - 1):
                if set(tables[i]._contents) != set(tables[i+1]._contents):
                    raise ValueError("cannot concatenate Tables with different fields")

            # create empty arrays  for each column with the most general dtype
            for n in tables[0]._contents:
                dtype = get_dtype([t[n] for t in tables])
                tablecontent[n] = np.zeros(n_content, dtype=dtype)

            for i in range(n_arrays):
                mask = get_mask(i)
                for n in tables[0]._contents:
                    tablecontent[n][mask] = tables[i][n]

            content = flatarrays[0].copy(content=awkward0.array.table.Table(**tablecontent))

        else:
            raise NotImplementedError("concatenate with axis=1 is not implemented for these types")

        return arrays[0].__class__(starts[::n_arrays], stops[n_arrays-1::n_arrays], content)

    @awkward0.util.bothmethod
    def zip(isclassmethod, cls_or_self, columns1={}, *columns2, **columns3):
        if isclassmethod:
            cls = cls_or_self
        else:
            self = cls_or_self
            cls = self.__class__
            if not (isinstance(columns1, dict) and len(columns1) == 0):
                columns2 = (columns1,) + columns2
            columns1 = self

        first = None
        def ready(x):
            starts, stops = x._starts.reshape(-1), x._stops.reshape(-1)
            if (x._canuseoffset() and len(starts) != 0 and starts[0] == 0) or (len(starts) != 0 and len(stops) != 0 and len(stops) >= len(starts) and starts[0] == 0 and stops[len(starts) - 1] >= starts[len(starts) - 1] and (starts[1:] == stops[:-1]).all()):
                return x
            else:
                offsets = x.counts2offsets(stops - starts)
                starts, stops = offsets[:-1], offsets[1:]
                starts = starts.reshape((-1,) + x._starts.shape[1:])
                stops = stops.reshape((-1,) + x._stops.shape[1:])
                return x._tojagged(starts, stops, copy=False)

        if isinstance(columns1, JaggedArray):
            columns1 = first = ready(columns1)

        if isinstance(columns1, dict):
            for n in columns1:
                x = columns1[n]
                if isinstance(x, JaggedArray):
                    if first is None:
                        columns1[n] = first = ready(x)
                    else:
                        columns1[n] = x._tojagged(first._starts, first._stops, copy=False)

        columns2 = list(columns2)
        for i in range(len(columns2)):
            x = columns2[i]
            if isinstance(x, JaggedArray):
                if first is None:
                    columns2[i] = first = ready(x)
                else:
                    columns2[i] = x._tojagged(first._starts, first._stops, copy=False)

        for n in columns3:
            x = columns3[n]
            if isinstance(x, JaggedArray):
                if first is None:
                    columns3[n] = first = ready(x)
                else:
                    columns3[n] = x._tojagged(first._starts, first._stops, copy=False)

        if first is None:
            raise TypeError("at least one argument in JaggedArray.zip must be a JaggedArray")

        if isclassmethod:
            numpy = cls.numpy
        else:
            numpy = first.numpy

        if isinstance(columns1, JaggedArray):
            columns1 = columns1._content
        elif isinstance(columns1, dict):
            for n in columns1:
                x = columns1[n]
                if isinstance(x, JaggedArray):
                    columns1[n] = x._content
                elif isinstance(x, Iterable):
                    columns1[n] = first.tojagged(x)._content
                elif isinstance(x, (numbers.Number, numpy.number, numpy.bool, numpy.bool_)):
                    columns1[n] = JaggedArray(first._starts, first._stops, numpy.full(first._stops.max(), columns1, dtype=type(columns1)))._content
                else:
                    raise TypeError("unrecognized type for JaggedArray.zip: {0}".format(type(x)))
        elif isinstance(columns1, Iterable):
            columns1 = first.tojagged(columns1)._content
        elif isinstance(columns1, (numbers.Number, numpy.number, numpy.bool, numpy.bool_)):
            columns1 = JaggedArray(first._starts, first._stops, numpy.full(first._stops.max(), columns1, dtype=type(columns1)))._content
        else:
            raise TypeError("unrecognized type for JaggedArray.zip: {0}".format(type(columns1)))

        for i in range(len(columns2)):
            x = columns2[i]
            if isinstance(x, JaggedArray):
                columns2[i] = x._content
            elif not isinstance(x, dict) and isinstance(x, Iterable):
                columns2[i] = first.tojagged(x)._content
            elif isinstance(x, (numbers.Number, numpy.number, numpy.bool, numpy.bool_)):
                columns2[i] = JaggedArray(first._starts, first._stops, numpy.full(first._stops.max(), x, dtype=type(x)))._content
            else:
                raise TypeError("unrecognized type for JaggedArray.zip: {0}".format(type(x)))

        for n in columns3:
            x = columns3[n]
            if isinstance(x, JaggedArray):
                columns3[n] = x._content
            elif not isinstance(x, dict) and isinstance(x, Iterable):
                columns3[n] = first.tojagged(x)._content
            elif isinstance(x, (numbers.Number, numpy.number, numpy.bool, numpy.bool_)):
                columns3[n] = JaggedArray(first._starts, first._stops, numpy.full(first._stops.max(), x, dtype=type(x)))._content
            else:
                raise TypeError("unrecognized type for JaggedArray.zip: {0}".format(type(x)))

        if isclassmethod:
            if isinstance(columns1, dict) or len(columns3) > 0:
                table = cls.Table.fget(None)(columns1, *columns2, **columns3)
            else:
                table = cls.Table.fget(None).named("tuple", columns1, *columns2)
            return cls.JaggedArray.fget(None)(first._starts, first._stops, table)
        else:
            if isinstance(columns1, dict) or len(columns3) > 0:
                table = first.Table(columns1, *columns2, **columns3)
            else:
                table = first.Table.named("tuple", columns1, *columns2)
            return first.JaggedArray(first._starts, first._stops, table)

    def pad(self, length, maskedwhen=True, clip=False, axis=0):
        if not self._util_isinteger(axis) or axis < 0:
            raise TypeError("axis must be a non-negative integer (can't count from the end)")

        if axis > 0:
            return type(self).fromcounts(self.counts, self._util_pad(self._content, length, maskedwhen, clip, axis - 1))

        flatstarts = self._starts.reshape(-1)

        if len(self._content) == 0:
            offsets = self.numpy.arange(0, length*len(flatstarts) + 1, length)
            starts, stops = offsets[:-1], offsets[1:]
            content = self.IndexedMaskedArray(self.numpy.full(length*len(flatstarts), -1, dtype=self.INDEXTYPE), self._content, maskedwhen=-1)
            return self.copy(starts=starts.reshape(self._starts.shape), stops=stops.reshape(self._starts.shape), content=content)

        if clip:
            almostflat = self._starts.reshape(-1, 1)
            index = self.numpy.arange(length, dtype=self.INDEXTYPE) + almostflat
            localindex = index - almostflat
            index = index.reshape(-1)
            comparator = self.counts.reshape(-1, 1)
            offsets = self.numpy.arange(0, length*len(flatstarts) + 1, length, dtype=self.INDEXTYPE)

        else:
            counts = self.numpy.maximum(self.counts.reshape(-1), length)
            offsets = self.counts2offsets(counts)
            parents = self.offsets2parents(offsets)
            localindex = self.numpy.arange(len(parents), dtype=self.INDEXTYPE) - offsets[:-1][parents]
            index = localindex + flatstarts[parents]
            comparator = self.counts.reshape(-1)[parents]

        if isinstance(maskedwhen, self.numpy.ma.core.MaskedConstant):
            if not isinstance(self._content, self.numpy.ndarray):
                raise TypeError("numpy.ma.masked can only be used if JaggedArray.content is a Numpy array")
            mask = (localindex >= comparator).reshape(-1)
        elif maskedwhen:
            mask = (localindex >= comparator).reshape(-1)
        else:
            mask = (localindex < comparator).reshape(-1)

        index[index >= len(self._content)] = -1
        content = self._content[index]

        starts = offsets[:-1].reshape((-1,) + self._starts.shape[1:])
        stops = offsets[1:].reshape((-1,) + self._starts.shape[1:])

        if isinstance(maskedwhen, self.numpy.ma.core.MaskedConstant):
            return self.copy(starts=starts, stops=stops, content=self.numpy.ma.MaskedArray(content, mask))
        else:
            return self.copy(starts=starts, stops=stops, content=self.MaskedArray(mask, content, maskedwhen=maskedwhen))

    def boolmask(self, maskedwhen=True):
        if maskedwhen:
            return self.numpy.zeros(len(self), dtype=self.MASKTYPE)
        else:
            return self.numpy.ones(len(self), dtype=self.MASKTYPE)

    _topandas_name = "JaggedSeries"

    def _topandas(self, seen):
        import awkward0.pandas
        if id(self) in seen:
            return seen[id(self)]
        else:
            out = seen[id(self)] = self.copy()
            out.__class__ = awkward0.pandas.mixin(type(self))
            if isinstance(self._content, awkward0.array.base.AwkwardArray):
                out._content = out._content._topandas(seen)
            return out
