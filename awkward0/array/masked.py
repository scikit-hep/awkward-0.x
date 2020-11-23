#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-0.x/blob/master/LICENSE

import collections
import numbers

import awkward0.array.base
import awkward0.type
import awkward0.util

class MaskedArray(awkward0.array.base.AwkwardArrayWithContent):
    """
    MaskedArray
    """

    masked = None

    def __init__(self, mask, content, maskedwhen=True):
        self.mask = mask
        self.content = content
        self.maskedwhen = maskedwhen

    @classmethod
    def fromcontent(cls, content, maskedwhen=True):
        if maskedwhen:
            mask = cls.numpy.zeros(len(content), dtype=cls.MASKTYPE)
        else:
            mask = cls.numpy.ones(len(content), dtype=cls.MASKTYPE)
        return cls(mask, content, maskedwhen=maskedwhen)

    def copy(self, mask=None, content=None, maskedwhen=None):
        out = self.__class__.__new__(self.__class__)
        out._mask = self._mask
        out._content = self._content
        out._maskedwhen = self._maskedwhen
        out._isvalid = self._isvalid
        if mask is not None:
            out.mask = mask
        if content is not None:
            out.content = content
        if maskedwhen is not None:
            out.maskedwhen = maskedwhen
        return out

    def deepcopy(self, mask=None, content=None):
        out = self.copy(mask=mask, content=content)
        out._mask = self._util_deepcopy(out._mask)
        out._content = self._util_deepcopy(out._content)
        return out

    def _mine(self, overrides):
        mine = {}
        mine["maskedwhen"] = overrides.pop("maskedwhen", self._maskedwhen)
        return mine

    def empty_like(self, **overrides):
        mine = self._mine(overrides)
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.empty_like(self._content), **mine)
        else:
            return self.copy(content=self._content.empty_like(**overrides), **mine)

    def zeros_like(self, **overrides):
        mine = self._mine(overrides)
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.zeros_like(self._content), **mine)
        else:
            return self.copy(content=self._content.zeros_like(**overrides), **mine)

    def ones_like(self, **overrides):
        mine = self._mine(overrides)
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.ones_like(self._content), **mine)
        else:
            return self.copy(content=self._content.ones_like(**overrides), **mine)

    def __awkward_serialize__(self, serializer):
        self._valid()
        return serializer.encode_call(
            ["awkward0", "MaskedArray"],
            serializer(self._mask, "MaskedArray.mask"),
            serializer(self._content, "MaskedArray.content"),
            {"json": bool(self._maskedwhen)},
        )

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        value = self._util_toarray(value, self.MASKTYPE, self.numpy.ndarray)
        if self.check_prop_valid:
            if len(value.shape) != 1:
                raise ValueError("mask must have 1-dimensional shape")
        if not issubclass(value.dtype.type, (self.numpy.bool_, self.numpy.bool)):
            value = (value != 0)
        self._mask = value
        self._isvalid = False

    def boolmask(self, maskedwhen=None):
        if maskedwhen is None:
            maskedwhen = self._maskedwhen
        if maskedwhen == self._maskedwhen:
            return self._mask
        else:
            return self.numpy.logical_not(self._mask)

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = self._util_toarray(value, self.DEFAULTTYPE)
        self._isvalid = False

    @property
    def maskedwhen(self):
        return self._maskedwhen

    @maskedwhen.setter
    def maskedwhen(self, value):
        self._maskedwhen = bool(value)

    def _getnbytes(self, seen):
        if id(self) in seen:
            return 0
        else:
            seen.add(id(self))
            return self._mask.nbytes + (self._content.nbytes if isinstance(self._content, self.numpy.ndarray) else self._content._getnbytes(seen))

    def __len__(self):
        return len(self._mask)

    def _gettype(self, seen):
        return awkward0.type.OptionType(awkward0.type._fromarray(self._content, seen))

    def _util_layout(self, position, seen, lookup):
        awkward0.type.LayoutNode(self._mask, position + (0,), seen, lookup)
        awkward0.type.LayoutNode(self._content, position + (1,), seen, lookup)
        return (awkward0.type.LayoutArg("mask", position + (0,)),
                awkward0.type.LayoutArg("content", position + (1,)),
                awkward0.type.LayoutArg("maskedwhen", self._maskedwhen))

    def _valid(self):
        if self.check_whole_valid:
            if not self._isvalid:
                if len(self._mask) > len(self._content):
                    raise ValueError("mask length ({0}) must be the same as (or shorter than) the content length ({1})".format(len(self._mask), len(self._content)))

                self._isvalid = True

    def __iter__(self, checkiter=True):
        if checkiter:
            self._checkiter()
        self._valid()

        mask = self._mask
        lenmask = len(mask)
        content = self._content
        maskedwhen = self._maskedwhen
        masked = self.masked

        i = 0
        while i < lenmask:
            if mask[i] == maskedwhen:
                yield masked
            else:
                yield content[i]
            i += 1

    def __getitem__(self, where):
        self._valid()

        if self._util_isstringslice(where):
            content = self._content[where]
            cls = awkward0.array.objects.Methods.maybemixin(type(content), self.MaskedArray)
            out = cls.__new__(cls)
            out.__dict__.update(self.__dict__)
            out._content = content
            return out

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        if self._util_isinteger(head):
            if self._mask[head] == self._maskedwhen:
                if tail != ():
                    raise ValueError("masked element ({0}) is not subscriptable".format(self.masked))
                return self.masked
            else:
                return self._content[:len(self._mask)][(head,) + tail]

        else:
            mask = self._mask[head]
            if tail != () and ((self._maskedwhen and mask.any()) or (not self._maskedwhen and not mask.all())):
                raise ValueError("masked element ({0}) is not subscriptable".format(self.masked))
            else:
                return self.copy(mask=mask, content=self._content[:len(self._mask)][(head,) + tail])

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            raise NotImplementedError("in-place operations not supported")

        if method != "__call__":
            return NotImplemented

        tokeep = None
        for x in inputs:
            if isinstance(x, MaskedArray):
                x._valid()
                if tokeep is None:
                    tokeep = x.boolmask(maskedwhen=False)
                else:
                    tokeep = tokeep & x.boolmask(maskedwhen=False)

        assert tokeep is not None

        inputs = list(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], IndexedMaskedArray):
                inputs[i] = inputs[i]._content[inputs[i]._mask[tokeep]]
            elif isinstance(inputs[i], MaskedArray):
                inputs[i] = inputs[i]._content[tokeep]
            elif isinstance(inputs[i], (self.numpy.ndarray, awkward0.array.base.AwkwardArray)):
                inputs[i] = inputs[i][tokeep]
            else:
                try:
                    for first in inputs[i]:
                        break
                except TypeError:
                    pass
                else:
                    inputs[i] = self.numpy.array(inputs[i], copy=False)[tokeep]

        # compute only the non-masked elements
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # put the masked out values back
        index = self.numpy.full(len(tokeep), -1, dtype=self.INDEXTYPE)
        index[tokeep] = self.numpy.arange(self.numpy.count_nonzero(tokeep))

        if isinstance(result, tuple):
            return tuple(self.Methods.maybemixin(type(x), IndexedMaskedArray)(index, x, maskedwhen=-1) if isinstance(x, (self.numpy.ndarray, awkward0.array.base.AwkwardArray)) else x for x in result)
        elif method == "at":
            return None
        else:
            return self.Methods.maybemixin(type(result), IndexedMaskedArray)(index, result, maskedwhen=-1)

    @property
    def counts(self):
        self._valid()
        content = self._util_counts(self._content)
        out = self.numpy.full(self.shape, -1, dtype=content.dtype)
        mask = self.boolmask(maskedwhen=False)
        out[mask] = content[mask]
        return out

    def choose(self, n):
        return self.copy(content=self._content.choose(n))

    def argchoose(self, n):
        return self.copy(content=self._content.argchoose(n))

    def distincts(self, nested=False):
        return self.copy(content=self._content.distincts(nested=nested))

    def argdistincts(self, nested=False):
        return self.copy(content=self._content.argdistincts(nested=nested))

    def pairs(self, nested=False):
        return self.copy(content=self._content.pairs(nested=nested))

    def argpairs(self, nested=False):
        return self.copy(content=self._content.argpairs(nested=nested))

    def cross(self, other, nested=False):
        return self.copy(content=self._content.cross(other, nested=nested))

    def argcross(self, other, nested=False):
        return self.copy(content=self._content.argcross(other, nested=nested))

    def flattentuple(self):
        return self.copy(content=self._util_flattentuple(self._content))

    def flatten(self, axis=0):
        mask = self.boolmask(maskedwhen=False)
        goodcontent = self._content[mask]
        content = self._util_flatten(goodcontent, axis)
        counts = self._util_counts(goodcontent)
        counts[counts < 0] = 1
        augcounts = self.numpy.ones(len(self), dtype=self.INDEXTYPE)
        augcounts[mask] = counts
        augparents = self.JaggedArray.offsets2parents(self.JaggedArray.counts2offsets(augcounts))
        augmask = mask[augparents]
        index = self.numpy.cumsum(augmask) - 1
        index[~augmask] = -1
        return self.IndexedMaskedArray(index, content)

    def pad(self, length, maskedwhen=True, clip=False, axis=0):
        return self.copy(content=self._util_pad(self._content, length, maskedwhen, clip, axis))

    def regular(self):
        self._valid()
        out = self._util_regular(self._content).astype(self.numpy.float64)
        out[self.boolmask(maskedwhen=True)] = float("nan")
        return out

    def indexed(self):
        maskindex = self.numpy.arange(len(self), dtype=self.INDEXTYPE)
        maskindex[self.boolmask(maskedwhen=True)] = -1
        return IndexedMaskedArray(maskindex, self._content, maskedwhen=-1)

    def _reduce(self, ufunc, identity, dtype):
        if self._util_hasjagged(self._content):
            return self.copy(content=self._content._reduce(ufunc, identity, dtype))

        elif isinstance(self._content, awkward0.array.table.Table):
            out = self._content.copy(contents={})
            for n, x in self._content._contents.items():
                out[n] = self.copy(content=x)
            return out._reduce(ufunc, identity, dtype)

        else:
            prepared = self._prepare(ufunc, identity, dtype)
            if ufunc is None:
                return (1 - self.numpy.isnan(prepared)).sum()
            elif ufunc is self.numpy.count_nonzero:
                return (1 - (prepared == 0)).sum()
            if issubclass(prepared.dtype.type, (self.numpy.floating, self.numpy.complexfloating)):
                prepared = self.numpy.where(self.numpy.isnan(prepared), identity, prepared)
            return ufunc.reduce(prepared)

    def _prepare(self, ufunc, identity, dtype):
        if isinstance(self._content, awkward0.array.table.Table):
            out = self._content.copy(contents={})
            for n, x in self._content._contents.items():
                out[n] = self.copy(content=x)._prepare(ufunc, identity, dtype)
            return out

        if isinstance(self._content, self.numpy.ndarray):
            if dtype is None and issubclass(self._content.dtype.type, (self.numpy.bool_, self.numpy.bool)):
                dtype = self.numpy.dtype(type(identity))
            if ufunc is None:
                content = self.numpy.zeros(self._content.shape, dtype=self.numpy.float32)
                content[self.numpy.isnan(self._content)] = self.numpy.nan
            elif ufunc is self.numpy.count_nonzero:
                content = self.numpy.ones(self._content.shape, dtype=self.numpy.int8)
                content[self.numpy.isnan(self._content)] = 0
                content[self._content == 0] = 0
            elif dtype is None:
                content = self._content
            else:
                content = self._content.astype(dtype)
        else:
            content = self._content._prepare(ufunc, identity, dtype)

        if content is self._content or not content.flags.owndata:
            content = content.copy()

        if ufunc is None:
            content[self.ismasked] = self.numpy.nan

        else:
            dtype = content.dtype

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

            content[self.ismasked] = identity

        return content

    def argmin(self):
        if self._util_hasjagged(self):
            return self.copy(content=self._content.argmin())
        else:
            index = self._content[self.isunmasked()].argmin()
            return self.numpy.searchsorted(self.numpy.cumsum(self.ismasked()), index, side="right")

    def argmax(self):
        if self._util_hasjagged(self):
            return self.copy(content=self._content.argmax())
        else:
            index = self._content[self.isunmasked()].argmax()
            return self.numpy.searchsorted(self.numpy.cumsum(self.ismasked()), index, side="right")

    def fillna(self, value):
        out = self._util_fillna(self._content, value)
        if not isinstance(out, self.numpy.ndarray):
            out = self.numpy.array(out)
        out[self.boolmask(maskedwhen=True)] = value
        return out

    @classmethod
    def _concatenate_axis0(cls, arrays):
        assert all(isinstance(x, MaskedArray) for x in arrays)
        mask = cls.numpy.concatenate([x.boolmask(maskedwhen=True) for x in arrays])
        content = awkward0.array.base.AwkwardArray.concatenate([x._content for x in arrays])
        return cls(mask, content, maskedwhen=True)

    _topandas_name = "MaskedSeries"

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

class BitMaskedArray(MaskedArray):
    """
    BitMaskedArray
    """

    # TODO for 1.0: need a maskshape parameter to apply length and multidimensional shape to the output

    def __init__(self, mask, content, maskedwhen=True, lsborder=False):
        super(BitMaskedArray, self).__init__(mask, content, maskedwhen=maskedwhen)
        self.lsborder = lsborder

    @classmethod
    def fromcontent(cls, content, maskedwhen=True, lsborder=False):
        if maskedwhen:
            mask = cls.numpy.zeros(cls._ceildiv8(len(content)), dtype=cls.BITMASKTYPE)
        else:
            mask = cls.numpy.ones(cls._ceildiv8(len(content)), dtype=cls.BITMASKTYPE)
        return cls(mask, content, maskedwhen=maskedwhen, lsborder=lsborder)

    @classmethod
    def fromboolmask(cls, mask, content, maskedwhen=True, lsborder=False):
        return cls(cls.bool2bit(mask, lsborder=lsborder), content, maskedwhen=maskedwhen, lsborder=lsborder)

    def copy(self, mask=None, content=None, maskedwhen=None, lsborder=None):
        out = super(BitMaskedArray, self).copy(mask=mask, content=content, maskedwhen=maskedwhen)
        out._lsborder = self._lsborder
        if lsborder is not None:
            out._lsborder = lsborder
        return out

    def _mine(self, overrides):
        mine = {}
        mine["maskedwhen"] = overrides.pop("maskedwhen", self._maskedwhen)
        mine["lsborder"] = overrides.pop("lsborder", self._lsborder)
        return mine

    def __awkward_serialize__(self, serializer):
        self._valid()
        return serializer.encode_call(
            ["awkward0", "BitMaskedArray"],
            serializer(self._mask, "BitMaskedArray.mask"),
            serializer(self._content, "BitMaskedArray.content"),
            {"json": bool(self._maskedwhen)},
            {"json": bool(self._lsborder)},
        )

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        value = self._util_toarray(value, self.BITMASKTYPE, self.numpy.ndarray)
        if self.check_prop_valid:
            if len(value.shape) != 1:
                raise ValueError("mask must have 1-dimensional shape")
        self._mask = value.view(self.BITMASKTYPE)
        self._isvalid = False

    def __len__(self):
        return len(self._content)

    @staticmethod
    def _ceildiv8(x):
        return -(-x >> 3)   # this is int(math.ceil(x / 8))

    @classmethod
    def bit2bool(cls, bitmask, lsborder=False):
        out = cls.numpy.unpackbits(bitmask)
        if lsborder:
            out = out.reshape(-1, 8)[:,::-1].reshape(-1)
        return out.view(cls.MASKTYPE)

    @classmethod
    def bool2bit(cls, boolmask, lsborder=False):
        boolmask = cls._util_toarray(boolmask, cls.MaskedArray.fget(None).MASKTYPE, cls.numpy.ndarray)
        if len(boolmask.shape) != 1:
            raise ValueError("boolmask must have 1-dimensional shape")
        if not issubclass(boolmask.dtype.type, (cls.numpy.bool_, cls.numpy.bool)):
            boolmask = (boolmask != 0)

        if lsborder:
            # maybe pad the length for reshape
            length = cls._ceildiv8(len(boolmask)) * 8
            if length != len(boolmask):
                out = cls.numpy.empty(length, dtype=boolmask.dtype)
                out[:len(boolmask)] = boolmask
            else:
                out = boolmask

            # reverse the order in groups of 8
            out = out.reshape(-1, 8)[:,::-1].reshape(-1)

        else:
            # numpy.packbits encodes as msb (most significant bit); already in the right order
            out = boolmask

        return cls.numpy.packbits(out)

    def boolmask(self, maskedwhen=None):
        if maskedwhen is None:
            maskedwhen = self._maskedwhen
        if maskedwhen == self._maskedwhen:
            bitmask = self._mask
        else:
            bitmask = self.numpy.bitwise_not(self._mask)
        return self.bit2bool(bitmask, lsborder=self._lsborder)[:len(self._content)]

    @property
    def lsborder(self):
        return self._lsborder

    @lsborder.setter
    def lsborder(self, value):
        self._lsborder = bool(value)

    def _util_layout(self, position, seen, lookup):
        awkward0.type.LayoutNode(self._mask, position + (0,), seen, lookup)
        awkward0.type.LayoutNode(self._content, position + (1,), seen, lookup)
        return (awkward0.type.LayoutArg("mask", position + (0,)),
                awkward0.type.LayoutArg("content", position + (1,)),
                awkward0.type.LayoutArg("maskedwhen", self._maskedwhen),
                awkward0.type.LayoutArg("lsborder", self._lsborder))

    def _valid(self):
        if self.check_whole_valid:
            if not self._isvalid:
                self._isvalid = True

    def __iter__(self, checkiter=True):
        if checkiter:
            self._checkiter()
        self._valid()

        one = self.numpy.uint8(1)
        zero = self.numpy.uint8(0)
        mask = self._mask
        content = self._content
        lencontent = len(content)
        maskedwhen = self._maskedwhen
        masked = self.masked

        if self._lsborder:
            byte = i = 0
            bit = start = self.numpy.uint8(1)
            while i < lencontent:
                if ((mask[byte] & bit) != 0) == self._maskedwhen:
                    yield masked
                else:
                    yield content[i]
                bit <<= one
                if bit == zero:
                    bit = start
                    byte += 1
                i += 1

        else:
            byte = i = 0
            bit = start = self.numpy.uint8(128)
            while i < lencontent:
                if ((mask[byte] & bit) != 0) == self._maskedwhen:
                    yield masked
                else:
                    yield content[i]
                bit >>= one
                if bit == zero:
                    bit = start
                    byte += 1
                i += 1

    def _maskat(self, where):
        bytepos = self.numpy.right_shift(where, 3)    # where // 8
        bitpos  = where - 8*bytepos                           # where % 8

        if self._lsborder:
            bitmask = self.numpy.left_shift(1, bitpos)
        else:
            bitmask = self.numpy.right_shift(128, bitpos)

        if isinstance(bitmask, self.numpy.ndarray):
            bitmask = bitmask.astype(self.BITMASKTYPE)
        else:
            bitmask = self.BITMASKTYPE.type(bitmask)

        return bytepos, bitmask

    def _maskwhere(self, where):
        if self._util_isinteger(where):
            bytepos, bitmask = self._maskat(where)
            return self.numpy.bitwise_and(self._mask[bytepos], bitmask) != 0

        elif isinstance(where, slice):
            # assumes a small slice; for a big slice, it could be faster to unpack the whole mask
            return self._maskwhere(self.numpy.arange(*where.indices(len(self._content))))

        else:
            where = self.numpy.array(where, copy=False)
            if len(where.shape) == 1 and self._util_isintegertype(where.dtype.type):
                byteposes, bitmasks = self._maskat(where)
                self.numpy.bitwise_and(bitmasks, self._mask[byteposes], bitmasks)
                return bitmasks.astype(self.numpy.bool_)

            elif len(where.shape) == 1 and issubclass(where.dtype.type, (self.numpy.bool, self.numpy.bool_)):
                # scales with the size of the mask anyway, so go ahead and unpack the whole mask
                unpacked = self.numpy.unpackbits(self._mask).view(self.MASKTYPE)

                if self._lsborder:
                    unpacked = unpacked.reshape(-1, 8)[:,::-1].reshape(-1)[:len(where)]
                else:
                    unpacked = unpacked[:len(where)]

                return unpacked[where]

            else:
                raise TypeError("cannot interpret shape {0}, dtype {1} as a fancy index or mask".format(where.shape, where.dtype))

    def __getitem__(self, where):
        self._valid()

        if self._util_isstringslice(where):
            content = self._content[where]
            cls = awkward0.array.objects.Methods.maybemixin(type(content), self.BitMaskedArray)
            out = cls.__new__(cls)
            out.__dict__.update(self.__dict__)
            out._content = content
            return out

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        if self._util_isinteger(head):
            if self._maskwhere(head) == self._maskedwhen:
                if tail != ():
                    raise ValueError("masked element ({0}) is not subscriptable".format(self.masked))
                return self.masked
            else:
                return self._content[(head,) + tail]

        else:
            mask = self._maskwhere(head)
            if tail != () and ((self._maskedwhen and mask.any()) or (not self._maskedwhen and not mask.all())):
                raise ValueError("masked element ({0}) is not subscriptable".format(self.masked))
            else:
                return self.copy(mask=self.bool2bit(mask, lsborder=self._lsborder), content=self._content[(head,) + tail], lsborder=self._lsborder)

    @classmethod
    def _concatenate_axis0(cls, arrays):
        raise NotImplementedError("concatenate not implemented for BitMaskedArray")

class IndexedMaskedArray(MaskedArray):
    """
    IndexedMaskedArray
    """

    # TODO for 1.0: remove maskedwhen and instead check for any negative value (can't be allowed to inherit methods that assume self.maskedwhen!)

    def __init__(self, mask, content, maskedwhen=-1):
        super(IndexedMaskedArray, self).__init__(mask, content, maskedwhen=maskedwhen)
        self._isvalid = False

    @classmethod
    def fromcontent(cls, content, maskedwhen=-1):
        mask = cls.numpy.arange(len(content), dtype=cls.INDEXTYPE)
        return cls(mask, content, maskedwhen=maskedwhen)

    def copy(self, mask=None, content=None, maskedwhen=None):
        out = self.__class__.__new__(self.__class__)
        out._mask = self._mask
        out._content = self._content
        out._maskedwhen = self._maskedwhen
        out._isvalid = self._isvalid
        if mask is not None:
            out._mask = mask
        if content is not None:
            out._content = content
        if maskedwhen is not None:
            out._maskedwhen = maskedwhen
        return out

    def __awkward_serialize__(self, serializer):
        self._valid()
        return serializer.encode_call(
            ["awkward0", "IndexedMaskedArray"],
            serializer(self._mask, "IndexedMaskedArray.mask"),
            serializer(self._content, "IndexedMaskedArray.content"),
            {"json": int(self._maskedwhen)},
        )

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        value = self._util_toarray(value, self.INDEXTYPE, self.numpy.ndarray)
        if self.check_prop_valid:
            if not self._util_isintegertype(value.dtype.type):
                raise TypeError("starts must have integer dtype")
            if len(value.shape) != 1:
                raise ValueError("mask must have 1-dimensional shape")
        self._mask = value
        self._isvalid = False

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = self._util_toarray(value, self.DEFAULTTYPE)
        self._isvalid = False

    @property
    def maskedwhen(self):
        return self._maskedwhen

    @maskedwhen.setter
    def maskedwhen(self, value):
        if self.check_prop_valid:
            if not self._util_isinteger(value):
                raise TypeError("maskedwhen must be an integer for IndexedMaskedArray")
        self._maskedwhen = value

    def boolmask(self, maskedwhen=True):
        if maskedwhen is None:
            raise TypeError("maskedwhen must be True or False")
        if maskedwhen:
            return self._mask == self._maskedwhen
        else:
            return self._mask != self._maskedwhen

    def _util_layout(self, position, seen, lookup):
        awkward0.type.LayoutNode(self._mask, position + (0,), seen, lookup)
        awkward0.type.LayoutNode(self._content, position + (1,), seen, lookup)
        return (awkward0.type.LayoutArg("mask", position + (0,)),
                awkward0.type.LayoutArg("content", position + (1,)),
                awkward0.type.LayoutArg("maskedwhen", self._maskedwhen))

    def _valid(self):
        if self.check_whole_valid:
            if not self._isvalid:
                if len(self._mask) != 0:
                    if self._mask.max() > len(self._content):
                        raise ValueError("maximum mask-index ({0}) is beyond the length of the content ({1})".format(self._mask.max(), len(self._content)))
                    if (self._mask[self._mask != self._maskedwhen] < 0).any():
                        raise ValueError("mask-index has negative values (other than maskedwhen)")

                self._isvalid = True

    def __iter__(self, checkiter=True):
        if checkiter:
            self._checkiter()
        self._valid()

        mask = self._mask
        lenmask = len(mask)
        content = self._content
        maskedwhen = self._maskedwhen
        masked = self.masked

        i = 0
        while i < lenmask:
            maskindex = mask[i]
            if maskindex == maskedwhen:
                yield masked
            else:
                yield content[maskindex]
            i += 1

    def __getitem__(self, where):
        self._valid()

        if self._util_isstringslice(where):
            content = self._content[where]
            cls = awkward0.array.objects.Methods.maybemixin(type(content), self.IndexedMaskedArray)
            out = cls.__new__(cls)
            out.__dict__.update(self.__dict__)
            out._content = content
            return out

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[0], where[1:]

        if self._util_isinteger(head):
            maskindex = self._mask[head]
            if maskindex == self._maskedwhen:
                if tail != ():
                    raise ValueError("masked element ({0}) is not subscriptable".format(self.masked))
                return self.masked
            else:
                return self._content[(maskindex,) + tail]

        else:
            maskindex = self._mask[head]
            if tail != () and (maskindex == self._maskedwhen).any():
                raise ValueError("masked element ({0}) is not subscriptable".format(self.masked))
            else:
                return self.copy(mask=maskindex)

    @property
    def counts(self):
        self._valid()
        out = self._util_counts(self._content)[self._index]
        out[self.boolmask(maskedwhen=True)] = -1
        return out

    def flatten(self, axis=0):
        mask = self.boolmask(maskedwhen=False)
        goodcontent = self._content[self._mask[self._mask >= 0]]
        content = self._util_flatten(goodcontent, axis)
        counts = self._util_counts(goodcontent)
        counts[counts < 0] = 1
        augcounts = self.numpy.ones(len(self), dtype=self.INDEXTYPE)
        augcounts[mask] = counts
        augparents = self.JaggedArray.offsets2parents(self.JaggedArray.counts2offsets(augcounts))
        augmask = mask[augparents]
        index = self.numpy.cumsum(augmask) - 1
        index[~augmask] = -1
        return self.IndexedMaskedArray(index, content)

    def regular(self):
        self._valid()
        out = self._util_regular(self._content).astype(self.numpy.float64)[self._index]
        out[self.boolmask(maskedwhen=True)] = float("nan")
        return out

    def indexed(self):
        return self

    def _prepare(self, ufunc, identity, dtype):
        if isinstance(self._content, awkward0.array.table.Table):
            out = self._content.copy(contents={})
            for n, x in self._content._contents.items():
                out[n] = self.copy(content=x)._prepare(ufunc, identity, dtype)
            return out

        if isinstance(self._content, self.numpy.ndarray):
            if dtype is None and issubclass(self._content.dtype.type, (self.numpy.bool_, self.numpy.bool)):
                dtype = self.numpy.dtype(type(identity))
            if dtype is None:
                content = self._content
            else:
                content = self._content.astype(dtype)
        else:
            content = self._content._prepare(ufunc, identity, dtype)

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

        out = self.numpy.full(self._mask.shape + content.shape[1:], identity, dtype=content.dtype)
        out[self.isunmasked] = content[self.mask[self.mask >= 0]]
        return out

    def argmin(self):
        if self._util_hasjagged(self):
            return self.copy(content=self._content.argmin())
        else:
            index = self._content[self._mask[self.isunmasked()]].argmin()
            return self.numpy.searchsorted(self.numpy.cumsum(self.ismasked()), index, side="right")

    def argmax(self):
        if self._util_hasjagged(self):
            return self.copy(content=self._content.argmax())
        else:
            index = self._content[self._mask[self.isunmasked()]].argmax()
            return self.numpy.searchsorted(self.numpy.cumsum(self.ismasked()), index, side="right")

    def fillna(self, value):
        out = self._util_fillna(self._content, value)
        if not isinstance(out, self.numpy.ndarray):
            out = self.numpy.array(out)
        out = self.numpy.append(out, value)
        if (self.mask < -1).any():
            mask = self.mask.copy()
            mask[mask < -1] = -1
        else:
            mask = self.mask
        return out[mask]

    @classmethod
    def _concatenate_axis0(cls, arrays):
        assert all(isinstance(x, IndexedMaskedArray) for x in arrays)

        indexes = []
        offset = 0
        for x in arrays:
            tmp = x._index.copy()
            tmp[tmp >= 0] += offset
            indexes.append(tmp)
            offset += len(x._content)
        index = cls.numpy.concatenate(indexes)

        content = awkward0.array.base.AwkwardArray.concatenate([x._content for x in arrays], axis=0)

        return cls(index, content, maskedwhen=maskedwhen)
