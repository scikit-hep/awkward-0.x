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

import collections
import numbers

import awkward.array.base
import awkward.type
import awkward.util

class MaskedArray(awkward.array.base.AwkwardArrayWithContent):
    """
    MaskedArray
    """

    masked = None

    def __init__(self, mask, content, maskedwhen=True):
        self.mask = mask
        self.content = content
        self.maskedwhen = maskedwhen

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

    def __awkward_persist__(self, ident, fill, prefix, suffix, schemasuffix, storage, compression, **kwargs):
        self._valid()
        return {"id": ident,
                "call": ["awkward", "MaskedArray"],
                "args": [fill(self._mask, "MaskedArray.mask", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                         fill(self._content, "MaskedArray.content", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                         {"json": bool(self._maskedwhen)}]}

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
    def ismasked(self):
        return self.boolmask(maskedwhen=True)

    @property
    def isunmasked(self):
        return self.boolmask(maskedwhen=False)

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
        return awkward.type.OptionType(awkward.type._fromarray(self._content, seen))

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
            cls = awkward.array.objects.Methods.maybemixin(type(content), self.MaskedArray)
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
            elif isinstance(inputs[i], (self.numpy.ndarray, awkward.array.base.AwkwardArray)):
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
            return tuple(self.Methods.maybemixin(type(x), IndexedMaskedArray)(index, x, maskedwhen=-1) if isinstance(x, (self.numpy.ndarray, awkward.array.base.AwkwardBase)) else x for x in result)
        elif method == "at":
            return None
        else:
            return self.Methods.maybemixin(type(result), IndexedMaskedArray)(index, result, maskedwhen=-1)

    def indexed(self):
        maskindex = self.numpy.arange(len(self), dtype=self.INDEXTYPE)
        maskindex[self.boolmask(maskedwhen=True)] = -1
        return IndexedMaskedArray(maskindex, self._content, maskedwhen=-1)

    def _reduce(self, ufunc, identity, dtype, regularaxis):
        if self._util_hasjagged(self._content):
            return self.copy(content=self._content._reduce(ufunc, identity, dtype, regularaxis))

        elif isinstance(self._content, awkward.array.table.Table):
            out = awkward.array.table.Table()
            for n, x in self._content._contents.items():
                out[n] = self.copy(content=x)
            return out._reduce(ufunc, identity, dtype, regularaxis)

        else:
            return ufunc.reduce(self._prepare(identity, dtype))

    def _prepare(self, identity, dtype):
        if isinstance(self._content, self.numpy.ndarray):
            if dtype is None and issubclass(self._content.dtype.type, (self.numpy.bool_, self.numpy.bool)):
                dtype = self.numpy.dtype(type(identity))
            if dtype is None:
                content = self._content
            else:
                content = self._content.astype(dtype)
        else:
            content = self._content._prepare(identity, dtype)

        if content is self._content or not content.flags.owndata:
            content = content.copy()

        content[self.ismasked] = identity
        return content

    def fillna(self, value):
        out = self._util_fillna(self._content, value)
        if not isinstance(out, self.numpy.ndarray):
            out = self.numpy.array(out)
        out[self.boolmask(maskedwhen=True)] = value
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

    def __awkward_persist__(self, ident, fill, prefix, suffix, schemasuffix, storage, compression, **kwargs):
        self._valid()
        return {"id": ident,
                "call": ["awkward", "BitMaskedArray"],
                "args": [fill(self._mask, "BitMaskedArray.mask", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                         fill(self._content, "BitMaskedArray.content", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                         {"json": bool(self._maskedwhen)},
                         {"json": bool(self._lsborder)}]}

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

    def _valid(self):
        if self.check_whole_valid:
            if not self._isvalid:
                if len(self._mask) != self._ceildiv8(len(self._content)):
                    raise ValueError("mask length ({0}) must be equal to ceil(content length / 8) ({1})".format(len(self._mask), self._ceildiv8(len(self._content))))

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
            cls = awkward.array.objects.Methods.maybemixin(type(content), self.BitMaskedArray)
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

class IndexedMaskedArray(MaskedArray):
    """
    IndexedMaskedArray
    """

    # TODO for 1.0: remove maskedwhen and instead check for any negative value (can't be allowed to inherit methods that assume self.maskedwhen!)

    def __init__(self, mask, content, maskedwhen=-1):
        super(IndexedMaskedArray, self).__init__(mask, content, maskedwhen=maskedwhen)
        self._isvalid = False

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

    def __awkward_persist__(self, ident, fill, prefix, suffix, schemasuffix, storage, compression, **kwargs):
        self._valid()
        return {"id": ident,
                "call": ["awkward", "IndexedMaskedArray"],
                "args": [fill(self._mask, "IndexedMaskedArray.mask", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                         fill(self._content, "IndexedMaskedArray.content", prefix, suffix, schemasuffix, storage, compression, **kwargs),
                         {"json": int(self._maskedwhen)}]}

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
            cls = awkward.array.objects.Methods.maybemixin(type(content), self.IndexedMaskedArray)
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

    def indexed(self):
        return self

    def _prepare(self, identity, dtype):
        if isinstance(self._content, self.numpy.ndarray):
            if dtype is None and issubclass(self._content.dtype.type, (self.numpy.bool_, self.numpy.bool)):
                dtype = self.numpy.dtype(type(identity))
            if dtype is None:
                content = self._content
            else:
                content = self._content.astype(dtype)
        else:
            content = self._content._prepare(identity, dtype)

        out = self.numpy.full(self._mask.shape + content.shape[1:], identity, dtype=content.dtype)
        out[self.isunmasked] = content[self.mask[self.mask >= 0]]
        return out

    def fillna(self, value):
        out = self._util_fillna(self._content, value)
        if not isinstance(out, self.numpy.ndarray):
            out = self.numpy.array(out)
        out[self.mask < 0] = value
        return out
