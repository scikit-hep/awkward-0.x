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

import collections
import numbers

import numpy

import awkward.array.base

class MaskedArray(awkward.array.base.AwkwardArray):
    def __init__(self, mask, content, maskedwhen=True):
        raise NotImplementedError

# class MaskedArray(awkward.array.base.AwkwardArray):
#     def __init__(self, mask, content, maskedwhen=True):
#         self.mask = mask
#         self.content = content
#         self.maskedwhen = maskedwhen

#     @property
#     def mask(self):
#         return self._mask

#     @mask.setter
#     def mask(self, value):
#         value = self._toarray(value, self.MASKTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

#         if len(value.shape) != 1:
#             raise TypeError("mask must have 1-dimensional shape")
#         if value.shape[0] == 0:
#             value = value.view(self.MASKTYPE)
#         if not issubclass(value.dtype.type, (numpy.bool_, numpy.bool)):
#             raise TypeError("mask must have boolean dtype")

#         self._mask = value

#     @property
#     def boolmask(self):
#         return self._mask

#     @boolmask.setter
#     def boolmask(self, value):
#         self.mask = value

#     @property
#     def content(self):
#         return self._content

#     @content.setter
#     def content(self, value):
#         self._content = self._toarray(value, self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))
        
#     @property
#     def maskedwhen(self):
#         return self._maskedwhen

#     @maskedwhen.setter
#     def maskedwhen(self, value):
#         self._maskedwhen = bool(value)

#     @property
#     def dtype(self):
#         return self._content.dtype

#     @property
#     def shape(self):
#         return self._content.shape

#     def __len__(self):
#         return len(self._content)

#     def __getitem__(self, where):
#         if self._isstring(where):
#             return MaskedArray(self._mask, self._content[where], maskedwhen=self._maskedwhen)

#         if not isinstance(where, tuple):
#             where = (where,)
#         head, tail = where[0], where[1:]

#         if isinstance(head, (numbers.Integral, numpy.integer)):
#             if self._mask[head] == self._maskedwhen:
#                 return numpy.ma.masked
#             else:
#                 return self._content[self._singleton(where)]
#         else:
#             return MaskedArray(self._mask[head], self._content[self._singleton(where)], maskedwhen=self._maskedwhen)

class BitMaskedArray(MaskedArray):
    def __init__(self, mask, content, maskedwhen=True, lsb=True):
        raise NotImplementedError

# class BitMaskedArray(MaskedArray):
#     @staticmethod
#     def fromboolmask(mask, content, maskedwhen=True, lsb=True):
#         out = BitMaskedArray([], content, maskedwhen=maskedwhen, lsb=lsb)
#         out.boolmask = mask
#         return out

#     def __init__(self, mask, content, maskedwhen=True, lsb=True):
#         self.mask = mask
#         self.content = content
#         self.maskedwhen = maskedwhen
#         self.lsb = lsb

#     @property
#     def mask(self):
#         return self._mask

#     @mask.setter
#     def mask(self, value):
#         value = self._toarray(value, self.BITMASKTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

#         if len(value.shape) != 1:
#             raise TypeError("mask must have 1-dimensional shape")

#         self._mask = value.view(self.BITMASKTYPE)

#     @property
#     def boolmask(self):
#         out = numpy.unpackbits(self._mask)
#         if self._lsb:
#             out = out.reshape(-1, 8)[:,::-1].reshape(-1)
#         return out.view(self.MASKTYPE)[:len(self._content)]

#     @boolmask.setter
#     def boolmask(self, value):
#         value = numpy.array(value, copy=False)

#         if len(value.shape) != 1:
#             raise TypeError("boolmask must have 1-dimensional shape")
#         if not issubclass(value.dtype.type, (numpy.bool, numpy.bool_)):
#             raise TypeError("boolmask must have boolean type")

#         if self._lsb:
#             # maybe pad the length for reshape
#             length = 8*((len(value) + 8 - 1) >> 3)   # ceil(len(value) / 8.0) * 8
#             if length != len(value):
#                 out = numpy.empty(length, dtype=numpy.bool_)
#                 out[:len(value)] = value
#             else:
#                 out = value

#             # reverse the order in groups of 8
#             out = out.reshape(-1, 8)[:,::-1].reshape(-1)

#         else:
#             # numpy.packbits encodes as msb (most significant bit); already in the right order
#             out = value

#         self._mask = numpy.packbits(out)
        
#     @property
#     def lsb(self):
#         return self._lsb

#     @lsb.setter
#     def lsb(self, value):
#         self._lsb = bool(value)

#     def _maskat(self, where):
#         bytepos = numpy.right_shift(where, 3)    # where // 8
#         bitpos  = where - 8*bytepos              # where % 8

#         if self.lsb:
#             bitmask = numpy.left_shift(1, bitpos)
#         else:
#             bitmask = numpy.right_shift(128, bitpos)

#         if isinstance(bitmask, numpy.ndarray):
#             bitmask = bitmask.astype(self.BITMASKTYPE)
#         else:
#             bitmask = self.BITMASKTYPE.type(bitmask)

#         return bytepos, bitmask

#     def _maskwhere(self, where):
#         if isinstance(where, (numbers.Integral, numpy.integer)):
#             bytepos, bitmask = self._maskat(where)
#             return numpy.bitwise_and(self._mask[bytepos], bitmask) != 0

#         elif isinstance(where, slice):
#             # assumes a small slice; for a big slice, it could be faster to unpack the whole mask
#             return self._maskwhere(numpy.arange(*where.indices(len(self._content))))

#         else:
#             where = numpy.array(where, copy=False)
#             if len(where.shape) == 1 and issubclass(where.dtype.type, numpy.integer):
#                 byteposes, bitmasks = self._maskat(where)
#                 numpy.bitwise_and(bitmasks, self._mask[byteposes], bitmasks)
#                 return bitmasks.astype(numpy.bool_)
        
#             elif len(where.shape) == 1 and issubclass(where.dtype.type, (numpy.bool, numpy.bool_)):
#                 # scales with the size of the mask anyway, so go ahead and unpack the whole mask
#                 unpacked = numpy.unpackbits(self._mask).view(self.MASKTYPE)

#                 if self.lsb:
#                     unpacked = unpacked.reshape(-1, 8)[:,::-1].reshape(-1)[:len(where)]
#                 else:
#                     unpacked = unpacked[:len(where)]

#                 return unpacked[where]

#             else:
#                 raise TypeError("cannot interpret shape {0}, dtype {1} as a fancy index or mask".format(where.shape, where.dtype))

#     def _setmask(self, where, valid):
#         if isinstance(where, (numbers.Integral, numpy.integer)):        
#             bytepos, bitmask = self._maskat(where)
#             if self._maskedwhen != valid:
#                 self._mask[bytepos] |= bitmask
#             else:
#                 self._mask[bytepos] &= numpy.bitwise_not(bitmask)

#         elif isinstance(where, slice):
#             # assumes a small slice; for a big slice, it could be faster to unpack the whole mask
#             self._setmask(numpy.arange(*where.indices(len(self._content))), valid)

#         else:
#             where = numpy.array(where, copy=False)
#             if len(where.shape) == 1 and issubclass(where.dtype.type, numpy.integer):
#                 bytepos, bitmask = self._maskat(where)
#                 if self._maskedwhen != valid:
#                     numpy.bitwise_or.at(self._mask, bytepos, bitmask)
#                 else:
#                     numpy.bitwise_and.at(self._mask, bytepos, numpy.bitwise_not(bitmask))

#             elif len(where.shape) == 1 and issubclass(where.dtype.type, (numpy.bool, numpy.bool_)):
#                 tmp = self.boolmask
#                 if self._maskedwhen != valid:
#                     tmp[where] = True
#                 else:
#                     tmp[where] = False
#                 self.boolmask = tmp

#             else:
#                 raise TypeError("cannot interpret shape {0}, dtype {1} as a fancy index or mask".format(where.shape, where.dtype))

#     def __getitem__(self, where):
#         if self._isstring(where):
#             return MaskedArray(self._mask, self._content[where], maskedwhen=self._maskedwhen)

#         if not isinstance(where, tuple):
#             where = (where,)
#         head, tail = where[0], where[1:]

#         if isinstance(head, (numbers.Integral, numpy.integer)):
#             if self._maskwhere(head) == self._maskedwhen:
#                 return numpy.ma.masked
#             else:
#                 return self._content[self._singleton(where)]

#         else:
#             return MaskedArray(self._maskwhere(head), self._content[self._singleton(where)], maskedwhen=self._maskedwhen)
