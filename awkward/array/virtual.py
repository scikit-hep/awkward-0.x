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

import numbers

import awkward.array.base
import awkward.util

class VirtualArray(awkward.array.base.AwkwardArray):
    def __init__(self, generator, cache=None, persistentkey=None, dtype=None, shape=None):
        raise NotImplementedError

# class VirtualArray(awkward.array.base.AwkwardArray):
#     class TransientKey(object):
#         def __init__(self, id):
#             self._id = id
#         def __repr__(self):
#             return "<VirtualArray.TransientKey {0}>".format(self._id)
#         def __hash__(self):
#             return hash((VirtualArray.TransientKey, self._id))
#         def __eq__(self, other):
#             return isinstance(other, VirtualArray.TransientKey) and self._id == other._id
#         def __ne__(self, other):
#             return not self.__eq__(other)
#         def __getstate__(self):
#             raise RuntimeError("VirtualArray.TransientKeys are not unique across processes, and hence should not be serialized")

#     def __init__(self, generator, cache=None, persistentkey=None, dtype=None, shape=None):
#         self.generator = generator
#         self.cache = cache
#         self.persistentkey = persistentkey
#         self._array = None

#         if dtype is None:
#             self._dtype = dtype
#         else:
#             self._dtype = numpy.dtype(dtype)

#         if shape is None or (isinstance(shape, tuple) and len(shape) != 0 and all(isinstance(x, (numbers.Integral, numpy.integer)) and x >= 0 for x in shape)):
#             self._shape = shape
#         else:
#             raise TypeError("shape must be None (unknown) or a non-empty tuple of non-negative integers")

#     @property
#     def generator(self):
#         return self._generator

#     @generator.setter
#     def generator(self, value):
#         if not callable(value):
#             raise TypeError("generator must be a callable (of zero arguments)")
#         self._generator = value

#     @property
#     def cache(self):
#         return self._cache

#     @cache.setter
#     def cache(self, value):
#         if not value is None and not (callable(getattr(value, "__getitem__", None)) and callable(getattr(value, "__setitem__", None)) and callable(getattr(value, "__delitem__", None))):
#             raise TypeError("cache must be a dict or have __getitem__/__setitem__/__delitem__ methods")
#         self._cache = value

#     @property
#     def persistentkey(self):
#         return self._persistentkey

#     @persistentkey.setter
#     def persistentkey(self, value):
#         if value is not None and not isinstance(value, awkward.util.string):
#             raise TypeError("persistentkey must be a string or None")
#         self._persistentkey = value

#     @property
#     def dtype(self):
#         if self._dtype is not None:
#             return self._dtype
#         else:
#             return self.array.dtype

#     @property
#     def shape(self):
#         if self._shape is not None:
#             return self._shape
#         else:
#             return self.array.shape

#     @property
#     def key(self):
#         if self._persistentkey is not None:
#             return self._persistentkey
#         else:
#             return self.TransientKey(id(self))

#     @property
#     def array(self):
#         # Normal states:   (1) no cache and _array is None: make a new one
#         #                  (2) no cache and _array is an array: return _array
#         #                  (3) have a cache and _array is None: make a new one (filling cache)
#         #                  (4) have a cache and _array is a key and cache[key] was evicted: make a new one (filling cache)
#         #                  (5) have a cache and _array is a key and cache[key] exists: return cache[key]
#         # 
#         # Abnormal states: (6) no cache and _array is a key (user removed _cache): make a new one
#         #                  (7) have a cache and _array is an array (user added _cache): fill cache and return _array

#         something = self._array

#         if something is None:
#             # states (1) and (3)
#             return self.materialize()

#         elif self._cache is None:
#             if isinstance(something, (VirtualArray.TransientKey, awkward.util.string)):
#                 # abnormal state (6)
#                 return self.materialize()
#             else:
#                 # state (2)
#                 return something

#         else:
#             if isinstance(something, (VirtualArray.TransientKey, awkward.util.string)):
#                 try:
#                     # state (5)
#                     return self._cache[something]
#                 except:
#                     # state (4), taking any error in __getitem__ as evidence that it was evicted
#                     return self.materialize()
#             else:
#                 # abnormal state (7)
#                 self._cache[self.key] = something
#                 return something

#     @property
#     def ismaterialized(self):
#         if self._cache is None:
#             return isinstance(self._array, (numpy.ndarray, awkward.array.base.AwkwardArray))
#         else:
#             return self._array is not None and self._array in self._cache

#     def materialize(self):
#         array = self._toarray(self.generator(), self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

#         if self._dtype is not None and self._dtype != array.dtype:
#             raise ValueError("materialized array has dtype {0}, expected dtype {1}".format(array.dtype, self._dtype))
#         if self._shape is not None and self._shape != array.shape:
#             raise ValueError("materialized array has shape {0}, expected shape {1}".format(array.shape, self._shape))
#         if len(array.shape) == 0:
#             raise ValueError("materialized object is scalar: {0}".format(array))

#         if self._cache is None:
#             # states (1), (2), and (6)
#             self._array = array
#         else:
#             # states (3) and (4)
#             self._array = self.key
#             self._cache[self._array] = array

#         return array

#     def __del__(self):
#         # TransientKeys are based on runtime ids, which Python may reuse after an object is garbage collected
#         # they *MUST* be removed from the cache to avoid confusion; persistentkeys can (and should) stay in
#         if self._cache is not None and isinstance(self._array, VirtualArray.TransientKey):
#             try:
#                 del self._cache[self._array]
#             except:
#                 pass

#     def __len__(self):
#         return self.shape[0]

#     def __getitem__(self, where):
#         return self.array[where]
