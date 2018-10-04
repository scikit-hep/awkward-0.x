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

import awkward.array.base

class UnionArray(awkward.array.base.AwkwardArray):
    def __init__(self, tags, index, contents):
        raise NotImplementedError

    def copy(self, index=None, content=None):
        raise NotImplementedError

    def deepcopy(self, index=None, content=None):
        raise NotImplementedError

    def empty_like(self, **overrides):
        raise NotImplementedError

    def zeros_like(self, **overrides):
        raise NotImplementedError

    def ones_like(self, **overrides):
        raise NotImplementedError

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        raise NotImplementedError

    @property
    def type(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def base(self):
        raise NotImplementedError

    def _valid(self):
        raise NotImplementedError

    def _argfields(self, function):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __getitem__(self, where):
        raise NotImplementedError

    def __setitem__(self, where, what):
        raise NotImplementedError

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError

    def any(self):
        raise NotImplementedError

    def all(self):
        raise NotImplementedError

    @classmethod
    def concat(cls, first, *rest):
        raise NotImplementedError

    @property
    def columns(self):
        raise NotImplementedError

    @property
    def allcolumns(self):
        raise NotImplementedError

    def pandas(self):
        raise NotImplementedError

# class UnionArray(awkward.array.base.AwkwardArray):
#     @classmethod
#     def fromtags(cls, tags, contents):
#         raise NotImplementedError

#     def __init__(self, tags, index, contents):
#         self.tags = tags
#         self.index = index
#         self.contents = contents

#     @property
#     def tags(self):
#         return self._tags

#     @tags.setter
#     def tags(self, value):
#         value = self._toarray(value, self.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

#         if len(value.shape) != 1:
#             raise TypeError("tags must have 1-dimensional shape")
#         if value.shape[0] == 0:
#             value = value.view(self.INDEXTYPE)
#         if not issubclass(value.dtype.type, numpy.integer):
#             raise TypeError("tags must have integer dtype")

#         self._tags = value

#     @property
#     def index(self):
#         return self._index

#     @index.setter
#     def index(self, value):
#         value = self._toarray(value, self.INDEXTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray))

#         if len(value.shape) != 1:
#             raise TypeError("index must have 1-dimensional shape")
#         if value.shape[0] == 0:
#             value = value.view(self.INDEXTYPE)
#         if not issubclass(value.dtype.type, numpy.integer):
#             raise TypeError("index must have integer dtype")

#         self._index = value

#     @property
#     def contents(self):
#         return self._contents

#     @contents.setter
#     def contents(self, value):
#         self._contents = tuple(self._toarray(x, self.CHARTYPE, (numpy.ndarray, awkward.array.base.AwkwardArray)) for x in value)

#     @property
#     def dtype(self):
#         return numpy.dtype(object)

#     @property
#     def shape(self):
#         return (len(self._tags),)

#     def __len__(self):
#         return len(self._tags)

#     def __getitem__(self, where):
#         if self._isstring(where):
#             return UnionArray(self._tags, self._index, tuple(x[where] for x in self._contents))

#         if self._tags.shape != self._index.shape:
#             raise ValueError("tags shape ({0}) does not match index shape ({1})".format(self._tags.shape, self._index.shape))

#         if not isinstance(where, tuple):
#             where = (where,)
#         head, tail = where[0], where[1:]

#         tags = self._tags[head]
#         index = self._index[head]
#         assert tags.shape == index.shape

#         uniques = numpy.unique(tags)
#         if len(uniques) == 1:
#             return self._contents[uniques[0]][self._singleton((index,) + tail)]
#         else:
#             return UnionArray(tags, index, self._contents)
