#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-0.x/blob/master/LICENSE

import pickle
import numbers

import awkward0.array.base
import awkward0.persist
import awkward0.type
import awkward0.util

class IndexedArray(awkward0.array.base.AwkwardArrayWithContent):
    """
    IndexedArray
    """

    @classmethod
    def invert(cls, permutation):
        if permutation.size == 0:
            return cls.numpy.zeros(0, dtype=cls.IndexedArray.fget(None).INDEXTYPE)
        permutation = permutation.reshape(-1)
        out = cls.numpy.zeros(permutation.max() + 1, dtype=cls.IndexedArray.fget(None).INDEXTYPE)
        identity = cls.numpy.arange(len(permutation))
        out[permutation] = identity
        if not cls.numpy.array_equal(out[permutation], identity):
            raise ValueError("cannot invert index; it contains duplicates")
        return out

    def __init__(self, index, content):
        self.index = index
        self.content = content

    def copy(self, index=None, content=None):
        out = self.__class__.__new__(self.__class__)
        out._index = self._index
        out._content = self._content
        out._inverse = self._inverse
        out._isvalid = self._isvalid
        if index is not None:
            out.index = index
        if content is not None:
            out.content = content
        return out

    def deepcopy(self, index=None, content=None):
        out = self.copy(index=index, content=content)
        out._index   = self._util_deepcopy(out._index)
        out._content = self._util_deepcopy(out._content)
        out._inverse = self._util_deepcopy(out._inverse)
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
        self._valid()
        return serializer.encode_call(
            ["awkward0", "IndexedArray"],
            serializer(self._index, "IndexedArray.index"),
            serializer(self._content, "IndexedArray.content"),
        )

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        value = self._util_toarray(value, self.INDEXTYPE, self.numpy.ndarray)
        if self.check_prop_valid:
            if not self._util_isintegertype(value.dtype.type):
                raise TypeError("index must have integer dtype")
            if (value < 0).any():
                raise ValueError("index must be a non-negative array")
        self._index = value
        self._inverse = None
        self._isvalid = False

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = self._util_toarray(value, self.DEFAULTTYPE)
        self._isvalid = False

    def _getnbytes(self, seen):
        if id(self) in seen:
            return 0
        else:
            seen.add(id(self))
            return self._index.nbytes + (self._content.nbytes if isinstance(self._content, self.numpy.ndarray) else self._content._getnbytes(seen))

    def __len__(self):
        return len(self._index)

    def _gettype(self, seen):
        out = awkward0.type._fromarray(self._content, seen)
        for x in self._index.shape[:0:-1]:
            out = awkward0.type.ArrayType(x, out)
        return out

    def _util_layout(self, position, seen, lookup):
        awkward0.type.LayoutNode(self._index, position + (0,), seen, lookup)
        awkward0.type.LayoutNode(self._content, position + (1,), seen, lookup)
        return (awkward0.type.LayoutArg("index", position + (0,)),
                awkward0.type.LayoutArg("content", position + (1,)))

    def _valid(self):
        if self.check_whole_valid:
            if not self._isvalid:
                if len(self._index) != 0 and self._index.reshape(-1).max() > len(self._content):
                    raise ValueError("maximum index ({0}) is beyond the length of the content ({1})".format(self._index.reshape(-1).max(), len(self._content)))

                self._isvalid = True

    def __iter__(self, checkiter=True):
        if checkiter:
            self._checkiter()
        self._valid()
        for i in self._index:
            yield self._content[i]

    def __getitem__(self, where):
        self._valid()

        if self._util_isstringslice(where):
            content = self._content[where]
            cls = awkward0.array.objects.Methods.maybemixin(type(content), self.IndexedArray)
            out = cls.__new__(cls)
            out.__dict__.update(self.__dict__)
            out._content = content
            return out

        if isinstance(where, tuple) and len(where) == 0:
            return self
        if not isinstance(where, tuple):
            where = (where,)
        head, tail = where[:len(self._index.shape)], where[len(self._index.shape):]

        head = self._index[head]
        if len(head.shape) != 0 and len(head) == 0:
            return self.numpy.empty(0, dtype=self._content.dtype)[tail]
        else:
            return self._content[(head,) + tail]

    def _invert(self, what):
        if self._inverse is None:
            self._inverse = self.invert(self._index)
        return IndexedArray(self._inverse, what)

    def __setitem__(self, where, what):
        if what.shape[:len(self._index.shape)] != self._index.shape:
            raise ValueError("array to assign does not have the same starting shape as index")

        if isinstance(where, awkward0.util.string):
            self._content[where] = self._invert(what)

        elif self._util_isstringslice(where):
            what = what.unzip()
            if len(where) != len(what):
                raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
            for x, y in zip(where, what):
                self._content[x] = self._invert(y)

        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            raise NotImplementedError("in-place operations not supported")

        if method != "__call__":
            return NotImplemented

        inputs = list(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], IndexedArray):
                inputs[i]._valid()
                inputs[i] = inputs[i][:]

        return getattr(ufunc, method)(*inputs, **kwargs)

    @property
    def counts(self):
        self._valid()
        return self._util_counts(self._content[self._index])

    def boolmask(self, maskedwhen=True):
        self._valid()
        return self._util_boolmask(self._content[self._index], maskedwhen)

    def choose(self, n):
        self._valid()
        return self._content[self._index].choose(n)

    def argchoose(self, n):
        self._valid()
        return self._content[self._index].argchoose(n)

    def distincts(self, nested=False):
        self._valid()
        return self._content[self._index].distincts(nested=nested)

    def argdistincts(self, nested=False):
        self._valid()
        return self._content[self._index].argdistincts(nested=nested)

    def pairs(self, nested=False):
        self._valid()
        return self._content[self._index].pairs(nested=nested)

    def argpairs(self, nested=False):
        self._valid()
        return self._content[self._index].argpairs(nested=nested)

    def cross(self, other, nested=False):
        self._valid()
        return self._content[self._index].cross(other, nested=nested)

    def argcross(self, other, nested=False):
        self._valid()
        return self._content[self._index].argcross(other, nested=nested)

    def flattentuple(self):
        self._valid()
        return self.copy(content=self._util_flattentuple(self._content))

    def flatten(self, axis=0):
        self._valid()
        return self._util_flatten(self._content[self._index], axis)

    def pad(self, length, maskedwhen=True, clip=False, axis=0):
        return self._util_pad(self._content[self._index], length, maskedwhen, clip, axis)

    def regular(self):
        self._valid()
        return self._util_regular(self._content[self._index])

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
        if isinstance(self._content, self.numpy.ndarray):
            return self._content[self._index]
        else:
            return self._content._prepare(ufunc, identity, dtype)[self._index]

    def argmin(self):
        return self._content[self._index].argmin()

    def argmax(self):
        return self._content[self._index].argmax()

    @classmethod
    def _concatenate_axis0(cls, arrays):
        assert all(isinstance(x, IndexedArray) for x in arrays)

        indexes = []
        offset = 0
        for x in arrays:
            indexes.append(x._index + offset)
            offset += len(x._content)
        index = cls.numpy.concatenate(indexes)

        content = awkward0.array.base.AwkwardArray.concatenate([x._content for x in arrays], axis=0)

        return cls(index, content)

    _topandas_name = "IndexedSeries"

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

class SparseArray(awkward0.array.base.AwkwardArrayWithContent):
    """
    SparseArray
    """

    # TODO for 1.0: replace length with an indexshape

    def __init__(self, length, index, content, default=None):
        self.length = length
        self.index = index
        self.content = content
        self.default = default

    def copy(self, length=None, index=None, content=None, default=None):
        out = self.__class__.__new__(self.__class__)
        out._length = self._length
        out._index = self._index
        out._content = self._content
        out._default = self._default
        out._inverse = self._inverse
        out._isvalid = self._isvalid
        if length is not None:
            out.length = length
        if index is not None:
            out.index = index
        if content is not None:
            out.content = content
        if default is not None:
            out.default = default
        return out

    def deepcopy(self, length=None, index=None, content=None, default=None):
        out = self.copy(length=length, index=index, content=content, default=default)
        out._index = self._util_deepcopy(out._index)
        out._content = self._util_deepcopy(out._content)
        out._inverse = self._util_deepcopy(out._inverse)
        return out

    def empty_like(self, **overrides):
        mine = {}
        mine = overrides.pop("length", self._length)
        mine = overrides.pop("default", self._default)
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.empty_like(self._content), **mine)
        else:
            return self.copy(content=self._content.empty_like(**overrides), **mine)

    def zeros_like(self, **overrides):
        mine = {}
        mine = overrides.pop("length", self._length)
        mine = overrides.pop("default", self._default)
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.zeros_like(self._content), **mine)
        else:
            return self.copy(content=self._content.zeros_like(**overrides), **mine)

    def ones_like(self, **overrides):
        mine = {}
        mine = overrides.pop("length", self._length)
        mine = overrides.pop("default", self._default)
        if isinstance(self._content, self.numpy.ndarray):
            return self.copy(content=self.numpy.ones_like(self._content), **mine)
        else:
            return self.copy(content=self._content.ones_like(**overrides), **mine)

    def __awkward_serialize__(self, serializer):
        self._valid()
        return serializer.encode_call(
            ["awkward0", "SparseArray"],
            {"json": int(self._length)},
            serializer(self._index, "SparseArray.index"),
            serializer(self._content, "SparseArray.content"),
            serializer(self._default, "SparseArray.default"),
        )

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        if self.check_prop_valid:
            if not self._util_isinteger(value):
                raise TypeError("length must be an integer")
            if value < 0:
                raise ValueError("length must be a non-negative integer")
        self._length = value

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        value = self._util_toarray(value, self.INDEXTYPE, self.numpy.ndarray)
        if self.check_prop_valid:
            if not self._util_isintegertype(value.dtype.type):
                raise TypeError("index must have integer dtype")
            if len(value.shape) != 1:
                raise ValueError("index must be one-dimensional")
            if (value < 0).any():
                raise ValueError("index must be a non-negative array")
            if len(value) > 0 and not (value[1:] >= value[:-1]).all():
                raise ValueError("index must be monatonically increasing")
        self._index = value
        self._inverse = None
        self._isvalid = False

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = self._util_toarray(value, self.DEFAULTTYPE)
        self._isvalid = False

    @property
    def default(self):
        import awkward0.array.jagged

        if self._default is None:
            if isinstance(self._content, awkward0.array.jagged.JaggedArray):
                return self.JaggedArray([0], [0], self._content.content)
            elif self._content.shape[1:] == ():
                return self._content.dtype.type(0)
            else:
                return self.numpy.zeros(self._content.shape[1:], dtype=self._content.dtype)

        else:
            return self._default

        self._isvalid = False

    @default.setter
    def default(self, value):
        self._default = value

    def _gettype(self, seen):
        return awkward0.type._fromarray(self._content, seen)

    def _util_layout(self, position, seen, lookup):
        awkward0.type.LayoutNode(self._index, position + (0,), seen, lookup)
        awkward0.type.LayoutNode(self._content, position + (1,), seen, lookup)
        return (awkward0.type.LayoutArg("length", self._length),
                awkward0.type.LayoutArg("index", position + (0,)),
                awkward0.type.LayoutArg("content", position + (1,)),
                awkward0.type.LayoutArg("default", self._default))

    def _getnbytes(self, seen):
        if id(self) in seen:
            return 0
        else:
            seen.add(id(self))
            return self._index.nbytes + (self._content.nbytes if isinstance(self._content, self.numpy.ndarray) else self._content._getnbytes(seen))

    def __len__(self):
        return self._length

    def _valid(self):
        if self.check_whole_valid:
            if not self._isvalid:
                if len(self._index) > len(self._content):
                    raise ValueError("length of index ({0}) must not be greater than the length of content ({1})".format(len(self._index), len(self._content)))

                self._isvalid = True

    def __iter__(self, checkiter=True):
        if checkiter:
            self._checkiter()
        self._valid()

        length = self._length
        index = self._index
        lenindex = len(self._index)
        content = self._content
        default = self.default

        i = 0
        j = self.numpy.searchsorted(index, 0, side="left")
        while i != length:
            if j == lenindex:
                yield default
            elif index[j] == i:
                yield content[j]
                while j != lenindex and index[j] == i:
                    j += 1
            else:
                yield default
            i += 1

    def __getitem__(self, where):
        self._valid()

        if self._util_isstringslice(where):
            content = self._content[where]
            cls = awkward0.array.objects.Methods.maybemixin(type(content), self.SparseArray)
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
            original_head = head
            if head < 0:
                head += self._length
            if not 0 <= head < self._length:
                raise IndexError("index {0} is out of bounds for size {1}".format(original_head, length))

            match = self.numpy.searchsorted(self._index, head, side="left")

            if match < len(self._index) and self._index[match] == head:
                return self._content[(match,) + tail]
            elif tail == ():
                return self.default
            else:
                return self.default[tail]

        elif isinstance(head, slice):
            start, stop, step = head.indices(self._length)

            if step == 0:
                raise ValueError("slice step cannot be zero")
            elif step > 0:
                mask = (self._index < stop)
                mask &= (self._index >= start)
                index = self._index - start
            elif step < 0:
                mask = (self._index > stop)
                mask &= (self._index <= start)
                index = start - self._index

            if (step > 0 and stop - start > 0) or (step < 0 and stop - start < 0):
                d, m = divmod(abs(start - stop), abs(step))
                length = d + (1 if m != 0 else 0)
            else:
                length = 0

            if abs(step) > 1:
                index, remainder = self.numpy.divmod(index, abs(step))
                mask[remainder != 0] = False

            index = index[mask]
            content = self._content[mask]
            if step < 0:
                index = index[::-1]
                content = content[::-1]

            return self.copy(length=length, index=index, content=content)[tail]

        elif isinstance(head, SparseArray) and len(head.shape) == 1 and issubclass(head.dtype.type, (self.numpy.bool, self.numpy.bool_)):
            head._valid()
            if self._length != head._length:
                raise IndexError("boolean index did not match indexed array along dimension 0; dimension is {0} but corresponding boolean dimension is {1}".format(self._length, head._length))

            # the new index is a cumsum (starting at zero) of the boolean values
            index = self.numpy.cumsum(head._content)
            length = index[-1]
            index[1:] = index[:-1]
            index[0] = 0

            # find my sparse elements in the mask's sparse elements
            match1 = self.numpy.searchsorted(head._index, self._index, side="left")
            match1[match1 >= len(head._index)] = len(head._index) - 1
            content = self._content[self.numpy.logical_and(head._index[match1] == self._index, head._content[match1])]

            # find the mask's sparse elements in my sparse elements
            match2 = self.numpy.searchsorted(self._index, head._index, side="left")
            match2[match2 >= len(head._index)] = len(head._index) - 1
            index = index[self.numpy.logical_and(self._index[match2] == head._index, head._content)]

            return self.copy(length=length, index=index, content=content)

        else:
            head = self._util_toarray(head, self.INDEXTYPE)
            if len(head.shape) == 1 and issubclass(head.dtype.type, (self.numpy.bool, self.numpy.bool_)):
                if self._length != len(head):
                    raise IndexError("boolean index did not match indexed array along dimension 0; dimension is {0} but corresponding boolean dimension is {1}".format(self._length, len(head)))

                head = self.numpy.arange(self._length, dtype=self.INDEXTYPE)[head]

            if len(head.shape) == 1 and self._util_isintegertype(head.dtype.type):
                mask = (head < 0)
                if mask.any():
                    head[mask] += self._length
                if (head < 0).any() or (head >= self._length).any():
                    raise IndexError("indexes out of bounds for size {0}".format(self._length))

                match = self.numpy.searchsorted(self._index, head, side="left")
                match[match >= len(self._index)] = len(self._index) - 1
                explicit = (self._index[match] == head)

                tags = self.numpy.zeros(len(head), dtype=self.TAGTYPE)
                index = self.numpy.zeros(len(head), dtype=self.INDEXTYPE)
                tags[explicit] = 1
                index[explicit] = self.numpy.arange(self.numpy.count_nonzero(explicit))

                content = self._content[match[explicit]]
                default = self.numpy.array([self.default])
                return self.UnionArray(tags, index, [default, content])[tail]

            else:
                raise TypeError("cannot interpret shape {0}, dtype {1} as a fancy index or mask".format(head.shape, head.dtype))

    def _getinverse(self):
        if self._inverse is None:
            self._inverse = self.numpy.searchsorted(self._index, self.numpy.arange(self._length, dtype=self.INDEXTYPE), side="left")
            if len(self._index) > 0:
                self._inverse[self._index[-1] + 1 :] = len(self._index) - 1
        return self._inverse

    @property
    def dense(self):
        self._valid()

        if isinstance(self._content, self.numpy.ndarray):
            out = self.numpy.full(self.shape, self.default, dtype=self.dtype)
            if len(self._index) != 0:
                mask = self.boolmask(maskedwhen=True)
                out[mask] = self._content[self._inverse[mask]]
            return out

        else:
            raise NotImplementedError(type(self._content))

    def boolmask(self, maskedwhen=True):
        self._valid()

        if len(self._index) == 0:
            return self.numpy.empty(0, dtype=self.numpy.bool_)

        if maskedwhen:
            return self._index[self._getinverse()] == self.numpy.arange(self._length, dtype=self.INDEXTYPE)
        else:
            return self._index[self._getinverse()] != self.numpy.arange(self._length, dtype=self.INDEXTYPE)

    def _invert(self, what):
        if len(what) != self._length:
            raise ValueError("cannot assign array of length {0} to sparse table of length {1}".format(len(what), self._length))

        test = what[self.boolmask(maskedwhen=False)].any()
        while not isinstance(test, bool):
            test = test.any()

        if test:
            raise ValueError("cannot assign an array with non-zero elements in the undefined spots of a sparse table")

        return IndexedArray(self._inverse, what)

    def __setitem__(self, where, what):
        if isinstance(where, awkward0.util.string):
            self._content[where] = self._invert(what)

        elif self._util_isstringslice(where):
            what = what.unzip()
            if len(where) != len(what):
                raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
            for x, y in zip(where, what):
                self._content[x] = self._invert(y)

        else:
            raise TypeError("invalid index for assigning column to Table: {0}".format(where))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            raise NotImplementedError("in-place operations not supported")

        if method != "__call__":
            return NotImplemented

        inputs = list(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], SparseArray):
                inputs[i]._valid()
                inputs[i] = inputs[i].dense   # FIXME: can do better (optimization)

        return getattr(ufunc, method)(*inputs, **kwargs)

    @property
    def counts(self):
        self._valid()
        content = self._util_counts(self._content)
        try:
            defaultlen = len(self.default)
        except:
            defaultlen = -1
        out = self.numpy.full(self.shape, defaultlen, dtype=content.dtype)
        if len(self._index) != 0:
            mask = self.boolmask(maskedwhen=True)
            out[mask] = content[self._inverse[mask]]
        return out

    def flattentuple(self):
        self._valid()
        return self.copy(content=self._util_flattentuple(self._content))

    def flatten(self, axis=0):
        self._valid()
        out = self._util_flatten(self._content, axis)
        out = self.numpy.full(self.shape, self.default, dtype=content.dtype)
        if len(self._index) != 0:
            mask = self.boolmask(maskedwhen=True)
            out[mask] = content[self._inverse[mask]]
        return out

    def pad(self, length, maskedwhen=True, clip=False, axis=0):
        return self._util_pad(self._content.dense, length, maskedwhen, clip, axis)

    def regular(self):
        self._valid()
        content = self._util_regular(self._content)
        out = self.numpy.full(self.shape, self.default, dtype=content.dtype)
        if len(self._index) != 0:
            mask = self.boolmask(maskedwhen=True)
            out[mask] = content[self._inverse[mask]]
        return out

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
        if isinstance(self._content, self.numpy.ndarray):
            return self.dense
        else:
            return self.copy(content=self._content._prepare(ufunc, identity, dtype)).dense

    def argmin(self):
        return self.dense.argmin()

    def argmax(self):
        return self.dense.argmax()

    _topandas_name = "SparseSeries"

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
