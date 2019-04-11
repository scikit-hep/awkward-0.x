#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

from __future__ import division

import unittest
import numbers
import operator
import numpy as np
import awkward

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_method_mixin(self):
        class TypeArrayMethods(awkward.Methods):
            def _initObjectArray(self, table):
                awkward.ObjectArray.__init__(self, table, lambda row: Type(row["x"]))
                self.content.rowname = "Type"

            @property
            def x(self):
                return self["x"]

            @x.setter
            def x(self, value):
                self["x"] = value

            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                if method != "__call__":
                    raise NotImplemented

                inputs = list(inputs)
                for i in range(len(inputs)):
                    if isinstance(inputs[i], awkward.util.numpy.ndarray) and inputs[i].dtype == awkward.util.numpy.dtype(object) and len(inputs[i]) > 0:
                        idarray = awkward.util.numpy.frombuffer(inputs[i], dtype=awkward.util.numpy.uintp)
                        if (idarray == idarray[0]).all():
                            inputs[i] = inputs[i][0]

                if ufunc is awkward.util.numpy.add or ufunc is awkward.util.numpy.subtract:
                    if not all(isinstance(x, (TypeArrayMethods, TypeMethods)) for x in inputs):
                        raise TypeError("(arrays of) Type can only be added to/subtracted from other (arrays of) Type")
                    out = self.empty_like()
                    out["x"] = getattr(ufunc, method)(*[x.x for x in inputs], **kwargs)
                    return out

                else:
                    return super(TypeArrayMethods, self).__array_ufunc__(ufunc, method, *inputs, **kwargs)

        class TypeMethods(awkward.Methods):
            _arraymethods = TypeArrayMethods

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, value):
                self._x = value

            def _number_op(self, operator, scalar, reverse=False):
                if not isinstance(scalar, (numbers.Number, awkward.util.numpy.number)):
                    raise TypeError("cannot {0} a Type with a {1}".format(operator.__name__, type(scalar).__name__))
                if reverse:
                    return Type(operator(scalar, self.x))
                else:
                    return Type(operator(self.x, scalar))
        
            def _type_op(self, operator, other, reverse=False):
                if isinstance(other, self._arraymethods):
                    # Give precedence to reverse op, implemented using self._arraymethods.__array_ufunc__
                    return NotImplemented
                if not isinstance(other, self.__class__):
                    raise TypeError("cannot {0} a Type with a {1}".format(operator.__name__, type(other).__name__))
                if reverse:
                    return Type(operator(other.x, self.x))
                else:
                    return Type(operator(self.x, other.x))

            def __mul__(self, other):
                return self._number_op(operator.mul, other)

            def __rmul__(self, other):
                return self._number_op(operator.mul, other, True)

            def __add__(self, other):
                return self._type_op(operator.add, other)

            def __radd__(self, other):
                return self._type_op(operator.add, other, True)

        class TypeArray(TypeArrayMethods, awkward.ObjectArray):
            def __init__(self, x):
                self._initObjectArray(awkward.Table())
                self["x"] = x

        class Type(TypeMethods):
            def __init__(self, x):
                self._x = x

        counts = np.array([1, 4, 2, 0, 15])
        x = np.arange(np.sum(counts))
        array = TypeArray(x)
        assert np.all(array.x == x)

        assert type(3.*array) is type(array)
        assert type(array*3.) is type(array)
        assert np.all(3.*array == 3.*x)
        assert np.all(array*3. == x*3.)

        scalar = Type(3.)
        assert type(array+scalar) is type(array)
        assert type(scalar+array) is type(array)
        assert np.all((array+scalar).x == x+3.)
        assert np.all((scalar+array).x == 3.+x)

        scalar2 = Type(4.)
        assert (scalar+scalar2).x == 7.

        JaggedTypeArray = awkward.Methods.mixin(TypeArrayMethods, awkward.JaggedArray)
        jagged_array = JaggedTypeArray.fromcounts(counts, array)
        assert np.all(jagged_array.x.flatten() == x)
        assert np.all(jagged_array.pairs().i0.x.counts == counts*(counts+1)//2)
