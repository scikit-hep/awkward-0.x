#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-array/blob/master/LICENSE

import numpy

# the following commented out lines allow JaggedArrayCpp to inherit from awkward-array
#from awkward.cpp.array.base import CppMethods
#from awkward.cpp.array.jagged import JaggedArrayCpp as JaggedArray

#__all__ = ["CppMethods", "JaggedArray"]

from awkward.cpp.array.array_impl import JaggedArray as JaggedArray

__all__ = ["JaggedArray"]
